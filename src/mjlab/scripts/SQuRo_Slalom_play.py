import re
import tyro
import torch
import pandas as pd
import yaml
from pathlib import Path
from typing import Literal
from mjlab.envs import ManagerBasedRlEnv
from dataclasses import asdict, dataclass, replace
from mjlab.viewer import NativeMujocoViewer
from mjlab.utils.wrappers import VideoRecorder
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import (
    _STEPS_PER_ITER,
    PHASE1_END_ITER,
    get_training_phase,
)
from mjlab.tasks.SQuRo_Slalom.mdp.entity import (
    PoleEntityCfg,
)
from mjlab.tasks.SQuRo_Slalom.mdp.reference import curvature_from_spacing, get_slalom_geometry
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.SQuRo_Slalom.mdp.reference import get_reference_joint_state
from mjlab.tasks.SQuRo_Slalom.mdp.indices import (
    _ACT_JOINT_NAMES, _MODEL_INDICES, FOOT_NAMES, FOOT_SITE_NAMES,
    resolve_model_indices, resolve_foot_force_ids,
)


# 任务配置
TASK_NAME = "Mjlab-SQuRo-Slalom"


@dataclass(frozen=True)
class PlayConfig:
    agent: Literal["zero", "random", "trained"] = "trained"
    wandb_run_path: str | None = None
    wandb_checkpoint_name: str | None = None
    checkpoint_file: str | None = None
    num_envs: int | None = 1
    device: str | None = None
    video: bool = True
    video_length: int = 1000
    video_height: int | None = 1080
    video_width: int | None = 1920
    record_data: bool = True
    # Slalom 任务相关配置
    command_source: Literal["转圈", "绕杆"] | None = None  # 转圈=一阶段, 绕杆=二阶段; None=按检查点轮数判定
    fixed_velocity: float | None = 0.1      # 基础速度; 最终命令还乘步频与曲率缩放
    fixed_height_f: float | None = 0.055    # 期望前肢高度; None = 任务默认
    fixed_height_h: float | None = 0.055    # 期望后肢高度; None = 任务默认
    fixed_gait_freq: float | None = 2.0     # 步频; None = 任务默认 (每回合 U(1,2) 采样)
    fixed_curvature: float | None = -15     # 仅转圈阶段生效; None = 任务默认 (每回合随机采样)
    fixed_pole_spacing: float | None = None  # 仅绕杆生效，映射到最近整数曲率对应的杆距
    fixed_slalom_curvature: int | None = None  # 绕杆峰值五至十五；None 为逐回合随机
    enable_collision: bool = False
    phase0_only: bool | None = None         # command_source 为 None 时的细粒度覆盖
    smoke_steps: int | None = None          # 无窗自检: 只跑 N 步打印统计后退出 (不启动 viewer)


# 固定值显示: None 表示交给任务默认
def fmt_fixed(value: float | None) -> str:
    return "任务默认" if value is None else f"{value:g}"


# 安全读取保存的配置字段，不构造 YAML 中的 Python 对象。
def read_saved_config(checkpoint: Path, name: str) -> dict:
    path = checkpoint.parent / "params" / f"{name}.yaml"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        return yaml.load(stream, Loader=yaml.BaseLoader) or {}


# 从 checkpoint 文件名提取训练轮次
def extract_iter_from_checkpoint(checkpoint_path: Path) -> int:
    m = re.search(r"model_(\d+)", checkpoint_path.name)
    if m:
        return int(m.group(1))
    return 0


# 从检查点路径提取视频名称
def extract_video_name_from_checkpoint(checkpoint_path: Path) -> str:
    checkpoint_name = checkpoint_path.stem
    if '_' in checkpoint_name:
        step = checkpoint_name.split('_')[-1]
    else:
        step = checkpoint_name

    run_dir_name = checkpoint_path.parent.parent.name

    if run_dir_name.startswith('run-'):
        timestamp_part = run_dir_name[4:]
        timestamp = timestamp_part.split('-')[0]
    else:
        timestamp = run_dir_name

    video_name = f"{timestamp}_{step}"
    return video_name


class JointDataRecorder:
    def __init__(self, log_dir, video_name, num_envs=1):
        self.log_dir = log_dir
        self.video_name = video_name
        self.num_envs = num_envs
        self.data_records = []
        self.step_count = 0

        # 14 个驱动关节 — 顺序必须与 entity actuator 顺序一致
        self.joint_names = [name.removesuffix("_joint") for name in _ACT_JOINT_NAMES]
        self._joint_ids_resolved = False
        self.action_names = self.joint_names  # 与 joint_names 同序
        self.foot_names = FOOT_NAMES
        self.foot_site_names = FOOT_SITE_NAMES
        self._foot_site_ids = None
        self._foot_force_ids = None

    def record_step_data(self, env, actions=None, rewards=None, dones=None):
        record = {'step': float(self.step_count)}

        # 记录动作
        if actions is not None:
            for action_idx in range(actions.shape[1]):
                if action_idx < len(self.action_names):
                    joint_name = self.action_names[action_idx]
                    record[f'{joint_name}_action'] = float(actions[0, action_idx].item())
                else:
                    record[f'action_{action_idx}'] = float(actions[0, action_idx].item())

        unwrapped = env.unwrapped
        asset = unwrapped.scene["robot"]
        env_idx = 0

        # 足部接触力
        contact_sensor = unwrapped.scene["feet_ground_contact"]
        if self._foot_force_ids is None:
            self._foot_force_ids = resolve_foot_force_ids(contact_sensor)
        feet_contact = contact_sensor.data.force[env_idx, self._foot_force_ids].reshape(len(FOOT_NAMES), -1, 3).sum(dim=1)
        for i, name in enumerate(self.foot_names):
            record[f'contact_{name}_x'] = float(feet_contact[i, 0].item())
            record[f'contact_{name}_y'] = float(feet_contact[i, 1].item())
            record[f'contact_{name}_z'] = float(feet_contact[i, 2].item())
            force_mag = torch.norm(feet_contact[i]).item()
            record[f'contact_{name}_mag'] = force_mag

        # 足端位置
        if self._foot_site_ids is None:
            self._foot_site_ids, _ = asset.find_sites(self.foot_site_names, preserve_order=True)
        foot_pos = asset.data.site_pos_w[env_idx, self._foot_site_ids]
        for i, name in enumerate(self.foot_names):
            record[f'foot_{name}_x'] = float(foot_pos[i, 0].item())
            record[f'foot_{name}_y'] = float(foot_pos[i, 1].item())
            record[f'foot_{name}_z'] = float(foot_pos[i, 2].item())

        # 关节状态 — 使用 ModelIndices 解析后的实体级关节索引
        if not self._joint_ids_resolved:
            resolve_model_indices(asset)
            self._joint_ids_resolved = True
        joint_ids = _MODEL_INDICES.joint_ids
        # 先减默认值再升维：[J] 或 [N,J] → 统一 [N,J]
        djp = asset.data.default_joint_pos
        djv = asset.data.default_joint_vel
        jp = (asset.data.joint_pos - djp) if djp is not None else asset.data.joint_pos
        jv = (asset.data.joint_vel - djv) if djv is not None else asset.data.joint_vel
        ja = asset.data.joint_acc
        if jp.dim() == 1:
            jp, jv, ja = jp.unsqueeze(0), jv.unsqueeze(0), ja.unsqueeze(0)
        joint_pos = jp[env_idx, joint_ids]
        joint_vel = jv[env_idx, joint_ids]
        joint_acc = ja[env_idx, joint_ids]
        actuator_force = asset.data.actuator_force[env_idx]

        # 基座位置
        base_pos = asset.data.root_link_pos_w[env_idx]
        base_lin_vel_w = asset.data.root_link_lin_vel_w[env_idx]
        base_ang_vel_w = asset.data.root_link_ang_vel_w[env_idx]

        # F_body / H_body 偏航角
        import math as _math
        if not hasattr(self, '_f_body_id') or not hasattr(self, '_h_body_id'):
            self._f_body_id = _MODEL_INDICES.f_body_id
            self._h_body_id = _MODEL_INDICES.h_body_id

        f_body_raw, h_body_raw = 0.0, 0.0
        if self._f_body_id is not None:
            quat = asset.data.body_link_quat_w[env_idx, self._f_body_id]
            w, x, y, z = quat[0], quat[1], quat[2], quat[3]
            f_body_raw = float(torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)).item())
        if self._h_body_id is not None:
            quat = asset.data.body_link_quat_w[env_idx, self._h_body_id]
            w, x, y, z = quat[0], quat[1], quat[2], quat[3]
            h_body_raw = float(torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)).item())

        # body+X heading (raw)
        record['f_body_raw_heading'] = f_body_raw
        record['h_body_raw_heading'] = h_body_raw
        # 物理前向 heading
        record['f_body_heading'] = float(torch.atan2(torch.sin(torch.tensor(f_body_raw - _math.pi/2)),
                                                       torch.cos(torch.tensor(f_body_raw - _math.pi/2))).item())
        record['h_body_heading'] = float(torch.atan2(torch.sin(torch.tensor(h_body_raw + _math.pi/2)),
                                                       torch.cos(torch.tensor(h_body_raw + _math.pi/2))).item())

        # 基座朝向
        heading = asset.data.heading_w[env_idx]
        record['heading'] = float(heading.item())

        # 控制命令等
        ref_joint_pos, ref_joint_vel = get_reference_joint_state(unwrapped)
        ref_joint_pos = ref_joint_pos[env_idx]
        ref_joint_vel = ref_joint_vel[env_idx]
        command = unwrapped.command_manager.get_command("slalom_cmd")[env_idx]

        for i, name in enumerate(self.joint_names):
            record[f'{name}_pos'] = float(joint_pos[i].item())
            record[f'{name}_vel'] = float(joint_vel[i].item())
            record[f'{name}_acc'] = float(joint_acc[i].item())
            record[f'{name}_torque'] = float(actuator_force[i].item())
            record[f'{name}_ref_pos'] = float(ref_joint_pos[i].item())
            record[f'{name}_ref_vel'] = float(ref_joint_vel[i].item())

        record['base_pos_x'] = float(base_pos[0].item())
        record['base_pos_y'] = float(base_pos[1].item())
        record['base_pos_z'] = float(base_pos[2].item())
        record['base_lin_vel_x'] = float(base_lin_vel_w[0].item())
        record['base_lin_vel_y'] = float(base_lin_vel_w[1].item())
        record['base_lin_vel_z'] = float(base_lin_vel_w[2].item())
        record['base_ang_vel_x'] = float(base_ang_vel_w[0].item())
        record['base_ang_vel_y'] = float(base_ang_vel_w[1].item())
        record['base_ang_vel_z'] = float(base_ang_vel_w[2].item())

        command_names = [
            'vel_command_x', 'height_f_command', 'height_h_command',
            'gait_freq_command', 'curvature_command',
        ]
        for i, name in enumerate(command_names):
            record[name] = float(command[i].item())

        if rewards is not None:
            record['reward'] = float(rewards[0].item()) if torch.is_tensor(rewards) else float(rewards[0])
        if dones is not None:
            record['done'] = float(dones[0].item() if torch.is_tensor(dones) else float(dones[0]))

        self.data_records.append(record)
        self.step_count += 1

    def save_to_csv(self):
        if not self.data_records:
            print("[WARN] 没有数据可保存")
            return

        video_dir = self.log_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        csv_path = video_dir / f"{self.video_name}.csv"
        df = pd.DataFrame(self.data_records)
        df['step'] = df['step'].astype(int)
        df.to_csv(csv_path, index=False)
        print(f"[INFO] 关节数据已保存到: {csv_path}")
        print(f"[INFO] 记录了 {len(self.data_records)} 步数据，{len(df.columns)} 列")


class DataRecordingEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, clip_actions=None, data_recorder=None, action_scale=1.0):
        super().__init__(env, clip_actions)
        self.data_recorder = data_recorder
        self.action_scale = action_scale

    def step(self, actions):
        scaled_actions = actions * self.action_scale

        obs_dict, rew, dones, extras = super().step(scaled_actions)

        if self.data_recorder:
            self.data_recorder.record_step_data(
                self.env,
                actions=scaled_actions,
                rewards=rew,
                dones=dones
            )

        return obs_dict, rew, dones, extras


def run_play(cfg: PlayConfig):
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(TASK_NAME, play=True)
    agent_cfg = load_rl_cfg(TASK_NAME)

    DUMMY_MODE = cfg.agent in {"zero", "random"}
    TRAINED_MODE = not DUMMY_MODE

    log_dir: Path | None = None
    resume_path: Path | None = None
    video_name: str | None = None

    if TRAINED_MODE:
        log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
        if cfg.checkpoint_file is not None:
            resume_path = Path(cfg.checkpoint_file)
            if not resume_path.exists():
                raise FileNotFoundError(f"未找到checkpoint文件: {resume_path}")
            video_name = extract_video_name_from_checkpoint(resume_path)
        else:
            if cfg.wandb_run_path is None:
                print("请输入 --checkpoint-file 路径:")
                checkpoint_file = input().strip()
                if checkpoint_file.startswith('"') and checkpoint_file.endswith('"'):
                    checkpoint_file = checkpoint_file[1:-1]
                elif checkpoint_file.startswith("'") and checkpoint_file.endswith("'"):
                    checkpoint_file = checkpoint_file[1:-1]
                if not checkpoint_file:
                    raise ValueError("必须提供checkpoint文件路径")
                resume_path = Path(checkpoint_file)
                if not resume_path.exists():
                    raise FileNotFoundError(f"未找到checkpoint文件: {resume_path}")
                video_name = extract_video_name_from_checkpoint(resume_path)
            else:
                resume_path, was_cached = get_wandb_checkpoint_path(
                    log_root_path,
                    Path(cfg.wandb_run_path),
                    cfg.wandb_checkpoint_name
                )
                video_name = extract_video_name_from_checkpoint(resume_path)

        log_dir = resume_path.parent

    # 设置环境参数
    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    # 自动识别训练阶段，注意：train_iter 仅用于提取，实际阶段判断用 align_iter（与 env 内部一致）
    train_iter = 0
    align_iter = 0
    cmd_cfg = env_cfg.commands["slalom_cmd"]
    saved_env = read_saved_config(resume_path, "env") if resume_path is not None else {}
    saved_agent = read_saved_config(resume_path, "agent") if resume_path is not None else {}
    saved_command = saved_env.get("commands", {}).get("slalom_cmd", {})
    if saved_agent:
        saved_clip = saved_agent.get("clip_actions", "null")
        agent_cfg.clip_actions = None if saved_clip.lower() in ("null", "none", "") else float(saved_clip)
        saved_steps = str(saved_agent.get("num_steps_per_env", "null")).lower()
        if saved_steps not in ("null", "none", "") and int(saved_steps) != _STEPS_PER_ITER:
            print(f"[WARN] 检查点按每轮 {int(saved_steps)} 步训练, 当前口径为 {_STEPS_PER_ITER} 步/轮; "
                  f"课程按轮次对齐故不受影响, 但检查点里的 step 计数不可直接比较")
    if saved_env:
        env_cfg.decimation = int(saved_env.get("decimation", env_cfg.decimation))
        saved_mujoco = saved_env.get("sim", {}).get("mujoco", {})
        env_cfg.sim.mujoco.timestep = float(saved_mujoco.get("timestep", env_cfg.sim.mujoco.timestep))
    if TRAINED_MODE and resume_path is not None:
        train_iter = extract_iter_from_checkpoint(resume_path)
        align_iter = max(0, train_iter - 10)  # -10 避免边界效应，与 env 内部对齐

    # 阶段判定优先级: command_source > phase0_only > 检查点记录 > 检查点轮数
    if cfg.command_source is not None:
        is_slalom_phase = cfg.command_source == "绕杆"
        stage_src = f"命令行 command_source={cfg.command_source}"
    elif cfg.phase0_only is not None:
        is_slalom_phase = get_training_phase(align_iter * _STEPS_PER_ITER, cfg.phase0_only) == 1
        stage_src = f"命令行 phase0_only={cfg.phase0_only}"
    elif "phase0_only" in saved_command:
        saved_phase0 = str(saved_command["phase0_only"]).lower() == "true"
        is_slalom_phase = get_training_phase(align_iter * _STEPS_PER_ITER, saved_phase0) == 1
        stage_src = f"检查点 phase0_only={saved_phase0}"
    else:
        is_slalom_phase = get_training_phase(align_iter * _STEPS_PER_ITER, False) == 1
        stage_src = (f"检查点轮数 iter {align_iter}" if resume_path is not None
                     else "无检查点, 默认一阶段")

    # 阶段由解析结果唯一决定, 回放期间不再翻转
    cmd_cfg.phase0_only = not is_slalom_phase  # type: ignore[attr-defined]
    if is_slalom_phase and align_iter < PHASE1_END_ITER:
        print(f"[WARN] iter {align_iter} 早于绕杆阶段起点 iter {PHASE1_END_ITER}, 按 {stage_src} 对齐到起点")
        align_iter = PHASE1_END_ITER
    align_step = align_iter * _STEPS_PER_ITER

    # 命令固定值: 转圈用 fixed_curvature, 绕杆用 fixed_pole_spacing, 两者互斥; 为 None 的字段用任务默认
    if cmd_cfg is not None:
        cmd_cfg.fixed_velocity = cfg.fixed_velocity  # type: ignore[assignment]
        cmd_cfg.fixed_height_f = cfg.fixed_height_f  # type: ignore[assignment]
        cmd_cfg.fixed_height_h = cfg.fixed_height_h  # type: ignore[assignment]
        cmd_cfg.fixed_gait_freq = cfg.fixed_gait_freq  # type: ignore[assignment]
        cmd_cfg.fixed_curvature = None if is_slalom_phase else cfg.fixed_curvature  # type: ignore[assignment]
        cmd_cfg.fixed_pole_spacing = cfg.fixed_pole_spacing if is_slalom_phase else None  # type: ignore[assignment]
        cmd_cfg.fixed_slalom_curvature = cfg.fixed_slalom_curvature if is_slalom_phase else None  # type: ignore
        if is_slalom_phase and cfg.fixed_pole_spacing is not None and cfg.fixed_slalom_curvature is not None:
            raise ValueError("固定绕杆峰值曲率和固定杆距只能指定一个")

    stage_name = "绕杆 (Phase 1)" if is_slalom_phase else "转圈 (Phase 0)"
    stage_value = (f"杆间距={fmt_fixed(cfg.fixed_pole_spacing)}"
                   if is_slalom_phase else f"曲率={fmt_fixed(cfg.fixed_curvature)}")
    print(f"[INFO] 阶段 = {stage_name} | 判定来源 = {stage_src}")
    print(f"[INFO] 命令固定值: vel={fmt_fixed(cfg.fixed_velocity)} h_f={fmt_fixed(cfg.fixed_height_f)} "
          f"h_h={fmt_fixed(cfg.fixed_height_h)} gait={fmt_fixed(cfg.fixed_gait_freq)} | {stage_value}")

    # 构建命令后缀（用于视频和CSV文件名）
    cmd_suffix_parts = ["raogan" if is_slalom_phase else "zhuanquan"]
    if is_slalom_phase:
        peak = (curvature_from_spacing(cfg.fixed_pole_spacing) if cfg.fixed_pole_spacing is not None
                else cfg.fixed_slalom_curvature)
        cmd_suffix_parts.append("kauto" if peak is None else f"k{peak}")
    else:
        cmd_suffix_parts.append("cuauto" if cfg.fixed_curvature is None else f"cu{cfg.fixed_curvature:g}")
    cmd_suffix = f"-{'-'.join(cmd_suffix_parts)}"
    if video_name is not None:
        video_name = f"{video_name}{cmd_suffix}"

    # 保留训练中的 mocap 杆实体，回合重置负责按实际曲率布置。
    mask = int(is_slalom_phase and cfg.enable_collision)
    env_cfg.scene.entities = {
        name: replace(entity, contype=mask, conaffinity=mask) if isinstance(entity, PoleEntityCfg) else entity
        for name, entity in env_cfg.scene.entities.items()
    }

    # 创建环境
    render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
    if cfg.video and DUMMY_MODE:
        print("[WARN] 虚拟智能体的视频录制已禁用")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    # 对齐 curriculum 阶段（用与阶段判定同一口径的 align_step, 回放期间阶段不再翻转）
    # 无检查点时为 0; 显式指定绕杆时已按阶段起点钳过, 故 dummy 模式也能跑二阶段
    env.common_step_counter = align_step
    if align_step > 0:
        print(f"[INFO] curriculum 对齐到 iter {align_iter} (step {align_step}) [{stage_name}]")

    # 固定杆距可能映射到最近整数曲率，打印实际采用的几何。
    if is_slalom_phase:
        if peak is None:
            print("[INFO] 绕杆峰值曲率每回合随机采样整数 5～15，实体杆按同一几何更新")
        else:
            geometry = get_slalom_geometry(peak)
            print(f"[INFO] 绕杆峰值曲率={peak}，实际杆心距={geometry['spacing']:.6f}m，输入杆距={cfg.fixed_pole_spacing}")

    # 初始化数据记录器
    data_recorder = None
    if cfg.record_data and TRAINED_MODE:
        print("[INFO] 启用关节数据记录")
        data_recorder = JointDataRecorder(log_dir, video_name, num_envs=env_cfg.scene.num_envs)

    # 添加视频录制器（文件名与CSV统一，使用 video_name 作为前缀）
    if TRAINED_MODE and cfg.video:
        print("[INFO] 播放期间录制视频")
        assert log_dir is not None
        video_folder = log_dir / "videos"
        assert video_name is not None
        env = VideoRecorder(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            name_prefix=video_name,
            disable_logger=False,
        )

    # 使用数据记录包装器
    env = DataRecordingEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
        data_recorder=data_recorder,
        action_scale=1.0
    )

    # 创建策略
    if DUMMY_MODE:
        action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
        if cfg.agent == "zero":
            class PolicyZero:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return torch.zeros(action_shape, device=env.unwrapped.device)
            policy = PolicyZero()
        else:
            class PolicyRandom:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1
            policy = PolicyRandom()
    else:
        runner_cls = load_runner_cls(TASK_NAME) or MjlabOnPolicyRunner
        agent_cfg_dict = asdict(agent_cfg)
        runner = runner_cls(env, agent_cfg_dict, str(log_dir), device=device)
        runner.load(
            str(resume_path),
            load_cfg={"actor": True},
            strict=True,
            map_location=device
        )
        env.unwrapped.common_step_counter = align_step
        policy = runner.get_inference_policy(device=device)

    # 无窗自检: 推 N 步打印统计后退出 (不启动 viewer)
    if cfg.smoke_steps is not None:
        obs_dict = env.reset()
        obs_in = obs_dict[0] if isinstance(obs_dict, tuple) else obs_dict
        # 必须 no_grad: 本任务的 actions 观测带 history_length, 历史缓冲的 where(out=) 不接受需要梯度的输入
        with torch.no_grad():
            for _ in range(cfg.smoke_steps):
                act = policy(obs_in)
                obs_dict, rew, dones, extras = env.step(act)
                obs_in = obs_dict[0] if isinstance(obs_dict, tuple) else obs_dict
        robot = env.unwrapped.scene["robot"]
        body_pos = robot.data.body_link_pos_w
        cmd = env.unwrapped.command_manager.get_command("slalom_cmd")
        print(f"[INFO] 自检 {cfg.smoke_steps} 步完成")
        print(f"[INFO] 命令 = {[round(float(v), 4) for v in cmd[0].tolist()]}")  # type: ignore
        print(f"[INFO] F_body x={float(body_pos[0, _MODEL_INDICES.f_body_id, 0]):.4f} "
              f"z={float(body_pos[0, _MODEL_INDICES.f_body_id, 2]):.4f} | "
              f"H_body x={float(body_pos[0, _MODEL_INDICES.h_body_id, 0]):.4f} "
              f"z={float(body_pos[0, _MODEL_INDICES.h_body_id, 2]):.4f}")
        print(f"[INFO] 最近一步奖励: {float(rew[0]):.4f}")
        if data_recorder:
            data_recorder.save_to_csv()
        env.close()
        return

    # 运行查看器
    try:
        viewer = NativeMujocoViewer(env, policy)
        viewer.run()
    except KeyboardInterrupt:
        print("[INFO] 用户中断播放")
    except Exception as e:
        print(f"[ERROR] 播放过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if data_recorder:
            data_recorder.save_to_csv()
        env.close()


def main():
    args = tyro.cli(PlayConfig, description="播放 SQuRo Slalom 智能体")
    run_play(args)


if __name__ == "__main__":
    main()
