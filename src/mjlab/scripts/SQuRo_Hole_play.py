# uv run python -B -m mjlab.scripts.SQuRo_Hole_play --checkpoint_file <path>
# uv run python -B -m mjlab.scripts.SQuRo_Hole_play --agent zero --smoke_steps 50 --no-video

import re
import tyro
import math
import torch
import pandas as pd
from pathlib import Path
from typing import Literal
from mjlab.envs import ManagerBasedRlEnv
from dataclasses import asdict, dataclass
from mjlab.viewer import NativeMujocoViewer
from mjlab.utils.wrappers import VideoRecorder
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.SQuRo_Hole.mdp.command import THRESHOLD_HEIGHT
from mjlab.tasks.SQuRo_Hole.mdp.config import (
    STAGE1_END,
    STAGE2_END,
    STAGE3_END,
    STEPS_PER_ITER,
    get_current_stage,
    stage_requires_collision,
)
from mjlab.tasks.SQuRo_Hole.mdp.hole import (
    apply_saved_layout,
    configure_hole_entities,
    set_obstacle_visibility,
)
from mjlab.tasks.SQuRo_Hole.rl.runner import read_env_step, read_hole_state
from mjlab.tasks.SQuRo_Hole.mdp.indices import _MODEL_INDICES, resolve_model_indices
from mjlab.tasks.SQuRo_Hole.mdp.reference import (
    get_reference_joint_state,
)


TASK_NAME = "Mjlab-SQuRo-Hole"


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
    # Hole 任务: 命令来源。schedule = 位置表(阶段 3/4 口径, 按位移自动推进);
    # fixed/random = 用 fixed_* 锁死或按阶段 1/2 随机采样 (位置表关闭)
    command_source: Literal["schedule", "fixed", "random"] = "fixed"
    fixed_velocity: float | None = None
    # 默认 None: 只有 fixed 模式才补默认高度, schedule/random 只在显式给出时覆盖
    fixed_height_F: float | None = 0.055
    fixed_height_H: float | None = 0.055
    stage: int | None = None               # None = 按 checkpoint 轮次推断; 1~4 = 强制该阶段
    enable_collision: bool | None = False   # None = 按 cfg (训练默认关, 阶段 4 才开)
    show_obstacles: bool = False            # False = 板体 rgba 设为全透明 (仅外观, 不影响碰撞)
    smoke_steps: int | None = None     # 无窗自检: 只跑 N 步打印统计后退出


# 从 checkpoint 文件名提取训练轮次
def extract_iter_from_checkpoint(checkpoint_path: Path) -> int:
    m = re.search(r"model_(\d+)", checkpoint_path.name)
    if m:
        return int(m.group(1))
    return 0


# 从检查点路径提取视频名称
def extract_video_name_from_checkpoint(checkpoint_path: Path) -> str:
    checkpoint_name = checkpoint_path.stem
    step = checkpoint_name.split('_')[-1] if '_' in checkpoint_name else checkpoint_name
    run_dir_name = checkpoint_path.parent.parent.name
    if run_dir_name.startswith('run-'):
        timestamp = run_dir_name[4:].split('-')[0]
    else:
        timestamp = run_dir_name
    return f"{timestamp}_{step}"


# 选择可写输出目录: 受限环境无法新建目录时退回系统临时目录
def resolve_output_dir(preferred: Path) -> Path:
    import tempfile
    videos_dir = preferred / "videos"
    try:
        videos_dir.mkdir(parents=True, exist_ok=True)
        probe = videos_dir / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
        return preferred
    except OSError as e:
        fallback = Path(tempfile.gettempdir()) / "mjlab_play" / preferred.name
        (fallback / "videos").mkdir(parents=True, exist_ok=True)
        print(f"[WARN] {preferred} 不可写 ({e}), 输出改到 {fallback}")
        return fallback


# 受限模式: 0 都高 / 1 前低 / 2 后低
def mode_of(height_F: float, height_H: float) -> str:
    low_F = height_F < THRESHOLD_HEIGHT
    low_H = height_H < THRESHOLD_HEIGHT
    if low_F and not low_H:
        return "前低后高"
    if low_H and not low_F:
        return "前高后低"
    if low_F and low_H:
        return "双低"
    return "都高"


class JointDataRecorder:
    def __init__(self, log_dir, video_name, num_envs=1):
        self.log_dir = log_dir
        self.video_name = video_name
        self.num_envs = num_envs
        self.data_records = []
        self.step_count = 0

        # 14 个执行器 (动作空间顺序)
        self.joint_names = [
            'F_spine1', 'F_body', 'Neck_yaw', 'Neck_pitch',
            'FL_shoulder', 'FL_elbow', 'FR_shoulder', 'FR_elbow',
            'H_spine1', 'H_body', 'HL_hip', 'HL_knee', 'HR_hip', 'HR_knee',
        ]
        self.action_names = self.joint_names
        self.foot_names = ['FL', 'FR', 'HL', 'HR']
        self.foot_site_names = ['FL_elbow_site', 'FR_elbow_site', 'HL_knee_site', 'HR_knee_site']
        self._foot_site_ids = None
        self._joint_ids_resolved = None   # ← 新增: 按名称缓存的关节 id

        # 参考表 12 列顺序: 前腿(4) + 后腿(4) + 脊柱(4)
        self.ref_names = ['FL_shoulder', 'FL_elbow', 'FR_shoulder', 'FR_elbow',
                          'HL_hip', 'HL_knee', 'HR_hip', 'HR_knee',
                          'F_spine1', 'F_body', 'H_spine1', 'H_body']

    def record_step_data(self, env, actions=None, rewards=None, dones=None):
        record = {'step': float(self.step_count)}
        unwrapped = env.unwrapped
        asset = unwrapped.scene["robot"]
        resolve_model_indices(asset)
        idx = 0

        # ---------- 动作 ----------
        if actions is not None:
            for i in range(actions.shape[1]):
                name = self.action_names[i] if i < len(self.action_names) else f'action_{i}'
                record[f'{name}_action'] = float(actions[0, i].item())

        # ---------- 足端接触力与位置 ----------
        contact_sensor = unwrapped.scene["feet_ground_contact"]
        feet_contact = contact_sensor.data.force.flatten(start_dim=1)
        if self._foot_site_ids is None:
            self._foot_site_ids, _ = asset.find_sites(self.foot_site_names, preserve_order=True)
        foot_pos = asset.data.site_pos_w[idx, self._foot_site_ids]
        for i, name in enumerate(self.foot_names):
            force = feet_contact[idx, i * 3: i * 3 + 3]
            record[f'contact_{name}_mag'] = float(torch.norm(force).item())
            record[f'contact_{name}_x'] = float(force[0].item())
            record[f'contact_{name}_y'] = float(force[1].item())
            record[f'contact_{name}_z'] = float(force[2].item())
            record[f'foot_{name}_x'] = float(foot_pos[i, 0].item())
            record[f'foot_{name}_y'] = float(foot_pos[i, 1].item())
            record[f'foot_{name}_z'] = float(foot_pos[i, 2].item())

        # ---------- 基座位姿/速度/角速度 ----------
        base_pos = asset.data.root_link_pos_w[idx]
        base_lin_vel = asset.data.root_link_lin_vel_w[idx]
        base_ang_vel = asset.data.root_link_ang_vel_w[idx]
        record['base_pos_x'] = float(base_pos[0].item())
        record['base_pos_y'] = float(base_pos[1].item())
        record['base_pos_z'] = float(base_pos[2].item())
        record['base_vel_x'] = float(base_lin_vel[0].item())
        record['base_vel_y'] = float(base_lin_vel[1].item())
        record['base_vel_z'] = float(base_lin_vel[2].item())
        # 分析脚本使用的别名
        record['base_lin_vel_x'] = float(base_lin_vel[0].item())
        record['base_lin_vel_y'] = float(base_lin_vel[1].item())
        record['base_lin_vel_z'] = float(base_lin_vel[2].item())
        record['base_ang_vel_x'] = float(base_ang_vel[0].item())
        record['base_ang_vel_y'] = float(base_ang_vel[1].item())
        record['base_ang_vel_z'] = float(base_ang_vel[2].item())

        # ---------- 朝向 ----------
        def _quat_to_yaw(q):
            w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        f_quat = asset.data.body_link_quat_w[idx, _MODEL_INDICES.f_body_id]
        h_quat = asset.data.body_link_quat_w[idx, _MODEL_INDICES.h_body_id]
        f_body_yaw = _quat_to_yaw(f_quat)
        h_body_yaw = _quat_to_yaw(h_quat)
        record['f_body_heading'] = f_body_yaw
        record['h_body_heading'] = h_body_yaw
        # 分析脚本回退分支用的 'heading': 用根链接四元数算; 无该属性时退用 f_body_yaw
        if hasattr(asset.data, 'root_link_quat_w'):
            record['heading'] = _quat_to_yaw(asset.data.root_link_quat_w[idx])
        else:
            record['heading'] = f_body_yaw

        # ---------- 前后躯干高度 ----------
        f_height = float(asset.data.body_link_pos_w[idx, _MODEL_INDICES.f_body_id, 2].item())
        h_height = float(asset.data.body_link_pos_w[idx, _MODEL_INDICES.h_body_id, 2].item())
        record['F_body_height'] = f_height
        record['H_body_height'] = h_height

        # ---------- 命令 ----------
        command = unwrapped.command_manager.get_command("hole_cmd")[idx]
        for i, name in enumerate(['vel_command_x', 'vel_command_y', 'vel_command_z',
                                  'height_F_command', 'height_H_command', 'angle_command']):
            record[name] = float(command[i].item())
        record['mode'] = mode_of(float(command[3]), float(command[4]))  # type: ignore
        record['height_F_error'] = f_height - float(command[3].item())
        record['height_H_error'] = h_height - float(command[4].item())
        # 步态基频 (Hole 任务参考表固定 2.0 Hz, 与 config.BASE_FREQ 一致)
        record['gait_freq_command'] = 2.0

        # ---------- 每个关节的 pos / vel / acc / torque ----------
        joint_ids = _MODEL_INDICES.joint_ids
        joint_pos_all = asset.data.joint_pos[idx, joint_ids]
        joint_vel_all = asset.data.joint_vel[idx, joint_ids]
        joint_acc_all = (
            asset.data.joint_acc[idx, joint_ids]
            if hasattr(asset.data, 'joint_acc') else None
        )
        # actuator_force 的顺序也是 actuator 顺序，与 self.joint_names 一一对应
        actuator_force_all = asset.data.actuator_force[idx]

        for i, name in enumerate(self.joint_names):
            record[f'{name}_pos'] = float(joint_pos_all[i].item())
            record[f'{name}_vel'] = float(joint_vel_all[i].item())
            if joint_acc_all is not None:
                record[f'{name}_acc'] = float(joint_acc_all[i].item())
            if i < actuator_force_all.shape[0]:
                record[f'{name}_torque'] = float(actuator_force_all[i].item())

        # ---------- 参考 ----------
        ref_pos_all, ref_vel_all = get_reference_joint_state(unwrapped)
        ref_pos = ref_pos_all[idx]
        ref_vel = ref_vel_all[idx]
        for i, name in enumerate(self.ref_names):
            record[f'{name}_ref_pos'] = float(ref_pos[i].item())
            record[f'{name}_ref_vel'] = float(ref_vel[i].item())

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
        print(f"[INFO] 数据已保存到: {csv_path} ({len(self.data_records)} 步 × {len(df.columns)} 列)")


class DataRecordingEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, clip_actions=None, data_recorder=None, action_scale=1.0):
        super().__init__(env, clip_actions)
        self.data_recorder = data_recorder
        self.action_scale = action_scale

    def step(self, actions):
        scaled_actions = actions * self.action_scale
        obs_dict, rew, dones, extras = super().step(scaled_actions)
        if self.data_recorder:
            self.data_recorder.record_step_data(self.env, actions=scaled_actions,
                                                rewards=rew, dones=dones)
        return obs_dict, rew, dones, extras

    # 推理步: 观测历史缓冲区的 in-place 写入不支持 autograd, 必须关梯度
    def step_inference(self, actions):
        with torch.inference_mode():
            return self.step(actions)


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
                raise FileNotFoundError(f"未找到 checkpoint 文件: {resume_path}")
            video_name = extract_video_name_from_checkpoint(resume_path)
        elif cfg.wandb_run_path is not None:
            resume_path, _ = get_wandb_checkpoint_path(
                log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name)
            video_name = extract_video_name_from_checkpoint(resume_path)
        else:
            print("请输入 --checkpoint-file 路径:")
            text = input().strip().strip('"').strip("'")
            if not text:
                raise ValueError("必须提供 checkpoint 文件路径")
            resume_path = Path(text)
            if not resume_path.exists():
                raise FileNotFoundError(f"未找到 checkpoint 文件: {resume_path}")
            video_name = extract_video_name_from_checkpoint(resume_path)
        log_dir = resume_path.parent
    else:
        log_dir = (Path("logs") / "rsl_rl" / TASK_NAME.replace("Mjlab-", "").replace("-", "_")
                   / "dummy").resolve()
        video_name = f"dummy_{cfg.agent}"

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    # 命令来源 (play 侧最高优先级): 显式覆盖 env_cfg 里的位置表设定。
    # schedule = 位置表 (fixed_* 仍可覆盖对应字段); fixed = 锁死给定值; random = 按阶段随机采样
    # fixed 模式补默认高度 (其余模式不给默认, 避免把 schedule 位置表或 random 采样锁死)
    fixed_hF = cfg.fixed_height_F
    fixed_hH = cfg.fixed_height_H
    if cfg.command_source == "fixed":
        fixed_hF = 0.055 if fixed_hF is None else fixed_hF
        fixed_hH = 0.055 if fixed_hH is None else fixed_hH

    cmd_cfg = env_cfg.commands.get("hole_cmd")
    if cmd_cfg is not None:
        if cfg.command_source != "schedule":
            # 关掉配置里的位置表与时间表, 并禁用 stage>=3 的内置兜底
            cmd_cfg.use_position_schedule = False  # type: ignore[attr-defined]
            cmd_cfg.position_schedule = None  # type: ignore[attr-defined]
            cmd_cfg.use_height_schedule = False  # type: ignore[attr-defined]
            cmd_cfg.height_schedule = None  # type: ignore[attr-defined]
            cmd_cfg.stage_schedule_fallback = False  # type: ignore[attr-defined]
        # 显式给出的字段一律写入 (schedule 下在位置表算完后覆盖, random 下只固定这些字段)
        if cfg.fixed_velocity is not None:
            cmd_cfg.fixed_velocity = cfg.fixed_velocity  # type: ignore[attr-defined]
        if fixed_hF is not None:
            cmd_cfg.fixed_height_F = fixed_hF  # type: ignore[attr-defined]
        if fixed_hH is not None:
            cmd_cfg.fixed_height_H = fixed_hH  # type: ignore[attr-defined]
        print(f"[INFO] 命令来源 = {cfg.command_source}"
              + (f" (固定 h_F={fixed_hF}, h_H={fixed_hH}, vel={cfg.fixed_velocity})"
                 if cfg.command_source == "fixed" else
                 (" (8 段位移位置表)" if cfg.command_source == "schedule" else " (按阶段随机采样)"))
              + ("" if cfg.command_source == "fixed" else
                 f" [显式覆盖: h_F={fixed_hF}, h_H={fixed_hH}, vel={cfg.fixed_velocity}]"
                 if (fixed_hF is not None or fixed_hH is not None
                     or cfg.fixed_velocity is not None) else ""))

    # 在建环境前统一解析"最终阶段"与"碰撞开关", 供碰撞与后续课程对齐共用
    saved_hole = read_hole_state(resume_path) if resume_path is not None else {}
    if cfg.stage is not None:
        final_step, stage_src = {1: 0, 2: STAGE1_END, 3: STAGE2_END, 4: STAGE3_END}[cfg.stage], "命令行 --stage"
    elif resume_path is not None:
        recorded = read_env_step(resume_path)
        if recorded is not None:
            final_step, stage_src = recorded, "检查点 env_state"
        else:
            train_iter = extract_iter_from_checkpoint(resume_path)
            final_step, stage_src = max(0, train_iter - 10) * STEPS_PER_ITER, "检查点文件名轮次"
    else:
        final_step, stage_src = 0, "默认 (无检查点)"
    final_stage = get_current_stage(final_step)

    # 碰撞优先级: 显式 > 检查点记录 > 按上面解析出的最终阶段推断
    if cfg.enable_collision is not None:
        collision, coll_src = cfg.enable_collision, "命令行显式指定"
    elif saved_hole.get("collision") is not None:
        collision, coll_src = bool(saved_hole["collision"]), "检查点 hole_state"
    else:
        collision, coll_src = stage_requires_collision(final_stage), f"按阶段推断 (stage {final_stage})"
    configure_hole_entities(env_cfg, enable_collision=collision)
    # 回放: 应用检查点保存的板几何与接触参数 (重建时保留自定义几何的同一套口径)
    applied = apply_saved_layout(env_cfg, saved_hole.get("layout") or [])
    # 障碍物可见性: 仅改 rgba, 不改碰撞; 放在几何应用之后以免被覆盖
    n_hidden = set_obstacle_visibility(env_cfg, visible=cfg.show_obstacles)
    print(f"[INFO] 阶段解析: step {final_step} → stage {final_stage} (来源: {stage_src})")
    print(f"[INFO] 限高板碰撞 = {'开' if collision else '关'} (来源: {coll_src})"
          + (f"; 已应用检查点几何 {applied} 块板" if applied else ""))
    if not cfg.show_obstacles:
        print(f"[INFO] 限高板已隐藏 (rgba=0,0,0,0, {n_hidden} 块); 碰撞不受影响")
    if saved_hole:
        print(f"[INFO] 检查点 hole_state: 版本={saved_hole.get('version')}, "
              f"stage={saved_hole.get('stage')}, 记录碰撞={saved_hole.get('collision')}")

    # 构建输出名后缀: command_source 始终带上 (三选一), 其余只记录实际生效的非默认配置
    suffix_parts = [cfg.command_source]
    if fixed_hF is not None:
        suffix_parts.append(f"hF{fixed_hF * 1000:.0f}")
    if fixed_hH is not None:
        suffix_parts.append(f"hH{fixed_hH * 1000:.0f}")
    if cfg.fixed_velocity is not None:
        suffix_parts.append(f"v{cfg.fixed_velocity:.2f}")
    if cfg.stage is not None:
        suffix_parts.append(f"s{cfg.stage}")
    if cfg.enable_collision is not None:
        suffix_parts.append("col1" if cfg.enable_collision else "col0")
    if not cfg.show_obstacles:
        suffix_parts.append("noobs")
    suffix = f"-{'-'.join(suffix_parts)}" if suffix_parts else ""
    if video_name is not None:
        video_name = f"{video_name}{suffix}"

    render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
    if cfg.video and DUMMY_MODE:
        print("[WARN] 虚拟智能体的视频录制已禁用")

    # render_mode 必须传进环境, 否则 VideoRecorder 抓不到帧 (rgb_array 才录)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    # 阶段对齐延后到 actor 加载之后 (基类 load 会用检查点的 common_step_counter 覆盖)

    data_recorder = None
    if cfg.record_data:
        assert log_dir is not None
        log_dir = resolve_output_dir(log_dir)
        data_recorder = JointDataRecorder(log_dir, video_name, num_envs=env_cfg.scene.num_envs)
        print("[INFO] 启用关节数据记录")

    if TRAINED_MODE and cfg.video:
        assert log_dir is not None and video_name is not None
        env = VideoRecorder(
            env,
            video_folder=log_dir / "videos",
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            name_prefix=video_name,
            disable_logger=False,
        )
        print("[INFO] 播放期间录制视频")

    env = DataRecordingEnvWrapper(env, clip_actions=agent_cfg.clip_actions,
                                  data_recorder=data_recorder, action_scale=1.0)

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
        runner = runner_cls(env, asdict(agent_cfg), str(log_dir), device=device)
        runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)
        # 基类 load 会恢复检查点的 common_step_counter, 显式阶段必须在它之后最终生效
        env = runner.env

    # 阶段对齐: 用建环境前就解析好的 final_step (与碰撞来源同一口径);
    # 基类 load 会用检查点的 common_step_counter 覆盖, 所以必须放在加载之后
    env.unwrapped.common_step_counter = final_step
    print(f"[INFO] 阶段对齐到 step {final_step} → stage {get_current_stage(final_step)}"
          + (" (命令行 --stage 强制)" if cfg.stage is not None else " (按检查点记录)"))

    if cfg.smoke_steps is not None:
        # 无窗自检: 推 N 步打印命令/高度统计
        obs_dict = env.reset()
        obs_in = obs_dict[0] if isinstance(obs_dict, tuple) else obs_dict
        cmd_hist = []
        for _ in range(cfg.smoke_steps):
            act = policy(obs_in)
            obs_dict, rew, dones, extras = env.step_inference(act) # type: ignore
            obs_in = obs_dict[0] if isinstance(obs_dict, tuple) else obs_dict
            cmd_hist.append(env.unwrapped.command_manager.get_command("hole_cmd")[0].clone()) # type: ignore
        cmds = torch.stack(cmd_hist)
        robot = env.unwrapped.scene["robot"]
        print(f"[INFO] 自检 {cfg.smoke_steps} 步完成")
        print(f"[INFO] 命令: vel_x {float(cmds[:, 0].mean()):.3f} m/s, "
              f"h_F {float(cmds[:, 3].mean()) * 1000:.1f} mm, h_H {float(cmds[:, 4].mean()) * 1000:.1f} mm")
        print(f"[INFO] 实测: 前躯干 {float(robot.data.body_link_pos_w[0, _MODEL_INDICES.f_body_id, 2]) * 1000:.1f} mm, "
              f"后躯干 {float(robot.data.body_link_pos_w[0, _MODEL_INDICES.h_body_id, 2]) * 1000:.1f} mm, "
              f"位移 {float(robot.data.root_link_pos_w[0, 0]) * 1000:.1f} mm")
        print(f"[INFO] 最近一步奖励: {float(rew[0]):.4f}")  # type: ignore
        if data_recorder:
            data_recorder.save_to_csv()
        env.close()
        return

    try:
        viewer = NativeMujocoViewer(env, policy)
        viewer.run()
    except KeyboardInterrupt:
        print("[INFO] 用户中断播放")
    finally:
        if data_recorder:
            data_recorder.save_to_csv()
        env.close()


def main():
    args = tyro.cli(PlayConfig, description="播放 SQuRo Hole 智能体 (虚拟碰撞版)")
    run_play(args)


if __name__ == "__main__":
    main()
