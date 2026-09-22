# uv run python -B -m mjlab.scripts.SQuRo_Tunnel_play --checkpoint_file <path>
# uv run python -B -m mjlab.scripts.SQuRo_Tunnel_play --agent zero   (无策略虚拟智能体)

import re
import tyro
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
from mjlab.tasks.SQuRo_Tunnel.mdp.curriculums import _STEPS_PER_ITER, get_training_phase
from mjlab.tasks.SQuRo_Tunnel.mdp.indices import _MODEL_INDICES, resolve_model_indices
from mjlab.tasks.SQuRo_Tunnel.mdp.path import OBSTACLE_LENGTH, TUNNEL_BOTTOM
from mjlab.tasks.SQuRo_Tunnel.mdp.reference import get_reference_joint_state


TASK_NAME = "Mjlab-SQuRo-Tunnel"
PLATE_HALF_THICKNESS = 0.005      # 限高板半厚 (= size[2], 与 env_cfg 占位实体一致)
PLATE_HALF_WIDTH = 0.1            # 限高板半宽 (= size[1])


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
    # Tunnel 任务相关配置
    fixed_tunnel_xs: str | None = "0.30,0.60,0.90"   # 固定洞位置 (洞左侧, 逗号分隔); 传 "sample" 则每 episode 随机采样
    fixed_velocity: float | None = None
    fixed_height_f: float | None = None
    fixed_height_h: float | None = None
    fixed_gait_freq: float | None = 1.0
    enable_collision: bool = False                    # 限高板是否参与碰撞 (默认关闭, 与训练一致)
    smoke_steps: int | None = None                    # 无窗自检: 只跑 N 步打印统计后退出 (不启动 viewer)


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


# 解析固定洞位置字符串 "0.30,0.60,0.90" -> tuple
def parse_fixed_tunnel_xs(text: str | None) -> tuple[float, ...] | None:
    if text is None or text.strip().lower() == "sample":
        return None
    return tuple(float(v) for v in text.split(",") if v.strip())


# 选择可写输出目录: 优先 logs 下, 受限环境无法新建目录时退回系统临时目录
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


class JointDataRecorder:
    def __init__(self, log_dir, video_name, num_envs=1):
        self.log_dir = log_dir
        self.video_name = video_name
        self.num_envs = num_envs
        self.data_records = []
        self.step_count = 0

        # 14 个驱动关节 — 顺序必须与 entity actuator 顺序一致
        self.joint_names = [
            'F_spine1', 'F_body',
            'Neck_yaw', 'Neck_pitch',
            'FL_shoulder', 'FL_elbow',
            'FR_shoulder', 'FR_elbow',
            'H_spine1', 'H_body',
            'HL_hip', 'HL_knee',
            'HR_hip', 'HR_knee',
        ]
        self._joint_ids_resolved = False
        self.action_names = self.joint_names  # 与 joint_names 同序
        self.foot_names = ['FL', 'FR', 'HL', 'HR']
        self.foot_site_names = ['FL_elbow_site', 'FR_elbow_site', 'HL_knee_site', 'HR_knee_site']
        self._foot_site_ids = None

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
        feet_contact = contact_sensor.data.force.flatten(start_dim=1)
        for i, name in enumerate(self.foot_names):
            record[f'contact_{name}_x'] = float(feet_contact[env_idx, i * 3].item())
            record[f'contact_{name}_y'] = float(feet_contact[env_idx, i * 3 + 1].item())
            record[f'contact_{name}_z'] = float(feet_contact[env_idx, i * 3 + 2].item())
            force_mag = torch.norm(feet_contact[env_idx, i * 3: i * 3 + 3]).item()
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
        joint_pos = jp[env_idx, joint_ids]  # type: ignore[call-overload]
        joint_vel = jv[env_idx, joint_ids]  # type: ignore[call-overload]
        joint_acc = ja[env_idx, joint_ids]  # type: ignore[call-overload]
        actuator_force = asset.data.actuator_force[env_idx]

        # 基座位置
        base_pos = asset.data.root_link_pos_w[env_idx]
        base_lin_vel_w = asset.data.root_link_lin_vel_w[env_idx]
        base_ang_vel_w = asset.data.root_link_ang_vel_w[env_idx]

        # 参考关节位置/速度
        ref_joint_pos, ref_joint_vel = get_reference_joint_state(unwrapped)
        ref_joint_pos = ref_joint_pos[env_idx]
        ref_joint_vel = ref_joint_vel[env_idx]

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

        # 朝向: base body+X heading + F/H 物理前向 heading (直行任务需监控侧偏)
        from mjlab.tasks.SQuRo_Tunnel.mdp.path import (
            get_body_heading,
            get_f_body_physical_heading,
            get_h_body_physical_heading,
        )
        record['base_heading'] = float(asset.data.heading_w[env_idx].item())
        record['base_raw_heading'] = float(get_body_heading(unwrapped)[env_idx].item())
        f_heading = get_f_body_physical_heading(unwrapped)[env_idx]
        h_heading = get_h_body_physical_heading(unwrapped)[env_idx]
        record['f_body_heading'] = float(f_heading.item())
        record['h_body_heading'] = float(h_heading.item())

        # 控制命令 [vel_x, height_f, height_h, gait_freq, curvature]
        command = unwrapped.command_manager.get_command("tunnel_cmd")[env_idx]
        command_names = [
            'vel_command_x', 'height_f_command', 'height_h_command',
            'gait_freq_command', 'curvature_command',
        ]
        for i, name in enumerate(command_names):
            record[name] = float(command[i].item())

        # Tunnel 专用: 前后肢高度 (实际/期望/误差) + 走廊超额
        from mjlab.tasks.SQuRo_Tunnel.mdp.path import (
            get_front_center_height,
            get_rear_center_height,
            compute_corridor_front_excess,
            compute_corridor_rear_excess,
        )
        body_pos_w = asset.data.body_link_pos_w
        x_f = body_pos_w[:, _MODEL_INDICES.f_body_id, 0]
        z_f = body_pos_w[:, _MODEL_INDICES.f_body_id, 2]
        x_h = body_pos_w[:, _MODEL_INDICES.h_body_id, 0]
        z_h = body_pos_w[:, _MODEL_INDICES.h_body_id, 2]
        z_f_ref = get_front_center_height(unwrapped, x_f)[env_idx]
        z_h_ref = get_rear_center_height(unwrapped, x_h)[env_idx]
        record['f_body_pos_x'] = float(x_f[env_idx].item())
        record['f_body_pos_z'] = float(z_f[env_idx].item())
        record['h_body_pos_x'] = float(x_h[env_idx].item())
        record['h_body_pos_z'] = float(z_h[env_idx].item())
        record['front_z_ref'] = float(z_f_ref.item())
        record['rear_z_ref'] = float(z_h_ref.item())
        record['front_z_err'] = float((z_f[env_idx] - z_f_ref).item())
        record['rear_z_err'] = float((z_h[env_idx] - z_h_ref).item())
        record['corridor_front_excess'] = float(compute_corridor_front_excess(unwrapped)[env_idx].item())
        record['corridor_rear_excess'] = float(compute_corridor_rear_excess(unwrapped)[env_idx].item())

        # 当前 episode 的洞位置 (洞左侧 x)
        tunnel_xs = getattr(unwrapped, "_tunnel_xs", None)
        if tunnel_xs is not None:
            for i in range(tunnel_xs.shape[1]):
                record[f'tunnel{i}_x'] = float(tunnel_xs[env_idx, i].item())

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


# 按真实洞位置重建限高板实体 (覆盖 env_cfg 中的单个占位洞)
def rebuild_tunnel_entities(env_cfg, tunnel_xs: tuple[float, ...] | None, enable_collision: bool) -> None:
    from mjlab.tasks.SQuRo_Tunnel.mdp.entity import HoleEntityCfg

    if tunnel_xs is None:
        print("[INFO] 洞位置设为随机采样, 场景限高板保持占位实体 (位置与实际洞不一致, 仅作视觉提示)")
        return

    col = 1 if enable_collision else 0
    tunnel_dict = {}
    for i, x in enumerate(tunnel_xs):
        tunnel_dict[f"tunnel{i}"] = HoleEntityCfg(
            name=f"tunnel{i}",
            position=(x, 0.0, TUNNEL_BOTTOM),
            size=(OBSTACLE_LENGTH / 2, PLATE_HALF_WIDTH, PLATE_HALF_THICKNESS),
            contype=col,
            conaffinity=col,
        )
    # 覆盖 env_cfg 里的 "tunnel1" 占位实体, 保证板与实际高度轨迹对齐
    entities = {k: v for k, v in env_cfg.scene.entities.items() if not k.startswith("tunnel")}
    env_cfg.scene.entities = {**entities, **tunnel_dict}  # type: ignore[index]
    print(f"[INFO] 已按固定洞位置重建限高板: {[round(x, 3) for x in tunnel_xs]} (碰撞={'开' if enable_collision else '关'})")


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
        elif cfg.wandb_run_path is not None:
            resume_path, was_cached = get_wandb_checkpoint_path(
                log_root_path,
                Path(cfg.wandb_run_path),
                cfg.wandb_checkpoint_name
            )
            video_name = extract_video_name_from_checkpoint(resume_path)
        else:
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

        log_dir = resume_path.parent
    else:
        # 虚拟智能体: 无 checkpoint, 产物 (CSV) 落到 logs/rsl_rl/SQuRo_Tunnel/dummy
        log_dir = (Path("logs") / "rsl_rl" / TASK_NAME.replace("Mjlab-", "").replace("-", "_") / "dummy").resolve()
        video_name = f"dummy_{cfg.agent}"

    # 设置环境参数
    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    # 自动识别训练阶段, train_iter 仅用于提取, 实际阶段判断用 align_iter (与 env 内部一致)
    train_iter = 0
    align_iter = 0
    if TRAINED_MODE and resume_path is not None:
        train_iter = extract_iter_from_checkpoint(resume_path)
        align_iter = max(0, train_iter - 10)  # -10 避免边界效应, 与 env 内部对齐

    align_step = align_iter * _STEPS_PER_ITER
    is_tunnel_phase = get_training_phase(align_step) == 1

    fixed_tunnel_xs = parse_fixed_tunnel_xs(cfg.fixed_tunnel_xs)

    # 命令固定值覆盖 — Phase0 高度随机(可用 fixed_height 锁死), Phase1 高度由轨迹动态生成
    cmd_cfg = env_cfg.commands.get("tunnel_cmd")
    if cmd_cfg is not None and TRAINED_MODE:
        cmd_cfg.fixed_tunnel_xs = fixed_tunnel_xs  # type: ignore[attr-defined]
        if cfg.fixed_velocity is not None:
            cmd_cfg.fixed_velocity = cfg.fixed_velocity  # type: ignore
        if cfg.fixed_gait_freq is not None:
            cmd_cfg.fixed_gait_freq = cfg.fixed_gait_freq  # type: ignore
        if not is_tunnel_phase:
            # Phase 0: 无洞, 高度用固定值便于观察 (默认不覆盖, 保留训练期随机高度)
            if cfg.fixed_height_f is not None:
                cmd_cfg.fixed_height_f = cfg.fixed_height_f  # type: ignore
            if cfg.fixed_height_h is not None:
                cmd_cfg.fixed_height_h = cfg.fixed_height_h  # type: ignore

    # 构建文件名后缀
    cmd_suffix_parts = []
    if is_tunnel_phase:
        if fixed_tunnel_xs is not None:
            cmd_suffix_parts.append("tn" + "-".join(f"{x:.2f}" for x in fixed_tunnel_xs))
        else:
            cmd_suffix_parts.append("tnrand")
    else:
        cmd_suffix_parts.append("phase0")
    cmd_suffix = f"-{'-'.join(cmd_suffix_parts)}"
    if video_name is not None:
        video_name = f"{video_name}{cmd_suffix}"

    # 按固定洞位置重建限高板 (Phase1 有效, Phase0 无洞则隐藏)
    if not is_tunnel_phase:
        rebuild_tunnel_entities(env_cfg, None, cfg.enable_collision)
        print("[INFO] Phase0 (无洞) — 场景仅保留占位板, 高度命令为随机/固定值")
    else:
        rebuild_tunnel_entities(env_cfg, fixed_tunnel_xs, cfg.enable_collision)

    # 创建环境
    render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
    if cfg.video and DUMMY_MODE:
        print("[WARN] 虚拟智能体的视频录制已禁用")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    # 对齐 curriculum 阶段
    if TRAINED_MODE and resume_path is not None and train_iter > 0:
        env.common_step_counter = align_step
        phase_name = "钻洞阶段" if is_tunnel_phase else "无洞阶段"
        print(f"[INFO] curriculum 对齐到 iter {align_iter} (step {align_step}) [{phase_name}]")

    # 确认本 episode 的洞位置
    tunnel_xs_env = getattr(env, "_tunnel_xs", None)
    if tunnel_xs_env is not None and env.num_envs >= 1:
        print(f"[INFO] 当前 episode 洞位置 (env0): {[round(float(v), 3) for v in tunnel_xs_env[0].tolist()]}")

    # 初始化数据记录器
    data_recorder = None
    if cfg.record_data:
        # 虚拟智能体也允许记录 (CSV 落到 dummy 目录, 便于无 checkpoint 时验证脚本与轨迹)
        assert log_dir is not None
        print("[INFO] 启用关节数据记录")
        log_dir = resolve_output_dir(log_dir)
        data_recorder = JointDataRecorder(log_dir, video_name, num_envs=env_cfg.scene.num_envs)
    # 添加视频录制器
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
        policy = runner.get_inference_policy(device=device)

    # 运行查看器
    if cfg.smoke_steps is not None:
        # 无窗自检路径: 直接推 N 步, 打印高度/命令统计, 不启动 viewer
        obs_dict = env.reset()
        obs_in = obs_dict[0] if isinstance(obs_dict, tuple) else obs_dict
        for _ in range(cfg.smoke_steps):
            act = policy(obs_in)
            obs_dict, rew, dones, extras = env.step(act)
            obs_in = obs_dict[0] if isinstance(obs_dict, tuple) else obs_dict
        from mjlab.tasks.SQuRo_Tunnel.mdp.path import (
            get_front_center_height,
            get_rear_center_height,
        )
        robot = env.unwrapped.scene["robot"]
        body_pos_w = robot.data.body_link_pos_w
        x_f = body_pos_w[:, _MODEL_INDICES.f_body_id, 0]
        x_h = body_pos_w[:, _MODEL_INDICES.h_body_id, 0]
        z_f = body_pos_w[:, _MODEL_INDICES.f_body_id, 2]
        z_h = body_pos_w[:, _MODEL_INDICES.h_body_id, 2]
        z_f_ref = get_front_center_height(env.unwrapped, x_f)
        z_h_ref = get_rear_center_height(env.unwrapped, x_h)
        print(f"[INFO] 自检 {cfg.smoke_steps} 步完成")
        print(f"[INFO] 前肢中心 x={float(x_f[0]):.4f} z={float(z_f[0]):.4f} 期望={float(z_f_ref[0]):.4f}")
        print(f"[INFO] 后肢中心 x={float(x_h[0]):.4f} z={float(z_h[0]):.4f} 期望={float(z_h_ref[0]):.4f}")
        print(f"[INFO] 最近一步奖励: {float(rew[0]):.4f}")
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
    args = tyro.cli(PlayConfig, description="播放 SQuRo Tunnel 智能体")
    run_play(args)


if __name__ == "__main__":
    main()
