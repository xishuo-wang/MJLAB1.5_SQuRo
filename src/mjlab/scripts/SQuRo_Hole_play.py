# uv run python -B -m mjlab.scripts.SQuRo_Hole_play --checkpoint_file <path>
# uv run python -B -m mjlab.scripts.SQuRo_Hole_play --agent zero --smoke_steps 50 --no-video

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
from mjlab.tasks.SQuRo_Hole.mdp.command import HEIGHT_THRESHOLD
from mjlab.tasks.SQuRo_Hole.mdp.indices import _MODEL_INDICES, resolve_model_indices
from mjlab.tasks.SQuRo_Hole.mdp.reference import (
    get_reference_joint_pos,
    get_reference_joint_vel,
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
    # Hole 任务: 命令固定值 (None 表示用 cfg 里的位置表/随机采样)
    fixed_velocity: float | None = None
    fixed_height_F: float | None = None
    fixed_height_H: float | None = None
    enable_collision: bool | None = None   # None = 按 cfg (训练默认关, 阶段 4 才开)
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
    low_F = height_F < HEIGHT_THRESHOLD
    low_H = height_H < HEIGHT_THRESHOLD
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

        if actions is not None:
            for i in range(actions.shape[1]):
                name = self.action_names[i] if i < len(self.action_names) else f'action_{i}'
                record[f'{name}_action'] = float(actions[0, i].item())

        # 足端接触力与位置
        contact_sensor = unwrapped.scene["feet_ground_contact"]
        feet_contact = contact_sensor.data.force.flatten(start_dim=1)
        if self._foot_site_ids is None:
            self._foot_site_ids, _ = asset.find_sites(self.foot_site_names, preserve_order=True)
        foot_pos = asset.data.site_pos_w[idx, self._foot_site_ids]
        for i, name in enumerate(self.foot_names):
            force = feet_contact[idx, i * 3: i * 3 + 3]
            record[f'contact_{name}_mag'] = float(torch.norm(force).item())
            record[f'foot_{name}_x'] = float(foot_pos[i, 0].item())
            record[f'foot_{name}_z'] = float(foot_pos[i, 2].item())

        # 36 个关节的全量状态 (旧版观测口径)
        record['base_pos_x'] = float(asset.data.root_link_pos_w[idx, 0].item())
        record['base_pos_y'] = float(asset.data.root_link_pos_w[idx, 1].item())
        record['base_pos_z'] = float(asset.data.root_link_pos_w[idx, 2].item())
        record['base_vel_x'] = float(asset.data.root_link_lin_vel_w[idx, 0].item())
        record['base_vel_y'] = float(asset.data.root_link_lin_vel_w[idx, 1].item())
        record['base_vel_z'] = float(asset.data.root_link_lin_vel_w[idx, 2].item())

        # 前后躯干高度 (奖励口径: F_body / H_body)
        f_height = float(asset.data.body_link_pos_w[idx, _MODEL_INDICES.f_body_id, 2].item())
        h_height = float(asset.data.body_link_pos_w[idx, _MODEL_INDICES.h_body_id, 2].item())
        record['F_body_height'] = f_height
        record['H_body_height'] = h_height

        # 命令与参考
        command = unwrapped.command_manager.get_command("hole_cmd")[idx]
        for i, name in enumerate(['vel_command_x', 'vel_command_y', 'vel_command_z',
                                  'height_F_command', 'height_H_command', 'angle_command']):
            record[name] = float(command[i].item())
        record['mode'] = mode_of(float(command[3]), float(command[4]))
        record['height_F_error'] = f_height - float(command[3].item())
        record['height_H_error'] = h_height - float(command[4].item())

        ref_pos = get_reference_joint_pos(unwrapped)[idx]
        ref_vel = get_reference_joint_vel(unwrapped)[idx]
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

    # 命令固定值覆盖 (回放时锁死高度/速度, 便于单帧对比)
    cmd_cfg = env_cfg.commands.get("hole_cmd")
    if cmd_cfg is not None and TRAINED_MODE:
        if cfg.fixed_velocity is not None:
            cmd_cfg.fixed_velocity = cfg.fixed_velocity  # type: ignore[attr-defined]
        if cfg.fixed_height_F is not None:
            cmd_cfg.fixed_height_F = cfg.fixed_height_F  # type: ignore[attr-defined]
        if cfg.fixed_height_H is not None:
            cmd_cfg.fixed_height_H = cfg.fixed_height_H  # type: ignore[attr-defined]

    # 限高板碰撞开关 (编译期固化): 显式传入优先, 否则用 cfg 默认
    if cfg.enable_collision is not None:
        from mjlab.tasks.SQuRo_Hole.SQuRo_Hole_env_cfg import configure_hole_collision
        configure_hole_collision(env_cfg, enable_collision=cfg.enable_collision)
        print(f"[INFO] 限高板碰撞 = {'开' if cfg.enable_collision else '关'}")

    suffix = f"-it{extract_iter_from_checkpoint(resume_path)}" if resume_path else ""
    if video_name is not None:
        video_name = f"{video_name}{suffix}"

    render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
    if cfg.video and DUMMY_MODE:
        print("[WARN] 虚拟智能体的视频录制已禁用")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

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

    if cfg.smoke_steps is not None:
        # 无窗自检: 推 N 步打印命令/高度统计
        obs_dict = env.reset()
        obs_in = obs_dict[0] if isinstance(obs_dict, tuple) else obs_dict
        cmd_hist = []
        for _ in range(cfg.smoke_steps):
            act = policy(obs_in)
            obs_dict, rew, dones, extras = env.step(act)
            obs_in = obs_dict[0] if isinstance(obs_dict, tuple) else obs_dict
            cmd_hist.append(env.unwrapped.command_manager.get_command("hole_cmd")[0].clone())
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
