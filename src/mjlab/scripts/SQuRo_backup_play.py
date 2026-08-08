"""SQuRo Backup 跌倒爬起策略回放脚本。

加载训练好的策略, 固定参考时间缩放 λ (demo 用 1.4), 统计复位时间并可选录制视频。

用法:
    uv run python src/mjlab/scripts/SQuRo_backup_play.py --checkpoint-file <model_XXX.pt>
    uv run python src/mjlab/scripts/SQuRo_backup_play.py --checkpoint-file <model_XXX.pt> --time-scale 1.4 --video
    uv run python src/mjlab/scripts/SQuRo_backup_play.py --checkpoint-file <model_XXX.pt> --agent zero

输出: 每 episode 的复位时间 (从跌倒到站稳 0.5s), 平均复位时间与成功率; 可选 mp4 视频。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import tyro
import torch

import mjlab.tasks  # noqa: F401  触发任务注册
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


TASK_NAME = "Mjlab-SQuRo-Backup"


@dataclass(frozen=True)
class BackupPlayConfig:
    agent: Literal["trained", "zero", "random"] = "trained"
    checkpoint_file: str | None = None
    """训练 checkpoint 路径 (trained 模式必填)。"""
    time_scale: float | None = 1.4
    """固定参考时间缩放 (demo 用 1.4, ~1.4s 复位); None=按课程采样。"""
    num_envs: int = 1
    device: str | None = None
    num_episodes: int = 5
    """统计复位时间的 episode 数。"""
    video: bool = True
    video_length: int = 400
    """单个视频最大帧数 (3s @ step_dt=0.008 ≈ 375)。"""


def run_play(cfg: BackupPlayConfig) -> None:
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(TASK_NAME, play=True)
    agent_cfg = load_rl_cfg(TASK_NAME)

    DUMMY = cfg.agent in {"zero", "random"}
    if not DUMMY and cfg.checkpoint_file is None:
        raise ValueError("trained 模式需要 --checkpoint-file")
    resume_path = Path(cfg.checkpoint_file) if cfg.checkpoint_file else None
    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {resume_path}")

    # 固定参考时间缩放 (回放/demo): 与 Slalom fixed_curvature 风格一致
    if cfg.time_scale is not None:
        env_cfg.commands["backup_cmd"].fixed_time_scale = cfg.time_scale  # type: ignore[attr-defined]
        print(f"[INFO] 固定 time_scale λ = {cfg.time_scale}")

    env_cfg.scene.num_envs = cfg.num_envs
    render_mode = "rgb_array" if (not DUMMY and cfg.video) else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    log_dir = resume_path.parent if resume_path is not None else Path("logs/rsl_rl/SQuRo_Backup")
    if not DUMMY and cfg.video:
        video_folder = log_dir / "videos" / "play"
        env = VideoRecorder(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            name_prefix=f"backup_play_ts{cfg.time_scale}",
            disable_logger=False,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if DUMMY:
        action_shape = env.unwrapped.action_space.shape
        if cfg.agent == "zero":
            policy = lambda obs: torch.zeros(action_shape, device=env.unwrapped.device)  # noqa: E731
        else:
            policy = lambda obs: 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1  # noqa: E731
    else:
        assert resume_path is not None
        runner_cls = load_runner_cls(TASK_NAME) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), str(log_dir), device=device)
        runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)
        print(f"[INFO] 已加载 checkpoint: {resume_path.name}")

    # 运行多 episode 统计复位时间
    reset_times: list[float] = []
    asset = env.unwrapped.scene.entities["robot"]
    step_dt = env.unwrapped.step_dt
    from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES

    for ep in range(cfg.num_episodes):
        env.unwrapped.reset()
        obs = env.get_observations()   # RslRlVecEnvWrapper 观测 (tensor)
        stand_steps = 0
        reset_time = None
        for i in range(cfg.video_length):
            with torch.no_grad():
                action = policy(obs)
            obs, rew, dones, infos = env.step(action)
            up = asset.data.projected_gravity_b[:, 2]
            h = 0.5 * (asset.data.body_link_pos_w[:, _MODEL_INDICES.f_body_id, 2]
                       + asset.data.body_link_pos_w[:, _MODEL_INDICES.h_body_id, 2])
            standing = (up > 0.9) & (h > 0.05)
            if bool(standing.all()):
                stand_steps += 1
            else:
                stand_steps = 0
            if stand_steps >= int(0.5 / step_dt) and reset_time is None:
                reset_time = float(i - stand_steps + 1) * step_dt
                break
        if reset_time is not None:
            reset_times.append(reset_time)
            print(f"[episode {ep}] 复位时间 = {reset_time:.3f}s")
        else:
            print(f"[episode {ep}] 未在 {cfg.video_length * step_dt:.2f}s 内稳定站起")

    if reset_times:
        print(f"\n[RESULT] {len(reset_times)}/{cfg.num_episodes} 成功, "
              f"平均复位时间 = {sum(reset_times)/len(reset_times):.3f}s")
    else:
        print("\n[RESULT] 无成功复位")
    env.close()


def main() -> None:
    cfg = tyro.cli(BackupPlayConfig)
    run_play(cfg)


if __name__ == "__main__":
    main()
