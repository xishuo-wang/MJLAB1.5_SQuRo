"""SQuRo Backup 参考轨迹重放验证脚本（MJLAB / mujoco-warp 后端）。

开环将参考轨迹（翻身 -> 保持支撑 -> 过渡站立）通过位置控制重放到环境中，
验证机器人能否从仰面跌倒姿态稳定爬起，支持三种可视化模式：

- ``--visualize none``  : 无头模式, 打印 base_z/高度/uprightness 时间线 + 爬起判定
- ``--visualize video`` : 录制 mp4 视频到 --video-dir (离屏渲染, 无需 GUI)
- ``--visualize viewer``: 打开 MuJoCo 交互式查看器实时观察 (需本机 GUI)

用法:
    uv run python src/mjlab/scripts/SQuRo_backup_ref_replay.py --num-envs 4
    uv run python src/mjlab/scripts/SQuRo_backup_ref_replay.py --visualize video
    uv run python src/mjlab/scripts/SQuRo_backup_ref_replay.py --visualize viewer --num-envs 1

输出: 每 0.4s 打印 base_z / F/H body 高度 / uprightness, 末尾给出爬起判定。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mjlab.tasks  # noqa: F401  触发任务注册
import tyro
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Backup.mdp.indices import (
    _MODEL_INDICES,
    resolve_model_indices,
)
from mjlab.tasks.SQuRo_Backup.mdp.reference import (
    REF_TOTAL_TIME,
    get_reference_joint_state,
)
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer


@dataclass(frozen=True)
class ReplayConfig:
    num_envs: int = 4
    """并行环境数。"""
    device: str | None = None
    """计算设备, 默认 cuda:0 (无 GPU 时 cpu)。"""
    duration: float = REF_TOTAL_TIME
    """重放时长 (s), 默认覆盖整段参考轨迹。"""
    action_scale: float | None = None
    """位置动作缩放; 默认自动从 env_cfg.actions["joint_pos"].scale 读取,
    必须与 env_cfg 一致, 否则开环重放的参考动作会被错误缩放。"""
    print_interval: float = 0.4
    """打印时间间隔 (s)。"""
    visualize: Literal["none", "video", "viewer"] = "viewer"
    """可视化模式: none=无头打印, video=录制 mp4, viewer=交互式查看器。"""
    video_dir: str = "logs/rsl_rl/SQuRo_Backup/replay_videos"
    """视频输出目录 (--visualize video 时生效)。"""


# 开环参考"策略": 忽略观测, 每步按 episode 时间输出参考动作
class ReferencePolicy:
    def __init__(self, env: ManagerBasedRlEnv, action_scale: float) -> None:
        self.unwrapped = env.unwrapped
        self.asset = self.unwrapped.scene.entities["robot"]
        # 确保模型索引已解析 (viewer 模式在 env.reset() 之前构造, 索引可能为空)
        resolve_model_indices(self.asset)
        self.default = self.asset.data.default_joint_pos[:, _MODEL_INDICES.joint_ids]
        self.action_scale = action_scale

    def __call__(self, obs: Any) -> torch.Tensor:
        """忽略观测, 按 episode 时间输出参考动作。

        BaseViewer 传入的 obs 是观测字典 (get_observations), 与训练时的
        RslRlVecEnvWrapper 张量不同; 开环重放不需要观测。
        """
        del obs
        ref_pos, _ = get_reference_joint_state(self.unwrapped)
        return (ref_pos - self.default) / self.action_scale


def compute_ref_action(
    env: ManagerBasedRlEnv, default: torch.Tensor, action_scale: float
) -> torch.Tensor:
    """按当前 episode 时间计算参考位置动作。"""
    ref_pos, _ = get_reference_joint_state(env)
    return (ref_pos - default) / action_scale


def main() -> None:
    args = tyro.cli(ReplayConfig)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    env_cfg.scene.num_envs = args.num_envs
    # 动作缩放: 未显式指定时从 env_cfg 读取, 保证与训练/环境施加一致
    # type: ignore[attr-defined]  # actions 实际是 JointPositionActionCfg(有 scale), 基类标注无该属性
    cfg_scale = env_cfg.actions["joint_pos"].scale  # type: ignore[attr-defined]
    action_scale = args.action_scale if args.action_scale is not None else float(cfg_scale)
    print(f"[INFO] 动作缩放 action_scale={action_scale} (env_cfg scale={cfg_scale})")

    # 创建环境: video 模式需离屏渲染 (rgb_array), viewer 模式用原生渲染
    render_mode = "rgb_array" if args.visualize == "video" else None
    print(f"[INFO] 创建环境 num_envs={args.num_envs}, device={device}, "
          f"visualize={args.visualize}")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if args.visualize == "viewer":
        # 交互式查看器: 开环参考策略实时驱动
        if args.num_envs != 1:
            print("[WARN] viewer 模式建议 --num-envs 1 (查看器只显示单环境)")
        # BaseViewer 不会自动 reset 环境, 这里先重置为跌倒初始姿态
        # (否则 viewer 显示的是 env 构造后的站立姿态, 参考动作会把机器人"翻倒")
        env.reset()
        policy = ReferencePolicy(env, action_scale)
        # type: ignore[arg-type]  # 框架 EnvProtocol 与 ManagerBasedRlEnv.step 返回类型标注不完全匹配
        viewer = NativeMujocoViewer(env, policy, frame_rate=60)  # type: ignore[arg-type]
        viewer.run(num_steps=int(args.duration / env.step_dt))
        env.close()
        return

    if args.visualize == "video":
        video_folder = Path(args.video_dir)
        video_folder.mkdir(parents=True, exist_ok=True)
        n_video_frames = int(args.duration / env.step_dt) + 1
        env = VideoRecorder(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=n_video_frames,
            name_prefix="backup_ref_replay",
            disable_logger=False,
        )

    env.reset()
    unwrapped = env.unwrapped
    asset = unwrapped.scene.entities["robot"]
    default = asset.data.default_joint_pos[:, _MODEL_INDICES.joint_ids]

    print(f"[INFO] 开环重放参考轨迹 {args.duration:.1f}s "
          f"(step_dt={env.step_dt:.4f}s, {int(args.duration/env.step_dt)} env steps)")

    n_steps = int(args.duration / env.step_dt)
    print_every = max(1, int(args.print_interval / env.step_dt))
    z_tail: list[float] = []

    for i in range(n_steps):
        action = compute_ref_action(unwrapped, default, action_scale)
        obs, rew, dones, to, extras = env.step(action)
        base_z = asset.data.root_link_pos_w[:, 2]
        if i >= n_steps - int(1.0 / env.step_dt):
            z_tail.append(float(base_z.mean().item()))
        if i % print_every == 0:
            log = extras.get("log", {})
            print(
                f"  t={i*env.step_dt:4.2f}s rew={rew.mean().item():+.4f} "
                f"base_z={base_z.mean().item():.3f} "
                f"upright={log.get('Data/uprightness', float('nan')):+.3f} "
                f"stand_hold={log.get('Progress/stand_hold', float('nan')):.3f}"
            )

    tail_avg = sum(z_tail) / max(1, len(z_tail))
    success = tail_avg > 0.05
    print(f"\n[RESULT] 最后 1s 平均 base_z = {tail_avg:.3f} m -> "
          f"{'[OK] 成功爬起并站立' if success else '[FAIL] 未稳定站起'}")
    env.close()


if __name__ == "__main__":
    main()
