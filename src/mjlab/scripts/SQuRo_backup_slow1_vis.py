"""SQuRo Backup slow1 手调三段动作的可视化调试脚本 (MJLAB / mujoco-warp)。

复刻 D:\\Code\\SQuRo-MuJoCo\\Loco\\Loco_Backup_slow1.py 的三段动作,
在 MJLAB 环境中开环执行, 对比两种控制方式:
- ``--control position``: 位置控制直接跟踪 slow1 目标轨迹 (MJLAB 标准位置控制)
- ``--control pd``     : 复刻用户 PD 控制 (每步 ctrl = clamp(KP*(target-pos)+KD*(-vel), ±0.2))

用法:
    # 本机 GUI 交互查看
    uv run python src/mjlab/scripts/SQuRo_backup_slow1_vis.py --scale 2 --control pd --visualize viewer
    # 录制视频 (无 GUI 也可)
    uv run python src/mjlab/scripts/SQuRo_backup_slow1_vis.py --scale 2 --control pd --visualize video
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tyro
import torch

import mjlab.tasks  # noqa: F401  触发任务注册
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Backup.mdp.indices import (
    _MODEL_INDICES,
    resolve_model_indices,
)
from mjlab.tasks.SQuRo_Backup.mdp.reference import (
    _FL_HOLD,
    _HL_HOLD,
)
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer


# ===== slow1 三段动作参数 (Loco_Backup_slow1.py) =====
KP, KD, TORQUE_LIMIT = 2.5, 0.01, 0.2   # PD 参数
_LEG_INIT = [0.1, -0.3, 0.1, -0.3, -0.1, 0.3, -0.1, 0.3]


# slow1 目标轨迹 — 返回 MJLAB actuator 顺序 [F_spine1, F_body, Neck_yaw, Neck_pitch,
#   FL_sh, FL_el, FR_sh, FR_el, H_spine1, H_body, HL_hip, HL_knee, HR_hip, HR_knee]
def slow1_target(t: float, scale: float) -> list[float]:
    t1, t2, t3, t4 = 0.0, 0.65 * scale, 0.8 * scale, 0.95 * scale
    d1, d2, d3 = t2 - t1, t3 - t2, t4 - t3
    if t <= t1:
        return _LEG_INIT + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    r = _LEG_INIT + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # 腿支撑位 (MJLAB 列: FL/FR 4-7, HL/HR 10-13)
    r[4], r[5] = _FL_HOLD; r[6], r[7] = _FL_HOLD
    r[10], r[11] = _HL_HOLD; r[12], r[13] = _HL_HOLD
    # 脊柱 (MJLAB 列: F_spine1=0, F_body=1, H_spine1=8, H_body=9)
    if t < t2:
        u = (t - t1) / d1
        r[0] = 0.8 * u; r[1] = -1.57 * u; r[8] = 0.8 * u; r[9] = 1.57 * u
    elif t < t3:
        u = (t - t2) / d2
        r[0] = 0.8 - 0.8 * u; r[1] = -1.57; r[8] = 0.8 - 0.8 * u; r[9] = 1.57
    elif t < t4:
        u = (t - t3) / d3
        r[0] = 0.8 * u; r[1] = -1.57 + 1.57 * u; r[8] = 0.0; r[9] = 1.57 - 1.57 * u
    # t >= t4: 回站立角 (默认)
    return r


# 开环策略: 忽略观测, 按 episode 时间执行 slow1
class Slow1OpenLoopPolicy:
    def __init__(self, env: ManagerBasedRlEnv, scale: float, control: str,
                 action_scale: float) -> None:
        self.unwrapped = env.unwrapped
        self.asset = self.unwrapped.scene.entities["robot"]
        resolve_model_indices(self.asset)
        self.default = self.asset.data.default_joint_pos[:, _MODEL_INDICES.joint_ids]
        self.scale = scale
        self.control = control
        self.action_scale = action_scale

    def __call__(self, obs: Any) -> torch.Tensor:
        del obs
        t = (self.unwrapped.episode_length_buf.float() * self.unwrapped.step_dt)[0].item()
        target = slow1_target(t, self.scale)
        target_t = torch.tensor(target, device=self.default.device, dtype=torch.float32)
        if self.control == "position":
            pos_cmd = target_t
        else:
            cur = self.asset.data.joint_pos[0, _MODEL_INDICES.joint_ids]
            vel = self.asset.data.joint_vel[0, _MODEL_INDICES.joint_ids]
            err = target_t - cur
            pos_cmd = (KP * err + KD * (-vel)).clamp(-TORQUE_LIMIT, TORQUE_LIMIT)
        action = (pos_cmd - self.default[0]) / self.action_scale
        return action.unsqueeze(0)


@dataclass(frozen=True)
class VisConfig:
    scale: float = 2.0
    """slow1 时间缩放 (1-10, 最小 1 = 0.95s 动作)。"""
    control: Literal["position", "pd"] = "pd"
    """position=位置控制跟踪目标; pd=复刻用户 PD 平滑控制。"""
    visualize: Literal["none", "viewer", "video"] = "viewer"
    """none=无头打印; viewer=交互查看(需 GUI); video=录制 mp4。"""
    num_envs: int = 1
    device: str | None = None
    duration: float = 4.0
    """回放时长 (s)。"""
    video_dir: str = "logs/rsl_rl/SQuRo_Backup/replay_videos"


def main() -> None:
    args = tyro.cli(VisConfig)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.commands["backup_cmd"].fixed_time_scale = args.scale  # type: ignore[attr-defined]

    render_mode = "rgb_array" if args.visualize == "video" else None
    print(f"[INFO] slow1 可视化: scale={args.scale}, control={args.control}, "
          f"visualize={args.visualize}, device={device}")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    env.reset()

    if args.visualize == "video":
        video_folder = Path(args.video_dir)
        video_folder.mkdir(parents=True, exist_ok=True)
        n_frames = int(args.duration / env.step_dt) + 1
        env = VideoRecorder(
            env, video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=n_frames,
            name_prefix=f"slow1_sc{args.scale}_{args.control}",
            disable_logger=False,
        )

    # 无头/视频: 手动循环打印
    if args.visualize in ("none", "video"):
        env.reset()
        asset = env.unwrapped.scene.entities["robot"]
        policy = Slow1OpenLoopPolicy(env, args.scale, args.control, 0.3)
        n = int(args.duration / env.step_dt)
        for i in range(n):
            with torch.no_grad():
                action = policy(env.unwrapped.get_observations())
            obs, rew, dones, to, extras = env.step(action)
            if i % 50 == 0:
                up = env.unwrapped.scene.entities["robot"].data.projected_gravity_b[:, 2]
                print(f"t={i*env.step_dt:4.2f} base_z={asset.data.root_link_pos_w[0,2]:.3f} "
                      f"up={up[0].item():+.2f}", flush=True)
        env.close()
        return

    # viewer: 交互查看
    env.reset()
    policy = Slow1OpenLoopPolicy(env, args.scale, args.control, 0.3)
    viewer = NativeMujocoViewer(env, policy, frame_rate=60)  # type: ignore[arg-type]
    viewer.run(num_steps=int(args.duration / env.step_dt))
    env.close()


if __name__ == "__main__":
    main()
