"""SQuRo Backup 参考轨迹重放验证脚本（MJLAB / mujoco-warp 后端）。

开环将参考轨迹（翻身 -> 保持支撑 -> 过渡站立）通过位置控制重放到环境中，
验证机器人能否从仰面跌倒姿态稳定爬起。

用法:
    uv run python src/mjlab/scripts/SQuRo_backup_ref_replay.py --num-envs 4
    uv run python -m mjlab.scripts.SQuRo_backup_ref_replay --num-envs 8 --duration 8.0

输出: 每 0.4s 打印 base_z / F/H body 高度 / uprightness, 末尾给出爬起判定。
"""

from __future__ import annotations

import tyro
import torch
from dataclasses import dataclass

import mjlab.tasks  # noqa: F401  触发任务注册
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES
from mjlab.tasks.SQuRo_Backup.mdp.reference import (
    REF_TOTAL_TIME,
    get_reference_joint_state,
)


@dataclass(frozen=True)
class ReplayConfig:
    num_envs: int = 4
    """并行环境数。"""
    device: str | None = None
    """计算设备, 默认 cuda:0 (无 GPU 时 cpu)。"""
    duration: float = REF_TOTAL_TIME
    """重放时长 (s), 默认覆盖整段参考轨迹。"""
    action_scale: float = 0.5
    """位置动作缩放 (需与 env_cfg 中 JointPositionActionCfg.scale 一致)。"""
    print_interval: float = 0.4
    """打印时间间隔 (s)。"""


def main() -> None:
    args = tyro.cli(ReplayConfig)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    env_cfg.scene.num_envs = args.num_envs

    print(f"[INFO] 创建环境 num_envs={args.num_envs}, device={device}")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()
    asset = env.scene.entities["robot"]
    default = asset.data.default_joint_pos[:, _MODEL_INDICES.joint_ids]

    print(f"[INFO] 开环重放参考轨迹 {args.duration:.1f}s "
          f"(step_dt={env.step_dt:.4f}s, {int(args.duration/env.step_dt)} env steps)")

    n_steps = int(args.duration / env.step_dt)
    print_every = max(1, int(args.print_interval / env.step_dt))
    z_tail: list[float] = []

    for i in range(n_steps):
        ref_pos, _ = get_reference_joint_state(env)  # [N,14]
        action = (ref_pos - default) / args.action_scale
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
                f"h={log.get('Data/height_actual', float('nan')):.3f}"
            )

    tail_avg = sum(z_tail) / max(1, len(z_tail))
    success = tail_avg > 0.05
    print(f"\n[RESULT] 最后 1s 平均 base_z = {tail_avg:.3f} m -> "
          f"{'[OK] 成功爬起并站立' if success else '[FAIL] 未稳定站起'}")
    env.close()


if __name__ == "__main__":
    main()
