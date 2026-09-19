# 重置初态实测 — 打印 reset 后各关节的实际角与速度, 并给出与参考 t=0 的差
# 用途: 核实"重置腿姿态"这一前提 (技术细节 §7 起步瞬态一节)
# 用法: uv run python -B -m mjlab.scripts.Backup.probe_reset_pose
from __future__ import annotations

from dataclasses import dataclass

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.SQuRo_Backup.mdp.indices import _ACTUATED_JOINT_NAMES, _MODEL_INDICES, resolve_model_indices
from mjlab.tasks.SQuRo_Backup.mdp.reference import _generate_reference_table
from mjlab.tasks.registry import load_env_cfg


@dataclass
class ProbeCfg:
    device: str = "cuda:0"
    # 速度阈值: 超过则视为非稳态初态
    vel_warn: float = 0.5
    # 静止测试步数 (零动作), 检查初态是否为 PD 平衡点
    settle_steps: int = 50


def main() -> None:
    cfg = tyro.cli(ProbeCfg)
    env_cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    env_cfg.scene.num_envs = 1
    env_cfg.commands["backup_cmd"].fixed_time_scale = 1.0
    env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
    env.reset()

    asset = env.scene.entities["robot"]
    resolve_model_indices(asset)
    names = _ACTUATED_JOINT_NAMES
    jids = {n: asset.joint_names.index(n) for n in names}
    pos = {n: float(asset.data.joint_pos[0, jids[n]]) for n in names}
    vel = {n: float(asset.data.joint_vel[0, jids[n]]) for n in names}
    _, ref = _generate_reference_table()

    print("重置初态实测 (env.reset 之后、未 step):")
    print("%-22s %12s %12s %12s %10s" % ("关节", "初态角", "初态速度", "参考 t=0", "差"))
    print("-" * 74)
    bad = []
    for k, n in enumerate(names):
        d = pos[n] - float(ref[0, k])
        flag = "  <-- 非零速度" if abs(vel[n]) > cfg.vel_warn else ""
        print("%-22s %+12.3f %+12.3f %+12.3f %+10.3f%s" % (n, pos[n], vel[n], float(ref[0, k]), d, flag))
        if abs(vel[n]) > cfg.vel_warn:
            bad.append(n)
    print("-" * 74)
    print(f"|速度| > {cfg.vel_warn} 的关节数: {len(bad)}  {bad}")
    print()
    # 基座
    bz = float(asset.data.body_link_pos_w[0, 0, 2]) if hasattr(asset.data, "body_link_pos_w") else float("nan")
    print(f"基座高度: {bz:.4f} m")

    # 零动作静止测试: 初态若远离 PD 平衡点, 松手后会漂移
    if cfg.settle_steps > 0:
        print()
        print(f"零动作静止测试 ({cfg.settle_steps} 步 = {cfg.settle_steps * env.step_dt:.2f} s):")
        print("%6s %10s | %s" % ("step", "基座高度", " | ".join("%-18s" % n.replace("_joint", "") for n in names)))
        print("%6s %10s | %s" % ("", "", " | ".join("%8s%9s" % ("角", "速度") for _ in names)))
        zero = torch.zeros(1, 14, device=env.device)
        for i in range(cfg.settle_steps + 1):
            if i % 10 == 0:
                b = float(asset.data.body_link_pos_w[0, 0, 2])
                cells = " | ".join("%+8.3f %+9.3f" % (
                    float(asset.data.joint_pos[0, jids[n]]), float(asset.data.joint_vel[0, jids[n]])) for n in names)
                print("%6d %10.4f | %s" % (i, b, cells))
            env.step(zero)


if __name__ == "__main__":
    main()
