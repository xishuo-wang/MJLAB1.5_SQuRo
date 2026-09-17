from __future__ import annotations
import sys
import torch
import numpy as np
import mujoco
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.SQuRo_Backup_Replay import StateMachinePolicy
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES, resolve_model_indices
import mjlab.asset_zoo.robots.SQuRo as _sq

XML = Path(_sq.__file__).resolve().parent / "xmls" / "SQuRo.xml"


# 三口径对拍 (手调开环回放, 逐步):
#   A. command.py / 状态机用的 _body_up  = sign * 2(yz + wx)   (仅 body 局部四元数)
#   B. body 局部系背腹轴在该公式下的另一分量 2(wy - xz)
#   C. 真值: body 背腹轴在世界系的 Z 分量 (用 base 四元数 + 局部四元数做四元数乘法)
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


# 世界系下 body 的 +Y 轴
def world_up(q_world):
    w, x, y, z = q_world
    return np.array([
        2 * (x * y - w * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z + w * x),
    ])


def main() -> None:
    env_cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    env_cfg.scene.num_envs = 1
    env_cfg.commands["backup_cmd"].fixed_time_scale = 1.0  # type: ignore[attr-defined]
    for tc in env_cfg.terminations.values():
        tc.func = lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env_cfg.episode_length_s = 5.0

    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
    env.reset()
    asset = env.unwrapped.scene.entities["robot"]
    resolve_model_indices(asset)
    policy = StateMachinePolicy(env, 1.0, max_retry=5, buffer=0.3, quiet=True)

    fb, hb = _MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id
    print(f"{'t':>5} {'phase':>4} | {'A_upF':>7} {'A_upH':>7} | {'C_wF':>7} {'C_wH':>7} | "
          f"{'base_z':>7} {'base_up':>7} | {'S1(A)':>5} {'S1(C)':>5}")
    with torch.no_grad():
        for i in range(500):
            env.step(policy(env.unwrapped.get_observations()))
            d = asset.data
            qb = d.root_link_quat_w[0].cpu().numpy().astype(float)
            qF = d.body_link_quat_w[0, fb].cpu().numpy().astype(float)
            qH = d.body_link_quat_w[0, hb].cpu().numpy().astype(float)
            # A: 状态机口径
            A_F = 2.0 * (qF[2] * qF[3] + qF[0] * qF[1])
            A_H = -2.0 * (qH[2] * qH[3] + qH[0] * qH[1])
            # C: 真值 — 先把局部四元数复合到世界
            wF = qmul(qb, qF)
            wH = qmul(qb, qH)
            C_F = world_up(wF)[2]
            C_H = -world_up(wH)[2]
            fz = float(d.body_link_pos_w[0, fb, 2])
            hz = float(d.body_link_pos_w[0, hb, 2])
            if i % 15 == 0:
                s1a = (A_F > 0.5) and (A_H < -0.5) and fz < 0.03 and hz < 0.03
                s1c = (C_F > 0.5) and (C_H < -0.5) and fz < 0.03 and hz < 0.03
                print(f"{policy._elapsed:5.2f} {policy.phase:>4} | {A_F:+7.3f} {A_H:+7.3f} | "
                      f"{C_F:+7.3f} {C_H:+7.3f} | {float(d.root_link_pos_w[0,2]):7.4f} "
                      f"{float(d.projected_gravity_b[0,2]):+7.3f} | "
                      f"{str(s1a):>5} {str(s1c):>5}")
            if policy.phase == "DONE":
                print(f"--- DONE @ t={policy._elapsed:.2f}s  重试 P1={policy.retry['P1']} P2={policy.retry['P2']}")
                break
    env.close()


if __name__ == "__main__":
    main()
