from __future__ import annotations
import sys
import torch
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.SQuRo_Backup_Replay import StateMachinePolicy
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES, resolve_model_indices


# 一次性审计: 同一帧打印 关节角 / 相对四元数 raw / 世界系 FK 预测 / 世界系四元数复合
# 目标: 确定 body_link_quat_w 的口径, 并给出与"每段身体 projected_gravity"对应的量

def q2m(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def main() -> None:
    lam = 1.0
    cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    cfg.scene.num_envs = 1
    cfg.commands["backup_cmd"].fixed_time_scale = lam  # type: ignore[attr-defined]
    for tc in cfg.terminations.values():
        tc.func = lambda e: torch.zeros(e.num_envs, dtype=torch.bool, device=e.device)
    cfg.episode_length_s = 3.0

    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
    env.reset()
    a = env.unwrapped.scene.entities["robot"]
    resolve_model_indices(a)
    pol = StateMachinePolicy(env, lam, 5, 0.3, log_events=False)
    fb, hb = _MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id

    # 关节角 -> 世界系背腹轴 Z 分量的解析预测 (θ_s + θ_b 的正弦)
    print("列说明:")
    print("  thF = F_spine1 + F_body ; thH = H_spine1 + H_body")
    print("  sin   = sin(θ_s+θ_b)                    ← 世界系解析预测")
    print("  rawF  = 2(yz+wx) of body_link_quat_w[F]  ← mjlab 原始量")
    print("  wldF  = (q_root ⊗ q_rel) 的 2(yz+wx)      ← 复合 root 后的量")
    print("  pgF   = 1-2(x²+y²) of body_link_quat_w[F] ← 若该四元数是世界系, 这就是 gravity[2]")
    print()
    print(f"{'t':>5} {'t_nom':>6} {'ph':>3} | {'F_sp1':>7} {'F_body':>7} {'H_sp1':>7} {'H_body':>7} | "
          f"{'sinF':>7} {'sinH':>7} | {'rawF':>7} {'rawH':>7} | {'wldF':>7} {'wldH':>7} | "
          f"{'pgF':>7} {'pgH':>7}")
    with torch.no_grad():
        for i in range(300):
            before = pol.phase
            env.step(pol(env.unwrapped.get_observations()))
            d = a.data
            jp = d.joint_pos[0, _MODEL_INDICES.joint_ids].detach().cpu().numpy()
            f_sp1, f_bd, h_sp1, h_bd = float(jp[0]), float(jp[1]), float(jp[8]), float(jp[9])
            sF, sH = np.sin(f_sp1 + f_bd), np.sin(h_sp1 + h_bd)
            qr = d.root_link_quat_w[0].detach().cpu().numpy().astype(float)
            qF = d.body_link_quat_w[0, fb].detach().cpu().numpy().astype(float)
            qH = d.body_link_quat_w[0, hb].detach().cpu().numpy().astype(float)
            rawF = 2 * (qF[2] * qF[3] + qF[0] * qF[1])
            rawH = 2 * (qH[2] * qH[3] + qH[0] * qH[1])
            tF, tH = qmul(qr, qF), qmul(qr, qH)
            wldF = 2 * (tF[2] * tF[3] + tF[0] * tF[1])
            wldH = 2 * (tH[2] * tH[3] + tH[0] * tH[1])
            pgF = 1 - 2 * (qF[1] ** 2 + qF[2] ** 2)
            pgH = 1 - 2 * (qH[1] ** 2 + qH[2] ** 2)
            t = pol._elapsed
            anchor = pol.phase != before
            if anchor or i % 8 == 0:
                mark = "  <== 相位切换" if anchor else ""
                print(f"{t:5.2f} {t/lam:6.3f} {pol.phase:>3} | {f_sp1:+7.3f} {f_bd:+7.3f} "
                      f"{h_sp1:+7.3f} {h_bd:+7.3f} | {sF:+7.3f} {sH:+7.3f} | "
                      f"{rawF:+7.3f} {rawH:+7.3f} | {wldF:+7.3f} {wldH:+7.3f} | "
                      f"{pgF:+7.3f} {pgH:+7.3f}{mark}")
            if pol.phase == "DONE":
                break
    env.close()


if __name__ == "__main__":
    main()
