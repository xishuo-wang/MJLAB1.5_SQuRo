from __future__ import annotations
import sys
import torch
import numpy as np
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.SQuRo_Backup_Replay import StateMachinePolicy, slow1_target
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES, resolve_model_indices


# 逐帧观察 S1/S2 达成瞬间的关节角与几何条件, 判断是"端点达成"还是"瞬态掠过"

UP_TH = 0.5
G_TH = 0.03
G_TH2 = 0.04


def main() -> None:
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    dur = 2.5 * lam + 1.0
    cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    cfg.scene.num_envs = 1
    cfg.commands["backup_cmd"].fixed_time_scale = lam  # type: ignore[attr-defined]
    for tc in cfg.terminations.values():
        tc.func = lambda e: torch.zeros(e.num_envs, dtype=torch.bool, device=e.device)
    cfg.episode_length_s = dur + 1.0

    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
    env.reset()
    a = env.unwrapped.scene.entities["robot"]
    resolve_model_indices(a)
    pol = StateMachinePolicy(env, lam, 5, 0.3, log_events=True)

    fb, hb = _MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id
    print(f"λ={lam}  名义 P1_END=0.80s → 实际 {0.80 * lam:.2f}s ; P2_END=0.95s → 实际 {0.95 * lam:.2f}s")
    print(f"\n{'t':>6} {'t_nom':>6} {'ph':>3} | {'ref F':>7} {'ref B':>7} | "
          f"{'F_sp1':>7} {'F_body':>7} {'H_sp1':>7} {'H_body':>7} | "
          f"{'A_upF':>7} {'A_upH':>7} | {'zF':>7} {'zH':>7} | S1 S2")
    with torch.no_grad():
        for i in range(int(dur / env.step_dt)):
            env.step(pol(env.unwrapped.get_observations()))
            d = a.data
            qF = d.body_link_quat_w[0, fb]
            qH = d.body_link_quat_w[0, hb]
            A_F = 2.0 * (float(qF[2]) * float(qF[3]) + float(qF[0]) * float(qF[1]))
            A_H = -2.0 * (float(qH[2]) * float(qH[3]) + float(qH[0]) * float(qH[1]))
            jp = d.joint_pos[0, _MODEL_INDICES.joint_ids]
            fz = float(d.body_link_pos_w[0, fb, 2])
            hz = float(d.body_link_pos_w[0, hb, 2])
            s1 = A_F > UP_TH and A_H < -UP_TH and fz < G_TH and hz < G_TH
            s2 = A_F < -UP_TH and A_H < -UP_TH and fz < G_TH2 and hz < G_TH2
            t = pol._elapsed
            tn = t / lam
            # 只打印 0.55~1.15 名义秒窗口 + 每次相位切换处
            if 0.55 <= tn <= 1.10 and i % 4 == 0:
                ref = slow1_target(1.0 + tn * lam, lam)
                print(f"{t:6.2f} {tn:6.3f} {pol.phase:>3} | {ref[0]:+7.3f} {ref[1]:+7.3f} | "
                      f"{float(jp[0]):+7.3f} {float(jp[1]):+7.3f} {float(jp[8]):+7.3f} {float(jp[9]):+7.3f} | "
                      f"{A_F:+7.3f} {A_H:+7.3f} | {fz:7.4f} {hz:7.4f} | {int(s1)}  {int(s2)}")
            if pol.phase == "DONE":
                print(f"--- DONE @ t={t:.2f}s (t_nom={tn:.3f}s)  重试 P1={pol.retry['P1']} P2={pol.retry['P2']}")
                break
    env.close()


if __name__ == "__main__":
    main()
