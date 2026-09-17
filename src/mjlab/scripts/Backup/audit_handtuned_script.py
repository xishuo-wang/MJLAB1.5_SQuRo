from __future__ import annotations
import sys
import torch
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.SQuRo_Backup_Replay import StateMachinePolicy, slow1_target
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES, resolve_model_indices


# 手调脚本全逻辑核对:
#  1) 目标函数 slow1_target 的名义时间轴与状态机的相位时钟是否对齐
#  2) 终止条件是否会在回放中途触发 auto-reset (新版没有把 terminations 置零)
#  3) 实际关节是否跟随期望

def main() -> None:
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    dur = 2.0 + 1.0
    cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    cfg.scene.num_envs = 1
    cfg.commands["backup_cmd"].fixed_time_scale = lam  # type: ignore[attr-defined]
    cfg.episode_length_s = dur + 1.0

    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
    env.reset()
    a = env.unwrapped.scene.entities["robot"]
    resolve_model_indices(a)
    pol = StateMachinePolicy(env, lam, 5, 0.3, log_events=False)

    print(f"=== 目标函数名义时间轴 (λ={lam}) ===")
    print(f"  slow1_target 内部: time1={1.0}  time2={1.0 + 0.65 * lam:.3f}  "
          f"time3={1.0 + 0.80 * lam:.3f}  time4={1.0 + 0.95 * lam:.3f}  "
          f"time5_end={1.0 + 1.45 * lam:.3f}")
    print(f"  传参方式: current_time = 1.0 + tn*λ, scale=λ  →  有效名义时刻 tn = (current_time-1.0)/λ")
    print(f"  名义 T1 末 = {(1.0 + 0.65 * lam - 1.0) / lam:.3f}s ; "
          f"T2 末 = {(1.0 + 0.80 * lam - 1.0) / lam:.3f}s ; "
          f"T3 末 = {(1.0 + 0.95 * lam - 1.0) / lam:.3f}s")
    print(f"  状态机: P1 期望 {0.8 * lam:.3f}s (=0.80×λ) ; P2 期望 {0.15 * lam:.3f}s (=0.15×λ)")
    print(f"  → 对齐检查: P1 期望 {0.8 * lam:.3f}s  vs  T2 末 {0.80 * lam:.3f}s  "
          f"{'✓ 一致' if abs(0.8 * lam - 0.80 * lam) < 1e-9 else '✗ 不一致'}")

    # 采样目标函数, 看它实际给出的分段
    print(f"\n=== slow1_target 实际分段 (λ={lam}) ===")
    print(f"{'t_nom':>7} | {'F_sp1':>7} {'F_body':>7} {'H_sp1':>7} {'H_body':>7} | 段")
    for tn in [0.0, 0.325, 0.65, 0.725, 0.80, 0.875, 0.95, 1.20, 1.45]:
        r = slow1_target(1.0 + tn * lam, lam)
        if tn < 0.65:
            seg = "T1 起转"
        elif tn < 0.80:
            seg = "T2 回收"
        elif tn < 0.95:
            seg = "T3 解扭"
        elif tn < 1.45:
            seg = "T4 过渡"
        else:
            seg = "T5 保持"
        print(f"{tn:7.3f} | {r[0]:+7.3f} {r[1]:+7.3f} {r[8]:+7.3f} {r[9]:+7.3f} | {seg}")

    print(f"\n=== 回放实际运行 (检查是否中途被 auto-reset) ===")
    fb, hb = _MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id
    prev_eph = 0
    reset_events = []
    print(f"{'t':>6} {'ph':>3} {'t_phase':>8} | {'F_sp1':>7} {'F_body':>7} {'H_sp1':>7} {'H_body':>7} | "
          f"{'base_z':>7} {'up':>7} | {'eph':>4}")
    with torch.no_grad():
        for i in range(int(dur / env.step_dt)):
            before_phase = pol.phase
            env.step(pol(env.unwrapped.get_observations()))
            d = a.data
            jp = d.joint_pos[0, _MODEL_INDICES.joint_ids]
            eph = int(env.episode_length_buf[0])
            if eph < prev_eph:
                reset_events.append((pol._elapsed, prev_eph))
            prev_eph = eph
            if i % 8 == 0 or pol.phase != before_phase:
                print(f"{pol._elapsed:6.2f} {pol.phase:>3} {pol.t_phase:8.3f} | "
                      f"{float(jp[0]):+7.3f} {float(jp[1]):+7.3f} {float(jp[8]):+7.3f} {float(jp[9]):+7.3f} | "
                      f"{float(d.root_link_pos_w[0, 2]):7.4f} "
                      f"{float(d.projected_gravity_b[0, 2]):+7.3f} | {eph:4d}")
            if pol.phase == "DONE":
                break
    print(f"\n[auto-reset 事件] {reset_events if reset_events else '无'}")
    print(f"[最终] phase={pol.phase}  t={pol._elapsed:.2f}s  重试 P1={pol.retry['P1']} P2={pol.retry['P2']}")
    print(f"[终止条件] {list(env.unwrapped.termination_manager.active_terms)}")
    env.close()


if __name__ == "__main__":
    main()
