from __future__ import annotations
import sys
import torch
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.SQuRo_Backup_Replay import (
    StateMachinePolicy,
    slow1_target,
    T_SEG1_END,
    T_SEG2_END,
    T_SEG3_END,
    T_SEG4_END,
)
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES, resolve_model_indices
from mjlab.tasks.SQuRo_Backup.mdp.reference import get_reference_joint_state
from mjlab.tasks.SQuRo_Backup.mdp import config as T
from mjlab.tasks.SQuRo_Backup.mdp.config import STAND_TRANSITION_END as _STAND_TRANSITION_END
from mjlab.tasks.SQuRo_Backup.mdp.reference import P1_ONSET as _P1_ONSET


# 对拍两件事:
#   ① 参考轨迹: 手调脚本 slow1_target  vs  训练参考表 reference.py
#   ② 检测判据: 手调脚本 _is_S1/_is_S2/_is_both_inverted(实为直接调 command._check_*)
#               vs  训练环境 BackupCommand._check_S1/_check_S2/_check_both_inverted
#
# 检测对拍方式: 两者都指向同一个 command 实例的判据 (脚本按设计复用训练判据),
#   因此在同一物理状态上必须逐帧一致 —— 这里直接断言该点, 并同时记录
#   手调脚本自身的相位推进与训练 command 的相位推进是否同步。

SPINE_IDX = [0, 1, 8, 9]
LEG_IDX = [4, 5, 6, 7, 10, 11, 12, 13]
ALL_IDX = SPINE_IDX + LEG_IDX
# 参考表存的是 float32, 且比对要走一次线性插值, 故容差不能取 1e-6 (约为 f32 在 1.57 处的
# 一个 ulp); 1e-5 rad 相对量程仍是 1e-5 量级, 足以发现真实公式分歧。
REF_TOL = 1e-5


def cmp_reference() -> bool:
    print("=" * 78)
    print("① 参考轨迹对拍: slow1_target (手调脚本)  vs  _get_ref_table (训练参考表)")
    print("=" * 78)
    print(f"  手调分段边界: T1={T_SEG1_END} T2={T_SEG2_END} T3={T_SEG3_END} T4={T_SEG4_END} (名义秒)")
    print(f"  参考表边界  : P1_END={T.P1_END} P2_END={T.P2_END} "
          f"TRANS_END={_STAND_TRANSITION_END} TOTAL={T.REFERENCE_TOTAL_TIME}")
    print(f"  分段时长    : 手调 T1={T_SEG1_END:.3f} T2={T_SEG2_END-T_SEG1_END:.3f} "
          f"T3={T_SEG3_END-T_SEG2_END:.3f} T4={T_SEG4_END-T_SEG3_END:.3f}")

    env_cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    env_cfg.scene.num_envs = 1
    env_cfg.commands["backup_cmd"].fixed_time_scale = 1.0  # type: ignore[attr-defined]
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    resolve_model_indices(env.unwrapped.scene.entities["robot"])

    # 训练参考表: 用同一套查询接口 (λ=1)
    # 手调轴以 T1 起点为 T_OFFSET, 参考表以 P1_ONSET 为 T1 起点, 故 手调时刻 = 表时刻 + P1_ONSET。
    # 全部 14 个关节都参与比对: 只比脊柱无法证明 P0 收腿轨迹一致。
    ts = np.arange(0.0, T.STAND_TRANSITION_END + 1e-9, 0.005)
    errs = []
    rows = []
    for tn in ts:
        # 训练参考: get_reference_joint_state 按 phase/t_phase 查询, 这里直接造一个最小代理
        ref_gym = _ref_table_at(env, tn)
        ref_hand = np.array(slow1_target(tn + _P1_ONSET, 1.0))[ALL_IDX]
        e = np.abs(ref_gym - ref_hand).max()
        errs.append(e)
        rows.append((tn, ref_gym, ref_hand, e))
    errs = np.array(errs)
    print(f"\n  采样 {len(ts)} 点, 名义时间 0..{ts[-1]:.3f}s")
    print(f"  14 关节(含 P0 收腿段) |参考表 - 手调| 最大偏差 = {errs.max():.6f} rad")
    print(f"  偏差 > {REF_TOL:g} 的点数 = {int((errs > REF_TOL).sum())}")
    print()
    print(f"  {'t_nom':>6} | {'参考表 F_sp1 F_body H_sp1 H_body':>34} | {'手调 F_sp1 F_body H_sp1 H_body':>34} | {'max|Δ|':>8}")
    for tn, rg, rh, e in rows[::20]:
        print(f"  {tn:6.2f} | {rg[0]:+8.3f} {rg[1]:+8.3f} {rg[2]:+8.3f} {rg[3]:+8.3f} | "
              f"{rh[0]:+8.3f} {rh[1]:+8.3f} {rh[2]:+8.3f} {rh[3]:+8.3f} | {e:8.6f}")

    env.close()
    ok = bool(errs.max() < REF_TOL)
    print(f"\n  [参考对拍] {'一致' if ok else '** 不一致 **'}")
    return ok


# 在给定名义时间 tn 处查询训练参考表 (λ=1, 按 P1/P2/P3 分段)
def _ref_table_at(env, tn: float) -> np.ndarray:
    cmd = env.unwrapped.command_manager.get_term("backup_cmd")
    if tn < T.P1_END:
        phase, t_local = 0, tn
    elif tn < T.P2_END:
        phase, t_local = 1, tn - T.P1_END
    else:
        phase, t_local = 2, tn - T.P2_END
    cmd.phase[:] = phase
    cmd.t_phase[:] = t_local
    cmd.command_tensor[:, 5] = 1.0
    pos, _ = get_reference_joint_state(env.unwrapped)
    return pos[0, ALL_IDX].detach().cpu().numpy()


def cmp_detection(lam: float) -> bool:
    print()
    print("=" * 78)
    print(f"② 检测判据对拍 (λ={lam}): 手调脚本 vs 训练 command, 同一物理状态逐帧比对")
    print("=" * 78)

    cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    cfg.scene.num_envs = 1
    cfg.commands["backup_cmd"].fixed_time_scale = lam  # type: ignore[attr-defined]
    for tc in cfg.terminations.values():
        tc.func = lambda e: torch.zeros(e.num_envs, dtype=torch.bool, device=e.device)
    cfg.episode_length_s = 5.0 * lam + 2.0

    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
    env.reset()
    resolve_model_indices(env.unwrapped.scene.entities["robot"])
    cmd = env.unwrapped.command_manager.get_term("backup_cmd")
    pol = StateMachinePolicy(env, lam, 0, log_events=False)

    n_mismatch = {"S1": 0, "S2": 0, "INV": 0}
    rows = []
    with torch.no_grad():
        for _ in range(int((5.0 * lam + 1.0) / env.step_dt)):
            env.step(pol(env.unwrapped.get_observations()))
            # 手调脚本侧
            h_s1 = bool(pol._is_S1())
            h_s2 = bool(pol._is_S2())
            h_inv = bool(pol._is_both_inverted())
            # 训练 command 侧
            c_s1 = bool(cmd._check_S1()[0])
            c_s2 = bool(cmd._check_S2()[0])
            c_inv = bool(cmd._check_both_inverted()[0])
            for k, a, b in (("S1", h_s1, c_s1), ("S2", h_s2, c_s2), ("INV", h_inv, c_inv)):
                if a != b:
                    n_mismatch[k] += 1
            rows.append({
                "t": pol._elapsed, "ph": pol.phase,
                "h": (h_s1, h_s2, h_inv), "c": (c_s1, c_s2, c_inv),
                "cph": int(cmd.phase[0]), "cS1": pol._s1_confirm_t, "cINV": pol._inverted_confirm_t,
            })
            if pol.phase == "DONE":
                break

    print(f"  帧数 {len(rows)}  最终 phase={pol.phase}")
    print(f"  逐帧不一致次数: S1={n_mismatch['S1']}  S2={n_mismatch['S2']}  INV={n_mismatch['INV']}")

    def first(key: str, which: int) -> str:
        for r in rows:
            if r[key][which]:
                return f"t={r['t']:.2f}s"
        return "从未"

    print()
    print(f"  {'判据':>5} | {'手调脚本首次':>14} | {'训练 command 首次':>18}")
    for i, k in enumerate(("S1", "S2", "INV")):
        print(f"  {k:>5} | {first('h', i):>14} | {first('c', i):>18}")

    print()
    print("  确认计时对拍 (手调脚本内部计时):")
    for k in ("cS1", "cINV"):
        vals = [r[k] for r in rows]
        print(f"    {k}: max={max(vals):.3f}s")
    print(f"  pose_confirm_s(训练配置)={float(cmd.cfg.pose_confirm_s)}  "
          f"脚本读取值={pol.pose_confirm_s}")
    print(f"  inverted_confirm_s(训练配置)={float(cmd.cfg.inverted_confirm_s)}  "
          f"脚本读取值={pol.inverted_confirm_s}")
    print(f"  window_late_s(训练配置)={float(cmd.cfg.window_late_s)}  "
          f"脚本读取值={pol.window_late_s}")

    print()
    print("  相位推进对拍 (手调脚本 phase vs 训练 command phase, 逐帧):")
    phase_name = {0: "P1", 1: "P2", 2: "P3"}
    ph_mismatch = 0
    first_ph_mismatch = None
    for r in rows:
        # 必须用逐帧记录的 cph: 在循环外读 cmd.phase 会拿"最终相位"比"逐帧相位", 恒报不一致。
        h_ph = {"P1": 0, "P2": 1, "P3": 2}.get(r["ph"], r["cph"])
        if r["cph"] != h_ph:
            ph_mismatch += 1
            if first_ph_mismatch is None:
                first_ph_mismatch = (r["t"], h_ph, r["cph"])
    print(f"    不一致帧数 = {ph_mismatch} / {len(rows)}")
    if first_ph_mismatch is not None:
        t0, h_ph, c_ph = first_ph_mismatch
        print(f"    首个不一致: t={t0:.2f}s 手调={phase_name.get(h_ph, '?')} 训练={phase_name.get(c_ph, '?')}")
    # 每处相位边界允许 1 帧错位: 手调侧在 env.step 之前推进时钟, 训练 command 在 step 之内,
    # 两者天生相差一步。超过"边界数×1 帧"才是真的状态机分歧。
    print(f"    说明: ≤3 帧(三个边界各 1 帧)属固有错位, >3 帧才是状态机分歧")

    env.close()
    ok = sum(n_mismatch.values()) == 0
    print(f"\n  [检测对拍] {'逐帧一致' if ok else '** 存在不一致 **'}")
    return ok


def main() -> None:
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    ok_ref = cmp_reference()
    ok_det = cmp_detection(lam)
    print()
    print("=" * 78)
    print(f"总判定: 参考 {'一致' if ok_ref else '不一致'} | 检测 {'逐帧一致' if ok_det else '不一致'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
