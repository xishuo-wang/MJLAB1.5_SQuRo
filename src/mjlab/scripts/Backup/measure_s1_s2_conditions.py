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


# 手调回放的 S1/S2 分项测量
#
# 判据来源: 直接调用训练环境的 BackupCommand._check_S1 / _check_S2 / _check_both_inverted
#          (不做任何重实现, 避免"用近似式代替真判据"的错误)
# 分项来源: 同一帧读出 — 两段的归一化姿态余弦 u 与 body 高度
#
# 目的: 回答 backup_s1_ok / backup_s2_ok 长期为 0 时, 到底卡在朝向还是高度
#
# S1 = (前段倒置) & (后段正置) & (zF < 0.03) & (zH < 0.03)
# S2 = (前段正置) & (后段正置) & (zF < 0.04) & (zH < 0.04)

G_TH_S1 = 0.03
G_TH_S2 = 0.04
COS45 = 0.7071067811865476


def main() -> None:
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0 * lam

    cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    cfg.scene.num_envs = 1
    cfg.commands["backup_cmd"].fixed_time_scale = lam  # type: ignore[attr-defined]
    for tc in cfg.terminations.values():
        tc.func = lambda e: torch.zeros(e.num_envs, dtype=torch.bool, device=e.device)
    cfg.episode_length_s = dur + 1.0

    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
    env.reset()
    asset = env.unwrapped.scene.entities["robot"]
    resolve_model_indices(asset)
    cmd = env.unwrapped.command_manager.get_term("backup_cmd")
    pol = StateMachinePolicy(env, lam, 0, log_events=False)

    fb, hb = _MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id
    pairs = _MODEL_INDICES.segment_belly_back_ids
    assert pairs is not None

    def u_norm(idx: int) -> float:
        b, k = pairs[idx]
        d = (asset.data.site_pos_w[0, k] - asset.data.site_pos_w[0, b]).detach().cpu().numpy()
        n = float(np.linalg.norm(d))
        return float(d[2] / n) if n > 1e-12 else float("nan")

    rows = []
    with torch.no_grad():
        for _ in range(int(dur / env.step_dt)):
            env.step(pol(env.unwrapped.get_observations()))
            d = asset.data
            rows.append({
                "t": pol._elapsed,
                "phase": pol.phase,
                "uF": u_norm(0),
                "uH": u_norm(1),
                "zF": float(d.body_link_pos_w[0, fb, 2]),
                "zH": float(d.body_link_pos_w[0, hb, 2]),
                "S1": bool(cmd._check_S1()[0]),
                "S2": bool(cmd._check_S2()[0]),
                "INV": bool(cmd._check_both_inverted()[0]),
                "cS1": pol._s1_confirm_t,
                "cS2": pol._s2_confirm_t,
            })
            if pol.phase == "DONE":
                break

    print(f"λ={lam}  时长 {rows[-1]['t']:.2f}s  步数 {len(rows)}  最终 phase={pol.phase}")
    print(f"手调时序: T1={0.65} T2={0.3} T3={0.3} T4={0.5} (名义秒), P1 截止={0.95}×λ={0.95*lam:.2f}s")
    print()
    print(f"{'t':>6} {'t/λ':>6} {'ph':>3} | {'uF':>7} {'uH':>7} | {'zF':>7} {'zH':>7} | "
          f"{'F倒':>4} {'H正':>4} {'F正':>4} | {'S1':>3} {'S2':>3} {'INV':>4} | {'cS1':>5} {'cS2':>5}")
    step = max(1, len(rows) // 45)
    for i in range(0, len(rows), step):
        r = rows[i]
        f_inv = r["uF"] <= -COS45
        h_up = r["uH"] >= COS45
        f_up = r["uF"] >= COS45
        print(f"{r['t']:6.2f} {r['t']/lam:6.2f} {r['phase']:>3} | {r['uF']:+7.3f} {r['uH']:+7.3f} | "
              f"{r['zF']:7.4f} {r['zH']:7.4f} | "
              f"{int(f_inv):>4} {int(h_up):>4} {int(f_up):>4} | "
              f"{int(r['S1']):>3} {int(r['S2']):>3} {int(r['INV']):>4} | "
              f"{r['cS1']:5.2f} {r['cS2']:5.2f}")

    # ---- 汇总 ----
    def first_true(key: str) -> str:
        for r in rows:
            if r[key]:
                return f"t={r['t']:.2f}s (t/λ={r['t']/lam:.3f})"
        return "从未成立"

    print()
    print("=== 关键事件首次成立时刻 ===")
    for k, name in [("S1", "S1 四条件同时"), ("S2", "S2 四条件同时"), ("INV", "双倒")]:
        print(f"  {name:16s} {first_true(k)}")

    # 分项统计: 各单项在整个过程中成立过没有
    print()
    print("=== 各项条件是否曾成立 (用于定位卡点) ===")
    def ever(pred) -> str:
        hits = [r for r in rows if pred(r)]
        if not hits:
            return "从未成立"
        return f"成立 {len(hits)} 帧, t∈[{hits[0]['t']:.2f}, {hits[-1]['t']:.2f}]s"

    print(f"  前段倒置 (uF<=-0.707)      : {ever(lambda r: r['uF'] <= -COS45)}")
    print(f"  后段正置 (uH>=+0.707)      : {ever(lambda r: r['uH'] >= COS45)}")
    print(f"  前段正置 (uF>=+0.707)      : {ever(lambda r: r['uF'] >= COS45)}")
    print(f"  zF < {G_TH_S1} (S1)          : {ever(lambda r: r['zF'] < G_TH_S1)}")
    print(f"  zH < {G_TH_S1} (S1)          : {ever(lambda r: r['zH'] < G_TH_S1)}")
    print(f"  zF < {G_TH_S2} (S2)          : {ever(lambda r: r['zF'] < G_TH_S2)}")
    print(f"  zH < {G_TH_S2} (S2)          : {ever(lambda r: r['zH'] < G_TH_S2)}")

    # S1/S2 缺哪一项
    print()
    print("=== S1 各条件同时成立的帧数 (逐项缺口) ===")
    c = {
        "F倒": lambda r: r["uF"] <= -COS45,
        "H正": lambda r: r["uH"] >= COS45,
        "zF<0.03": lambda r: r["zF"] < G_TH_S1,
        "zH<0.03": lambda r: r["zH"] < G_TH_S1,
    }
    n = len(rows)
    for k, f in c.items():
        print(f"  {k:10s} {sum(1 for r in rows if f(r)):5d}/{n}")
    print(f"  {'全部同时':10s} {sum(1 for r in rows if all(f(r) for f in c.values())):5d}/{n}")

    print()
    print("=== S2 各条件同时成立的帧数 (逐项缺口) ===")
    c2 = {
        "F正": lambda r: r["uF"] >= COS45,
        "H正": lambda r: r["uH"] >= COS45,
        "zF<0.04": lambda r: r["zF"] < G_TH_S2,
        "zH<0.04": lambda r: r["zH"] < G_TH_S2,
    }
    for k, f in c2.items():
        print(f"  {k:10s} {sum(1 for r in rows if f(r)):5d}/{n}")
    print(f"  {'全部同时':10s} {sum(1 for r in rows if all(f(r) for f in c2.values())):5d}/{n}")

    env.close()


if __name__ == "__main__":
    main()
