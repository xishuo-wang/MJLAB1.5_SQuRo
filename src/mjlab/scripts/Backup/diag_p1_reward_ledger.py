# P1 段密集奖励账本 — 用回放 CSV 逐帧重算 P1 期间的 mimic_pos/mimic_vel/spine_target 金额
# 目的: 判断"跳过 P1 斜坡"要付出多少密集奖励, 与"早到"的时间收益对比。
# 核函数与 mdp/rewards.py 的 mimic_pos/mimic_vel 同源 (参照 CSV 的 *_ref_pos/*_ref_vel 列)。
# 用法: uv run python -B -m mjlab.scripts.Backup.diag_p1_reward_ledger --csv <回放csv>
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro

from mjlab.tasks.SQuRo_Backup.mdp.curriculums import _CURVES

# 与 rewards.py 的 mimic_pos 分组一致 (参考表顺序)
SPINE = ["F_spine1", "F_body", "H_spine1", "H_body"]
NECK = ["Neck_yaw", "Neck_pitch"]
LEG = ["FL_shoulder", "FL_elbow", "FR_shoulder", "FR_elbow",
       "HL_hip", "HL_knee", "HR_hip", "HR_knee"]


@dataclass
class LedgerCfg:
    csv: str = "logs/rsl_rl/SQuRo_Backup/2026-09-19_16-49-35/videos/SQuRo_Backup_2999-ts2.00.csv"
    # P1 段的名义时长; None 时按 λ 自动推 (0.80*λ)
    p1_end_s: float | None = None
    dt: float = 0.01


# 分组权重: 脊柱/颈部/腿各自的 σ, 与 curriculums 的 _CURVES 保持一致
def group_sigma() -> dict[str, float]:
    return {
        "spn": _CURVES["sigma_spn_pos"][0] * _CURVES["sigma_spn_pos"][0],
        "neck": _CURVES["sigma_neck_pos"][0] * _CURVES["sigma_neck_pos"][0],
        "leg": _CURVES["sigma_leg_pos"][0] * _CURVES["sigma_leg_pos"][0],
    }


def main() -> None:
    cfg = tyro.cli(LedgerCfg)
    df = pd.read_csv(cfg.csv)
    lam = 1.0
    stem = Path(cfg.csv).stem
    if "-ts" in stem:
        lam = float(stem.split("-ts")[-1])
    p1_end = cfg.p1_end_s if cfg.p1_end_s is not None else 0.80 * lam
    n_p1 = int(round(p1_end / cfg.dt))
    print(f"文件: {Path(cfg.csv).name}  λ={lam:g}  P1 名义时长={p1_end:.3f} s  取前 {n_p1} 帧")
    print(f"P1 帧数 {n_p1} / 总帧数 {len(df)} = {n_p1 / len(df):.1%} 的回合时间")
    print()

    w_pos = _CURVES["weight_mimic_pos"][0]
    w_vel = _CURVES["weight_mimic_vel"][0]
    w_spn = _CURVES["weight_spine_target"][0]
    sg = group_sigma()

    # 逐帧重算 exp 核: 按分组取平均
    def kernel(prefix: str, sigma: float, half: float = 0.5) -> np.ndarray:
        err = np.zeros(len(df))
        cnt = 0
        for j in (SPINE if prefix == "spn" else NECK if prefix == "neck" else LEG):
            pc, rc = j + "_pos", j + "_ref_pos"
            if pc in df.columns and rc in df.columns:
                err += (df[pc].values - df[rc].values) ** 2
                cnt += 1
        if cnt == 0:
            return np.zeros(len(df))
        return half * np.exp(-(err / cnt) / sigma)

    r_pos = (kernel("spn", sg["spn"]) + kernel("neck", sg["neck"]) + kernel("leg", sg["leg"])) * w_pos
    # mimic_vel 同构: 用 *_vel 列
    errv = np.zeros(len(df))
    cntv = 0
    for j in SPINE + NECK + LEG:
        vc, vr = j + "_vel", j + "_ref_vel"
        if vc in df.columns and vr in df.columns:
            errv += (df[vc].values - df[vr].values) ** 2
            cntv += 1
    r_vel = (0.5 * np.exp(-(errv / max(cntv, 1)) / (0.5 * 0.5))) * w_vel if cntv else np.zeros(len(df))
    # spine_target: -mean((target-ref)^2) * 2 ; target = default(0) + action*0.3
    tgt = np.stack([df[j + "_action"].values * 0.3 for j in SPINE], axis=1)
    ref = np.stack([df[j + "_ref_pos"].values for j in SPINE], axis=1)
    r_spn = -((tgt - ref) ** 2).mean(axis=1) * w_spn

    p1 = slice(0, n_p1)
    # Episode_Reward 口径: 回合累计 / 回合总长(s)
    ep_len = len(df) * cfg.dt
    print("%-16s %14s %14s %14s" % ("项", "P1累计", "整回合累计", "P1占整回合"))
    for name, arr in (("mimic_pos", r_pos), ("mimic_vel", r_vel), ("spine_target", r_spn)):
        tot = float(arr.sum()) * cfg.dt
        p1v = float(arr[p1].sum()) * cfg.dt
        share = p1v / tot if abs(tot) > 1e-9 else float("nan")
        print("%-16s %14.4f %14.4f %13.1f%%" % (name, p1v, tot, share * 100))
    print()
    print("换算到 TensorBoard 的 Episode_Reward 口径 (÷回合总长 %.2f s):" % ep_len)
    for name, arr in (("mimic_pos", r_pos), ("mimic_vel", r_vel), ("spine_target", r_spn)):
        print("  %-16s P1 贡献 %+.4f /s   整回合 %+.4f /s" % (
            name, float(arr[p1].sum()) * cfg.dt / ep_len, float(arr.sum()) * cfg.dt / ep_len))
    print()
    # 名义跟踪(假设实际=参考)在同一窗口能拿多少 —— 即"这笔钱的上限"
    print("参考: 若 P1 期间实际角完全等于参考角, mimic_pos 在 P1 的上限 = %.4f /s" % (
        (0.5 * 3) * w_pos * n_p1 * cfg.dt / ep_len))


if __name__ == "__main__":
    main()
