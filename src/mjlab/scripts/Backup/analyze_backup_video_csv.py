from __future__ import annotations
import sys
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from mjlab.tasks.SQuRo_Backup.mdp.reference import _generate_reference_table

# 回放 CSV 分析 — 只读录像 CSV, 用 *_ref_pos / *_ref_vel 列反推策略当时处在哪一段参考
# 用途: 判断策略是"没走到 S2"、"走到 S2 又退回 P2", 还是"根本没翻过去"。
#
# 说明: 该 CSV 由录像器写出, 不含 backup_phase 列, 因此这里反推名义时间与所属段。
# 两个必须注意的坑:
#   1) P3 起点的 14 维参考位置与 P1 起点完全相同 (脊柱全零 + 腿支撑位), 只匹配位置无法区分,
#      必须把参考速度一起匹配 (T1 起点脊柱速度非零, T4 起点脊柱速度为零而腿速度非零)。
#   2) P2 的名义时间被限幅在 0.95, 与 P3 起点重合; 只能用腿参考是否偏离支撑位来区分。

JOINT_NAMES = ("F_spine1", "F_body", "Neck_yaw", "Neck_pitch",
               "FL_shoulder", "FL_elbow", "FR_shoulder", "FR_elbow",
               "H_spine1", "H_body", "HL_hip", "HL_knee", "HR_hip", "HR_knee")
LEG_NAMES = ("FL_shoulder", "FL_elbow", "FR_shoulder", "FR_elbow",
             "HL_hip", "HL_knee", "HR_hip", "HR_knee")
SPINE_NAMES = ("F_spine1", "F_body", "H_spine1", "H_body")
SUPPORT_POSE = np.array([-0.28, 0.55, -0.28, 0.55, -1.50, -0.25, -1.50, -0.25])
SEGMENTS = ("T1", "T2", "T3", "P2保持", "T4/T5")
DT = 0.01
_REF_DT = 0.005


def infer_nominal_time(df: pd.DataFrame, lam: float) -> tuple[np.ndarray, np.ndarray]:
    t_tab, ref = _generate_reference_table()
    # 训练侧参考速度 = d(ref)/dt_nom / λ, 匹配前必须同样除以 λ。
    vel_tab = np.gradient(ref, _REF_DT, axis=0) / max(lam, 1e-6)
    cand = np.concatenate([ref, vel_tab], axis=1)
    got = np.concatenate([
        df[[f"{n}_ref_pos" for n in JOINT_NAMES]].to_numpy(dtype=np.float64),
        df[[f"{n}_ref_vel" for n in JOINT_NAMES]].to_numpy(dtype=np.float64),
    ], axis=1)
    scale = np.array([1.0] * 14 + [0.5] * 14)
    d = np.linalg.norm((got[:, None, :] - cand[None, :, :]) * scale, axis=2)
    idx = np.argmin(d, axis=1)
    return t_tab[idx], d[np.arange(len(idx)), idx]


def segment_of(df: pd.DataFrame, t_nom: np.ndarray) -> np.ndarray:
    leg_ref = df[[f"{n}_ref_pos" for n in LEG_NAMES]].to_numpy(dtype=np.float64)
    leg_dev = np.abs(leg_ref - SUPPORT_POSE).max(axis=1)
    seg = np.full(t_nom.shape, "T1", dtype=object)
    seg[(t_nom >= 0.65) & (t_nom < 0.80)] = "T2"
    seg[(t_nom >= 0.80) & (t_nom < 0.95 - 1e-9)] = "T3"
    seg[t_nom >= 0.95 - 1e-9] = "T4/T5"
    seg[(t_nom >= 0.95 - 1e-9) & (leg_dev < 1e-6)] = "P2保持"
    return seg


def main() -> None:
    path = sys.argv[1]
    every = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    df = pd.read_csv(path)
    n = len(df)
    t = df["step"].to_numpy(dtype=float) * DT
    lam = float(df["time_scale_command"].mean())
    t_nom, err = infer_nominal_time(df, lam)
    seg = segment_of(df, t_nom)
    r = df["reward"].to_numpy(dtype=float)
    print(f"文件: {path}")
    print(f"步数 {n} ({n*DT:.2f}s 实际, λ={lam:.2f} -> 名义 {n*DT/lam:.2f}s)  "
          f"参考匹配误差 max={err.max():.4f} mean={err.mean():.4f}")
    print(f"nominal 范围 {t_nom.min():.3f} .. {t_nom.max():.3f}")

    print("\n段停留占比:")
    for name in SEGMENTS:
        m = seg == name
        mr = r[m].mean() if m.any() else float("nan")
        print(f"  {name:8s} {m.mean()*100:6.2f}%  ({m.sum()} 步)  步奖励均值 {mr:+.4f}")
    switches = int((seg[1:] != seg[:-1]).sum())
    print(f"段切换次数: {switches}  (平均每 {n/max(switches,1):.1f} 步一次)")

    print(f"\n时间表 (每 {every} 步):")
    print(f"{'t':>5} {'nom':>6} {'seg':>7} {'upright':>8} {'h_act':>7} " +
          " ".join(f"{s+'_pos':>8} {s+'_ref':>8}" for s in SPINE_NAMES) +
          f" {'腿|err|':>8} {'reward':>8} {'done':>4}")
    for i in range(0, n, every):
        row = df.iloc[i]
        legd = np.mean([abs(row[f"{c}_pos"] - row[f"{c}_ref_pos"]) for c in LEG_NAMES])
        spn = " ".join(f"{row[f'{c}_pos']:+8.3f} {row[f'{c}_ref_pos']:+8.3f}" for c in SPINE_NAMES)
        print(f"{t[i]:5.2f} {t_nom[i]:6.3f} {seg[i]:>7} {row['uprightness']:+8.3f} "
              f"{row['height_actual']:7.4f} {spn} {legd:8.4f} {row['reward']:+8.3f} {int(row['done']):4d}")

    up = df["uprightness"].to_numpy(dtype=float)
    h = df["height_actual"].to_numpy(dtype=float)
    print("\n姿态统计 (uprightness: +1=俯卧/背朝上, -1=仰卧/腹朝上):")
    print(f"  min={up.min():+.3f} max={up.max():+.3f} 末值={up[-1]:+.3f}")
    print(f"  俯卧(>0.7)占比 {(up > 0.7).mean()*100:.1f}%   仰卧(<-0.7)占比 {(up < -0.7).mean()*100:.1f}%")
    print(f"  height_actual: 起 {h[0]:.4f} 末 {h[-1]:.4f} max {h.max():.4f}  (站立目标 0.055)")
    print(f"  步奖励均值 {r.mean():+.4f}  累计 {r.sum():+.3f}")

    print("\n分段统计 (reward 为单步总奖励, 已含 dt 与权重):")
    print(f"  {'段':>8} {'步数':>5} {'步奖励均值':>10} {'奖励合计':>10} {'剔除脉冲后':>10} {'脊柱|err|':>10} {'腿|err|':>10}")
    for name in SEGMENTS:
        m = seg == name
        if not m.any():
            continue
        spn_err = np.mean([np.abs(df[f"{c}_pos"] - df[f"{c}_ref_pos"]).to_numpy()[m] for c in SPINE_NAMES])
        leg_err = np.mean([np.abs(df[f"{c}_pos"] - df[f"{c}_ref_pos"]).to_numpy()[m] for c in LEG_NAMES])
        # 里程碑是一次性脉冲 (10/15/35), 会计入所在段的均值, 比较"段之间每步收益"时必须剔除
        clean = r[m] * (np.abs(r[m]) <= 1.0)
        print(f"  {name:>8} {m.sum():5d} {r[m].mean():10.4f} {r[m].sum():10.3f} "
              f"{clean.sum()/m.sum():10.4f} {spn_err:10.4f} {leg_err:10.4f}")

    print("\n一次性脉冲行 (|reward| > 1, 即里程碑入账步):")
    for i in np.where(np.abs(r) > 1.0)[0]:
        print(f"  t={t[i]:5.2f} nom={t_nom[i]:.3f} seg={seg[i]:>7} reward={r[i]:+8.3f} "
              f"upright={up[i]:+.3f} h={h[i]:.4f}")

    print("\n腿跟踪 (全段均值):")
    print(f"  {'关节':>13} {'pos均值':>9} {'ref均值':>9} {'|err|均值':>10}")
    for c in LEG_NAMES:
        e = np.abs(df[f"{c}_pos"] - df[f"{c}_ref_pos"])
        print(f"  {c:>13} {df[f'{c}_pos'].mean():+9.3f} {df[f'{c}_ref_pos'].mean():+9.3f} {e.mean():10.4f}")


if __name__ == "__main__":
    main()
