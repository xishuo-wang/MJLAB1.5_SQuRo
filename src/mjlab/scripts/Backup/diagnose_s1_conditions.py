from __future__ import annotations
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


# ==================================================================================================
# S1/S2 达标诊断: 逐帧检查状态机的 4 个几何条件, 定位是哪一个不满足
# S1: 背腹F>0.5 且 背腹H<-0.5 且 zF<0.03 且 zH<0.03
# 输入 CSV 需含 F_body_Link / H_body_Link 的高度与姿态列

UP_TH = 0.5
GROUND_TH = 0.03
GROUND_TH_S2 = 0.04


def dorsoventral(f_sp1, f_bd, h_sp1, h_bd):
    return np.sin(f_sp1 + f_bd), np.sin(h_sp1 + h_bd)


def load(csv_path: Path, dt: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    t = df["time"].to_numpy(float) if "time" in df.columns else df["step"].to_numpy(float) * dt
    out = pd.DataFrame({"t": t})
    for c in df.columns:
        out[c] = df[c]
    if "F_spine1_pos" in df.columns:
        out["upF"], out["upH"] = dorsoventral(
            df["F_spine1_pos"].to_numpy(float), df["F_body_pos"].to_numpy(float),
            df["H_spine1_pos"].to_numpy(float), df["H_body_pos"].to_numpy(float))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--t-min", type=float, default=0.0)
    ap.add_argument("--t-max", type=float, default=1e9)
    args = ap.parse_args()

    df = load(Path(args.csv), args.dt)
    print(f"列: {[c for c in df.columns][:40]}")
    print(f"行数 {len(df)}, t 范围 [{df['t'].iloc[0]:.2f}, {df['t'].iloc[-1]:.2f}]")
    print()

    # 猜测 body 高度列名
    zf_col = next((c for c in df.columns if c.lower() in ("f_body_z", "body_f_z", "zf")), None)
    zh_col = next((c for c in df.columns if c.lower() in ("h_body_z", "body_h_z", "zh")), None)
    print(f"高度列: zF={zf_col}  zH={zh_col}")

    if zf_col is None or zh_col is None:
        # 退而求其次: 从 base_pos_z + 姿态推算
        if "base_pos_z" in df.columns and "upF" in df.columns:
            # F/H body 中心相对 base 沿 base 局部 X 的距离 ~0.0468 m
            OFF = 0.0468
            df["zF_est"] = df["base_pos_z"].to_numpy() + OFF * df["upF"].to_numpy()
            df["zH_est"] = df["base_pos_z"].to_numpy() + OFF * df["upH"].to_numpy()
            zf_col, zh_col = "zF_est", "zH_est"
            print(f"  未找到 body 高度列, 用 base_pos_z + 0.0468*背腹轴 估算")

    m = (df["t"] >= args.t_min) & (df["t"] <= args.t_max)
    d = df[m].copy()
    if len(d) == 0:
        print("时间窗内无数据")
        return

    print(f"\n{'t':>6} {'背腹F':>7} {'背腹H':>7} {'zF':>7} {'zH':>7} | "
          f"{'c1':>3} {'c2':>3} {'c3':>3} {'c4':>3} | S1  S2")
    step = max(1, len(d) // 40)
    for i in range(0, len(d), step):
        r = d.iloc[i]
        c1 = r["upF"] > UP_TH
        c2 = r["upH"] < -UP_TH
        c3 = r[zf_col] < GROUND_TH
        c4 = r[zh_col] < GROUND_TH
        s1 = c1 and c2 and c3 and c4
        s2 = (r["upF"] < -UP_TH) and (r["upH"] < -UP_TH) and \
             (r[zf_col] < GROUND_TH_S2) and (r[zh_col] < GROUND_TH_S2)
        print(f"{r['t']:6.2f} {r['upF']:+7.3f} {r['upH']:+7.3f} {r[zf_col]:7.4f} {r[zh_col]:7.4f} | "
              f"{int(c1):>3} {int(c2):>3} {int(c3):>3} {int(c4):>3} | "
              f"{'✔' if s1 else '✘':>2}  {'✔' if s2 else '✘':>2}")

    c1 = (d["upF"] > UP_TH).sum(); c2 = (d["upH"] < -UP_TH).sum()
    c3 = (d[zf_col] < GROUND_TH).sum(); c4 = (d[zh_col] < GROUND_TH).sum()
    s1 = ((d["upF"] > UP_TH) & (d["upH"] < -UP_TH) & (d[zf_col] < GROUND_TH) & (d[zh_col] < GROUND_TH)).sum()
    print(f"\n[条件命中帧数 / {len(d)}]")
    print(f"  c1 背腹F>{UP_TH}      : {c1:4d}")
    print(f"  c2 背腹H<-{UP_TH}     : {c2:4d}")
    print(f"  c3 zF<{GROUND_TH}       : {c3:4d}")
    print(f"  c4 zH<{GROUND_TH}       : {c4:4d}")
    print(f"  S1 四条件同时满足      : {s1:4d}")
    print(f"\n  背腹F: [{d['upF'].min():+.3f}, {d['upF'].max():+.3f}]")
    print(f"  背腹H: [{d['upH'].min():+.3f}, {d['upH'].max():+.3f}]")
    print(f"  zF   : [{d[zf_col].min():.4f}, {d[zf_col].max():.4f}]   最小值@t={d['t'].iloc[int(d[zf_col].to_numpy().argmin())]:.2f}s")
    print(f"  zH   : [{d[zh_col].min():.4f}, {d[zh_col].max():.4f}]   最小值@t={d['t'].iloc[int(d[zh_col].to_numpy().argmin())]:.2f}s")
    if "base_pos_z" in d.columns:
        print(f"  base_z: [{d['base_pos_z'].min():.4f}, {d['base_pos_z'].max():.4f}]")


if __name__ == "__main__":
    main()
