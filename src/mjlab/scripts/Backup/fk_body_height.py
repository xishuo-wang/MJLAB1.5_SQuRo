from __future__ import annotations
import sys
import argparse
import numpy as np
import pandas as pd
import mujoco
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


# ==================================================================================================
# 用 MuJoCo 正向运动学精确计算 F/H body 高度 (CSV 未记录该量)
# 输入: 策略/回放 CSV 的 4 个脊柱角 + 8 个腿角 + base 位姿
# 输出: 逐帧 zF / zH + S1/S2 四条件判定

import mjlab.asset_zoo.robots.SQuRo as _sq_pkg
XML = Path(_sq_pkg.__file__).resolve().parent / "xmls" / "SQuRo.xml"

UP_TH = 0.5
GROUND_TH = 0.03
GROUND_TH_S2 = 0.04

# reference.py 的时变期望高度参考 (从 backup_body_traj.npy 得到), 仅作对照
LEG_JOINTS = ["FL_shoulder_joint", "FL_elbow_joint", "FR_shoulder_joint", "FR_elbow_joint",
              "HL_hip_joint", "HL_knee_joint", "HR_hip_joint", "HR_knee_joint"]
SPINE_JOINTS = ["F_spine1_joint", "F_body_joint", "H_spine1_joint", "H_body_joint"]
CSV_SPINE = ["F_spine1_pos", "F_body_pos", "H_spine1_pos", "H_body_pos"]
CSV_LEG = ["FL_shoulder_pos", "FL_elbow_pos", "FR_shoulder_pos", "FR_elbow_pos",
           "HL_hip_pos", "HL_knee_pos", "HR_hip_pos", "HR_knee_pos"]


def build() -> tuple[mujoco.MjModel, mujoco.MjData, dict, dict]:
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    jadr = {}
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        jadr[nm] = m.jnt_qposadr[j]
    bid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b): b for b in range(m.nbody)}
    return m, d, jadr, bid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--t-min", type=float, default=0.0)
    ap.add_argument("--t-max", type=float, default=1e9)
    ap.add_argument("--step", type=int, default=0, help="打印步长, 0=自动")
    args = ap.parse_args()

    m, d, jadr, bid = build()
    fb, hb = bid["F_body_Link"], bid["H_body_Link"]

    df = pd.read_csv(args.csv)
    have_legs = all(c in df.columns for c in CSV_LEG)
    print(f"CSV: {Path(args.csv).name}  行数={len(df)}")
    print(f"腿角列完整: {have_legs}" + ("" if have_legs else "  → 腿角按 0 处理(会有偏差)"))

    t = df["time"].to_numpy(float) if "time" in df.columns else df["step"].to_numpy(float) * args.dt
    base = df[["base_pos_x", "base_pos_y", "base_pos_z"]].to_numpy(float) if "base_pos_z" in df.columns \
        else np.zeros((len(df), 3))

    zF = np.zeros(len(df)); zH = np.zeros(len(df))
    yF = np.zeros(len(df)); yH = np.zeros(len(df))
    upF = np.zeros(len(df)); upH = np.zeros(len(df))
    for i in range(len(df)):
        d.qpos[:] = 0.0
        d.qpos[0:3] = base[i]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]      # CSV 未记 base 四元数, 用单位四元数近似
        for nm, col in zip(SPINE_JOINTS, CSV_SPINE):
            d.qpos[jadr[nm]] = float(df[col].iloc[i])
        if have_legs:
            for nm, col in zip(LEG_JOINTS, CSV_LEG):
                d.qpos[jadr[nm]] = float(df[col].iloc[i])
        mujoco.mj_kinematics(m, d)
        pF, pH = d.xpos[fb], d.xpos[hb]
        zF[i], zH[i] = pF[2], pH[2]
        yF[i], yH[i] = pF[1], pH[1]
        # 背腹轴 = body 局部 +Y 在世界系 Z 的分量
        upF[i] = d.xmat[fb].reshape(3, 3)[1, 2]
        upH[i] = -d.xmat[hb].reshape(3, 3)[1, 2]

    out = pd.DataFrame({"t": t, "zF": zF, "zH": zH, "yF": yF, "yH": yH, "upF": upF, "upH": upH})
    m2 = (out["t"] >= args.t_min) & (out["t"] <= args.t_max)
    dsp = out[m2].reset_index(drop=True)
    if len(dsp) == 0:
        print("时间窗内无数据")
        return

    print(f"\n注意: base 四元数未记录, 高度用 base_pos_z + 单位四元数近似 (FK 主要误差来自姿态翻转)")
    print(f"\n{'t':>6} {'upF':>7} {'upH':>7} {'zF':>8} {'zH':>8} | {'c1':>3} {'c2':>3} {'c3':>3} {'c4':>3} | S1")
    step = args.step or max(1, len(dsp) // 35)
    for i in range(0, len(dsp), step):
        r = dsp.iloc[i]
        c1, c2 = r["upF"] > UP_TH, r["upH"] < -UP_TH
        c3, c4 = r["zF"] < GROUND_TH, r["zH"] < GROUND_TH
        s1 = c1 and c2 and c3 and c4
        print(f"{r['t']:6.2f} {r['upF']:+7.3f} {r['upH']:+7.3f} {r['zF']:8.4f} {r['zH']:8.4f} | "
              f"{int(c1):>3} {int(c2):>3} {int(c3):>3} {int(c4):>3} | {'✔' if s1 else '✘'}")

    c1 = int((dsp["upF"] > UP_TH).sum()); c2 = int((dsp["upH"] < -UP_TH).sum())
    c3 = int((dsp["zF"] < GROUND_TH).sum()); c4 = int((dsp["zH"] < GROUND_TH).sum())
    s1 = int(((dsp["upF"] > UP_TH) & (dsp["upH"] < -UP_TH) & (dsp["zF"] < GROUND_TH) & (dsp["zH"] < GROUND_TH)).sum())
    print(f"\n[条件命中 / {len(dsp)}]  c1={c1}  c2={c2}  c3={c3}  c4={c4}   S1={s1}")
    print(f"  zF: [{dsp['zF'].min():+.4f}, {dsp['zF'].max():+.4f}]   zH: [{dsp['zH'].min():+.4f}, {dsp['zH'].max():+.4f}]")
    print(f"  upF: [{dsp['upF'].min():+.3f}, {dsp['upF'].max():+.3f}]   upH: [{dsp['upH'].min():+.3f}, {dsp['upH'].max():+.3f}]")
    print(f"  yF: [{dsp['yF'].min():+.4f}, {dsp['yF'].max():+.4f}]   yH: [{dsp['yH'].min():+.4f}, {dsp['yH'].max():+.4f}]")


if __name__ == "__main__":
    main()
