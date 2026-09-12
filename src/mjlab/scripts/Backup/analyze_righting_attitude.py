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
# 翻正任务离线分析: 从关节角反推 F/H body 的真实空间姿态, 对比手调参考
#
# 关节链 (SQuRo.xml):
#   base_Link -> FU_spine -> F_spine1 -> F_spine2 -> F_body
#   base_Link -> HL/HR_spine -> H_spine1 -> H_spine2 -> H_body
# 固定安装旋转折叠后, 两链的净姿态均为:
#   R(θ_s, θ_b) = Ry(θ_s) · Rx(-π/2) · Ry(θ_b) · Rx(+π/2)
# 由此解析可得 (与 command.py/_body_up 的 2(yz+wx) 完全等价):
#   背腹轴(F 局部 +Y, H 局部 +Y 取负) 的世界 Z 分量 = sin(θ_s + θ_b)
#   前后轴(局部 +X)              的世界 Z 分量 = -sin(θ_b)·cos(θ_s)
# 注: θ_s = 侧摆/俯仰关节 (F_spine1 / H_spine1), θ_b = 扭转关节 (F_body / H_body)

DT_DEFAULT = 0.01

# 手调参考表 (reference.py _generate_reference_table) 的分段, 名义秒 (λ=1)
SEG1_END = 0.65          # P1 构建段末 (扭转到位)
SEG2_END = 1.30          # P1 回收段末 (侧摆/俯仰回零)
ACTION_END = 1.45        # P2 动作段末 (扭转回零)
TRANS_END = 1.95         # 站立过渡段末


# 背腹轴世界 Z 分量 (±1=完全正置/倒置, 0=侧立)
def dorsoventral(f_sp1, f_bd, h_sp1, h_bd):
    return np.sin(f_sp1 + f_bd), np.sin(h_sp1 + h_bd)


# 前后轴世界 Z 分量 (±1=躯干竖直, 0=躯干水平)
def longitudinal(f_sp1, f_bd, h_sp1, h_bd):
    return -np.sin(f_bd) * np.cos(f_sp1), -np.sin(h_bd) * np.cos(h_sp1)


# 姿态判读标签
def judge(up: float, ax: float) -> str:
    s = "正置" if up > 0.5 else ("倒置" if up < -0.5 else "侧立")
    if abs(ax) > 0.7:
        s += "·躯干竖直"
    else:
        s += "·躯干水平"
    return s


# 手调参考的名义轨迹 (与 reference.py 分段线性完全同源)
def reference_trace(lam: float, n: int, dt: float) -> pd.DataFrame:
    t = np.arange(n) * dt
    tn = t / lam
    f_sp1 = np.zeros(n); f_bd = np.zeros(n); h_sp1 = np.zeros(n); h_bd = np.zeros(n)
    for i, s in enumerate(tn):
        if s < SEG1_END:
            u = s / SEG1_END
            f_sp1[i], f_bd[i], h_sp1[i], h_bd[i] = 0.6 * u, -1.57 * u, 0.6 * u, 1.57 * u
        elif s < SEG2_END:
            u = (s - SEG1_END) / (SEG2_END - SEG1_END)
            f_sp1[i], f_bd[i], h_sp1[i], h_bd[i] = 0.6 - 0.6 * u, -1.57, 0.6 - 0.6 * u, 1.57
        elif s < ACTION_END:
            u = (s - SEG2_END) / (ACTION_END - SEG2_END)
            f_sp1[i], f_bd[i], h_sp1[i], h_bd[i] = 0.6 * u, -1.57 + 1.57 * u, 0.0, 1.57 - 1.57 * u
        elif s < TRANS_END:
            u = (s - ACTION_END) / (TRANS_END - ACTION_END)
            f_sp1[i], f_bd[i], h_sp1[i], h_bd[i] = 0.6 * (1 - u), 0.0, 0.0, 0.0
    upF, upH = dorsoventral(f_sp1, f_bd, h_sp1, h_bd)
    axF, axH = longitudinal(f_sp1, f_bd, h_sp1, h_bd)
    return pd.DataFrame({
        "t": t, "tn": tn,
        "F_spine1": f_sp1, "F_body": f_bd, "H_spine1": h_sp1, "H_body": h_bd,
        "upF": upF, "upH": upH, "axF_z": axF, "axH_z": axH,
    })


# 读策略/回放 CSV
def load_trace(csv_path: Path, dt: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = ["F_spine1_pos", "F_body_pos", "H_spine1_pos", "H_body_pos"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"{csv_path.name} 缺少列: {miss}")
    t = df["time"].to_numpy(float) if "time" in df.columns else df["step"].to_numpy(float) * dt
    f_sp1 = df["F_spine1_pos"].to_numpy(float)
    f_bd = df["F_body_pos"].to_numpy(float)
    h_sp1 = df["H_spine1_pos"].to_numpy(float)
    h_bd = df["H_body_pos"].to_numpy(float)
    upF, upH = dorsoventral(f_sp1, f_bd, h_sp1, h_bd)
    axF, axH = longitudinal(f_sp1, f_bd, h_sp1, h_bd)
    out = pd.DataFrame({
        "t": t, "tn": t,
        "F_spine1": f_sp1, "F_body": f_bd, "H_spine1": h_sp1, "H_body": h_bd,
        "upF": upF, "upH": upH, "axF_z": axF, "axH_z": axH,
    })
    for extra in ["F_body_ref_pos", "F_spine1_ref_pos", "H_spine1_ref_pos", "H_body_ref_pos"]:
        if extra in df.columns:
            out[extra] = df[extra].to_numpy(float)
    return out


def show(df: pd.DataFrame, tag: str, key: str, times) -> None:
    print(f"\n=== {tag} ===")
    print(f"{key:>6} {'F_sp1':>7} {'F_body':>7} {'H_sp1':>7} {'H_body':>7} | "
          f"{'背腹F':>7} {'背腹H':>7} | {'前后F':>7} {'前后H':>7} | 判读")
    for tt in times:
        i = int(np.argmin(np.abs(df[key].to_numpy() - tt)))
        r = df.iloc[i]
        print(f"{r[key]:6.2f} {r['F_spine1']:+7.3f} {r['F_body']:+7.3f} {r['H_spine1']:+7.3f} "
              f"{r['H_body']:+7.3f} | {r['upF']:+7.3f} {r['upH']:+7.3f} | "
              f"{r['axF_z']:+7.3f} {r['axH_z']:+7.3f} | "
              f"F{judge(r['upF'], r['axF_z'])} H{judge(r['upH'], r['axH_z'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description="翻正: 关节角 -> 身体姿态 反推与对比")
    ap.add_argument("csv", nargs="?", help="策略/回放 CSV 路径")
    ap.add_argument("--dt", type=float, default=DT_DEFAULT)
    ap.add_argument("--lam", type=float, default=3.0, help="策略所用 time_scale")
    ap.add_argument("--duration", type=float, default=6.0)
    args = ap.parse_args()

    n = int(args.duration / args.dt)
    ref = reference_trace(args.lam, n, args.dt)
    show(ref, f"手调参考 (λ={args.lam})", "tn", np.linspace(0, args.duration, 16))

    print(f"\n[参考关键节点]  (λ={args.lam} → 实际时刻 = tn × λ)")
    for name, tn in [("起  始", 0.0), ("P1 构建段末", SEG1_END), ("P1 回收段末", SEG2_END),
                     ("P2 动作段末", ACTION_END), ("站立过渡末", TRANS_END)]:
        i = int(np.argmin(np.abs(ref["tn"].to_numpy() - tn)))
        r = ref.iloc[i]
        print(f"  {name:12s} tn={tn:.2f}s (t={tn * args.lam:5.2f}s)  "
              f"F_sp1={r['F_spine1']:+.3f} F_body={r['F_body']:+.3f} "
              f"H_sp1={r['H_spine1']:+.3f} H_body={r['H_body']:+.3f} | "
              f"背腹F={r['upF']:+.3f} 背腹H={r['upH']:+.3f} | "
              f"前后F={r['axF_z']:+.3f} 前后H={r['axH_z']:+.3f} | "
              f"F{judge(r['upF'], r['axF_z'])} H{judge(r['upH'], r['axH_z'])}")

    if args.csv:
        p = Path(args.csv)
        if not p.exists():
            raise FileNotFoundError(p)
        act = load_trace(p, args.dt)
        show(act, f"策略实际 ({p.name})", "t", np.linspace(0, act["t"].iloc[-1], 18))

        print("\n[策略实际统计]")
        for j in ["F_spine1", "F_body", "H_spine1", "H_body"]:
            v = act[j].to_numpy()
            print(f"  {j:9s} 范围 [{v.min():+.3f}, {v.max():+.3f}]  终值 {v[-1]:+.3f}")
        for j in ["upF", "upH", "axF_z", "axH_z"]:
            v = act[j].to_numpy()
            print(f"  {j:9s} 范围 [{v.min():+.3f}, {v.max():+.3f}]  终值 {v[-1]:+.3f}")
        # 是否真的到达过手调参考的 P1 末姿态 (F 正置 + H 倒置 + 躯干水平)
        tgt = np.array([0.0, -1.57, 0.0, 1.57])   # P1 回收段末的目标脊柱角
        Q = act[["F_spine1", "F_body", "H_spine1", "H_body"]].to_numpy()
        err = np.abs(Q - tgt).max(axis=1)
        k = int(np.argmin(err))
        print(f"\n  与 P1 回收段末目标 [0, -1.57, 0, +1.57] 最接近的一帧: "
              f"t={act['t'].iloc[k]:.2f}s  最大偏差 {err[k]:.3f} rad")
        print(f"    该帧实际: {Q[k].round(3).tolist()}")
        print(f"    P1 构建段末目标 [0.6, -1.57, 0.6, +1.57] 的最小最大偏差 = "
              f"{np.abs(Q - np.array([0.6, -1.57, 0.6, 1.57])).max(axis=1).min():.3f} rad")
        # 是否有任一帧满足 S1 的朝向判据 (背腹轴)
        s1_ori = (act["upF"] > 0.5) & (act["upH"] < -0.5)
        print(f"  S1 朝向判据 (背腹F>0.5 且 背腹H<-0.5) 命中帧数: {int(s1_ori.sum())} / {len(act)}")


if __name__ == "__main__":
    main()
