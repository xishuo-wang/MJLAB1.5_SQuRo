"""Phase1 绕杆 LUT 曲率对称平滑 (多周期) 可视化

设计: 每个曲率跳变处插入线性过渡段 (时长 t, 弧长 v·t), 包括:
  - S1: -20 平台 (L90 - v·t/2, 第一周期起点直接 -20) → 过渡(-20→0, t)
  - S2: 过渡(0→+20, t) → +20 平台 (L90 - v·t) → 过渡(+20→0, t)
  - 直行段 κ=0 保持, 后续所有弧段对称处理 (进入/退出过渡各 t)

验证性质 (第一切换点, κ 第一次到 0 处):
  - t=0: 起点 y0=0, 切换点 x=Rmin (原始 90° 弧)
  - t>0: 起点 y0>0, 切换点 x>Rmin (等效转弯半径增大)

用法:
  uv run python src/mjlab/scripts/plot_slalom_smooth.py [--no-show] [--out xxx.png]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import CURVATURE_TARGET, CURVATURE_TARGET_MAX
from mjlab.tasks.SQuRo_Slalom.mdp.command import FIXED_VEL, VEL_MIN
from mjlab.tasks.SQuRo_Slalom.mdp.pole import POLE_Y, POLE_RADIUS
from mjlab.tasks.SQuRo_Slalom.mdp.path import _generate_slalom_lut_one_period


# ===== 文件头常量 =====
GAIT_FREQ = 1.0             # Phase 1 使用的步频 (Hz)
SMOOTH_TIME = 1.0           # 每段曲率过渡的时间 t (s): -20→0 与 0→+20 各 t
POLE_SPACING = 0.15         # 杆间距 (m)
N_PERIODS = 3               # 展示周期数
RMIN = 1.0 / CURVATURE_TARGET   # 最小转弯半径 (= 0.05m, κ=20)


def main():
    parser = argparse.ArgumentParser(description="Phase1 LUT 曲率对称平滑 (多周期)")
    parser.add_argument("--out", type=str, default="slalom_smooth_diagram.png")
    parser.add_argument("--no-show", action="store_true", help="不弹窗展示, 直接保存")
    args = parser.parse_args()

    t = SMOOTH_TIME
    v = FIXED_VEL * GAIT_FREQ * (1.0 - (1.0 - VEL_MIN) * CURVATURE_TARGET / CURVATURE_TARGET_MAX)
    tr = v * t                          # 单段过渡弧长
    L90 = np.pi / 2 * RMIN              # 90° 弧长
    straight = max(0.0, POLE_SPACING - 2 * RMIN)   # 直行段长度
    platform_std = L90 - tr             # 标准弧段平台 (含进出过渡)
    platform_s1 = L90 - tr / 2          # 第一周期 S1 平台 (无进入过渡)

    # ---- 构造 κ(s) 分段 (所有跳变线性过渡) ----
    K = CURVATURE_TARGET
    segs = []   # (弧长, κ起点, κ终点)
    for p in range(N_PERIODS):
        if p == 0:
            segs.append((platform_s1, -K, -K))   # S1 平台
        else:
            segs.append((tr, 0.0, -K))           # 直行→S1 过渡
            segs.append((platform_std, -K, -K))  # S1' 平台
        segs.append((tr, -K, 0.0))               # S1→S2 过渡1
        segs.append((tr, 0.0, K))                # S1→S2 过渡2
        segs.append((platform_std, K, K))        # S2 平台
        segs.append((tr, K, 0.0))                # S2→S3 过渡
        segs.append((straight, 0.0, 0.0))        # S3 直行
        segs.append((tr, 0.0, K))                # S3→S4 过渡
        segs.append((platform_std, K, K))        # S4 平台
        segs.append((tr, K, 0.0))                # S4→S5 过渡1
        segs.append((tr, 0.0, -K))               # S4→S5 过渡2
        segs.append((platform_std, -K, -K))      # S5 平台
        segs.append((tr, -K, 0.0))               # S5→S6 过渡
        segs.append((straight, 0.0, 0.0))        # S6 直行

    # 逐段生成 (s, κ) 密集网格
    ds = 1e-4
    s_pts, k_pts = [], []
    s = 0.0
    for L, k0, k1 in segs:
        n = max(2, int(L / ds))
        s_pts.extend(np.linspace(s, s + L, n, endpoint=False))
        k_pts.extend(np.linspace(k0, k1, n, endpoint=False))
        s += L
    s_pts.append(s)
    k_pts.append(k_pts[-1])
    s_grid = np.array(s_pts)
    kappa = np.array(k_pts)

    # ---- 积分重建路径 (起点朝 +X, 从 (0,0) 出发) ----
    h = np.cumsum(kappa) * ds
    x_int = np.cumsum(np.cos(h)) * ds
    y_int = np.cumsum(np.sin(h)) * ds

    # 第一切换点 (κ 第一次到达 0): 第一个 tr 过渡结束
    i_sw = np.searchsorted(s_grid, platform_s1 + tr) - 1
    x_sw, y_sw = x_int[i_sw], y_int[i_sw]
    y_off = -RMIN - y_sw
    y0 = y_off
    y_path = y_int + y_off

    # ---- 原始 LUT (对比, 一个周期) ----
    arc_o, xs_o, ys_o, hd_o, kp_o = _generate_slalom_lut_one_period(POLE_SPACING)

    # ---- 测量输出 ----
    delta_y = y0
    delta_x = x_sw - RMIN
    print("=" * 60)
    print(f"平滑时间 t = {t:.2f}s, v = {v:.4f} m/s, 单段过渡弧长 = {tr*1000:.1f}mm")
    print(f"第一切换点 (κ 首次到 0):")
    print(f"  原始: (0, 0) → 切换点 ({RMIN:.4f}, {-RMIN:.4f})")
    print(f"  平滑: (0, {y0:+.4f}) → 切换点 ({x_sw:.4f}, {-RMIN:.4f})")
    print(f"  Δy0 = {delta_y:+.4f} m, Δx = {delta_x*1000:+.2f} mm")
    print(f"周期弧长: 原始 {arc_o[-1]:.4f}m, 平滑 {s_grid[-1]/N_PERIODS:.4f}m")
    print("=" * 60)

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                             gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.25})
    ax_path, ax_k = axes

    # 主图: 平滑路径 (多周期) + 原始 LUT + 杆
    ax_path.plot(np.concatenate([[0.0], x_int]), np.concatenate([[y0], y_path]),
                 "-", color="crimson", linewidth=2.2,
                 label=f"平滑路径 ({N_PERIODS} 周期, t={t:.1f}s)")
    for k in range(N_PERIODS):
        off = k * 2 * POLE_SPACING
        ax_path.plot(np.asarray(xs_o) + off, ys_o, "--", color="gray",
                     linewidth=1.2, alpha=0.7)
    # 杆
    for i in range(N_PERIODS * 4):
        px, py = i * POLE_SPACING, POLE_Y
        if px > N_PERIODS * 2 * POLE_SPACING:
            break
        ax_path.add_patch(Circle((px, py), radius=POLE_RADIUS, facecolor="gray",
                                 edgecolor="black", zorder=5))
    # 第一切换点标注
    ax_path.scatter([0.0], [y0], marker="o", color="green", s=70, zorder=6)
    ax_path.annotate(f"起点 (0, {y0:+.3f})", xy=(0.0, y0), xytext=(0.004, y0 + 0.012),
                     fontsize=9, color="green")
    ax_path.scatter([x_sw], [-RMIN], marker="s", color="red", s=70, zorder=6)
    ax_path.annotate(f"切换点 ({x_sw:.3f}, {-RMIN:.3f})", xy=(x_sw, -RMIN),
                     xytext=(x_sw + 0.01, -RMIN - 0.024), fontsize=9, color="red")
    ax_path.set_aspect("equal")
    ax_path.grid(True, alpha=0.3)
    ax_path.legend(fontsize=8, loc="upper left")
    ax_path.set_xlabel("X (m)")
    ax_path.set_ylabel("Y (m)")
    ax_path.set_title(f"对称平滑路径 (所有跳变线性过渡 {t:.1f}s, {N_PERIODS} 周期)")

    # 子图: κ(s) 对比 (2 周期)
    n2 = int(len(s_grid) * 2 / N_PERIODS)
    ax_k.plot(s_grid[:n2], kappa[:n2], "-", color="crimson", linewidth=1.5)
    ax_k.axhline(0, color="green", linestyle="--", linewidth=0.8)
    ax_k.set_xlabel("弧长 s (m)")
    ax_k.set_ylabel("κ (rad/m)")
    ax_k.set_title(f"κ(s): 所有跳变对称线性过渡 (±{K:.0f}, t={t:.1f}s)")
    ax_k.grid(True, alpha=0.3)
    ax_k.set_ylim(-K * 1.25, K * 1.25)

    fig.suptitle(f"曲率对称平滑 (Δy0={delta_y*1000:+.1f}mm, Δx={delta_x*1000:+.2f}mm, "
                 f"每周期弧长 {s_grid[-1]/N_PERIODS*1000:.0f}mm)",
                 fontsize=12, fontweight="bold")

    if not args.no_show:
        plt.show()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"已保存: {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
