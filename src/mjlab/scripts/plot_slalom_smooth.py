"""Phase1 绕杆 LUT 曲率平滑对比可视化

问题: LUT 中 CW 弧 (κ=+20) 与 CCW 弧 (κ=-20) 直接相切拼接, 曲率瞬时跳变。
方案: 对 κ(s) 做峰值缩放 + 环形滑动平均, 再数值积分重建路径, 使切换平滑,
      等效增大转弯半径 (可相应增大杆的 Y 坐标)。

本脚本先绘制现有曲线, 再绘制平滑后曲线, 并读取实际转弯半径增大了多少。

用法:
  uv run python src/mjlab/scripts/plot_slalom_smooth.py [--no-show] [--out xxx.png]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from mjlab.tasks.SQuRo_Slalom.mdp.path import (
    _RMIN,
    _INIT_DIST,
    _generate_slalom_lut_one_period,
)
from mjlab.tasks.SQuRo_Slalom.mdp.pole import POLE_Y, POLE_RADIUS
from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import CURVATURE_TARGET


# ===== 文件头常量 =====
GAIT_FREQ = 1.0             # Phase 1 使用的步频 (Hz)
POLE_SPACING = 0.15         # 杆间距 (m)
KAPPA_SMOOTH = 15.0         # 平滑后的目标峰值曲率 (rad/m, 原始 20)
SMOOTH_WINDOW = 7           # 曲率环形滑动平均窗口 (点数, 奇数)
EPISODE_LEN = 20.0          # episode 时长 (s)


def moving_average_cyclic(x, w):
    """环形滑动平均 (w 为奇数), 保持周期连续"""
    pad = w // 2
    xp = np.concatenate([x[-pad:], x, x[:pad]])
    kernel = np.ones(w) / w
    return np.convolve(xp, kernel, mode="valid")


def integrate_from_kappa(kappa_s, s_grid, x0=0.0, y0=0.0, h0=0.0):
    """由曲率曲线 κ(s) 数值积分重建路径: h=∫κds, x=∫cos h ds, y=∫sin h ds"""
    ds = s_grid[1] - s_grid[0]
    heading = h0 + np.cumsum(kappa_s) * ds
    x = x0 + np.cumsum(np.cos(heading)) * ds
    y = y0 + np.cumsum(np.sin(heading)) * ds
    return x, y, heading


def main():
    parser = argparse.ArgumentParser(description="Phase1 LUT 曲率平滑对比")
    parser.add_argument("--out", type=str, default="slalom_smooth_diagram.png")
    parser.add_argument("--no-show", action="store_true", help="不弹窗展示, 直接保存")
    args = parser.parse_args()

    r_orig = _RMIN
    # ===== 1. 现有 LUT (不平滑) =====
    arc, xs, ys, headings, kappa = _generate_slalom_lut_one_period(POLE_SPACING)
    arc = np.asarray(arc)
    kappa = np.asarray(kappa)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    s_period = float(arc[-1])
    rmin_orig = 1.0 / CURVATURE_TARGET

    # ===== 2. 曲率平滑: 峰值缩放 + 环形滑动平均 =====
    kappa_scaled = kappa * (KAPPA_SMOOTH / CURVATURE_TARGET)
    kappa_smooth = moving_average_cyclic(kappa_scaled, SMOOTH_WINDOW)
    # 平滑后实际峰值曲率 → 实际最小转弯半径
    kappa_peak_actual = np.max(np.abs(kappa_smooth))
    rmin_actual = 1.0 / kappa_peak_actual

    # ===== 3. 均匀弧长网格上插值 + 积分重建平滑路径 =====
    M = 4000
    s_grid = np.linspace(0.0, s_period, M)
    kappa_grid = np.interp(s_grid, arc, kappa_smooth)
    x_new, y_new, _ = integrate_from_kappa(kappa_grid, s_grid)

    # ===== 测量输出 =====
    delta_r = rmin_actual - rmin_orig
    print("=" * 56)
    print(f"杆间距 spacing = {POLE_SPACING}m, 周期弧长 = {s_period:.4f}m")
    print(f"原始: 峰值 κ = ±{CURVATURE_TARGET:.0f}, Rmin = {rmin_orig:.4f} m, POLE_Y = {POLE_Y:.4f}")
    print(f"平滑: 目标 κ = ±{KAPPA_SMOOTH:.0f} (窗口{SMOOTH_WINDOW}点)")
    print(f"      实际峰值 κ = ±{kappa_peak_actual:.3f} → Rmin = {rmin_actual:.4f} m")
    print(f"转弯半径增大: ΔR = {delta_r:.4f} m (+{delta_r/rmin_orig*100:.1f}%)")
    print(f"建议 POLE_Y: {POLE_Y:.4f} → {-rmin_actual:.4f} (Y 移动 {-delta_r:.4f} m)")
    print(f"平滑后周期 X 跨度: {2*POLE_SPACING:.4f} → {x_new[-1]:.4f} m")
    print("=" * 56)

    # ===== 4. 绘图 =====
    fig, axes = plt.subplots(2, 2, figsize=(13, 9),
                             gridspec_kw={"height_ratios": [1.7, 1.0], "hspace": 0.4})
    ax_path, ax_k, ax_zoom, ax_r = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # 主图: 原始 vs 平滑路径 + 杆
    ax_path.plot(xs, ys, "-", color="gray", linewidth=2.0, alpha=0.6,
                 label=f"现有 LUT (κ=±{CURVATURE_TARGET:.0f}, R={rmin_orig:.3f}m)")
    ax_path.plot(x_new, y_new, "-", color="crimson", linewidth=2.2,
                 label=f"平滑后 (实际 Rmin={rmin_actual:.3f}m)")
    for i in range(4):
        px, py = i * POLE_SPACING, POLE_Y
        ax_path.add_patch(Circle((px, py), radius=POLE_RADIUS, facecolor="gray",
                                 edgecolor="black", zorder=5))
    ax_path.scatter([0.0], [0.0], marker="o", color="green", s=60, zorder=6, label="起点")
    ax_path.set_aspect("equal")
    ax_path.grid(True, alpha=0.3)
    ax_path.legend(fontsize=9, loc="upper right")
    ax_path.set_xlabel("X (m)")
    ax_path.set_ylabel("Y (m)")
    ax_path.set_title(f"路径对比 (spacing={POLE_SPACING}m, 杆位置不变)")

    # κ(s) 对比
    ax_k.plot(arc, kappa, "-", color="gray", linewidth=1.6, label="现有 κ(s)")
    ax_k.plot(arc, kappa_smooth, "-", color="crimson", linewidth=1.8,
              label=f"平滑 κ(s) (峰值 ±{kappa_peak_actual:.1f})")
    ax_k.axhline(0, color="green", linestyle="--", linewidth=0.8)
    ax_k.set_xlabel("弧长 s (m)")
    ax_k.set_ylabel("κ (rad/m)")
    ax_k.set_title("曲率沿弧长对比")
    ax_k.legend(fontsize=8)
    ax_k.grid(True, alpha=0.3)
    ax_k.set_ylim(-CURVATURE_TARGET * 1.25, CURVATURE_TARGET * 1.25)

    # 局部放大: 第一个 CW→CCW 切换处
    zoom_x0, zoom_x1 = 0.0, 2 * _RMIN + 0.02
    zoom_y0, zoom_y1 = -2 * _RMIN - 0.02, 0.01
    ax_zoom.plot(xs, ys, "-", color="gray", linewidth=2.0, alpha=0.7)
    ax_zoom.plot(x_new, y_new, "-", color="crimson", linewidth=2.2)
    ax_zoom.set_xlim(zoom_x0, zoom_x1)
    ax_zoom.set_ylim(zoom_y0, zoom_y1)
    ax_zoom.set_aspect("equal")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.set_title("CW→CCW 切换处放大 (曲率跳变 vs 平滑)")

    # 转弯半径 R(s) = 1/|κ(s)|
    with np.errstate(divide="ignore", invalid="ignore"):
        r_orig_s = np.where(np.abs(kappa) > 1e-6, 1.0 / np.abs(kappa), np.nan)
        r_smooth_s = np.where(np.abs(kappa_smooth) > 1e-6, 1.0 / np.abs(kappa_smooth), np.nan)
    ax_r.plot(arc, r_orig_s, "-", color="gray", linewidth=1.6, label="现有 R(s)")
    ax_r.plot(arc, r_smooth_s, "-", color="crimson", linewidth=1.8, label="平滑后 R(s)")
    ax_r.axhline(rmin_orig, color="gray", linestyle=":", linewidth=0.8)
    ax_r.axhline(rmin_actual, color="crimson", linestyle=":", linewidth=0.8)
    ax_r.annotate(f"Rmin: {rmin_orig:.3f} → {rmin_actual:.3f} m (+{delta_r*1000:.1f}mm)",
                  xy=(0.02, max(rmin_orig, rmin_actual) * 1.15), fontsize=9)
    ax_r.set_xlabel("弧长 s (m)")
    ax_r.set_ylabel("R (m)")
    ax_r.set_title("局部转弯半径对比")
    ax_r.legend(fontsize=8)
    ax_r.grid(True, alpha=0.3)

    fig.suptitle(f"Phase1 LUT 曲率平滑: κ±{CURVATURE_TARGET:.0f} → 目标±{KAPPA_SMOOTH:.0f}, "
                 f"实际 Rmin {rmin_orig:.3f}→{rmin_actual:.3f} m", fontsize=13, fontweight="bold")

    if not args.no_show:
        plt.show()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"已保存: {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
