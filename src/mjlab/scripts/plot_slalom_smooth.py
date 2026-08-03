"""Phase1 绕杆弧段曲率平滑设计 (按平滑时间 t 参数化)

场景 (第一段 CW 弧 κ=-20 → 切换 CCW 弧 κ=+20):
  - 初始从 (0, y0) 出发, 以 κ=-20 行驶弧长 s1 = L90 - v·t/2
  - 然后 κ 线性增大到 0 (平滑段, 弧长 v·t), 此时路径刚好到 (x, -Rmin)
  - 之后切换为 κ=+20 (下一段弧)

验证性质:
  - t=0: 起点 y0=0, 切换点 x=Rmin (即原始 90° 弧)
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


# ===== 文件头常量 =====
GAIT_FREQ = 1.0             # Phase 1 使用的步频 (Hz)
SMOOTH_TIME = 1.0           # 平滑时间 t (s): 曲率从 -20 线性过渡到 0 的时长
RMIN = 1.0 / CURVATURE_TARGET   # 最小转弯半径 (= 0.05m, κ=20)


def integrate_from_kappa(kappa_s, s_grid, h0, x0, y0):
    """由曲率曲线 κ(s) 数值积分重建路径: h=∫κds, x=∫cos h ds, y=∫sin h ds"""
    ds = s_grid[1] - s_grid[0]
    heading = h0 + np.cumsum(kappa_s) * ds
    x = x0 + np.cumsum(np.cos(heading)) * ds
    y = y0 + np.cumsum(np.sin(heading)) * ds
    return x, y, heading


def main():
    parser = argparse.ArgumentParser(description="Phase1 弧段曲率平滑 (时间参数化)")
    parser.add_argument("--out", type=str, default="slalom_smooth_diagram.png")
    parser.add_argument("--no-show", action="store_true", help="不弹窗展示, 直接保存")
    args = parser.parse_args()

    t = SMOOTH_TIME
    v = FIXED_VEL * GAIT_FREQ * (1.0 - (1.0 - VEL_MIN) * CURVATURE_TARGET / CURVATURE_TARGET_MAX)
    L90 = np.pi / 2 * RMIN              # 90° 弧长
    l_smooth = v * t                    # 平滑段弧长
    s1 = L90 - l_smooth / 2             # 前段 (κ=-20) 弧长

    # ---- κ(s) 设计: -20 (弧长 s1) → 线性 → 0 (平滑段) → +20 (后段) ----
    # 按弧长节点构造分段线性 κ, 再由 s_grid 插值 (保证各段弧长精确)
    M_TOTAL = 3600
    s_grid = np.linspace(0.0, s1 + l_smooth + L90, M_TOTAL)
    s_nodes = np.array([0.0, s1, s1 + l_smooth, s1 + l_smooth + L90])
    k_nodes = np.array([-CURVATURE_TARGET, -CURVATURE_TARGET, 0.0, CURVATURE_TARGET])
    kappa = np.interp(s_grid, s_nodes, k_nodes)

    # ---- 积分 (起点朝 +X, 从 (0,0) 出发), 再平移 y 使平滑段终点 (κ=0) 落在 y=-Rmin ----
    x_int, y_int, h_int = integrate_from_kappa(kappa, s_grid, 0.0, 0.0, 0.0)
    kappa_zero_idx = np.searchsorted(s_grid, s1 + l_smooth) - 1   # κ 到达 0 的位置
    x_switch, y_switch = x_int[kappa_zero_idx], y_int[kappa_zero_idx]
    y_off = -RMIN - y_switch            # y 平移量
    y0 = y_off                          # 平移后起点 y 坐标
    x = x_switch                        # 切换点 x 坐标 (x 不平移)

    # 平移后路径
    y_path = y_int + y_off

    # ---- 原始 LUT 弧 (对比) ----
    th = np.linspace(np.pi / 2, 0.0, 200)          # 绕圆心 (0,-Rmin), CW
    x_orig = RMIN * np.cos(th)
    y_orig = -RMIN + RMIN * np.sin(th)

    # ---- 测量输出 ----
    delta_y = y0 - 0.0
    delta_x = x - RMIN
    print("=" * 60)
    print(f"平滑时间 t = {t:.2f}s, 速度 v = {v:.4f} m/s, 平滑段弧长 = {l_smooth*1000:.1f}mm")
    print(f"前段 (κ=-{CURVATURE_TARGET:.0f}) 弧长 s1 = {s1*1000:.1f}mm")
    print(f"原始: 起点 (0, 0), 切换点 ({RMIN:.4f}, {-RMIN:.4f})")
    print(f"平滑: 起点 (0, {y0:+.4f}), 切换点 ({x:.4f}, {-RMIN:.4f})")
    print(f"变化: Δy0 = {delta_y:+.4f} m, Δx = {delta_x:+.4f} m")
    print("=" * 60)

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5),
                             gridspec_kw={"width_ratios": [1.4, 1.0], "wspace": 0.28})
    ax_path, ax_k = axes

    # 主图: 路径
    ax_path.plot(x_orig, y_orig, "--", color="gray", linewidth=2.0,
                 label=f"原始弧 (t=0, 起点(0,0), 切换点({RMIN:.3f},{-RMIN:.3f}))")
    ax_path.plot(x_path := np.concatenate([[0.0], x_int]),
                 np.concatenate([[y0], y_path]),
                 "-", color="crimson", linewidth=2.4,
                 label=f"平滑路径 (t={t:.1f}s)")
    # 杆
    ax_path.add_patch(Circle((0.0, POLE_Y), radius=POLE_RADIUS, facecolor="gray",
                             edgecolor="black", zorder=5))
    ax_path.annotate("pole0", xy=(0.0, POLE_Y - 0.012), ha="center", fontsize=8)
    # 起点/切换点标注
    ax_path.scatter([0.0], [y0], marker="o", color="green", s=80, zorder=6)
    ax_path.annotate(f"起点 (0, {y0:+.3f})", xy=(0.0, y0), xytext=(0.002, y0 + 0.012),
                     fontsize=9, color="green")
    ax_path.scatter([x_switch], [-RMIN], marker="s", color="red", s=80, zorder=6)
    ax_path.annotate(f"切换点 ({x:.3f}, {-RMIN:.3f})\nκ: -20→0→+20",
                     xy=(x_switch, -RMIN), xytext=(x_switch + 0.008, -RMIN - 0.022),
                     fontsize=9, color="red")
    ax_path.axhline(-RMIN, color="gray", linestyle=":", linewidth=0.8)
    ax_path.set_aspect("equal")
    ax_path.grid(True, alpha=0.3)
    ax_path.legend(fontsize=8, loc="upper left")
    ax_path.set_xlabel("X (m)")
    ax_path.set_ylabel("Y (m)")
    ax_path.set_title(f"平滑设计: κ -{CURVATURE_TARGET:.0f} → 0 → +{CURVATURE_TARGET:.0f} "
                      f"(t={t:.1f}s)")

    # 子图: κ(s)
    ax_k.plot(s_grid, kappa, "-", color="crimson", linewidth=1.8)
    ax_k.axhline(0, color="green", linestyle="--", linewidth=0.8)
    ax_k.axvline(s1, color="orange", linestyle=":", linewidth=1.0)
    ax_k.axvline(s1 + l_smooth, color="red", linestyle=":", linewidth=1.0)
    ax_k.annotate("s1", xy=(s1, CURVATURE_TARGET * 0.8), fontsize=8, color="orange")
    ax_k.annotate("平滑段", xy=(s1 + l_smooth / 2, CURVATURE_TARGET * 0.8),
                  ha="center", fontsize=8, color="red")
    ax_k.set_xlabel("弧长 s (m)")
    ax_k.set_ylabel("κ (rad/m)")
    ax_k.set_title("κ(s): -20 → 线性 → 0 → 切换 +20")
    ax_k.grid(True, alpha=0.3)
    ax_k.set_ylim(-CURVATURE_TARGET * 1.25, CURVATURE_TARGET * 1.25)

    fig.suptitle(f"弧段曲率平滑 (s1={s1*1000:.1f}mm, 平滑段={l_smooth*1000:.1f}mm, "
                 f"Δy0={delta_y:+.1f}mm, Δx={delta_x*1000:+.1f}mm)",
                 fontsize=12, fontweight="bold")

    if not args.no_show:
        plt.show()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"已保存: {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
