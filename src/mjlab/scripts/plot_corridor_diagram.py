"""绘制走廊一致性 (Corridor Conformance) 奖励示意图

场景: 机器人沿圆弧路径 (绕杆弧段, 真实尺寸) 行驶, 身体朝向与路径切线存在
偏差 Δθ 且附带横向偏移 d_lat, 超出走廊带 (±C) 的身体部分用阴影标出。

用法:
  uv run python src/mjlab/scripts/plot_corridor_diagram.py [--delta_theta 28] [--d_lat 0.008] [--out xxx.png]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ===== 真实尺寸参数 (与 path.py / curriculums.py 一致) =====
R = 0.05            # 弧半径 Rmin = 1/CURVATURE_TARGET (CURVATURE_TARGET=20)
CORRIDOR_C = 0.04   # 走廊半宽 (CORRIDOR_HALF_WIDTH)
BODY_L = 0.04       # body 半长 (F/H_BODY_HALF_LENGTH)
BODY_W = 0.035      # body 半宽 (F/H_BODY_HALF_WIDTH)
OFFSET = 0.04       # F/H 中心距 base 偏移 (BODY_REF_OFFSET)

THETA_RANGE = (-0.9, 0.9)   # 路径弧角范围 (rad)
N_SAMPLE = 60               # 阴影采样密度 (每边)


def main():
    parser = argparse.ArgumentParser(description="走廊一致性奖励示意图")
    parser.add_argument("--delta_theta", type=float, default=28.0,
                        help="身体朝向与路径切线的偏差角度 (度)")
    parser.add_argument("--d_lat", type=float, default=0.008,
                        help="机器人横向偏移 (m, 沿路径法向)")
    parser.add_argument("--out", type=str, default="corridor_diagram.png",
                        help="输出图片路径")
    args = parser.parse_args()

    delta_theta = np.deg2rad(args.delta_theta)
    d_lat = args.d_lat

    # ---- 路径与走廊带 ----
    theta = np.linspace(*THETA_RANGE, 300)
    path_x, path_y = R * np.cos(theta), R * np.sin(theta)
    inner_x, inner_y = (R - CORRIDOR_C) * np.cos(theta), (R - CORRIDOR_C) * np.sin(theta)
    outer_x, outer_y = (R + CORRIDOR_C) * np.cos(theta), (R + CORRIDOR_C) * np.sin(theta)

    # ---- 机器人 (base 在路径弧长中点, 可加横向偏移) ----
    theta_b = 0.0
    base = np.array([R * np.cos(theta_b), R * np.sin(theta_b)])
    radial = np.array([np.cos(theta_b), np.sin(theta_b)])          # 路径法向 (径向)
    tangent = np.array([-np.sin(theta_b), np.cos(theta_b)])        # 路径切线方向
    base = base + radial * d_lat

    # 身体轴 = 切线旋转 Δθ (朝向偏差)
    a = np.array([tangent[0] * np.cos(delta_theta) - tangent[1] * np.sin(delta_theta),
                  tangent[0] * np.sin(delta_theta) + tangent[1] * np.cos(delta_theta)])
    b = np.array([-a[1], a[0]])                                    # 身体侧向

    f_center = base + OFFSET * a
    h_center = base - OFFSET * a

    def rect_pts(c, half_l, half_w):
        return np.array([
            c + half_l * a + half_w * b,
            c + half_l * a - half_w * b,
            c - half_l * a - half_w * b,
            c - half_l * a + half_w * b,
        ])

    def in_corridor(px, py):
        r = np.hypot(px, py)
        return (r >= R - CORRIDOR_C) & (r <= R + CORRIDOR_C)

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(9.5, 9.5))

    # 走廊带填充
    ax.fill(np.concatenate([outer_x, inner_x[::-1]]),
            np.concatenate([outer_y, inner_y[::-1]]),
            color="lightsteelblue", alpha=0.45, label="走廊带 (±C=±0.04m)")
    ax.plot(outer_x, outer_y, "--", color="steelblue", linewidth=1.0, alpha=0.8)
    ax.plot(inner_x, inner_y, "--", color="steelblue", linewidth=1.0, alpha=0.8)
    # 路径中心线
    ax.plot(path_x, path_y, "-", color="navy", linewidth=2.5,
            label=f"参考路径 (κ={1/R:.0f}, R={R:.2f}m)")

    # 机器人 (F_body / H_body / base)
    for name, c in [("F_body", f_center), ("H_body", h_center)]:
        ax.add_patch(Polygon(rect_pts(c, BODY_L, BODY_W), closed=True,
                             facecolor="lightcoral", edgecolor="darkred",
                             linewidth=1.6, alpha=0.5, label=name))
    ax.add_patch(Polygon(rect_pts(base, 0.008, 0.008), closed=True,
                         facecolor="gray", edgecolor="black", linewidth=1.2, label="base"))
    # 脊柱连线 (H → base → F)
    ax.plot([h_center[0], base[0], f_center[0]],
            [h_center[1], base[1], f_center[1]], "k-", linewidth=1.6, alpha=0.8)

    # 阴影: 身体超出走廊的采样点
    n_out = 0
    for c in (f_center, h_center):
        xs, ys = [], []
        for u in np.linspace(-BODY_L, BODY_L, N_SAMPLE):
            for v in np.linspace(-BODY_W, BODY_W, N_SAMPLE):
                p = c + u * a + v * b
                if not in_corridor(p[0], p[1]):
                    xs.append(p[0]); ys.append(p[1])
        n_out += len(xs)
        if xs:
            ax.scatter(xs, ys, s=5, color="darkorange", alpha=0.85, zorder=5,
                       label="超出走廊 (阴影)" if c is f_center else None)

    # ---- 标注 ----
    # 走廊宽度
    ax.annotate("", xy=(R + CORRIDOR_C, 0), xytext=(R - CORRIDOR_C, 0),
                arrowprops=dict(arrowstyle="<->", color="steelblue", lw=1.2))
    ax.text(R, -CORRIDOR_C - 0.012, "走廊宽度 2C = 0.08m", ha="center",
            fontsize=9, color="steelblue")
    # Δθ 角弧
    arc_t = np.linspace(0, delta_theta, 30)
    ax.plot(0.018 * np.cos(arc_t), 0.018 * np.sin(arc_t), "-", color="purple", linewidth=1.5)
    ax.text(0.022, 0.010, f"Δθ={np.rad2deg(delta_theta):.0f}°", fontsize=10, color="purple")
    # d_lat
    ax.annotate("", xy=base, xytext=(R, 0),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.2))
    ax.text(R - 0.008, -0.045, f"d_lat={d_lat:.3f}m", fontsize=9, color="green")

    # 奖励公式
    e_now = abs(d_lat) + BODY_L * abs(np.sin(delta_theta)) + BODY_W * abs(np.cos(delta_theta))
    formula = (f"走廊超额: e = |d_lat| + |L·sinΔθ| + |W·cosΔθ|\n"
               f"惩罚量:   v = max(0, e - C),  C = {CORRIDOR_C}m\n"
               f"奖励:     r = exp(-σ·v^2)\n\n"
               f"当前: e = {e_now:.3f} m"
               f"  -> 超出 {max(0.0, e_now - CORRIDOR_C):.3f} m"
               f" ({n_out} 采样点)")
    ax.text(0.02, 0.97, formula, transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFF8DC", alpha=0.95))

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title(f"走廊一致性奖励示意 — 绕杆弧段 (真实尺寸, Δθ={np.rad2deg(delta_theta):.0f}°, d_lat={d_lat:.3f}m)",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"已保存: {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
