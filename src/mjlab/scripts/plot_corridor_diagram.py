"""绘制走廊一致性 (Corridor Conformance) 奖励示意图 (v2)

场景: 机器人沿圆弧路径行驶, 整体朝向相对路径切线有偏差 Δθ 且附带横向
偏移 d_lat; 脊柱弯曲 (F/H body 相对 base 偏转 ±spine_angle) 模拟绕杆姿态。
身体超出走廊带 (±C) 的部分用阴影标出。

用法:
  uv run python src/mjlab/scripts/plot_corridor_diagram.py [--radius 0.15]
      [--delta_theta 28] [--d_lat 0.008] [--spine_angle 12] [--out xxx.png]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ===== 真实尺寸参数 (与 path.py / curriculums.py 一致) =====
CORRIDOR_C = 0.04   # 走廊半宽 (CORRIDOR_HALF_WIDTH)
BODY_L = 0.04       # body 半长 (F/H_BODY_HALF_LENGTH)
BODY_W = 0.035      # body 半宽 (F/H_BODY_HALF_WIDTH)
OFFSET = 0.04       # F/H 中心距 base 偏移 (BODY_REF_OFFSET)

THETA_RANGE = (-1.5, 1.5)   # 路径弧角范围 (rad, 约 ±86°)
N_SAMPLE = 60               # 阴影采样密度 (每边)


def circle_through(p1, p2, p3):
    """三点定圆, 返回 (cx, cy, r); 近似共线时返回 None"""
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    A = np.array([[2.0 * (x2 - x1), 2.0 * (y2 - y1)],
                  [2.0 * (x3 - x1), 2.0 * (y3 - y1)]])
    b = np.array([x2 ** 2 + y2 ** 2 - x1 ** 2 - y1 ** 2,
                  x3 ** 2 + y3 ** 2 - x1 ** 2 - y1 ** 2])
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-12:
        return None
    cx, cy = np.linalg.solve(A, b)
    return cx, cy, np.hypot(x1 - cx, y1 - cy)


def arc_between(p1, p2, p3, n=60):
    """过三点画弧 (p2 在弧上), 返回采样点; 共线时返回直线"""
    circ = circle_through(p1, p2, p3)
    if circ is None:
        return np.array([p1, p3])
    cx, cy, r = circ
    a1 = np.arctan2(p1[1] - cy, p1[0] - cx)
    a2 = np.arctan2(p2[1] - cy, p2[0] - cx)
    a3 = np.arctan2(p3[1] - cy, p3[0] - cx)
    # 选包含 a2 的短弧方向
    d = a3 - a1
    if d > np.pi:
        d -= 2 * np.pi
    elif d < -np.pi:
        d += 2 * np.pi
    if not (min(a1, a3) - 1e-6 <= a2 <= max(a1, a3) + 1e-6):
        d = -d  # 反向
    thetas = np.linspace(a1, a1 + d, n)
    return np.column_stack([cx + r * np.cos(thetas), cy + r * np.sin(thetas)])


def main():
    parser = argparse.ArgumentParser(description="走廊一致性奖励示意图")
    parser.add_argument("--radius", type=float, default=0.15,
                        help="弧半径 (m, 示意尺寸; 真实 Rmin=0.05)")
    parser.add_argument("--delta_theta", type=float, default=28.0,
                        help="身体整体朝向与路径切线的偏差角度 (度)")
    parser.add_argument("--d_lat", type=float, default=0.008,
                        help="机器人横向偏移 (m, 沿路径法向)")
    parser.add_argument("--spine_angle", type=float, default=12.0,
                        help="脊柱弯曲角 (度): F/H body 相对 base 各偏转 ±角度")
    parser.add_argument("--out", type=str, default="corridor_diagram.png",
                        help="输出图片路径")
    args = parser.parse_args()

    R = args.radius
    delta_theta = np.deg2rad(args.delta_theta)
    spine_angle = np.deg2rad(args.spine_angle)
    d_lat = args.d_lat

    # ---- 路径与走廊带 ----
    theta = np.linspace(*THETA_RANGE, 400)
    path_x, path_y = R * np.cos(theta), R * np.sin(theta)
    inner_x, inner_y = (R - CORRIDOR_C) * np.cos(theta), (R - CORRIDOR_C) * np.sin(theta)
    outer_x, outer_y = (R + CORRIDOR_C) * np.cos(theta), (R + CORRIDOR_C) * np.sin(theta)

    # ---- 机器人 (base 在弧长中点, 带横向偏移) ----
    theta_b = 0.0
    base = np.array([R * np.cos(theta_b), R * np.sin(theta_b)])
    radial = np.array([np.cos(theta_b), np.sin(theta_b)])
    tangent = np.array([-np.sin(theta_b), np.cos(theta_b)])
    base = base + radial * d_lat

    beta = np.arctan2(tangent[1], tangent[0]) + delta_theta   # base 朝向角 (整体偏差)

    def u(angle):
        return np.array([np.cos(angle), np.sin(angle)])

    # 脊柱弯曲: F 偏 +spine, H 偏 -spine (相对 base 轴线)
    f_dir = u(beta + spine_angle)
    h_dir = u(beta - spine_angle)
    f_center = base + OFFSET * f_dir
    h_center = base - OFFSET * h_dir

    def rect_pts(c, half_l, half_w, angle):
        ax_v = u(angle)
        ay_v = np.array([-ax_v[1], ax_v[0]])
        return np.array([
            c + half_l * ax_v + half_w * ay_v,
            c + half_l * ax_v - half_w * ay_v,
            c - half_l * ax_v - half_w * ay_v,
            c - half_l * ax_v + half_w * ay_v,
        ])

    def in_corridor(px, py):
        r = np.hypot(px, py)
        return (r >= R - CORRIDOR_C) & (r <= R + CORRIDOR_C)

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    # 走廊带填充 (无边框说明)
    ax.fill(np.concatenate([outer_x, inner_x[::-1]]),
            np.concatenate([outer_y, inner_y[::-1]]),
            color="lightsteelblue", alpha=0.45)
    ax.plot(outer_x, outer_y, "--", color="steelblue", linewidth=1.0, alpha=0.7)
    ax.plot(inner_x, inner_y, "--", color="steelblue", linewidth=1.0, alpha=0.7)
    # 路径中心线
    ax.plot(path_x, path_y, "-", color="navy", linewidth=2.5)

    # 脊柱弧线 (H -> base -> F)
    spine_pts = arc_between(h_center, base, f_center)
    ax.plot(spine_pts[:, 0], spine_pts[:, 1], "-", color="black", linewidth=1.8, alpha=0.8)

    # 机器人 (F_body / H_body 矩形, 各自朝向含脊柱弯曲)
    ax.add_patch(Polygon(rect_pts(f_center, BODY_L, BODY_W, beta + spine_angle),
                         closed=True, facecolor="lightcoral", edgecolor="darkred",
                         linewidth=1.6, alpha=0.55))
    ax.add_patch(Polygon(rect_pts(h_center, BODY_L, BODY_W, beta - spine_angle),
                         closed=True, facecolor="lightcoral", edgecolor="darkred",
                         linewidth=1.6, alpha=0.55))
    ax.add_patch(Polygon(rect_pts(base, 0.008, 0.008, beta),
                         closed=True, facecolor="gray", edgecolor="black", linewidth=1.2))

    # 阴影: 身体超出走廊的采样点
    for c, ang in [(f_center, beta + spine_angle), (h_center, beta - spine_angle)]:
        ax_v = u(ang)
        ay_v = np.array([-ax_v[1], ax_v[0]])
        xs, ys = [], []
        for du in np.linspace(-BODY_L, BODY_L, N_SAMPLE):
            for dv in np.linspace(-BODY_W, BODY_W, N_SAMPLE):
                p = c + du * ax_v + dv * ay_v
                if not in_corridor(p[0], p[1]):
                    xs.append(p[0]); ys.append(p[1])
        if xs:
            ax.scatter(xs, ys, s=5, color="darkorange", alpha=0.85, zorder=5)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title("走廊一致性奖励示意 (含脊柱弯曲)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"已保存: {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
