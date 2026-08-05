"""绘制走廊一致性奖励示意图 (绕杆 slalom 路径 v5)
变更 (v5):
  - 杆按 (x,y),(2x,y),(3x,y) 摆放 (第一根杆在 x=POLE_SPACING 处)
  - 有直行模式: 直行段合并为一段 (两杆之间), 中点对齐第二根杆的 x 坐标
  - 轨迹整体右移, 起点 = 第一根杆正上方
用法:
  python plot_corridor_slalom.py [--delta_theta 28] [--d_lat 0.008] [--spine_angle 12] [--out test.png]
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon, Circle

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 用户可配置项 ====================
OUTPUT_DIR = r"D:\MuJoCoLab_1.5\src\mjlab\scripts\Viz_Path\Result"  # 输出目录，自动创建

# ----- 身体尺寸 -----
CORRIDOR_C = 0.06               # 走廊半宽
BODY_L = 0.10                   # 身体半长
BODY_W = 0.055                  # 身体半宽
OFFSET = BODY_L                 # F/H 中心距 base 偏移

# ----- 期望轨迹参数 -----
POLE_SPACING = 0.2              # 杆间距 (m)
RMIN = 0.06                     # 最小转弯半径
SMOOTH_VEL = 0.3                # 平滑过渡速度 (m/s)
SMOOTH_TIME = 0.1               # 平滑过渡时间 (s)
CURVATURE_TARGET = 1.0 / RMIN   # 目标曲率 (1/m)，由最小半径自动计算

# ----- 杆显示参数 -----
POLE_RADIUS = 0.01              # 杆半径 (m)
POLE_COLOR = (0.9, 0.35, 0.2)   # 橙红色 RGB
POLE_ZORDER = 8                 # 绘图层级（高于机器人）

# ----- 绘图杂项 -----
N_SAMPLE = 60                   # 身体阴影采样密度 (每边)
# ======================================================


# ========== 平滑绕杆 LUT 生成 ==========
def _measure_s1_xsw(segs, ds=1e-4) -> float:
    s_pts, k_pts, s = [], [], 0.0
    for L, k0, k1 in segs:
        n = max(2, int(L / ds))
        s_pts.extend(np.linspace(s, s + L, n, endpoint=False))
        k_pts.extend(np.linspace(k0, k1, n, endpoint=False))
        s += L
    s_pts.append(s)
    k_pts.append(k_pts[-1])
    k_g = np.array(k_pts)
    h = np.cumsum(k_g) * ds
    return float(np.sum(np.cos(h)) * ds)


def _get_smooth_xsw(tr: float) -> tuple[float, float]:
    K = CURVATURE_TARGET
    L90 = np.pi / 2 * RMIN
    platform = L90 - tr
    platform_half = L90 - tr / 2
    x_sw_half = _measure_s1_xsw([(platform_half, -K, -K), (tr, -K, 0.0)])
    x_sw_full = _measure_s1_xsw([(tr, 0.0, -K), (platform, -K, -K), (tr, -K, 0.0)])
    return x_sw_half, x_sw_full


def generate_slalom_lut_smooth_period(X: float, smooth_time=None, vel=None):
    """生成一个周期平滑绕杆路径的弧长、坐标、航向、曲率 LUT"""
    t = SMOOTH_TIME if smooth_time is None else smooth_time
    v = SMOOTH_VEL if vel is None else vel
    K = CURVATURE_TARGET
    tr = v * t
    L90 = np.pi / 2 * RMIN
    platform = L90 - tr
    platform_half = L90 - tr / 2

    x_sw_half, x_sw_full = _get_smooth_xsw(tr)

    if X >= 2 * x_sw_full - 1e-6:
        straight = max(0.0, X - 2 * x_sw_full)
        segs = [
            (tr, 0.0, -K), (platform, -K, -K), (tr, -K, 0.0),
            (tr, 0.0, K), (platform, K, K), (tr, K, 0.0),
            (2 * straight, 0.0, 0.0),          # 直行段: 起点=2*x_sw, 终点=2X-2*x_sw, 中点=X (第二根杆)
            (tr, 0.0, K), (platform, K, K), (tr, K, 0.0),
            (tr, 0.0, -K), (platform, -K, -K), (tr, -K, 0.0),
        ]
    else:
        segs = [
            (platform_half, -K, -K), (tr, -K, 0.0),
            (tr, 0.0, K), (platform_half, K, K),
            (platform_half, K, K), (tr, K, 0.0),
            (tr, 0.0, -K), (platform_half, -K, -K),
        ]

    ds = 1e-4
    s_vals, k_vals, s = [], [], 0.0
    for L, k0, k1 in segs:
        n = max(2, int(L / ds))
        s_vals.extend(np.linspace(s, s + L, n, endpoint=False))
        k_vals.extend(np.linspace(k0, k1, n, endpoint=False))
        s += L
    s_vals.append(s)
    k_vals.append(k_vals[-1])
    s_g = np.array(s_vals)
    k_g = np.array(k_vals)
    h = np.cumsum(k_g) * ds
    x = np.cumsum(np.cos(h)) * ds
    y = np.cumsum(np.sin(h)) * ds
    return s_g, x, y, h, k_g


def circle_through(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    A = np.array([[2.0 * (x2 - x1), 2.0 * (y2 - y1)],
                  [2.0 * (x3 - x1), 2.0 * (y3 - y1)]])
    b = np.array([x2**2 + y2**2 - x1**2 - y1**2,
                  x3**2 + y3**2 - x1**2 - y1**2])
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-12:
        return None
    cx, cy = np.linalg.solve(A, b)
    return cx, cy, np.hypot(x1 - cx, y1 - cy)


def arc_between(p1, p2, p3, n=60):
    circ = circle_through(p1, p2, p3)
    if circ is None:
        return np.array([p1, p3])
    cx, cy, r = circ
    a1 = np.arctan2(p1[1] - cy, p1[0] - cx)
    a2 = np.arctan2(p2[1] - cy, p2[0] - cx)
    a3 = np.arctan2(p3[1] - cy, p3[0] - cx)
    d = a3 - a1
    if d > np.pi:
        d -= 2 * np.pi
    elif d < -np.pi:
        d += 2 * np.pi
    if not (min(a1, a3) - 1e-6 <= a2 <= max(a1, a3) + 1e-6):
        d = -d
    thetas = np.linspace(a1, a1 + d, n)
    return np.column_stack([cx + r * np.cos(thetas), cy + r * np.sin(thetas)])


# ========== 绘图主程序 ==========
def main():
    parser = argparse.ArgumentParser(description="走廊一致性奖励示意图 (slalom)")
    parser.add_argument("--delta_theta", type=float, default=28.0,
                        help="整体朝向偏差 (度)")
    parser.add_argument("--d_lat", type=float, default=0.008,
                        help="横向偏移 (m)")
    parser.add_argument("--spine_angle", type=float, default=12.0,
                        help="脊柱弯曲角 (度)")
    parser.add_argument("--out", type=str, default="corridor_slalom.png",
                        help="输出文件名（自动存入 OUTPUT_DIR）")
    args = parser.parse_args()

    X = POLE_SPACING
    delta_theta = np.deg2rad(args.delta_theta)
    spine_angle = np.deg2rad(args.spine_angle)
    d_lat = args.d_lat

    # ---- 计算杆的 Y 坐标 ----
    tr = SMOOTH_VEL * SMOOTH_TIME
    _, x_sw_full = _get_smooth_xsw(tr)
    pole_y = -x_sw_full

    # ---- 生成绕杆路径 LUT ----
    s_lut, x_lut, y_lut, hd_lut, k_lut = generate_slalom_lut_smooth_period(X)
    period = s_lut[-1]

    # 轨迹整体右移 X: 起点 = 第一根杆 (x=X) 正上方, 周期覆盖杆1(x) 与杆2(2x)
    path_x, path_y = x_lut + X, y_lut

    cos_h = np.cos(hd_lut)
    sin_h = np.sin(hd_lut)
    n_x = -sin_h
    n_y = cos_h

    outer_x = path_x + CORRIDOR_C * n_x
    outer_y = path_y + CORRIDOR_C * n_y
    inner_x = path_x - CORRIDOR_C * n_x
    inner_y = path_y - CORRIDOR_C * n_y

    corridor_poly_pts = np.vstack([
        np.column_stack([outer_x, outer_y]),
        np.column_stack([inner_x[::-1], inner_y[::-1]])
    ])
    corridor_path = MplPath(corridor_poly_pts)

    # ---- 机器人位姿（路径中点） ----
    s_mid = period / 2.0
    idx = np.searchsorted(s_lut, s_mid)
    idx = np.clip(idx, 1, len(s_lut) - 1)
    idx_prev = idx - 1
    frac = (s_mid - s_lut[idx_prev]) / (s_lut[idx] - s_lut[idx_prev] + 1e-12)
    x0 = x_lut[idx_prev] + frac * (x_lut[idx] - x_lut[idx_prev]) + X   # 与轨迹一致的整体右移 (杆1 在 x=X)
    y0 = y_lut[idx_prev] + frac * (y_lut[idx] - y_lut[idx_prev])
    h0 = hd_lut[idx_prev]

    tangent = np.array([np.cos(h0), np.sin(h0)])
    normal = np.array([-tangent[1], tangent[0]])
    base = np.array([x0, y0]) + d_lat * normal

    beta = h0 + delta_theta

    def u(angle):
        return np.array([np.cos(angle), np.sin(angle)])

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

    f_rect = rect_pts(f_center, BODY_L, BODY_W, beta + spine_angle)
    h_rect = rect_pts(h_center, BODY_L, BODY_W, beta - spine_angle)

    spine_pts = arc_between(h_center, base, f_center)

    # ---- 超出走廊采样点 ----
    out_x, out_y, out_dist = [], [], []
    path_pts = np.column_stack([x_lut, y_lut])
    for c, ang in [(f_center, beta + spine_angle), (h_center, beta - spine_angle)]:
        ax_v = u(ang)
        ay_v = np.array([-ax_v[1], ax_v[0]])
        for du in np.linspace(-BODY_L, BODY_L, N_SAMPLE):
            for dv in np.linspace(-BODY_W, BODY_W, N_SAMPLE):
                p = c + du * ax_v + dv * ay_v
                if not corridor_path.contains_point(p):
                    out_x.append(p[0])
                    out_y.append(p[1])
                    dists = np.hypot(path_pts[:, 0] - p[0], path_pts[:, 1] - p[1])
                    min_dist = np.min(dists)
                    exceed = max(0.0, min_dist - CORRIDOR_C)
                    out_dist.append(exceed)

    # ========== 绘图 ==========
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('white')

    # 走廊边界
    ax.plot(outer_x, outer_y, '--', color='gray', linewidth=1.2, alpha=0.8)
    ax.plot(inner_x, inner_y, '--', color='gray', linewidth=1.2, alpha=0.8)

    # 路径中线
    ax.plot(path_x, path_y, '--', color='dimgrey', linewidth=2.2, dashes=(8, 4), alpha=0.9, zorder=2)

    # 三根杆 (用户几何: 杆1 在 (x,y), 杆2 在 (2x,y), 杆3 在 (3x,y))
    pole_xs = [POLE_SPACING, 2 * POLE_SPACING, 3 * POLE_SPACING]
    for px in pole_xs:
        ax.add_patch(Circle((px, pole_y), radius=POLE_RADIUS,
                                color=POLE_COLOR, zorder=POLE_ZORDER, ec='black', linewidth=0.5))

    # 脊柱弧线
    ax.plot(spine_pts[:, 0], spine_pts[:, 1], '-', color='#555555', linewidth=2.0, alpha=0.9, zorder=5)

    # 躯干
    ax.add_patch(Polygon(f_rect, closed=True, facecolor='#FFD700', edgecolor='#B8860B',
                         linewidth=1.6, alpha=0.4, zorder=4))
    ax.add_patch(Polygon(h_rect, closed=True, facecolor='#FFD700', edgecolor='#B8860B',
                         linewidth=1.6, alpha=0.4, zorder=4))

    # 脊柱关节
    ax.add_patch(Circle(base, radius=0.012, facecolor='dimgrey', edgecolor='black',
                        linewidth=1.5, zorder=6))

    # 惩罚阴影
    if out_x:
        out_x = np.array(out_x)
        out_y = np.array(out_y)
        out_dist = np.array(out_dist)
        max_exceed = out_dist.max() if len(out_dist) > 0 else 1.0
        norm = Normalize(0, max_exceed)
        sc = ax.scatter(out_x, out_y, s=18, c=out_dist, cmap='plasma_r',
                        edgecolors='none', alpha=0.95, norm=norm, zorder=7)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('超出距离 (m)', fontsize=10)

    ax.set_aspect('equal')
    ax.set_axis_off()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, args.out)
    plt.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"已保存至: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
