import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon, Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ======================================================
# 配置区
OUTPUT_DIR = r"D:\MuJoCoLab_1.5\src\mjlab\scripts\Viz_Path\Result"      # 输出目录
OUTPUT_FILENAME = "corridor_slalom.png"                                 # 输出文件名

# 机器人尺寸
CORRIDOR_C = 0.06               # 走廊半宽
BODY_L = 0.10                   # 身体半长
BODY_W = 0.055                  # 身体半宽
OFFSET = BODY_L                 # F/H 中心距 base 偏移

# 机器人姿态
DELTA_THETA = 15              # 整体朝向偏差 (度)
SPINE_ANGLE = 25              # 脊柱弯曲角 (度)
D_LAT = 0.0                   # 横向偏移 (m)
ROBO_POS = 0.6               # 机器人在一个周期路径上的位置比例 (0~1)

# 期望轨迹
POLE_SPACING = 0.2              # 杆间距 (m)
RMIN = 0.06                     # 最小转弯半径
SMOOTH_VEL = 0.3                # 平滑过渡速度 (m/s)
SMOOTH_TIME = 0.1               # 平滑过渡时间 (s)
CURVATURE_TARGET = 1.0 / RMIN   # 目标曲率 (1/m)

# 杆
POLE_RADIUS = 0.015             # 杆半径 (m)
POLE_COLOR = '#2E86C1'          # 杆颜色

# 绘图配置
N_SAMPLE = 60                   # 身体阴影采样密度 (每边)

# 走廊边界
CORRIDOR_BOUNDARY_COLOR = '#B0B0B0'
CORRIDOR_BOUNDARY_LINEWIDTH = 1.2
CORRIDOR_BOUNDARY_ALPHA = 0.9

# 路径中线
PATH_MIDLINE_COLOR = '#FFFFFF'
PATH_MIDLINE_LINEWIDTH = 2.2
PATH_MIDLINE_ALPHA = 0.9

# 躯干（安全区域）
BODY_FILL_COLOR = '#F4D03F'
BODY_EDGE_COLOR = '#B8860B'
BODY_ALPHA = 1.0
BODY_LINEWIDTH = 1.5

# 脊柱关节
JOINT_FILL_COLOR = "#FC8727"
JOINT_EDGE_COLOR = 'black'
JOINT_RADIUS = 0.014
JOINT_LINEWIDTH = 1.2

# 杆边框
POLE_EDGE_COLOR = 'black'
POLE_EDGE_LINEWIDTH = 0.8

# 惩罚阴影
PENALTY_CMAP = 'plasma_r'
PENALTY_POINT_SIZE = 25
PENALTY_EDGE_COLOR = '#2C3E50'
PENALTY_EDGE_LINEWIDTH = 0.3
PENALTY_ALPHA = 1.0
PENALTY_CBAR_LABEL = 'Exceedance (m)'

# 3D 视图
ELEVATION = 90.0
AZIMUTH = -90.0
GROUND_VISIBLE = False         # 是否显示灰色地面 (论文图通常不需要)
GROUND_COLOR = (0.5, 0.5, 0.5, 1.0)
GROUND_MARGIN = 0.12
# ======================================================

# ---------- 辅助函数（与之前相同） ----------
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
            (straight / 2, 0.0, 0.0),
            (tr, 0.0, -K), (platform, -K, -K), (tr, -K, 0.0),
            (tr, 0.0, K), (platform, K, K), (tr, K, 0.0),
            (straight, 0.0, 0.0),
            (tr, 0.0, K), (platform, K, K), (tr, K, 0.0),
            (tr, 0.0, -K), (platform, -K, -K), (tr, -K, 0.0),
            (straight / 2, 0.0, 0.0),
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

# ---------- 主函数 ----------
def main():
    delta_theta = np.deg2rad(DELTA_THETA)
    spine_angle = np.deg2rad(SPINE_ANGLE)
    d_lat = D_LAT
    X = POLE_SPACING

    tr = SMOOTH_VEL * SMOOTH_TIME
    _, x_sw_full = _get_smooth_xsw(tr)
    pole_y = -x_sw_full

    s_lut, x_lut, y_lut, hd_lut, k_lut = generate_slalom_lut_smooth_period(X)
    period = s_lut[-1]

    # 扩展轨迹使直行段完整显示
    if X >= 2 * x_sw_full - 1e-6:
        half = (X - 2 * x_sw_full) / 2.0
        mask_tail = x_lut >= 2 * X - half - 1e-9
        mask_head = x_lut <= half + 1e-9
        x_ext = np.concatenate([x_lut[mask_tail] - 2 * X, x_lut, x_lut[mask_head] + 2 * X])
        y_ext = np.concatenate([y_lut[mask_tail], y_lut, y_lut[mask_head]])
        h_ext = np.concatenate([hd_lut[mask_tail], hd_lut, hd_lut[mask_head]])
        path_x, path_y, hd_ext = x_ext + X, y_ext, h_ext
    else:
        path_x, path_y, hd_ext = x_lut + X, y_lut, hd_lut

    cos_h = np.cos(hd_ext)
    sin_h = np.sin(hd_ext)
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

    # ---- 机器人位姿 ----
    s_mid = period * ROBO_POS
    idx = np.searchsorted(s_lut, s_mid)
    idx = np.clip(idx, 1, len(s_lut) - 1)
    idx_prev = idx - 1
    frac = (s_mid - s_lut[idx_prev]) / (s_lut[idx] - s_lut[idx_prev] + 1e-12)
    x0 = x_lut[idx_prev] + frac * (x_lut[idx] - x_lut[idx_prev]) + X
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

    # ---- 超出点采样 ----
    out_x, out_y, out_dist = [], [], []
    path_pts = np.column_stack([path_x, path_y])
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

    # ========== 3D 视图绘图 ==========
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    z_plane = 0.0

    # --- 地面（可选，默认关闭） ---
    if GROUND_VISIBLE:
        gx0, gx1 = path_x.min() - GROUND_MARGIN, path_x.max() + GROUND_MARGIN
        gy0, gy1 = path_y.min() - GROUND_MARGIN, path_y.max() + GROUND_MARGIN
        ground_verts = [(gx0, gy0, z_plane), (gx1, gy0, z_plane),
                        (gx1, gy1, z_plane), (gx0, gy1, z_plane)]
        ax.add_collection3d(Poly3DCollection([ground_verts], facecolor=GROUND_COLOR[:3],
                                            alpha=GROUND_COLOR[3], edgecolor='none'))

    # --- 走廊边界 ---
    ax.plot(outer_x, outer_y, z_plane, '--', color=CORRIDOR_BOUNDARY_COLOR,
            linewidth=CORRIDOR_BOUNDARY_LINEWIDTH, alpha=CORRIDOR_BOUNDARY_ALPHA)
    ax.plot(inner_x, inner_y, z_plane, '--', color=CORRIDOR_BOUNDARY_COLOR,
            linewidth=CORRIDOR_BOUNDARY_LINEWIDTH, alpha=CORRIDOR_BOUNDARY_ALPHA)

    # --- 路径中线 ---
    ax.plot(path_x, path_y, z_plane, '--', color=PATH_MIDLINE_COLOR,
            linewidth=PATH_MIDLINE_LINEWIDTH, dashes=(8, 4), alpha=PATH_MIDLINE_ALPHA)

    # --- 杆（平面圆盘） ---
    theta = np.linspace(0, 2 * np.pi, 30)
    pole_xs = [POLE_SPACING, 2 * POLE_SPACING, 3 * POLE_SPACING]
    for px in pole_xs:
        x_circle = px + POLE_RADIUS * np.cos(theta)
        y_circle = pole_y + POLE_RADIUS * np.sin(theta)
        verts = [list(zip(x_circle, y_circle, np.full_like(x_circle, z_plane)))]
        ax.add_collection3d(Poly3DCollection(verts, facecolor=POLE_COLOR,
                                            edgecolor=POLE_EDGE_COLOR,
                                            linewidth=POLE_EDGE_LINEWIDTH))

    # --- 脊柱弧线 ---
    ax.plot(spine_pts[:, 0], spine_pts[:, 1], z_plane, '-', color='#404040', linewidth=2.0)

    # --- 躯干矩形 ---
    for rect in (f_rect, h_rect):
        verts = [(x, y, z_plane) for x, y in rect]
        ax.add_collection3d(Poly3DCollection([verts], facecolor=BODY_FILL_COLOR,
                                            edgecolor=BODY_EDGE_COLOR,
                                            linewidth=BODY_LINEWIDTH, alpha=BODY_ALPHA))

    # --- 脊柱关节（平面圆盘） ---
    jt = np.linspace(0, 2 * np.pi, 20)
    jx = base[0] + JOINT_RADIUS * np.cos(jt)
    jy = base[1] + JOINT_RADIUS * np.sin(jt)
    ax.add_collection3d(Poly3DCollection([list(zip(jx, jy, np.full_like(jx, z_plane)))],
                                        facecolor=JOINT_FILL_COLOR,
                                        edgecolor=JOINT_EDGE_COLOR,
                                        linewidth=JOINT_LINEWIDTH))

    # --- 惩罚阴影（散点，近似平面圆点） ---
    # 由于点数量巨大，此处保持 scatter；若需严格平面圆点可改用微型多边形，但会影响性能。
    if out_x:
        out_x = np.array(out_x)
        out_y = np.array(out_y)
        out_dist = np.array(out_dist)
        max_exceed = out_dist.max() if len(out_dist) > 0 else 1.0
        norm = Normalize(0, max_exceed)
        sc = ax.scatter(out_x, out_y, 0, s=PENALTY_POINT_SIZE,
                        c=out_dist, cmap=PENALTY_CMAP, norm=norm,
                        edgecolors=PENALTY_EDGE_COLOR, linewidth=PENALTY_EDGE_LINEWIDTH,
                        alpha=PENALTY_ALPHA)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label(PENALTY_CBAR_LABEL, fontsize=10, labelpad=8)
        cbar.ax.tick_params(labelsize=9, width=0.8)

    # --- 视角设置 ---
    # 计算数据跨度
    x_span = 0.1
    y_span = 0.05
    z_span = 0.01          # Z 轴跨度极小，仅用于保留 3D 结构

    ax.set_zlim(z_plane - z_span, z_plane + z_span)
    ax.set_box_aspect((x_span, y_span, z_span))   # 关键：保证 XY 单位长度相等
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)

    # 坐标轴与网格
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_zlabel('')
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_axis_off()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    main()