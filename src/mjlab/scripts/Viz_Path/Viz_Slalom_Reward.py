import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon, Circle


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
DELTA_THETA = 25              # 整体朝向偏差 (度)
SPINE_ANGLE = 25              # 脊柱弯曲角 (度)
D_LAT = 0.0                   # 横向偏移 (m)
ROBO_POS = 0.56       # 机器人在一个周期路径上的位置比例 (0~1)

# 期望轨迹
POLE_SPACING = 0.2              # 杆间距 (m)
RMIN = 0.06                     # 最小转弯半径
SMOOTH_VEL = 0.3                # 平滑过渡速度 (m/s)
SMOOTH_TIME = 0.1               # 平滑过渡时间 (s)
CURVATURE_TARGET = 1.0 / RMIN   # 目标曲率 (1/m)，由最小半径自动计算

# 杆
POLE_RADIUS = 0.015             # 杆半径 (m)
POLE_COLOR = '#2E86C1'          # 杆颜色 (蓝色, 与灰色地面/橙黄机器人区分)
POLE_ZORDER = 8                 # 绘图层级（高于机器人）

# 绘图配置
N_SAMPLE = 60                   # 身体阴影采样密度 (每边)
# 走廊边界
CORRIDOR_BOUNDARY_COLOR = '#B0B0B0'
CORRIDOR_BOUNDARY_LINEWIDTH = 1.2
CORRIDOR_BOUNDARY_ALPHA = 0.9

# 路径中线
PATH_MIDLINE_COLOR = '#FFFFFF'   # 路径中线 (白色, 灰色地面上更明显)
PATH_MIDLINE_LINEWIDTH = 2.2
PATH_MIDLINE_ALPHA = 0.9

# 躯干（走廊内的安全区域）
BODY_FILL_COLOR = '#F4D03F'
BODY_EDGE_COLOR = '#B8860B'
BODY_ALPHA = 1.0
BODY_LINEWIDTH = 1.5

# 脊柱关节
JOINT_FILL_COLOR = "#FC8727"
JOINT_EDGE_COLOR = 'black'
JOINT_RADIUS = 0.014
JOINT_LINEWIDTH = 1.2

# 杆 (保留原 POLE_COLOR，补充边框属性)
POLE_EDGE_COLOR = 'black'
POLE_EDGE_LINEWIDTH = 0.8

# 惩罚阴影（超出区域）
PENALTY_CMAP = 'coolwarm'       # 奖励色: 蓝(小)→红(大), 与躯干橙黄区分
PENALTY_POINT_SIZE = 25
PENALTY_EDGE_COLOR = '#2C3E50'
PENALTY_EDGE_LINEWIDTH = 0.3
PENALTY_ALPHA = 1.0
PENALTY_CBAR_LABEL = 'Exceedance (m)'

# 3D 视图 (自投影等距视图: 3D坐标→2D, zorder 控制层叠; 地面 zorder=1 最底层)
ELEVATION = 25.0                     # 仰角 (度): 90=正俯视, 0=水平
AZIMUTH = -45.0                      # 方位角 (度): 绕 Z 轴旋转, 灵活调整视角
GROUND_COLOR = (0.5, 0.5, 0.5, 1.0)  # 地面颜色 (有限长方形)
GROUND_MARGIN = 0.12                 # 地面超出轨迹范围的边距 (m)
# 注意: 所有内容(杆/机器人/轨迹/走廊/阴影)均位于 XOY 平面 z=0,
#       采用自投影等距视图仅为旋转视角看图 (无 matplotlib 3D 层叠问题)
# ======================================================


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
            (straight / 2, 0.0, 0.0),          # S_a 直行 y=0 (杆1 上方直行右半, 起点=杆1正上方)
            (tr, 0.0, -K), (platform, -K, -K), (tr, -K, 0.0),
            (tr, 0.0, K), (platform, K, K), (tr, K, 0.0),
            (straight, 0.0, 0.0),              # S_b 直行 y=-2*x_sw (杆2 下方, 中点=杆2 x)
            (tr, 0.0, K), (platform, K, K), (tr, K, 0.0),
            (tr, 0.0, -K), (platform, -K, -K), (tr, -K, 0.0),
            (straight / 2, 0.0, 0.0),          # S_c 直行 y=0 (杆3 上方直行左半, 终点=杆3正上方)
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


def main():
    delta_theta = np.deg2rad(DELTA_THETA)
    spine_angle = np.deg2rad(SPINE_ANGLE)
    d_lat = D_LAT
    X = POLE_SPACING

    #计算杆的 Y 坐标
    tr = SMOOTH_VEL * SMOOTH_TIME
    _, x_sw_full = _get_smooth_xsw(tr)
    pole_y = -x_sw_full

    # 生成绕杆路径 LUT
    s_lut, x_lut, y_lut, hd_lut, k_lut = generate_slalom_lut_smooth_period(X)
    period = s_lut[-1]

    # 使杆1/2/3 的完整直行段均可见 (杆位于直行段正中间)
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

    # ---- 机器人位姿（路径中点） ----
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

    # ---- 超出走廊采样点 (path_pts 必须用平移后的轨迹, 与采样点坐标一致) ----
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

    # ========== 绘图 (自投影等距视图: 3D坐标→2D, zorder 控制层叠) ==========
    az_r = np.deg2rad(AZIMUTH)
    el_r = np.deg2rad(ELEVATION)
    _ca, _sa = np.cos(az_r), np.sin(az_r)
    _ce, _se = np.cos(el_r), np.sin(el_r)

    def proj(x, y, z=0.0):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        return x * _ca - y * _sa, x * _sa * _se + y * _ca * _se + z * _ce

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('white')

    # 地面 (XOY 平面有限长方形, 灰色填充, zorder=1 最底层)
    gx0, gx1 = path_x.min() - GROUND_MARGIN, path_x.max() + GROUND_MARGIN
    gy0, gy1 = path_y.min() - GROUND_MARGIN, path_y.max() + GROUND_MARGIN
    g_px, g_py = proj([gx0, gx1, gx1, gx0], [gy0, gy0, gy1, gy1], 0.0)
    ax.add_patch(Polygon(np.column_stack([g_px, g_py]), closed=True,
                         facecolor=GROUND_COLOR, edgecolor=(0.35, 0.35, 0.35),
                         linewidth=0.8, zorder=1))

    # 走廊边界 (z=0)
    o_px, o_py = proj(outer_x, outer_y)
    i_px, i_py = proj(inner_x, inner_y)
    ax.plot(o_px, o_py, '--', color=CORRIDOR_BOUNDARY_COLOR,
            linewidth=CORRIDOR_BOUNDARY_LINEWIDTH, alpha=CORRIDOR_BOUNDARY_ALPHA, zorder=2)
    ax.plot(i_px, i_py, '--', color=CORRIDOR_BOUNDARY_COLOR,
            linewidth=CORRIDOR_BOUNDARY_LINEWIDTH, alpha=CORRIDOR_BOUNDARY_ALPHA, zorder=2)

    # 路径中线 (z=0)
    m_px, m_py = proj(path_x, path_y)
    ax.plot(m_px, m_py, '--', color=PATH_MIDLINE_COLOR, linewidth=PATH_MIDLINE_LINEWIDTH,
            dashes=(8, 4), alpha=PATH_MIDLINE_ALPHA, zorder=2)

    # 三根杆 (2D 圆, z=0, zorder=3)
    pole_xs = [POLE_SPACING, 2 * POLE_SPACING, 3 * POLE_SPACING]
    for px in pole_xs:
        c_px, c_py = proj(px, pole_y, 0.0)
        ax.add_patch(Circle((c_px, c_py), radius=POLE_RADIUS, facecolor=POLE_COLOR,
                            edgecolor=POLE_EDGE_COLOR, linewidth=POLE_EDGE_LINEWIDTH, zorder=3))

    # 机器人躯干 (z=0 矩形, zorder=4) 及脊柱关节
    for rect in (f_rect, h_rect):
        r_px, r_py = proj(rect[:, 0], rect[:, 1], 0.0)
        ax.add_patch(Polygon(np.column_stack([r_px, r_py]), closed=True,
                             facecolor=BODY_FILL_COLOR, edgecolor=BODY_EDGE_COLOR,
                             linewidth=BODY_LINEWIDTH, alpha=BODY_ALPHA, zorder=4))
    b_px, b_py = proj(base[0], base[1], 0.0)
    ax.scatter([b_px], [b_py], s=80, color=JOINT_FILL_COLOR, edgecolors=JOINT_EDGE_COLOR,
               linewidth=JOINT_LINEWIDTH, zorder=5)

    # 惩罚阴影 (z=0 散点, zorder=6)
    if out_x:
        out_x = np.array(out_x)
        out_y = np.array(out_y)
        out_dist = np.array(out_dist)
        max_exceed = out_dist.max() if len(out_dist) > 0 else 1.0
        norm = Normalize(0, max_exceed)
        s_px, s_py = proj(out_x, out_y, 0.0)
        sc = ax.scatter(s_px, s_py, s=PENALTY_POINT_SIZE, c=out_dist, cmap=PENALTY_CMAP,
                        edgecolors=PENALTY_EDGE_COLOR, linewidth=PENALTY_EDGE_LINEWIDTH,
                        alpha=PENALTY_ALPHA, norm=norm, zorder=6)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label(PENALTY_CBAR_LABEL, fontsize=10, labelpad=8)
        cbar.ax.tick_params(labelsize=9, width=0.8)
        cbar.outline.set_linewidth(0.6)  # type: ignore

    ax.set_aspect('equal')
    ax.set_axis_off()

    # 保存图片
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"已保存至: {save_path} (elevation={ELEVATION}, azimuth={AZIMUTH})")
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    main()
