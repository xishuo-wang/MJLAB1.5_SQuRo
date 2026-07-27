"""绕杆期望轨迹对比 — 圆弧拼接 vs 平滑曲率

路径生成逻辑（一个完整周期从 (0,0) 出发）:
  1. 顺时针 1/4弧: (0,0) → (Rmin, -Rmin)
  2. 逆时针 1/4弧: (Rmin, -Rmin) → (2Rmin, -2Rmin)
  3. 直线:          (2Rmin, -2Rmin) → (X, -2Rmin)      [长度 X-2Rmin]
  4. 逆时针 1/4弧: (X, -2Rmin) → (X+Rmin, -Rmin)
  5. 顺时针 1/4弧: (X+Rmin, -Rmin) → (X+2Rmin, 0)
  6. 直线:          (X+2Rmin, 0) → (2X, 0)             [长度 X-2Rmin]

平滑版本: 在圆弧段和直线段之间插入余弦缓变过渡段，
  κ 从 arc_κ 连续过渡到 straight_κ=0 (或反向)，
  过渡段长度 = Rmin (一个半径距离)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec

# ========== 参数 ==========
Rmin = 0.10
pole_radius = 0.005
velocity = 0.1          # 前进速度，仅用于计算曲率剖面时间轴

# ========== 辅助函数：给定起终点和半径、方向生成1/4圆弧 ==========
def arc_from_start_end(start, end, r, clockwise, steps=30):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    mid = (start + end) / 2
    chord_vec = end - start
    chord_len = np.linalg.norm(chord_vec)
    if chord_len > 2 * r:
        raise ValueError("No circle with this radius can connect the points")
    d = np.sqrt(max(0, r**2 - (chord_len / 2)**2))
    perp = np.array([-chord_vec[1], chord_vec[0]]) / chord_len
    sign = -1 if clockwise else 1
    center = mid + sign * d * perp

    v_start = start - center
    v_end = end - center
    ang_start = np.arctan2(v_start[1], v_start[0])
    ang_end = np.arctan2(v_end[1], v_end[0])

    if clockwise:
        if ang_end > ang_start:
            ang_end -= 2 * np.pi
    else:
        if ang_end < ang_start:
            ang_end += 2 * np.pi

    theta = np.linspace(ang_start, ang_end, steps)
    return center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)


# ========== 辅助函数：定角圆弧 ==========
def arc_from_center(cx, cy, r, theta_start, theta_end, clockwise, steps=30):
    theta = np.linspace(theta_start, theta_end, steps)
    if clockwise:
        theta = theta[::-1]  # 反转方向
    return cx + r * np.cos(theta), cy + r * np.sin(theta)


# ========== 生成一个周期的路径 (圆弧拼接) — 同时返回分段信息用于曲率分析 ==========
def generate_one_period_arc(X, Rmin):
    x0, y0 = 0.0, 0.0
    path_x, path_y = [x0], [y0]
    # 记录每段的 (start_arc_idx, end_arc_idx, kappa, segment_type)
    segments = []
    idx = 0

    # S1: 顺时针 1/4弧
    ax, ay = arc_from_start_end((x0, y0), (x0 + Rmin, y0 - Rmin), Rmin, clockwise=True)
    path_x.extend(ax); path_y.extend(ay)
    idx += len(ax)
    segments.append(("CW_arc",  +1/Rmin, len(ax)))

    # S2: 逆时针 1/4弧
    ax, ay = arc_from_start_end((x0 + Rmin, y0 - Rmin), (x0 + 2*Rmin, y0 - 2*Rmin), Rmin, clockwise=False)
    path_x.extend(ax); path_y.extend(ay)
    idx += len(ax)
    segments.append(("CCW_arc", -1/Rmin, len(ax)))

    # S3: 直线
    if X - 2 * Rmin > 1e-9:
        path_x.append(x0 + X)
        path_y.append(y0 - 2 * Rmin)
        idx += 1
        segments.append(("line", 0.0, 1))

    # S4: 逆时针 1/4弧
    ax, ay = arc_from_start_end((x0 + X, y0 - 2*Rmin), (x0 + X + Rmin, y0 - Rmin), Rmin, clockwise=False)
    path_x.extend(ax); path_y.extend(ay)
    idx += len(ax)
    segments.append(("CCW_arc", -1/Rmin, len(ax)))

    # S5: 顺时针 1/4弧
    ax, ay = arc_from_start_end((x0 + X + Rmin, y0 - Rmin), (x0 + X + 2*Rmin, y0), Rmin, clockwise=True)
    path_x.extend(ax); path_y.extend(ay)
    idx += len(ax)
    segments.append(("CW_arc",  +1/Rmin, len(ax)))

    # S6: 直线
    if X - 2 * Rmin > 1e-9:
        path_x.append(x0 + 2 * X)
        path_y.append(y0)
        idx += 1
        segments.append(("line", 0.0, 1))

    return np.array(path_x), np.array(path_y), segments


# ========== 多周期路径 (圆弧拼接) ==========
def generate_arc_path(X, Rmin, num_periods=3):
    x_all, y_all = [0.0], [0.0]
    all_segments = []
    cur_x, cur_y = 0.0, 0.0
    for k in range(num_periods):
        px, py, segs = generate_one_period_arc(X, Rmin)
        px_shifted = px[1:] + cur_x
        py_shifted = py[1:] + cur_y
        x_all.extend(px_shifted)
        y_all.extend(py_shifted)
        cur_x = x_all[-1]
        cur_y = y_all[-1]
        all_segments.extend(segs)
    return np.array(x_all), np.array(y_all), all_segments


# ========== 构建曲率剖面 (阶跃版) ==========
def build_kappa_profile_arc(segments):
    """从圆弧段描述构建逐点曲率剖面 (阶跃)"""
    kappa = []
    for _, k_val, n_pts in segments:
        kappa.extend([k_val] * n_pts)
    return np.array(kappa)


# ========== 构建曲率剖面 (平滑版) ==========
def build_kappa_profile_smooth(segments, trans_steps=15):
    """在段与段之间插入余弦缓变过渡"""
    kappa_smooth = []
    prev_k = 0.0

    for _, k_val, n_pts in segments:
        # 插入过渡段: prev_k → k_val (余弦缓变)
        if abs(prev_k - k_val) > 1e-9:
            for i in range(trans_steps):
                t = i / trans_steps                   # 0 → 1
                k_trans = prev_k + (k_val - prev_k) * (1 - np.cos(np.pi * t)) / 2
                kappa_smooth.append(k_trans)
        # 主段: 恒定曲率 (或直线段)
        kappa_smooth.extend([k_val] * n_pts)
        prev_k = k_val

    return np.array(kappa_smooth)


# ========== 从曲率剖面重建路径 (数值积分) ==========
def reconstruct_path_from_kappa(kappa_profile, ds_per_step=0.002):
    """κ(s) → θ(s) = ∫κ(s)ds → (x,y) = (∫cosθ ds, ∫sinθ ds)"""
    n = len(kappa_profile)
    x = np.zeros(n)
    y = np.zeros(n)
    theta = 0.0
    for i in range(1, n):
        kappa_avg = (kappa_profile[i - 1] + kappa_profile[i]) / 2
        theta += kappa_avg * ds_per_step
        x[i] = x[i - 1] + np.cos(theta) * ds_per_step
        y[i] = y[i - 1] + np.sin(theta) * ds_per_step
    return x, y


# ========== 主可视化 ==========
X_values = [0.2, 0.3]
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

for col, X in enumerate(X_values):
    title_extra = "X=2Rmin (全弧无直行)" if abs(X - 2*Rmin) < 1e-9 else f"X={X/Rmin}Rmin (含直行段)"

    # ---- 子图1: 路径对比 ----
    ax_path = fig.add_subplot(gs[0, col])

    # 杆
    num_poles = 6
    poles_x = np.arange(num_poles) * X
    poles_y = np.full(num_poles, -Rmin)
    for px, py in zip(poles_x, poles_y):
        ax_path.add_patch(Circle((px, py), pole_radius, color='red', alpha=0.5))
    ax_path.plot(poles_x, poles_y, 'rx', markersize=8)

    # 圆弧拼接路径
    arc_x, arc_y, segments = generate_arc_path(X, Rmin, num_periods=3)
    ax_path.plot(arc_x, arc_y, 'orange', linewidth=2.5, alpha=0.7, label='圆弧拼接 (κ阶跃)')

    # 平滑路径
    kappa_step = build_kappa_profile_arc(segments)
    kappa_smooth = build_kappa_profile_smooth(segments, trans_steps=10)
    smooth_x, smooth_y = reconstruct_path_from_kappa(kappa_smooth)

    ax_path.plot(smooth_x, smooth_y, 'b-', linewidth=2, label='平滑曲率 (κ连续)')

    # 标注 waypoints
    for period_idx in range(3):
        base_x = period_idx * 2 * X
        wp = [(base_x, 0), (base_x+X, -2*Rmin), (base_x+2*X, 0)]
        for wx, wy in wp:
            ax_path.plot(wx, wy, 'go', markersize=4, alpha=0.5)

    ax_path.plot(0, 0, 'go', markersize=10, label='起点')
    ax_path.axhline(0, color='gray', linestyle=':')
    ax_path.axhline(-Rmin, color='red', linestyle='--', alpha=0.3)
    ax_path.axhline(-2*Rmin, color='blue', linestyle='--', alpha=0.3)
    ax_path.set_xlabel('X (m)'); ax_path.set_ylabel('Y (m)')
    ax_path.set_title(f'路径对比 — {title_extra}')
    ax_path.axis('equal'); ax_path.grid(True); ax_path.legend(fontsize=8)

    # ---- 子图2: 局部放大 (第一周期) ----
    ax_zoom = fig.add_subplot(gs[1, col])
    zoom_xlim = (-0.05, 2*X + 0.05)
    zoom_ylim = (-2*Rmin - 0.05, 0.05)

    for px, py in zip(poles_x[:3], poles_y[:3]):
        ax_zoom.add_patch(Circle((px, py), pole_radius, color='red', alpha=0.6))
    ax_zoom.plot(arc_x, arc_y, 'orange', linewidth=2.5, alpha=0.7)
    ax_zoom.plot(smooth_x, smooth_y, 'b-', linewidth=2)

    # waypoints
    for wx, wy in [(0,0), (Rmin,-Rmin), (2*Rmin,-2*Rmin), (X,-2*Rmin),
                    (X+Rmin,-Rmin), (X+2*Rmin,0), (2*X,0)]:
        if wx <= zoom_xlim[1]:
            ax_zoom.plot(wx, wy, 'go', markersize=6)
            ax_zoom.annotate(f'({wx:.1f},{wy:.1f})', (wx, wy),
                             textcoords="offset points", xytext=(5, -10), fontsize=7)

    ax_zoom.set_xlim(zoom_xlim); ax_zoom.set_ylim(zoom_ylim)
    ax_zoom.set_xlabel('X (m)'); ax_zoom.set_ylabel('Y (m)')
    ax_zoom.set_title(f'第一周期放大 — {title_extra}')
    ax_zoom.axis('equal'); ax_zoom.grid(True)

# ---- 子图3: 曲率剖面对比 (取 X=0.3) ----
ax_k = fig.add_subplot(gs[2, :])
X_k = 0.3
_, _, segs_k = generate_one_period_arc(X_k, Rmin)
kappa_step_k = build_kappa_profile_arc(segs_k)
kappa_smooth_k = build_kappa_profile_smooth(segs_k, trans_steps=10)

# 沿弧长的位置
s_step = np.arange(len(kappa_step_k)) * 0.002
s_smooth = np.arange(len(kappa_smooth_k)) * 0.002

ax_k.step(s_step, kappa_step_k, 'orange', linewidth=2, where='post',
          alpha=0.8, label='圆弧拼接 κ(s)')
ax_k.plot(s_smooth, kappa_smooth_k, 'b-', linewidth=2,
          alpha=0.8, label='平滑曲率 κ(s) (余弦过渡)')

# 标注过渡段
for i in range(1, len(kappa_smooth_k)):
    if abs(kappa_smooth_k[i] - kappa_smooth_k[i-1]) > 0.1:
        if abs(kappa_smooth_k[i] - kappa_smooth_k[i-1]) / 0.002 < 50:  # 排除段内变化
            ax_k.axvspan(s_smooth[i-10], s_smooth[i+10], alpha=0.1, color='green')

# 关键 κ 值标注
ax_k.axhline(y=+1/Rmin, color='gray', linestyle=':', alpha=0.5)
ax_k.axhline(y=-1/Rmin, color='gray', linestyle=':', alpha=0.5)
ax_k.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax_k.text(s_step[-1]*0.02, +1/Rmin, f'+1/Rmin={1/Rmin:.0f}', va='bottom', fontsize=8, alpha=0.6)
ax_k.text(s_step[-1]*0.02, -1/Rmin, f'-1/Rmin=-{1/Rmin:.0f}', va='top', fontsize=8, alpha=0.6)

ax_k.set_xlabel('弧长 s (m)')
ax_k.set_ylabel('曲率 κ (1/m)')
ax_k.set_title(f'曲率剖面对比 (X={X_k}m) — 橙色: κ 阶跃 | 蓝色: 余弦过渡平滑 | 绿色区域: 过渡段')
ax_k.grid(True); ax_k.legend()
ax_k.set_xlim(0, s_smooth[-1])

plt.tight_layout()
plt.show()
