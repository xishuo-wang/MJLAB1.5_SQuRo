"""SQuRo 虚拟碰撞几何模型示意 — 躯干矩形 + 腿线段 + 杆圆

展示用于碰撞检测的简化几何体在 XoY 平面上的投影。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D

# ========== 几何参数 (与 path.py 一致) ==========
F_L, F_W = 0.04, 0.035   # F_body 半长、半宽
H_L, H_W = 0.04, 0.035   # H_body 半长、半宽
BODY_OFFSET = 0.04       # F/H 中心距 base 的偏移
POLE_R = 0.005           # 杆半径 (直径 1cm)
POLE_Y = -1/15           # 杆心 Y

# 腿附着点 — 身体短边边框的中点 (长边的中点, body 局部坐标)
# F_body: 长边中点 (0, ±W_f), H_body: 长边中点 (0, ±W_h)
ATTACH_FL = np.array([ 0.0, -F_W])   # F_body 右侧中点
ATTACH_FR = np.array([ 0.0, +F_W])   # F_body 左侧中点
ATTACH_HL = np.array([ 0.0, -H_W])   # H_body 右侧中点
ATTACH_HR = np.array([ 0.0, +H_W])   # H_body 左侧中点

# 示例足端位置 (世界坐标, 模拟站立姿态)
FOOT_POS = {
    "FL": np.array([ 0.05, -0.06]),
    "FR": np.array([ 0.05, -0.02]),
    "HL": np.array([-0.05, -0.06]),
    "HR": np.array([-0.05, -0.02]),
}


def rotate_point(pt, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * pt[0] - s * pt[1], s * pt[0] + c * pt[1]])


def draw_body(ax, center, angle, L, W, color, label, alpha=0.3):
    """绘制旋转矩形 (身体环节)"""
    # 矩形左下角在局部坐标 (-L, -W)
    rect = Rectangle(
        (center[0] - L, center[1] - W), 2*L, 2*W,
        angle=np.degrees(angle), rotation_point='center',
        facecolor=color, edgecolor='black', alpha=alpha, label=label,
    )
    ax.add_patch(rect)
    # 中心点
    ax.plot(center[0], center[1], 'o', color=color, markersize=6)
    # 前向箭头
    arrow_len = L * 0.8
    ax.arrow(center[0], center[1],
             arrow_len * np.cos(angle), arrow_len * np.sin(angle),
             head_width=0.006, color=color, alpha=0.7)


def draw_leg(ax, attach_pt, foot_pt, color):
    """绘制腿线段"""
    ax.plot([attach_pt[0], foot_pt[0]], [attach_pt[1], foot_pt[1]],
            '-', color=color, linewidth=2)
    ax.plot(foot_pt[0], foot_pt[1], 'o', color=color, markersize=5)


def point_to_rect_dist(px, py, cx, cy, angle, L, W):
    """点到旋转矩形的最短距离 (用于碰撞检测)"""
    dx, dy = px - cx, py - cy
    local_x =  dx * np.cos(-angle) + dy * np.sin(-angle)
    local_y = -dx * np.sin(-angle) + dy * np.cos(-angle)
    clamp_x = np.clip(local_x, -L, L)
    clamp_y = np.clip(local_y, -W, W)
    return np.hypot(local_x - clamp_x, local_y - clamp_y)


def point_to_segment_dist(px, py, x1, y1, x2, y2):
    """点到线段的最短距离"""
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return np.hypot(px - x1, py - y1)
    t = np.clip(((px - x1)*dx + (py - y1)*dy) / (dx*dx + dy*dy), 0, 1)
    return np.hypot(px - (x1 + t*dx), py - (y1 + t*dy))


# ========== 两种身体姿态 ==========
scenarios = [
    ("直行 (heading=0)", 0.0, 0.0, 0.0),
    ("右转 C 形 (F=-10°, H=+10°)", -0.0, np.radians(-15), np.radians(+15)),
    ("右转加大 (F=-15°, H=+15°)", -0.0, np.radians(-30), np.radians(+30)),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (title, base_angle, f_angle, h_angle) in zip(axes, scenarios):
    # 身体中心位置
    base = np.array([0.05, -0.01])
    f_center = base + rotate_point(np.array([ BODY_OFFSET, 0]), base_angle)
    h_center = base + rotate_point(np.array([-BODY_OFFSET, 0]), base_angle)

    # 绘制身体
    draw_body(ax, f_center, base_angle + f_angle, F_L, F_W, 'dodgerblue', 'F_body')
    draw_body(ax, h_center, base_angle + h_angle, H_L, H_W, 'darkorange', 'H_body')
    ax.plot(base[0], base[1], 's', color='gray', markersize=8, label='base_Link')

    # 腿附着点 (世界坐标) — 身体长边中点
    f_attach_L = f_center + rotate_point(ATTACH_FL, base_angle + f_angle)
    f_attach_R = f_center + rotate_point(ATTACH_FR, base_angle + f_angle)
    h_attach_L = h_center + rotate_point(ATTACH_HL, base_angle + h_angle)
    h_attach_R = h_center + rotate_point(ATTACH_HR, base_angle + h_angle)

    # 绘制腿
    draw_leg(ax, f_attach_L, FOOT_POS["FL"], 'steelblue')
    draw_leg(ax, f_attach_R, FOOT_POS["FR"], 'steelblue')
    draw_leg(ax, h_attach_L, FOOT_POS["HL"], 'peru')
    draw_leg(ax, h_attach_R, FOOT_POS["HR"], 'peru')

    # 杆
    for px in [0.0, 0.15, 0.30, 0.45, 0.6]:
        pole = Circle((px, POLE_Y), POLE_R, color='red', alpha=0.6)
        ax.add_patch(pole)

    # 身体-杆 碰撞检测示意 (标注最近点)
    for px in [0.0, 0.15, 0.30]:
        for body_c, body_a, L, W, c in [
            (f_center, base_angle + f_angle, F_L, F_W, 'cyan'),
            (h_center, base_angle + h_angle, H_L, H_W, 'orange'),
        ]:
            d = point_to_rect_dist(px, POLE_Y, body_c[0], body_c[1], body_a, L, W)
            if d < POLE_R + 0.02:  # 接近或碰撞
                # 找到矩形上最近点
                dx, dy = px - body_c[0], POLE_Y - body_c[1]
                lx =  dx * np.cos(-body_a) + dy * np.sin(-body_a)
                ly = -dx * np.sin(-body_a) + dy * np.cos(-body_a)
                cx = np.clip(lx, -L, L)
                cy = np.clip(ly, -W, W)
                nearest = body_c + rotate_point(np.array([cx, cy]), body_a)
                ax.plot([px, nearest[0]], [POLE_Y, nearest[1]], '--', color='red', linewidth=0.8)
                ax.plot(nearest[0], nearest[1], 'x', color='red', markersize=4)
                if d < POLE_R:
                    ax.annotate(f'碰撞', (nearest[0], nearest[1]),
                                fontsize=8, color='red', fontweight='bold',
                                xytext=(5, -10), textcoords='offset points')

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.axis('equal'); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper right')

plt.suptitle('SQuRo 虚拟碰撞几何模型 — 躯干(矩形) + 腿(线段) + 杆(圆)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
