import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.widgets import Slider

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def rodrigues(v, k, angle):
    return v * np.cos(angle) + np.cross(k, v) * np.sin(angle) + k * np.dot(k, v) * (1 - np.cos(angle))

def rot_matrix_axis_angle(axis, angle):
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def draw_spine(ax, theta_f=0.0, theta_h=0.0, alpha=0.0, beta=0.0):
    # 尺寸 (米)
    spine_len = 0.039
    base_radius = 0.01
    body_half_x = 0.019   # X半长
    body_half_y = 0.025
    body_half_z = 0.025

    X = np.array([1.,0.,0.])
    Y = np.array([0.,1.,0.])
    Z = np.array([0.,0.,1.])

    # baselink 中心 (原点)
    base_center = np.array([0.,0.,0.])

    # 扭转角
    phi = (theta_f + theta_h) / 2.0

    # 扭转后的关节轴
    axis_alpha = rodrigues(Z, X, phi)   # F_spine1 轴
    axis_beta  = rodrigues(Y, X, phi)   # H_spine1 轴

    # 连杆从球心出发
    f_joint_base = base_center
    h_joint_base = base_center

    # F_spine1 Link 方向
    f_link_dir = rodrigues(X, axis_alpha, alpha)
    # 连接点：F_body -X 面中心 (后表面)
    f_connect = f_joint_base + spine_len * f_link_dir

    # H_spine1 Link 方向 (初始 -X)
    h_link_dir = rodrigues(-X, axis_beta, beta)
    # 连接点：H_body +X 面中心 (前表面)
    h_connect = h_joint_base + spine_len * h_link_dir

    # F_body 方块中心：连接点沿前向偏移半长
    f_body_center = f_connect + body_half_x * f_link_dir
    # H_body 方块中心：连接点沿后向偏移半长 (h_link_dir 指向 -X 方向，所以减半长即向 +X)
    h_body_center = h_connect + body_half_x * h_link_dir  # 注意 h_link_dir 是 -X，减负 = 加正

    # 朝向计算 (同前)
    R_link_f = rot_matrix_axis_angle(axis_alpha, alpha)
    f_body_x = f_link_dir
    f_body_z = R_link_f @ Z
    R_twist_f = rot_matrix_axis_angle(f_body_x, theta_f)
    f_body_z = R_twist_f @ f_body_z
    f_body_y = np.cross(f_body_z, f_body_x)

    R_link_h = rot_matrix_axis_angle(axis_beta, beta)
    h_body_x = h_link_dir
    h_body_z = R_link_h @ Z
    R_twist_h = rot_matrix_axis_angle(h_body_x, theta_h)
    h_body_z = R_twist_h @ h_body_z
    h_body_y = np.cross(h_body_z, h_body_x)

    # 绘图
    ax.clear()
    ax.set_xlim(-0.12, 0.12)
    ax.set_ylim(-0.12, 0.12)
    ax.set_zlim(-0.12, 0.12)
    ax.set_xlabel('X (前)')
    ax.set_ylabel('Y (左)')
    ax.set_zlabel('Z (上)')
    ax.set_box_aspect((1,1,1))

    # baselink 小球
    u = np.linspace(0, 2*np.pi, 16)
    v = np.linspace(0, np.pi, 16)
    xs = base_radius * np.outer(np.cos(u), np.sin(v))
    ys = base_radius * np.outer(np.sin(u), np.sin(v))
    zs = base_radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='gray', alpha=0.4)

    # 绘制连杆 (球心 → 连接点)
    ax.plot([0, f_connect[0]], [0, f_connect[1]], [0, f_connect[2]],
            'r-', lw=3, label='F_spine1 Link')
    ax.plot([0, h_connect[0]], [0, h_connect[1]], [0, h_connect[2]],
            'b-', lw=3, label='H_spine1 Link')

    # 绘制身体方块
    def draw_rect_prism(ax, center, x_axis, y_axis, z_axis, hx, hy, hz, color):
        corners = []
        for dx in (-hx, hx):
            for dy in (-hy, hy):
                for dz in (-hz, hz):
                    corners.append(center + dx*x_axis + dy*y_axis + dz*z_axis)
        edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
        for i,j in edges:
            ax.plot([corners[i][0], corners[j][0]],
                    [corners[i][1], corners[j][1]],
                    [corners[i][2], corners[j][2]], color=color)

    draw_rect_prism(ax, f_body_center, f_body_x, f_body_y, f_body_z,
                    body_half_x, body_half_y, body_half_z, 'green')
    draw_rect_prism(ax, h_body_center, h_body_x, h_body_y, h_body_z,
                    body_half_x, body_half_y, body_half_z, 'orange')

    # 标记连接点 (关节位置)
    ax.scatter(*f_connect, c='darkgreen', s=40, marker='o', label='F_body关节')
    ax.scatter(*h_connect, c='darkorange', s=40, marker='o', label='H_body关节')

    # 关节轴 (球心)
    ax.quiver(0,0,0, *(axis_alpha*0.05), color='cyan', label='F_spine1轴')
    ax.quiver(0,0,0, *(axis_beta*0.05), color='magenta', label='H_spine1轴')
    ax.quiver(0,0,0, 0.08,0,0, color='black', alpha=0.5, label='X扭转轴')

    # 世界坐标系提示
    ax.text2D(0.02, 0.02, '世界: X前 Y左 Z上', transform=ax.transAxes,
              fontsize=9, color='navy', bbox=dict(facecolor='white', alpha=0.8))
    ax.legend(loc='upper left', fontsize=8)

# 交互界面
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')
draw_spine(ax, 0,0,0,0)

# 滑块 (弧度，范围符合关节限位)
ax_f = plt.axes((0.15, 0.12, 0.3, 0.03))
ax_h = plt.axes((0.15, 0.07, 0.3, 0.03))
ax_a = plt.axes((0.55, 0.12, 0.3, 0.03))
ax_b = plt.axes((0.55, 0.07, 0.3, 0.03))

s_f = Slider(ax_f, 'F_body扭转 (rad)', -1.57, 1.57, valinit=0.0, valstep=0.01)
s_h = Slider(ax_h, 'H_body扭转 (rad)', -1.57, 1.57, valinit=0.0, valstep=0.01)
s_a = Slider(ax_a, 'F_spine1 (rad)', -0.6, 0.6, valinit=0.0, valstep=0.01)
s_b = Slider(ax_b, 'H_spine1 (rad)', -0.6, 0.6, valinit=0.0, valstep=0.01)

def update(val):
    # 控制取反
    theta_f = -s_f.val
    theta_h =  s_h.val
    alpha   = -s_a.val
    beta    = -s_b.val
    draw_spine(ax, theta_f, theta_h, alpha, beta)
    fig.canvas.draw_idle()

s_f.on_changed(update)
s_h.on_changed(update)
s_a.on_changed(update)
s_b.on_changed(update)

plt.show()