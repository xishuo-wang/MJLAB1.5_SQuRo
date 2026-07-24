import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.widgets import Slider

# 字体设置：优先使用 SimHei 显示中文，DejaVu Sans 确保负号等符号正常
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False   # 明确允许使用 ASCII 负号

def rodrigues(v, k, angle):
    v, k = np.asarray(v), np.asarray(k)
    return v * np.cos(angle) + np.cross(k, v) * np.sin(angle) + k * np.dot(k, v) * (1 - np.cos(angle))

def rot_matrix_axis_angle(axis, angle):
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def draw_spine(ax, theta_f=0.0, theta_h=0.0, alpha=0.0, beta=0.0, show_circle=True):
    spine_len = 0.039          # 连杆长度
    base_radius = 0.01         # baselink 球半径
    body_half_x = 0.019        # 身体方块半长
    body_half_y = 0.025
    body_half_z = 0.025

    X = np.array([1.,0.,0.])
    Y = np.array([0.,1.,0.])
    Z = np.array([0.,0.,1.])

    phi = (theta_f + theta_h) / 2.0   # baselink 平均扭转

    # 关节轴
    axis_alpha = rodrigues(Z, X, phi)
    axis_beta  = rodrigues(Y, X, phi)

    # ---- 前躯干 ----
    f_link_dir = rodrigues(X, axis_alpha, alpha)
    f_connect = spine_len * f_link_dir
    f_body_center = f_connect + body_half_x * f_link_dir

    R_link_f = rot_matrix_axis_angle(axis_alpha, alpha)
    f_body_x = f_link_dir
    f_body_z = R_link_f @ Z
    R_twist_f = rot_matrix_axis_angle(f_body_x, theta_f)
    f_body_z = R_twist_f @ f_body_z
    f_body_y = np.cross(f_body_z, f_body_x)

    # ---- 后躯干 ----
    h_link_dir = rodrigues(-X, axis_beta, beta)
    h_connect = spine_len * h_link_dir
    h_body_center = h_connect + body_half_x * h_link_dir

    R_link_h = rot_matrix_axis_angle(axis_beta, beta)
    h_body_x = h_link_dir
    h_body_z = R_link_h @ Z
    R_twist_h = rot_matrix_axis_angle(h_body_x, theta_h)
    h_body_z = R_twist_h @ h_body_z
    h_body_y = np.cross(h_body_z, h_body_x)

    # ---- 绘图 ----
    ax.clear()
    ax.set_xlim(-0.15, 0.15)
    ax.set_ylim(-0.15, 0.15)
    ax.set_zlim(-0.15, 0.15)
    ax.set_xlabel('X (前)')
    ax.set_ylabel('Y (左)')
    ax.set_zlabel('Z (上)')
    ax.set_box_aspect((1,1,1))

    # baselink 球
    u = np.linspace(0, 2*np.pi, 16)
    v = np.linspace(0, np.pi, 16)
    xs = base_radius * np.outer(np.cos(u), np.sin(v))
    ys = base_radius * np.outer(np.sin(u), np.sin(v))
    zs = base_radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='gray', alpha=0.4)

    # 连杆
    ax.plot([0, f_connect[0]], [0, f_connect[1]], [0, f_connect[2]], 'r-', lw=3)
    ax.plot([0, h_connect[0]], [0, h_connect[1]], [0, h_connect[2]], 'b-', lw=3)

    # 身体方块
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

    # 局部 Y 轴
    ax.quiver(*f_body_center, *(f_body_y*0.05), color='lime', lw=2, label='F_body Y轴')
    ax.quiver(*h_body_center, *(h_body_y*0.05), color='darkorange', lw=2, label='H_body Y轴')

    # 期望半径计算与显示
    if show_circle:
        p1, d1 = f_body_center, f_body_y
        p2, d2 = h_body_center, h_body_y
        d1d2 = np.dot(d1, d2)
        denom = 1 - d1d2**2
        if abs(denom) > 1e-6:
            t = np.dot(p2 - p1, d1 - d1d2*d2) / denom
            s = np.dot(p1 - p2, d2 - d1d2*d1) / denom
            point1 = p1 + t * d1
            point2 = p2 + s * d2
            center_circle = (point1 + point2) / 2
            radius = np.linalg.norm(f_body_center - center_circle)
            ax.scatter(*center_circle, color='purple', s=50, label='圆心')
            # 画圆
            n = np.cross(d1, d2)
            if np.linalg.norm(n) > 1e-6:
                n = n / np.linalg.norm(n)
                u = d1
                v = np.cross(n, u)
                if np.linalg.norm(v) < 1e-6:
                    v = np.cross(n, d2)
                v = v / np.linalg.norm(v)
                theta_circle = np.linspace(0, 2*np.pi, 100)
                circle_pts = []
                for ang in theta_circle:
                    pt = center_circle + radius * (np.cos(ang)*u + np.sin(ang)*v)
                    circle_pts.append(pt)
                circle_pts = np.array(circle_pts)
                ax.plot(circle_pts[:,0], circle_pts[:,1], circle_pts[:,2], color='purple', lw=1, alpha=0.7)
            # 在图形左上角显示半径文本
            ax.text2D(0.02, 0.90, f'期望半径: {radius:.3f} m', transform=ax.transAxes,
                      fontsize=10, color='navy', bbox=dict(facecolor='white', alpha=0.8))
        else:
            ax.text2D(0.02, 0.90, '期望半径: 无穷大 (直线)', transform=ax.transAxes,
                      fontsize=10, color='navy', bbox=dict(facecolor='white', alpha=0.8))

    ax.text2D(0.02, 0.98, '世界: X前 Y左 Z上', transform=ax.transAxes,
              fontsize=9, color='navy', bbox=dict(facecolor='white', alpha=0.8))
    ax.legend(loc='upper right', fontsize=7)

# 交互界面
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')

# 同步弯曲滑块
ax_theta = plt.axes((0.25, 0.15, 0.5, 0.03))
s_theta = Slider(ax_theta, 'α = β (rad)', -0.6, 0.6, valinit=0.0, valstep=0.01)

# 扭转自动显示
ax_twist = plt.axes((0.25, 0.10, 0.5, 0.03))
s_twist = Slider(ax_twist, '扭转角度 (自动 = α/2)', -0.3, 0.3, valinit=0.0, valstep=0.01)
s_twist.set_active(False)  # 只读显示

def update(val):
    theta = s_theta.val
    twist = theta / 2.0
    s_twist.set_val(twist)  # 更新显示
    draw_spine(ax, theta_f=-twist, theta_h=twist, alpha=theta, beta=theta, show_circle=True)
    fig.canvas.draw_idle()

s_theta.on_changed(update)
update(0)

plt.show()