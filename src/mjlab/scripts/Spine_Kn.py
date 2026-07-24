import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def draw_spine(ax, theta_f=0.0, theta_h=0.0, alpha=0.0, beta=0.0):
    """
    绘制脊柱构型简图。
    theta_f: F_body_joint 绕X轴扭转角度 (rad)
    theta_h: H_body_joint 绕X轴扭转角度 (rad)
    alpha:   F_spine1_joint 绕其当前轴转角 (初始绕Z)
    beta:    H_spine1_joint 绕其当前轴转角 (初始绕Y)
    """
    # 长度定义
    L_body = 0.1   # F/H body 正方体边长的一半（用于画块）
    L_spine = 0.15 # spine link 长度
    r_base = 0.05  # baselink 球半径

    # 初始位置（世界坐标）
    base_pos = np.array([0.0, 0.0, 0.0])

    # 初始轴 (世界坐标系)
    X_axis = np.array([1.0, 0.0, 0.0])
    Y_axis = np.array([0.0, 1.0, 0.0])
    Z_axis = np.array([0.0, 0.0, 1.0])

    # --- 计算 baselink 的扭转角 (假定由前后扭转平均或单独处理) ---
    # 这里简化为 baselink 扭转角 = (theta_f + theta_h)/2，实际取决于关节连接方式
    phi = (theta_f + theta_h) / 2.0

    # 扭转后的 F_spine1 和 H_spine1 轴 (绕 X 旋转 phi)
    # alpha 轴初始为 Z，beta 轴初始为 Y
    def rot_x(vec, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        return R @ vec

    axis_alpha = rot_x(Z_axis, phi)   # F_spine1 旋转轴
    axis_beta  = rot_x(Y_axis, phi)   # H_spine1 旋转轴

    # --- 构建各连杆位置和朝向 ---
    # 从 base 出发，向前经过 F_spine1_joint 到 F_spine1_Link 再到 F_body_joint 到 F_body_Link
    # 向后经过 H_spine1_joint 到 H_spine1_Link 再到 H_body_joint 到 H_body_Link

    # F_spine1_joint 位置 (base 前方距离 spine_link_len/2)
    # 我们先假设 spine_link 一半在 base 前，一半在后？根据描述：
    # 顺序: F_body_Link -- F_body_joint -- F_spine1_Link -- F_spine1_joint -- baselink -- ...
    # 因此 F_spine1_Link 连接 F_body_joint 和 F_spine1_joint，方向朝前。
    # 我们设 F_spine1_joint 位于 base 前方距离 spine_link_len 处。
    f_spine1_joint_pos = base_pos + L_spine * X_axis   # 向前

    # F_spine1_joint 旋转 (绕 axis_alpha 转 alpha)
    # 旋转后，F_spine1_Link 的指向（初始也沿 X 轴）发生偏转。
    # 使用罗德里格斯公式旋转 X 轴
    def rodrigues(v, k, angle):
        return v * np.cos(angle) + np.cross(k, v) * np.sin(angle) + k * np.dot(k, v) * (1 - np.cos(angle))

    f_spine_link_dir = rodrigues(X_axis, axis_alpha, alpha)

    # F_body_joint 位置 (F_spine1_Link 的远端)
    f_body_joint_pos = f_spine1_joint_pos + L_spine * f_spine_link_dir

    # F_body_joint 旋转 (绕 X 轴扭转 theta_f)，但这里 F_body_Link 的朝向还要叠加之前的旋转
    # F_body_Link 的最终 X 轴 = 先绕 axis_alpha 转 alpha，再绕新 X 轴 (即当前的前向) 转 theta_f
    # 不过为了简单，我们只画出 F_body 块的位置和局部坐标系。
    # F_body 的局部 Z 轴: 先绕 axis_alpha 转 alpha 后的 Z 轴再绕前向转 theta_f
    # 我们直接计算 F_body 的局部坐标系：
    # 1. 基础旋转 R_base = 绕 axis_alpha 转 alpha
    # 2. R_body = R_base * 绕 body X 转 theta_f
    # 这里的 body X 就是 f_spine_link_dir
    # 使用 scipy？手动实现旋转矩阵。
    def rot_matrix_from_axis_angle(axis, angle):
        k = axis / np.linalg.norm(axis)
        K = np.array([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    R_alpha = rot_matrix_from_axis_angle(axis_alpha, alpha)
    # F_body 的局部 X 轴
    f_body_x = f_spine_link_dir
    # F_body 的局部 Z 轴 (初始为世界 Z 经过 R_alpha 旋转)
    f_body_z = R_alpha @ Z_axis
    # 再绕 f_body_x 扭转 theta_f
    R_twist_f = rot_matrix_from_axis_angle(f_body_x, theta_f)
    f_body_z = R_twist_f @ f_body_z
    f_body_y = np.cross(f_body_z, f_body_x)  # 右手系

    # 后侧同理，方向相反 (向后)
    h_spine1_joint_pos = base_pos - L_spine * X_axis

    # H_spine1_joint 旋转 (绕 axis_beta 转 beta)
    h_spine_link_dir = rodrigues(-X_axis, axis_beta, beta)  # 初始指向 -X

    h_body_joint_pos = h_spine1_joint_pos + L_spine * h_spine_link_dir

    R_beta = rot_matrix_from_axis_angle(axis_beta, beta)
    h_body_x = h_spine_link_dir
    h_body_z = R_beta @ Z_axis
    R_twist_h = rot_matrix_from_axis_angle(h_body_x, theta_h)
    h_body_z = R_twist_h @ h_body_z
    h_body_y = np.cross(h_body_z, h_body_x)

    # ---- 绘制 ----
    ax.clear()
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.set_xlabel('X (前)')
    ax.set_ylabel('Y (左)')
    ax.set_zlabel('Z (上)')

    # 绘制 baselink 球
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = base_pos[0] + r_base * np.outer(np.cos(u), np.sin(v))
    y = base_pos[1] + r_base * np.outer(np.sin(u), np.sin(v))
    z = base_pos[2] + r_base * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='gray', alpha=0.3)

    # 绘制 spine links (细线)
    ax.plot([f_spine1_joint_pos[0], f_body_joint_pos[0]],
            [f_spine1_joint_pos[1], f_body_joint_pos[1]],
            [f_spine1_joint_pos[2], f_body_joint_pos[2]], 'r-', lw=2, label='F_spine1 Link')
    ax.plot([h_spine1_joint_pos[0], h_body_joint_pos[0]],
            [h_spine1_joint_pos[1], h_body_joint_pos[1]],
            [h_spine1_joint_pos[2], h_body_joint_pos[2]], 'b-', lw=2, label='H_spine1 Link')

    # 绘制 F_body 和 H_body 块 (用小立方体表示)
    def draw_cube(ax, center, x_axis, y_axis, z_axis, size, color='green'):
        corners = []
        for dx in (-size, size):
            for dy in (-size, size):
                for dz in (-size, size):
                    corner = center + dx * x_axis + dy * y_axis + dz * z_axis
                    corners.append(corner)
        corners = np.array(corners)
        # 只画棱边
        for i, c1 in enumerate(corners):
            for j, c2 in enumerate(corners):
                if i < j and np.sum((c1 - c2) ** 2) < (2 * size + 1e-5) ** 2:
                    diff = c1 - c2
                    if np.sum(np.abs(diff) < 1e-6) == 2:  # 两个坐标相等，表示棱边
                        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], color=color, lw=2)

    draw_cube(ax, f_body_joint_pos, f_body_x, f_body_y, f_body_z, L_body, 'green')
    draw_cube(ax, h_body_joint_pos, h_body_x, h_body_y, h_body_z, L_body, 'orange')

    # 标注关节位置
    ax.scatter(*f_spine1_joint_pos, c='k', marker='o')
    ax.scatter(*h_spine1_joint_pos, c='k', marker='o')
    ax.scatter(*f_body_joint_pos, c='k', marker='o')
    ax.scatter(*h_body_joint_pos, c='k', marker='o')

    # 绘制关节轴 (初始和当前)
    # F_spine1 轴
    ax.quiver(*f_spine1_joint_pos, *axis_alpha * 0.1, color='cyan', label='F_spine1 轴')
    # H_spine1 轴
    ax.quiver(*h_spine1_joint_pos, *axis_beta * 0.1, color='magenta', label='H_spine1 轴')
    # 扭转轴 (X) 示意
    ax.quiver(0, 0, 0, 0.3, 0, 0, color='black', alpha=0.5, label='X 扭转轴')

    ax.legend(loc='upper right')

def update(val):
    theta_f = np.deg2rad(slider_f.val)
    theta_h = np.deg2rad(slider_h.val)
    alpha = np.deg2rad(slider_alpha.val)
    beta = np.deg2rad(slider_beta.val)
    draw_spine(ax, theta_f, theta_h, alpha, beta)
    fig.canvas.draw_idle()

# 创建交互界面
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 默认初始姿态：所有角度为0
draw_spine(ax, 0, 0, 0, 0)

# 添加滑块
from matplotlib.widgets import Slider

axcolor = 'lightgoldenrodyellow'
ax_f = plt.axes([0.1, 0.15, 0.35, 0.03], facecolor=axcolor) # type: ignore
ax_h = plt.axes([0.1, 0.10, 0.35, 0.03], facecolor=axcolor) # type: ignore
ax_alpha = plt.axes([0.6, 0.15, 0.35, 0.03], facecolor=axcolor) # type: ignore
ax_beta = plt.axes([0.6, 0.10, 0.35, 0.03], facecolor=axcolor) # type: ignore

slider_f = Slider(ax_f, 'F_body twist (deg)', -180, 180, valinit=0, valstep=1)
slider_h = Slider(ax_h, 'H_body twist (deg)', -180, 180, valinit=0, valstep=1)
slider_alpha = Slider(ax_alpha, 'F_spine1 (deg)', -90, 90, valinit=0, valstep=1)
slider_beta = Slider(ax_beta, 'H_spine1 (deg)', -90, 90, valinit=0, valstep=1)

slider_f.on_changed(update)
slider_h.on_changed(update)
slider_alpha.on_changed(update)
slider_beta.on_changed(update)

plt.show()