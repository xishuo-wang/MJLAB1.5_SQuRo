import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brute
from matplotlib.widgets import Slider, Button


# =================================================
# 参数配置
SPINE_LEN    = 0.039
BODY_HALF_X  = 0.019
BODY_HALF_Y  = 0.025
BODY_HALF_Z  = 0.025
BASE_RADIUS  = 0.01

X_vec = np.array([1., 0., 0.])
Y_vec = np.array([0., 1., 0.])
Z_vec = np.array([0., 0., 1.])


# =================================================
# 字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =================================================
# 辅助函数
def rodrigues(v, k, angle):
    v, k = np.asarray(v), np.asarray(k)
    return v * np.cos(angle) + np.cross(k, v) * np.sin(angle) + k * np.dot(k, v) * (1 - np.cos(angle))

def rot_matrix_axis_angle(axis, angle):
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# =================================================
# 正运动学
def forward_kinematics_full(alpha, beta, theta_f, theta_h):
    phi = (theta_f + theta_h) / 2.0
    axis_alpha = rodrigues(Z_vec, X_vec, phi)
    axis_beta  = rodrigues(Y_vec, X_vec, phi)

    f_link_dir = rodrigues(X_vec, axis_alpha, alpha)
    f_center = (SPINE_LEN + BODY_HALF_X) * f_link_dir
    R_link_f = rot_matrix_axis_angle(axis_alpha, alpha)
    f_z = R_link_f @ Z_vec
    R_twist_f = rot_matrix_axis_angle(f_link_dir, theta_f)
    f_z = R_twist_f @ f_z
    f_z /= np.linalg.norm(f_z)
    f_y = np.cross(f_z, f_link_dir)
    f_y /= np.linalg.norm(f_y)

    h_link_dir = rodrigues(-X_vec, axis_beta, beta)
    h_center = (SPINE_LEN + BODY_HALF_X) * h_link_dir
    R_link_h = rot_matrix_axis_angle(axis_beta, beta)
    h_z = R_link_h @ Z_vec
    R_twist_h = rot_matrix_axis_angle(h_link_dir, theta_h)
    h_z = R_twist_h @ h_z
    h_z /= np.linalg.norm(h_z)
    h_y = np.cross(h_z, h_link_dir)
    h_y /= np.linalg.norm(h_y)

    return f_link_dir, f_center, f_y, f_z, h_link_dir, h_center, h_y, h_z


# =================================================
# 优化目标：Z轴平行
def objective_z_parallel(twist, alpha, beta):
    theta_f, theta_h = twist
    _, _, _, f_z, _, _, _, h_z = forward_kinematics_full(alpha, beta, theta_f, theta_h)
    dot = np.dot(f_z, h_z)
    return (1.0 - dot) ** 2


# =================================================
# 寻找最优扭转角
def find_optimal_twist(alpha, beta):
    bounds_brute = (slice(-1.57, 1.57, 0.1), slice(-1.57, 1.57, 0.1))
    res_brute = brute(lambda p: objective_z_parallel(p, alpha, beta), bounds_brute,
                      full_output=True, finish=None)
    x0 = res_brute[0]
    bounds_opt = [(-1.57, 1.57), (-1.57, 1.57)]
    res = minimize(objective_z_parallel, x0, args=(alpha, beta), bounds=bounds_opt,
                   method='L-BFGS-B', options={'ftol': 1e-12, 'gtol': 1e-12})
    if not res.success or res.fun > 1e-6:
        guess_f = -0.8
        guess_h = -0.7 * (abs(alpha) / 0.6) if abs(alpha) > 1e-9 else 0.0
        res2 = minimize(objective_z_parallel, [guess_f, guess_h], args=(alpha, beta),
                        bounds=bounds_opt, method='L-BFGS-B')
        if res2.fun < res.fun:
            res = res2
    return res.x[0], res.x[1], res.fun


# =================================================
# 绘图
def draw_spine(ax, alpha, beta, theta_f, theta_h):
    (f_link_dir, c_f, y_f, f_z,
     h_link_dir, c_h, y_h, h_z) = forward_kinematics_full(alpha, beta, theta_f, theta_h)

    # 弯曲角
    rear_forward = -h_link_dir
    dot_fwd = np.clip(np.dot(f_link_dir, rear_forward), -1.0, 1.0)
    bend_angle_rad = np.arccos(dot_fwd)
    cmd_bend_rad = abs(alpha)
    increase_rad = bend_angle_rad - cmd_bend_rad
    percent = (increase_rad / cmd_bend_rad * 100) if cmd_bend_rad > 1e-6 else 0.0

    # 圆心与半径
    cross = np.cross(y_f, y_h)
    cross_norm = np.linalg.norm(cross)
    radius = 0.0
    center = (c_f + c_h) / 2
    y_parallel = (cross_norm < 1e-6)
    if not y_parallel:
        d1d2 = np.dot(y_f, y_h)
        denom = 1 - d1d2**2
        if abs(denom) > 1e-12:
            A = np.array([[1, -d1d2], [-d1d2, 1]])
            b = np.array([np.dot(c_h - c_f, y_f), -np.dot(c_h - c_f, y_h)])
            try:
                ts = np.linalg.solve(A, b)
                p_f = c_f + ts[0] * y_f
                p_h = c_h + ts[1] * y_h
                center = (p_f + p_h) / 2
                radius = np.linalg.norm(c_f - center)
            except np.linalg.LinAlgError:
                pass

    ax.clear()
    ax.set_xlim(-0.12, 0.12)
    ax.set_ylim(-0.12, 0.12)
    ax.set_zlim(-0.12, 0.12)
    ax.set_xlabel('X (前)')
    ax.set_ylabel('Y (左)')
    ax.set_zlabel('Z (上)')
    ax.set_box_aspect((1,1,1))
    ax.grid(True, linestyle=':', alpha=0.3)

    # baselink 球
    u = np.linspace(0, 2*np.pi, 16)
    v = np.linspace(0, np.pi, 16)
    xs = BASE_RADIUS * np.outer(np.cos(u), np.sin(v))
    ys = BASE_RADIUS * np.outer(np.sin(u), np.sin(v))
    zs = BASE_RADIUS * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='lightgray', alpha=0.5)

    # 连杆
    f_conn = SPINE_LEN * f_link_dir
    h_conn = SPINE_LEN * h_link_dir
    ax.plot([0, f_conn[0]], [0, f_conn[1]], [0, f_conn[2]], color='#E06060', lw=3, label='F_spine1 Link')
    ax.plot([0, h_conn[0]], [0, h_conn[1]], [0, h_conn[2]], color='#7090C0', lw=3, label='H_spine1 Link')

    # 身体方块
    def draw_rect_prism(ax, center, y_axis, z_axis, color):
        x_axis = center / np.linalg.norm(center)
        z_axis = z_axis / np.linalg.norm(z_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        corners = []
        for dx in (-BODY_HALF_X, BODY_HALF_X):
            for dy in (-BODY_HALF_Y, BODY_HALF_Y):
                for dz in (-BODY_HALF_Z, BODY_HALF_Z):
                    corners.append(center + dx*x_axis + dy*y_axis + dz*z_axis)
        edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
        for i,j in edges:
            ax.plot([corners[i][0], corners[j][0]],
                    [corners[i][1], corners[j][1]],
                    [corners[i][2], corners[j][2]], color=color, lw=1.5)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        faces = [[0,2,6,4],[1,5,7,3],[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6]]
        for face in faces:
            poly = np.array([corners[idx] for idx in face])
            ax.add_collection3d(Poly3DCollection([poly], facecolors=color, linewidths=0, alpha=0.15))

    draw_rect_prism(ax, c_f, y_f, f_z, '#70B070')   # 柔和绿
    draw_rect_prism(ax, c_h, y_h, h_z, '#E0A060')   # 柔和橙

    # 局部坐标轴：Y 轴统一用青色，Z 轴统一用品红色
    ax.quiver(*c_f, *(y_f*0.05), color='cyan', lw=2, label='F_body Y')
    ax.quiver(*c_h, *(y_h*0.05), color='cyan', lw=2, label='H_body Y')
    ax.quiver(*c_f, *(f_z*0.05), color='magenta', lw=2, label='F_body Z')
    ax.quiver(*c_h, *(h_z*0.05), color='magenta', lw=2, label='H_body Z')

    # 圆
    if not y_parallel:
        ax.scatter(*center, color='purple', s=50, label='圆心')
        n = cross / cross_norm
        u_vec = y_f
        v_vec = np.cross(n, u_vec)
        v_vec /= np.linalg.norm(v_vec)
        theta_circ = np.linspace(0, 2*np.pi, 100)
        circ = center + radius * (np.cos(theta_circ)[:, None] * u_vec +
                                  np.sin(theta_circ)[:, None] * v_vec)
        ax.plot(circ[:,0], circ[:,1], circ[:,2], color='purple', lw=1, alpha=0.7)

    # 信息面板 (左侧)
    x_text, y_text = 0.02, 0.95
    z_dot = np.dot(f_z, h_z)
    converged = (abs(1.0 - z_dot) < 1e-6)
    lines = [
        f'转弯半径: {radius:.3f} m',
        f'命令侧摆叫: {cmd_bend_rad:.3f} rad',
        f'等效侧摆角: {bend_angle_rad:.3f} rad',
        f'增大: {increase_rad:+.3f} rad ({percent:+.1f}%)',
        f'Z点积: {z_dot:.6f}',
        '平面平行' if converged else f'未平行 (Δ={abs(1-z_dot):.1e})'
    ]
    for i, line in enumerate(lines):
        clr = 'navy'
        if '增大' in line: clr = 'red'
        if '平行' in line: clr = 'green' if converged else 'red'
        ax.text2D(x_text, y_text - i*0.06, line, transform=ax.transAxes,
                  fontsize=12, color=clr,
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2', edgecolor='gray'))


# ==================== 界面搭建 ====================
fig = plt.figure(figsize=(10,9))
ax = fig.add_subplot(111, projection='3d')

# 滑块配色（柔和）
c_alpha = '#E06060'   # 红
c_beta  = '#7090C0'   # 蓝
c_tf    = '#70B070'   # 绿
c_th    = '#E0A060'   # 橙

# 左组滑块 (x 坐标 0.15)
ax_alpha = plt.axes((0.15, 0.08, 0.2, 0.03))   # α 在上
ax_tf    = plt.axes((0.15, 0.03, 0.2, 0.03))   # θ_f 在下
s_alpha = Slider(ax_alpha, 'F_spine1(侧摆)', -0.6, 0.6, valinit=0.0, valstep=0.01, color=c_alpha)
s_tf    = Slider(ax_tf, 'F_body(前扭转)', -1.57, 1.57, valinit=0.0, valstep=0.01, color=c_tf)

# 右组滑块 (x 坐标 0.65)
ax_beta  = plt.axes((0.65, 0.08, 0.2, 0.03))   # β 在上
ax_th    = plt.axes((0.65, 0.03, 0.2, 0.03))   # θ_h 在下
s_beta = Slider(ax_beta,  'H_spine1(侧摆)', -0.6, 0.6, valinit=0.0, valstep=0.01, color=c_beta)
s_th   = Slider(ax_th, 'H_body(后扭转)', -1.57, 1.57, valinit=0.0, valstep=0.01, color=c_th)

# 按钮 (中央)
ax_btn = plt.axes((0.43, 0.03, 0.1, 0.09))
btn_opt = Button(ax_btn, '搜索', color='lightgoldenrodyellow', hovercolor='gold')


# =================================================
# 更新
def manual_update(val=None):
    alpha = s_alpha.val
    beta  = s_beta.val
    theta_f = s_tf.val
    theta_h = s_th.val
    draw_spine(ax, alpha, beta, theta_f, theta_h)
    fig.canvas.draw_idle()

def optimize_and_update(event):
    alpha = s_alpha.val
    beta  = s_beta.val
    theta_f_opt, theta_h_opt, _ = find_optimal_twist(alpha, beta)
    # 更新扭转滑块
    s_tf.eventson = False
    s_th.eventson = False
    s_tf.set_val(theta_f_opt)
    s_th.set_val(theta_h_opt)
    s_tf.eventson = True
    s_th.eventson = True
    manual_update()

s_alpha.on_changed(manual_update)
s_beta.on_changed(manual_update)
s_tf.on_changed(manual_update)
s_th.on_changed(manual_update)
btn_opt.on_clicked(optimize_and_update)

manual_update()
plt.show()