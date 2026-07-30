import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from matplotlib.patches import Circle

# ========== 参数 ==========
Rmin = 0.10          # 最小转弯半径 (m)
pole_radius = 0.005  # 杆的显示半径 (m)
X = 0.30             # 杆间距 (m)，需 >= 2*Rmin
num_periods = 3      # 周期数

# 平滑转弯的回旋线长度 (m)，必须 <= (pi/2)*Rmin ≈ 0.157
Lc = 0.04            # 过渡段长度，可根据需要调整

# 速度参数
v_max = 0.1          # 直线最大速度 (m/s)
v_min = 0.025        # 最大曲率时的最小速度 (m/s)

# 数值离散点数（每段）
steps = 50

# ========== 辅助函数 ==========
def clothoid_segment(start_xy, start_theta, kappa0, kappa1, L, steps=50):
    """
    生成曲率从 kappa0 线性变化到 kappa1 的回旋线段
    返回: x, y, 终点切线角, 曲率序列
    """
    s = np.linspace(0, L, steps)
    alpha = (kappa1 - kappa0) / L
    theta = start_theta + kappa0 * s + 0.5 * alpha * s**2
    # 数值积分求坐标
    x = start_xy[0] + cumulative_trapezoid(np.cos(theta), s, initial=0)
    y = start_xy[1] + cumulative_trapezoid(np.sin(theta), s, initial=0)
    kappa = kappa0 + alpha * s
    return x, y, theta[-1], kappa

def smooth_turn(start_xy, start_theta, clockwise, Rmin, Lc, steps=50):
    """
    生成一个 90° 平滑转弯（回旋线 + 定曲率圆弧 + 回旋线）
    转弯方向: clockwise=True 为顺时针
    返回: x, y, 终点切线角, 曲率序列
    """
    k_max = 1.0 / Rmin
    sign = -1 if clockwise else 1   # 曲率正负约定：逆时针为正
    k_peak = sign * k_max

    # 第1段回旋线: 0 -> k_peak
    x1, y1, theta1, kap1 = clothoid_segment(start_xy, start_theta, 0, k_peak, Lc, steps)
    # 圆弧段：剩余角度 = pi/2 - Lc/Rmin (因为回旋线总转角 = Lc/(2Rmin) * 2 = Lc/Rmin)
    theta_arc = np.pi/2 - Lc / Rmin
    if theta_arc < 0:
        raise ValueError("Lc too large, arc angle negative. Reduce Lc.")
    La = Rmin * theta_arc
    # 圆弧中心坐标
    # 从回旋线终点出发，半径方向垂直于切线方向
    # 顺时针时，圆心在切线右侧；逆时针时在左侧
    # 切线方向单位向量: (cos(theta1), sin(theta1))
    # 顺时针: 圆心 = 起点 + Rmin * (sin(theta1), -cos(theta1))   # 右侧法向量
    # 逆时针: 圆心 = 起点 + Rmin * (-sin(theta1), cos(theta1))  # 左侧法向量
    if clockwise:
        center_x = x1[-1] + Rmin * np.sin(theta1)
        center_y = y1[-1] - Rmin * np.cos(theta1)
    else:
        center_x = x1[-1] - Rmin * np.sin(theta1)
        center_y = y1[-1] + Rmin * np.cos(theta1)

    # 圆弧起止角度
    v_start = (x1[-1] - center_x, y1[-1] - center_y)
    ang_start = np.arctan2(v_start[1], v_start[0])
    if clockwise:
        ang_end = ang_start - theta_arc
    else:
        ang_end = ang_start + theta_arc

    # 圆弧离散点（至少要2个点，避免 linspace 出错）
    n_arc = max(2, int(steps * theta_arc / (np.pi/2)))
    theta_arr = np.linspace(ang_start, ang_end, n_arc)
    arc_x = center_x + Rmin * np.cos(theta_arr)
    arc_y = center_y + Rmin * np.sin(theta_arr)
    # 圆弧段曲率恒定
    kap_arc = np.full_like(theta_arr, k_peak)

    # 圆弧终点切线方向
    theta2 = theta1 + sign * theta_arc

    # 第2段回旋线: k_peak -> 0
    x3, y3, theta_end, kap3 = clothoid_segment((arc_x[-1], arc_y[-1]), theta2, k_peak, 0, Lc, steps)

    # 拼接坐标和曲率（去除重复连接点）
    x = np.concatenate([x1, arc_x[1:], x3])
    y = np.concatenate([y1, arc_y[1:], y3])
    kappa = np.concatenate([kap1, kap_arc[1:], kap3])

    return x, y, theta_end, kappa

def generate_one_period(start_xy, start_theta, X, Rmin, Lc):
    """
    生成一个周期的平滑路径（绕过两根杆）
    起点状态: (x, y, 切线角)
    返回: 路径点 x, y, 曲率序列, 终点位姿 (x_end, y_end, theta_end)
    """
    x_all, y_all, kap_all = [], [], []
    cur_x, cur_y = start_xy
    cur_theta = start_theta

    # 1. 顺时针 90° 转弯 (向右 -> 向下)
    x, y, cur_theta, kap = smooth_turn((cur_x, cur_y), cur_theta, clockwise=True, Rmin=Rmin, Lc=Lc)
    x_all.extend(x); y_all.extend(y); kap_all.extend(kap)
    # 2. 逆时针 90° 转弯 (向下 -> 向右)
    x, y, cur_theta, kap = smooth_turn((x_all[-1], y_all[-1]), cur_theta, clockwise=False, Rmin=Rmin, Lc=Lc)
    x_all.extend(x); y_all.extend(y); kap_all.extend(kap)
    # 3. 直线段 向右移动到 x = start_xy[0] + X （注意全局坐标）
    target_x = start_xy[0] + X
    if target_x - x_all[-1] > 1e-9:
        x_line = np.linspace(x_all[-1], target_x, max(2, int((target_x - x_all[-1]) / 0.005)))
        y_line = np.full_like(x_line, y_all[-1])
        x_all.extend(x_line); y_all.extend(y_line)
        kap_all.extend([0.0] * len(x_line))
    # 4. 逆时针 90° 转弯 (向右 -> 向上)
    x, y, cur_theta, kap = smooth_turn((x_all[-1], y_all[-1]), cur_theta, clockwise=False, Rmin=Rmin, Lc=Lc)
    x_all.extend(x); y_all.extend(y); kap_all.extend(kap)
    # 5. 顺时针 90° 转弯 (向上 -> 向右)
    x, y, cur_theta, kap = smooth_turn((x_all[-1], y_all[-1]), cur_theta, clockwise=True, Rmin=Rmin, Lc=Lc)
    x_all.extend(x); y_all.extend(y); kap_all.extend(kap)
    # 6. 直线段 向右移动到 start_xy[0] + 2*X
    target_x2 = start_xy[0] + 2*X
    if target_x2 - x_all[-1] > 1e-9:
        x_line = np.linspace(x_all[-1], target_x2, max(2, int((target_x2 - x_all[-1]) / 0.005)))
        y_line = np.full_like(x_line, y_all[-1])
        x_all.extend(x_line); y_all.extend(y_line)
        kap_all.extend([0.0] * len(x_line))

    return (np.array(x_all), np.array(y_all), np.array(kap_all),
            (x_all[-1], y_all[-1], cur_theta))

def generate_path(num_periods, X, Rmin, Lc):
    """生成多个周期的全局路径，同时返回曲率"""
    x_global, y_global, kap_global = [0.0], [0.0], [0.0]
    cur_x, cur_y = 0.0, 0.0
    cur_theta = 0.0
    for i in range(num_periods):
        xs, ys, kaps, end_pose = generate_one_period((cur_x, cur_y), cur_theta, X, Rmin, Lc)
        if i == 0:
            x_global = xs
            y_global = ys
            kap_global = kaps
        else:
            x_global = np.concatenate([x_global, xs[1:]])
            y_global = np.concatenate([y_global, ys[1:]])
            kap_global = np.concatenate([kap_global, kaps[1:]])
        cur_x, cur_y, cur_theta = end_pose
    return x_global, y_global, kap_global

def compute_min_distance_to_poles(path_x, path_y, pole_x_positions, pole_y):
    """计算所有杆到路径的最近距离，返回最小值"""
    min_dist = np.inf
    for px in pole_x_positions:
        dists = np.sqrt((path_x - px)**2 + (path_y - pole_y)**2)
        min_dist = min(min_dist, np.min(dists))
    return min_dist

# ========== 主程序 ==========
# 1. 先用临时杆坐标 y = -Rmin 生成路径，计算最小距离
x_path, y_path, kappa_path = generate_path(num_periods, X, Rmin, Lc)
pole_x_coords = np.arange(num_periods * 2 + 1) * X  # 杆的x坐标 (包括最后一个周期后的杆)
d0 = compute_min_distance_to_poles(x_path, y_path, pole_x_coords, pole_y=-Rmin)
# 计算所需的杆偏移：使最小距离恰好为 Rmin
y_pole = 2 * Rmin - d0   # 杆的新 y 坐标 = -y_pole
print(f"初始最小距离 (杆在 y=-{Rmin}): {d0:.4f} m")
print(f"调整后杆的 y 坐标: -{y_pole:.4f} m (向下移动 {y_pole - Rmin:.4f} m)")

# 2. 计算弧长和速度曲线
dx = np.diff(x_path)
dy = np.diff(y_path)
ds = np.sqrt(dx**2 + dy**2)
s = np.insert(np.cumsum(ds), 0, 0)  # 弧长

k_max = 1.0 / Rmin
# 速度与曲率绝对值成线性反比：曲率越大速度越小
v = v_min + (v_max - v_min) * (k_max - np.abs(kappa_path)) / k_max
v = np.clip(v, v_min, v_max)   # 数值安全

# 3. 绘图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图1：轨迹与杆
ax1 = axes[0, 0]
# 绘制路径
ax1.plot(x_path, y_path, 'b-', linewidth=2, label='Smooth reference path')
# 标记起点
ax1.plot(0, 0, 'go', markersize=10, label='Start')
# 画杆（调整后的位置）
for px in pole_x_coords:
    ax1.add_patch(Circle((px, -y_pole), pole_radius, color='red', alpha=0.6))
ax1.plot(pole_x_coords, np.full_like(pole_x_coords, -y_pole), 'rx', label='Pole centers')

# 辅助线
ax1.axhline(0, color='gray', linestyle=':')
ax1.axhline(-y_pole, color='red', linestyle='--', alpha=0.3)
ax1.axhline(-y_pole - Rmin, color='blue', linestyle='--', alpha=0.3, label=f'y = -y_pole - Rmin')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Trajectory with smooth curvature transitions')
ax1.axis('equal')
ax1.grid(True)
ax1.legend(fontsize=8)

# 图2：曲率随弧长变化
ax2 = axes[0, 1]
ax2.plot(s, kappa_path, 'b-', linewidth=1.5)
ax2.axhline(k_max, color='red', linestyle='--', label=f'+k_max = {k_max:.1f}')
ax2.axhline(-k_max, color='red', linestyle='--', label=f'-k_max = {-k_max:.1f}')
ax2.set_xlabel('Arc length (m)')
ax2.set_ylabel('Curvature (1/m)')
ax2.set_title('Curvature profile')
ax2.grid(True)
ax2.legend()

# 图3：速度随弧长变化
ax3 = axes[1, 0]
ax3.plot(s, v, 'g-', linewidth=1.5)
ax3.axhline(v_max, color='gray', linestyle='--', label=f'v_max = {v_max}')
ax3.axhline(v_min, color='gray', linestyle='--', label=f'v_min = {v_min}')
ax3.set_xlabel('Arc length (m)')
ax3.set_ylabel('Velocity (m/s)')
ax3.set_title('Speed profile (curvature-adaptive)')
ax3.grid(True)
ax3.legend()

# 图4：曲率与速度的关系（同一弧长）
ax4 = axes[1, 1]
ax4.plot(s, np.abs(kappa_path), 'b-', label='|curvature|')
ax4_twin = ax4.twinx()
ax4_twin.plot(s, v, 'g-', label='velocity')
ax4.set_xlabel('Arc length (m)')
ax4.set_ylabel('|Curvature| (1/m)', color='b')
ax4_twin.set_ylabel('Velocity (m/s)', color='g')
ax4.set_title('Curvature and velocity along the path')
ax4.grid(True)
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.show()