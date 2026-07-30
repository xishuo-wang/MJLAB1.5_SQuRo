import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from matplotlib.patches import Circle

# ========== 参数 ==========
Rmin = 0.10          # 最小转弯半径 (m)
pole_radius = 0.005  # 杆的显示半径 (m)
X = 0.20             # 杆间距，可设为 0.2, 0.3 等
num_periods = 3
v_max = 0.1          # 最大速度 (m/s)
v_min = 0.025        # 最大曲率时的最小速度 (m/s)
steps = 100

# 期望过渡段长度（空间不足时会自动缩减）
Ls_des = 0.04
L_rev_des = 0.10

# ========== 基础几何函数 ==========
def sine_rise(start_xy, start_theta, k_target, Ls, steps=50):
    s = np.linspace(0, Ls, steps)
    kappa = k_target * np.sin(np.pi * s / (2 * Ls))
    theta = start_theta + (2 * Ls / np.pi) * k_target * (1 - np.cos(np.pi * s / (2 * Ls)))
    x = start_xy[0] + cumulative_trapezoid(np.cos(theta), s, initial=0)
    y = start_xy[1] + cumulative_trapezoid(np.sin(theta), s, initial=0)
    return x, y, theta[-1], kappa

def sine_fall(start_xy, start_theta, k_start, Ls, steps=50):
    s = np.linspace(0, Ls, steps)
    kappa = k_start * np.cos(np.pi * s / (2 * Ls))
    theta = start_theta + (2 * Ls / np.pi) * k_start * np.sin(np.pi * s / (2 * Ls))
    x = start_xy[0] + cumulative_trapezoid(np.cos(theta), s, initial=0)
    y = start_xy[1] + cumulative_trapezoid(np.sin(theta), s, initial=0)
    return x, y, theta[-1], kappa

def pure_arc(start_xy, start_theta, clockwise, Rmin, angle, steps=50):
    sign = -1 if clockwise else 1
    if clockwise:
        cx = start_xy[0] + Rmin * np.sin(start_theta)
        cy = start_xy[1] - Rmin * np.cos(start_theta)
    else:
        cx = start_xy[0] - Rmin * np.sin(start_theta)
        cy = start_xy[1] + Rmin * np.cos(start_theta)
    v_start = np.array([start_xy[0] - cx, start_xy[1] - cy])
    ang_start = np.arctan2(v_start[1], v_start[0])
    ang_end = ang_start - angle if clockwise else ang_start + angle
    n = max(2, int(steps * angle / (np.pi/2)))
    th = np.linspace(ang_start, ang_end, n)
    x = cx + Rmin * np.cos(th)
    y = cy + Rmin * np.sin(th)
    kappa = np.full_like(th, sign / Rmin)
    theta_end = start_theta + sign * angle
    return x, y, theta_end, kappa

def reverse_transition(start_xy, start_theta, k_start, L_rev, steps=80):
    L = L_rev
    s = np.linspace(0, L, steps)
    t = s / L
    H = 6*t**5 - 15*t**4 + 10*t**3
    kappa = k_start * (1 - 2*H)
    theta = start_theta + cumulative_trapezoid(kappa, s, initial=0)
    x = start_xy[0] + cumulative_trapezoid(np.cos(theta), s, initial=0)
    y = start_xy[1] + cumulative_trapezoid(np.sin(theta), s, initial=0)
    return x, y, theta[-1], kappa

# ========== 复合转弯函数（根据曲率符号自动选择）==========
def turn_from_straight(start_xy, start_theta, clockwise, Rmin, Ls):
    """直线 → 圆弧（90°）"""
    k_max = 1.0 / Rmin
    sign = -1 if clockwise else 1
    k_peak = sign * k_max
    x1, y1, th1, kap1 = sine_rise(start_xy, start_theta, k_peak, Ls)
    theta_sine = 2 * k_max * Ls / np.pi
    theta_arc = np.pi/2 - theta_sine
    x2, y2, th2, kap2 = pure_arc((x1[-1], y1[-1]), th1, clockwise, Rmin, theta_arc)
    x = np.concatenate([x1, x2[1:]])
    y = np.concatenate([y1, y2[1:]])
    kappa = np.concatenate([kap1, kap2[1:]])
    return x, y, th2, kappa

def turn_to_straight(start_xy, start_theta, clockwise, Rmin, Ls):
    """圆弧 → 直线（90°）"""
    k_max = 1.0 / Rmin
    sign = -1 if clockwise else 1
    k_peak = sign * k_max
    theta_sine = 2 * k_max * Ls / np.pi
    theta_arc = np.pi/2 - theta_sine
    x1, y1, th1, kap1 = pure_arc(start_xy, start_theta, clockwise, Rmin, theta_arc)
    x2, y2, th2, kap2 = sine_fall((x1[-1], y1[-1]), th1, k_peak, Ls)
    x = np.concatenate([x1, x2[1:]])
    y = np.concatenate([y1, y2[1:]])
    kappa = np.concatenate([kap1, kap2[1:]])
    return x, y, th2, kappa

def turn_reverse_arc(start_xy, start_theta, k_start, Rmin, L_rev):
    """异号圆弧连接（90°）: k_start → -k_start"""
    x1, y1, th1, kap1 = reverse_transition(start_xy, start_theta, k_start, L_rev)
    clockwise_after = (k_start > 0)
    x2, y2, th2, kap2 = pure_arc((x1[-1], y1[-1]), th1, clockwise_after, Rmin, np.pi/2)
    x = np.concatenate([x1, x2[1:]])
    y = np.concatenate([y1, y2[1:]])
    kappa = np.concatenate([kap1, kap2[1:]])
    return x, y, th2, kappa

# ========== 智能转弯函数（根据当前曲率与目标曲率自动选择）==========
def smart_turn(state, target_k, angle, Rmin, Ls, L_rev):
    """
    state: (x, y, theta, k_cur)
    target_k: 目标曲率（弧度长度后的恒定曲率，0表示直线）
    angle: 总转角（弧度）
    返回: x, y, theta_end, kappa
    """
    x, y, th, k_cur = state
    # 情况1：直线 ↔ 圆弧
    if k_cur == 0 and target_k != 0:
        clockwise = (target_k < 0)
        # 仅支持 90° 转弯，若 angle 不是 π/2 则需要调整，这里默认 90°
        return turn_from_straight((x, y), th, clockwise, Rmin, Ls)
    if k_cur != 0 and target_k == 0:
        clockwise = (k_cur < 0)
        return turn_to_straight((x, y), th, clockwise, Rmin, Ls)
    # 情况2：同号非零 → 纯圆弧（无平滑）
    if k_cur != 0 and target_k != 0 and np.sign(k_cur) == np.sign(target_k):
        clockwise = (k_cur < 0)
        return pure_arc((x, y), th, clockwise, Rmin, angle)
    # 情况3：异号非零 → 反向平滑过渡
    if k_cur != 0 and target_k != 0 and np.sign(k_cur) != np.sign(target_k):
        return turn_reverse_arc((x, y), th, k_cur, Rmin, L_rev)
    # 其他（如 target_k == 0 且 angle == 0）视为直线，返回空
    return np.array([x]), np.array([y]), th, np.array([k_cur])

# ========== 生成无直线路径（X=2Rmin，完全由圆弧组成）==========
def generate_path_no_straight(num_periods, X, Rmin, Ls, L_rev):
    # 初始状态：位于 (0,0)，切线向右 (0)，曲率 0
    state = (0.0, 0.0, 0.0, 0.0)
    x_all, y_all, kap_all = [0.0], [0.0], [0.0]
    
    for _ in range(num_periods):
        # 转弯1: 当前曲率 → -kmax (顺时针90°)
        xs, ys, th, kaps = smart_turn(state, -1.0/Rmin, np.pi/2, Rmin, Ls, L_rev)
        x_all.extend(xs[1:]); y_all.extend(ys[1:]); kap_all.extend(kaps[1:])
        state = (x_all[-1], y_all[-1], th, -1.0/Rmin)
        
        # 转弯2: -kmax → +kmax (逆时针90°，异号)
        xs, ys, th, kaps = smart_turn(state, 1.0/Rmin, np.pi/2, Rmin, Ls, L_rev)
        x_all.extend(xs[1:]); y_all.extend(ys[1:]); kap_all.extend(kaps[1:])
        state = (x_all[-1], y_all[-1], th, 1.0/Rmin)
        
        # 转弯3: +kmax → +kmax (逆时针90°，同号 → 纯圆弧)
        xs, ys, th, kaps = smart_turn(state, 1.0/Rmin, np.pi/2, Rmin, Ls, L_rev)
        x_all.extend(xs[1:]); y_all.extend(ys[1:]); kap_all.extend(kaps[1:])
        state = (x_all[-1], y_all[-1], th, 1.0/Rmin)
        
        # 转弯4: +kmax → -kmax (顺时针90°，异号)
        xs, ys, th, kaps = smart_turn(state, -1.0/Rmin, np.pi/2, Rmin, Ls, L_rev)
        x_all.extend(xs[1:]); y_all.extend(ys[1:]); kap_all.extend(kaps[1:])
        state = (x_all[-1], y_all[-1], th, -1.0/Rmin)
        
        # 注意：周期结束状态曲率为 -kmax，下一个周期转弯1目标也是 -kmax，同号，smart_turn 会自动使用纯圆弧
        
    return np.array(x_all), np.array(y_all), np.array(kap_all)

# ========== 主程序（根据 X 选择模式，此处只演示 X=0.2 的无直线情况）==========
if X <= 2*Rmin + 1e-9:
    print("模式：无直线段，同号圆弧无平滑")
    x_path, y_path, kappa_path = generate_path_no_straight(num_periods, X, Rmin, Ls_des, L_rev_des)
else:
    # 当 X > 2Rmin 时，可调用带直线的自适应版本（之前已实现，此处略）
    print("当前仅支持 X=0.2 的演示，X>0.2 请使用之前的自适应代码")
    exit()

# 杆偏移计算
pole_xs = np.arange(num_periods * 2 + 1) * X
d0 = min(np.min(np.hypot(x_path - xp, y_path + Rmin)) for xp in pole_xs)
y_pole = 2 * Rmin - d0
print(f"杆调整至 y = -{y_pole:.4f} m")

# 速度、时间
dx = np.diff(x_path); dy = np.diff(y_path)
ds = np.hypot(dx, dy)
s = np.insert(np.cumsum(ds), 0, 0)
k_max = 1.0 / Rmin
v = v_min + (v_max - v_min) * (k_max - np.abs(kappa_path)) / k_max
v = np.clip(v, v_min, v_max)
dt = ds / v[:-1]
t = np.insert(np.cumsum(dt), 0, 0)

# ========== 绘图 ==========
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

ax = axes[0, 0]
ax.plot(x_path, y_path, 'b-', lw=2, label='Path')
ax.plot(0, 0, 'go', ms=10, label='Start')
for xp in pole_xs:
    ax.add_patch(Circle((xp, -y_pole), pole_radius, color='red', alpha=0.6))
ax.plot(pole_xs, np.full_like(pole_xs, -y_pole), 'rx', label='Poles')
ax.axhline(0, color='gray', ls=':')
ax.axhline(-y_pole, color='red', ls='--', alpha=0.3)
ax.axhline(-y_pole - Rmin, color='blue', ls='--', alpha=0.3, label=f'y = -y_pole - Rmin')
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
ax.set_title('Trajectory (cross‑period same‑arc no smoothing)')
ax.axis('equal'); ax.grid(True); ax.legend(fontsize=8)

ax = axes[0, 1]
ax.plot(s, kappa_path, 'b-', lw=1.5)
ax.axhline(k_max, color='red', ls='--', label=f'±{k_max:.1f}'); ax.axhline(-k_max, color='red', ls='--')
ax.set_xlabel('Arc length (m)'); ax.set_ylabel('Curvature (1/m)')
ax.set_title('Curvature vs arc length'); ax.grid(True); ax.legend()

ax = axes[0, 2]
ax.plot(s, v, 'g-', lw=1.5)
ax.axhline(v_max, color='gray', ls='--', label=f'v_max={v_max}'); ax.axhline(v_min, color='gray', ls='--', label=f'v_min={v_min}')
ax.set_xlabel('Arc length (m)'); ax.set_ylabel('Velocity (m/s)')
ax.set_title('Velocity vs arc length'); ax.grid(True); ax.legend()

ax = axes[1, 0]
ax.plot(t, kappa_path, 'b-', lw=1.5)
ax.axhline(k_max, color='red', ls='--'); ax.axhline(-k_max, color='red', ls='--')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Curvature (1/m)')
ax.set_title('Curvature vs time'); ax.grid(True)

ax = axes[1, 1]
ax.plot(t, v, 'g-', lw=1.5)
ax.set_xlabel('Time (s)'); ax.set_ylabel('Velocity (m/s)')
ax.set_title('Velocity vs time'); ax.grid(True)

ax = axes[1, 2]
ax.plot(t, np.abs(kappa_path), 'b-', label='|curvature|')
axt = ax.twinx()
axt.plot(t, v, 'g-', label='velocity')
ax.set_xlabel('Time (s)'); ax.set_ylabel('|Curvature| (1/m)', color='b')
axt.set_ylabel('Velocity (m/s)', color='g')
ax.set_title('Curvature & velocity over time')
ax.grid(True)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = axt.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.show()