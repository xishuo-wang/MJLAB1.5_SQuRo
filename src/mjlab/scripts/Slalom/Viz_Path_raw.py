import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle   # 修复 pylance 警告

# ========== 参数 ==========
Rmin = 0.10
pole_radius = 0.005

# ========== 辅助函数：已知起终点、半径和旋转方向生成1/4圆弧 ==========
def arc_from_start_end(start, end, r, clockwise, steps=30):
    start = np.asarray(start)
    end = np.asarray(end)
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
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return x, y

# ========== 生成一个周期的路径（起点在 (0,0)） ==========
def generate_one_period(X, Rmin):
    x0, y0 = 0.0, 0.0
    path_x, path_y = [x0], [y0]

    # 1. 顺时针1/4: (0,0) -> (Rmin, -Rmin)
    ax, ay = arc_from_start_end((x0, y0), (x0 + Rmin, y0 - Rmin), Rmin, clockwise=True)
    path_x.extend(ax); path_y.extend(ay)

    # 2. 逆时针1/4: -> (2Rmin, -2Rmin)
    ax, ay = arc_from_start_end((x0 + Rmin, y0 - Rmin), (x0 + 2*Rmin, y0 - 2*Rmin), Rmin, clockwise=False)
    path_x.extend(ax); path_y.extend(ay)

    # 3. 直线: -> (X, -2Rmin)  长度 X-2Rmin
    if X - 2 * Rmin > 1e-9:
        path_x.append(x0 + X)
        path_y.append(y0 - 2 * Rmin)

    # 4. 逆时针1/4: -> (X+Rmin, -Rmin)
    ax, ay = arc_from_start_end((x0 + X, y0 - 2*Rmin), (x0 + X + Rmin, y0 - Rmin), Rmin, clockwise=False)
    path_x.extend(ax); path_y.extend(ay)

    # 5. 顺时针1/4: -> (X+2Rmin, 0)
    ax, ay = arc_from_start_end((x0 + X + Rmin, y0 - Rmin), (x0 + X + 2*Rmin, y0), Rmin, clockwise=True)
    path_x.extend(ax); path_y.extend(ay)

    # 6. 直线: -> (2X, 0)  长度 X-2Rmin
    if X - 2 * Rmin > 1e-9:
        path_x.append(x0 + 2 * X)
        path_y.append(y0)

    return np.array(path_x), np.array(path_y)

# ========== 生成多个周期 ==========
def generate_path(X, Rmin, num_periods=3):
    x_all, y_all = [0.0], [0.0]
    for k in range(num_periods):
        px, py = generate_one_period(X, Rmin)
        x_all.extend(px[1:])
        y_all.extend(py[1:])
        # 更新下一个周期的起点（当前终点）
        # 注意 generate_one_period 已经是从 (0,0) 开始，我们需要全局坐标
        # 简单做法：用累积位移
    # 上面的循环不对，因为每个周期都从 (0,0) 开始。
    # 正确做法：每个周期衔接前一个周期的终点。
    x_all, y_all = [0.0], [0.0]
    cur_x, cur_y = 0.0, 0.0
    for k in range(num_periods):
        # 移动起点到 (cur_x, cur_y)
        px, py = generate_one_period(X, Rmin)
        px += cur_x
        py += cur_y
        if k == 0:
            x_all.extend(px[1:])
            y_all.extend(py[1:])
        else:
            x_all.extend(px[1:])
            y_all.extend(py[1:])
        cur_x = x_all[-1]
        cur_y = y_all[-1]
    return np.array(x_all), np.array(y_all)

# ========== 绘图 ==========
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, X, title in zip(axes, [0.2, 0.3], ['X = 2Rmin (no straight)', 'X = 3Rmin (with straight)']):
    # 生成杆位置（仅用于显示）
    num_poles = 6
    poles_x = np.arange(num_poles) * X
    poles_y = np.full(num_poles, -Rmin)

    # 画杆
    for px, py in zip(poles_x, poles_y):
        ax.add_patch(Circle((px, py), pole_radius, color='red', alpha=0.6))
    ax.plot(poles_x, poles_y, 'rx', label='Pole centers')

    # 生成路径（多个周期）
    path_x, path_y = generate_path(X, Rmin, num_periods=3)

    # 画路径
    ax.plot(path_x, path_y, 'b-', linewidth=2, label='Reference path')

    # 标记直线段（y≈0 或 y≈-2Rmin 的部分用红色虚线标出）
    straight = (np.abs(path_y) < 1e-6) | (np.abs(path_y + 2*Rmin) < 1e-6)
    ax.plot(path_x[straight], path_y[straight], 'r--', linewidth=2, label='Straight segments')

    # 起点
    ax.plot(0, 0, 'go', markersize=10, label='Start')

    # 辅助线
    ax.axhline(0, color='gray', linestyle=':')
    ax.axhline(-Rmin, color='red', linestyle='--', alpha=0.3)
    ax.axhline(-2*Rmin, color='blue', linestyle='--', alpha=0.3)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.axis('equal')
    ax.grid(True)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()