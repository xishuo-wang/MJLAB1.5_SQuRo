import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 身体尺寸
L = 0.08          # 矩形全长 (前后)
W = 0.06          # 矩形全宽 (左右)
half_L = L / 2
half_W = W / 2
C = 0.08          # 走廊半宽
path_y = 0.0      # 参考路径中心线 y=0

# 身体中心位置（存在侧向偏移）
center = np.array([0.0, 0.02])

# 两个对比案例：对齐 (0°) 与 不对齐 (30°)
theta_cases = [0.0, np.deg2rad(30)]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, theta in zip(axes, theta_cases):
    # 绘制走廊边界和参考路径
    ax.axhline(y=path_y + C, color='green', linestyle='--', label=f'Corridor boundary (C = {C:.2f})')
    ax.axhline(y=path_y - C, color='green', linestyle='--')
    ax.axhline(y=path_y, color='gray', linestyle='-', label='Reference path')
    
    # 计算身体矩形角点（局部坐标：x向前，y向左）
    corners_local = np.array([
        [ half_L, -half_W],
        [ half_L,  half_W],
        [-half_L,  half_W],
        [-half_L, -half_W]
    ])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    corners_world = center + corners_local @ R.T
    rect = patches.Polygon(corners_world, closed=True, facecolor='lightblue',
                           edgecolor='blue', alpha=0.5)
    ax.add_patch(rect)
    
    # 身体中心
    ax.plot(center[0], center[1], 'bo', label='Body center')
    
    # 计算侧向距离 d_lat（路径为水平，法向即 y 方向）
    d_lat = center[1]
    # 有效半宽：eff_hw = |L·sinΔθ| + |W·cosΔθ|
    eff_hw = half_L * abs(np.sin(theta)) + half_W * abs(np.cos(theta))
    env_min = d_lat - eff_hw
    env_max = d_lat + eff_hw
    
    # 绘制身体占据的法向范围
    ax.axhline(y=env_min, color='red', linestyle=':', label=f'Envelope (e_min={env_min:.3f})')
    ax.axhline(y=env_max, color='red', linestyle=':', label=f'Envelope (e_max={env_max:.3f})')
    
    # 如果超出走廊，用红色半透明填充
    if env_max > path_y + C:
        ax.fill_between([center[0]-0.05, center[0]+0.05], path_y+C, env_max,
                        color='red', alpha=0.3, label='Excess')
    if env_min < path_y - C:
        ax.fill_between([center[0]-0.05, center[0]+0.05], env_min, path_y-C,
                        color='red', alpha=0.3)
    
    # 标注侧向距离
    ax.annotate('', xy=(center[0]+0.02, d_lat), xytext=(center[0]+0.02, 0),
                arrowprops=dict(arrowstyle='<->', color='black'))
    ax.text(center[0]+0.03, d_lat/2, f'd_lat = {d_lat:.2f}', va='center')
    
    # 图例与坐标轴
    ax.set_xlim(-0.15, 0.15)
    ax.set_ylim(-0.12, 0.12)
    ax.set_aspect('equal')
    ax.set_title(f'θ (Δθ) = {np.rad2deg(theta):.0f}°')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlabel('X (along path)')
    ax.set_ylabel('Y (normal)')

plt.suptitle('Corridor Conformance Reward – Geometry Illustration', fontsize=14)
plt.tight_layout()
plt.show()