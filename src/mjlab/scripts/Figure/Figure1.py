import matplotlib.pyplot as plt
import numpy as np

# ==================================================================================================
# 参数配置
data = {
    'Tunnel': [
        [0.0369, 0.0145, 0.8889, 0.0318],
        [0.0892, 0.0134, 0.8206, 0.0768],
        [0.6195, 0.5082, 0.5026, 0.2902],
    ],
    'Slalom': [
        [0.4003, 0.7480, 0.7030, 0.3162],
        [0.3149, 0.2249, 0.2114, 0.2488],
        [0.5221, 0.5219, 0.5233, 0.5248],
    ],
    'Righting': [
        [0.4763, 0.6950, 0.3807, 0.4776],
        [0.3489, 0.1946, 0.1066, 0.3499],
        [0.3716, 0.5363, 0.3305, 0.3711],
    ]
}

n_groups = 3
n_sub = 3
n_per_sub = 4

INNER_RADIUS = 0.4
OUTER_RADIUS = 1.4
group_span = 2 * np.pi / n_groups

sub_frac = 0.8
sub_span = (group_span / n_sub) * sub_frac
gap = (group_span / n_sub) * (1 - sub_frac)

step_sub = sub_span / n_per_sub
bar_width = step_sub * 0.65

categories = ['Tunnel', 'Slalom', 'Righting']

# 大类背景色（浅色）
bg_colors = ['#EAF2F8', '#FDEDEC', '#FEF9E7']   # 浅蓝、浅粉、浅黄

# 大类箭头/圆弧颜色（同色系深色）
arrow_colors = ['#4A90E2', '#E74C3C', '#F1C40F']  # 亮蓝、亮红、金黄色

# 小类柱子颜色（三个子块统一使用，跨大类一致）
sub_colors = ['#9575CD', '#F06292', '#4DB6AC']  # 淡紫、粉红、青绿

# ==================================================================================================
# 绘图
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

r_ticks = np.linspace(INNER_RADIUS, OUTER_RADIUS, 6)

for g, category in enumerate(categories):
    start_angle = g * group_span
    category_data = data[category]

    # 背景扇形
    bg_width = group_span * 0.95
    bg_angle = start_angle + group_span / 2
    bg_height = OUTER_RADIUS - INNER_RADIUS
    ax.bar(bg_angle, bg_height, width=bg_width, bottom=INNER_RADIUS,
           color=bg_colors[g], edgecolor='none', alpha=0.5, zorder=1)

    # 内部径向参考线
    theta_start = start_angle + (group_span - bg_width) / 2
    theta_end = start_angle + group_span - (group_span - bg_width) / 2
    theta = np.linspace(theta_start, theta_end, 100)
    for r in r_ticks:
        ax.plot(theta, [r] * len(theta), color='gray', linestyle='--',
                linewidth=0.5, alpha=0.5, zorder=1.5)

    # ---------- 独立缩放 ----------
    all_values = [v for sub in category_data for v in sub]
    max_val = max(all_values) * 1.1
    scale = (OUTER_RADIUS - INNER_RADIUS) / max_val if max_val > 0 else 1.0

    # 数据柱（使用小类颜色 sub_colors[k]）
    for k, sub_values in enumerate(category_data):
        sub_start = start_angle + k * (group_span / n_sub) + gap / 2
        for i, value in enumerate(sub_values):
            scaled_value = value * scale
            angle = sub_start + (i + 0.5) * step_sub
            ax.bar(angle, scaled_value, width=bar_width, bottom=INNER_RADIUS,
                   color=sub_colors[k], edgecolor='white', linewidth=0.5,
                   alpha=0.9, zorder=2)

# ---------- 大类间隙径向轴线（柱状+三角形箭头） ----------
axis_angles = [g * group_span for g in range(n_groups)]
AXIS_END_R = OUTER_RADIUS + 0.25
wedge_half_angle = 0.03
arrow_length = 0.15

for angle in axis_angles:
    shaft_end_r = AXIS_END_R - arrow_length

    theta_shaft = [angle, angle - wedge_half_angle, angle + wedge_half_angle]
    r_shaft = [0, shaft_end_r, shaft_end_r]
    ax.fill(theta_shaft, r_shaft, color='gray', alpha=0.4,
            edgecolor='none', zorder=1.8)

    theta_arrow = [angle - 2*wedge_half_angle, angle + 2*wedge_half_angle, angle]
    r_arrow = [shaft_end_r, shaft_end_r, AXIS_END_R]
    ax.fill(theta_arrow, r_arrow, color='gray', alpha=0.4,
            edgecolor='none', zorder=1.8)

# ---------- 弧形箭头 ----------
ARC_RADIUS = OUTER_RADIUS + 0.1
arc_span_frac = 0.9
arc_half_width = 0.03

for g, category in enumerate(categories):
    start_angle = g * group_span
    arc_start = start_angle + (group_span * (1 - arc_span_frac)) / 2
    arc_end   = start_angle + group_span - (group_span * (1 - arc_span_frac)) / 2

    color = arrow_colors[g]

    theta_arc = np.linspace(arc_start, arc_end, 100)
    ax.plot(theta_arc, [ARC_RADIUS] * len(theta_arc),
            color=color, linewidth=4.0, linestyle='-', alpha=0.9, zorder=2.5)

    ax.annotate('', xy=(arc_start - arc_half_width, ARC_RADIUS),
                xytext=(arc_start, ARC_RADIUS),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5, shrinkA=0, shrinkB=0),
                zorder=3)

    ax.annotate('', xy=(arc_end + arc_half_width, ARC_RADIUS),
                xytext=(arc_end, ARC_RADIUS),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5, shrinkA=0, shrinkB=0),
                zorder=3)

# 清除角度刻度
ax.set_xticks([])
ax.set_xticklabels([])

ax.set_ylim(0, ARC_RADIUS + 0.2)
ax.set_yticklabels([])

ax.grid(False)

fig.set_facecolor('none')
ax.set_facecolor('none')
ax.set_theta_zero_location('N') # type: ignore
ax.set_theta_direction(-1) # type: ignore
ax.spines['polar'].set_visible(False)

plt.show()