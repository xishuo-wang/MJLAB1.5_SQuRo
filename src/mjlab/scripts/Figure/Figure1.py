import os
import numpy as np
import matplotlib.pyplot as plt

# ==================================================================================================
# 保存配置
SAVE_DIR = r"D:\MuJoCoLab_1.5\src\mjlab\scripts\Figure\Figure"
FILENAME_BASE = "Figure1"
os.makedirs(SAVE_DIR, exist_ok=True)

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

BG_WIDTH_FRAC = 0.85                     # 大类间隙：背景扇形宽度占大类角度的比例
bg_width = group_span * BG_WIDTH_FRAC

# 子块布局参数（基于背景扇形宽度 bg_width）
sub_frac = 0.8
sub_span = (bg_width / n_sub) * sub_frac
gap = (bg_width / n_sub) * (1 - sub_frac)

# 柱子的角度步长
step_sub = sub_span / n_per_sub
bar_width = step_sub * 0.65

categories = ['Tunnel', 'Slalom', 'Righting']

# 大类背景色（浅色）
bg_colors = ['#EAF2F8', '#FDEDEC', '#FEF9E7']

# 大类箭头/圆弧颜色
arrow_colors = ['#4A90E2', '#E74C3C', '#F1C40F']

# 小类柱子颜色
sub_colors = ['#9575CD', '#F06292', '#4DB6AC']

# 偏移与旋转参数
axis_offset = 0.05                     # 轴线偏移量（弧度），正值顺时针
global_rotation = -axis_offset         # 全局旋转，使偏移后的轴线回到正上方

# ==================================================================================================
# 绘图函数
def create_plot(annotated):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

    r_ticks = np.linspace(INNER_RADIUS, OUTER_RADIUS, 6)

    # ---------- 绘制背景扇形、参考线、数据柱 ----------
    for g, category in enumerate(categories):
        # 大类起始角度：加入全局旋转（背景和柱子偏移）
        start_angle = g * group_span + global_rotation
        category_data = data[category]

        # 背景扇形实际角度范围
        theta_start = start_angle + (group_span - bg_width) / 2
        theta_end = start_angle + group_span - (group_span - bg_width) / 2

        # 背景扇形
        bg_angle = start_angle + group_span / 2
        bg_height = OUTER_RADIUS - INNER_RADIUS
        ax.bar(bg_angle, bg_height, width=bg_width, bottom=INNER_RADIUS,
               color=bg_colors[g], edgecolor='none', alpha=0.5, zorder=1)

        # 内部径向参考线
        theta = np.linspace(theta_start, theta_end, 100)
        for r in r_ticks:
            ax.plot(theta, [r] * len(theta), color='gray', linestyle='--',
                    linewidth=0.5, alpha=0.5, zorder=1.5)

        # 独立缩放
        all_values = [v for sub in category_data for v in sub]
        max_val = max(all_values) * 1.1
        scale = (OUTER_RADIUS - INNER_RADIUS) / max_val if max_val > 0 else 1.0

        # 数据柱
        for k, sub_values in enumerate(category_data):
            sub_start = theta_start + k * (bg_width / n_sub) + gap / 2
            for i, value in enumerate(sub_values):
                scaled_value = value * scale
                angle = sub_start + (i + 0.5) * step_sub
                ax.bar(angle, scaled_value, width=bar_width, bottom=INNER_RADIUS,
                       color=sub_colors[k], edgecolor='white', linewidth=0.5,
                       alpha=0.9, zorder=2)

    # ---------- 径向轴线（位于间隙中心，正上方） ----------
    # 轴线角度：原始间隙中心（因为 axis_offset + global_rotation = 0）
    axis_angles = [g * group_span for g in range(n_groups)]
    AXIS_END_R = OUTER_RADIUS + 0.25
    wedge_half_angle = 0.03
    arrow_length = 0.15

    for angle in axis_angles:
        shaft_end_r = AXIS_END_R - arrow_length

        # 轴身
        theta_shaft = [angle, angle - wedge_half_angle, angle + wedge_half_angle]
        r_shaft = [0, shaft_end_r, shaft_end_r]
        ax.fill(theta_shaft, r_shaft, color='gray', alpha=0.4,
                edgecolor='none', zorder=1.8)

        # 轴末端箭头
        theta_arrow = [angle - 2*wedge_half_angle, angle + 2*wedge_half_angle, angle]
        r_arrow = [shaft_end_r, shaft_end_r, AXIS_END_R]
        ax.fill(theta_arrow, r_arrow, color='gray', alpha=0.4,
                edgecolor='none', zorder=1.8)

    # ---------- 弧形箭头（位于间隙中心，与轴线对齐，不偏移） ----------
    ARC_RADIUS = OUTER_RADIUS + 0.1
    arc_span_frac = 0.9
    arc_half_width = 0.03

    for g, category in enumerate(categories):
        # 弧形箭头角度：直接使用间隙中心，不包含global_rotation
        start_angle = g * group_span
        arc_start = start_angle + (group_span * (1 - arc_span_frac)) / 2
        arc_end   = start_angle + group_span - (group_span * (1 - arc_span_frac)) / 2

        color = arrow_colors[g]

        theta_arc = np.linspace(arc_start, arc_end, 100)
        ax.plot(theta_arc, [ARC_RADIUS] * len(theta_arc),
                color=color, linewidth=4.0, linestyle='-', alpha=0.9, zorder=2.5)

        # 两端箭头
        ax.annotate('', xy=(arc_start - arc_half_width, ARC_RADIUS),
                    xytext=(arc_start, ARC_RADIUS),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.5, shrinkA=0, shrinkB=0),
                    zorder=3)

        ax.annotate('', xy=(arc_end + arc_half_width, ARC_RADIUS),
                    xytext=(arc_end, ARC_RADIUS),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.5, shrinkA=0, shrinkB=0),
                    zorder=3)

    # ---------- 标注控制 ----------
    if annotated:
        label_radius = ARC_RADIUS + 0.25
        info_radius = ARC_RADIUS + 0.12
        for g, cat in enumerate(categories):
            # 标注角度：间隙中心，与轴线、弧形箭头一致
            angle = g * group_span
            all_vals = [v for sub in data[cat] for v in sub]
            max_orig = max(all_vals)
            min_orig = min(all_vals)
            ax.text(angle, label_radius, cat,
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color=arrow_colors[g])
            ax.text(angle, info_radius, f'[{min_orig:.2f}, {max_orig:.2f}]',
                    ha='center', va='center', fontsize=8, color='gray')

        ax.set_yticks(r_ticks)
        ax.set_yticklabels([f'{r:.1f}' for r in r_ticks], fontsize=8)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])

    # 公共设置
    ax.set_ylim(0, ARC_RADIUS + 0.4)
    ax.grid(False)

    fig.set_facecolor('none')
    ax.set_facecolor('none')
    ax.set_theta_zero_location('N') # type: ignore
    ax.set_theta_direction(-1) # type: ignore
    ax.spines['polar'].set_visible(False)

    return fig

# ==================================================================================================
# 生成、保存并显示两张图
fig1 = create_plot(annotated=False)
fig1.savefig(os.path.join(SAVE_DIR, f"{FILENAME_BASE}.png"), transparent=True, bbox_inches='tight', dpi=300)

fig2 = create_plot(annotated=True)
fig2.savefig(os.path.join(SAVE_DIR, f"{FILENAME_BASE}_labels.png"), transparent=True, bbox_inches='tight', dpi=300)

plt.show()