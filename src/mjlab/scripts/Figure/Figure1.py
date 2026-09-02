import matplotlib.pyplot as plt
import numpy as np

# =========================================
# 1. 参数配置
# =========================================
categories = ['1/COT', 'Prink', 'Average line']
n_groups = 3                     # 大类数量
n_sub = 3                        # 每个大类中的子块数量
n_per_sub = 4                    # 每个子块中的柱子数量
total_bars = n_sub * n_per_sub   # 12

inner_radius = 0.4               # 内半径
group_span = 2 * np.pi / n_groups  # 每个大类角度跨度（120°）

# 子块占大类角度的比例，剩余为间隙
sub_frac = 0.8
sub_span = (group_span / n_sub) * sub_frac   # 每个子块实际宽度
gap = (group_span / n_sub) * (1 - sub_frac)  # 子块间间隙

# 每个子块内部柱子的角度步长
step_sub = sub_span / n_per_sub
bar_width = step_sub * 0.7       # 柱子宽度

# ---------- 颜色配置 ----------
# 大类背景色（浅色，透明度0.5）
bg_colors = ['#FFD1D1', '#C5E8E0', '#FFF5C2']   # 浅红、浅青、浅黄

# Nature 风格的 4 色（低饱和度，学术感）
# sub_colors = ['#3F7FBF', '#E58B4C', '#6BB36B', '#C97C7C']
# sub_colors = ['#8FA6B4', '#A3B693', '#E0A87C', '#2F6EB5']
# sub_colors = ['#56B4E9', '#E69F00', '#009E73', '#D55E00']
sub_colors = ['#E64B35', '#4DBBD5', '#8A8A8A', '#F39B7F']
# 可选其他 Nature 配色：
# sub_colors = ['#2E5A88', '#D55E00', '#56B4E9', '#CC79A7']

# =========================================
# 2. 创建极坐标子图
# =========================================
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

# =========================================
# 3. 绘制每个大类
# =========================================
np.random.seed(2026)  # 固定随机种子

for g in range(n_groups):
    start_angle = g * group_span   # 大类的起始角度

    # ---- 为该大类独立生成 12 个随机数值 ----
    values = np.round(np.random.uniform(0.5, 1.5, total_bars), 2).tolist()
    max_value = max(values)

    # ---- 浅色背景扇形（整个大类） ----
    bg_width = group_span * 0.95
    bg_angle = start_angle + group_span / 2
    ax.bar(bg_angle, max_value, width=bg_width, bottom=inner_radius,
           color=bg_colors[g], edgecolor='none', alpha=0.5, zorder=1)

    # ---- 绘制三个子块 ----
    for k in range(n_sub):
        # 子块的起始角度（带间隙）
        sub_start = start_angle + k * (group_span / n_sub) + gap / 2

        # 子块内部的 4 根柱子
        for i in range(n_per_sub):
            angle = sub_start + (i + 0.5) * step_sub
            idx = k * n_per_sub + i
            # 柱子颜色按位置 i 选择（所有子块统一）
            ax.bar(angle, values[idx], width=bar_width, bottom=inner_radius,
                   color=sub_colors[i],          # 关键修改：按 i 取色
                   edgecolor='white', linewidth=0.5,
                   alpha=0.9, zorder=2)

# =========================================
# 4. 美化与标签
# =========================================
# 大类标签
group_ticks = [group_span/2 + g * group_span for g in range(n_groups)]
ax.set_xticks(group_ticks)
ax.set_xticklabels(categories, fontsize=14, fontweight='bold')

ax.set_ylim(0, inner_radius + 1.7)
ax.set_yticklabels([])
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.grid(False)

plt.title('3 Categories, Each with 3 Sub-groups of 4 Bars (Nature Colors)', 
          pad=25, fontsize=14)
plt.show()