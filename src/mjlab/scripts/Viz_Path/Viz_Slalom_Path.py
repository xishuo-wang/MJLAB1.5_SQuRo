import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 直接从 Slalom 任务导入 RL 实际使用的平滑 LUT 路径生成函数 (保证展示 = 训练使用)
from mjlab.tasks.SQuRo_Slalom.mdp.path import (
    _approach_rev_table,
    _generate_slalom_lut_smooth_period,
    BODY_REF_OFFSET,
    CORRIDOR_HALF_WIDTH,
    _INIT_DIST,
)
from mjlab.tasks.SQuRo_Slalom.mdp.pole import POLE_Y, generate_pole_positions
from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import SMOOTH_TIME, SMOOTH_VEL


# ==================== 配置 ====================
POLE_SPACING = 0.15             # 杆间距 (m) — 与回放默认 fixed_pole_spacing 一致
NUM_PERIODS = 3                 # 周期数
NUM_POLES = 8                   # 显示杆数量
CORRIDOR_HALF_WIDTH = 0.04      # 走廊半宽 (m) — 与 Slalom 任务 corridor 一致
BODY_REF_OFFSET = 0.04          # F/H_body 中心沿路径切线偏移 (m) — 与 Slalom 任务一致

X_RANGE = (0.0, 1.10)           # 显示 x 范围 (m)
Y_RANGE = (-0.20, 0.20)         # 显示 y 范围 (m)
N_SAMPLES = 400                 # 采样点数 (路径由 LUT 直接给出, 仅用于重采样)
SAVE_PATH = Path(__file__).parent / "Path" / "Viz_Slalom_Path.png"


# 生成 RL 实际使用的 base 期望路径 (接近段 + 多周期平滑 LUT)
def generate_base_path(spacing: float, num_periods: int):
    # 一个周期的平滑 LUT (起点 (0,0), 周期 x 位移 = 2*spacing)
    _, xs, ys, hd, _ = _generate_slalom_lut_smooth_period(spacing)
    px, py, ph = [], [], []
    for k in range(num_periods):
        px.extend(xs + spacing + k * 2 * spacing)
        py.extend(ys)
        ph.extend(hd)
    return np.array(px), np.array(py), np.array(ph)


# 沿路径切线偏移 offset 得到前/后肢中心路径 (F +offset, H -offset)
def offset_path(px, py, ph, offset: float):
    tangent = np.column_stack([np.cos(ph), np.sin(ph)])
    return px + offset * tangent[:, 0], py + offset * tangent[:, 1]


# 生成接近段 (圆弧: 平台 -K + 过渡 -K→0, 终点 = (spacing, 0), 与 RL 一致)
def generate_approach(spacing: float):
    tbl = _approach_rev_table(_INIT_DIST, SMOOTH_VEL * SMOOTH_TIME)
    return np.array(tbl["x"]) + spacing, np.array(tbl["y"])


# 绘制单个子图: 路径 + 杆 + 走廊带 + 接近段
def draw_path_subplot(ax, px, py, ph, title: str, desc: str, color: str,
                      draw_corridor: bool, spacing: float) -> None:
    # 杆 (仅显示)
    poles = generate_pole_positions(spacing=spacing, num_poles=NUM_POLES,
                                    start_x=spacing, start_y=POLE_Y)
    for pole_x, pole_y, _ in poles:
        ax.add_patch(Circle((pole_x, pole_y), 0.01, facecolor="red", edgecolor="darkred",
                            alpha=0.6, zorder=4))
        ax.plot(pole_x, pole_y, "rx", markersize=5, zorder=5)

    # 走廊带 (沿路径法向 ± 半宽)
    if draw_corridor:
        n_x = -np.sin(ph)
        n_y = np.cos(ph)
        up_x = px + CORRIDOR_HALF_WIDTH * n_x
        up_y = py + CORRIDOR_HALF_WIDTH * n_y
        dn_x = px - CORRIDOR_HALF_WIDTH * n_x
        dn_y = py - CORRIDOR_HALF_WIDTH * n_y
        ax.fill(np.concatenate([up_x, dn_x[::-1]]), np.concatenate([up_y, dn_y[::-1]]),
                color="skyblue", alpha=0.25, zorder=1,
                label=f"走廊 (±{CORRIDOR_HALF_WIDTH:.2f})")

    # 接近段 (RL 实际使用)
    app_x, app_y = generate_approach(spacing)
    ax.plot(app_x, app_y, "g--", linewidth=1.5, zorder=3, label="接近段 (圆弧)")

    # 期望路径
    ax.plot(px, py, color=color, linewidth=2.2, zorder=6, label="期望路径 (平滑 LUT)")

    # 直行段参考线
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, zorder=2)
    ax.axhline(2 * POLE_Y, color="gray", linestyle=":", linewidth=1.0, zorder=2)

    ax.set_title(f"{title}  ( {desc} )", fontsize=10)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=8, loc="upper right")


# 绘制三个子图 (前肢中心 / 基座 / 后肢中心), 展示 RL 实际使用的平滑 LUT 路径
def plot_slalom_path() -> None:
    spacing = POLE_SPACING
    base_x, base_y, base_h = generate_base_path(spacing, NUM_PERIODS)
    f_x, f_y = offset_path(base_x, base_y, base_h, +BODY_REF_OFFSET)
    h_x, h_y = offset_path(base_x, base_y, base_h, -BODY_REF_OFFSET)

    specs = [
        ("前肢中心 (F_body)", f"路径沿切线 +{BODY_REF_OFFSET:.3f}m",
         f_x, f_y, base_h, "#1f77b4", True),
        ("基座 (base)", "RL 不对基座判走廊 (仅 F/H)",
         base_x, base_y, base_h, "#2ca02c", False),
        ("后肢中心 (H_body)", f"路径沿切线 -{BODY_REF_OFFSET:.3f}m",
         h_x, h_y, base_h, "#d62728", True),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    for ax, (title, desc, px, py, ph, color, draw_cor) in zip(axes, specs):
        draw_path_subplot(ax, px, py, ph, title, desc, color, draw_cor, spacing)

    axes[-1].set_xlabel("X (m)")
    fig.suptitle(f"Slalom 三点期望路径 (XoY, 平滑 LUT, 杆间距 {spacing:.2f}m, 接近段+{NUM_PERIODS}周期)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if SAVE_PATH is not None:
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(SAVE_PATH), dpi=150)
        print(f"[Viz_Slalom_Path] 图片已保存: {SAVE_PATH}")
    plt.show()


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC"]
    plt.rcParams["axes.unicode_minus"] = False
    plot_slalom_path()
