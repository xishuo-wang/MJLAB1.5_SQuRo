# Viz_Tunnel_Path: 简化 Tunnel (门洞式洞) 任务的期望高度轨迹可视化
# 建模平面: XoZ (高度走廊)
# 期望高度由单一参考轨迹 z_ref(x) 表示, 前肢中心/基座/后肢中心共享同一条轨迹
# 机器人简化建模参考 Slalom 任务: 两个铰接矩形 (F_body + H_body) + 基座
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

# 中文字体配置 (Windows: Microsoft YaHei / SimHei, macOS: PingFang SC)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 配置 ====================
# --- 洞配置 (参考 SQuRo_Tunnel/mdp/hole.py 的 HoleEntityCfg, BOX 几何) ---
# 洞下沿 = position z (门洞净空, 用户确认 0.05m); 限高板以 position z 为中心, 厚度 = 2*size[2]
HOLE_X_CENTER = 0.20      # 洞 x 中心 (m)
HOLE_HALF_X = 0.015       # 洞 x 半宽 size[0] (m) → 洞 x 范围 [0.185, 0.215]
HOLE_Y_HALF = 0.10        # 洞 y 半深 size[1] (m), 跨走廊宽度 (XoZ 平面不显示)
HOLE_BOTTOM = 0.055        # 洞下沿高度 (m) = hole.py position[2]
HOLE_THICKNESS = 0.01     # 限高板厚度 (m) = 2*size[2] = 0.01
X1 = HOLE_X_CENTER - HOLE_HALF_X   # 洞最左侧 X1 (m) = 0.185
X2 = HOLE_X_CENTER + HOLE_HALF_X   # 洞最右侧 X2 (m) = 0.215

# --- 期望高度规则 ---
HEIGHT_NORMAL = 0.06      # 正常段期望高度 (m)
HEIGHT_CLEARANCE = 0.025  # 过洞时相对洞下沿的下降量 (m) = 走廊半高, 保证走廊上沿 ≤ 洞下沿
HEIGHT_HOLE = HOLE_BOTTOM - HEIGHT_CLEARANCE  # 洞内期望高度 = 0.055 - 0.025 = 0.03
TRANSITION_LENGTH = 0.04  # 线性过渡长度 (m): 正常↔低高度各 4cm

# --- 走廊 (高度上下限) 配置 ---
CORRIDOR_HALF_HEIGHT = 0.025  # 走廊半高 (m): 期望高度 ± 半高 (> BODY_HALF_HEIGHT 0.021, 能包住机器人矩形)

# --- 机器人简化建模 (参考 Slalom: 两个铰接矩形 F_body + H_body) ---
BODY_REF_OFFSET = 0.038    # F/H_body 中心距基座的 X 向偏移 (m)
BODY_HALF_LENGTH = 0.038   # 矩形半长 (沿 X 方向, m)
BODY_HALF_HEIGHT = 0.021   # 矩形半高 (沿 Z 方向, m)
ROBOT_BASE_X = 0.10       # 机器人快照的基座 X 位置 (m, 默认对准洞中心)

# --- 轨迹范围与输出 ---
X_RANGE = (0.0, 0.4)      # 轨迹 x 范围 (m)
N_SAMPLES = 400           # 采样点数
SAVE_PATH = Path(__file__).parent / "Result" / "Viz_Tunnel_Path.png"  # 输出图片


# 计算期望高度轨迹 (梯形波): 洞内 [X1, X2] = 低高度, 边界外各 4cm 线性过渡
def compute_height_trajectory(xs: np.ndarray) -> np.ndarray:
    breakpoints = [
        X1 - TRANSITION_LENGTH,   # 左过渡起点 (正常)
        X1,                       # 左过渡终点 (低)
        X2,                       # 右过渡起点 (低)
        X2 + TRANSITION_LENGTH,   # 右过渡终点 (正常)
    ]
    values = [HEIGHT_NORMAL, HEIGHT_HOLE, HEIGHT_HOLE, HEIGHT_NORMAL]
    return np.interp(xs, breakpoints, values, left=HEIGHT_NORMAL, right=HEIGHT_NORMAL)


# 在指定位置绘制简化机器人 (两个铰接矩形 + 基座), 共享同一期望高度 z_ref
def draw_robot(ax, base_x: float, z_ref: float) -> None:
    # 基座点
    ax.plot(base_x, z_ref, "ko", markersize=7, zorder=6, label="基座 base_Link")

    # 前体 F_body 矩形 (中心 +BODY_REF_OFFSET)
    f_x = base_x + BODY_REF_OFFSET
    ax.add_patch(
        Rectangle(
            (f_x - BODY_HALF_LENGTH, z_ref - BODY_HALF_HEIGHT),
            2 * BODY_HALF_LENGTH, 2 * BODY_HALF_HEIGHT,
            facecolor="tab:blue", edgecolor="navy", alpha=0.75, zorder=5,
            label="前体 F_body",
        )
    )
    ax.plot(f_x, z_ref, "bo", markersize=4, zorder=6)

    # 后体 H_body 矩形 (中心 -BODY_REF_OFFSET)
    h_x = base_x - BODY_REF_OFFSET
    ax.add_patch(
        Rectangle(
            (h_x - BODY_HALF_LENGTH, z_ref - BODY_HALF_HEIGHT),
            2 * BODY_HALF_LENGTH, 2 * BODY_HALF_HEIGHT,
            facecolor="tab:red", edgecolor="darkred", alpha=0.75, zorder=5,
            label="后体 H_body",
        )
    )
    ax.plot(h_x, z_ref, "ro", markersize=4, zorder=6)

    # 三点标注 (前肢中心 / 基座 / 后肢中心, 共享同一 z_ref)
    ax.annotate("前肢中心", (f_x, z_ref), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=8, color="navy")
    ax.annotate("后肢中心", (h_x, z_ref), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=8, color="darkred")


# 绘制 XoZ 平面期望高度轨迹 + 洞 + 走廊 + 机器人简化模型
def plot_tunnel_path() -> None:
    xs = np.linspace(*X_RANGE, N_SAMPLES)
    z_ref = compute_height_trajectory(xs)
    x_left = HOLE_X_CENTER - HOLE_HALF_X
    x_right = HOLE_X_CENTER + HOLE_HALF_X

    fig, ax = plt.subplots(figsize=(12, 5.5))

    # 走廊带 (期望高度 ± 半高)
    ax.fill_between(
        xs, z_ref - CORRIDOR_HALF_HEIGHT, z_ref + CORRIDOR_HALF_HEIGHT,
        color="skyblue", alpha=0.25, zorder=1,
        label=f"走廊 (高度 ±{CORRIDOR_HALF_HEIGHT:.2f})",
    )

    # 洞 (门洞限高板剖面, 板中心 = 洞下沿)
    ax.add_patch(
        Rectangle(
            (x_left, HOLE_BOTTOM - HOLE_THICKNESS / 2),
            x_right - x_left, HOLE_THICKNESS,
            facecolor="orange", edgecolor="darkorange", alpha=0.85, zorder=4,
            label=f"洞 (限高板, x∈[{x_left:.3f},{x_right:.3f}])",
        )
    )

    # 洞下沿参考线
    ax.axhline(
        HOLE_BOTTOM, color="darkorange", linestyle="--", linewidth=1.2, zorder=3,
        label=f"洞下沿 {HOLE_BOTTOM:.2f}",
    )

    # 线性过渡带 (4cm, 正常↔低高度)
    ax.axvspan(X1 - TRANSITION_LENGTH, X1, color="gray", alpha=0.15, zorder=1)
    ax.axvspan(X2, X2 + TRANSITION_LENGTH, color="gray", alpha=0.15, zorder=1)

    # 期望高度轨迹 (单条, 三点共享)
    ax.plot(xs, z_ref, "b-", linewidth=2.4, zorder=7,
            label=f"期望高度轨迹 (洞内 {HEIGHT_HOLE:.3f}, 过渡 {TRANSITION_LENGTH*100:.0f}cm)")

    # 机器人简化模型 (快照, 基座对准洞中心)
    base_z = float(compute_height_trajectory(np.array([ROBOT_BASE_X]))[0])
    draw_robot(ax, ROBOT_BASE_X, base_z)

    # 正常高度 / 洞内期望高度参考线
    ax.axhline(HEIGHT_NORMAL, color="gray", linestyle=":", linewidth=1.2, zorder=2,
               label=f"正常高度 {HEIGHT_NORMAL:.2f}")
    ax.axhline(HEIGHT_HOLE, color="red", linestyle=":", linewidth=1.2, zorder=2,
               label=f"洞内期望高度 {HEIGHT_HOLE:.3f}")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Tunnel 简化建模: 期望高度轨迹 + 机器人简化模型 (XoZ 平面)")
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(0.0, 0.10)
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    if SAVE_PATH is not None:
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(SAVE_PATH), dpi=150)
        print(f"[Viz_Tunnel_Path] 图片已保存: {SAVE_PATH}")
    plt.show()


if __name__ == "__main__":
    plot_tunnel_path()
