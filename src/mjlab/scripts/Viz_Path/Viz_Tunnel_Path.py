import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ==================== 配置 ====================
OBSTACLE_X_LEFT_CM = 18.5                       # 障碍物左侧 x (cm)
OBSTACLE_LENGTH_CM = 3.0                        # 障碍物长度 a (cm) → 右侧 x+a = 21.5cm
HOLE_BOTTOM = 0.050                             # 洞下沿高度 (m) = hole.py position z (对齐旧版 hole1)
HOLE_THICKNESS = 0.01                           # 限高板厚度 (m) = 2*size[2]

HEIGHT_NORMAL = 0.055                           # 正常段期望高度 (m) — 对齐 Slalom
HEIGHT_HOLE = 0.02                              # 低高度期望 (m)
TRANSITION_LENGTH_CM = 0.0                      # 线性过渡长度 (cm): 正常↔低高度

FRONT_DOWN_CM = OBSTACLE_X_LEFT_CM - 9.0                            # 前肢中心 降起点 x-9
FRONT_UP_CM = OBSTACLE_X_LEFT_CM + 5.0 + OBSTACLE_LENGTH_CM / 2     # 前肢中心 升起点 x+5+a/2
BASE_DOWN_CM = OBSTACLE_X_LEFT_CM - 14.0                            # baselink 降起点 x-14
BASE_UP_CM = OBSTACLE_X_LEFT_CM + 8.0 + OBSTACLE_LENGTH_CM          # baselink 升起点 x+8+a
REAR_DOWN_CM = OBSTACLE_X_LEFT_CM - 4.0 + OBSTACLE_LENGTH_CM / 2    # 后肢中心 降起点 x-4+a/2
REAR_UP_CM = OBSTACLE_X_LEFT_CM + 4.0 + OBSTACLE_LENGTH_CM          # 后肢中心 升起点 x+4+a

CORRIDOR_HALF_HEIGHT = 0.025                    # 走廊半高 (m):  期望高度 ± 半高

X_RANGE = (0.0, 0.42)     # 轨迹 x 范围 (m)
N_SAMPLES = 600           # 采样点数
SAVE_PATH = Path(__file__).parent / "Path" / "Viz_Tunnel_Path.png"


# 计算单条梯形波期望高度轨迹 (d0 处开始降, d0+过渡达到低; u0 处开始升, u0+过渡恢复)
def compute_point_trajectory(xs: np.ndarray, down_start_m: float, up_start_m: float, trans_m: float) -> np.ndarray:
    breakpoints = [down_start_m, down_start_m + trans_m, up_start_m, up_start_m + trans_m,]
    values = [HEIGHT_NORMAL, HEIGHT_HOLE, HEIGHT_HOLE, HEIGHT_NORMAL]
    return np.interp(xs, breakpoints, values, left=HEIGHT_NORMAL, right=HEIGHT_NORMAL)


# 绘制单个子图: 轨迹 + 走廊 + 洞 + 过渡带
def draw_trajectory_subplot(ax, xs, z_ref, title: str, desc: str, down_cm: float, up_cm: float, color: str) -> None:
    cm = 0.01
    trans_m = TRANSITION_LENGTH_CM * cm
    down_m, up_m = down_cm * cm, up_cm * cm
    hole_left = OBSTACLE_X_LEFT_CM * cm
    hole_right = (OBSTACLE_X_LEFT_CM + OBSTACLE_LENGTH_CM) * cm

    # 走廊带 (期望高度 ± 半高)
    ax.fill_between(
        xs, z_ref - CORRIDOR_HALF_HEIGHT, z_ref + CORRIDOR_HALF_HEIGHT,
        color="skyblue", alpha=0.25, zorder=1,
        label=f"走廊 (±{CORRIDOR_HALF_HEIGHT:.2f})",
    )

    # 洞 (门洞限高板剖面)
    ax.add_patch(
        Rectangle(
            (hole_left, HOLE_BOTTOM - HOLE_THICKNESS / 2),
            hole_right - hole_left, HOLE_THICKNESS,
            facecolor="orange", edgecolor="darkorange", alpha=0.85, zorder=4,
            label=f"洞 [{hole_left*100:.1f}, {hole_right*100:.1f}]cm",
        )
    )
    ax.axhline(HOLE_BOTTOM, color="darkorange", linestyle="--", linewidth=1.2, zorder=3, label=f"洞下沿 {HOLE_BOTTOM:.2f}")

    # 低高度平台区 [d0+过渡, u0]
    ax.axvspan(down_m + trans_m, up_m, color="gold", alpha=0.12, zorder=1, label=f"低高度区 [{down_m+trans_m:.3f}, {up_m:.3f}]")

    # 线性过渡带 (4cm)
    ax.axvspan(down_m, down_m + trans_m, color="gray", alpha=0.15, zorder=1)
    ax.axvspan(up_m, up_m + trans_m, color="gray", alpha=0.15, zorder=1)
    ax.text(down_m + trans_m / 2, 0.015, "降过渡", ha="center", fontsize=7, color="gray")
    ax.text(up_m + trans_m / 2, 0.015, "升过渡", ha="center", fontsize=7, color="gray")

    # 期望高度轨迹
    ax.plot(xs, z_ref, color=color, linewidth=2.2, zorder=7, label=f"期望高度 (低 {HEIGHT_HOLE:.3f})")

    # 参考线
    ax.axhline(HEIGHT_NORMAL, color="gray", linestyle=":", linewidth=1.2, zorder=2, label=f"正常 {HEIGHT_NORMAL:.2f}")
    ax.axhline(HEIGHT_HOLE, color="red", linestyle=":", linewidth=1.2, zorder=2, label=f"低 {HEIGHT_HOLE:.3f}")

    ax.set_title(f"{title}  ( {desc} )", fontsize=10)
    ax.set_ylabel("Z (m)")
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(0.0, 0.10)
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=7, loc="upper left", ncol=2)


# 绘制三个子图 (前肢中心 / baselink / 后肢中心), 不放机器人
def plot_tunnel_path() -> None:
    cm = 0.01
    trans_m = TRANSITION_LENGTH_CM * cm
    xs = np.linspace(*X_RANGE, N_SAMPLES)

    # 三条轨迹
    specs = [
        ("前肢中心", f"降 x-9={FRONT_DOWN_CM:.1f}cm, 升 x+5+a/2={FRONT_UP_CM:.1f}cm",
         FRONT_DOWN_CM, FRONT_UP_CM, "#1f77b4"),
        ("baselink", f"降 x-14={BASE_DOWN_CM:.1f}cm, 升 x+8+a={BASE_UP_CM:.1f}cm",
         BASE_DOWN_CM, BASE_UP_CM, "#2ca02c"),
        ("后肢中心", f"降 x-4-a/2={REAR_DOWN_CM:.1f}cm, 升 x+4+a={REAR_UP_CM:.1f}cm",
         REAR_DOWN_CM, REAR_UP_CM, "#d62728"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    for ax, (title, desc, down_cm, up_cm, color) in zip(axes, specs):
        z_ref = compute_point_trajectory(xs, down_cm * cm, up_cm * cm, trans_m)
        draw_trajectory_subplot(ax, xs, z_ref, title, desc, down_cm, up_cm, color)

    axes[-1].set_xlabel("X (m)")
    fig.suptitle("Tunnel 三点期望高度轨迹 (XoZ, 障碍物 x=18.5cm, a=3cm, 过渡4cm)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if SAVE_PATH is not None:
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(SAVE_PATH), dpi=150)
        print(f"[Viz_Tunnel_Path] 图片已保存: {SAVE_PATH}")
    plt.show()


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC"]
    plt.rcParams["axes.unicode_minus"] = False
    plot_tunnel_path()
