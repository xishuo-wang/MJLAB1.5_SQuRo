import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mjlab.tasks.SQuRo_Tunnel.mdp.path import (
    FRONT_DOWN_OFF,
    FRONT_UP_OFF,
    REAR_DOWN_OFF,
    REAR_UP_OFF,
    CORRIDOR_FRONT_HALF_HEIGHT,
    CORRIDOR_REAR_HALF_HEIGHT,
    BODY_SEG_HALF_HEIGHT,
    HEIGHT_NORMAL,
    HEIGHT_LOW,
    OBSTACLE_LENGTH,
    TUNNEL_BOTTOM,
    TUNNEL_THICKNESS,
    TRANSITION_LENGTH,
    _height_from_tunnels,
)


# 障碍物位置 (仅绘图用, 运行时洞位置由 path.sample_tunnel_positions 采样)
OBSTACLE_X_LEFT = 0.185      # 洞左侧 x (m)
# baselink 参考轨迹偏移 (对齐 c0921e6 设计: 降 x-14, 升 x+8+a; 代码中已不作为控制点)
BASE_DOWN_OFF = 0.14
BASE_UP_OFF = 0.08 + OBSTACLE_LENGTH

X_RANGE = (0.0, 0.42)        # 轨迹 x 范围 (m)
N_SAMPLES = 600              # 采样点数
SAVE_PATH = Path(__file__).parent / "Path" / "Viz_Tunnel_Path.png"


# 单条期望高度轨迹: 直接调用 path.py 的方波规则, 保证图与代码一致
def compute_point_trajectory(xs: np.ndarray, down_off: float, up_off: float) -> np.ndarray:
    xs_t = torch.tensor(np.asarray(xs, dtype=np.float32))
    tunnel_xs = torch.full((len(xs_t), 1), OBSTACLE_X_LEFT, dtype=torch.float32)
    z = _height_from_tunnels(xs_t, tunnel_xs, down_off, up_off)
    return z.numpy().astype(np.float64)


# 绘制单个子图: 期望高度 + 走廊带 + 洞剖面 + 地面
def draw_trajectory_subplot(ax, xs, z_ref, title: str, desc: str,
                            corridor_half: float, color: str) -> None:
    seg_half = BODY_SEG_HALF_HEIGHT
    band = corridor_half - seg_half
    hole_left = OBSTACLE_X_LEFT
    hole_right = OBSTACLE_X_LEFT + OBSTACLE_LENGTH

    # 走廊带 (段中心允许范围: 期望高度 ±(走廊半高 - 段半高))
    ax.fill_between(
        xs, z_ref - band, z_ref + band,
        color="skyblue", alpha=0.30, zorder=1,
        label=f"走廊 (中心 ±{band*100:.1f}cm, 半高{corridor_half*100:.0f}mm)",
    )

    # 洞剖面 (门洞限高板)
    ax.add_patch(
        Rectangle(
            (hole_left, TUNNEL_BOTTOM - TUNNEL_THICKNESS / 2),
            hole_right - hole_left, TUNNEL_THICKNESS,
            facecolor="orange", edgecolor="darkorange", alpha=0.85, zorder=4,
            label=f"限高板 [{hole_left*100:.1f}, {hole_right*100:.1f}]cm",
        )
    )
    ax.axhline(TUNNEL_BOTTOM, color="darkorange", linestyle="--", linewidth=1.2,
               zorder=3, label=f"洞下沿 {TUNNEL_BOTTOM:.3f}")

    # 低高度平台区
    ax.axvspan(hole_left - 0.0, hole_right, color="gold", alpha=0.12, zorder=1)
    if TRANSITION_LENGTH > 0.0:
        ax.axvspan(hole_left - TRANSITION_LENGTH, hole_left, color="gray", alpha=0.15, zorder=1)

    # 期望高度轨迹
    ax.plot(xs, z_ref, color=color, linewidth=2.2, zorder=7, label=f"期望高度 (低 {HEIGHT_LOW:.3f})")

    # 地面与高度参考线
    ax.axhline(0.0, color="black", linewidth=1.0, zorder=2, label="地面")
    ax.axhline(HEIGHT_NORMAL, color="gray", linestyle=":", linewidth=1.2, zorder=2,
               label=f"正常 {HEIGHT_NORMAL:.3f}")
    ax.axhline(HEIGHT_LOW, color="red", linestyle=":", linewidth=1.2, zorder=2,
               label=f"低 {HEIGHT_LOW:.3f}")

    ax.set_title(f"{title}  ( {desc} )", fontsize=10)
    ax.set_ylabel("Z (m)")
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(0.0, 0.10)
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=7, loc="upper left", ncol=2)


# 绘制三个子图 (前肢中心 / baselink / 后肢中心), 不放机器人
def plot_tunnel_path() -> None:
    xs = np.linspace(*X_RANGE, N_SAMPLES)

    specs = [
        ("前肢中心", f"降 x-{FRONT_DOWN_OFF*100:.1f}cm, 升 x+{FRONT_UP_OFF*100:.1f}cm",
         FRONT_DOWN_OFF, FRONT_UP_OFF, CORRIDOR_FRONT_HALF_HEIGHT, "#1f77b4"),
        ("baselink (参考)", f"降 x-{BASE_DOWN_OFF*100:.1f}cm, 升 x+{BASE_UP_OFF*100:.1f}cm",
         BASE_DOWN_OFF, BASE_UP_OFF, CORRIDOR_FRONT_HALF_HEIGHT, "#2ca02c"),
        ("后肢中心", f"降 x-{REAR_DOWN_OFF*100:.1f}cm, 升 x+{REAR_UP_OFF*100:.1f}cm",
         REAR_DOWN_OFF, REAR_UP_OFF, CORRIDOR_REAR_HALF_HEIGHT, "#d62728"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    for ax, (title, desc, down_off, up_off, corridor_half, color) in zip(axes, specs):
        z_ref = compute_point_trajectory(xs, down_off, up_off)
        draw_trajectory_subplot(ax, xs, z_ref, title, desc, corridor_half, color)

    axes[-1].set_xlabel("X (m)")
    fig.suptitle(
        f"Tunnel 三点期望高度轨迹 (XoZ, 洞左侧 x={OBSTACLE_X_LEFT*100:.1f}cm, "
        f"a={OBSTACLE_LENGTH*100:.0f}cm, 过渡 {TRANSITION_LENGTH*100:.0f}cm)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if SAVE_PATH is not None:
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(SAVE_PATH), dpi=150)
        print(f"[Viz_Tunnel_Path] 图片已保存: {SAVE_PATH}")
    plt.show()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC"]
    plt.rcParams["axes.unicode_minus"] = False
    plot_tunnel_path()
