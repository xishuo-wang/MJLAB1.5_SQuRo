# Viz_Tunnel_Path: 简化 Tunnel (门洞式洞) 任务的期望高度轨迹可视化
# 建模平面: XoZ (高度走廊)
# 三个关键点: 前肢中心 / 基座 / 后肢中心, 共享同一条期望高度规则 z_ref(x)
# (前/后肢中心在 X 方向按 BODY_REF_OFFSET 与基座错开, 因此三点轨迹互为平移)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

# 中文字体配置 (Windows: Microsoft YaHei / SimHei, macOS: PingFang SC)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 配置 ====================
# --- 洞配置 (参考 SQuRo_Tunnel/mdp/hole.py 的 HoleEntityCfg, BOX 几何) ---
# 几何说明: body pos = 板中心; geom 偏移 +size[2]/2, 半高 = size[2]
# → 板 z 范围 = [z - size[2]/2, z + 3*size[2]/2], 洞下边沿 = z - size[2]/2
HOLE_X_CENTER = 0.20      # 洞 x 中心 (m)
HOLE_HALF_X = 0.015       # 洞 x 半宽 size[0] (m) → 洞 x 范围 [0.185, 0.215]
HOLE_Y_HALF = 0.10        # 洞 y 半深 size[1] (m), 跨走廊宽度 (XoZ 平面不显示)
HOLE_Z_CENTER = 0.05      # 洞板 z 中心 position[2] (m)
HOLE_Z_HALF = 0.005       # 洞板 z 半厚 size[2] (m)
HOLE_BOTTOM = HOLE_Z_CENTER - HOLE_Z_HALF   # 洞下边沿 (m)
HOLE_TOP = HOLE_Z_CENTER + HOLE_Z_HALF      # 洞上边沿 (m)

# --- 期望高度规则 ---
HEIGHT_NORMAL = 0.06      # 正常段期望高度 (m)
HEIGHT_CLEARANCE = 0.02   # 过洞时相对洞下边沿的下降余量 (m)
HEIGHT_HOLE = HOLE_BOTTOM - HEIGHT_CLEARANCE  # 洞内期望高度 = 洞下边沿 - 0.02

# --- 走廊 (高度上下限) 配置 ---
CORRIDOR_HALF_HEIGHT = 0.01   # 走廊半高 (m): 期望高度 ± 半高

# --- 三点建模配置 ---
BODY_REF_OFFSET = 0.04    # 前/后肢中心距基座的 X 向偏移 (m, 同 Slalom path.py)
SHOW_THREE_POINTS = True  # 是否叠加显示三点轨迹 (前肢中心/基座/后肢中心)

# --- 轨迹范围与输出 ---
X_RANGE = (0.0, 0.4)      # 轨迹 x 范围 (m)
N_SAMPLES = 400           # 采样点数
SAVE_PATH = Path(__file__).parent / "Result" / "Viz_Tunnel_Path.png"  # 输出图片


# 计算期望高度轨迹: 洞内 = 洞下边沿 - 余量, 其余 = 正常高度
def compute_height_trajectory(xs: np.ndarray) -> np.ndarray:
    x_left = HOLE_X_CENTER - HOLE_HALF_X
    x_right = HOLE_X_CENTER + HOLE_HALF_X
    inside = (xs >= x_left) & (xs <= x_right)
    return np.where(inside, HEIGHT_HOLE, HEIGHT_NORMAL)


# 绘制 XoZ 平面期望高度轨迹 + 洞 + 走廊 (含三点轨迹)
def plot_tunnel_path() -> None:
    xs = np.linspace(*X_RANGE, N_SAMPLES)
    z_ref = compute_height_trajectory(xs)
    x_left = HOLE_X_CENTER - HOLE_HALF_X
    x_right = HOLE_X_CENTER + HOLE_HALF_X

    fig, ax = plt.subplots(figsize=(12, 5))

    # 走廊带 (期望高度 ± 半高)
    ax.fill_between(
        xs, z_ref - CORRIDOR_HALF_HEIGHT, z_ref + CORRIDOR_HALF_HEIGHT,
        color="skyblue", alpha=0.25, label=f"走廊 (高度 ±{CORRIDOR_HALF_HEIGHT:.2f})",
    )

    # 洞 (门洞限高板剖面)
    ax.add_patch(
        Rectangle(
            (x_left, HOLE_BOTTOM), x_right - x_left, HOLE_TOP - HOLE_BOTTOM,
            facecolor="orange", edgecolor="darkorange", alpha=0.75,
            label=f"洞 (限高板, x∈[{x_left:.3f},{x_right:.3f}])",
        )
    )

    # 洞下边沿参考线
    ax.axhline(
        HOLE_BOTTOM, color="darkorange", linestyle="--", linewidth=1.2,
        label=f"洞下边沿 {HOLE_BOTTOM:.4f}",
    )

    # 基座期望高度轨迹 (主轨迹)
    ax.plot(
        xs, z_ref, "b-", linewidth=2.4,
        label=f"期望高度轨迹 (洞内 {HEIGHT_HOLE:.4f})",
    )

    # 三点轨迹: 前/后肢中心按 BODY_REF_OFFSET 与基座错开 (同一条 z_ref(x) 规则)
    if SHOW_THREE_POINTS:
        ax.plot(xs, compute_height_trajectory(xs + BODY_REF_OFFSET), "g--", linewidth=1.2,
                label=f"前肢中心轨迹 (x 偏移 +{BODY_REF_OFFSET:.2f})")
        ax.plot(xs, compute_height_trajectory(xs - BODY_REF_OFFSET), "r--", linewidth=1.2,
                label=f"后肢中心轨迹 (x 偏移 -{BODY_REF_OFFSET:.2f})")

    # 正常高度 / 洞内期望高度参考线
    ax.axhline(HEIGHT_NORMAL, color="gray", linestyle=":", linewidth=1.2,
               label=f"正常高度 {HEIGHT_NORMAL:.2f}")
    ax.axhline(HEIGHT_HOLE, color="red", linestyle=":", linewidth=1.2,
               label=f"洞内期望高度 {HEIGHT_HOLE:.4f}")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Tunnel 简化建模: 期望高度轨迹 (XoZ 平面)")
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
