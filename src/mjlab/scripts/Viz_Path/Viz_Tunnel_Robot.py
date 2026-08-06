# Viz_Tunnel_Robot: Tunnel 任务简化机器人模型 (XoZ 侧视)
# 结构 (三块长方形, 从前往后): 头部 → 前躯干及前脊柱 → 后躯干及后脊柱
# 每块高度中间有一个俯仰关节 (绕 Y 轴, 侧视表现为矩形绕自身中心旋转)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from pathlib import Path

# 中文字体配置 (Windows: Microsoft YaHei / SimHei, macOS: PingFang SC)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 配置 ====================
# --- 三段尺寸 (单位 mm, 绘图时转换为 m) ---
HEAD_LENGTH_MM = 40      # 头部长度 (mm)
HEAD_HEIGHT_MM = 50      # 头部高度 (mm)
FRONT_LENGTH_MM = 100    # 前躯干及前脊柱长度 (mm)
FRONT_HEIGHT_MM = 50     # 前躯干及前脊柱高度 (mm)
REAR_LENGTH_MM = 80      # 后躯干及后脊柱长度 (mm)
REAR_HEIGHT_MM = 50      # 后躯干及后脊柱高度 (mm)

# --- 俯仰关节配置 (rad, 绕块中心 Y 轴旋转; 0 = 水平) ---
HEAD_PITCH = 0.0         # 头部俯仰角
FRONT_PITCH = 0.0        # 前躯干俯仰角
REAR_PITCH = 0.0         # 后躯干俯仰角

# --- 布局 ---
BASE_X = 0.0             # 基准点 X (前躯干与后躯干交界处, m)
BASE_Z = 0.05            # 机身中心高度 (m), 三块中心 Z
JOINT_RADIUS = 0.004     # 关节圆点半径 (m)

# --- 外观 (参考 GUI_Spine_Kinematics 配色) ---
COLOR_HEAD = "#9ECAE1"   # 头部 浅蓝
COLOR_FRONT = "#70B070"  # 前躯干 绿
COLOR_REAR = "#E0A060"   # 后躯干 橙
COLOR_BASE = "#808080"   # base 交界点 灰

# --- 输出 ---
SAVE_PATH = Path(__file__).parent / "Result" / "Viz_Tunnel_Robot.png"


# 计算绕中心旋转后的矩形角点 (XoZ 平面, 绕 Y 轴俯仰)
def rotated_rect_corners(cx: float, cz: float, half_l: float, half_h: float, pitch: float) -> np.ndarray:
    c, s = np.cos(pitch), np.sin(pitch)
    local = np.array([[-half_l, -half_h], [half_l, -half_h], [half_l, half_h], [-half_l, half_h]])
    return local @ np.array([[c, -s], [s, c]]).T + np.array([cx, cz])


# 绘制 Tunnel 简化机器人 (三段长方形 + 俯仰关节)
def draw_tunnel_robot(ax) -> None:
    mm = 1e-3  # mm → m
    half_h = 0.5 * mm  # 通用半高占位 (按各段实际高度重算)

    # 各段几何 (中心 x, 半长, 半高)
    blocks = [
        # (名称, 中心 x, 半长, 半高, 俯仰角, 颜色)
        ("头部", BASE_X + FRONT_LENGTH_MM * mm + HEAD_LENGTH_MM * mm / 2,
         HEAD_LENGTH_MM * mm / 2, HEAD_HEIGHT_MM * mm / 2, HEAD_PITCH, COLOR_HEAD),
        ("前躯干及前脊柱", BASE_X + FRONT_LENGTH_MM * mm / 2,
         FRONT_LENGTH_MM * mm / 2, FRONT_HEIGHT_MM * mm / 2, FRONT_PITCH, COLOR_FRONT),
        ("后躯干及后脊柱", BASE_X - REAR_LENGTH_MM * mm / 2,
         REAR_LENGTH_MM * mm / 2, REAR_HEIGHT_MM * mm / 2, REAR_PITCH, COLOR_REAR),
    ]

    for name, cx, half_l, half_h_i, pitch, color in blocks:
        # 长方形
        pts = rotated_rect_corners(cx, BASE_Z, half_l, half_h_i, pitch)
        ax.add_patch(
            Polygon(pts, closed=True, facecolor=color, edgecolor="black",
                    linewidth=1.5, alpha=0.85, zorder=4)
        )
        # 俯仰关节 (位于块高度中点 = 块中心)
        ax.add_patch(
            Circle((cx, BASE_Z), JOINT_RADIUS, facecolor="white", edgecolor=COLOR_BASE,
                   linewidth=1.5, zorder=6)
        )
        # 标签: 名称 + 尺寸
        ax.text(cx, BASE_Z + half_h_i + 0.006, f"{name}\n{int(half_l*2/mm)}×{int(half_h_i*2/mm)}mm",
                ha="center", va="bottom", fontsize=9, zorder=5)

    # base 交界点 (前/后躯干之间)
    ax.plot(BASE_X, BASE_Z, "o", markersize=5, color=COLOR_BASE, zorder=6)
    ax.annotate("base", (BASE_X, BASE_Z), textcoords="offset points",
                xytext=(-4, -12), ha="right", fontsize=8, color=COLOR_BASE)

    # 地面线
    ax.axhline(0.0, color="black", linestyle="-", linewidth=1.2, zorder=2)
    ax.text(0.005, 0.002, "地面 (z=0)", fontsize=8, color="gray")


# 绘制并展示
def plot_tunnel_robot() -> None:
    mm = 1e-3
    total_len = (HEAD_LENGTH_MM + FRONT_LENGTH_MM + REAR_LENGTH_MM) * mm
    max_h = max(HEAD_HEIGHT_MM, FRONT_HEIGHT_MM, REAR_HEIGHT_MM) * mm

    fig, ax = plt.subplots(figsize=(12, 4))
    draw_tunnel_robot(ax)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Tunnel 简化机器人模型 (XoZ 侧视, 三段长方形 + 高度中点俯仰关节)")
    ax.set_xlim(BASE_X - total_len * 0.2, BASE_X + total_len * 1.25)
    ax.set_ylim(-0.03, BASE_Z + max_h + 0.04)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_aspect("equal")

    plt.tight_layout()
    if SAVE_PATH is not None:
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(SAVE_PATH), dpi=150)
        print(f"[Viz_Tunnel_Robot] 图片已保存: {SAVE_PATH}")
    plt.show()


if __name__ == "__main__":
    plot_tunnel_robot()
