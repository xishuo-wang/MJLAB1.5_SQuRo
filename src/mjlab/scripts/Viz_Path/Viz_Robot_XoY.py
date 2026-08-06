import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle


# ==================== 配置 ====================
HEAD_LENGTH_MM = 40         # 头部长度 (mm, 前后方向 X)
HEAD_WIDTH_MM = 30          # 头部宽度 (mm, 左右方向 Y)
FRONT_LENGTH_MM =100        # 前躯干及前脊柱长度 (mm)
FRONT_WIDTH_MM = 75         # 前躯干及前脊柱宽度 (mm)
REAR_LENGTH_MM = 80         # 后躯干及后脊柱长度 (mm)
REAR_WIDTH_MM = 75          # 后躯干及后脊柱宽度 (mm)

JOINT1_YAW_DEG = 0.0        # 关节1 相对角: 头部相对前躯干的偏航 (绕关节1)
JOINT2_YAW_DEG = 0.0        # 关节2 相对角: 前躯干相对后躯干的偏航 (绕关节2, 后躯干保持直行)

BASE_X = 0.0                # 关节2 X 位置 (前躯干与后躯干相接边, m)
BASE_Y = 0.0                # 身体中心线 Y (=0)
JOINT_RADIUS = 0.005        # 关节圆点半径 (m)

COLOR_HEAD = "#9ECAE1"    # 头部 浅蓝
COLOR_FRONT = "#70B070"   # 前躯干 绿
COLOR_REAR = "#E0A060"    # 后躯干 橙
COLOR_BASE = "#808080"    # 关节/base 灰

SAVE_PATH = Path(__file__).parent / "Robot" / "Viz_Robot_XoY.png"       # 保存路径


# 2D 点绕旋转中心旋转 (XoY 平面, 绕 Z 轴偏航)
def rotate_point(p, center, angle: float) -> np.ndarray:
    d = np.asarray(p, dtype=float) - np.asarray(center, dtype=float)
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(center, dtype=float) + np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c])


# 生成矩形角点并绕指定旋转中心旋转, 可附加平移 (半宽对应 Y 方向)
def rect_points(center, half_l: float, half_w: float, yaw: float, rot_center, extra_shift=(0.0, 0.0)) -> np.ndarray:
    corners = np.array([[-half_l, -half_w], [half_l, -half_w], [half_l, half_w], [-half_l, half_w]]) + np.asarray(center)
    out = np.array([rotate_point(p, rot_center, yaw) for p in corners])
    return out + np.asarray(extra_shift)


# 绘制 Slalom 简化机器人 (三段长方形 + 两两相接边中点偏航关节 ×2)
def draw_slalom_robot(ax) -> None:
    mm = 1e-3

    # 关节2 (前-后躯干相接边中点, 固定) / 关节1 初始 (头-前躯干相接边中点)
    j2 = np.array([BASE_X, BASE_Y])
    j1_init = np.array([BASE_X + FRONT_LENGTH_MM * mm, BASE_Y])

    # 各段初始中心 (直行时)
    head_center = np.array([j1_init[0] + HEAD_LENGTH_MM * mm / 2, BASE_Y])
    front_center = np.array([j2[0] + FRONT_LENGTH_MM * mm / 2, BASE_Y])
    rear_center = np.array([j2[0] - REAR_LENGTH_MM * mm / 2, BASE_Y])

    # 铰接运动学 (相对角链式累积): 头部世界角 = JOINT2 + JOINT1
    j1_yaw = np.deg2rad(JOINT1_YAW_DEG)
    j2_yaw = np.deg2rad(JOINT2_YAW_DEG)
    j1_new = rotate_point(j1_init, j2, j2_yaw)

    head_pts = rect_points(head_center, HEAD_LENGTH_MM * mm / 2, HEAD_WIDTH_MM * mm / 2, j1_yaw + j2_yaw, j1_init, j1_new - j1_init)
    front_pts = rect_points(front_center, FRONT_LENGTH_MM * mm / 2, FRONT_WIDTH_MM * mm / 2, j2_yaw, j2)
    rear_pts = rect_points(rear_center, REAR_LENGTH_MM * mm / 2, REAR_WIDTH_MM * mm / 2, 0.0, j2)  # 后躯干保持直行

    # 三段长方形
    blocks = [
        ("头部", head_pts, HEAD_LENGTH_MM, HEAD_WIDTH_MM, COLOR_HEAD),
        ("前躯干及前脊柱", front_pts, FRONT_LENGTH_MM, FRONT_WIDTH_MM, COLOR_FRONT),
        ("后躯干及后脊柱", rear_pts, REAR_LENGTH_MM, REAR_WIDTH_MM, COLOR_REAR),
    ]
    for name, pts, len_mm, w_mm, color in blocks:
        ax.add_patch(Polygon(pts.tolist(), closed=True, facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.85, zorder=4))
        ctr = pts.mean(axis=0)
        ax.text(float(ctr[0]), float(ctr[1]) + w_mm * mm / 2 + 0.004,
                f"{name}\n{int(len_mm)}×{int(w_mm)}mm",
                ha="center", va="bottom", fontsize=9, zorder=5)

    # 两个偏航关节 (两两相接边中点)
    for j, name in [
        (j2, "关节2\n(前-后躯干)"),
        (j1_new, "关节1\n(头部-前躯干)"),
    ]:
        ax.add_patch(Circle((float(j[0]), float(j[1])), JOINT_RADIUS, facecolor="white", edgecolor=COLOR_BASE, linewidth=1.5, zorder=6))
        ax.annotate(name, (float(j[0]), float(j[1])), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8, color=COLOR_BASE)

    # base 标注 (关节2 即 base 位置)
    ax.text(float(j2[0]), float(j2[1]) + JOINT_RADIUS + 0.002, "base", ha="center", va="bottom", fontsize=8, color=COLOR_BASE)

    # 前进方向箭头 (沿头部朝向)
    head_dir = (head_pts[1] + head_pts[2]) / 2 - (head_pts[0] + head_pts[3]) / 2
    head_dir = head_dir / np.linalg.norm(head_dir)
    arrow_tip = np.array([BASE_X, BASE_Y]) + head_dir * (REAR_LENGTH_MM + FRONT_LENGTH_MM) * mm * 0.9
    ax.annotate("", xy=(float(arrow_tip[0]), float(arrow_tip[1])), xytext=(float(BASE_X), float(BASE_Y)), arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(float(arrow_tip[0]), float(arrow_tip[1]) + 0.003, "前进方向", ha="center", fontsize=8, color="black")

    # 期望路径中心线 (X 轴)
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.2, zorder=2)
    ax.text(0.005, -0.018, "路径中心线 (y=0)", fontsize=8, color="gray")


# 绘制并展示
def plot_robot_xoy() -> None:
    mm = 1e-3
    total_len = (HEAD_LENGTH_MM + FRONT_LENGTH_MM + REAR_LENGTH_MM) * mm
    max_w = max(HEAD_WIDTH_MM, FRONT_WIDTH_MM, REAR_WIDTH_MM) * mm

    fig, ax = plt.subplots(figsize=(12, 5))
    draw_slalom_robot(ax)

    ax.set_xlabel("X (前) (m)")
    ax.set_ylabel("Y (左) (m)")
    ax.set_title("Slalom 简化机器人模型 (XoY 俯视, 三段长方形 + 相接边中点偏航关节×2)")
    ax.set_xlim(BASE_X - total_len * 0.25, BASE_X + total_len * 1.25)
    ax.set_ylim(-max_w * 1.6, max_w * 1.6)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_aspect("equal")

    plt.tight_layout()
    if SAVE_PATH is not None:
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(SAVE_PATH), dpi=150)
        print(f"[Viz_Slalom_Robot] 图片已保存: {SAVE_PATH}")
    plt.show()


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC"]
    plt.rcParams["axes.unicode_minus"] = False
    plot_robot_xoy()
