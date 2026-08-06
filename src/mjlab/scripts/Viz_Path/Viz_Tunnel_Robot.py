# Viz_Tunnel_Robot: Tunnel 任务简化机器人模型 (XoZ 侧视)
# 结构 (三块长方形, 从前往后): 头部 → 前躯干及前脊柱 → 后躯干及后脊柱
# 俯仰关节共 2 个, 位于两两相接的边的中点:
#   关节1: 头部与前躯干相接边中点
#   关节2: 前躯干与后躯干相接边中点 (= base 位置)
# 运动学: 后躯干绕关节2, 前躯干绕关节2, 头部绕关节1 (随前躯干联动)
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

# --- 俯仰关节控制角度 (单位: 度, 共 2 个关节各对应一个角度; 0 = 水平) ---
JOINT1_PITCH_DEG = 0.0    # 关节1 (头部-前躯干): 头部绕关节1 俯仰
JOINT2_PITCH_DEG = 20.0   # 关节2 (前-后躯干): 前躯干绕关节2 俯仰 (后躯干保持水平)

# --- 布局 ---
BASE_X = 0.0             # 关节2 X 位置 (前躯干与后躯干相接边, m)
BASE_Z = 0.05            # 机身中心高度 (m), 关节位于高度中点
JOINT_RADIUS = 0.005     # 关节圆点半径 (m)

# --- 外观 (参考 GUI_Spine_Kinematics 配色) ---
COLOR_HEAD = "#9ECAE1"   # 头部 浅蓝
COLOR_FRONT = "#70B070"  # 前躯干 绿
COLOR_REAR = "#E0A060"   # 后躯干 橙
COLOR_BASE = "#808080"   # 关节/base 灰

# --- 输出 ---
SAVE_PATH = Path(__file__).parent / "Result" / "Viz_Tunnel_Robot.png"


# 2D 点绕旋转中心旋转 (XoZ 平面, 绕 Y 轴俯仰)
def rotate_point(p, center, angle: float) -> np.ndarray:
    d = np.asarray(p, dtype=float) - np.asarray(center, dtype=float)
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(center, dtype=float) + np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c])


# 生成矩形角点并绕指定旋转中心旋转, 可附加平移
def rect_points(center, half_l: float, half_h: float, pitch: float,
                rot_center, extra_shift=(0.0, 0.0)) -> np.ndarray:
    corners = np.array(
        [[-half_l, -half_h], [half_l, -half_h], [half_l, half_h], [-half_l, half_h]]
    ) + np.asarray(center)
    out = np.array([rotate_point(p, rot_center, pitch) for p in corners])
    return out + np.asarray(extra_shift)


# 绘制 Tunnel 简化机器人 (三段长方形 + 两两相接边中点俯仰关节 ×2)
def draw_tunnel_robot(ax) -> None:
    mm = 1e-3

    # 关节2 (前-后躯干相接边中点, 固定) / 关节1 初始 (头-前躯干相接边中点)
    j2 = np.array([BASE_X, BASE_Z])
    j1_init = np.array([BASE_X + FRONT_LENGTH_MM * mm, BASE_Z])

    # 各段初始中心 (直行水平时)
    head_center = np.array([j1_init[0] + HEAD_LENGTH_MM * mm / 2, BASE_Z])
    front_center = np.array([j2[0] + FRONT_LENGTH_MM * mm / 2, BASE_Z])
    rear_center = np.array([j2[0] - REAR_LENGTH_MM * mm / 2, BASE_Z])

    # 铰接运动学: 前躯干绕 j2 (JOINT2) → 关节1 新位置 → 头部绕关节1 (JOINT1)
    j1_pitch = np.deg2rad(JOINT1_PITCH_DEG)
    j2_pitch = np.deg2rad(JOINT2_PITCH_DEG)
    j1_new = rotate_point(j1_init, j2, j2_pitch)

    head_pts = rect_points(head_center, HEAD_LENGTH_MM * mm / 2, HEAD_HEIGHT_MM * mm / 2,
                           j1_pitch, j1_init, j1_new - j1_init)
    front_pts = rect_points(front_center, FRONT_LENGTH_MM * mm / 2, FRONT_HEIGHT_MM * mm / 2,
                            j2_pitch, j2)
    rear_pts = rect_points(rear_center, REAR_LENGTH_MM * mm / 2, REAR_HEIGHT_MM * mm / 2,
                           0.0, j2)  # 后躯干保持水平

    # 三段长方形
    blocks = [
        ("头部", head_pts, HEAD_LENGTH_MM, HEAD_HEIGHT_MM, COLOR_HEAD),
        ("前躯干及前脊柱", front_pts, FRONT_LENGTH_MM, FRONT_HEIGHT_MM, COLOR_FRONT),
        ("后躯干及后脊柱", rear_pts, REAR_LENGTH_MM, REAR_HEIGHT_MM, COLOR_REAR),
    ]
    for name, pts, len_mm, h_mm, color in blocks:
        ax.add_patch(
            Polygon(pts.tolist(), closed=True, facecolor=color, edgecolor="black",
                    linewidth=1.5, alpha=0.85, zorder=4)
        )
        ctr = pts.mean(axis=0)
        ax.text(float(ctr[0]), float(ctr[1]) + h_mm * mm / 2 + 0.006,
                f"{name}\n{int(len_mm)}×{int(h_mm)}mm",
                ha="center", va="bottom", fontsize=9, zorder=5)

    # 两个俯仰关节 (两两相接边中点)
    for j, name in [
        (j2, "关节2\n(前-后躯干)"),
        (j1_new, "关节1\n(头部-前躯干)"),
    ]:
        ax.add_patch(
            Circle((float(j[0]), float(j[1])), JOINT_RADIUS, facecolor="white", edgecolor=COLOR_BASE,
                   linewidth=1.5, zorder=6)
        )
        ax.annotate(name, (float(j[0]), float(j[1])), textcoords="offset points",
                    xytext=(-2, -14), ha="center", fontsize=8, color=COLOR_BASE)

    # base 标注 (关节2 即 base 位置)
    ax.text(float(j2[0]), float(j2[1]) + JOINT_RADIUS + 0.002, "base", ha="center",
            va="bottom", fontsize=8, color=COLOR_BASE)

    # 地面线
    ax.axhline(0.0, color="black", linestyle="-", linewidth=1.2, zorder=2)
    ax.text(0.005, 0.002, "地面 (z=0)", fontsize=8, color="gray")


# 绘制并展示
def plot_tunnel_robot() -> None:
    mm = 1e-3
    total_len = (HEAD_LENGTH_MM + FRONT_LENGTH_MM + REAR_LENGTH_MM) * mm
    max_h = max(HEAD_HEIGHT_MM, FRONT_HEIGHT_MM, REAR_HEIGHT_MM) * mm

    fig, ax = plt.subplots(figsize=(12, 4.5))
    draw_tunnel_robot(ax)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Tunnel 简化机器人模型 (XoZ 侧视, 三段长方形 + 相接边中点俯仰关节×2)")
    ax.set_xlim(BASE_X - total_len * 0.25, BASE_X + total_len * 1.25)
    ax.set_ylim(-0.03, BASE_Z + max_h + 0.05)
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
