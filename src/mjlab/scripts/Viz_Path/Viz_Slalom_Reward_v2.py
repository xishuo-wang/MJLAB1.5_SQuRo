"""走廊一致性 (Corridor Conformance) 奖励示意图 v2

相对 v1 的修复：
  - 补全缺失的 circle_through / arc_between 函数（v1 无法运行）
  - 色标语义与公式对齐：越界点按奖励 r = exp(-σ·v²) 着色，而非"超出距离"
  - 新增奖励曲面子图：r 随 (d_lat, Δθ) 变化，展示死区 C 与指数衰减
  - 增加几何标注：走廊半宽 C、朝向偏差 Δθ、横向偏差 d_lat、前进方向 +X
  - 轨迹方向改为"右为前"（顺时针旋转 90°，前进方向 = world +X，与实物一致）
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrowPatch, Arc
from matplotlib.colors import Normalize

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===== 真实尺寸参数 =====
CORRIDOR_C = 0.04      # 走廊半宽 (m)，即公式中的死区 C
BODY_L = 0.04          # 包络半长 (m)
BODY_W = 0.035         # 包络半宽 (m)
OFFSET = 0.04          # F/H 中心距 base 偏移 (m)
SIGMA = 50.0           # 课程后期 sigma_corridor = 50

THETA_RANGE = (-np.pi * 0.95, np.pi * 0.95)   # 接近一圈，留小缺口
N_SAMPLE = 40          # 采样密度（示意用 40，避免过密糊图）


# ===== 几何工具：补全 v1 缺失的函数 =====
def circle_through(p1, p2, p3):
    """三点定圆，返回 (圆心, 半径)；三点近似共线时返回中点与半弦长"""
    p1, p2, p3 = map(np.asarray, (p1, p2, p3))
    a = p2 - p1
    b = p3 - p1
    denom = 2.0 * (a[0] * b[1] - a[1] * b[0])
    if abs(denom) < 1e-12:                     # 共线退化：取线性中点
        center = (p1 + p3) / 2.0
        return center, np.linalg.norm(p3 - p1) / 2.0
    a2 = np.dot(a, a)
    b2 = np.dot(b, b)
    ux = (b[1] * a2 - a[1] * b2) / denom
    uy = (a[0] * b2 - b[0] * a2) / denom
    center = p1 + np.array([ux, uy])
    return center, np.linalg.norm(p1 - center)


def arc_between(p1, p2, p3, n=80):
    """从 p1 经 p2 到 p3 的圆弧插值点；共线时退化为直线插值"""
    c, r = circle_through(p1, p2, p3)
    d1 = p1 - c
    d2 = p2 - c
    d3 = p3 - c
    a1 = np.arctan2(d1[1], d1[0])
    a2 = np.arctan2(d2[1], d2[0])
    a3 = np.arctan2(d3[1], d3[0])
    if a2 < a1:
        a2 += 2 * np.pi
    while a3 < a2:
        a3 += 2 * np.pi
    ang = np.linspace(a1, a3, n)
    return c + r * np.column_stack([np.cos(ang), np.sin(ang)])


# ===== 走廊一致性奖励（与 mdp/rewards.py 一致） =====
def corridor_reward(d_lat, delta_theta, L=BODY_L, W=BODY_W, C=CORRIDOR_C, sigma=SIGMA):
    """r = exp(-σ·max(0, e-C)²)，e = |d_lat| + |L·sinΔθ| + |W·cosΔθ|"""
    e = np.abs(d_lat) + L * np.abs(np.sin(delta_theta)) + W * np.abs(np.cos(delta_theta))
    v = np.maximum(0.0, e - C)
    return np.exp(-sigma * v ** 2)


def main():
    parser = argparse.ArgumentParser(description="走廊一致性奖励示意图 v2")
    parser.add_argument("--radius", type=float, default=0.15)
    parser.add_argument("--delta_theta", type=float, default=28.0)
    parser.add_argument("--d_lat", type=float, default=0.008)
    parser.add_argument("--spine_angle", type=float, default=12.0)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--out", type=str, default="corridor_reward_diagram_v2.png")
    args = parser.parse_args()

    R = args.radius
    delta_theta = np.deg2rad(args.delta_theta)
    spine_angle = np.deg2rad(args.spine_angle)
    d_lat = args.d_lat
    sigma = args.sigma

    # ---- 路径与走廊带（原始坐标，xOy：+X 右为前） ----
    theta = np.linspace(*THETA_RANGE, 800)
    path_x, path_y = R * np.cos(theta), R * np.sin(theta)
    inner_x, inner_y = (R - CORRIDOR_C) * np.cos(theta), (R - CORRIDOR_C) * np.sin(theta)
    outer_x, outer_y = (R + CORRIDOR_C) * np.cos(theta), (R + CORRIDOR_C) * np.sin(theta)

    # ---- 机器人位姿 ----
    theta_b = 0.0
    base = np.array([R * np.cos(theta_b), R * np.sin(theta_b)])
    radial = np.array([np.cos(theta_b), np.sin(theta_b)])
    tangent = np.array([-np.sin(theta_b), np.cos(theta_b)])   # 原始坐标中切向 = +Y
    base = base + radial * d_lat

    beta = np.arctan2(tangent[1], tangent[0]) + delta_theta

    def u(angle):
        return np.array([np.cos(angle), np.sin(angle)])

    f_dir = u(beta + spine_angle)
    h_dir = u(beta - spine_angle)
    f_center = base + OFFSET * f_dir
    h_center = base - OFFSET * h_dir

    def rect_pts(c, half_l, half_w, angle):
        ax_v = u(angle)
        ay_v = np.array([-ax_v[1], ax_v[0]])
        return np.array([
            c + half_l * ax_v + half_w * ay_v,
            c + half_l * ax_v - half_w * ay_v,
            c - half_l * ax_v - half_w * ay_v,
            c - half_l * ax_v + half_w * ay_v,
        ])

    spine_pts = arc_between(h_center, base, f_center)

    # ---- 顺时针旋转 90°：轨迹变为横向，前进方向 = +X（右为前，与实际一致） ----
    def cw_rot(x, y):
        return y, -x

    path_x, path_y = cw_rot(path_x, path_y)
    inner_x, inner_y = cw_rot(inner_x, inner_y)
    outer_x, outer_y = cw_rot(outer_x, outer_y)
    base_x, base_y = cw_rot(base[0], base[1])
    f_center_x, f_center_y = cw_rot(f_center[0], f_center[1])
    h_center_x, h_center_y = cw_rot(h_center[0], h_center[1])
    spine_pts_x, spine_pts_y = cw_rot(spine_pts[:, 0], spine_pts[:, 1])

    f_rect = rect_pts(f_center, BODY_L, BODY_W, beta + spine_angle)
    f_rect_x, f_rect_y = cw_rot(f_rect[:, 0], f_rect[:, 1])
    h_rect = rect_pts(h_center, BODY_L, BODY_W, beta - spine_angle)
    h_rect_x, h_rect_y = cw_rot(h_rect[:, 0], h_rect[:, 1])

    # ---- 越界采样点：按奖励 r = exp(-σ·v²) 着色（v 为该点到走廊边界的径向越界量） ----
    out_x, out_y, out_v = [], [], []
    for c, ang in [(f_center, beta + spine_angle), (h_center, beta - spine_angle)]:
        ax_v = u(ang)
        ay_v = np.array([-ax_v[1], ax_v[0]])
        for du in np.linspace(-BODY_L, BODY_L, N_SAMPLE):
            for dv in np.linspace(-BODY_W, BODY_W, N_SAMPLE):
                p = c + du * ax_v + dv * ay_v
                r = np.hypot(p[0], p[1])
                v = max(r - (R + CORRIDOR_C), (R - CORRIDOR_C) - r)
                if v > 0:
                    out_x.append(p[0])
                    out_y.append(p[1])
                    out_v.append(v)

    if out_x:
        out_x_rot, out_y_rot = cw_rot(np.array(out_x), np.array(out_y))
        out_v = np.array(out_v)
        out_r = np.exp(-sigma * out_v ** 2)      # 局部奖励密度
    else:
        out_x_rot, out_y_rot, out_r = [], [], np.array([])

    # ========== 绘图：左 = 几何示意，右 = 奖励曲面 ==========
    fig = plt.figure(figsize=(13.5, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.18)

    # ---------- 左：几何示意 ----------
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("white")
    # 走廊带
    ax1.fill(np.concatenate([outer_x, inner_x[::-1]]),
             np.concatenate([outer_y, inner_y[::-1]]),
             color="whitesmoke", alpha=0.7, zorder=0)
    ax1.plot(outer_x, outer_y, "--", color="gray", linewidth=1.2, alpha=0.8)
    ax1.plot(inner_x, inner_y, "--", color="gray", linewidth=1.2, alpha=0.8)
    # 路径中线
    ax1.plot(path_x, path_y, "--", color="dimgrey", linewidth=2.2, dashes=(8, 4), alpha=0.9, zorder=2)
    # 脊柱弧线
    ax1.plot(spine_pts_x, spine_pts_y, "-", color="#555555", linewidth=2.0, alpha=0.9, zorder=5)
    # 躯干（金色半透明）
    ax1.add_patch(Polygon(np.column_stack([f_rect_x, f_rect_y]),
                          closed=True, facecolor="#FFD700", edgecolor="#B8860B",
                          linewidth=1.6, alpha=0.4, zorder=4))
    ax1.add_patch(Polygon(np.column_stack([h_rect_x, h_rect_y]),
                          closed=True, facecolor="#FFD700", edgecolor="#B8860B",
                          linewidth=1.6, alpha=0.4, zorder=4))
    # 脊柱关节
    ax1.add_patch(Circle((base_x, base_y), radius=0.012, facecolor="dimgrey",
                         edgecolor="black", linewidth=1.5, zorder=6))
    # 越界点：按局部奖励 r 着色（viridis_r：奖励低 = 深紫，奖励高 = 黄）
    if len(out_r) > 0:
        norm = Normalize(0.0, 1.0)
        sc = ax1.scatter(out_x_rot, out_y_rot, s=20, c=out_r, cmap="viridis_r",
                         edgecolors="none", alpha=0.95, norm=norm, zorder=7)
        cbar = fig.colorbar(sc, ax=ax1, fraction=0.045, pad=0.04)
        cbar.set_label("局部奖励 $r = \\exp(-\\sigma v^2)$（越界惩罚密度）", fontsize=10)

    # ---- 几何标注 ----
    # 前进方向 +X（右为前）
    ax1.add_patch(FancyArrowPatch((base_x, base_y), (base_x + 0.055, base_y),
                                  arrowstyle="-|>", mutation_scale=18,
                                  linewidth=2.2, color="#C62828", zorder=8))
    ax1.text(base_x + 0.062, base_y + 0.012, "+X（前）", fontsize=11, color="#C62828", zorder=8)
    # 走廊半宽 C（径向双箭头，取 theta=π/2 处）
    ang_c = np.pi / 2.0
    p_outer = np.array([R + CORRIDOR_C, 0.0]) * 0 + (R + CORRIDOR_C) * np.array([np.cos(ang_c), np.sin(ang_c)])
    p_inner = (R - CORRIDOR_C) * np.array([np.cos(ang_c), np.sin(ang_c)])
    po_x, po_y = cw_rot(p_outer[0], p_outer[1])
    pi_x, pi_y = cw_rot(p_inner[0], p_inner[1])
    ax1.add_patch(FancyArrowPatch((po_x, po_y), (pi_x, pi_y),
                                  arrowstyle="<|-|>", mutation_scale=14,
                                  linewidth=1.6, color="#37474F", zorder=8))
    ax1.text((po_x + pi_x) / 2 + 0.01, (po_y + pi_y) / 2 + 0.006,
             "C（死区半宽）", fontsize=10, color="#37474F", zorder=8)
    # 朝向偏差 Δθ（在 base 处画弧线）
    arc_r = 0.05
    ax1.add_patch(Arc((base_x, base_y), 2 * arc_r, 2 * arc_r, angle=0,
                      theta1=0, theta2=delta_theta * 180 / np.pi,
                      color="#1565C0", linewidth=1.8, zorder=8))
    ax1.text(base_x + 0.055, base_y - 0.028, "Δθ", fontsize=11, color="#1565C0", zorder=8)
    # 横向偏差 d_lat（base 到路径中线的径向小箭头）
    mid_x, mid_y = cw_rot(R * np.cos(theta_b), R * np.sin(theta_b))
    ax1.add_patch(FancyArrowPatch((mid_x, mid_y), (base_x, base_y),
                                  arrowstyle="-|>", mutation_scale=12,
                                  linewidth=1.4, color="#6A1B9A", zorder=8))
    ax1.text((mid_x + base_x) / 2 + 0.012, (mid_y + base_y) / 2 - 0.012,
             "d_lat", fontsize=10, color="#6A1B9A", zorder=8)

    ax1.set_aspect("equal")
    ax1.set_axis_off()
    ax1.set_title("走廊一致性：包络越界惩罚示意（XoY 俯视，右为前 +X）", fontsize=12)

    # ---------- 右：奖励曲面 ----------
    ax2 = fig.add_subplot(gs[1])
    d_lat_grid = np.linspace(-0.02, 0.10, 220)
    dtheta_grid = np.linspace(np.deg2rad(-45), np.deg2rad(45), 220)
    DL, DT = np.meshgrid(d_lat_grid, dtheta_grid)
    REW = corridor_reward(DL, DT, sigma=sigma)
    im = ax2.pcolormesh(DL, DT * 180 / np.pi, REW, cmap="viridis", shading="auto")
    # 死区边界（v=0 ⇔ e=C）
    Cline = ax2.contour(DL, DT * 180 / np.pi, REW, levels=[0.999], colors=["white"], linewidths=1.6)
    ax2.clabel(Cline, fmt={0.999: "死区边界  e = C"}, fontsize=9)
    ax2.set_xlabel("横向偏差 $d_{lat}$ (m)", fontsize=11)
    ax2.set_ylabel("朝向偏差 $\\Delta\\theta$ (°)", fontsize=11)
    ax2.set_title("奖励曲面 $r = \\exp(-\\sigma v^2)$，$\\sigma = %.0f$" % sigma, fontsize=12)
    cb2 = fig.colorbar(im, ax=ax2, fraction=0.045, pad=0.04)
    cb2.set_label("奖励 $r$", fontsize=10)
    ax2.annotate("死区：$e \\leq C \\rightarrow r = 1$（不惩罚）",
                 xy=(0.03, 20), xytext=(0.045, 34),
                 fontsize=9.5, color="#37474F",
                 arrowprops=dict(arrowstyle="->", color="#37474F"))

    fig.suptitle("走廊一致性（Corridor Conformance）奖励示意 ｜  $e = |d_{lat}| + |L\\sin\\Delta\\theta| + |W\\cos\\Delta\\theta|$,  $v = \\max(0, e-C)$",
                 fontsize=13, y=0.99)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"已保存: {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
