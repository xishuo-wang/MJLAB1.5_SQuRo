"""Phase 1 绕杆 LUT 轨迹生成可视化

展示 mdp/path.py 中 _generate_slalom_lut_one_period 的构造过程:
  - 6 段: CW弧 → CCW弧 → 直行 → CCW弧 → CW弧 → 直行 (一个周期)
  - 弧段圆心/辅助圆/直行段
  - 杆位置 (y = POLE_Y) 与弧的几何关系
  - 接近段 (机器人 X=-_INIT_DIST → 路径原点)
  - κ(s) 与 heading(s) 沿弧长分布

用法:
  uv run python src/mjlab/scripts/plot_slalom_lut.py [--spacing 0.15]
      [--n_periods 1] [--out xxx.png] [--no-show]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from mjlab.tasks.SQuRo_Slalom.mdp.path import (
    _RMIN,
    _INIT_DIST,
    CORRIDOR_HALF_WIDTH,
    _generate_slalom_lut_one_period,
)
from mjlab.tasks.SQuRo_Slalom.mdp.pole import POLE_Y, POLE_RADIUS, POLE_NUM
from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import CURVATURE_TARGET
from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import CURVATURE_TARGET_MAX
from mjlab.tasks.SQuRo_Slalom.mdp.command import FIXED_VEL, VEL_MIN


EPISODE_LEN = 20.0          # episode 时长 (s, 与 env_cfg 一致)
GAIT_FREQ = 1.0             # Phase 1 使用的步频 (Hz, 文件开头统一定义)


# 与 path.py 内部 _arc_np 相同的单段弧生成 (用于画辅助圆/圆心)
def _arc_np(start, end, r, clockwise, steps=15):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    mid = (start + end) / 2
    chord_vec = end - start
    chord_len = np.linalg.norm(chord_vec)
    d = np.sqrt(max(0.0, r ** 2 - (chord_len / 2) ** 2))
    perp = np.array([-chord_vec[1], chord_vec[0]]) / chord_len
    sign = -1 if clockwise else 1
    center = mid + sign * d * perp
    v_s = start - center
    v_e = end - center
    a_s = np.arctan2(v_s[1], v_s[0])
    a_e = np.arctan2(v_e[1], v_e[0])
    if clockwise:
        if a_e > a_s:
            a_e -= 2 * np.pi
    else:
        if a_e < a_s:
            a_e += 2 * np.pi
    theta = np.linspace(a_s, a_e, steps)
    return center[0] + r * np.cos(theta), center[1] + r * np.sin(theta), center


SEG_STYLES = [
    ("S1: CW 90°弧", "#E41A1C"),
    ("S2: CCW 90°弧", "#FF7F00"),
    ("S3: 直行", "#4DAF4A"),
    ("S4: CCW 90°弧", "#377EB8"),
    ("S5: CW 90°弧", "#984EA3"),
    ("S6: 直行", "#A65628"),
]


def build_segments(spacing, n_arc_pts=15):
    """与 _generate_slalom_lut_one_period 相同的分段构造, 返回每段 (名称, 点集, 圆心或None)"""
    r = _RMIN
    x0, y0 = 0.0, 0.0
    pts_x, pts_y = [x0], [y0]
    segs = []

    def add_arc(start, end, r, cw, name):
        ax, ay, center = _arc_np(start, end, r, cw, steps=n_arc_pts)
        segs.append((name, ax[1:], ay[1:], center))
        pts_x.extend(ax[1:])
        pts_y.extend(ay[1:])

    def add_straight(start, end, name):
        sx = np.linspace(start[0], end[0], 8)[1:]
        sy = np.full_like(sx, start[1])
        segs.append((name, sx, sy, None))
        pts_x.extend(sx)
        pts_y.extend(sy)

    arc_len = r * np.pi / 2
    straight_len = spacing - 2 * r
    n_straight = max(2, int(straight_len / arc_len * n_arc_pts)) if straight_len > 1e-9 else 0

    add_arc((x0, y0), (x0 + r, y0 - r), r, True, "S1: CW 90°弧")
    add_arc((x0 + r, y0 - r), (x0 + 2 * r, y0 - 2 * r), r, False, "S2: CCW 90°弧")
    if n_straight > 0:
        add_straight((x0 + 2 * r, y0 - 2 * r), (x0 + spacing, y0 - 2 * r), "S3: 直行")
    add_arc((x0 + spacing, y0 - 2 * r), (x0 + spacing + r, y0 - r), r, False, "S4: CCW 90°弧")
    add_arc((x0 + spacing + r, y0 - r), (x0 + spacing + 2 * r, y0), r, True, "S5: CW 90°弧")
    if n_straight > 0:
        add_straight((x0 + spacing + 2 * r, y0), (x0 + 2 * spacing, y0), "S6: 直行")
    return segs


def main():
    parser = argparse.ArgumentParser(description="Phase1 绕杆 LUT 轨迹生成可视化")
    parser.add_argument("--spacing", type=float, default=0.15, help="杆间距 (m)")
    parser.add_argument("--n_periods", type=int, default=1, help="展示周期数")
    parser.add_argument("--out", type=str, default="slalom_lut_diagram.png", help="输出路径")
    parser.add_argument("--no-show", action="store_true", help="不弹窗展示, 直接保存")
    args = parser.parse_args()

    spacing = args.spacing
    n_periods = args.n_periods
    r = _RMIN

    # 真实 LUT (用于 κ/heading 曲线)
    arc, xs, ys, headings, kappa = _generate_slalom_lut_one_period(spacing)
    arc = np.asarray(arc)

    # 分段构造 (用于着色/圆心标注)
    segs = build_segments(spacing)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.6, 1.0], hspace=0.35, wspace=0.25)
    ax_path = fig.add_subplot(gs[0, :])
    ax_kappa = fig.add_subplot(gs[1, 0])
    ax_head = fig.add_subplot(gs[1, 1])
    ax_s = fig.add_subplot(gs[1, 2])

    # ============ 主图: 路径构造 ============
    # 接近段
    ax_path.plot([-_INIT_DIST, 0.0], [0.0, 0.0], "--", color="green", linewidth=2.0)
    ax_path.annotate("接近段", xy=(-_INIT_DIST / 2, 0.004), ha="center", fontsize=9, color="green")
    ax_path.scatter([-_INIT_DIST], [0.0], marker="o", color="green", s=70, zorder=6)

    # 周期路径 (分段着色)
    for k in range(n_periods):
        off_x = k * 2 * spacing
        for (name, sx, sy, center) in segs:
            color = dict(SEG_STYLES)[name]
            ax_path.plot(sx + off_x, sy, "-", color=color, linewidth=2.5)
            if center is not None:
                # 辅助圆 + 圆心
                circ = Circle((center[0] + off_x, center[1]), radius=r, fill=False,
                              linestyle=":", edgecolor=color, alpha=0.5, linewidth=1.0)
                ax_path.add_patch(circ)
                ax_path.scatter([center[0] + off_x], [center[1]], marker="x",
                                color=color, s=45, zorder=5)
        # 周期起点/终点
        ax_path.scatter([off_x], [0.0], marker="o", color="black", s=35, zorder=6)
        ax_path.scatter([off_x + 2 * spacing], [0.0], marker="s", color="black", s=35, zorder=6)

    # 段标注 (第一个周期, 取每段中点)
    for (name, sx, sy, center) in segs:
        mid = len(sx) // 2
        ax_path.annotate(name, xy=(sx[mid], sy[mid]), xytext=(sx[mid], sy[mid] + 0.012),
                         ha="center", fontsize=8.5, color=dict(SEG_STYLES)[name],
                         arrowprops=dict(arrowstyle="-", color=dict(SEG_STYLES)[name], lw=0.8))

    # 杆
    for i in range(POLE_NUM):
        px, py = i * spacing, POLE_Y
        if px > 2 * spacing * n_periods + 0.02:
            break
        ax_path.add_patch(Circle((px, py), radius=POLE_RADIUS, facecolor="gray",
                                 edgecolor="black", zorder=4))
        if i < 3:
            ax_path.annotate(f"pole{i}", xy=(px, py - 0.013), ha="center", fontsize=8)

    # 走廊带 (第一周期)
    th = np.linspace(0.0, np.pi, 200)
    # 走廊简化: 沿路径中心线两侧 ±C (逐点法向)
    for k in range(min(n_periods, 1)):
        off_x = k * 2 * spacing
        xs_arr = np.asarray(xs) + off_x
        ys_arr = np.asarray(ys)
        dx = np.gradient(xs_arr)
        dy = np.gradient(ys_arr)
        nrm = np.hypot(dx, dy) + 1e-9
        nx = -dy / nrm
        ny = dx / nrm
        ax_path.plot(xs_arr + CORRIDOR_HALF_WIDTH * nx, ys_arr + CORRIDOR_HALF_WIDTH * ny,
                     "--", color="lightsteelblue", linewidth=1.0)
        ax_path.plot(xs_arr - CORRIDOR_HALF_WIDTH * nx, ys_arr - CORRIDOR_HALF_WIDTH * ny,
                     "--", color="lightsteelblue", linewidth=1.0)

    ax_path.set_aspect("equal")
    ax_path.grid(True, alpha=0.3)
    ax_path.set_xlabel("X (m)")
    ax_path.set_ylabel("Y (m)")
    ax_path.set_title(
        f"Phase1 绕杆 LUT 轨迹构造 (spacing={spacing}m, Rmin={r:.3f}m, κ={CURVATURE_TARGET:.0f}, "
        f"{n_periods} 周期)",
        fontsize=12, fontweight="bold")

    # ============ 子图: κ(s) ============
    ax_kappa.plot(arc, kappa, "-", color="crimson", linewidth=1.8)
    ax_kappa.axhline(CURVATURE_TARGET, color="gray", linestyle="--", linewidth=0.8)
    ax_kappa.axhline(-CURVATURE_TARGET, color="gray", linestyle="--", linewidth=0.8)
    ax_kappa.fill_between(arc, kappa, 0, color="crimson", alpha=0.15)
    ax_kappa.set_xlabel("弧长 s (m)")
    ax_kappa.set_ylabel("κ (rad/m)")
    ax_kappa.set_title(f"曲率沿弧长分布 (±{CURVATURE_TARGET:.0f} / 直行 0)")
    ax_kappa.grid(True, alpha=0.3)
    ax_kappa.set_ylim(-CURVATURE_TARGET * 1.2, CURVATURE_TARGET * 1.2)

    # ============ 子图: heading(s) ============
    h_unwrapped = np.unwrap(headings)
    ax_head.plot(arc, np.rad2deg(h_unwrapped), "-", color="navy", linewidth=1.8)
    ax_head.set_xlabel("弧长 s (m)")
    ax_head.set_ylabel("heading (°)")
    ax_head.set_title("朝向沿弧长分布 (unwrapped)")
    ax_head.grid(True, alpha=0.3)

    # ============ 子图: 曲率随时间变化 (期望速度驱动) ============
    scale_phase1 = 1.0 - (1.0 - VEL_MIN) * CURVATURE_TARGET / CURVATURE_TARGET_MAX
    s_period = float(arc[-1])
    vel = FIXED_VEL * GAIT_FREQ * scale_phase1     # 期望速度 (恒速)
    t_arr = np.linspace(0.0, EPISODE_LEN, 800)
    s_arr = vel * t_arr - _INIT_DIST               # 弧长: 接近段为负, 之后进入 LUT
    kappa_t = np.zeros_like(s_arr)
    pos_mask = s_arr >= 0.0
    s_pos = s_arr[pos_mask]
    s_mod = s_pos % s_period
    idx = np.searchsorted(arc, s_mod).clip(1, len(arc) - 1)
    frac = (s_mod - arc[idx - 1]) / (arc[idx] - arc[idx - 1] + 1e-12)
    kappa_t[pos_mask] = kappa[idx - 1] + frac * (kappa[idx] - kappa[idx - 1])
    ax_s.plot(t_arr, kappa_t, "-", color="crimson", linewidth=1.6,
              label=f"κ(t), vel={vel:.4f}m/s (gait={GAIT_FREQ:.1f}Hz)")
    ax_s.axhline(0.0, color="green", linestyle="--", linewidth=0.8)
    # 周期边界时刻
    for k in (1, 2, 3):
        t_k = (k * s_period + _INIT_DIST) / vel
        if t_k <= EPISODE_LEN:
            ax_s.axvline(t_k, color="gray", linestyle=":", linewidth=0.8)
    # 接近段标注
    t_app = _INIT_DIST / vel
    ax_s.axvspan(0.0, t_app, color="green", alpha=0.12)
    ax_s.annotate(f"接近段\n({t_app:.2f}s)", xy=(t_app / 2, -CURVATURE_TARGET * 0.9),
                  ha="center", fontsize=8, color="green")
    ax_s.set_xlabel("时间 t (s)")
    ax_s.set_ylabel("κ (rad/m)")
    ax_s.set_title(f"曲率随时间变化 (期望速度驱动, episode {EPISODE_LEN:.0f}s)")
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(fontsize=8, loc="upper left")
    ax_s.set_ylim(-CURVATURE_TARGET * 1.2, CURVATURE_TARGET * 1.2)

    # 先展示, 再保存
    if not args.no_show:
        plt.show()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"已保存: {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
