# uv run python -B -m mjlab.scripts.Viz_Path.Viz_Tunnel_Cases
# 两种受限情况 (限高板下沿 0.050 / 0.065 m) 的包络对比与预计算表, 只做分析不改训练路径

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mjlab.tasks.SQuRo_Tunnel.mdp.path import (
    FRONT_DOWN_OFF,
    FRONT_UP_OFF,
    REAR_DOWN_OFF,
    REAR_UP_OFF,
    BODY_SEG_HALF_HEIGHT,
    CORRIDOR_FRONT_HALF_HEIGHT,
    HEIGHT_NORMAL,
    HEIGHT_LOW,
    OBSTACLE_LENGTH,
    TUNNEL_BOTTOM,
    TUNNEL_THICKNESS,
    TUNNEL_CASES,
    PASS_CLEARANCE,
    low_height_for_bottom,
    _height_from_tunnels,
)

# 与 command.py 对齐的速度规则常量 (直行, 步频 1 Hz)
BASE_VEL = 0.1
VEL_LOW = 0.05
VEL_STOP = 0.0
HEIGHT_THRESHOLD = (HEIGHT_NORMAL + HEIGHT_LOW) / 2

# 分析范围与分辨率
TUNNEL_X_LEFT = 0.15          # 洞左侧 x (m), 只影响绘图平移
X_MIN, X_MAX = 0.0, 0.30      # x 采样范围 (m)
GRID_N = 601                  # 与预计算表分辨率一致 (1 mm)
SAVE_PNG = Path(__file__).parent / "Path" / "Viz_Tunnel_Cases.png"
SAVE_NPZ = Path(__file__).parent / "Path" / "Tunnel_Envelope_Cases.npz"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    plate_bottom: float
    height_low: float


# 两种受限情况: 只有板底下沿不同, 低高度期望值按几何推导
def build_cases() -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    for i, bottom in enumerate(TUNNEL_CASES):
        low = 0.02 if abs(bottom - TUNNEL_BOTTOM) < 1e-9 else low_height_for_bottom(bottom)
        cases.append(CaseSpec(name=f"case{i + 1}", plate_bottom=bottom, height_low=low))
    return cases


# 单条期望高度方波: 直接调用 path.py 的规则, 保证与代码一致
def square_wave(xs: np.ndarray, down_off: float, up_off: float,
                height_low: float) -> np.ndarray:
    xs_t = torch.tensor(xs, dtype=torch.float32)
    tunnel_xs = torch.full((len(xs_t), 1), TUNNEL_X_LEFT, dtype=torch.float32)
    z = _height_from_tunnels(xs_t, tunnel_xs, down_off, up_off)
    # _height_from_tunnels 用常量 HEIGHT_LOW, 这里按情况替换低段取值
    return torch.where(z < HEIGHT_NORMAL, torch.tensor(height_low), z).numpy().astype(np.float64)


# 预计算包络表: z_front/z_rear [情况, x]
def build_envelope_tables(cases: list[CaseSpec], xs: np.ndarray) -> dict[str, np.ndarray]:
    z_front = np.stack([square_wave(xs, FRONT_DOWN_OFF, FRONT_UP_OFF, c.height_low) for c in cases])
    z_rear = np.stack([square_wave(xs, REAR_DOWN_OFF, REAR_UP_OFF, c.height_low) for c in cases])
    return {
        "x": xs.astype(np.float32),
        "plate_bottom": np.array([c.plate_bottom for c in cases], dtype=np.float32),
        "height_low": np.array([c.height_low for c in cases], dtype=np.float32),
        "z_front": z_front.astype(np.float32),
        "z_rear": z_rear.astype(np.float32),
    }


# 统计单条轨迹的低高度区间与按速度规则的耗时
def low_intervals(xs: np.ndarray, z: np.ndarray, height_low: float) -> list[tuple[float, float]]:
    low = z < (HEIGHT_NORMAL + height_low) / 2
    intervals: list[tuple[float, float]] = []
    start: float | None = None
    for x, is_low in zip(xs, low):
        if is_low and start is None:
            start = float(x)
        elif not is_low and start is not None:
            intervals.append((start, float(x)))
            start = None
    if start is not None:
        intervals.append((start, float(xs[-1])))
    return intervals


# 按 双正常 0.1 / 单低 0.05 / 双低 0 的速度规则积分总耗时
def travel_time(xs: np.ndarray, z_front: np.ndarray, z_rear: np.ndarray,
                height_low: float) -> float:
    thr = (HEIGHT_NORMAL + height_low) / 2
    low_f = z_front < thr
    low_h = z_rear < thr
    speed = np.where(low_f | low_h, VEL_LOW, BASE_VEL)
    speed = np.where(low_f & low_h, VEL_STOP, speed)
    dx = np.diff(xs)
    v_mid = 0.5 * (speed[:-1] + speed[1:])
    dt = np.where(v_mid > 1e-9, dx / np.maximum(v_mid, 1e-9), 0.0)
    return float(dt.sum())


# 绘制单个情况的 XoZ 剖面: 限高板 + 前后肢包络 + 走廊带
def draw_case(ax, xs: np.ndarray, spec: CaseSpec, z_front: np.ndarray, z_rear: np.ndarray) -> None:
    hole_left = TUNNEL_X_LEFT
    hole_right = TUNNEL_X_LEFT + OBSTACLE_LENGTH
    seg = BODY_SEG_HALF_HEIGHT

    # 限高板剖面
    ax.add_patch(Rectangle(
        (hole_left, spec.plate_bottom), hole_right - hole_left, TUNNEL_THICKNESS,
        facecolor="orange", edgecolor="darkorange", alpha=0.9, zorder=5,
        label=f"限高板 下沿 {spec.plate_bottom * 1000:.0f}mm"))
    ax.axhline(spec.plate_bottom, color="darkorange", linestyle="--", linewidth=1.2, zorder=4)
    # 允许的段中心上限 (板底 - 段半高)
    ax.axhline(spec.plate_bottom - seg, color="red", linestyle=":", linewidth=1.3, zorder=4,
               label=f"段中心上限 {(spec.plate_bottom - seg) * 1000:.0f}mm")

    # 前后肢包络 (段中心期望高度)
    ax.plot(xs, z_front, color="#1f77b4", linewidth=2.4, zorder=7,
            label=f"前肢期望高度 (低 {spec.height_low * 1000:.1f}mm)")
    ax.plot(xs, z_rear, color="#d62728", linewidth=2.4, linestyle="--", zorder=7,
            label=f"后肢期望高度 (低 {spec.height_low * 1000:.1f}mm)")

    # 包络带 (段中心 ± 段半高 = 身体占位)
    ax.fill_between(xs, z_front - seg, z_front + seg, color="#1f77b4", alpha=0.12, zorder=2)
    ax.fill_between(xs, z_rear - seg, z_rear + seg, color="#d62728", alpha=0.10, zorder=2)

    # 走廊带 (段中心允许范围 ±(走廊半高 - 段半高))
    band = CORRIDOR_FRONT_HALF_HEIGHT - seg
    ax.fill_between(xs, z_front - band, z_front + band, color="skyblue", alpha=0.22, zorder=1,
                    label=f"走廊 (中心 ±{band * 100:.1f}cm)")

    ax.axhline(0.0, color="black", linewidth=1.0, zorder=3, label="地面")
    ax.axhline(HEIGHT_NORMAL, color="gray", linestyle=":", linewidth=1.1, zorder=3,
               label=f"正常段 {HEIGHT_NORMAL * 1000:.0f}mm")

    ax.set_title(f"{spec.name}: 板底 {spec.plate_bottom * 1000:.0f} mm, 低高度 {spec.height_low * 1000:.1f} mm, "
                 f"通过余量 {(spec.plate_bottom - spec.height_low - seg) * 1000:.1f} mm", fontsize=10)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0.0, 0.10)
    ax.set_ylabel("Z (m)")
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=7, loc="upper left", ncol=2)


# 绘制情况对比与关键时序
def plot_cases() -> None:
    cases = build_cases()
    xs = np.linspace(X_MIN, X_MAX, GRID_N)
    tables = build_envelope_tables(cases, xs)

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=False)
    for i, spec in enumerate(cases):
        draw_case(axes[i], xs, spec, tables["z_front"][i], tables["z_rear"][i])

    # 第三栏: 情况间的差异与时序
    ax = axes[2]
    ax.axis("off")
    lines = ["情况对比 (步频 1 Hz, 巡航规则 双正常 0.1 / 单低 0.05 / 双低 0 m/s)", ""]
    lines.append(f"{'情况':<8}{'板底(mm)':>10}{'低高度(mm)':>12}{'余量(mm)':>10}"
                 f"{'前低区间(mm)':>22}{'后低区间(mm)':>22}{'双低(mm)':>18}{'0..0.25m耗时(s)':>18}")
    for i, spec in enumerate(cases):
        zf, zr = tables["z_front"][i], tables["z_rear"][i]
        thr = (HEIGHT_NORMAL + spec.height_low) / 2
        f_int = low_intervals(xs, zf, spec.height_low)
        r_int = low_intervals(xs, zr, spec.height_low)
        both = (zf < thr) & (zr < thr)
        both_w = float(xs[both].max() - xs[both].min()) if both.any() else 0.0
        t_all = travel_time(xs, zf, zr, spec.height_low)
        f_txt = ",".join(f"[{a * 1000:.0f},{b * 1000:.0f}]" for a, b in f_int)
        r_txt = ",".join(f"[{a * 1000:.0f},{b * 1000:.0f}]" for a, b in r_int)
        lines.append(f"{spec.name:<8}{spec.plate_bottom * 1000:>10.0f}{spec.height_low * 1000:>12.1f}"
                     f"{(spec.plate_bottom - spec.height_low - BODY_SEG_HALF_HEIGHT) * 1000:>10.1f}"
                     f"{f_txt:>22}{r_txt:>22}{both_w * 1000:>18.0f}{t_all:>18.3f}")
    lines.append("")
    lines.append(f"洞左侧 x = {TUNNEL_X_LEFT * 1000:.0f} mm, 洞宽 a = {OBSTACLE_LENGTH * 1000:.0f} mm, "
                 f"过渡段 = 方波 (TRANSITION_LENGTH = 0)")
    lines.append(f"段半高 = {BODY_SEG_HALF_HEIGHT * 1000:.0f} mm, 通过余量常量 PASS_CLEARANCE = "
                 f"{PASS_CLEARANCE * 1000:.0f} mm")
    lines.append("低高度期望值 = 板底 - 段半高 - 通过余量; case1 的 0.020 m 即现有 HEIGHT_LOW")
    ax.text(0.01, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)

    fig.suptitle("Tunnel 两种受限情况的包络 (XoZ)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    SAVE_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(SAVE_PNG), dpi=150)
    print(f"[Viz_Tunnel_Cases] 图片已保存: {SAVE_PNG}")
    np.savez(str(SAVE_NPZ), **tables)
    print(f"[Viz_Tunnel_Cases] 包络表已保存: {SAVE_NPZ} "
          f"(x {tables['x'].shape}, z_front {tables['z_front'].shape})")

    # 控制台复述关键数字
    for i, spec in enumerate(cases):
        zf, zr = tables["z_front"][i], tables["z_rear"][i]
        thr = (HEIGHT_NORMAL + spec.height_low) / 2
        both = (zf < thr) & (zr < thr)
        both_w = float(xs[both].max() - xs[both].min()) if both.any() else 0.0
        print(f"[情况 {spec.name}] 板底 {spec.plate_bottom * 1000:.0f} mm, 低高度 {spec.height_low * 1000:.1f} mm, "
              f"段中心上限 {(spec.plate_bottom - BODY_SEG_HALF_HEIGHT) * 1000:.1f} mm, "
              f"双低区宽 {both_w * 1000:.1f} mm, "
              f"穿过 0..0.25 m 耗时 {travel_time(xs, zf, zr, spec.height_low):.3f} s")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC"]
    plt.rcParams["axes.unicode_minus"] = False
    plot_cases()
