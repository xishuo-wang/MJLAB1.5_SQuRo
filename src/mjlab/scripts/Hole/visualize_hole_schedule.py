from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import torch

from mjlab.tasks.SQuRo_Hole.mdp.command import (
    compute_cmd_vel_x,
    get_height_mode,
)
from mjlab.tasks.SQuRo_Hole.mdp.config import (
    BASE_HEIGHT,
    HEIGHT_LIST,
    MODE_BOTH_HIGH,
    MODE_BOTH_LOW,
    MODE_FRONT_LOW,
    MODE_HIND_LOW,
    THRESHOLD_HEIGHT,
    OBSTACLE_POSITION,
    OBSTACLE_SIZE,
    HEIGHT_INITIAL_MIN,
    VIRTUAL_CLEARANCE_MARGIN,
    BODY_HEIGHT_OFFSET,
    ENTRY_MARGIN,
    EXIT_MARGIN,
    OBSTACLE_HEIGHT_SPLIT,
    HEIGHT_MAX as PREVIEW_HEIGHT_MAX,
    HEIGHT_MIN as PREVIEW_HEIGHT_MIN,
)
from mjlab.tasks.SQuRo_Hole.mdp.curriculums import (
    compute_base_heights,
    obstacle_rule_bounds as _shared_rule_bounds,
)

MODE_LABELS = {
    MODE_BOTH_HIGH: "双高",
    MODE_FRONT_LOW: "前低后高",
    MODE_HIND_LOW: "前高后低",
    MODE_BOTH_LOW: "双低",
}
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "logs/rsl_rl/SQuRo_Hole/schedule_visualization"
)
PREVIEW_X_PADDING = 0.025


@dataclass(frozen=True)
class ScheduleCase:
    label: str
    board: tuple[str, float, float, float, float, float]
    x_min: float
    x_max: float


# 预览真实短板和独立长板示例，长板不进入训练场景。
def build_cases(start_x: float) -> list[ScheduleCase]:
    boards = [
        ("obstacle", *OBSTACLE_POSITION, OBSTACLE_SIZE[0], OBSTACLE_SIZE[2]),
        ("LongPreview", 0.6, OBSTACLE_POSITION[1], HEIGHT_INITIAL_MIN, 0.100, OBSTACLE_SIZE[2]),
    ]
    short = min(boards, key=lambda board: board[4])
    long = max(boards, key=lambda board: board[4])
    if short[4] == long[4]:
        raise ValueError("当前布局需要同时包含短板和长板")
    cases = []
    for label, board in (("短板", short), ("长板", long)):
        index = boards.index(board)
        left = (
            min(0.0, board[1] - board[4] - start_x)
            if index == 0
            else (boards[index - 1][1] + board[1]) / 2 - start_x
        )
        right = (
            (board[1] + boards[index + 1][1]) / 2 - start_x
            if index + 1 < len(boards)
            else board[1] + board[4] - start_x + 0.2
        )
        rule_left, _, rule_right = _shared_rule_bounds(board[1], board[4])
        padding = PREVIEW_X_PADDING
        left = min(left, rule_left - start_x - padding)
        right = max(right, rule_right - start_x + padding)
        cases.append(ScheduleCase(label, board, left, right))
    return cases


# 按板的世界坐标计算生效区间与板中心。
def rule_bounds(board: tuple) -> tuple[float, float, float]:
    _, center_x, _, _, half_length, _ = board
    return _shared_rule_bounds(center_x, half_length)


# 返回前后端共用的中点切换边界。
def low_rule_bounds(board: tuple) -> tuple[float, float, float, float]:
    left, middle, right = rule_bounds(board)
    return left, middle, middle, right


# 收集 base_Link 高度表的全部切换节点。
def rule_knots(board: tuple, start_x: float = 0.0) -> np.ndarray:
    return np.array(rule_bounds(board)) - start_x


# 按 base_Link 的世界 X 查询同一时刻的前后目标高度。
def sample_rule_schedule(
    positions: np.ndarray, board: tuple, bottom_z: float, start_x: float = 0.0
) -> dict[str, np.ndarray]:
    position = np.asarray(positions, dtype=float)
    query_world = position.copy() + start_x
    left, _, right = rule_bounds(board)
    _, front_end, hind_start, _ = low_rule_bounds(board)
    boundaries = np.array([left, hind_start, front_end, right])
    tolerance = 8 * np.finfo(float).eps * max(
        1.0, abs(start_x), float(np.abs(boundaries).max())
    )
    for boundary in boundaries:
        query_world[np.abs(query_world - boundary) <= tolerance] = boundary
    query = torch.as_tensor(query_world, dtype=torch.float64)
    front, hind = compute_base_heights(query, board[1], board[4], bottom_z)
    return {
        "position": position,
        "height_f": front.cpu().numpy(),
        "height_h": hind.cpu().numpy(),
    }


# 按同一个 base_Link 位置计算高度、速度和参考档位。
def sample_base_commands(
    positions: np.ndarray, board: tuple, bottom_z: float, start_x: float = 0.0
) -> dict[str, np.ndarray]:
    position = np.asarray(positions, dtype=float)
    values = sample_rule_schedule(position, board, bottom_z, start_x)
    front = values["height_f"]
    hind = values["height_h"]
    front_tensor = torch.as_tensor(front, dtype=torch.float32)
    hind_tensor = torch.as_tensor(hind, dtype=torch.float32)
    heights = torch.tensor(HEIGHT_LIST)
    return {
        "position": position,
        "base_position_world": position + start_x,
        "height_f": front,
        "height_h": hind,
        "velocity_x": compute_cmd_vel_x(front_tensor, hind_tensor).numpy(),
        "reference_height_f": torch.where(front_tensor < THRESHOLD_HEIGHT, front_tensor,
            heights[(front_tensor[:, None] - heights).abs().argmin(dim=1)]).numpy(),
        "reference_height_h": torch.where(hind_tensor < THRESHOLD_HEIGHT, hind_tensor,
            heights[(hind_tensor[:, None] - heights).abs().argmin(dim=1)]).numpy(),
    }


# 加入切换节点及其右侧点，保留阶跃和右端点包含的口径。
def make_grid(x_min: float, x_max: float, knots: np.ndarray) -> np.ndarray:
    right_probe = knots + 32 * np.finfo(float).eps * np.maximum(1.0, np.abs(knots))
    knots = np.concatenate((knots, right_probe))
    return np.unique(np.concatenate((np.linspace(x_min, x_max, 800), knots[(knots >= x_min) & (knots <= x_max)])))


# 选择可用的中文字体。
def configure_fonts() -> None:
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]
    matplotlib.rcParams["font.sans-serif"] = [name for name in preferred if name in available] + ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


# 绘制高度命令和对应板实体，保持中心高度与表面净空的含义分开。
def draw_panel(
    ax, x_min: float, x_max: float, boards: list, start_x: float, title: str,
    values: dict[str, np.ndarray], knots: np.ndarray,
    xlabel: str = "机体相对复位起点的 X 位移（m）",
):
    from matplotlib.patches import Rectangle

    front_line, = ax.step(values["position"], values["height_f"] * 1000, where="post", color="#2563eb", linewidth=2.3, label="前躯干目标高度")
    hind_line, = ax.step(values["position"], values["height_h"] * 1000, where="post", color="#ea580c", linestyle="--", linewidth=2.3, label="后躯干目标高度")
    rectangles, virtual_lines = [], []
    for index, (_, x, _, bottom, half_length, half_thickness) in enumerate(boards):
        left = x - half_length - start_x
        right = x + half_length - start_x
        rectangle = Rectangle((left, bottom * 1000), 2 * half_length, 2 * half_thickness * 1000,
                              facecolor="#94a3b8", edgecolor="#475569", alpha=0.55,
                              label="限高板实体" if index == 0 else None)
        ax.add_patch(rectangle)
        rectangles.append(rectangle)
        virtual_lines.append(ax.hlines(
            (bottom - VIRTUAL_CLEARANCE_MARGIN) * 1000, left, right,
            colors="#be123c", linestyles="--", linewidth=1.5,
            label="虚拟净空阈值" if index == 0 else None,
        ))
    ax.axhline(THRESHOLD_HEIGHT * 1000, color="#64748b", linestyle=":", linewidth=1, label="高低模式分界")
    markers = []
    for position in knots:
        if x_min < position < x_max:
            markers.append(ax.axvline(position, color="#cbd5e1", linewidth=0.8))
    upper = max(PREVIEW_HEIGHT_MAX + 2 * OBSTACLE_SIZE[2], max(HEIGHT_LIST)) * 1000
    ax.set(xlim=(x_min, x_max), ylim=(0, upper + 12), title=title,
           xlabel=xlabel, ylabel="高度（mm）")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    return SimpleNamespace(front=front_line, hind=hind_line, rectangles=rectangles,
                           virtual_lines=virtual_lines, markers=markers)


# 同步板高、base_Link 游标与同一时刻的前后命令读数。
def add_rule_controls(fig, ax, case: ScheduleCase, artists, start_x: float,
                      column: int, initial_height: float):
    from matplotlib.widgets import Slider

    base_cursor = ax.axvline(case.board[1] - start_x,
                            color="#475569", linestyle=":", linewidth=1.5)
    front_point, = ax.plot([], [], "o", color="#2563eb", markersize=6)
    hind_point, = ax.plot([], [], "o", color="#ea580c", markersize=6)
    info = fig.text(0.075 + column * 0.48, 0.265, "", va="top", fontsize=9, color="#334155")
    height_slider = Slider(fig.add_axes([0.15 + column * 0.47, 0.115, 0.30, 0.025]),
                           "板底（mm）", PREVIEW_HEIGHT_MIN * 1000, PREVIEW_HEIGHT_MAX * 1000,
                           valinit=initial_height * 1000, valfmt="%.1f", valstep=0.1, color="#94a3b8")
    position_slider = Slider(fig.add_axes([0.15 + column * 0.47, 0.065, 0.30, 0.025]),
                             "基座 X（m）", case.x_min, case.x_max,
                             valinit=case.board[1] - start_x, valfmt="%.3f", color="#64748b")
    left, _, right = rule_bounds(case.board)
    _, front_end, hind_start, _ = low_rule_bounds(case.board)
    knots = rule_knots(case.board, start_x)
    grid = make_grid(case.x_min, case.x_max, knots)

    # 按用户规则更新曲线，速度与参考档位继续使用当前任务口径。
    def update(_value: float) -> None:
        bottom = float(height_slider.val) / 1000
        position = float(position_slider.val)
        values = sample_rule_schedule(grid, case.board, bottom, start_x)
        artists.front.set_ydata(values["height_f"] * 1000)
        artists.hind.set_ydata(values["height_h"] * 1000)
        artists.rectangles[0].set_y(bottom * 1000)
        board_left = case.board[1] - case.board[4] - start_x
        board_right = case.board[1] + case.board[4] - start_x
        threshold = (bottom - VIRTUAL_CLEARANCE_MARGIN) * 1000
        artists.virtual_lines[0].set_segments([[(board_left, threshold), (board_right, threshold)]])
        for marker in artists.markers:
            marker_x = float(marker.get_xdata()[0])
            is_outer = any(abs(marker_x - bound) < 1e-9 for bound in (left - start_x, right - start_x))
            marker.set_visible(is_outer or bottom < OBSTACLE_HEIGHT_SPLIT)
        current = sample_base_commands(np.array([position]), case.board, bottom, start_x)
        front, hind = float(current["height_f"][0]), float(current["height_h"][0])
        mode = MODE_LABELS[get_height_mode(front, hind)]
        rule = (f"高板：区间内前后同取 min(h−{BODY_HEIGHT_OFFSET * 1000:g} mm, {BASE_HEIGHT * 1000:g} mm)"
                if bottom >= OBSTACLE_HEIGHT_SPLIT else "低板：按基座 X 在板中点切换前低／后低")
        interval = (
            f"基座区间：[{left - start_x:.3f}, {right - start_x:.3f}] m"
            if bottom >= OBSTACLE_HEIGHT_SPLIT
            else f"前低：[{left - start_x:.3f}, {front_end - start_x:.3f})；后低：[{hind_start - start_x:.3f}, {right - start_x:.3f}] m"
        )
        info.set_text(
            f"{rule}\n"
            f"{interval}\n"
            f"当前基座世界 X：{current['base_position_world'][0]:.3f} m\n"
            f"同时刻目标前／后：{front * 1000:.1f}／{hind * 1000:.1f} mm　{mode}\n"
            f"参考档前／后：{current['reference_height_f'][0] * 1000:.1f}／"
            f"{current['reference_height_h'][0] * 1000:.1f} mm；速度：{current['velocity_x'][0]:.4f} m/s"
        )
        ax.set_title(f"{case.label}新规则：板底={bottom * 1000:.1f} mm，长度={2 * case.board[4] * 1000:.0f} mm")
        base_cursor.set_xdata([position, position])
        front_point.set_data([position], [front * 1000])
        hind_point.set_data([position], [hind * 1000])
        fig.canvas.draw_idle()

    height_slider.on_changed(update)
    position_slider.on_changed(update)
    update(float(position_slider.val))
    return SimpleNamespace(height_slider=height_slider, position_slider=position_slider,
                           info=info, base_cursor=base_cursor,
                           front_point=front_point, hind_point=hind_point)


# 导出当前高度规则、布局和两种板型，便于复核图片来源。
def save_source_data(output_dir: Path, cases: list[ScheduleCase], start_x: float,
                     initial_heights: list[float]) -> None:
    data = {
        "start_x": start_x,
        "reference_height_list": HEIGHT_LIST,
        "virtual_clearance_margin": VIRTUAL_CLEARANCE_MARGIN,
        "rule": {
            "obstacle_height_split": OBSTACLE_HEIGHT_SPLIT,
            "entry_margin": ENTRY_MARGIN,
            "exit_margin": EXIT_MARGIN,
            "body_height_offset": BODY_HEIGHT_OFFSET,
            "low_target_rule": "板底高度减躯干余量，受限侧使用固定角标定表插值",
            "default_height": BASE_HEIGHT,
            "position_source": "base_Link_world_x",
            "front_low_end_inclusive": False,
            "hind_low_start_inclusive": True,
            "right_endpoint_inclusive": True,
            "high_target_capped": True,
        },
        "cases": [{"label": case.label, "board": case.board, "x_min": case.x_min,
                   "x_max": case.x_max, "preview_bottom_z": height,
                   "rule_bounds_world": rule_bounds(case.board),
                   "low_rule_bounds_world": low_rule_bounds(case.board)}
                  for case, height in zip(cases, initial_heights, strict=True)],
    }
    (output_dir / "位置表数据.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# 预览短板与长板随板高变化的当前规则。
def main() -> None:
    parser = argparse.ArgumentParser(description=f"预览以 {OBSTACLE_HEIGHT_SPLIT * 1000:g} mm 为分界的障碍物高度—位置表联动规则")
    parser.add_argument("--no-show", action="store_true", help="仅导出图片，不打开交互窗口")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="图片和数据的输出目录")
    parser.add_argument("--start-x", type=float, default=0.0, help="复位起点的世界 X 坐标（m）")
    parser.add_argument("--short-height", type=float, default=None, help="短板预览的初始板底高度（m）")
    parser.add_argument("--long-height", type=float, default=None, help="长板预览的初始板底高度（m）")
    args = parser.parse_args()
    if not np.isfinite(args.start_x):
        parser.error("复位起点必须是有限数值")
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_fonts()
    cases = build_cases(args.start_x)
    initial_heights = [case.board[3] if height is None else height
                       for case, height in zip(cases, (args.short_height, args.long_height), strict=True)]
    if any(not np.isfinite(height) or not PREVIEW_HEIGHT_MIN <= height <= PREVIEW_HEIGHT_MAX
           for height in initial_heights):
        parser.error(f"预览板底高度须在 [{PREVIEW_HEIGHT_MIN}, {PREVIEW_HEIGHT_MAX}] m 内")
    fig = plt.figure(figsize=(15, 7), facecolor="white")
    grid = fig.add_gridspec(1, 2, left=0.075, right=0.97,
                           top=0.80, bottom=0.36, wspace=0.2)
    controls = []
    for column, (case, height) in enumerate(zip(cases, initial_heights, strict=True)):
        ax = fig.add_subplot(grid[0, column])
        knots = rule_knots(case.board, args.start_x)
        values = sample_rule_schedule(make_grid(case.x_min, case.x_max, knots), case.board, height, args.start_x)
        artists = draw_panel(ax, case.x_min, case.x_max, [case.board], args.start_x,
                             case.label, values, knots,
                             xlabel="base_Link 相对起点的 X 坐标（m）")
        controls.append(add_rule_controls(fig, ax, case, artists, args.start_x, column, height))
    fig.suptitle("障碍物板高与高度—位置表联动预览", fontsize=17, y=0.985)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.95),
               ncol=5, frameon=False, fontsize=9)
    fig.text(0.075, 0.015, "前后目标共用 base_Link 的世界 X；低板在板中心立即切换为前高后低，两个低姿态区间互斥。", fontsize=9, color="#475569")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = args.output_dir / f"短板长板位置表.{suffix}"
        fig.savefig(path, dpi=180, facecolor="white")
        print(f"已保存图像：{path.resolve()}")
    save_source_data(args.output_dir, cases, args.start_x, initial_heights)
    print(f"已保存位置表数据：{(args.output_dir / '位置表数据.json').resolve()}")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
