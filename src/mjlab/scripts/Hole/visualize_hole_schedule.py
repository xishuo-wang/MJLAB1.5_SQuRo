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
    STAGE3_POSITION_SCHEDULE,
    HoleCommand,
    HoleCommandCfg,
    compute_command_velocity,
    get_height_mode,
)
from mjlab.tasks.SQuRo_Hole.mdp.config import (
    BASE_HEIGHT,
    CMD_ANGLE_IDS,
    CMD_HEIGHT_F_IDS,
    CMD_HEIGHT_H_IDS,
    CMD_VEL_IDS,
    CMD_VEL_X_IDS,
    HEIGHT_LIST,
    MODE_BOTH_HIGH,
    MODE_BOTH_LOW,
    MODE_FRONT_LOW,
    MODE_HIND_LOW,
    THRESHOLD_HEIGHT,
)
from mjlab.tasks.SQuRo_Hole.mdp.entity import (
    HOLE_LAYOUT,
    VIRTUAL_CLEARANCE_MARGIN,
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
OBSTACLE_HEIGHT_SPLIT = 0.070
ENTRY_MARGIN = 0.015
EXIT_MARGIN = 0.010
BODY_HEIGHT_OFFSET = 0.030
LOW_TARGET_HEIGHT = 0.020
PREVIEW_HEIGHT_MIN = 0.048
PREVIEW_HEIGHT_MAX = 0.090


@dataclass(frozen=True)
class ScheduleCase:
    label: str
    board: tuple[str, float, float, float, float, float]
    x_min: float
    x_max: float


# 从现有布局选取第一块短板和第一块长板，按相邻板中点划分展示区间。
def build_cases(start_x: float) -> list[ScheduleCase]:
    boards = sorted(HOLE_LAYOUT, key=lambda board: board[1])
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
        cases.append(ScheduleCase(label, board, left, right))
    return cases


# 用轻量状态调用生产查表函数，无需创建仿真环境；见 docs/SQuRo_Hole_位置表可视化.md。
def sample_schedule(positions: np.ndarray) -> dict[str, np.ndarray]:
    position = torch.as_tensor(positions, dtype=torch.float32)
    count = len(position)
    command = torch.zeros(count, CMD_ANGLE_IDS + 1)
    probe = SimpleNamespace(
        device="cpu",
        cfg=HoleCommandCfg(),
        robot=SimpleNamespace(
            data=SimpleNamespace(
                root_link_pos_w=torch.stack((position, torch.zeros_like(position), torch.zeros_like(position)), dim=1)
            )
        ),
        start_positions=torch.zeros(count, 3),
        command_tensor=command,
        vel_command_w=command[:, CMD_VEL_IDS],
        height_F_command=command[:, CMD_HEIGHT_F_IDS],
        height_H_command=command[:, CMD_HEIGHT_H_IDS],
        angle_command=command[:, CMD_ANGLE_IDS],
        has_printed_current_cmd=torch.zeros(count, dtype=torch.bool),
    )
    HoleCommand._update_command_from_schedule(
        probe, torch.arange(count), STAGE3_POSITION_SCHEDULE
    )
    heights = torch.tensor(HEIGHT_LIST)
    front = command[:, CMD_HEIGHT_F_IDS]
    hind = command[:, CMD_HEIGHT_H_IDS]
    return {
        "position": position.numpy(),
        "height_f": front.numpy(),
        "height_h": hind.numpy(),
        "velocity_x": command[:, CMD_VEL_X_IDS].numpy(),
        "reference_height_f": heights[(front[:, None] - heights).abs().argmin(dim=1)].numpy(),
        "reference_height_h": heights[(hind[:, None] - heights).abs().argmin(dim=1)].numpy(),
    }


# 按板的世界坐标计算提前压低、中心切换和恢复边界。
def rule_bounds(board: tuple) -> tuple[float, float, float]:
    _, center_x, _, _, half_length, _ = board
    return center_x - half_length - ENTRY_MARGIN, center_x, center_x + half_length + EXIT_MARGIN


# 预览用户给定的两段高度规则；本函数不接入训练命令，见可视化文档。
def sample_rule_schedule(
    positions: np.ndarray, board: tuple, bottom_z: float, start_x: float = 0.0
) -> dict[str, np.ndarray]:
    position = np.asarray(positions, dtype=float)
    left, middle, right = rule_bounds(board)
    world_x = position + start_x
    front = np.full_like(position, BASE_HEIGHT)
    hind = np.full_like(position, BASE_HEIGHT)
    if bottom_z >= OBSTACLE_HEIGHT_SPLIT:
        active = (world_x >= left) & (world_x <= right)
        front[active] = bottom_z - BODY_HEIGHT_OFFSET
        hind[active] = bottom_z - BODY_HEIGHT_OFFSET
    else:
        front[(world_x >= left) & (world_x < middle)] = LOW_TARGET_HEIGHT
        hind[(world_x >= middle) & (world_x <= right)] = LOW_TARGET_HEIGHT
    front_tensor = torch.as_tensor(front, dtype=torch.float32)
    hind_tensor = torch.as_tensor(hind, dtype=torch.float32)
    heights = torch.tensor(HEIGHT_LIST)
    return {
        "position": position,
        "height_f": front,
        "height_h": hind,
        "velocity_x": compute_command_velocity(front_tensor, hind_tensor).numpy(),
        "reference_height_f": heights[(front_tensor[:, None] - heights).abs().argmin(dim=1)].numpy(),
        "reference_height_h": heights[(hind_tensor[:, None] - heights).abs().argmin(dim=1)].numpy(),
    }


# 加入切换节点及其右侧点，保留阶跃和右端点包含的口径。
def make_grid(x_min: float, x_max: float, knots: np.ndarray | None = None) -> np.ndarray:
    if knots is None:
        knots = np.array([row[0] for row in STAGE3_POSITION_SCHEDULE])
    knots = np.concatenate((knots, np.nextafter(knots, np.inf)))
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
    values: dict[str, np.ndarray] | None = None, knots: np.ndarray | None = None,
):
    from matplotlib.patches import Rectangle

    if values is None:
        values = sample_schedule(make_grid(x_min, x_max))
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
    if knots is None:
        knots = np.array([row[0] for row in STAGE3_POSITION_SCHEDULE])
    markers = []
    for position in knots:
        if x_min < position < x_max:
            markers.append(ax.axvline(position, color="#cbd5e1", linewidth=0.8))
    upper = max(PREVIEW_HEIGHT_MAX + max(2 * board[5] for board in HOLE_LAYOUT), max(HEIGHT_LIST)) * 1000
    ax.set(xlim=(x_min, x_max), ylim=(0, upper + 12), title=title,
           xlabel="相对复位起点的 X 位移（m）", ylabel="高度（mm）")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    return SimpleNamespace(front=front_line, hind=hind_line, rectangles=rectangles,
                           virtual_lines=virtual_lines, markers=markers)


# 同步板高滑块、位置滑块、两段目标曲线与当前位置读数。
def add_rule_controls(fig, ax, case: ScheduleCase, artists, start_x: float,
                      column: int, initial_height: float):
    from matplotlib.widgets import Slider

    cursor = ax.axvline(case.board[1] - start_x, color="#0f172a", linestyle="--", linewidth=1)
    info = fig.text(0.075 + column * 0.48, 0.265, "", va="top", fontsize=9, color="#334155")
    height_slider = Slider(fig.add_axes([0.15 + column * 0.47, 0.115, 0.30, 0.025]),
                           "板底（mm）", PREVIEW_HEIGHT_MIN * 1000, PREVIEW_HEIGHT_MAX * 1000,
                           valinit=initial_height * 1000, valfmt="%.1f", valstep=0.1, color="#94a3b8")
    position_slider = Slider(fig.add_axes([0.15 + column * 0.47, 0.065, 0.30, 0.025]),
                             "位置（m）", case.x_min, case.x_max,
                             valinit=case.board[1] - start_x, valfmt="%.3f", color="#64748b")
    left, middle, right = rule_bounds(case.board)
    knots = np.array([left, middle, right]) - start_x
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
            is_middle = abs(float(marker.get_xdata()[0]) - (middle - start_x)) < 1e-9
            marker.set_visible(not is_middle or bottom < OBSTACLE_HEIGHT_SPLIT)
        current = sample_rule_schedule(np.array([position]), case.board, bottom, start_x)
        front, hind = float(current["height_f"][0]), float(current["height_h"][0])
        mode = MODE_LABELS[get_height_mode(front, hind)]
        rule = "高板：双侧取板底减 30 mm" if bottom >= OBSTACLE_HEIGHT_SPLIT else "低板：先前低，过中点后后低"
        info.set_text(
            f"{rule}\n"
            f"生效区间：[{left - start_x:.3f}, {right - start_x:.3f}] m；中点：{middle - start_x:.3f} m\n"
            f"目标前／后：{front * 1000:.1f}／{hind * 1000:.1f} mm　{mode}\n"
            f"参考档前／后：{current['reference_height_f'][0] * 1000:.1f}／"
            f"{current['reference_height_h'][0] * 1000:.1f} mm；速度：{current['velocity_x'][0]:.4f} m/s"
        )
        ax.set_title(f"{case.label}新规则：板底={bottom * 1000:.1f} mm，长度={2 * case.board[4] * 1000:.0f} mm")
        cursor.set_xdata([position, position])
        fig.canvas.draw_idle()

    height_slider.on_changed(update)
    position_slider.on_changed(update)
    update(float(position_slider.val))
    return SimpleNamespace(height_slider=height_slider, position_slider=position_slider,
                           info=info, cursor=cursor)


# 导出当前源码中的位置表、布局和两种情况，便于复核图片来源。
def save_source_data(output_dir: Path, cases: list[ScheduleCase], start_x: float,
                     initial_heights: list[float]) -> None:
    data = {
        "start_x": start_x,
        "position_schedule": STAGE3_POSITION_SCHEDULE,
        "hole_layout": HOLE_LAYOUT,
        "reference_height_list": HEIGHT_LIST,
        "virtual_clearance_margin": VIRTUAL_CLEARANCE_MARGIN,
        "rule": {
            "obstacle_height_split": OBSTACLE_HEIGHT_SPLIT,
            "entry_margin": ENTRY_MARGIN,
            "exit_margin": EXIT_MARGIN,
            "body_height_offset": BODY_HEIGHT_OFFSET,
            "low_target_height": LOW_TARGET_HEIGHT,
            "default_height": BASE_HEIGHT,
            "midpoint_belongs_to": "hind_low",
            "right_endpoint_inclusive": True,
            "high_target_capped": False,
        },
        "cases": [{"label": case.label, "board": case.board, "x_min": case.x_min,
                   "x_max": case.x_max, "preview_bottom_z": height,
                   "rule_bounds_world": rule_bounds(case.board)}
                  for case, height in zip(cases, initial_heights, strict=True)],
    }
    (output_dir / "位置表数据.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# 对照原始位置表，预览短板与长板随板高变化的新规则。
def main() -> None:
    parser = argparse.ArgumentParser(description="预览以 70 mm 为分界的障碍物高度—位置表联动规则")
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
    fig = plt.figure(figsize=(15, 10), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=(0.9, 1), left=0.075, right=0.97,
                           top=0.86, bottom=0.36, hspace=0.52, wspace=0.2)
    overview = fig.add_subplot(grid[0, :])
    x_min = min(0.0, min(board[1] - board[4] for board in HOLE_LAYOUT) - args.start_x)
    x_max = max(STAGE3_POSITION_SCHEDULE[-1][0], max(board[1] + board[4] for board in HOLE_LAYOUT) - args.start_x) + 0.1
    draw_panel(overview, x_min, x_max, list(HOLE_LAYOUT), args.start_x, "原始训练位置表（参照，保持不变）")
    controls = []
    for column, (case, height) in enumerate(zip(cases, initial_heights, strict=True)):
        ax = fig.add_subplot(grid[1, column])
        knots = np.array(rule_bounds(case.board)) - args.start_x
        values = sample_rule_schedule(make_grid(case.x_min, case.x_max, knots), case.board, height, args.start_x)
        artists = draw_panel(ax, case.x_min, case.x_max, [case.board], args.start_x,
                             case.label, values, knots)
        controls.append(add_rule_controls(fig, ax, case, artists, args.start_x, column, height))
    fig.suptitle("障碍物板高与高度—位置表联动预览", fontsize=17, y=0.985)
    fig.legend(*overview.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.52, 0.95), ncol=5, frameon=False, fontsize=10)
    fig.text(0.075, 0.015, "下图为新规则预览：中点进入后低段，右端点仍在区间内；高板直接取 h−30 mm，不设 55 mm 上限。", fontsize=9, color="#475569")
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
