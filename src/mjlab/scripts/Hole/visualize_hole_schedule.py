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
    get_height_mode,
)
from mjlab.tasks.SQuRo_Hole.mdp.config import (
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


# 把原始切换节点加入采样网格，保留位置表的阶跃边界。
def make_grid(x_min: float, x_max: float) -> np.ndarray:
    knots = np.array([row[0] for row in STAGE3_POSITION_SCHEDULE])
    return np.unique(np.concatenate((np.linspace(x_min, x_max, 800), knots[(knots >= x_min) & (knots <= x_max)])))


# 选择可用的中文字体。
def configure_fonts() -> None:
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]
    matplotlib.rcParams["font.sans-serif"] = [name for name in preferred if name in available] + ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


# 绘制高度命令和对应板实体，保持中心高度与表面净空的含义分开。
def draw_panel(ax, x_min: float, x_max: float, boards: list, start_x: float, title: str) -> None:
    from matplotlib.patches import Rectangle

    values = sample_schedule(make_grid(x_min, x_max))
    ax.step(values["position"], values["height_f"] * 1000, where="post", color="#2563eb", linewidth=2.3, label="前躯干目标高度")
    ax.step(values["position"], values["height_h"] * 1000, where="post", color="#ea580c", linestyle="--", linewidth=2.3, label="后躯干目标高度")
    for index, (_, x, _, bottom, half_length, half_thickness) in enumerate(boards):
        left = x - half_length - start_x
        right = x + half_length - start_x
        ax.add_patch(Rectangle((left, bottom * 1000), 2 * half_length, 2 * half_thickness * 1000,
                               facecolor="#94a3b8", edgecolor="#475569", alpha=0.55,
                               label="限高板实体" if index == 0 else None))
        ax.hlines((bottom - VIRTUAL_CLEARANCE_MARGIN) * 1000, left, right,
                  colors="#be123c", linestyles="--", linewidth=1.5,
                  label="虚拟净空阈值" if index == 0 else None)
    ax.axhline(THRESHOLD_HEIGHT * 1000, color="#64748b", linestyle=":", linewidth=1, label="高低模式分界")
    for position, _, _ in STAGE3_POSITION_SCHEDULE:
        if x_min < position < x_max:
            ax.axvline(position, color="#cbd5e1", linewidth=0.8)
    upper = max(max(board[3] + 2 * board[5] for board in HOLE_LAYOUT), max(HEIGHT_LIST)) * 1000
    ax.set(xlim=(x_min, x_max), ylim=(0, upper + 12), title=title,
           xlabel="相对复位起点的 X 位移（m）", ylabel="高度（mm）")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


# 添加位置滑块，读取当前位置的高度、速度、模式与参考档位。
def add_position_slider(fig, ax, case: ScheduleCase, bounds: list[float], info_x: float):
    from matplotlib.widgets import Slider

    cursor = ax.axvline(case.x_min, color="#0f172a", linestyle="--", linewidth=1)
    info = fig.text(info_x, 0.1, "", fontsize=9, color="#334155")
    slider = Slider(fig.add_axes(bounds), "位置（m）", case.x_min, case.x_max,
                    valinit=case.x_min, valfmt="%.3f", color="#64748b")

    # 按生产查表结果更新游标和读数。
    def update(position: float) -> None:
        values = sample_schedule(np.array([position]))
        front, hind = float(values["height_f"][0]), float(values["height_h"][0])
        mode = MODE_LABELS[get_height_mode(front, hind)]
        info.set_text(
            f"目标前／后：{front * 1000:.1f}／{hind * 1000:.1f} mm　{mode}\n"
            f"参考档前／后：{values['reference_height_f'][0] * 1000:.1f}／"
            f"{values['reference_height_h'][0] * 1000:.1f} mm\n"
            f"速度命令：{values['velocity_x'][0]:.4f} m/s"
        )
        cursor.set_xdata([position, position])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(case.x_min)
    return slider


# 导出当前源码中的位置表、布局和两种情况，便于复核图片来源。
def save_source_data(output_dir: Path, cases: list[ScheduleCase], start_x: float) -> None:
    data = {
        "start_x": start_x,
        "position_schedule": STAGE3_POSITION_SCHEDULE,
        "hole_layout": HOLE_LAYOUT,
        "reference_height_list": HEIGHT_LIST,
        "virtual_clearance_margin": VIRTUAL_CLEARANCE_MARGIN,
        "cases": [{"label": case.label, "board": case.board, "x_min": case.x_min, "x_max": case.x_max} for case in cases],
    }
    (output_dir / "位置表数据.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# 生成当前短板与长板位置表的交互图和静态图片。
def main() -> None:
    parser = argparse.ArgumentParser(description="查看现有短板和长板对应的高度—位置表")
    parser.add_argument("--no-show", action="store_true", help="仅导出图片，不打开交互窗口")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="图片和数据的输出目录")
    parser.add_argument("--start-x", type=float, default=0.0, help="复位起点的世界 X 坐标（m）")
    args = parser.parse_args()
    if not np.isfinite(args.start_x):
        parser.error("复位起点必须是有限数值")
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_fonts()
    cases = build_cases(args.start_x)
    fig = plt.figure(figsize=(15, 9), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=(0.9, 1), left=0.075, right=0.97,
                           top=0.86, bottom=0.25, hspace=0.48, wspace=0.2)
    overview = fig.add_subplot(grid[0, :])
    x_min = min(0.0, min(board[1] - board[4] for board in HOLE_LAYOUT) - args.start_x)
    x_max = max(STAGE3_POSITION_SCHEDULE[-1][0], max(board[1] + board[4] for board in HOLE_LAYOUT) - args.start_x) + 0.1
    draw_panel(overview, x_min, x_max, list(HOLE_LAYOUT), args.start_x, "当前完整位置表与三块板的位置关系")
    sliders = []
    for column, case in enumerate(cases):
        ax = fig.add_subplot(grid[1, column])
        _, x, _, bottom, half_length, _ = case.board
        title = f"{case.label}：世界 X={x:.3f} m，板底={bottom * 1000:.0f} mm，长度={2 * half_length * 1000:.0f} mm"
        draw_panel(ax, case.x_min, case.x_max, [case.board], args.start_x, title)
        sliders.append(add_position_slider(fig, ax, case, [0.13 + column * 0.48, 0.055, 0.32, 0.025], 0.075 + column * 0.48))
    fig.suptitle("现有短板／长板的高度—位置表", fontsize=17, y=0.985)
    fig.legend(*overview.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.52, 0.95), ncol=5, frameon=False, fontsize=10)
    fig.text(0.075, 0.015, "目标曲线表示躯干中心高度；板底表示表面净空。当前展示原始配置，连续板高的联动规则尚未定义。", fontsize=9, color="#475569")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = args.output_dir / f"短板长板位置表.{suffix}"
        fig.savefig(path, dpi=180, facecolor="white")
        print(f"已保存图像：{path.resolve()}")
    save_source_data(args.output_dir, cases, args.start_x)
    print(f"已保存位置表数据：{(args.output_dir / '位置表数据.json').resolve()}")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
