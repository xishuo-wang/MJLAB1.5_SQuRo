import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ==================================================================================================
# 保存配置
SAVE_DIR = r"D:\MuJoCoLab_1.5\src\mjlab\scripts\Figure\Figure"
FILENAME_BASE = "Figure_backup_wall_width"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==================================================================================================
# 数据来源
DEFAULT_RUN = (r"D:\MuJoCoLab_1.5\logs\rsl_rl\SQuRo_Backup"
               r"\2026-09-27_14-33-27_v1")

CLEAR_EASIEST = 0.23
STAGE1_END = 3000

X_MIN, X_MAX, X_STEP = 0, 7000, 1000
Y_MIN, Y_MAX, Y_STEP = 0.075, 0.250, 0.025

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_scalars(run_dir: str) -> EventAccumulator:
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not files:
        raise SystemExit(f"没找到 event 文件: {run_dir}")
    acc = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    acc.Reload()
    return acc


# 可给多个候选标签: 净宽改名前的老 run 记的是 clear_width, 改名后记 wall_width。
# 两者都要能读, 否则初版基线那个 run 就画不出来了。
def series(acc: EventAccumulator, *tags: str) -> tuple[np.ndarray, np.ndarray]:
    have = acc.Tags()["scalars"]
    for tag in tags:
        if tag in have:
            if tag != tags[0]:
                print(f"[INFO] 该 run 用的是旧标签 {tag}")
            ev = acc.Scalars(tag)
            return (np.array([e.step for e in ev], dtype=float),
                    np.array([e.value for e in ev], dtype=float))
    raise SystemExit(f"该 run 没有记录 {' / '.join(tags)}; 无法画图")


def main() -> None:
    ap = argparse.ArgumentParser(description="绘制训练轮次-墙宽自适应课程学习曲线")
    ap.add_argument("--run", default=DEFAULT_RUN, help="训练 run 目录 (含 events.out.tfevents.*)")
    ap.add_argument("--out", default=FILENAME_BASE, help="输出文件名前缀")
    ap.add_argument("--size", type=float, default=5.6, help="正方形图像边长 (英寸)")
    args = ap.parse_args()

    acc = load_scalars(args.run)
    step, clear = series(acc, "Curriculum/wall_width", "Curriculum/clear_width")
    _, d_min = series(acc, "Curriculum/wall_d_min")
    clear_hi = clear + (0.20 - d_min)

    mask = (step >= X_MIN) & (step <= X_MAX)
    step, clear, clear_hi = step[mask], clear[mask], clear_hi[mask]
    if len(step) == 0:
        raise SystemExit(f"{X_MIN} ~ {X_MAX} 轮范围内没有数据。")

    mask2 = step >= STAGE1_END
    step2, clear2, clear_hi2 = step[mask2], clear[mask2], clear_hi[mask2]

    fig, ax = plt.subplots(figsize=(args.size, args.size), dpi=160)
    ax.set_box_aspect(1)

    # 阶段 I 背景与示意线
    ax.axvspan(X_MIN, STAGE1_END, color="#95a5a6", alpha=0.08, zorder=0)
    ax.hlines(CLEAR_EASIEST, X_MIN, STAGE1_END, color="#7f8c8d", lw=1.8, ls="--",
              label="阶段 I：无受限训练", zorder=3)
    ax.axvline(STAGE1_END, color="#7f8c8d", lw=1.1, ls="--", zorder=2)

    # 阶段 II
    if len(step2) > 0:
        ax.fill_between(step2, clear2, clear_hi2, color="#4c78a8", alpha=0.20,
                        label="随机采样净宽范围", zorder=1)
        ax.plot(step2, clear_hi2, color="#4c78a8", lw=1.2, ls=":",
                label="采样上界", zorder=3)
        ax.plot(step2, clear2, color="#c0392b", lw=2.4,
                label="课程最小宽度", zorder=4)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks(np.arange(X_MIN, X_MAX + X_STEP, X_STEP))
    ax.set_yticks(np.arange(Y_MIN, Y_MAX + Y_STEP / 2, Y_STEP))
    ax.set_yticklabels([f"{y:.3f}" for y in np.arange(Y_MIN, Y_MAX + Y_STEP / 2, Y_STEP)])
    ax.grid(alpha=0.28, ls=":", lw=0.7)

    # 坐标轴刻度字体放大
    ax.tick_params(axis="both", labelsize=12)

    # 图例使用中文, 并进一步放大 1.5x
    s = 1.5
    ax.legend(loc="lower left", fontsize=8.5 * s, framealpha=0.90,
              handlelength=2.0 * s, handletextpad=0.8 * s,
              labelspacing=0.5 * s, borderpad=0.4 * s)

    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = os.path.join(SAVE_DIR, f"{args.out}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"[INFO] 已保存 {path}")
    plt.close(fig)

    print(f"[INFO] 横轴 {X_MIN} ~ {X_MAX} iteration")
    if len(step2) > 0:
        print(f"[INFO] 第二阶段课程最小宽度: {clear2[0]:.4f} -> {clear2[-1]:.4f} m")


if __name__ == "__main__":
    main()