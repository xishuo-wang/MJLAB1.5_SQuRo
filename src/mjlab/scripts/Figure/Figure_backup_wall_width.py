import argparse
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ==================================================================================================
# 保存配置
SAVE_DIR = r"D:\MuJoCoLab_1.5\src\mjlab\scripts\Figure\Figure"
FILENAME_BASE = "Figure_backup_wall_width"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==================================================================================================
# 数据来源: tensorboard 的 Curriculum/* 标量 (每轮一个点, 直接反映训练当时真实生效的值)
DEFAULT_RUN = r"D:\MuJoCoLab_1.5\logs\rsl_rl\SQuRo_Backup\2026-09-27_14-33-27_v1"

# 环境里最宽/最窄的净宽 (m), 用于画采样上下沿; 想换成从常量推导也可以, 这里硬写是为了
# 让本脚本不依赖任务模块 (可直接 python 运行)。
CLEAR_EASIEST = 0.23
CLEAR_HARDEST = 0.09

# 中文字体 (Windows 常见可用字体, 按顺序回退)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# 读一个 run 目录里最新的 event 文件
def load_scalars(run_dir: str) -> EventAccumulator:
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not files:
        raise SystemExit(f"没找到 event 文件: {run_dir}")
    acc = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    acc.Reload()
    return acc


# 取一条标量曲线为 (轮次, 值) 数组
def series(acc: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    if tag not in acc.Tags()["scalars"]:
        raise SystemExit(f"该 run 没有记录 {tag}; 无法画图")
    ev = acc.Scalars(tag)
    return (np.array([e.step for e in ev], dtype=float),
            np.array([e.value for e in ev], dtype=float))


# 下界第一次到达最窄档的轮次; 没到过则返回 None。
# 容差取 1e-6: tensorboard 存的是 float32, 0.09 会存成 0.090000003, 用 1e-9 判不出来。
def first_reach(step: np.ndarray, clear: np.ndarray, target: float) -> float | None:
    hit = np.nonzero(clear <= target + 1e-6)[0]
    return float(step[hit[0]]) if len(hit) else None


def main() -> None:
    ap = argparse.ArgumentParser(description="绘制 epoch-墙宽(净宽) 课程曲线")
    ap.add_argument("--run", default=DEFAULT_RUN, help="训练 run 目录 (含 events.out.tfevents.*)")
    ap.add_argument("--out", default=FILENAME_BASE, help="输出文件名前缀")
    ap.add_argument("--width", type=float, default=6.2, help="图宽 (英寸)")
    ap.add_argument("--height", type=float, default=3.4, help="图高 (英寸)")
    args = ap.parse_args()

    acc = load_scalars(args.run)
    step, clear = series(acc, "Curriculum/clear_width")
    _, d_min = series(acc, "Curriculum/wall_d_min")
    # 采样上沿 = 左墙取到最宽位置 (d=0.20) 时的净宽。用已记录的量反推, 不再另设口径:
    #   clear(d) = d + x_pos - 2t  =>  clear(0.20) - clear(d_min) = 0.20 - d_min
    clear_hi = clear + (0.20 - d_min)

    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=160)

    # 采样带: 每个回合左墙在 [d_min, 0.20] 内随机, 于是净宽落在 [下沿, 上沿]
    ax.fill_between(step, clear, clear_hi, color="#4c78a8", alpha=0.20,
                    label="每回合随机采样的净宽范围")
    ax.plot(step, clear_hi, color="#4c78a8", lw=1.1, ls=":", label=f"最宽档 {CLEAR_EASIEST:.2f} m")
    # 课程量本体: 下沿 (越窄越难)
    ax.plot(step, clear, color="#c0392b", lw=2.4, label="课程下界 (d_min)")
    ax.axhline(CLEAR_HARDEST, color="#7f8c8d", lw=1.0, ls="--",
               label=f"下界下限 {CLEAR_HARDEST:.2f} m")

    reached = first_reach(step, clear, CLEAR_HARDEST)
    if reached is not None:
        ax.axvline(reached, color="#7f8c8d", lw=0.9, ls="-.")
        ax.annotate(f"第 {int(reached)} 轮到达下界下限",
                    xy=(reached, CLEAR_EASIEST), xytext=(reached + 0.04 * step[-1], CLEAR_EASIEST - 0.028),
                    fontsize=9, color="#2c3e50",
                    arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=0.9))

    ax.set_xlabel("训练轮次 (iteration)")
    ax.set_ylabel("走廊净宽 (m)")
    ax.set_title("翻正任务: 逐环境随机墙位的课程收窄过程", fontsize=11)
    ax.set_xlim(step[0], step[-1])
    ax.set_ylim(min(clear.min(), CLEAR_HARDEST) - 0.02, CLEAR_EASIEST + 0.02)
    ax.grid(alpha=0.25, ls=":", lw=0.6)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = os.path.join(SAVE_DIR, f"{args.out}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"[INFO] 已保存 {path}")
    plt.close(fig)

    print(f"[INFO] 轮次范围 {int(step[0])}..{int(step[-1])}, "
          f"净宽 {clear[0]:.4f} -> {clear[-1]:.4f} m, "
          f"到达 {CLEAR_HARDEST:.2f} m 的轮次: {None if reached is None else int(reached)}")


if __name__ == "__main__":
    main()
