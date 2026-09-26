# 多 run 标量对照 — 把两次(或多次)训练的同一标量并排比, 用于单变量 A/B。只读, 不落盘。
# 数据源优先 tensorboard; 没有事件文件时退回解析 output.log 的逐轮指标。
# 用法:
#   uv run python -B -m mjlab.scripts.Backup.diag_run_compare --list-tags \
#       --runs "A=logs/rsl_rl/SQuRo_Backup/2026-09-26_14-59-27"
#   uv run python -B -m mjlab.scripts.Backup.diag_run_compare \
#       --runs "旧=logs/rsl_rl/SQuRo_Backup/<run1>,新=logs/rsl_rl/SQuRo_Backup/<run2>" \
#       --tags Progress/stand_mean_vel,Data/leg_pose_rmse_hold --bins 6
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

# 默认关注的量: 课程 / 站姿 / 成功率 / 探索
DEFAULT_TAGS = ("Curriculum/level,Curriculum/clear_width,Curriculum/p_stood,"
                "Curriculum/p_done,Progress/stand_mean_vel,Progress/standing,"
                "Data/leg_pose_rmse_hold,Data/leg_pose_hold_frac,Policy/mean_std,"
                "Cycle/per_episode,Train/mean_reward")


@dataclass
class Cfg:
    runs: str = ""                 # "名字=路径[,名字=路径...]"
    tags: str = ""                 # 逗号分隔; 留空用 DEFAULT_TAGS
    bins: int = 6                  # 按轮次等分箱数
    list_tags: bool = False


# 从 tensorboard 读全量标量 (取最新的事件文件)
def read_tb(run_dir: Path) -> dict[str, dict[int, float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not files:
        return {}
    acc = EventAccumulator(str(files[-1]), size_guidance={"scalars": 0})
    acc.Reload()
    return {t: {int(e.step): float(e.value) for e in acc.Scalars(t)}
            for t in acc.Tags()["scalars"]}


# 从 output.log 解析逐轮指标: 见到 "Learning iteration N/" 就切当前轮次
def read_log(run_dir: Path) -> dict[str, dict[int, float]]:
    path = run_dir / "output.log"
    if not path.exists():
        return {}
    out: dict[str, dict[int, float]] = {}
    cur = None
    for line in path.open(encoding="utf-8", errors="ignore"):
        if "Learning iteration" in line:
            head = line.split("Learning iteration")[1].strip().split("/")[0].strip()
            cur = int(head) if head.isdigit() else None
            continue
        if cur is None or ":" not in line:
            continue
        name, _, val = line.strip().partition(":")
        try:
            out.setdefault(name.strip(), {})[cur] = float(val.strip())
        except ValueError:
            continue
    return out


# 读取一个 run: 先 tensorboard, 空则退回 output.log; 返回 (数据, 来源)
def read_run(run_dir: Path) -> tuple[dict[str, dict[int, float]], str]:
    tb = read_tb(run_dir)
    if tb:
        return tb, "tensorboard"
    log = read_log(run_dir)
    return log, ("output.log" if log else "无数据")


# 按轮次等分箱取均值; 返回 [(起, 止, 均值)]
def binned(series: dict[int, float], bins: int) -> list[tuple[int, int, float]]:
    if not series:
        return []
    lo, hi = min(series), max(series)
    if hi <= lo:
        return [(lo, hi, series[lo])]
    width = max(1, -(-(hi - lo + 1) // bins))          # 向上取整
    out = []
    for b in range(bins):
        a = lo + b * width
        z = a + width
        vals = [v for k, v in series.items() if a <= k < z]
        if vals:
            out.append((a, min(z - 1, hi), sum(vals) / len(vals)))
    return out


def main() -> None:
    cfg = tyro.cli(Cfg)
    runs: dict[str, Path] = {}
    for item in filter(None, (s.strip() for s in cfg.runs.split(","))):
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--runs 需要 名字=路径 的形式, 收到 {item!r}")
        runs[name.strip()] = Path(path.strip())
    if not runs:
        raise SystemExit("必须给 --runs, 例如 --runs \"A=logs/rsl_rl/SQuRo_Backup/<run>\"")

    data: dict[str, dict[str, dict[int, float]]] = {}
    for name, d in runs.items():
        if not d.exists():
            raise SystemExit(f"路径不存在: {d}")
        parsed, src = read_run(d)
        data[name] = parsed
        print(f"[{name}] {d}  标量 {len(parsed)} 个 (来源 {src})")

    if cfg.list_tags:
        alltags = sorted({t for d in data.values() for t in d})
        print(f"\n可用标签 {len(alltags)} 个:")
        for t in alltags:
            print("  " + t)
        return

    tags = [t.strip() for t in (cfg.tags or DEFAULT_TAGS).split(",") if t.strip()]
    names = list(data)
    for tag in tags:
        series = {n: data[n].get(tag, {}) for n in names}
        if not any(series.values()):
            print(f"\n=== {tag} ===  (各 run 都没有这个标签)")
            continue
        print(f"\n=== {tag} ===")
        blocks = {n: binned(s, cfg.bins) for n, s in series.items()}
        ref = next((b for b in blocks.values() if b), [])
        print(f"  {'':14s}" + "".join(f"{f'{a}-{z}':>11s}" for a, z, _ in ref))
        for n in names:
            b = blocks[n]
            if not b:
                print(f"  {n:14s}无数据")
                continue
            print(f"  {n:14s}" + "".join(f"{v:>11.4f}" for _, _, v in b))
        # 两个 run 时给出逐箱差 (后一个减前一个), 便于读单变量效果
        if len(names) == 2:
            b0, b1 = blocks[names[0]], blocks[names[1]]
            if len(b0) == len(b1):
                d = "".join(f"{y[2] - x[2]:>+11.4f}" for x, y in zip(b0, b1))
                print(f"  {'差(' + names[1] + '-' + names[0] + ')':14s}{d}")


if __name__ == "__main__":
    main()
