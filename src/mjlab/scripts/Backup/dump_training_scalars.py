from __future__ import annotations
import glob
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# 训练日志标量导出 — 只读 tensorboard event 文件, 不写磁盘, 不启动训练
# 用法:
#   uv run python -m mjlab.scripts.Backup.dump_training_scalars <run_dir> [标签子串 ...]
#   默认按固定轮次采样打印; 传入 --list 只列标签名。

_DEFAULT_POINTS = 14


def load_scalars(run_dir: str) -> dict[str, list[tuple[int, float]]]:
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not files:
        raise SystemExit(f"未找到 event 文件: {run_dir}")
    acc = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    acc.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in acc.Tags().get("scalars", []):
        out[tag] = [(e.step, e.value) for e in acc.Scalars(tag)]
    return out


def sample(series: list[tuple[int, float]], n: int) -> list[tuple[int, float]]:
    if len(series) <= n:
        return series
    idx = [round(i * (len(series) - 1) / (n - 1)) for i in range(n)]
    return [series[i] for i in idx]


def main() -> None:
    args = [a for a in sys.argv[1:]]
    if not args:
        raise SystemExit("用法: dump_training_scalars.py <run_dir> [标签子串 ...] [--list]")
    run_dir = args[0]
    rest = args[1:]
    list_only = "--list" in rest
    filters = [a for a in rest if not a.startswith("--")]

    scalars = load_scalars(run_dir)
    print(f"=== {run_dir} ===")
    print(f"标签数: {len(scalars)}")
    if list_only:
        for tag in sorted(scalars):
            print(f"  {tag}  (n={len(scalars[tag])})")
        return

    tags = sorted(scalars) if not filters else [t for t in sorted(scalars) if any(f in t for f in filters)]
    for tag in tags:
        series = scalars[tag]
        if not series:
            continue
        vals = [v for _, v in series]
        pts = sample(series, _DEFAULT_POINTS)
        body = "  ".join(f"{s}:{v:+.4g}" for s, v in pts)
        print(f"\n{tag}   n={len(series)}  min={min(vals):+.4g} max={max(vals):+.4g}")
        print(f"  {body}")


if __name__ == "__main__":
    main()
