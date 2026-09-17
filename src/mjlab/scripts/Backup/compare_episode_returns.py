from __future__ import annotations
import sys
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

# 逐回合回报对照: 回放 CSV 的 reward 列是单步总奖励(已含 dt 与权重), done 标记回合结束

BASE = "logs/rsl_rl/SQuRo_Backup/2026-09-17_15-00-33/videos/SQuRo_Backup_"


def main() -> None:
    for tag in ("600-ts3", "900-ts3", "600-ts1", "900-ts1"):
        d = pd.read_csv(BASE + tag + ".00.csv")
        r = d["reward"].to_numpy(float)
        done = d["done"].to_numpy(int)
        ends = np.where(done > 0)[0]
        print(f"=== {tag}   步数 {len(d)}   完成回合 {len(ends)}")
        print(f"  全段累计 {r.sum():.2f}   平均每步 {r.mean():+.4f}")
        prev = -1
        for k, e in enumerate(ends):
            seg = r[prev + 1:e + 1]
            prev = e
            print(f"    第{k+1}回合: {len(seg)} 步 ({len(seg)*0.01:.2f} s)  累计 {seg.sum():.2f}  末步(成功脉冲) {seg[-1]:+.2f}")
        if prev < len(d) - 1:
            seg = r[prev + 1:]
            print(f"    末尾未完成: {len(seg)} 步  累计 {seg.sum():.2f}")
        print()


if __name__ == "__main__":
    main()
