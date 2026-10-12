# 数值签名比对: 逐项对比两份签名, 报告最大绝对/相对偏差
# 用法: uv run python -B -m mjlab.scripts.Hole.compare_numeric_signature <基准.json> <候选.json>

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# 允许的最大偏差 (float32 往返噪声量级)
TOL_ABS = 1e-6
TOL_REL = 1e-6


# 递归比较签名，并拒绝非有限数。
def walk(a, b, path: str, diffs: list, stats: dict) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a) | set(b))
        if set(a) != set(b):
            only_a = sorted(set(a) - set(b))
            only_b = sorted(set(b) - set(a))
            diffs.append(f"{path}: 键集合不同 仅在基准={only_a[:6]} 仅在候选={only_b[:6]}")
        for k in keys:
            if k in a and k in b:
                walk(a[k], b[k], f"{path}.{k}", diffs, stats)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: 长度不同 {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", diffs, stats)
        return
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        if a != b:
            diffs.append(f"{path}: {a!r} vs {b!r}")
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isfinite(a) or not math.isfinite(b):
            diffs.append(f"{path}: 存在非有限数 {a!r} vs {b!r}")
            return
        d = abs(float(a) - float(b))
        scale = max(abs(float(a)), abs(float(b)), 1e-12)
        rel = d / scale
        stats["n"] = stats.get("n", 0) + 1
        stats["max_abs"] = max(stats.get("max_abs", 0.0), d)
        stats["max_rel"] = max(stats.get("max_rel", 0.0), rel)
        stats["sum_abs"] = stats.get("sum_abs", 0.0) + d
        if d > TOL_ABS and rel > TOL_REL:
            diffs.append(f"{path}: {a!r} vs {b!r} (Δ={d:.3e}, 相对={rel:.3e})")
        return
    if a != b:
        diffs.append(f"{path}: {a!r} vs {b!r}")


# 读取两份签名并报告比较结果。
def main() -> None:
    base = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    cand = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    diffs: list[str] = []
    stats: dict = {}
    walk(base, cand, "$", diffs, stats)

    print(f"比对完成: 标量 {stats.get('n', 0)} 个, "
          f"最大绝对偏差 {stats.get('max_abs', 0.0):.3e}, "
          f"最大相对偏差 {stats.get('max_rel', 0.0):.3e}, "
          f"累计绝对偏差 {stats.get('sum_abs', 0.0):.3e}")
    if diffs:
        print(f"\n[不一致] 共 {len(diffs)} 处, 前 30 条:")
        for line in diffs[:30]:
            print("  " + line)
        sys.exit(1)
    print("\n[一致] 数值签名完全一致")


if __name__ == "__main__":
    main()
