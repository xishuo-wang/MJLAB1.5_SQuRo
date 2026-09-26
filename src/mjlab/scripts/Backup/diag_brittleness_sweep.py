# 诊断: 多种子扫描, 区分"初态运气"与"真实脆弱性"。只读。
# 对每个 (检查点, seed) 跑一次 diag_s1_reach, 汇总某个指标列。
# 列按**表头名字**定位, 不按位置 —— diag_s1_reach 的 keys 顺序改动不会让本脚本静默读错列。
# 用法:
#   uv run python -B -m mjlab.scripts.Backup.diag_brittleness_sweep \
#       --run logs/rsl_rl/SQuRo_Backup/<run> --ckpts 1000,1500,2900 --seeds 1,3,7,42 \
#       --key strict_rate
import argparse
import subprocess
import sys
from pathlib import Path


# 跑一个 (检查点, seed) 组合, 返回 {列名: 字符串} 与 ckpt
def run_one(py: str, run: Path, it: int, seed: int, num_envs: int, steps: int,
            align_iter: int) -> dict[str, str] | None:
    args = [py, "-B", "-m", "mjlab.scripts.Backup.diag_s1_reach",
            "--ckpt", str(run / f"model_{it}.pt"),
            "--num-envs", str(num_envs), "--steps", str(steps),
            "--align-iter", str(it if align_iter < 0 else align_iter),
            "--seed", str(seed)]
    p = subprocess.run(args, capture_output=True, text=True, errors="ignore")
    header: list[str] | None = None
    for line in p.stdout.splitlines():
        if line.startswith("checkpoint "):
            header = line.split()[1:]
            continue
        if header and line.startswith("model_"):
            cells = line.split()
            # 第一列是检查点名, 其余按 header 对齐
            return dict(zip(header, cells[1:])) if len(cells) - 1 == len(header) else None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="训练目录 (含 model_<iter>.pt)")
    ap.add_argument("--ckpts", default="1000,1500,2900", help="逗号分隔的轮次")
    ap.add_argument("--seeds", default="1,3,7,11,42,101", help="逗号分隔的 seed")
    ap.add_argument("--key", default="strict_rate", help="要汇总的指标列名")
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--align-iter", type=int, default=-1,
                    help="λ 课程对齐到的轮次; -1 = 用各检查点自身轮次")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    run = Path(args.run)
    ckpts = [int(s) for s in args.ckpts.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not run.exists():
        raise SystemExit(f"目录不存在: {run}")

    print(f"run = {run}")
    print(f"列 = {args.key}   {len(ckpts)} 检查点 x {len(seeds)} seed")
    print("%-10s %s" % ("检查点", "  ".join("%-7s" % f"s={s}" for s in seeds)))
    summary: dict[int, list[float]] = {}
    for it in ckpts:
        vals, cells = [], []
        for seed in seeds:
            r = run_one(args.python, run, it, seed, args.num_envs, args.steps, args.align_iter)
            if r is None or args.key not in r:
                cells.append("%-7s" % "err")
                continue
            try:
                v = float(r[args.key])
            except ValueError:
                cells.append("%-7s" % r[args.key][:7])
                continue
            vals.append(v)
            cells.append("%-7.3f" % v)
        summary[it] = vals
        print("%-10d %s" % (it, "  ".join(cells)))

    print("\n=== 汇总 ===")
    print("%-10s %-10s %-10s %-12s" % ("检查点", "均值", "最小值", "满值种子数"))
    for it in ckpts:
        v = summary[it]
        if not v:
            print("%-10d 无有效样本" % it)
            continue
        print("%-10d %-10.3f %-10.3f %d/%d"
              % (it, sum(v) / len(v), min(v), sum(1 for x in v if x >= 0.999), len(v)))
    print("\n判读: 若某检查点多数 seed 都达不到 1.0, 说明该阶段策略真的脆弱;")
    print("      若各 seed 都满值而单条回放失败, 则是单次初态的运气问题。")


if __name__ == "__main__":
    main()
