import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from mjlab.scripts._diagnostics import HOLE_DIAGNOSTICS_ROOT


# 用检查点完整回合与批次历史计算课程通过率。
def checkpoint(path):
    saved = torch.load(path, weights_only=False, map_location="cpu")
    info = saved.get("infos") or {}
    hole = info.get("hole_state") or {}
    course = hole.get("height_course") or {}
    result = {key: value for key, value in course.items() if key != "history"}
    result.update(iteration=saved.get("iter"), env_step=(info.get("env_state") or {}).get("common_step_counter"),
                  stage=hole.get("stage"), collision=hole.get("collision"))
    result["levels"] = []
    history = course.get("history", [])
    for height in dict.fromkeys(row["z_min"] for row in history):
        rows = [row for row in history if row["z_min"] == height]
        count = sum(row["count"] for row in rows)
        successes = sum(row["successes"] for row in rows)
        result["levels"].append({"height": height, "batches": len(rows), "count": count,
            "successes": successes, "success_rate": successes / count,
            "first_iteration": rows[0]["iteration"], "last_iteration": rows[-1]["iteration"],
            "last10_rate": sum(row["successes"] for row in rows[-10:]) / sum(row["count"] for row in rows[-10:]),
            "last10_batches": rows[-10:]})
    return result, history


# 汇总全量标量中的有限性、分阶段均值和终段趋势。
def scalars(run):
    events = EventAccumulator(str(run), size_guidance={"scalars": 0}).Reload()
    summary = {}
    selected = ("Train/mean_reward", "Train/mean_episode_length", "Data/act_vel_x", "Data/cmd_vel_x",
        "Data/error_height_F", "Data/error_height_H", "Metrics/Zmin_m", "Metrics/success_rate",
        "Loss/value_function", "Loss/surrogate", "Policy/mean_noise_std", "Episode_Termination/fallen")
    all_names = events.Tags()["scalars"]
    raw = {}
    for name in all_names:
        values = events.Scalars(name)
        steps = np.array([value.step for value in values])
        numbers = np.array([value.value for value in values])
        raw[name] = pd.DataFrame({"step": steps, "value": numbers})
        row = {"count": len(numbers), "finite": bool(np.isfinite(numbers).all()), "last_step": int(steps[-1]), "last": float(numbers[-1])}
        if name in selected or name.startswith("Episode_Reward/"):
            row["windows"] = {}
            for left, right in ((800, 999), (1800, 1999), (2800, 2999), (3000, 3199), (4000, 4499), (max(0, int(steps[-1]) - 499), int(steps[-1])), (max(0, int(steps[-1]) - 99), int(steps[-1]))):
                active = numbers[(steps >= left) & (steps <= right)]
                if len(active):
                    row["windows"][f"{left}-{right}"] = {"mean": float(active.mean()), "min": float(active.min()), "max": float(active.max())}
        summary[name] = row
    return summary, raw


# 汇总任意训练目录及可选对照目录，保留最低档完整批次口径。
def main():
    parser = argparse.ArgumentParser(description="汇总Hole检查点课程统计与TensorBoard标量")
    parser.add_argument("run_dir", type=Path, help="待分析的训练目录")
    parser.add_argument("--compare-run", type=Path, action="append", default=[], help="可重复指定对照训练目录")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir or HOLE_DIAGNOSTICS_ROOT / "training_review" / args.run_dir.name
    runs = []
    for run in (args.run_dir, *args.compare_run):
        paths = [path for path in run.glob("model_*.pt") if path.stem.removeprefix("model_").isdigit()]
        if not paths:
            parser.error(f"没有找到检查点：{run}")
        runs.append((run, max(paths, key=lambda path: int(path.stem.removeprefix("model_")))))
    output.mkdir(parents=True, exist_ok=True)
    result = []
    for index, (run, path) in enumerate(runs):
        course, history = checkpoint(path)
        values, raw = scalars(run)
        label = f"{index}_{run.name}"
        result.append({"run": str(run.resolve()), "checkpoint": str(path.resolve()),
                       "final": course, "history": history, "scalars": values})
        pd.DataFrame(history).to_csv(output / f"{label}_课程完整批次.csv", index=False, encoding="utf-8-sig")
        for name in ("Metrics/Zmin_m", "Metrics/success_rate", "Train/mean_reward", "Data/error_height_F", "Data/error_height_H"):
            if name in raw:
                raw[name].to_csv(output / f"{label}_{name.replace('/', '_')}.csv", index=False)
        print(f"训练{run.name}：检查点{path.name}，最低板高{course.get('z_min')}，所有标量有限={all(row['finite'] for row in values.values())}")
        for level in course["levels"]:
            print(f"  板高{level['height'] * 1000:g}毫米：完整批次{level['batches']}，成功{level['successes']}/{level['count']}，末十批成功率{level['last10_rate']:.2%}")
    (output / "training_summary.json").write_text(json.dumps({"runs": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存至：{output.resolve()}")


if __name__ == "__main__":
    main()
