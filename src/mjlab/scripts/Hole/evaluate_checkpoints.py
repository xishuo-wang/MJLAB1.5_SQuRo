import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import MethodType

import torch
import warp as wp

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts._diagnostics import HOLE_DIAGNOSTICS_ROOT, WARP_CACHE_DIR
from mjlab.tasks.SQuRo_Hole.SQuRo_Hole_env_cfg import SQuRo_Hole_Env_Cfg
from mjlab.tasks.SQuRo_Hole.config.rl_cfg import SQuRo_Hole_PPO_Runner_Cfg
from mjlab.tasks.SQuRo_Hole.mdp.curriculums import STAGE3_STEP, stage_uses_height_course
from mjlab.tasks.SQuRo_Hole.mdp.entity import configure_obstacle_entity, write_obstacle_mocap
from mjlab.tasks.SQuRo_Hole.mdp.terminations import monitor_restricted_passage
from mjlab.tasks.SQuRo_Hole.rl.runner import SQuRoHoleOnPolicyRunner, read_obstacle_state


# 按指定真实板高分别评估均值动作和独立策略噪声，冻结网络与观测归一化。
def main():
    parser = argparse.ArgumentParser(description="用当前任务代码评估课程检查点的完整穿板回合")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--heights", type=float, nargs="+", default=[0.047, 0.048, 0.049, 0.055])
    parser.add_argument("--samples", type=int, default=32, help="每档随机动作回合数，另加一个确定性回合")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--high-obstacle-mode", choices=("saved", "equal", "segmented", "mixed"), default="saved")
    parser.add_argument("--output-dir", type=Path, default=HOLE_DIAGNOSTICS_ROOT / "checkpoint_evaluation")
    args = parser.parse_args()
    if args.samples < 0 or any(not math.isfinite(height) or height <= 0 for height in args.heights):
        parser.error("随机回合数不能为负，板底高度必须为正有限数，单位为米")
    for path in args.checkpoint:
        if not path.is_file():
            parser.error(f"检查点不存在：{path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(WARP_CACHE_DIR)
    cfg = SQuRo_Hole_Env_Cfg()
    cfg.scene.num_envs = len(args.heights) * (args.samples + 1)
    cfg.commands["hole_cmd"].enable_height_course = True
    configure_obstacle_entity(cfg, enable_collision=True)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    env.common_step_counter = STAGE3_STEP
    settings = asdict(SQuRo_Hole_PPO_Runner_Cfg())
    settings["logger"] = "tensorboard"
    settings["upload_model"] = False
    runner = SQuRoHoleOnPolicyRunner(RslRlVecEnvWrapper(env), settings, log_dir=None, device=env.device)
    heights = torch.tensor(args.heights, device=env.device).repeat_interleave(args.samples + 1)
    deterministic = torch.arange(env.num_envs, device=env.device) % (args.samples + 1) == 0

    # 覆盖采样方法，将每个环境的板高固定到对应评估档。
    def sample_heights(command, env_ids):
        command._sample_high_obstacle_mode(env_ids)
        values = heights[env_ids]
        command.obstacle_sampled_bottom_z[env_ids] = values
        command.obstacle_height_sampled[env_ids] = True
        write_obstacle_mocap(command.obstacle, values, env_ids)
        command._update_obstacle_height_command()

    summaries = []
    try:
        for path in args.checkpoint:
            saved = read_obstacle_state(path)
            if not stage_uses_height_course(saved.get("stage", 0)):
                raise ValueError("仅支持已进入真实单板课程的检查点")
            runner.load(str(path), load_cfg={"actor": True}, strict=True, map_location=env.device)
            env = runner.env.unwrapped
            command = env.command_manager.get_term("hole_cmd")
            command.cfg.high_obstacle_equal_height = (
                saved.get("high_obstacle_equal_height") if args.high_obstacle_mode == "saved"
                else {"equal": True, "segmented": False, "mixed": None}[args.high_obstacle_mode]
            )
            command._sample_obstacle_height = MethodType(sample_heights, command)
            torch.manual_seed(args.seed)
            obs, _ = runner.env.reset()
            selected_modes = command.obstacle_equal_height.clone()
            monitor_restricted_passage(env)
            monitor = env._hole_passage_monitor
            monitor.consume()
            policy = runner.get_inference_policy(device=env.device)
            max_x = torch.full((env.num_envs,), -torch.inf, device=env.device)
            records = {}
            print(f"开始评估{path.name}：{len(args.heights)}档，每档1个确定性和{args.samples}个随机回合", flush=True)
            for _ in range(env.max_episode_length):
                with torch.inference_mode():
                    current_x = env.scene["robot"].data.root_link_pos_w[:, 0]
                    active = torch.tensor([index not in records for index in range(env.num_envs)], device=env.device)
                    max_x = torch.where(active, torch.maximum(max_x, current_x), max_x)
                    mean = policy(obs)
                    noisy = policy(obs, stochastic_output=True) if args.samples else mean
                    obs, _, _, _ = runner.env.step(torch.where(deterministic[:, None], mean, noisy))
                for row in monitor.consume():
                    records.setdefault(row["env_id"], row)
                if len(records) == env.num_envs:
                    break
            if len(records) != env.num_envs:
                raise RuntimeError(f"回合记录不完整：{len(records)}/{env.num_envs}")
            for index, height in enumerate(args.heights):
                offset = index * (args.samples + 1)
                episodes = [records[env_id] for env_id in range(offset + 1, offset + args.samples + 1)]
                row = {"checkpoint": str(path.resolve()), "height": height,
                       "deterministic": records[offset], "deterministic_max_x": float(max_x[offset]),
                       "deterministic_equal_height": bool(selected_modes[offset]),
                       "stochastic_samples": args.samples, "stochastic_successes": sum(item["success"] for item in episodes),
                       "outcome_counts": {code: sum(item["failure_code"] == code for item in episodes) for code in range(6)}}
                summaries.append(row)
                print(f"板底{height * 1000:g}毫米：确定性成功={records[offset]['success']}，随机成功{row['stochastic_successes']}/{args.samples}", flush=True)
            result = {"seed": args.seed, "task_code": "当前工作区代码", "results": summaries}
            (args.output_dir / "checkpoint_evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        runner.env.close()


if __name__ == "__main__":
    main()
