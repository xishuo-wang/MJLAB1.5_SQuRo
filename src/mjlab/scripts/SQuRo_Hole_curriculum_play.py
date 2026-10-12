from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.SQuRo_Hole.mdp.config import COURSE_START
from mjlab.tasks.SQuRo_Hole.mdp.entity import (
    apply_saved_layout,
    configure_single_hole_entities,
)
from mjlab.tasks.SQuRo_Hole.mdp.height_curriculum import HeightCurriculumState
from mjlab.tasks.SQuRo_Hole.mdp.height_schedule import HEIGHT_MAX, HEIGHT_MIN
from mjlab.tasks.SQuRo_Hole.rl.runner import read_env_step, read_hole_state
from mjlab.viewer import NativeMujocoViewer

TASK = "Mjlab-SQuRo-Hole"


# 解析课程回放参数。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单板真实碰撞高度课程回放")
    parser.add_argument("--checkpoint", required=True, help="策略 checkpoint 路径")
    parser.add_argument("--obstacle-height", type=float, help="固定板底高度，默认 checkpoint 当前 Zmin")
    parser.add_argument("--random-height", action="store_true", help="按 checkpoint 当前范围随机采样")
    parser.add_argument("--steps", type=int, default=2000, help="无头运行步数")
    parser.add_argument("--num-envs", type=int, default=1, help="并行环境数")
    parser.add_argument("--device", default=None, help="运行设备，默认自动选择")
    parser.add_argument("--no-show", action="store_true", help="无窗口运行并导出回合 CSV")
    parser.add_argument("--output", type=Path, default=Path("logs/SQuRo_Hole_curriculum_play.csv"))
    return parser.parse_args()


# 按 checkpoint 几何、板高和碰撞配置构造课程回放环境。
def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.steps < 1 or args.num_envs < 1:
        raise ValueError("steps 和 num-envs 必须为正数")
    if args.random_height and args.obstacle_height is not None:
        raise ValueError("--random-height 与 --obstacle-height 不能同时使用")

    saved = read_hole_state(checkpoint)
    course_state = saved.get("height_course") or HeightCurriculumState().state_dict()
    z_min = float(course_state["z_min"])
    z_max = float(course_state["z_max"])
    if not HEIGHT_MIN <= z_min <= z_max <= HEIGHT_MAX:
        raise ValueError("checkpoint 高度范围必须位于课程上下界内")
    if args.obstacle_height is not None:
        if not HEIGHT_MIN <= args.obstacle_height <= HEIGHT_MAX:
            raise ValueError("板底高度必须在 0.048～0.090 m 范围内")
        course_state["z_min"] = args.obstacle_height
        course_state["z_max"] = args.obstacle_height
    elif not args.random_height:
        course_state["z_max"] = course_state["z_min"]

    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = args.num_envs
    configure_single_hole_entities(env_cfg, enable_collision=True)
    saved_layout = [
        item for item in saved.get("layout", [])
        if str(item.get("name", "")).lower() == "hole1"
    ]
    apply_saved_layout(env_cfg, saved_layout)
    command_cfg = env_cfg.commands["hole_cmd"]
    command_cfg.enable_height_course = True
    command_cfg.enable_course_progress = False
    command_cfg.height_curriculum_state = course_state

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    raw_env.common_step_counter = max(read_env_step(checkpoint) or 0, COURSE_START)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=load_rl_cfg(TASK).clip_actions)
    try:
        agent_cfg = load_rl_cfg(TASK)
        runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
        raw_env.common_step_counter = max(raw_env.common_step_counter, COURSE_START)
        obs, _ = env.reset()
        policy = runner.get_inference_policy(device=device)

        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        if args.no_show:
            obs = obs.to(device)
            with torch.inference_mode():
                for _ in range(args.steps):
                    actions = policy(obs)
                    obs, _, _, _ = env.step(actions)
                    obs = obs.to(device)
                    monitor = getattr(raw_env, "_hole_passage_monitor", None)
                    if monitor is not None:
                        rows.extend(monitor.consume())
        else:
            NativeMujocoViewer(env, policy).run()
            monitor = getattr(raw_env, "_hole_passage_monitor", None)
            if monitor is not None:
                rows.extend(monitor.consume())

        with output.open("w", newline="", encoding="utf-8-sig") as file:
            columns = ("seq", "env_id", "height", "valid", "success", "failure_code", "advanced")
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows({key: row.get(key) for key in columns} for row in rows)
        successes = sum(bool(row["success"]) for row in rows)
        valid = sum(bool(row["valid"]) for row in rows)
        print(f"回放完成：回合 {len(rows)}，有效 {valid}，成功 {successes}，CSV={output}")
        print(f"课程进度冻结；板底采样范围 [{course_state['z_min']:.3f}, {course_state['z_max']:.3f}] m")
    finally:
        env.close()


if __name__ == "__main__":
    main()
