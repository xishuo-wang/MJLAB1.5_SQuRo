import json
from dataclasses import asdict
from unittest.mock import patch

import torch
import warp as wp

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts._diagnostics import HOLE_DIAGNOSTICS_ROOT, WARP_CACHE_DIR
from mjlab.tasks.SQuRo_Hole.SQuRo_Hole_env_cfg import SQuRo_Hole_Env_Cfg
from mjlab.tasks.SQuRo_Hole.config.rl_cfg import SQuRo_Hole_PPO_Runner_Cfg
from mjlab.tasks.SQuRo_Hole.rl.runner import SQuRoHoleOnPolicyRunner
from mjlab.tasks.SQuRo_Hole.mdp.curriculums import STAGE3_ITER, STAGE3_STEP, STEPS_PER_ITER
from mjlab.tasks.SQuRo_Hole.mdp.entity import obstacle_collision_enabled


OUT = HOLE_DIAGNOSTICS_ROOT / "rebuild_timing"
wp.config.kernel_cache_dir = str(WARP_CACHE_DIR)


# 实际短训跨过3000轮，验证重建计时错开、首回合过滤以及重复learn不再次随机化。
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(197)
    cfg = SQuRo_Hole_Env_Cfg()
    cfg.scene.num_envs = 16
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    settings = asdict(SQuRo_Hole_PPO_Runner_Cfg())
    settings["logger"] = "tensorboard"
    settings["upload_model"] = False
    runner = SQuRoHoleOnPolicyRunner(RslRlVecEnvWrapper(env), settings, log_dir=None, device="cuda:0")
    runner.current_learning_iteration = STAGE3_ITER - 1
    env.common_step_counter = (STAGE3_ITER - 1) * STEPS_PER_ITER
    calls = []
    episodes = []
    original = runner._randomize_episode_lengths
    original_consume = runner._consume_passage_results

    # 记录真实方法调用后的局部计时和全局步数，不改变被测方法。
    def observe():
        original()
        clock = runner.env.episode_length_buf.clone()
        command = runner._obstacle_command
        assert (clock >= 0).all() and (clock < runner.env.max_episode_length).all()
        assert clock.unique().numel() > 1
        assert torch.equal(command.course_episode_eligible, clock == 0)
        calls.append({"step": runner.env.unwrapped.common_step_counter,
                      "episode_steps": clock.cpu().tolist(),
                      "has_obstacle": "obstacle" in runner.env.unwrapped.scene.entities,
                      "collision": obstacle_collision_enabled(runner.env.unwrapped)})

    # 保存课程回合快照后交还原消费函数，不改变训练统计。
    def observe_results(extras):
        current = runner.env.unwrapped
        monitor = getattr(current, "_pass_monitor", None)
        if monitor is not None and "obstacle" in current.scene.entities:
            episodes.extend(monitor.completed)
        original_consume(extras)

    try:
        with patch.object(runner, "_randomize_episode_lengths", side_effect=observe), \
             patch.object(runner, "_consume_passage_results", side_effect=observe_results):
            runner.learn(3, init_at_random_ep_len=True)
            assert len(calls) == 2 and calls[-1]["step"] == STAGE3_STEP
            assert calls[0]["has_obstacle"] is False and calls[1]["has_obstacle"] is True
            assert calls[1]["collision"] is False
            assert runner.env.unwrapped.common_step_counter == (STAGE3_ITER + 2) * STEPS_PER_ITER
            command = runner._obstacle_command
            assert len(episodes) > 0 and command.height_curriculum_state.all_height_episode_count == 0
            runner.learn(1, init_at_random_ep_len=True)
            assert len(calls) == 2
            assert command.height_curriculum_state.all_height_episode_count == 0
        first_episodes = {}
        for row in episodes:
            first_episodes.setdefault(row["env_id"], row)
        for env_id, row in first_episodes.items():
            assert row["valid"] == (calls[-1]["episode_steps"][env_id] == 0)
        result = {"randomization_calls": calls, "final_step": runner.env.unwrapped.common_step_counter,
                  "curriculum_count": command.height_curriculum_state.all_height_episode_count,
                  "training_collision": obstacle_collision_enabled(runner.env.unwrapped),
                  "completed_episodes": episodes,
                  "repeat_learn_randomized_again": False}
        (OUT / "实际重建随机计时.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        runner.env.close()


if __name__ == "__main__":
    main()
