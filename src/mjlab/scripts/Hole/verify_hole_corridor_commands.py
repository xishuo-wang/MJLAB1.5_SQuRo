from __future__ import annotations

import math

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.Hole.verify_hole_obstacle_curriculum import make_env, read_geom_bottom_z
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Hole.mdp.config import (
    BASE_FREQ,
    CMD_GAIT_FREQ_IDS,
    CMD_HEIGHT_F_IDS,
    CMD_HEIGHT_H_IDS,
    CMD_OBSTACLE_HEIGHT_IDS,
    COMMAND_FIELDS,
    COMMAND_SCHEMA_VERSION,
    COURSE_START,
    STAGE2_END,
    STEPS_PER_ITER,
    get_current_stage,
    heights_for_stage,
    stage_requires_collision,
)
from mjlab.tasks.SQuRo_Hole.mdp.curriculums import reward_weight_curriculum
from mjlab.tasks.SQuRo_Hole.mdp.entity import write_hole_bottom_z
from mjlab.tasks.SQuRo_Hole.mdp.indices import _MODEL_INDICES
from mjlab.tasks.SQuRo_Hole.mdp.reference import get_reference_joint_state
from mjlab.tasks.SQuRo_Hole.mdp.rewards import compute_corridor_reward
from mjlab.tasks.SQuRo_Hole.rl.runner import validate_hole_command_schema


# 验证三个边界两侧的阶段、奖励权重和课程门控。
def verify_stages() -> None:
    probes = ((999, 1), (1000, 2), (1999, 2), (2000, 3), (2999, 3), (3000, 4))
    for iteration, expected in probes:
        stage = get_current_stage(iteration * STEPS_PER_ITER)
        assert stage == expected
        assert stage_requires_collision(stage) == (expected == 4)
    previous = reward_weight_curriculum.get_reward_weights(1000 * STEPS_PER_ITER)
    warmup = reward_weight_curriculum.get_reward_weights(STAGE2_END)
    assert all(warmup[key] == value for key, value in previous.items() if key != "corridor")
    assert previous["corridor"] == 0.0 and warmup["corridor"] == 5.0
    assert heights_for_stage(2) == heights_for_stage(3)
    physical = reward_weight_curriculum.get_reward_weights(COURSE_START)
    assert physical["corridor"] == physical["body_contact"] == 5.0
    print("[通过] 前两段设置保留，第三段仅增加走廊，3000轮开始真实碰撞")


# 使用真实场景与独立最高点状态验证单侧走廊约束。
def verify_corridor() -> None:
    cfg = load_env_cfg("Mjlab-SQuRo-Hole")
    cfg.scene.num_envs = 4
    cfg.commands["hole_cmd"].enable_height_course = False
    cfg.commands["hole_cmd"].stage_schedule_fallback = False
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        env.common_step_counter = STAGE2_END
        env.reset(seed=47)
        command = env.command_manager.get_term("hole_cmd")
        obs = env.observation_manager.compute()["actor"]
        assert obs.shape == (4, 113) and env.action_space.shape[-1] == 14
        torch.testing.assert_close(command.command[:, CMD_GAIT_FREQ_IDS], torch.full((4,), BASE_FREQ))
        torch.testing.assert_close(command.command[:, CMD_OBSTACLE_HEIGHT_IDS], torch.full((4,), 0.050))
        torch.testing.assert_close(obs[:, -6:], command.command)

        command.command[:, CMD_HEIGHT_F_IDS] = torch.tensor([0.020, 0.020, 0.055, 0.055])
        command.command[:, CMD_HEIGHT_H_IDS] = torch.tensor([0.020, 0.055, 0.020, 0.055])
        robot = env.scene["robot"]
        front = robot.indexing.site_ids[torch.tensor(_MODEL_INDICES.f_body_site_ids)]
        hind = robot.indexing.site_ids[torch.tensor(_MODEL_INDICES.h_body_site_ids)]
        data = robot.data.data.site_xpos
        with torch.inference_mode():
            data[:, torch.cat((front, hind)), 2] = 0.010
            data[:, front[-1], 2] = torch.tensor([0.045, 0.051, 0.095, 0.085])
            data[:, hind[-1], 2] = torch.tensor([0.045, 0.075, 0.040, 0.087])
            data[:, torch.cat((front, hind)), 0] = torch.tensor([-0.5, 0.5, 0.2, 1.0])[:, None]
        expected = torch.tensor([
            0.0,
            -2.5 * (1.0 - math.exp(-2000.0 * 0.001 ** 2)),
            -2.5 * (1.0 - math.exp(-2000.0 * 0.010 ** 2)),
            -2.5 * (1.0 - math.exp(-2000.0 * 0.002 ** 2)),
        ])
        torch.testing.assert_close(compute_corridor_reward(env), expected, atol=1e-6, rtol=1e-5)
        env.common_step_counter = STAGE2_END - 1
        assert not bool(compute_corridor_reward(env).any())
        env.common_step_counter = COURSE_START
        torch.testing.assert_close(compute_corridor_reward(env), expected, atol=1e-6, rtol=1e-5)
        print("[通过] 两段分别读取最高site，无位置门控，低于上限不罚，2k启用且3k后保留")

        env.common_step_counter = STAGE2_END
        env.reset()
        env.episode_length_buf[:] = 17
        reference_pos, reference_vel = get_reference_joint_state(env)
        command.command[:, CMD_GAIT_FREQ_IDS] = BASE_FREQ / 2
        env.episode_length_buf[:] = 34
        slower_pos, slower_vel = get_reference_joint_state(env)
        torch.testing.assert_close(slower_pos, reference_pos)
        torch.testing.assert_close(slower_vel, reference_vel / 2)
        command.command[:, CMD_GAIT_FREQ_IDS] = BASE_FREQ
        env.reset()
        for _ in range(5):
            obs, rewards, _, _, _ = env.step(torch.zeros(4, 14))
            assert torch.isfinite(obs["actor"]).all() and torch.isfinite(rewards).all()
        print("[通过] 六维命令进入观测，步频与参考相位/速度一致，环境步进无非有限值")
    finally:
        env.close()


# 从碰撞几何独立校验板高先验、逐环境写入与部分复位。
def verify_obstacle_prior() -> None:
    env = make_env()
    try:
        command = env.command_manager.get_term("hole_cmd")
        board = env.scene.entities["hole1"]
        target = torch.tensor([0.048, 0.052, 0.070, 0.090])
        write_hole_bottom_z(board, target)
        actual = read_geom_bottom_z(env)
        command._update_command()
        torch.testing.assert_close(actual, target, atol=1e-6, rtol=0)
        torch.testing.assert_close(command.command[:, CMD_OBSTACLE_HEIGHT_IDS], actual)
        previous = actual[[1, 3]].clone()
        env.reset(env_ids=torch.tensor([0, 2]))
        actual = read_geom_bottom_z(env)
        torch.testing.assert_close(actual[[1, 3]], previous)
        torch.testing.assert_close(command.command[:, CMD_OBSTACLE_HEIGHT_IDS], actual)
        obs = env.observation_manager.compute()["actor"]
        torch.testing.assert_close(obs[:, -6:], command.command)
        print("[通过] 板高先验等于真实碰撞几何，部分复位不污染其他环境")
    finally:
        env.close()


# 同维度旧检查点必须拒绝，新格式字段必须完全一致。
def verify_schema() -> None:
    validate_hole_command_schema({
        "command_schema_version": COMMAND_SCHEMA_VERSION,
        "command_fields": list(COMMAND_FIELDS),
    })
    for state in ({}, {"version": 2}, {
        "command_schema_version": COMMAND_SCHEMA_VERSION,
        "command_fields": list(reversed(COMMAND_FIELDS)),
    }):
        try:
            validate_hole_command_schema(state)
        except ValueError:
            continue
        raise AssertionError("错误命令格式未被拒绝")
    print("[通过] 新旧命令格式隔离，不会因观测同为113维而静默误用")


# 执行走廊热身与命令接口验收。
def main() -> None:
    torch.set_num_threads(1)
    verify_stages()
    verify_corridor()
    verify_obstacle_prior()
    verify_schema()
    print("[结论] 走廊热身与六维命令验收全部通过")


if __name__ == "__main__":
    main()
