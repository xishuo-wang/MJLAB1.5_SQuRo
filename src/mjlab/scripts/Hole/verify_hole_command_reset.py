# 用法：uv run python -B -m mjlab.scripts.Hole.verify_hole_command_reset
# 验证真实环境的复位起点、部分复位及基座高度规则边界。

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Hole.mdp import events
from mjlab.tasks.SQuRo_Hole.mdp.config import (
    CMD_HEIGHT_F_IDS,
    CMD_HEIGHT_H_IDS,
)
from mjlab.tasks.SQuRo_Hole.mdp.curriculums import (
    STAGE3_STEP,
    HeightCurriculumState,
    compute_base_heights,
    obstacle_rule_bounds,
)
from mjlab.tasks.SQuRo_Hole.mdp.entity import configure_obstacle_entity


# 构造只保留超时终止的真实 CPU 环境。
def make_env(reset_func=events.reset_model) -> ManagerBasedRlEnv:
    cfg = load_env_cfg("Mjlab-SQuRo-Hole")
    cfg.scene.num_envs = 4
    configure_obstacle_entity(cfg, enable_collision=True)
    state = HeightCurriculumState()
    state.z_min = state.z_max = 0.064
    cfg.commands["hole_cmd"].height_curriculum_state = state.state_dict()
    cfg.commands["hole_cmd"].enable_course_progress = False
    cfg.terminations = {"timeout": cfg.terminations["timeout"]}
    cfg.events["reset_all"].func = reset_func
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.common_step_counter = STAGE3_STEP
    env.reset(seed=42)
    return env


# 将机器人放到指定位置并刷新世界状态。
def write_positions(env: ManagerBasedRlEnv, x_values: tuple[float, ...]) -> None:
    robot = env.scene["robot"]
    pose = robot.data.root_link_pose_w.clone()
    pose[:, 0] = torch.tensor(x_values, device=env.device)
    pose[:, 1] = torch.linspace(-0.03, 0.03, env.num_envs, device=env.device)
    robot.write_root_link_pose_to_sim(pose)
    env.sim.forward()


# 比较实际根位置与位移诊断起点。
def check_origins(env: ManagerBasedRlEnv, label: str) -> None:
    command = env.command_manager.get_term("hole_cmd")
    torch.testing.assert_close(
        command.start_positions, env.scene["robot"].data.root_link_pos_w,
        atol=1e-6, rtol=0, msg=label,
    )


# 反复从非零旧位置复位，要求起点与实际根位置一致。
def check_repeated_reset(env: ManagerBasedRlEnv) -> None:
    for positions in ((0.2, 0.7, -0.1, 1.4), (1.0, -0.4, 0.32, 0.8)):
        write_positions(env, positions)
        env.reset()
        check_origins(env, "连续复位仍引用旧位置")
    print("[通过] 非零位置、负位置及连续复位的起点正确")


# 只复位选中的环境，其他环境的起点和状态保持原值。
def check_partial_reset(env: ManagerBasedRlEnv) -> None:
    env.reset()
    write_positions(env, (0.4, 0.7, 1.1, 0.9))
    command = env.command_manager.get_term("hole_cmd")
    before_origins = command.start_positions.clone()
    before_positions = env.scene["robot"].data.root_link_pos_w.clone()
    reset_ids = torch.tensor([0, 2], device=env.device)
    keep_ids = torch.tensor([1, 3], device=env.device)
    env.reset(env_ids=reset_ids)
    reset_positions = env.scene["robot"].data.root_link_pos_w[reset_ids]
    torch.testing.assert_close(command.start_positions[reset_ids], reset_positions)
    torch.testing.assert_close(reset_positions[:, :2], torch.zeros(2, 2))
    torch.testing.assert_close(command.start_positions[keep_ids], before_origins[keep_ids])
    torch.testing.assert_close(env.scene["robot"].data.root_link_pos_w[keep_ids], before_positions[keep_ids])
    print("[通过] 部分环境复位不污染其他环境的起点与位置")


# 在连续复位后检查当前单板高度规则的三个边界。
def check_schedule_boundaries(env: ManagerBasedRlEnv) -> None:
    obstacle = env.scene["obstacle"].cfg
    left, middle, right = obstacle_rule_bounds(obstacle.position[0], obstacle.size[0])
    board_bottom = torch.tensor(0.064)
    cases = (
        (left - 1e-6, left + 1e-6, middle - 1e-6, middle + 1e-6),
        (right - 1e-6, right + 1e-6, -0.01, 0.4),
    )
    for positions in cases:
        env.reset()
        write_positions(env, positions)
        env.command_manager.compute(dt=0.0)
        command = env.command_manager.get_command("hole_cmd")
        front, hind = compute_base_heights(
            torch.tensor(positions), obstacle.position[0], obstacle.size[0], board_bottom, True)
        expected = torch.stack((front, hind), dim=-1)
        torch.testing.assert_close(
            command[:, [CMD_HEIGHT_F_IDS, CMD_HEIGHT_H_IDS]], expected,
            atol=1e-6, rtol=0, msg="基座高度规则边界在复位后发生偏移",
        )
    print("[通过] 连续复位后仍按基座世界 X 查表，覆盖进入、中点及离开边界")


# 通过 env.step 触发异步超时，检查训练所用的自动复位路径。
def check_timeout_reset(env: ManagerBasedRlEnv) -> None:
    env.reset()
    write_positions(env, (0.4, 0.7, 1.1, 0.9))
    env.episode_length_buf[:] = 0
    reset_ids = torch.tensor([1, 3], device=env.device)
    env.episode_length_buf[reset_ids] = env.max_episode_length - 1
    actions = torch.zeros(env.num_envs, env.action_space.shape[-1], device=env.device)
    _, _, _, timeouts, _ = env.step(actions)
    assert timeouts.tolist() == [False, True, False, True], "超时复位的环境掩码不正确"
    command = env.command_manager.get_term("hole_cmd")
    reset_positions = env.scene["robot"].data.root_link_pos_w[reset_ids]
    torch.testing.assert_close(command.start_positions[reset_ids], reset_positions)
    torch.testing.assert_close(reset_positions[:, :2], torch.zeros(2, 2))
    torch.testing.assert_close(
        command.command[reset_ids][:, [CMD_HEIGHT_F_IDS, CMD_HEIGHT_H_IDS]],
        torch.tensor([[0.055, 0.055], [0.055, 0.055]]), atol=1e-6, rtol=0,
    )
    print("[通过] 训练 step 中的异步超时复位恢复板前默认高度")


# 测试替身在复位事件中写入非零根位置。
def reset_at_offset(env, env_ids) -> None:
    events.reset_model(env, env_ids)
    robot = env.scene["robot"]
    pose_ids = robot.indexing.free_joint_q_adr
    assert pose_ids is not None
    pose = robot.data.data.qpos[env_ids][:, pose_ids].clone()
    pose[:, 0] = 0.125
    robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)


# 自定义复位位置也必须成为起点，避免用固定零值掩盖错误。
def check_nonzero_reset_pose() -> None:
    env = make_env(reset_at_offset)
    try:
        write_positions(env, (0.7, -0.2, 1.4, 0.32))
        env.reset()
        check_origins(env, "非零复位位置没有成为起点")
        positions = (0.034, 0.125, 0.201, 0.316)
        write_positions(env, positions)
        env.command_manager.compute(dt=0.0)
        command = env.command_manager.get_command("hole_cmd")
        obstacle = env.scene["obstacle"].cfg
        front, hind = compute_base_heights(
            torch.tensor(positions), obstacle.position[0], obstacle.size[0],
            torch.tensor(0.064), True)
        torch.testing.assert_close(
            command[:, [CMD_HEIGHT_F_IDS, CMD_HEIGHT_H_IDS]],
            torch.stack((front, hind), dim=-1),
            atol=1e-6, rtol=0,
        )
    finally:
        env.close()
    print("[通过] 非零复位位置成为诊断起点，查表边界保持世界坐标不变")


# 执行复位与基座高度规则回归验收。
def main() -> None:
    torch.set_num_threads(1)
    env = make_env()
    try:
        check_repeated_reset(env)
        check_partial_reset(env)
        check_schedule_boundaries(env)
        check_timeout_reset(env)
    finally:
        env.close()
    check_nonzero_reset_pose()
    print("[结论] Hole 复位与基座高度规则边界验收全部通过")


if __name__ == "__main__":
    main()
