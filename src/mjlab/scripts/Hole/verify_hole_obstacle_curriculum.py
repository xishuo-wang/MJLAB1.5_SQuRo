from __future__ import annotations

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.Hole.visualize_hole_schedule import (
    sample_base_commands,
    sample_rule_schedule,
)
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Hole.mdp.config import (
    CMD_HEIGHT_F_IDS,
    CMD_HEIGHT_H_IDS,
    CMD_VEL_X_IDS,
    THRESHOLD_HEIGHT,
    HEIGHT_INITIAL_MIN,
    HEIGHT_MAX,
    HEIGHT_MIN,
    HEIGHT_MIN_DWELL_ITERATIONS,
    HEIGHT_MIN_SUCCESS_BATCH,
    HEIGHT_MIN_SUCCESS_RATE,
    HEIGHT_REQUIRED_BATCHES,
)
from mjlab.tasks.SQuRo_Hole.mdp.curriculums import (
    COURSE_START,
    COURSE_START_ITER,
    HEIGHT_COURSE_STAGE,
    STAGE2_END_ITER,
    STEPS_PER_ITER,
    HeightCurriculumState,
    get_current_stage,
    rule_bounds,
    sample_base_heights,
)
from mjlab.tasks.SQuRo_Hole.mdp.entity import (
    configure_single_hole_entities,
    hole_collision_enabled,
    read_bottom_z,
    write_hole_bottom_z,
)
from mjlab.tasks.SQuRo_Hole.mdp.indices import (
    _ACT_SPN_IDS,
    _MODEL_INDICES,
    resolve_model_indices,
)
from mjlab.tasks.SQuRo_Hole.mdp.rewards import compute_body_contact_reward
from mjlab.tasks.SQuRo_Hole.mdp.terminations import (
    check_fallen,
    monitor_hole_passage,
)

FAILURES: list[str] = []


# 记录验收结果并汇总失败。
def check(name: str, passed: bool) -> None:
    print(f"[{'通过' if passed else '失败'}] {name}")
    if not passed:
        FAILURES.append(name)


# 构造启用真实单板碰撞的 CPU 课程环境。
def make_env() -> ManagerBasedRlEnv:
    cfg = load_env_cfg("Mjlab-SQuRo-Hole", play=True)
    cfg.scene.num_envs = 4
    configure_single_hole_entities(cfg, enable_collision=True)
    command = cfg.commands["hole_cmd"]
    command.enable_height_course = True
    command.enable_course_progress = False
    command.height_curriculum_state = HeightCurriculumState().state_dict()
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.common_step_counter = COURSE_START
    env.reset(seed=37)
    return env


# 验证基座查表的边界、低板互斥和可视化一致性。
def verify_shared_rule() -> None:
    check(
        "三段热身后在 3000 iter 进入第四阶段课程",
        get_current_stage(STAGE2_END_ITER * STEPS_PER_ITER - 1) == 2
        and get_current_stage(STAGE2_END_ITER * STEPS_PER_ITER) == 3
        and get_current_stage(COURSE_START - 1) == 3
        and get_current_stage(COURSE_START) == HEIGHT_COURSE_STAGE == 4,
    )
    for center, half_length in ((0.2, 0.015), (0.6, 0.100)):
        left, middle, right = rule_bounds(center, half_length)
        board = ("Hole", center, 0.0, 0.07, half_length, 0.005)
        for dtype in (torch.float32, torch.float64):
            positions = torch.tensor([
                left - 1e-6, left, middle - 1e-6, middle,
                middle + 1e-6, right, right + 1e-6,
            ], dtype=dtype)
            for height in (0.048, 0.064, 0.069999, 0.070, 0.090):
                front, hind = sample_base_heights(positions, center, half_length, height)
                if height < 0.070:
                    expected_f = positions.new_tensor([0.055, 0.020, 0.020, 0.055, 0.055, 0.055, 0.055])
                    expected_h = positions.new_tensor([0.055, 0.055, 0.055, 0.020, 0.020, 0.020, 0.055])
                else:
                    expected_f = positions.new_tensor([0.055, *([height - 0.030] * 5), 0.055])
                    expected_h = expected_f
                check(f"板长 {half_length * 2000:.0f} mm、板高 {height * 1000:g} mm、{dtype} 边界正确",
                      torch.allclose(front, expected_f) and torch.allclose(hind, expected_h))
        query = torch.linspace(left - 0.1, right + 0.1, 2001, dtype=torch.float64)
        for height in (0.048, 0.064, 0.068, 0.070, 0.090):
            front, hind = sample_base_heights(query, center, half_length, height)
            check(f"板长 {half_length * 2000:.0f} mm、板高 {height * 1000:g} mm 全区间没有双低",
                  not bool(((front < THRESHOLD_HEIGHT) & (hind < THRESHOLD_HEIGHT)).any()))
            preview = sample_rule_schedule(query.numpy() - 0.1, board, height, start_x=0.1)
            current = sample_base_commands(query.numpy() - 0.1, board, height, start_x=0.1)
            check("非零起点下生产、曲线及游标读数一致且速度为正",
                  torch.allclose(torch.as_tensor(preview["height_f"]), front)
                  and torch.allclose(torch.as_tensor(current["height_h"]), hind)
                  and bool((torch.as_tensor(current["velocity_x"]) > 0).all()))


# 验证课程门控、批次失败重置、最低样本过滤与驻留下界。
def verify_curriculum_state() -> None:
    state = HeightCurriculumState(start_iteration=COURSE_START_ITER)
    for _ in range(HEIGHT_MIN_SUCCESS_BATCH):
        state.record_episode(HEIGHT_INITIAL_MIN, True, STAGE2_END_ITER + 1, 2, True)
    check("第二阶段和无碰撞回合不会推进课程", state.z_min == HEIGHT_INITIAL_MIN and state.all_height_episode_count == 0)
    state = HeightCurriculumState(start_iteration=0)
    for _ in range(HEIGHT_MIN_SUCCESS_BATCH):
        state.record_episode(HEIGHT_INITIAL_MIN, True, 100, HEIGHT_COURSE_STAGE, True)
    for _ in range(HEIGHT_MIN_SUCCESS_BATCH):
        state.record_episode(HEIGHT_INITIAL_MIN, False, 100, HEIGHT_COURSE_STAGE, True)
    check("失败批清零连续达标批数", state.consecutive_success_batches == 0)
    for iteration in range(HEIGHT_MIN_DWELL_ITERATIONS + 1):
        for _ in range(HEIGHT_MIN_SUCCESS_BATCH):
            state.record_episode(HEIGHT_INITIAL_MIN, True, iteration, HEIGHT_COURSE_STAGE, True)
        if state.z_min < HEIGHT_INITIAL_MIN:
            break
    check("达到两批、驻留期和 60% 后才降档", state.z_min == HEIGHT_INITIAL_MIN - 0.002 and state.consecutive_success_batches == 0)
    state.z_min = HEIGHT_MIN
    for _ in range(HEIGHT_MIN_SUCCESS_BATCH * HEIGHT_REQUIRED_BATCHES):
        state.record_episode(HEIGHT_MIN, True, 1000, HEIGHT_COURSE_STAGE, True)
    check("最低档不会跌破课程下界", state.z_min == HEIGHT_MIN)
    restored = HeightCurriculumState()
    restored.load_state_dict(state.state_dict())
    check("课程 checkpoint 状态往返", restored.state_dict() == state.state_dict())
    check("最低成功率门槛为 60%", HEIGHT_MIN_SUCCESS_RATE == 0.60 and HEIGHT_MAX == 0.090)

    delayed = HeightCurriculumState(start_iteration=0)
    for _ in range(HEIGHT_MIN_SUCCESS_BATCH * HEIGHT_REQUIRED_BATCHES):
        delayed.record_episode(HEIGHT_INITIAL_MIN, True, 99, HEIGHT_COURSE_STAGE, True)
    delayed_result = delayed.record_episode(
        HEIGHT_INITIAL_MIN, True, 100, HEIGHT_COURSE_STAGE, True
    )
    check(
        "驻留达标后的部分批次降档也返回完整日志字段",
        delayed.z_min == HEIGHT_INITIAL_MIN - 0.002
        and all(key in delayed_result for key in (
            "previous_z_min", "z_min", "success_rate", "sample_count"
        )),
    )


# 把九个采样点放在指定位置，驱动监测器的成功与失败路径。
def set_sites(
    env: ManagerBasedRlEnv,
    x: float,
    y: float = 0.0,
    z: float = 0.04,
    env_ids: torch.Tensor | None = None,
) -> None:
    robot = env.scene["robot"]
    resolve_model_indices(robot)
    site_ids = list(_MODEL_INDICES.f_body_site_ids) + list(_MODEL_INDICES.h_body_site_ids)
    global_site_ids = robot.indexing.site_ids[torch.tensor(site_ids, device=env.device)]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    raw_site_pos = robot.data.data.site_xpos
    with torch.inference_mode():
        row_ids = env_ids[:, None]
        site_ids = global_site_ids[None, :]
        raw_site_pos[row_ids, site_ids, 0] = x
        raw_site_pos[row_ids, site_ids, 1] = y
        raw_site_pos[row_ids, site_ids, 2] = z


# 检查 monitor 能识别穿板成功、侧绕、上穿、跌倒和未通过超时。
def verify_passage_monitor(env: ManagerBasedRlEnv) -> None:
    command = env.command_manager.get_term("hole_cmd")
    command.obstacle_height_sampled[:] = True
    with torch.inference_mode():
        monitor_hole_passage(env)
    monitor = env._hole_passage_monitor
    env.reset(env_ids=torch.tensor([0, 2], device=env.device))
    check("inference step 后可从外部执行部分 reset", not torch.is_inference(monitor.approached))
    board = env.scene.entities["hole1"]
    write_hole_bottom_z(board, 0.070)
    x1 = float(board.cfg.position[0]) - float(board.cfg.size[0])
    x2 = float(board.cfg.position[0]) + float(board.cfg.size[0])
    bottom = float(read_bottom_z(board)[0])

    set_sites(env, x1 - 0.01)
    monitor.compute()
    set_sites(env, x1 + 0.005, z=bottom)
    monitor.compute()
    set_sites(env, x2 + 0.01, z=bottom)
    env.episode_length_buf[0] = env.max_episode_length
    monitor.compute()
    success = next(row for row in monitor.consume() if row["env_id"] == 0)
    check("合法贴板穿越在 timeout 回合中仍成功", success["success"] and success["failure_code"] == 0)

    monitor.reset(torch.tensor([1, 2, 3], device=env.device))
    set_sites(env, x1 - 0.01)
    monitor.compute()
    set_sites(env, x1 + 0.005, z=bottom)
    set_sites(env, x1 + 0.005, y=0.15, z=bottom + 0.02, env_ids=torch.tensor([1], device=env.device))
    set_sites(env, x1 + 0.005, z=bottom + 0.02, env_ids=torch.tensor([2], device=env.device))
    monitor.compute()
    set_sites(env, x2 + 0.01, y=0.0, z=bottom)
    env.episode_length_buf[1] = env.max_episode_length
    env.episode_length_buf[2] = env.max_episode_length
    env.episode_length_buf[3] = env.max_episode_length
    robot = env.scene["robot"]
    pose = robot.data.root_link_pose_w.clone()
    pose[3, 3:7] = pose.new_tensor([0.7071068, 0.7071068, 0.0, 0.0])
    with torch.inference_mode():
        robot.write_root_link_pose_to_sim(pose)
        env.sim.forward()
        set_sites(env, x2 + 0.01, y=0.0, z=bottom)
    check("跌倒姿态注入命中既有跌倒判据", bool(check_fallen(env)[3]))
    monitor.compute()
    rows = {row["env_id"]: row for row in monitor.consume()}
    check("侧绕优先于高度超限判为失败", rows[1]["failure_code"] == 2 and not rows[1]["success"])
    check("上穿不计成功", rows[2]["failure_code"] == 3 and not rows[2]["success"])
    check("跌倒失败优先级最高", rows[3]["failure_code"] == 4 and not rows[3]["success"])
    monitor.reset(torch.tensor([0], device=env.device))
    set_sites(env, x1 - 0.01, env_ids=torch.tensor([0], device=env.device))
    monitor.compute()
    env.episode_length_buf[0] = env.max_episode_length
    monitor.compute()
    timeout = next(row for row in monitor.consume() if row["env_id"] == 0)
    check("未通过就 timeout 记失败分母", timeout["failure_code"] == 5 and not timeout["success"])


# 验证周期命令重采样不移动障碍，以及虚拟净空随实际 mocap 高度变化。
def verify_dynamic_sources(env: ManagerBasedRlEnv) -> None:
    command = env.command_manager.get_term("hole_cmd")
    board = env.scene.entities["hole1"]
    previous = read_bottom_z(board).clone()
    command._resample_command(torch.arange(env.num_envs, device=env.device))
    check("周期命令重采样不改变回合板高", torch.equal(read_bottom_z(board), previous))

    robot = env.scene["robot"]
    resolve_model_indices(robot)
    body_ids = robot.indexing.body_ids[
        torch.tensor([_MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id], device=env.device)
    ]
    local_sites = list(_MODEL_INDICES.f_body_site_ids) + list(_MODEL_INDICES.h_body_site_ids)
    site_ids = robot.indexing.site_ids[torch.tensor(local_sites, device=env.device)]
    with torch.inference_mode():
        robot.data.data.xpos[0, body_ids, 0] = float(board.cfg.position[0])
        robot.data.data.site_xpos[0, site_ids, 2] = 0.080
    write_hole_bottom_z(board, 0.048)
    low_reward = compute_body_contact_reward(env)[0]
    write_hole_bottom_z(board, 0.090)
    high_reward = compute_body_contact_reward(env)[0]
    check("虚拟净空奖励阈值跟随实际板底", high_reward > low_reward)


# 从独立的碰撞几何与编译尺寸读取实际板底。
def read_geom_bottom_z(env: ManagerBasedRlEnv) -> torch.Tensor:
    board = env.scene.entities["hole1"]
    local_ids, _ = board.find_geoms([f"{board.cfg.name}_geom"], preserve_order=True)
    geom_id = local_ids[0]
    global_id = int(board.indexing.geom_ids[geom_id])
    half_thickness = float(env.sim.mj_model.geom_size[global_id, 2])
    env.sim.forward()
    env.scene.update(0.0)
    return board.data.geom_pos_w[:, geom_id, 2] - half_thickness


# 改变脊柱姿态后，命令仍只跟随 base_Link 的世界 X。
def verify_base_height_commands(env: ManagerBasedRlEnv) -> None:
    env.reset()
    robot = env.scene["robot"]
    board = env.scene.entities["hole1"]
    command = env.command_manager.get_term("hole_cmd")
    resolve_model_indices(robot)
    pose = robot.data.root_link_pose_w.clone()
    pose[:, 0] = pose.new_tensor([0.034, 0.199, 0.200, 0.316])
    robot.write_root_link_pose_to_sim(pose)
    joint_pos = robot.data.joint_pos.clone()
    spine_ids = [_MODEL_INDICES.joint_ids[i] for i in _ACT_SPN_IDS]
    joint_pos[:, spine_ids] = joint_pos.new_tensor([
        [0.4, 0.0, 0.3, 0.0], [-0.4, 0.0, -0.3, 0.0],
        [0.3, 0.0, -0.2, 0.0], [-0.3, 0.0, 0.2, 0.0],
    ])
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    write_hole_bottom_z(board, 0.064)
    read_geom_bottom_z(env)
    command._update_command_from_height_rule()
    values = command.command
    check("脊柱侧摆和俯仰变化时，低板仍按基座 X 在中点互斥切换",
          torch.allclose(values[:, CMD_HEIGHT_F_IDS], values.new_tensor([0.055, 0.020, 0.055, 0.055]))
          and torch.allclose(values[:, CMD_HEIGHT_H_IDS], values.new_tensor([0.055, 0.055, 0.020, 0.055]))
          and bool((values[:, CMD_VEL_X_IDS] > 0).all()))
    write_hole_bottom_z(board, 0.090)
    read_geom_bottom_z(env)
    command._update_command_from_height_rule()
    expected = values.new_tensor([0.055, 0.060, 0.060, 0.055])
    check("高板前后端同步进入和离开基座区间",
          torch.allclose(values[:, CMD_HEIGHT_F_IDS], expected)
          and torch.allclose(values[:, CMD_HEIGHT_H_IDS], expected))
    env.reset()


# 验收真实碰撞几何、逐环境复位和动态奖励阈值。
def verify_real_env() -> None:
    env = make_env()
    board = env.scene.entities["hole1"]
    check("训练环境只保留单板且碰撞已编译", list(name for name in env.scene.entities if name.startswith("hole")) == ["hole1"] and hole_collision_enabled(env))
    write_hole_bottom_z(board, torch.tensor([0.048, 0.052, 0.070, 0.090]), torch.arange(4, device=env.device))
    written = read_bottom_z(board).clone()
    check("板体 mocap 逐环境写入与即时读取", torch.allclose(written, torch.tensor([0.048, 0.052, 0.070, 0.090])))
    check("独立碰撞几何的板底随四个采样高度移动", torch.allclose(
        read_geom_bottom_z(env), torch.tensor([0.048, 0.052, 0.070, 0.090]), atol=1e-6,
    ))
    keep = written[[1, 3]].clone()
    write_hole_bottom_z(board, 0.051, torch.tensor([0, 2], device=env.device))
    check("部分写入不污染其他环境", torch.equal(read_bottom_z(board)[[1, 3]], keep))
    check("部分写入只移动指定环境的实际碰撞板", torch.allclose(
        read_geom_bottom_z(env), torch.tensor([0.051, 0.052, 0.051, 0.090]), atol=1e-6,
    ))
    write_hole_bottom_z(board, torch.tensor([0.054, 0.056]), slice(1, 4, 2))
    check("切片写入正确移动实际碰撞板", torch.allclose(
        read_geom_bottom_z(env), torch.tensor([0.051, 0.054, 0.051, 0.056]), atol=1e-6,
    ))
    keep = read_geom_bottom_z(env)[[1, 3]].clone()
    with torch.inference_mode():
        monitor_hole_passage(env)
    monitor = env._hole_passage_monitor
    env.reset(env_ids=torch.tensor([0, 2], device=env.device))
    after_reset = read_bottom_z(board).clone()
    check("部分 reset 重采样且保留其他环境板高", torch.equal(after_reset[[1, 3]], keep) and torch.all(after_reset[[0, 2]] >= HEIGHT_INITIAL_MIN))
    actual = read_geom_bottom_z(env)
    check("部分 reset 后碰撞几何与采样高度一致", torch.allclose(actual, after_reset, atol=1e-6)
          and torch.equal(actual[[1, 3]], keep))
    check("inference step 后可从外部执行部分 reset", not torch.is_inference(monitor.approached))
    verify_base_height_commands(env)
    verify_dynamic_sources(env)
    verify_passage_monitor(env)
    env.close()


# 运行全部课程验收。
def main() -> None:
    print("单板受限空间高度课程验收")
    verify_shared_rule()
    verify_curriculum_state()
    verify_real_env()
    if FAILURES:
        raise SystemExit(f"失败项: {FAILURES}")
    print("验收通过")


if __name__ == "__main__":
    main()
