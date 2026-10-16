from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

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
    CMD_GAIT_FREQ_IDS,
    COMMAND_DIM,
    BASE_FREQ,
    BASE_HEIGHT,
    FULL_HEIGHT_VALUES,
    RANDOM_HEIGHT_VALUES,
    HEIGHT_LIST,
    LOW_POSE_CALIBRATION,
    LOW_SPINE_PITCH,
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
    STAGE3_STEP,
    STAGE3_ITER,
    HEIGHT_RULE_STAGE,
    STAGE2_ITER,
    STEPS_PER_ITER,
    HeightCurriculumState,
    get_current_stage,
    obstacle_rule_bounds,
    compute_base_heights,
)
from mjlab.tasks.SQuRo_Hole.mdp.entity import (
    configure_obstacle_entity,
    obstacle_collision_enabled,
    read_obstacle_bottom_z,
    write_obstacle_mocap,
)
from mjlab.tasks.SQuRo_Hole.mdp.indices import (
    _ACT_SPN_IDS,
    _ACT_LEG_F_IDS,
    _ACT_LEG_H_IDS,
    _ACT_SPN_H_PITCH_ID,
    _MODEL_INDICES,
    resolve_model_indices,
)
from mjlab.tasks.SQuRo_Hole.mdp.command import get_height_mode
from mjlab.tasks.SQuRo_Hole.mdp.reference import get_reference_joint_state, initialize_tables
from mjlab.tasks.SQuRo_Hole.mdp.terminations import (
    check_fallen,
    monitor_restricted_passage,
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
    configure_obstacle_entity(cfg, enable_collision=True)
    command = cfg.commands["hole_cmd"]
    command.enable_height_course = True
    command.enable_course_progress = False
    command.height_curriculum_state = HeightCurriculumState().state_dict()
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.common_step_counter = STAGE3_STEP
    env.reset(seed=37)
    return env


# 验证基座查表的边界、低板互斥和可视化一致性。
def verify_shared_rule() -> None:
    check(
        "三段热身后在 3000 iter 进入第四阶段课程",
        get_current_stage(STAGE2_ITER * STEPS_PER_ITER - 1) == 2
        and get_current_stage(STAGE2_ITER * STEPS_PER_ITER) == 3
        and get_current_stage(STAGE3_STEP - 1) == 3
        and get_current_stage(STAGE3_STEP) == HEIGHT_RULE_STAGE == 4,
    )
    for center, half_length in ((0.2, 0.015), (0.6, 0.100)):
        left, middle, right = obstacle_rule_bounds(center, half_length)
        obstacle = ("Hole", center, 0.0, 0.07, half_length, 0.005)
        for dtype in (torch.float32, torch.float64):
            positions = torch.tensor([
                left - 1e-6, left, middle - 1e-6, middle,
                middle + 1e-6, right, right + 1e-6,
            ], dtype=dtype)
            low_cases = (
                (0.045, 0.020), (HEIGHT_MIN, 0.020), (0.047, 0.020), (0.049999, 0.020),
                (0.050, 0.025), (0.050001, 0.025), (0.054999, 0.025),
                (0.055, 0.030), (0.055001, 0.030), (0.059999, 0.030),
                (0.060, 0.035), (0.060001, 0.035), (0.064999, 0.035),
            )
            high_cases = (
                (0.065, 0.040), (0.069999, 0.040), (0.070, 0.045), (0.074999, 0.045),
                (0.075, 0.050), (0.079999, 0.050), (0.080, 0.055), (0.090, 0.055),
            )
            for height, target in (*low_cases, *high_cases):
                front, hind = compute_base_heights(positions, center, half_length, height)
                if height < 0.065:
                    expected_f = positions.new_tensor([0.055, target, target, 0.055, 0.055, 0.055, 0.055])
                    expected_h = positions.new_tensor([0.055, 0.055, 0.055, target, target, target, 0.055])
                else:
                    expected_f = positions.new_tensor([0.055, *([target] * 5), 0.055])
                    expected_h = expected_f
                check(f"板长 {half_length * 2000:.0f} mm、板高 {height * 1000:g} mm、{dtype} 边界正确",
                      torch.allclose(front, expected_f) and torch.allclose(hind, expected_h))
                expected_modes = [0, 1, 1, 2, 2, 2, 0] if height < 0.065 else [0] * 7
                check("各板高离散挡位与高低模式判定一致",
                      get_height_mode(front, hind).tolist() == expected_modes)
        query = torch.linspace(left - 0.1, right + 0.1, 2001, dtype=torch.float64)
        for height in (HEIGHT_MIN, 0.048, 0.064, 0.065, 0.068, 0.070, 0.080, 0.090):
            front, hind = compute_base_heights(query, center, half_length, height)
            check(f"板长 {half_length * 2000:.0f} mm、板高 {height * 1000:g} mm 全区间没有双低",
                  not bool(((front < THRESHOLD_HEIGHT) & (hind < THRESHOLD_HEIGHT)).any()))
            preview = sample_rule_schedule(query.numpy() - 0.1, obstacle, height, start_x=0.1)
            current = sample_base_commands(query.numpy() - 0.1, obstacle, height, start_x=0.1)
            check("非零起点下生产、曲线及游标读数一致且速度为正",
                  torch.allclose(torch.as_tensor(preview["height_f"]), front)
                  and torch.allclose(torch.as_tensor(current["height_h"]), hind)
                  and bool((torch.as_tensor(current["velocity_x"]) > 0).all()))


# 验证课程门控、批次失败重置、最低样本过滤与驻留下界。
def verify_curriculum_state() -> None:
    state = HeightCurriculumState(level_start_iteration=STAGE3_ITER)
    for _ in range(HEIGHT_MIN_SUCCESS_BATCH):
        state.record_episode(HEIGHT_INITIAL_MIN, True, STAGE2_ITER + 1, 2, True)
    check("第二阶段和无碰撞回合不会推进课程", state.z_min == HEIGHT_INITIAL_MIN and state.all_height_episode_count == 0)
    state = HeightCurriculumState(level_start_iteration=0)
    for _ in range(HEIGHT_MIN_SUCCESS_BATCH):
        state.record_episode(HEIGHT_INITIAL_MIN, True, 100, HEIGHT_RULE_STAGE, True)
    for _ in range(HEIGHT_MIN_SUCCESS_BATCH):
        state.record_episode(HEIGHT_INITIAL_MIN, False, 100, HEIGHT_RULE_STAGE, True)
    check("失败批清零连续达标批数", state.consecutive_success_batches == 0)
    for iteration in range(HEIGHT_MIN_DWELL_ITERATIONS + 1):
        for _ in range(HEIGHT_MIN_SUCCESS_BATCH):
            state.record_episode(HEIGHT_INITIAL_MIN, True, iteration, HEIGHT_RULE_STAGE, True)
        if state.z_min < HEIGHT_INITIAL_MIN:
            break
    check("达到两批、驻留期和 60% 后才降档", state.z_min == HEIGHT_INITIAL_MIN - 0.002 and state.consecutive_success_batches == 0)
    state.z_min = HEIGHT_MIN
    for _ in range(HEIGHT_MIN_SUCCESS_BATCH * HEIGHT_REQUIRED_BATCHES):
        state.record_episode(HEIGHT_MIN, True, 1000, HEIGHT_RULE_STAGE, True)
    check("最低档不会跌破课程下界", state.z_min == HEIGHT_MIN)
    restored = HeightCurriculumState()
    restored.load_state_dict(state.state_dict())
    check("课程 checkpoint 状态往返", restored.state_dict() == state.state_dict())
    check("最低成功率门槛为 60%", HEIGHT_MIN_SUCCESS_RATE == 0.60 and HEIGHT_MAX == 0.090)

    delayed = HeightCurriculumState(level_start_iteration=0)
    for _ in range(HEIGHT_MIN_SUCCESS_BATCH * HEIGHT_REQUIRED_BATCHES):
        delayed.record_episode(HEIGHT_INITIAL_MIN, True, 99, HEIGHT_RULE_STAGE, True)
    delayed_result = delayed.record_episode(
        HEIGHT_INITIAL_MIN, True, 100, HEIGHT_RULE_STAGE, True
    )
    check(
        "驻留达标后的部分批次降档也返回完整日志字段",
        delayed.z_min == HEIGHT_INITIAL_MIN - 0.002
        and all(key in delayed_result for key in (
            "previous_z_min", "z_min", "success_rate", "sample_count"
        )),
    )


# 验证50毫米边界的细步长、46毫米下限和旧检查点续训。
def verify_fine_height_steps() -> None:
    expected_mm = (70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 49, 48, 47, 46)
    state = HeightCurriculumState()
    for index, height_mm in enumerate(expected_mm[1:], start=1):
        previous_height = state.z_min
        iteration = STAGE3_ITER + index * HEIGHT_MIN_DWELL_ITERATIONS
        for _ in range(HEIGHT_MIN_SUCCESS_BATCH * HEIGHT_REQUIRED_BATCHES):
            state.record_episode(previous_height, True, iteration, HEIGHT_RULE_STAGE, True)
        check(f"课程由 {previous_height * 1000:g} 降到 {height_mm} mm",
              state.z_min == height_mm / 1000 and state.consecutive_success_batches == 0)
    check("46毫米下限与完整降档序列一致", HEIGHT_MIN == expected_mm[-1] / 1000)
    for _ in range(HEIGHT_MIN_SUCCESS_BATCH * HEIGHT_REQUIRED_BATCHES):
        state.record_episode(state.z_min, True, iteration + HEIGHT_MIN_DWELL_ITERATIONS, HEIGHT_RULE_STAGE, True)
    check("46毫米达标后保持下限", state.z_min == HEIGHT_MIN)

    fine = HeightCurriculumState(level_start_iteration=0)
    fine.z_min = 0.050
    for _ in range(HEIGHT_MIN_SUCCESS_BATCH * HEIGHT_REQUIRED_BATCHES):
        fine.record_episode(fine.z_min, True, HEIGHT_MIN_DWELL_ITERATIONS - 1, HEIGHT_RULE_STAGE, True)
    check("50毫米档成功达标后仍须满足驻留时间", fine.z_min == 0.050)
    fine.record_episode(fine.z_min, True, HEIGHT_MIN_DWELL_ITERATIONS, HEIGHT_RULE_STAGE, True)
    check("50毫米驻留达标后只降低1毫米", fine.z_min == 0.049)

    failed = HeightCurriculumState(level_start_iteration=0)
    failed.z_min = 0.050
    for passed in (True, False):
        for _ in range(HEIGHT_MIN_SUCCESS_BATCH):
            failed.record_episode(failed.z_min, passed, HEIGHT_MIN_DWELL_ITERATIONS, HEIGHT_RULE_STAGE, True)
    check("细步长阶段失败批仍清零且不降档",
          failed.z_min == 0.050 and failed.consecutive_success_batches == 0)

    old = HeightCurriculumState(level_start_iteration=0)
    old.z_min = 0.048
    restored = HeightCurriculumState()
    restored.load_state_dict(old.state_dict())
    check("旧48毫米检查点加载时保留原课程状态", restored.state_dict() == old.state_dict())
    for _ in range(HEIGHT_MIN_SUCCESS_BATCH * HEIGHT_REQUIRED_BATCHES):
        restored.record_episode(0.048, True, HEIGHT_MIN_DWELL_ITERATIONS, HEIGHT_RULE_STAGE, True)
    check("旧48毫米检查点达标后按新规则降至47毫米", restored.z_min == 0.047)


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
        monitor_restricted_passage(env)
    monitor = env._hole_passage_monitor
    env.reset(env_ids=torch.tensor([0, 2], device=env.device))
    check("inference step 后可从外部执行部分 reset", not torch.is_inference(monitor.approached))
    obstacle = env.scene.entities["obstacle"]
    write_obstacle_mocap(obstacle, 0.070)
    x1 = float(obstacle.cfg.position[0]) - float(obstacle.cfg.size[0])
    x2 = float(obstacle.cfg.position[0]) + float(obstacle.cfg.size[0])
    bottom = float(read_obstacle_bottom_z(obstacle)[0])

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


# 验证周期命令重采样不移动障碍。
def verify_dynamic_sources(env: ManagerBasedRlEnv) -> None:
    command = env.command_manager.get_term("hole_cmd")
    obstacle = env.scene.entities["obstacle"]
    previous = read_obstacle_bottom_z(obstacle).clone()
    command._resample_command(torch.arange(env.num_envs, device=env.device))
    check("周期命令重采样不改变回合板高", torch.equal(read_obstacle_bottom_z(obstacle), previous))


# 随机计时的首回合无论成败均排除，完整回合恢复正常统计。
def verify_partial_episode_filter(env: ManagerBasedRlEnv) -> None:
    command = env.command_manager.get_term("hole_cmd")
    command.height_curriculum_state = HeightCurriculumState()
    command.enable_course_progress = True
    env.reset()
    obstacle = env.scene.entities["obstacle"]
    write_obstacle_mocap(obstacle, HEIGHT_INITIAL_MIN)
    command.course_episode_eligible[:] = torch.tensor([False, False, True, True], device=env.device)
    monitor = env._hole_passage_monitor
    monitor.consume()
    x1 = obstacle.cfg.position[0] - obstacle.cfg.size[0]
    x2 = obstacle.cfg.position[0] + obstacle.cfg.size[0]
    passed_ids = torch.tensor([1, 3], device=env.device)
    set_sites(env, x1 - 0.01)
    monitor.compute()
    set_sites(env, x1 + 0.005, env_ids=passed_ids)
    monitor.compute()
    set_sites(env, x2 + 0.01, env_ids=passed_ids)
    env.episode_length_buf[:] = env.max_episode_length
    monitor.compute()
    rows = {row["env_id"]: row for row in monitor.consume()}
    state = command.height_curriculum_state
    check("截短首回合成功和失败均不计课程分母",
          not rows[0]["valid"] and not rows[1]["valid"] and rows[1]["success"]
          and state.all_height_episode_count == 2 and state.all_height_success_count == 1
          and state.level_episode_count == 2 and state.level_success_count == 1)
    env.reset(env_ids=torch.tensor([0], device=env.device))
    check("首次自然复位恢复当前环境资格且不影响其他截短回合",
          command.course_episode_eligible.tolist() == [True, False, True, True])
    env.reset()
    write_obstacle_mocap(obstacle, HEIGHT_INITIAL_MIN)
    set_sites(env, x1 - 0.01)
    monitor.compute()
    set_sites(env, x1 + 0.005)
    monitor.compute()
    set_sites(env, x2 + 0.01)
    env.episode_length_buf[:] = env.max_episode_length
    monitor.compute()
    monitor.consume()
    check("后续完整回合全部恢复课程统计",
          command.course_episode_eligible.all()
          and state.all_height_episode_count == 6 and state.all_height_success_count == 5)
    command.enable_course_progress = False
    env.reset()


# 从独立的碰撞几何与编译尺寸读取实际板底。
def read_geom_bottom_z(env: ManagerBasedRlEnv) -> torch.Tensor:
    obstacle = env.scene.entities["obstacle"]
    local_ids, _ = obstacle.find_geoms([f"{obstacle.cfg.name}_geom"], preserve_order=True)
    geom_id = local_ids[0]
    global_id = int(obstacle.indexing.geom_ids[geom_id])
    half_thickness = float(env.sim.mj_model.geom_size[global_id, 2])
    env.sim.forward()
    env.scene.update(0.0)
    return obstacle.data.geom_pos_w[:, geom_id, 2] - half_thickness


# 改变脊柱姿态后，命令仍只跟随 base_Link 的世界 X。
def verify_base_height_commands(env: ManagerBasedRlEnv) -> None:
    env.reset()
    robot = env.scene["robot"]
    obstacle = env.scene.entities["obstacle"]
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
    write_obstacle_mocap(obstacle, 0.064)
    read_geom_bottom_z(env)
    command._update_command_by_height_rule()
    values = command.command
    check("脊柱侧摆和俯仰变化时，低板仍按基座 X 在中点互斥切换",
          torch.allclose(values[:, CMD_HEIGHT_F_IDS], values.new_tensor([0.055, 0.035, 0.055, 0.055]))
          and torch.allclose(values[:, CMD_HEIGHT_H_IDS], values.new_tensor([0.055, 0.055, 0.035, 0.055]))
          and bool((values[:, CMD_VEL_X_IDS] > 0).all()))
    for height, target in ((0.065, 0.040), (0.069, 0.040), (0.070, 0.045), (0.074, 0.045),
                           (0.075, 0.050), (0.079, 0.050), (0.080, 0.055), (0.090, 0.055)):
        write_obstacle_mocap(obstacle, height)
        read_geom_bottom_z(env)
        command._update_command_by_height_rule()
        expected = values.new_tensor([0.055, target, target, 0.055])
        check(f"{height * 1000:g}毫米板底对应两段{target * 1000:g}毫米目标，区间外仍55毫米",
              torch.allclose(values[:, CMD_HEIGHT_F_IDS], expected)
              and torch.allclose(values[:, CMD_HEIGHT_H_IDS], expected))
    write_obstacle_mocap(obstacle, values.new_tensor([0.065, 0.070, 0.075, 0.090]))
    read_geom_bottom_z(env)
    command._update_command_by_height_rule()
    expected = values.new_tensor([0.055, 0.045, 0.050, 0.055])
    check("不同环境板高独立决定命令，高板目标不超过基准55毫米",
          torch.allclose(values[:, CMD_HEIGHT_F_IDS], expected)
          and torch.allclose(values[:, CMD_HEIGHT_H_IDS], expected))
    pose[:, 0] = 0.1
    robot.write_root_link_pose_to_sim(pose)
    write_obstacle_mocap(obstacle, values.new_tensor([0.047, 0.050, 0.055, 0.060]))
    read_geom_bottom_z(env)
    command._update_command_by_height_rule()
    check("逐环境真实板高分别映射20／25／30／35毫米低姿态且先验仍为实际板高",
          torch.allclose(values[:, CMD_HEIGHT_F_IDS], values.new_tensor([0.020, 0.025, 0.030, 0.035]))
          and torch.allclose(values[:, CMD_HEIGHT_H_IDS], values.new_full((4,), BASE_HEIGHT))
          and torch.allclose(command.command_obstacle_height, values.new_tensor([0.047, 0.050, 0.055, 0.060])))
    env.reset()


# 验证固定角标定插值、脊柱高度差、双低独立性及表格一致性。
def verify_continuous_reference(env: ManagerBasedRlEnv) -> None:
    check("前1000轮高度池保持不变", RANDOM_HEIGHT_VALUES == [0.04, 0.045, 0.05, 0.055])
    check("1000至3000轮扩展为八档", FULL_HEIGHT_VALUES == [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055])
    pairs = torch.cartesian_prod(torch.tensor(FULL_HEIGHT_VALUES), torch.tensor(FULL_HEIGHT_VALUES))
    counts = torch.bincount(get_height_mode(pairs[:, 0], pairs[:, 1]), minlength=4)
    check("前后独立均匀采样的64种组合中四种模式各占四分之一", counts.tolist() == [16] * 4)
    extra = torch.tensor([[0.021, 0.055], [0.055, 0.022], [0.0273, 0.0317],
                          [0.0398, 0.055], [0.055, 0.039999]])
    heights = torch.cat((pairs, extra)).to(env.device)
    command = torch.zeros(len(heights), COMMAND_DIM, device=env.device)
    command[:, CMD_HEIGHT_F_IDS] = heights[:, 0]
    command[:, CMD_HEIGHT_H_IDS] = heights[:, 1]
    command[:, CMD_GAIT_FREQ_IDS] = BASE_FREQ
    probe = SimpleNamespace(device=env.device, step_dt=env.step_dt,
        episode_length_buf=torch.arange(len(heights), device=env.device),
        command_manager=SimpleNamespace(_terms={"hole_cmd": SimpleNamespace(command=command)}))
    tables = initialize_tables(env.device)
    pos, vel = get_reference_joint_state(probe)
    original = pos.clone()
    check("连续参考与速度均为有限值", bool(torch.isfinite(pos).all() and torch.isfinite(vel).all()))
    for height, leg_ids, is_front in ((heights[:, 0], _ACT_LEG_F_IDS, True), (heights[:, 1], _ACT_LEG_H_IDS, False)):
        low = height < THRESHOLD_HEIGHT
        expected = []
        for value in height[low].tolist():
            value = min(max(value, LOW_POSE_CALIBRATION[0][0]), LOW_POSE_CALIBRATION[-1][0])
            upper = next((index for index, row in enumerate(LOW_POSE_CALIBRATION) if row[0] >= value), len(LOW_POSE_CALIBRATION) - 1)
            lower = max(0, upper - 1)
            left, right = LOW_POSE_CALIBRATION[lower], LOW_POSE_CALIBRATION[upper]
            fraction = (value - left[0]) / (right[0] - left[0]) if upper != lower else 0.0
            left_angles = left[1] if is_front else left[2]
            right_angles = right[1] if is_front else right[2]
            expected.append([a + fraction * (b - a) for a, b in zip(left_angles, right_angles)])
        expected_tensor = pos.new_tensor(expected).repeat(1, 2)
        label = "前" if is_front else "后"
        check(f"{label}腿连续高度严格按标定角度分段插值",
              torch.allclose(pos[low][:, leg_ids], expected_tensor, atol=3e-7, rtol=0))
        check(f"{label}受限侧参考速度严格为零", bool((vel[low][:, leg_ids] == 0).all()))
        for index, height_value in enumerate(HEIGHT_LIST):
            if height_value >= THRESHOLD_HEIGHT:
                continue
            rows = low & torch.isclose(height, height.new_tensor(height_value), atol=1e-8, rtol=0)
            mode = get_height_mode(heights[:, 0], heights[:, 1])[rows]
            key = "front_pos" if is_front else "hind_pos"
            check(f"{label}腿{height_value * 1000:g}毫米预计算与标定插值一致",
                  torch.allclose(pos[rows][:, leg_ids], tables[key][mode, index, 0][:, leg_ids], atol=1e-6))
    difference = (heights[:, 0] - heights[:, 1]).abs()
    expected_pitch = LOW_SPINE_PITCH * (difference / (BASE_HEIGHT - LOW_POSE_CALIBRATION[0][0])).clamp(max=1.0)
    check("四种模式均按实际前后高度差计算脊柱俯仰",
          torch.allclose(pos[:, _ACT_SPN_H_PITCH_ID], expected_pitch, atol=1e-7, rtol=0))
    equal_heights = heights[:, 0] == heights[:, 1]
    check("所有同高组合的脊柱俯仰均为零",
          bool((pos[equal_heights, _ACT_SPN_H_PITCH_ID] == 0).all()))
    for front, hind in ((0.040, 0.050), (0.050, 0.040), (0.045, 0.055), (0.055, 0.045)):
        row = torch.isclose(heights[:, 0], heights.new_tensor(front)) & torch.isclose(heights[:, 1], heights.new_tensor(hind))
        check(f"双高{front * 1000:g}／{hind * 1000:g}毫米也使用高度差俯仰且相位速度为零",
              torch.allclose(pos[row, _ACT_SPN_H_PITCH_ID], pos.new_full((int(row.sum()),), -0.65 * 0.010 / 0.035), atol=2e-7)
              and bool((vel[row, _ACT_SPN_H_PITCH_ID] == 0).all()))
    for height_value in (0.020, 0.025, 0.030, 0.035):
        expected_value = LOW_SPINE_PITCH * (BASE_HEIGHT - height_value) / (BASE_HEIGHT - LOW_POSE_CALIBRATION[0][0])
        mask = torch.isclose(heights.min(dim=1).values, heights.new_tensor(height_value)) & (heights.max(dim=1).values == BASE_HEIGHT)
        check(f"{height_value * 1000:g}/55毫米前后两种模式脊柱端点正确",
              torch.allclose(pos[mask, _ACT_SPN_H_PITCH_ID], pos.new_full((int(mask.sum()),), expected_value), atol=1e-7, rtol=0))
    with patch("mjlab.tasks.SQuRo_Hole.mdp.reference.inverse_kinematics", side_effect=AssertionError("在线参考不应计算IK")):
        direct, _ = get_reference_joint_state(probe)
    check("在线参考无需调用逆运动学", torch.equal(pos, direct))
    probe.episode_length_buf += 37
    shifted, _ = get_reference_joint_state(probe)
    for column, leg_ids in ((0, _ACT_LEG_F_IDS), (1, _ACT_LEG_H_IDS)):
        low = heights[:, column] < THRESHOLD_HEIGHT
        check("固定足端参考不受步态相位影响", torch.equal(original[low][:, leg_ids], shifted[low][:, leg_ids]))
    permutation = torch.arange(len(heights) - 1, -1, -1, device=env.device)
    command[:] = command[permutation].clone()
    probe.episode_length_buf[:] = probe.episode_length_buf[permutation].clone()
    reordered, _ = get_reference_joint_state(probe)
    check("批量参考不受其他环境或排列影响", torch.equal(reordered, shifted[permutation]))
    check("高低阈值与65毫米减25毫米目标衔接", THRESHOLD_HEIGHT == 0.040 and BASE_HEIGHT == 0.055)


# 验收真实碰撞几何、逐环境复位和动态奖励阈值。
def verify_real_env() -> None:
    env = make_env()
    obstacle = env.scene.entities["obstacle"]
    check("训练环境只保留单板且碰撞已编译", list(name for name in env.scene.entities if name.startswith("obstacle")) == ["obstacle"] and obstacle_collision_enabled(env))
    targets = torch.tensor([HEIGHT_MIN, 0.052, 0.070, 0.090], device=env.device)
    write_obstacle_mocap(obstacle, targets, torch.arange(4, device=env.device))
    written = read_obstacle_bottom_z(obstacle).clone()
    check("板体 mocap 逐环境写入与即时读取", torch.allclose(written, targets))
    check("独立碰撞几何的板底随四个采样高度移动", torch.allclose(
        read_geom_bottom_z(env), targets, atol=1e-6,
    ))
    keep = written[[1, 3]].clone()
    write_obstacle_mocap(obstacle, 0.051, torch.tensor([0, 2], device=env.device))
    check("部分写入不污染其他环境", torch.equal(read_obstacle_bottom_z(obstacle)[[1, 3]], keep))
    check("部分写入只移动指定环境的实际碰撞板", torch.allclose(
        read_geom_bottom_z(env), torch.tensor([0.051, 0.052, 0.051, 0.090]), atol=1e-6,
    ))
    write_obstacle_mocap(obstacle, torch.tensor([0.054, 0.056]), slice(1, 4, 2))
    check("切片写入正确移动实际碰撞板", torch.allclose(
        read_geom_bottom_z(env), torch.tensor([0.051, 0.054, 0.051, 0.056]), atol=1e-6,
    ))
    keep = read_geom_bottom_z(env)[[1, 3]].clone()
    with torch.inference_mode():
        monitor_restricted_passage(env)
    monitor = env._hole_passage_monitor
    env.reset(env_ids=torch.tensor([0, 2], device=env.device))
    after_reset = read_obstacle_bottom_z(obstacle).clone()
    check("部分 reset 重采样且保留其他环境板高", torch.equal(after_reset[[1, 3]], keep) and torch.all(after_reset[[0, 2]] >= HEIGHT_INITIAL_MIN))
    actual = read_geom_bottom_z(env)
    check("部分 reset 后碰撞几何与采样高度一致", torch.allclose(actual, after_reset, atol=1e-6)
          and torch.equal(actual[[1, 3]], keep))
    check("inference step 后可从外部执行部分 reset", not torch.is_inference(monitor.approached))
    verify_base_height_commands(env)
    verify_continuous_reference(env)
    verify_dynamic_sources(env)
    verify_passage_monitor(env)
    verify_partial_episode_filter(env)
    command = env.command_manager.get_term("hole_cmd")
    command.height_curriculum_state.z_min = HEIGHT_MIN
    command.height_curriculum_state.z_max = HEIGHT_MIN
    env.reset()
    check("46毫米复位采样、实际碰撞几何和命令先验一致",
          torch.allclose(read_obstacle_bottom_z(obstacle), torch.full_like(targets, HEIGHT_MIN))
          and torch.allclose(read_geom_bottom_z(env), torch.full_like(targets, HEIGHT_MIN), atol=1e-6)
          and torch.allclose(command.command_obstacle_height, torch.full_like(targets, HEIGHT_MIN)))
    env.close()


# 运行全部课程验收。
def main() -> None:
    print("单板受限空间高度课程验收")
    verify_shared_rule()
    verify_curriculum_state()
    verify_fine_height_steps()
    verify_real_env()
    if FAILURES:
        raise SystemExit(f"失败项: {FAILURES}")
    print("验收通过")


if __name__ == "__main__":
    main()
