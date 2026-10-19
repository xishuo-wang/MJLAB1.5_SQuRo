from __future__ import annotations

import math
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
    BASE_SPEED,
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
    HEIGHT_MIN_SUCCESS_RATE,
    HEIGHT_STEP,
    HEIGHT_FINE_STEP,
    FINAL_ACCEPT_RATE,
    NORMAL_BIN_TARGET_RATE,
    RETREAT_FAILURE_LIMIT,
    RETREAT_REQUIRED_PASSES,
    VIRTUAL_CLEARANCE_MARGIN,
)
from mjlab.tasks.SQuRo_Hole.mdp.curriculums import (
    STAGE3_STEP,
    STAGE3_ITER,
    STAGE2_STEP,
    HEIGHT_RULE_STAGE,
    STAGE2_ITER,
    STEPS_PER_ITER,
    HEIGHT_EVALUATION_INTERVAL,
    HEIGHT_EVALUATION_FINAL_INTERVAL,
    HEIGHT_EVALUATION_PRETRAIN_ITERATIONS,
    HEIGHT_BIN_COUNT,
    HEIGHT_BIN_EDGES,
    HeightCurriculumState,
    get_current_stage,
    obstacle_rule_bounds,
    compute_height_commands,
    sample_bin_heights,
    sample_range_bin_heights,
    height_bin_indices,
)
from mjlab.tasks.SQuRo_Hole.mdp.entity import (
    configure_obstacle_entity,
    obstacle_collision_enabled,
    read_obstacle_bottom_z,
    write_obstacle_mocap,
)
from mjlab.tasks.SQuRo_Hole.mdp.indices import (
    _ACT_NUM,
    _ACT_SPN_IDS,
    _ACT_LEG_F_IDS,
    _ACT_LEG_H_IDS,
    _ACT_SPN_H_PITCH_ID,
    _MODEL_INDICES,
    resolve_model_indices,
)
from mjlab.tasks.SQuRo_Hole.mdp.command import HoleCommand, get_height_mode, compute_cmd_vel_x
from mjlab.tasks.SQuRo_Hole.mdp.rewards import (
    compute_contact_penalty,
    compute_leg_motion_penalty,
)
from mjlab.tasks.SQuRo_Hole.mdp.reference import get_reference_joint_state, initialize_tables
from mjlab.tasks.SQuRo_Hole.mdp.terminations import (
    check_fallen,
    collect_passage_state,
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
    command.high_obstacle_equal_height = True
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
                front, hind = compute_height_commands(positions, center, half_length, height)
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
            front, hind = compute_height_commands(query, center, half_length, height)
            check(f"板长 {half_length * 2000:.0f} mm、板高 {height * 1000:g} mm 全区间没有双低",
                  not bool(((front < THRESHOLD_HEIGHT) & (hind < THRESHOLD_HEIGHT)).any()))
            preview = sample_rule_schedule(query.numpy() - 0.1, obstacle, height, start_x=0.1)
            current = sample_base_commands(query.numpy() - 0.1, obstacle, height, start_x=0.1)
            check("非零起点下生产、曲线及游标读数一致且速度为正",
                  torch.allclose(torch.as_tensor(preview["height_f"]), front)
                  and torch.allclose(torch.as_tensor(current["height_h"]), hind)
                  and bool((torch.as_tensor(current["velocity_x"]) > 0).all()))


# 验证八档速度的连续减速、前后对称与停步边界。
def verify_velocity_commands() -> None:
    for device in ("cpu", "cuda:0") if torch.cuda.is_available() else ("cpu",):
        heights = torch.tensor([0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055], device=device)
        factors = heights.new_tensor([0.0, 7 / 33, 14 / 33, 7 / 11, 8 / 11, 9 / 11, 10 / 11, 1.0])
        ratios = heights.new_tensor([0.75, 0.8125, 0.875, 0.9375, 1.0, 1.0, 1.0, 1.0])
        front, hind = torch.meshgrid(heights, heights, indexing="ij")
        velocity = compute_cmd_vel_x(front, hind)
        expected = BASE_SPEED * (factors[:, None] + factors[None, :]) / 2 * torch.minimum(ratios[:, None], ratios[None, :])
        check(f"{device}全部64种高度组合的速度符合低姿态连续减速", torch.allclose(velocity, expected, atol=3e-8, rtol=0))
        check("速度前后交换对称，只有20／20毫米停步", torch.equal(velocity, velocity.T) and int((velocity == 0).sum()) == 1)
        check("任一端高度增加时速度不下降", bool((velocity.diff(dim=0) >= 0).all() and (velocity.diff(dim=1) >= 0).all()))
        check("47毫米两段姿态均为0.075米每秒", torch.allclose(velocity[[0, -1], [-1, 0]], heights.new_full((2,), 0.075), atol=1e-8, rtol=0))
        old_high = BASE_SPEED * (torch.stack((front[4:, 4:], hind[4:, 4:]), dim=-1) / BASE_HEIGHT).mean(dim=-1)
        check("所有双高组合速度保持原值", torch.equal(velocity[4:, 4:], old_high))
        points = heights.new_tensor([0.020, 0.035, 0.040])
        left = compute_cmd_vel_x(points - 1e-6, heights.new_tensor(0.055))
        right = compute_cmd_vel_x(points + 1e-6, heights.new_tensor(0.055))
        check("20、35及40毫米分界连续", bool(((right - left).abs() < 2e-5).all()))


# 用错序模型关节验收四种模式、低侧选择、静止目标及逐环境隔离。
def verify_low_leg_hold() -> None:
    height_mm = [20, 25, 30, 35, 40, 45, 50, 55]
    strengths = [1.0, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0]
    joint_ids = tuple(range(_ACT_NUM, 0, -1))
    front_ids = [joint_ids[i] for i in _ACT_LEG_F_IDS]
    hind_ids = [joint_ids[i] for i in _ACT_LEG_H_IDS]
    expected = [
        -0.2 * strengths[i] * 7.5 if f < 40 <= h else
        -0.2 * strengths[j] * 30.0 if h < 40 <= f else 0.0
        for i, f in enumerate(height_mm) for j, h in enumerate(height_mm)
    ]
    for device in ("cpu", "cuda:0") if torch.cuda.is_available() else ("cpu",):
        heights = torch.tensor(height_mm, device=device) / 1000
        front, hind = torch.meshgrid(heights, heights, indexing="ij")
        command = torch.zeros(front.numel(), COMMAND_DIM, device=device)
        command[:, CMD_HEIGHT_F_IDS] = front.flatten()
        command[:, CMD_HEIGHT_H_IDS] = hind.flatten()
        velocity = torch.full((front.numel(), _ACT_NUM + 1), 99.0, device=device)
        velocity[:, front_ids] = velocity.new_tensor([1.0, -2.0, 3.0, -4.0])
        velocity[:, hind_ids] = velocity.new_tensor([2.0, -4.0, 6.0, -8.0])
        asset = SimpleNamespace(data=SimpleNamespace(joint_vel=velocity))
        probe = SimpleNamespace(scene={"robot": asset}, common_step_counter=STAGE3_STEP,
            command_manager=SimpleNamespace(_terms={"hole_cmd": SimpleNamespace(command=command)}))
        with patch.object(_MODEL_INDICES, "joint_ids", joint_ids), \
             patch.object(_MODEL_INDICES, "f_body_id", 0), \
             patch.object(_MODEL_INDICES, "linkage_joint_ids", (_ACT_NUM + 1,)):
            reward = compute_leg_motion_penalty(probe)
            check(f"{device}静止惩罚覆盖64种组合且仅选中低侧四关节",
                  torch.allclose(reward, velocity.new_tensor(expected), atol=1e-6, rtol=0))
            single_front_low = (front.flatten() < THRESHOLD_HEIGHT) & (hind.flatten() >= THRESHOLD_HEIGHT)
            velocity[:, hind_ids] *= 100
            changed = compute_leg_motion_penalty(probe)
            check("前低时运动侧后腿速度不影响静止惩罚", torch.equal(changed[single_front_low], reward[single_front_low]))
            velocity[:, hind_ids] /= 100
            permutation = torch.arange(front.numel() - 1, -1, -1, device=device)
            command[:] = command[permutation].clone()
            reordered = compute_leg_motion_penalty(probe)
            check("其他环境及批量排列不改变单环境静止惩罚", torch.allclose(reordered, reward[permutation], atol=1e-6, rtol=0))
            velocity.zero_()
            check("低侧四关节静止时惩罚为零", bool((compute_leg_motion_penalty(probe) == 0).all()))


# 虚拟碰撞探针场景: 同时提供按名字取机器人与实体表。
class _StubScene(dict):
    def __init__(self, robot, entities):
        super().__init__(robot=robot)
        self.entities = entities


# 验证虚拟碰撞惩罚的 X 区间、板底余量、数值口径与课程开关。
def verify_virtual_contact() -> None:
    num_envs = 8
    board_position = (0.2, 0.0, 0.047)
    board_size = (0.015, 0.1, 0.005)
    bottom = torch.full((num_envs,), board_position[2])
    threshold = board_position[2] - VIRTUAL_CLEARANCE_MARGIN
    x_min = board_position[0] - board_size[0]
    x_max = board_position[0] + board_size[0]
    body_x = torch.tensor([-0.10, x_min, x_max, 0.20, 0.20, 0.20, 0.20, 0.20])
    front_z = torch.tensor([0.100, threshold + 0.001, threshold + 0.001, threshold + 0.002,
                            threshold, threshold, threshold - 0.001, threshold + 0.002])
    hind_z = torch.tensor([0.100, threshold, threshold, threshold,
                           threshold + 0.001, threshold, threshold, threshold])

    # 构造只含奖励输入的最小探针环境
    def build_probe(front, hind, positions, step_counter=STAGE3_STEP):
        site_pos = torch.zeros(num_envs, 18, 3)
        site_pos[:, :9, 2] = front.unsqueeze(1)
        site_pos[:, 9:, 2] = hind.unsqueeze(1)
        link_pos = torch.zeros(num_envs, 2, 3)
        link_pos[:, 0, 0] = positions
        link_pos[:, 1, 0] = positions
        robot = SimpleNamespace(data=SimpleNamespace(body_link_pos_w=link_pos, site_pos_w=site_pos))
        obstacle = SimpleNamespace(cfg=SimpleNamespace(position=board_position, size=board_size))
        scene = _StubScene(robot, {"obstacle": obstacle})
        return SimpleNamespace(scene=scene, num_envs=num_envs, device="cpu",
                               common_step_counter=step_counter, extras={"log": {}})

    sigma, weight = 2000.0, 5.0
    inside = (body_x >= x_min) & (body_x <= x_max)
    e_front = torch.where(inside, (front_z - threshold).clamp_min(0.0), torch.zeros(num_envs))
    e_hind = torch.where(inside, (hind_z - threshold).clamp_min(0.0), torch.zeros(num_envs))
    expected = -weight * (1.0 - torch.exp(-sigma * e_front.square())
                          + 1.0 - torch.exp(-sigma * e_hind.square())) / 2
    with patch.object(_MODEL_INDICES, "f_body_id", 0), \
         patch.object(_MODEL_INDICES, "h_body_id", 1), \
         patch.object(_MODEL_INDICES, "f_body_site_ids", tuple(range(9))), \
         patch.object(_MODEL_INDICES, "h_body_site_ids", tuple(range(9, 18))), \
         patch("mjlab.tasks.SQuRo_Hole.mdp.rewards.resolve_model_indices", lambda asset: None), \
         patch("mjlab.tasks.SQuRo_Hole.mdp.rewards.read_obstacle_bottom_z", return_value=bottom):
        penalty = compute_contact_penalty(build_probe(front_z, hind_z, body_x))
        check("虚拟碰撞惩罚按实际板底与 X 区间数值正确", torch.allclose(penalty, expected, atol=1e-6, rtol=0))
        check("板外环境不受虚拟碰撞约束", float(penalty[0]) == 0.0)
        check("X 左右边界均计入板体区间", bool((penalty[[1, 2]] < 0).all()))
        check("恰好板底减余量时惩罚为零", float(penalty[5]) == 0.0 and float(penalty[6]) == 0.0)
        check("虚拟碰撞惩罚非正且有界", bool((penalty <= 0).all() and (penalty >= -weight).all()))
        stage3 = build_probe(front_z, hind_z, body_x, step_counter=STAGE2_STEP)
        check("第三阶段虚拟碰撞权重为零", bool((compute_contact_penalty(stage3) == 0).all()))
        swapped = build_probe(hind_z, front_z, body_x)
        check("前后段交换不改变虚拟碰撞惩罚", torch.allclose(compute_contact_penalty(swapped), penalty, atol=1e-7, rtol=0))
        permutation = torch.arange(num_envs - 1, -1, -1)
        permuted = build_probe(front_z[permutation], hind_z[permutation], body_x[permutation])
        check("环境排列不改变单环境虚拟碰撞惩罚",
              torch.allclose(compute_contact_penalty(permuted), penalty[permutation], atol=1e-7, rtol=0))


# 验证高板两种策略等概率采样，且不会改变低板规则。
def verify_high_obstacle_modes() -> None:
    positions = torch.tensor([0.1, 0.1, 0.25, 0.25])
    choices = torch.tensor([True, False, True, False])
    front, hind = compute_height_commands(positions, 0.2, 0.015, 0.074, choices)
    check("74毫米高板逐环境选择等高或分段", torch.allclose(front, front.new_tensor([0.045, 0.045, 0.045, 0.055]))
          and torch.allclose(hind, hind.new_tensor([0.045, 0.055, 0.045, 0.045])))
    for height, target in ((0.064999, 0.035), (0.065, 0.040), (0.080, 0.055)):
        front, hind = compute_height_commands(positions, 0.2, 0.015, height, False)
        check(f"{height * 1000:g}毫米分段正确覆盖65毫米边界与55毫米封顶",
              torch.allclose(front, front.new_tensor([target, target, 0.055, 0.055]))
              and torch.allclose(hind, hind.new_tensor([0.055, 0.055, target, target])))
    low_equal = compute_height_commands(positions, 0.2, 0.015, 0.047, True)
    low_split = compute_height_commands(positions, 0.2, 0.015, 0.047, False)
    check("47毫米低板忽略高板等高选项", all(torch.equal(a, b) for a, b in zip(low_equal, low_split)))
    fake = SimpleNamespace(cfg=SimpleNamespace(high_obstacle_equal_height=None), device="cpu",
                           obstacle_equal_height=torch.ones(4096, dtype=torch.bool))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(83)
        HoleCommand._sample_high_obstacle_mode(fake, torch.arange(4096))
    fraction = float(fake.obstacle_equal_height.float().mean())
    check("默认4096回合的等高采样比例接近50%", 0.47 < fraction < 0.53)


# 验证真实环境的固定策略、部分复位和回合内策略保持。
def verify_high_obstacle_env(env: ManagerBasedRlEnv) -> None:
    command = env.command_manager.get_term("hole_cmd")
    command.cfg.high_obstacle_equal_height = None
    with patch("mjlab.tasks.SQuRo_Hole.mdp.command.torch.rand", return_value=torch.tensor([0.1, 0.9, 0.1, 0.9])):
        command._sample_high_obstacle_mode(torch.arange(env.num_envs))
    robot = env.scene["robot"]
    pose = robot.data.root_link_pose_w.clone()
    pose[:, 0] = pose.new_tensor([0.1, 0.1, 0.25, 0.25])
    robot.write_root_link_pose_to_sim(pose)
    write_obstacle_mocap(env.scene["obstacle"], 0.074)
    read_geom_bottom_z(env)
    command._update_command_by_height_rule()
    values = command.command
    check("实际高板对应等高45／45或分段45／55、55／45毫米命令",
          torch.allclose(values[:, CMD_HEIGHT_F_IDS], values.new_tensor([0.045, 0.045, 0.045, 0.055]))
          and torch.allclose(values[:, CMD_HEIGHT_H_IDS], values.new_tensor([0.045, 0.055, 0.045, 0.045])))
    keep = command.obstacle_equal_height.clone()
    command._resample_command(torch.arange(env.num_envs))
    check("回合中周期重采样不会改变高板姿态策略", torch.equal(command.obstacle_equal_height, keep))
    env.reset(env_ids=torch.tensor([1, 3]))
    check("部分复位不会改变其他环境的策略", torch.equal(command.obstacle_equal_height[[0, 2]], keep[[0, 2]]))
    for fixed in (True, False):
        command.cfg.high_obstacle_equal_height = fixed
        env.reset()
        check(f"显式高板策略{fixed}覆盖所有复位环境", bool((command.obstacle_equal_height == fixed).all()))
    command.cfg.high_obstacle_equal_height = True
    env.reset()


# 构造一批考核结果, 最低档与较易档混合
def _evaluation_results(z_min: float, min_count: int, min_success: int, other_count: int = 0) -> list[dict]:
    results = [
        {"env_id": i, "height": z_min, "success": i < min_success, "valid": True,
         "failure_code": 0 if i < min_success else 3}
        for i in range(min_count)
    ]
    results.extend(
        {"env_id": min_count + i, "height": z_min + 0.010, "success": True, "valid": True,
         "failure_code": 0}
        for i in range(other_count)
    )
    return results


# 构造最终档全挡位考核结果: 每挡 rounds 回合, successes_per_bin 指定各挡成功数
def _final_results(rounds_per_bin: int, successes_per_bin: list[int]) -> list[dict]:
    results = []
    env_id = 0
    for index in range(HEIGHT_BIN_COUNT):
        height = (HEIGHT_BIN_EDGES[index] + HEIGHT_BIN_EDGES[index + 1]) / 2
        for item in range(rounds_per_bin):
            success = item < successes_per_bin[index]
            results.append({"env_id": env_id, "height": height, "success": success,
                            "valid": True, "failure_code": 0 if success else 5})
            env_id += 1
    return results


# 验证独立考核的调度、最低档样本门槛、最终档均匀采样、验收与退档保护。
def verify_evaluation_curriculum() -> None:
    state = HeightCurriculumState()
    first_iteration = STAGE3_ITER + HEIGHT_EVALUATION_PRETRAIN_ITERATIONS
    check("考核在预训练轮数之前不触发", not state.should_evaluate(first_iteration - HEIGHT_EVALUATION_INTERVAL))
    check("考核按 100 轮间隔触发", state.should_evaluate(first_iteration)
          and state.should_evaluate(first_iteration + HEIGHT_EVALUATION_INTERVAL))
    check("考核间隔之外不触发", not state.should_evaluate(first_iteration + 1))

    # 同一考核内一批失败、一批成功: 整次考核不达标; 反向顺序同样不达标
    fail_then_pass = [state.measure_batch(_evaluation_results(state.z_min, 256, 100)),
                      state.measure_batch(_evaluation_results(state.z_min, 256, 256))]
    first = state.record_evaluation(fail_then_pass, first_iteration)
    pass_then_fail = [state.measure_batch(_evaluation_results(state.z_min, 256, 256)),
                      state.measure_batch(_evaluation_results(state.z_min, 256, 100))]
    second = state.record_evaluation(pass_then_fail, first_iteration + HEIGHT_EVALUATION_INTERVAL)
    check("单批失败的成功组合跨考核也不降档",
          not first["advanced"] and not second["advanced"]
          and state.z_min == HEIGHT_INITIAL_MIN and state.consecutive_success_batches == 0)

    shortage = state.measure_batch(_evaluation_results(state.z_min, 100, 100, other_count=300))
    check("最低档样本不足 256 时该批不达标", not shortage["passed"])

    boundary = HeightCurriculumState()
    check("最低档通过率 59.8% 不达标",
          not boundary.measure_batch(_evaluation_results(0.070, 256, 153))["passed"])
    check("最低档通过率 60.2% 达标",
          boundary.measure_batch(_evaluation_results(0.070, 256, 154))["passed"])

    passing = [state.measure_batch(_evaluation_results(state.z_min, 256, 200)),
               state.measure_batch(_evaluation_results(state.z_min, 256, 200))]
    advanced = state.record_evaluation(passing, first_iteration + 2 * HEIGHT_EVALUATION_INTERVAL)
    metrics = state.metrics()
    check("同一考核两批达标后降档 2 毫米",
          advanced["advanced"] and state.z_min == HEIGHT_INITIAL_MIN - HEIGHT_STEP)
    check("降档后日志沿用最近一次考核成功率",
          math.isclose(metrics["success_rate"], 200 / 256, rel_tol=0, abs_tol=1e-9)
          and math.isclose(metrics["failure_rate"], 1 - 200 / 256, rel_tol=0, abs_tol=1e-9))

    # 47mm 档通过一次考核后进入最终模式
    final_state = HeightCurriculumState()
    final_state.z_min = HEIGHT_MIN
    final_state.level_start_iteration = first_iteration - HEIGHT_MIN_DWELL_ITERATIONS
    min_passing = [final_state.measure_batch(_evaluation_results(HEIGHT_MIN, 256, 256)),
                   final_state.measure_batch(_evaluation_results(HEIGHT_MIN, 256, 256))]
    final_result = final_state.record_evaluation(min_passing, first_iteration)
    check("47毫米档通过一次考核后进入最终模式",
          final_result["entered_final_mode"] and final_state.final_level_passed
          and final_state.z_min == HEIGHT_MIN)
    check("最终档验收门槛为 80%, 退档后需连续两轮达标恢复",
          FINAL_ACCEPT_RATE == 0.80 and RETREAT_REQUIRED_PASSES == 2
          and RETREAT_FAILURE_LIMIT == 5)
    check("最终模式初始化均匀挡位权重",
          all(abs(value - 1.0) < 1e-9 for value in final_state.bin_weights))
    check("最终档仍继续考核且间隔为 500 轮",
          final_state.should_evaluate(first_iteration + HEIGHT_EVALUATION_FINAL_INTERVAL)
          and not final_state.should_evaluate(first_iteration + HEIGHT_EVALUATION_INTERVAL))

    # 全挡位 100% 通过时完成验收
    accepted_batches = [final_state.measure_final_batch(_final_results(64, [64] * HEIGHT_BIN_COUNT)),
                        final_state.measure_final_batch(_final_results(64, [64] * HEIGHT_BIN_COUNT))]
    accepted = final_state.record_evaluation(
        accepted_batches, first_iteration + HEIGHT_EVALUATION_FINAL_INTERVAL)
    check("最终档全挡位 100% 通过即完成验收", final_state.final_accepted and accepted["mode"] == "final")
    check("最终档记录各挡位通过率",
          len(accepted["bin_rate"]) == HEIGHT_BIN_COUNT and all(rate == 1.0 for rate in accepted["bin_rate"]))

    # 80% 门槛: 50% 不验收, 81.25% 验收; 未达标挡位提高训练采样权重
    low_state = HeightCurriculumState()
    low_state.z_min = HEIGHT_MIN
    low_state._enter_final_mode()
    low_batches = [low_state.measure_final_batch(_final_results(64, [32] * HEIGHT_BIN_COUNT)),
                   low_state.measure_final_batch(_final_results(64, [32] * HEIGHT_BIN_COUNT))]
    low_result = low_state.record_evaluation(low_batches, first_iteration + HEIGHT_EVALUATION_FINAL_INTERVAL)
    check("最终档整体通过率 50% 不验收", not low_state.final_accepted and not low_result["passed"])
    check("最终档按挡位缺口提高训练采样权重",
          all(weight > 1.0 for weight in low_state.bin_weights))
    near_state = HeightCurriculumState()
    near_state.z_min = HEIGHT_MIN
    near_state._enter_final_mode()
    near_batches = [near_state.measure_final_batch(_final_results(64, [52] * HEIGHT_BIN_COUNT)),
                    near_state.measure_final_batch(_final_results(64, [52] * HEIGHT_BIN_COUNT))]
    near_state.record_evaluation(near_batches, first_iteration + HEIGHT_EVALUATION_FINAL_INTERVAL)
    check("最终档整体通过率 81.25% 完成验收", near_state.final_accepted
          and abs(near_state.metrics()["success_rate"] - 52 / 64) < 1e-9)

    # 退档保护: 连续 5 次失败退一档, 恢复需要连续两轮达标
    retreat = HeightCurriculumState()
    retreat.z_min = HEIGHT_MIN
    retreat_fail = [retreat.measure_batch(_evaluation_results(HEIGHT_MIN, 256, 100)),
                    retreat.measure_batch(_evaluation_results(HEIGHT_MIN, 256, 100))]
    outcome = None
    for index in range(RETREAT_FAILURE_LIMIT):
        outcome = retreat.record_evaluation(retreat_fail, first_iteration + index * HEIGHT_EVALUATION_INTERVAL)
    check(f"连续 {RETREAT_FAILURE_LIMIT} 次失败后退一档",
          outcome["retreated"] and retreat.z_min == HEIGHT_MIN + HEIGHT_FINE_STEP
          and retreat.retreats == 1 and retreat.retreat_pending_advance)
    level = retreat.z_min
    level_passing = [retreat.measure_batch(_evaluation_results(level, 256, 256)),
                     retreat.measure_batch(_evaluation_results(level, 256, 256))]
    first_pass = retreat.record_evaluation(level_passing, first_iteration + 600)
    check("退档后首次达标不立即推进", not first_pass["advanced"])
    second_pass = retreat.record_evaluation(level_passing, first_iteration + 700)
    check("退档后连续两轮达标恢复上一档", second_pass["advanced"] and retreat.z_min == HEIGHT_MIN)
    min_passing = [retreat.measure_batch(_evaluation_results(HEIGHT_MIN, 256, 256)),
                   retreat.measure_batch(_evaluation_results(HEIGHT_MIN, 256, 256))]
    re_enter = retreat.record_evaluation(min_passing, first_iteration + 800)
    check("恢复 47mm 后再次通过进入最终模式",
          re_enter["entered_final_mode"] and retreat.final_level_passed)

    # float32 分桶一致性: 60mm 命令对应 35mm 挡, 统计必须落在同一挡
    aligned_state = HeightCurriculumState()
    aligned_state.z_min = 0.060
    aligned_batch = aligned_state.measure_batch(
        [{"env_id": 0, "height": 0.060, "success": True, "valid": True, "failure_code": 0}])
    check("分挡位统计与命令按 float32 同精度分桶",
          int(height_bin_indices(torch.tensor([0.060]))[0]) == 3
          and int(height_bin_indices(torch.tensor([0.050]))[0]) == 1
          and aligned_batch["bin_count"][3] == 1 and aligned_batch["bin_count"][2] == 0)

    # 50mm 连续失败: 返回上一档 52mm, 恢复推进回到 50mm 而不是跳到 49mm
    fifty = HeightCurriculumState()
    fifty.z_min = 0.050
    fifty_fail = [fifty.measure_batch(_evaluation_results(0.050, 256, 100)),
                  fifty.measure_batch(_evaluation_results(0.050, 256, 100))]
    fifty_outcome = None
    for index in range(RETREAT_FAILURE_LIMIT):
        fifty_outcome = fifty.record_evaluation(
            fifty_fail, first_iteration + index * HEIGHT_EVALUATION_INTERVAL)
    check("50毫米连续失败退回上一档52毫米",
          fifty_outcome["retreated"] and fifty.z_min == 0.052)
    fifty_passing = [fifty.measure_batch(_evaluation_results(fifty.z_min, 256, 256)),
                     fifty.measure_batch(_evaluation_results(fifty.z_min, 256, 256))]
    fifty_first = fifty.record_evaluation(fifty_passing, first_iteration + 600)
    fifty_second = fifty.record_evaluation(fifty_passing, first_iteration + 700)
    check("退档后从52毫米恢复推进到50毫米",
          not fifty_first["advanced"] and fifty_second["advanced"] and fifty.z_min == 0.050)

    # 最终档全挡位均匀采样与权重
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        uniform = sample_bin_heights(8192, [1.0] * HEIGHT_BIN_COUNT, "cpu")
        torch.manual_seed(2027)
        biased = sample_bin_heights(8192, [4.0] + [1.0] * (HEIGHT_BIN_COUNT - 1), "cpu")
    edges = torch.tensor(HEIGHT_BIN_EDGES, dtype=torch.float64)
    uniform_index = torch.bucketize(uniform.to(torch.float64), edges[1:-1], right=True)
    biased_index = torch.bucketize(biased.to(torch.float64), edges[1:-1], right=True)
    uniform_fraction = torch.bincount(uniform_index, minlength=HEIGHT_BIN_COUNT).float() / len(uniform)
    biased_fraction = torch.bincount(biased_index, minlength=HEIGHT_BIN_COUNT).float() / len(biased)
    check("最终档均匀采样覆盖全部挡位",
          bool((uniform_fraction > 0.08).all()) and bool((uniform_fraction < 0.17).all()))
    check("挡位权重提高对应挡位采样比例",
          float(biased_fraction[0]) > float(uniform_fraction[0]) * 2.0)
    check("最终档采样范围正确",
          float(uniform.min()) >= HEIGHT_MIN and float(uniform.max()) <= HEIGHT_MAX)

    # 课程状态往返（当前版本检查点字段）
    state = HeightCurriculumState()
    restored = HeightCurriculumState()
    restored.load_state_dict(state.state_dict())
    check("课程 checkpoint 状态往返", restored.state_dict() == state.state_dict())


# 验证最终档命令采样: 全挡位随机与显式固定板高
def verify_final_mode_command_sampling() -> None:
    env_ids = torch.arange(64)
    state = HeightCurriculumState()
    state.final_level_passed = True
    state.z_min = HEIGHT_MIN
    class _CommandProbe(HoleCommand):
        @property
        def device(self):
            return "cpu"

    probe = _CommandProbe.__new__(_CommandProbe)
    probe.obstacle = SimpleNamespace()
    probe.height_curriculum_state = state
    probe.obstacle_sampled_bottom_z = torch.zeros(64)
    probe.obstacle_height_sampled = torch.zeros(64, dtype=torch.bool)
    probe.cfg = SimpleNamespace(high_obstacle_equal_height=True)
    probe.obstacle_equal_height = torch.zeros(64, dtype=torch.bool)
    probe.obstacle_height_command = torch.zeros(64)
    probe.adaptive_sampling = True
    probe._update_obstacle_height_command = lambda: None
    with patch("mjlab.tasks.SQuRo_Hole.mdp.command.write_obstacle_mocap",
               lambda *args, **kwargs: None), torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        HoleCommand._sample_obstacle_height(probe, env_ids)
    heights = probe.obstacle_sampled_bottom_z
    check("最终档训练命令按挡位均匀采样板底",
          float(heights.min()) >= HEIGHT_MIN and float(heights.max()) <= HEIGHT_MAX
          and torch.unique(heights).numel() > 8)
    state.z_min = 0.05
    state.z_max = 0.05
    with patch("mjlab.tasks.SQuRo_Hole.mdp.command.write_obstacle_mocap",
               lambda *args, **kwargs: None), torch.random.fork_rng(devices=[]):
        HoleCommand._sample_obstacle_height(probe, env_ids)
    check("显式固定板高时仍按固定值采样",
          bool(torch.allclose(probe.obstacle_sampled_bottom_z, torch.full((64,), 0.05))))


# 验证普通模式 70% 部分按挡位加权, 评估环境保持固定 30% + 均匀采样
def verify_normal_adaptive_sampling() -> None:
    class _CommandProbe(HoleCommand):
        @property
        def device(self):
            return "cpu"

    def sample_heights(weights, adaptive: bool, count: int = 20000) -> torch.Tensor:
        state = HeightCurriculumState()
        state.z_min = 0.060
        state.z_max = HEIGHT_MAX
        state.bin_weights = list(weights)
        probe = _CommandProbe.__new__(_CommandProbe)
        probe.obstacle = SimpleNamespace()
        probe.height_curriculum_state = state
        probe.obstacle_sampled_bottom_z = torch.zeros(count)
        probe.obstacle_height_sampled = torch.zeros(count, dtype=torch.bool)
        probe.cfg = SimpleNamespace(high_obstacle_equal_height=True)
        probe.obstacle_equal_height = torch.zeros(count, dtype=torch.bool)
        probe.obstacle_height_command = torch.zeros(count)
        probe.adaptive_sampling = adaptive
        probe._update_obstacle_height_command = lambda: None
        with patch("mjlab.tasks.SQuRo_Hole.mdp.command.write_obstacle_mocap",
                   lambda *args, **kwargs: None), torch.random.fork_rng(devices=[]):
            torch.manual_seed(41)
            HoleCommand._sample_obstacle_height(probe, torch.arange(count))
        return probe.obstacle_sampled_bottom_z.clone()

    biased = [1.0] * HEIGHT_BIN_COUNT
    biased[3] = 6.0
    weighted = sample_heights(biased, True)
    uniform = sample_heights([1.0] * HEIGHT_BIN_COUNT, True)
    minimum_fraction = float((weighted == 0.060).float().mean())
    check("普通模式最低档固定采样保持 30%", 0.28 < minimum_fraction < 0.32)
    check("普通模式加权采样不越界",
          float(weighted.min()) >= 0.060 - 1e-6 and float(weighted.max()) <= HEIGHT_MAX + 1e-6)

    edges = torch.tensor(HEIGHT_BIN_EDGES, dtype=torch.float64)
    def bin_fraction(values: torch.Tensor) -> torch.Tensor:
        index = torch.bucketize(values.to(torch.float64), edges[1:-1], right=True)
        return torch.bincount(index, minlength=HEIGHT_BIN_COUNT).float() / len(values)

    weighted_fraction = bin_fraction(weighted)
    uniform_fraction = bin_fraction(uniform)
    check("普通模式加权只改变 70% 部分的比例",
          float(weighted_fraction[3] - uniform_fraction[3]) > 0.02)

    eval_weighted = sample_heights(biased, False, count=12000)
    eval_uniform = sample_heights([1.0] * HEIGHT_BIN_COUNT, False, count=12000)
    check("评估采样与挡位权重无关", torch.equal(eval_weighted, eval_uniform))
    range_samples = sample_range_bin_heights(10000, biased, 0.060, HEIGHT_MAX, "cpu")
    check("普通模式 70% 加权采样范围正确",
          float(range_samples.min()) >= 0.060 and float(range_samples.max()) <= HEIGHT_MAX)

    state = HeightCurriculumState()
    state.z_min = 0.048
    state.level_start_iteration = 0
    results = _evaluation_results(0.048, 256, 256)
    results += [{"env_id": 300 + index, "height": 0.052, "success": index < 20,
                 "valid": True, "failure_code": 0 if index < 20 else 5} for index in range(100)]
    results += [{"env_id": 400 + index, "height": 0.072, "success": True,
                 "valid": True, "failure_code": 0} for index in range(100)]
    batch = state.measure_batch(results)
    check("普通考核返回分挡位统计",
          len(batch["bin_rate"]) == HEIGHT_BIN_COUNT and batch["bin_count"][1] == 100
          and abs(batch["bin_rate"][1] - 0.2) < 1e-9)
    outcome = state.record_evaluation([batch], 500)
    check("普通模式按挡位缺口更新 70% 采样权重",
          abs(state.bin_weights[1] - 1.8) < 1e-9 and abs(state.bin_weights[4] - 1.0) < 1e-9
          and abs(outcome["bin_rate"][1] - 0.2) < 1e-9)
    check("普通模式加权目标与推进成功率一致", NORMAL_BIN_TARGET_RATE == HEIGHT_MIN_SUCCESS_RATE)


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
        collect_passage_state(env)
    monitor = env._pass_monitor
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
    set_sites(env, x1 + 0.005, y=0.35, z=bottom + 0.02, env_ids=torch.tensor([1], device=env.device))
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
        collect_passage_state(env)
    monitor = env._pass_monitor
    env.reset(env_ids=torch.tensor([0, 2], device=env.device))
    after_reset = read_obstacle_bottom_z(obstacle).clone()
    check("部分 reset 重采样且保留其他环境板高", torch.equal(after_reset[[1, 3]], keep) and torch.all(after_reset[[0, 2]] >= HEIGHT_INITIAL_MIN))
    actual = read_geom_bottom_z(env)
    check("部分 reset 后碰撞几何与采样高度一致", torch.allclose(actual, after_reset, atol=1e-6)
          and torch.equal(actual[[1, 3]], keep))
    check("inference step 后可从外部执行部分 reset", not torch.is_inference(monitor.approached))
    verify_base_height_commands(env)
    verify_high_obstacle_env(env)
    verify_continuous_reference(env)
    verify_dynamic_sources(env)
    verify_passage_monitor(env)
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
    verify_velocity_commands()
    verify_low_leg_hold()
    verify_virtual_contact()
    verify_high_obstacle_modes()
    verify_evaluation_curriculum()
    verify_final_mode_command_sampling()
    verify_normal_adaptive_sampling()
    verify_real_env()
    if FAILURES:
        raise SystemExit(f"失败项: {FAILURES}")
    print("验收通过")


if __name__ == "__main__":
    main()
