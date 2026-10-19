# uv run python -B -m mjlab.scripts.Hole.verify_hole_baseline
# Hole 验证：参考表、单板高度规则、四阶段课程、单板场景与观测维度
# 当前口径: 14 执行器 (含头颈), 参考表 14 列

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

# 阶段边界与高度阈值的唯一来源: 参数在 config, 阶段边界与判定在 curriculums
from mjlab.tasks.SQuRo_Hole.mdp.config import (
    CMD_HEIGHT_F_IDS,
    CMD_HEIGHT_H_IDS,
    CMD_VEL_X_IDS,
    NUM_MODES,
    TABLE_RESOLUTION,
    THRESHOLD_HEIGHT,
)
from mjlab.tasks.SQuRo_Hole.mdp.curriculums import (
    NUM_STAGES,
    STAGE1_STEP,
    STAGE2_STEP,
    STAGE3_STEP,
    STEPS_PER_ITER,
    HeightCurriculumState,
    get_current_stage,
    heights_for_stage,
    reward_weight_curriculum,
    obstacle_rule_bounds,
    stage_uses_height_course,
)
from mjlab.tasks.SQuRo_Hole.mdp.entity import (
    configure_obstacle_entity,
    obstacle_collision_enabled,
    write_obstacle_mocap,
)
from mjlab.tasks.SQuRo_Hole.mdp.indices import (
    _ACT_NUM,
    _ACT_FL_LEG_IDS,
    _ACT_HL_LEG_IDS,
    _ACT_SPN_H_PITCH_ID,
)
from mjlab.tasks.SQuRo_Hole.mdp.reference import (
    HEIGHT_LIST,
    initialize_tables,
)

# 参考表结构与模式/高度档的作用
def check_reference() -> None:
    tables = initialize_tables("cpu")
    front = tables["front_pos"]
    num_modes = front.shape[0]
    assert num_modes == NUM_MODES
    print(f"[1] 参考表 front_pos {tuple(front.shape)} "
          f"(模式 {num_modes} × 高度档 {len(HEIGHT_LIST)} × 相位 {TABLE_RESOLUTION} × {_ACT_NUM} 关节) "
          f"被控关节数 {_ACT_NUM}")
    print(f"{'模式':>6}{'高度(mm)':>10}{'FL肩幅度':>12}{'HL髋幅度':>12}{'脊柱H_spine1':>14}")
    for mode in range(num_modes):
        for h_idx, h in enumerate(HEIGHT_LIST):
            # 表按执行器序存储, 列下标 = 执行器下标
            fl_shoulder = front[mode, h_idx, :, _ACT_FL_LEG_IDS[0]]
            hl_hip = tables["hind_pos"][mode, h_idx, :, _ACT_HL_LEG_IDS[0]]
            h_spine1 = tables["spine_pos"][mode, h_idx, :, _ACT_SPN_H_PITCH_ID]
            print(f"{mode:>6}{h * 1000:>10.0f}"
                  f"{float(fl_shoulder.max() - fl_shoulder.min()):>12.4f}"
                  f"{float(hl_hip.max() - hl_hip.min()):>12.4f}"
                  f"{float(h_spine1.mean()):>14.4f}")
    print(f"     高度阈值 THRESHOLD_HEIGHT = {THRESHOLD_HEIGHT} m (低于此值判为'压低'档)")


# 命令: 单板高度规则与阶段边界
def check_command() -> None:
    cfg = load_env_cfg("Mjlab-SQuRo-Hole", play=True)
    cfg.scene.num_envs = 8
    configure_obstacle_entity(cfg, enable_collision=False)
    state = HeightCurriculumState()
    state.z_min = state.z_max = 0.070
    cfg.commands["hole_cmd"].enable_height_course = True
    cfg.commands["hole_cmd"].height_curriculum_state = state.state_dict()
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    cmd = env.command_manager.get_command("hole_cmd")
    obstacle = env.scene["obstacle"].cfg
    left, middle, right = obstacle_rule_bounds(obstacle.position[0], obstacle.size[0])
    print(f"\n[2] 单板高度规则: 进入 {left:.3f} m，中点 {middle:.3f} m，离开 {right:.3f} m")
    print(f"     reset 后命令: vel_x {float(cmd[0, CMD_VEL_X_IDS]):.3f} m/s, "
          f"h_F {float(cmd[0, CMD_HEIGHT_F_IDS]) * 1000:.0f} mm, h_H {float(cmd[0, CMD_HEIGHT_H_IDS]) * 1000:.0f} mm (热身随机高度池)")
    print(f"[3] 命令阶段边界: step 0 → stage {get_current_stage(0)}, "
          f"step {STAGE1_STEP} → stage {get_current_stage(STAGE1_STEP)}, "
          f"step {STAGE2_STEP} → stage {get_current_stage(STAGE2_STEP)}, "
          f"step {STAGE3_STEP} → stage {get_current_stage(STAGE3_STEP)}")
    print("[3b] 四阶段定义 (边界唯一定义在 mdp/curriculums.py):")
    print(f"     {'阶段':>4}{'iter 区间':>16}{'命令来源':>26}{'高度课程':>8}")
    bounds = [0, STAGE1_STEP // STEPS_PER_ITER, STAGE2_STEP // STEPS_PER_ITER, STAGE3_STEP // STEPS_PER_ITER]
    for s in range(1, NUM_STAGES + 1):
        lo = bounds[s - 1]
        hi = bounds[s] - 1 if s < NUM_STAGES else "∞"
        pool = heights_for_stage(s)
        src = "单板高度课程" if pool is None else f"随机 {len(pool)} 档"
        print(f"     {s:>4}{f'{lo}-{hi}':>16}{src:>26}"
              f"{str(stage_uses_height_course(s)):>8}")

    # 将阶段和板底设置为真实课程口径。
    env.common_step_counter = STAGE3_STEP
    env.reset()
    write_obstacle_mocap(env.scene["obstacle"], 0.064)
    robot = env.scene["robot"]
    state = torch.zeros(env.num_envs, 13, device=env.device)
    state[:, 2] = 0.06
    state[:, 3] = 0.0
    state[:, 4] = -0.70710678
    state[:, 5] = -0.70710678
    state[:, 0] = 0.25          # 位于板中心之后
    robot.write_root_state_to_sim(state)
    env.sim.forward()               # 写 qpos 后必须 forward, 否则读到的是旧位置
    env.command_manager.compute(dt=env.step_dt)
    cmd2 = env.command_manager.get_command("hole_cmd")
    torch.testing.assert_close(
        cmd2[:, [CMD_HEIGHT_F_IDS, CMD_HEIGHT_H_IDS]],
        torch.tensor([[0.055, 0.035]]).expand(env.num_envs, -1), atol=1e-6, rtol=0,
    )
    print(f"     x=0.25 m 时命令: h_F {float(cmd2[0, CMD_HEIGHT_F_IDS]) * 1000:.0f} mm, "
          f"h_H {float(cmd2[0, CMD_HEIGHT_H_IDS]) * 1000:.0f} mm (64毫米低板后半段 55/35)")
    env.close()


# 课程权重: 阶段阈值与生效段
def check_curriculum() -> None:
    # 阶段阈值 = iter 边界 × 每 iter 步数
    thresholds = [0, STAGE1_STEP, STAGE2_STEP, STAGE3_STEP]
    print(f"\n[4] 奖励权重课程 (阈值单位 = 全局步数, {len(thresholds)} 段):")
    keys = ["weight_corridor", "weight_height", "weight_mimic_pos", "sigma_height"]
    header = "".join(f"{k:>16}" for k in keys)
    print(f"{'阶段阈值':>12}{'iter':>8}{header}{'训练碰撞':>14}")
    for step in thresholds:
        w = reward_weight_curriculum.get_curriculum_values(step)
        row = "".join(f"{w.get(k, float('nan')):>16.2f}" for k in keys)
        print(f"{step:>12}{step // STEPS_PER_ITER:>8}{row}"
              f"{'关':>14}")
    eff = reward_weight_curriculum.get_curriculum_values(STAGE3_STEP)
    print(f"     末段生效权重: mimic_pos={eff['weight_mimic_pos']}, height={eff['weight_height']}, "
          f"vel={eff['weight_vel']}, corridor={eff['weight_corridor']}, "
          f"训练碰撞=关 (独立考核环境开启)")


# 实体碰撞与观测维度
def check_env() -> None:
    for play in (False, True):
        cfg = load_env_cfg("Mjlab-SQuRo-Hole", play=play)
        cfg.scene.num_envs = 2
        entities = cfg.scene.entities
        tag = "回放" if play else "训练"
        if not play:
            print(f"\n[5] 场景实体: {list(entities.keys())}")
            assert list(entities) == ["robot"], "热身场景不应创建障碍物"
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
        obs = env.reset()
        obs_dict = obs[0] if isinstance(obs, tuple) else obs
        print(f"     [{tag}] 初始限高板碰撞={obstacle_collision_enabled(env)} (前三阶段无板 / 课程阶段重建单板且无碰撞), "
              f"actor {tuple(obs_dict['actor'].shape)}, "
              f"动作 {env.action_space.shape[-1]}, 参考表被控关节数 {_ACT_NUM}")
        if not play:
            print(f"     仿真: timestep={cfg.sim.mujoco.timestep}, decimation={cfg.decimation}, "
                  f"episode={cfg.episode_length_s}s → {env.max_episode_length} 步/回合")
            print(f"     终止: {[t for t in cfg.terminations.keys()]}")
        env.close()


def main() -> None:
    check_reference()
    check_command()
    check_curriculum()
    check_env()
    print(f"\n[结论] 参考表 {NUM_MODES} 模式 × {len(HEIGHT_LIST)} 高度档 × {TABLE_RESOLUTION} 相位 × {_ACT_NUM} 关节 (含头颈)、四阶段课程 "
          "(0-1k 正常档随机 / 1k-2k 全档随机 / 2k-3k 走廊热身 / 3k+ 单板高度课程)、"
          "动作 14 维 / 观测 113 维；高度规则按基座世界 X 查表")
    print("     注: 训练环境不开启实体碰撞, 3000轮由 rl/runner.py 重建单板; "
          "接触只在独立真实碰撞考核环境开启")


if __name__ == "__main__":
    main()
