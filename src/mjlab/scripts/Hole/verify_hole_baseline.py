# uv run python -B -m mjlab.scripts.Hole.verify_hole_baseline
# Hole 兼容性验证: 参考表 / 历史位置表 / 三阶段课程 / 实体碰撞 / 观测维度
# 当前口径: 14 执行器 (含头颈), 参考表 14 列

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Hole.mdp.command import STAGE3_POSITION_SCHEDULE

# 阶段边界与高度阈值的唯一来源都是 config (command 只是转出)
from mjlab.tasks.SQuRo_Hole.mdp.config import (
    CMD_HEIGHT_F_IDS,
    CMD_HEIGHT_H_IDS,
    CMD_VEL_X_IDS,
    NUM_STAGES,
    STAGE1_END,
    STAGE2_END,
    STAGE3_END,
    STEPS_PER_ITER,
    TABLE_RESOLUTION,
    THRESHOLD_HEIGHT,
    get_current_stage,
    heights_for_stage,
    stage_requires_collision,
    stage_uses_schedule,
)
from mjlab.tasks.SQuRo_Hole.mdp.curriculums import reward_weight_curriculum
from mjlab.tasks.SQuRo_Hole.mdp.indices import (
    REF_FL_LEG_SLICE,
    REF_HL_LEG_SLICE,
    REF_SPN_SLICE,
    SPN_LOCAL_H_PITCH,
)
from mjlab.tasks.SQuRo_Hole.mdp.reference import (
    ACTUATOR_NUM,
    CYCLOID_PARAMS,
    HEIGHT_LIST,
    Initialize_Tables,
)

# 模式数由 CYCLOID_PARAMS 决定 (normal / front_low / hind_low)
NUM_MODES = len(CYCLOID_PARAMS)


# 从实体本身读限高板碰撞开关, 不缓存"以为改成了什么"
def hole_collision_enabled(env) -> bool:
    entity = env.scene.entities.get("hole1")
    if entity is None:
        return False
    return int(getattr(entity.cfg, "contype", 0)) > 0


# 参考表结构与模式/高度档的作用
def check_reference() -> None:
    tables = Initialize_Tables("cpu")
    front = tables["front_pos"]
    print(f"[1] 参考表 front_pos {tuple(front.shape)} "
          f"(模式 {NUM_MODES} × 高度档 {len(HEIGHT_LIST)} × 相位 {TABLE_RESOLUTION} × 4 关节) "
          f"被控关节数 {ACTUATOR_NUM}")
    print(f"{'模式':>6}{'高度(mm)':>10}{'FL肩幅度':>12}{'HL髋幅度':>12}{'脊柱H_spine1':>14}")
    for mode in range(NUM_MODES):
        for h_idx, h in enumerate(HEIGHT_LIST):
            sp = tables["spine_pos"][mode, h_idx]
            print(f"{mode:>6}{h * 1000:>10.0f}"
                  f"{float(front[mode, h_idx, :, REF_FL_LEG_SLICE.start].max() - front[mode, h_idx, :, REF_FL_LEG_SLICE.start].min()):>12.4f}"
                  f"{float(tables['hind_pos'][mode, h_idx, :, REF_HL_LEG_SLICE.start].max() - tables['hind_pos'][mode, h_idx, :, REF_HL_LEG_SLICE.start].min()):>12.4f}"
                  f"{float(sp[:, REF_SPN_SLICE.start + SPN_LOCAL_H_PITCH].mean()):>14.4f}")
    print(f"     高度阈值 THRESHOLD_HEIGHT = {THRESHOLD_HEIGHT} m (低于此值判为'压低'档)")


# 命令: 回放位置表与阶段边界
def check_command() -> None:
    cfg = load_env_cfg("Mjlab-SQuRo-Hole", play=True)
    cfg.scene.num_envs = 8
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    cmd = env.command_manager.get_command("hole_cmd")
    print(f"\n[2] 回放位置表 ({len(STAGE3_POSITION_SCHEDULE)} 段):")
    for dist, h_f, h_h in STAGE3_POSITION_SCHEDULE:
        print(f"     位移 {dist:.2f} m → h_F {h_f * 1000:.0f} mm, h_H {h_h * 1000:.0f} mm")
    print(f"     reset 后命令: vel_x {float(cmd[0, CMD_VEL_X_IDS]):.3f} m/s, "
          f"h_F {float(cmd[0, CMD_HEIGHT_F_IDS]) * 1000:.0f} mm, h_H {float(cmd[0, CMD_HEIGHT_H_IDS]) * 1000:.0f} mm (期望第一段 20/50)")
    print(f"[3] 命令阶段边界: step 0 → stage {get_current_stage(0)}, "
          f"step {STAGE1_END} → stage {get_current_stage(STAGE1_END)}, "
          f"step {STAGE2_END} → stage {get_current_stage(STAGE2_END)}, "
          f"step {STAGE3_END} → stage {get_current_stage(STAGE3_END)}")
    print("[3b] 三阶段定义 (边界唯一定义在 mdp/config.py):")
    print(f"     {'阶段':>4}{'iter 区间':>16}{'命令来源':>26}{'位置表':>8}{'要求碰撞':>10}")
    bounds = [0, STAGE1_END // STEPS_PER_ITER, STAGE2_END // STEPS_PER_ITER]
    for s in range(1, NUM_STAGES + 1):
        lo = bounds[s - 1]
        hi = bounds[s] - 1 if s < NUM_STAGES else "∞"
        pool = heights_for_stage(s)
        src = "单板高度课程" if pool is None else f"随机 {len(pool)} 档"
        print(f"     {s:>4}{f'{lo}-{hi}':>16}{src:>26}"
              f"{str(stage_uses_schedule(s)):>8}{str(stage_requires_collision(s)):>10}")

    # 位移推进查表: 位置表只在 stage 3 生效, 先把计数器推到该阶段
    env.common_step_counter = STAGE2_END
    robot = env.scene["robot"]
    state = torch.zeros(env.num_envs, 13, device=env.device)
    state[:, 2] = 0.06
    state[:, 3] = 0.0
    state[:, 4] = -0.70710678
    state[:, 5] = -0.70710678
    state[:, 0] = 0.25          # 超过第一段 0.2 m
    robot.write_root_state_to_sim(state)
    env.sim.forward()               # 写 qpos 后必须 forward, 否则读到的是旧位置
    env.command_manager.compute(dt=env.step_dt)
    cmd2 = env.command_manager.get_command("hole_cmd")
    print(f"     x=0.25 m 时命令: h_F {float(cmd2[0, CMD_HEIGHT_F_IDS]) * 1000:.0f} mm, "
          f"h_H {float(cmd2[0, CMD_HEIGHT_H_IDS]) * 1000:.0f} mm (期望第二段 55/20)")
    env.close()


# 课程权重: 阶段阈值与生效段
def check_curriculum() -> None:
    print(f"\n[4] 奖励权重课程 (阈值单位 = 全局步数, {len(reward_weight_curriculum.weight_stages)} 段):")
    keys = ["body_contact", "height", "mimic_pos", "height_sigma"]
    header = "".join(f"{k:>16}" for k in keys)
    print(f"{'阶段阈值':>12}{'iter':>8}{header}{'enable_holes':>14}")
    for step in sorted(reward_weight_curriculum.weight_stages.keys()):
        w = reward_weight_curriculum.weight_stages[step]
        row = "".join(f"{w.get(k, float('nan')):>16.2f}" for k in keys)
        print(f"{step:>12}{step // STEPS_PER_ITER:>8}{row}"
              f"{str(reward_weight_curriculum.should_enable_holes(step)):>14}")
    eff = reward_weight_curriculum.get_reward_weights(STAGE3_END)
    print(f"     末段生效权重: mimic_pos={eff['mimic_pos']}, height={eff['height']}, "
          f"vel={eff['vel']}, body_contact={eff['body_contact']}, "
          f"enable_holes={reward_weight_curriculum.should_enable_holes(STAGE3_END)}")


# 实体碰撞与观测维度
def check_env() -> None:
    for play in (False, True):
        cfg = load_env_cfg("Mjlab-SQuRo-Hole", play=play)
        cfg.scene.num_envs = 2
        entities = cfg.scene.entities
        tag = "回放" if play else "训练"
        if not play:
            print(f"\n[5] 场景实体: {list(entities.keys())}")
            for key in ("hole1", "hole2", "hole3"):
                e = entities[key]
                print(f"     {key}: pos={e.position}, size={e.size}, contype={e.contype}, "
                      f"conaffinity={e.conaffinity}")
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
        obs = env.reset()
        obs_dict = obs[0] if isinstance(obs, tuple) else obs
        print(f"     [{tag}] 初始限高板碰撞={hole_collision_enabled(env)} (前两阶段关 / 课程阶段重建开), "
              f"actor {tuple(obs_dict['actor'].shape)}, "
              f"动作 {env.action_space.shape[-1]}, 参考表被控关节数 {ACTUATOR_NUM}")
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
    print("\n[结论] 参考表 3 模式 × 4 高度档 × 500 相位 × 14 关节 (含头颈)、三阶段课程 "
          "(0-1k 正常档随机 / 1k-2k 含低高度全档随机 / 2k+ 单板真实碰撞高度课程)、"
          "动作 14 维 / 观测 113 维；历史位置表保持可回放")
    print("     注: 课程阶段的实体碰撞由 rl/runner.py 在阶段边界重建环境完成 "
          "(contype 编译期固化, 运行期改无效)")


if __name__ == "__main__":
    main()
