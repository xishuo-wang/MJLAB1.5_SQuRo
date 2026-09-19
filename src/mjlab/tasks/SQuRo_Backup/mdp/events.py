from __future__ import annotations
import torch
from .indices import resolve_model_indices
from .timing import FL_HOLD, HL_HOLD


# 仰卧初态常量: 基座 (0, 0, FALLEN_HEIGHT) + identity 四元数 + 初始关节角。
# 回合重置 (events.reset) 与循环复位 (command 内) 共用同一份, 避免两处漂移。
FALLEN_ROOT_HEIGHT = 0.024
FALLEN_JOINT_INDICES = [6, 8, 12, 14, 24, 26, 30, 32, 1, 3, 21, 23] + [
    7, 9, 10, 11, 13, 15, 16, 17, 25, 27, 28, 29, 31, 33, 34, 35
]
# 腿部 8 个驱动关节置为参考表 t=0 的支撑角 HOLD (参考表腿角从第一帧就是 HOLD, 无斜坡)。
# 顺序 = FL_shoulder, FL_elbow, FR_shoulder, FR_elbow, HL_hip, HL_knee, HR_hip, HR_knee。
#
# 必须与 SQuRo_Backup_env_cfg.py 的任务级 init_state 覆盖**同时**生效, 原因见那里的长注释:
# 本函数经 write_joint_state_to_sim 写的是 qpos(起点), 而驱动器零动作目标由
# default_joint_pos 决定(终点)。只改这里会被弹簧在 0.2 s 内拉回默认姿态
# (实测 HL_hip 速度峰值 10.7 rad/s, 基座 0.0240->0.0275->0.0238)。
FALLEN_LEG_POSITIONS = [FL_HOLD[0], FL_HOLD[1], FL_HOLD[0], FL_HOLD[1],
                        HL_HOLD[0], HL_HOLD[1], HL_HOLD[0], HL_HOLD[1]]
# 后 4 项为脊柱四关节 (置 0), 与参考表 t=0 的脊柱列一致。
FALLEN_SPINE_POSITIONS = [0, 0, 0, 0]
FALLEN_JOINT_POSITIONS = FALLEN_LEG_POSITIONS + FALLEN_SPINE_POSITIONS + [
    -0.0943, 0.3867, 0.0943, 0.3862, -0.0943, 0.3867, 0.0942, -0.3862,
    0.0978, -0.3905, -0.0978, -0.3905, 0.0978, -0.3905, -0.0978, -0.3905
]


# 把指定环境的机器人写回仰卧初态 — 只写物理状态, 不碰任何管理器或回合计数。
# 注意: FALLEN_JOINT_INDICES 目前仍是硬编码下标 (技术细节 §8 遗留问题 2), 本函数只做"消重",
# 并没有消除这个漂移风险; XML 改关节顺序时这里与 indices.py 都要复核。
def apply_fallen_state(env, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if n == 0:
        return

    robot_entity = env.scene.entities["robot"]
    resolve_model_indices(robot_entity)

    # 基座: identity (仰面跌倒), 位置贴地, 避免悬空落地阶段
    root_state = torch.zeros(n, 13, device=env.device)
    root_state[:, 2] = FALLEN_ROOT_HEIGHT
    root_state[:, 3] = 1.0  # quat w
    robot_entity.write_root_state_to_sim(root_state, env_ids=env_ids)

    # 关节: 零速度 + 站立初始角
    joint_pos = torch.zeros(n, robot_entity.num_joints, device=env.device)
    joint_vel = torch.zeros(n, robot_entity.num_joints, device=env.device)
    for i, idx in enumerate(FALLEN_JOINT_INDICES):
        if idx < robot_entity.num_joints:
            joint_pos[:, idx] = FALLEN_JOINT_POSITIONS[i]
    robot_entity.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


# 重置模型 — 回合级重置入口 (EventTermCfg mode="reset")
def reset_model(env, env_ids) -> None:
    n = len(env_ids)
    if n == 0:
        return
    # 站立窗口与循环级状态现在归 BackupCommand 所有, 由 command_manager.reset →
    # _resample_command → _clear_cycle_state 统一清理。这里**不再**动 env._stand_* ——
    # 那些旧字段早已不是归属地, 留着只会让人以为完整回合重置清干净了(实际没有)。
    # 顺序保证: ManagerBasedRlEnv._reset_idx 先 sim.reset/scene.reset, 再 command_manager.reset。
    apply_fallen_state(env, env_ids)
