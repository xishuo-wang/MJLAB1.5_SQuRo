from __future__ import annotations
import torch
from .indices import resolve_model_indices
from .config import LEG_INIT


# 仰卧初态常量: 基座 (0, 0, FALLEN_HEIGHT) + identity 四元数 + 初始关节角。
# 回合重置 (events.reset) 与循环复位 (command 内) 共用同一份, 避免两处漂移。
FALLEN_ROOT_HEIGHT = 0.024
FALLEN_JOINT_INDICES = [6, 8, 12, 14, 24, 26, 30, 32, 1, 3, 21, 23] + [
    7, 9, 10, 11, 13, 15, 16, 17, 25, 27, 28, 29, 31, 33, 34, 35
]
# 腿部 8 个驱动关节置为站立角 LEG_INIT, 与 default_joint_pos(零动作目标)一致;
# 收腿动作由参考轨迹的前置段 P0 完成。顺序 = FL/FR shoulder,elbow, HL/HR hip,knee。
FALLEN_LEG_POSITIONS = list(LEG_INIT)
# 后 4 项为脊柱四关节 (置 0), 与参考表 t=0 的脊柱列一致。
FALLEN_SPINE_POSITIONS = [0, 0, 0, 0]
FALLEN_JOINT_POSITIONS = FALLEN_LEG_POSITIONS + FALLEN_SPINE_POSITIONS + [
    -0.0943, 0.3867, 0.0943, 0.3862, -0.0943, 0.3867, 0.0942, -0.3862,
    0.0978, -0.3905, -0.0978, -0.3905, 0.0978, -0.3905, -0.0978, -0.3905
]


# 把指定环境的机器人写回仰卧初态 — 只写物理状态, 不碰任何管理器或回合计数。
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


# 重置模型
def reset_model(env, env_ids) -> None:
    n = len(env_ids)
    if n == 0:
        return
    apply_fallen_state(env, env_ids)
