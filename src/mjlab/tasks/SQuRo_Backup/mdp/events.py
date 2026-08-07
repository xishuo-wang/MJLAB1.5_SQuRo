from __future__ import annotations
import torch
from .indices import _ACTUATED_JOINT_NAMES, resolve_model_indices


# 重置模型 — 仰面跌倒初始状态 (identity quat, 站立初始关节角)
def reset_model(env, env_ids):
    n = len(env_ids)
    if n == 0:
        return

    # 获取机器人实体
    robot_entity = env.scene.entities["robot"]
    resolve_model_indices(robot_entity)

    # 基座: identity (仰面跌倒), 位置 (0,0,0.024) — 贴地初始 (仰面稳定躺姿 base_z≈0.0236),
    # 避免悬空落地阶段, 可直接开始翻身复位
    root_state = torch.zeros(n, 13, device=env.device)
    root_state[:, 0] = 0.0
    root_state[:, 1] = 0.0
    root_state[:, 2] = 0.024
    root_state[:, 3] = 1.0  # quat w
    robot_entity.write_root_state_to_sim(root_state, env_ids=env_ids)

    # 重置关节状态
    joint_pos = torch.zeros(n, robot_entity.num_joints, device=env.device)
    joint_vel = torch.zeros(n, robot_entity.num_joints, device=env.device)

    joint_indices = [6, 8, 12, 14, 24, 26, 30, 32, 1, 3, 21, 23] + [
        7, 9, 10, 11, 13, 15, 16, 17, 25, 27, 28, 29, 31, 33, 34, 35
    ]
    
    joint_positions = [0.1, -0.3, 0.1, -0.3, -0.1, 0.3, -0.1, 0.3, 0, 0, 0, 0] + [
        -0.0943, 0.3867, 0.0943, 0.3862, -0.0943, 0.3867, 0.0942, -0.3862,
        0.0978, -0.3905, -0.0978, -0.3905, 0.0978, -0.3905, -0.0978, -0.3905
    ]

    for i, idx in enumerate(joint_indices):
        if idx < robot_entity.num_joints:
            joint_pos[:, idx] = joint_positions[i]

    robot_entity.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
