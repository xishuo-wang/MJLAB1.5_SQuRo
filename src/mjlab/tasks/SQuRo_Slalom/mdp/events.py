from __future__ import annotations
import torch
from .indices import _MODEL_INDICES, resolve_model_indices
from .path import _INIT_DIST


# 重置模型
def reset_model(env, env_ids):
    n = len(env_ids)
    if n == 0:
      return
    
    # 获取机器人实体
    robot_entity = env.scene.entities["robot"]
    resolve_model_indices(robot_entity)
    
    # 先写中性位姿，命令管理器随后按本回合采样的曲率写最终起点。
    root_state = torch.zeros(n, 13, device=env.device)
    ax, ay = -_INIT_DIST, 0.0
    qx, qy = -0.70710678, -0.70710678

    root_state[:, 0] = ax           # x
    root_state[:, 1] = ay           # y
    root_state[:, 2] = 0.06         # z
    root_state[:, 3] = 0            # quat w
    root_state[:, 4] = qx           # quat x
    root_state[:, 5] = qy           # quat y
    root_state[:, 6] = 0.0          # quat z
    
    robot_entity.write_root_state_to_sim(root_state, env_ids=env_ids)
    
    # 重置关节状态
    joint_pos = torch.zeros(n, robot_entity.num_joints, device=env.device)
    joint_vel = torch.zeros(n, robot_entity.num_joints, device=env.device)

    joint_indices = _MODEL_INDICES.joint_ids + _MODEL_INDICES.linkage_joint_ids
    
    joint_positions = [
        0.0, 0.0, 0.0, -0.3, 0.1, -0.3, 0.1, -0.3,
        0.0, 0.0, -0.1, 0.3, -0.1, 0.3,
    ] + [
        -0.0943, 0.3867, 0.0943, 0.3862, -0.0943, 0.3867, 0.0942, -0.3862,
        0.0978, -0.3905, -0.0978, -0.3905, 0.0978, -0.3905, -0.0978, -0.3905
    ]

    for i, idx in enumerate(joint_indices):
        if idx < robot_entity.num_joints:
            joint_pos[:, idx] = joint_positions[i]

    robot_entity.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
