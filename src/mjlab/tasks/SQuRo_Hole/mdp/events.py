from __future__ import annotations
import torch
from .indices import (
    _MODEL_INDICES,
    _ACT_JOINT_IDS,
    resolve_model_indices,
)



# 驱动关节
_ACT_JOINT_POS = (0.0, 0.0, 0.0, -0.3, 0.1, -0.3, 0.1, -0.3, 0.0, 0.0, -0.1, 0.3, -0.1, 0.3)

# 非驱动关节
_UNACT_JOINT_IDS = (7, 9, 10, 11, 13, 15, 16, 17, 25, 27, 28, 29, 31, 33, 34, 35)
_UNACT_JOINT_POS = (
    -0.0943, 0.3867, 0.0943, 0.3862, -0.0943, 0.3867, 0.0942, -0.3862,
    0.0978, -0.3905, -0.0978, -0.3905, 0.0978, -0.3905, -0.0978, -0.3905,
)



# 重置模型
def reset_model(env, env_ids):
    n = len(env_ids)
    if n == 0:
        return

    # 获取机器人实体
    robot_entity = env.scene.entities["robot"]
    resolve_model_indices(robot_entity)

    # 重置基座状态
    root_state = torch.zeros(n, 13, device=env.device)
    root_state[:, 0] = 0.0        # x
    root_state[:, 1] = 0.0        # y
    root_state[:, 2] = 0.06       # z
    root_state[:, 3] = 0          # quat w
    root_state[:, 4] = -0.707107  # quat x
    root_state[:, 5] = -0.707107  # quat y
    root_state[:, 6] = 0.0        # quat z
    robot_entity.write_root_state_to_sim(root_state, env_ids=env_ids)

    # 重置关节状态
    joint_pos = torch.zeros(n, robot_entity.num_joints, device=env.device)
    joint_vel = torch.zeros(n, robot_entity.num_joints, device=env.device)
    model_joint_ids = _MODEL_INDICES.joint_ids
    driver_joint_indices = [model_joint_ids[i] for i in _ACT_JOINT_IDS]
    joint_indices = driver_joint_indices + list(_UNACT_JOINT_IDS)
    joint_positions = list(_ACT_JOINT_POS) + list(_UNACT_JOINT_POS)

    for i, idx in enumerate(joint_indices):
        if idx < robot_entity.num_joints:
            joint_pos[:, idx] = joint_positions[i]

    robot_entity.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)