from __future__ import annotations
import torch
from .config import LEG_INIT
from .indices import resolve_model_indices


# 重置所用关节索引
JOINT_INDICES = [6, 8, 12, 14, 24, 26, 30, 32, 1, 3, 21, 23] + [
    7, 9, 10, 11, 13, 15, 16, 17, 25, 27, 28, 29, 31, 33, 34, 35
]



# 重置所用关节位置
JOINT_POSITIONS = list(LEG_INIT) + [0, 0, 0, 0] + [
    -0.0943, 0.3867, 0.0943, 0.3862, -0.0943, 0.3867, 0.0942, -0.3862,
    0.0978, -0.3905, -0.0978, -0.3905, 0.0978, -0.3905, -0.0978, -0.3905
]



# 重置模型
def reset_model(env, env_ids):
    n = len(env_ids)
    if n == 0:
        return

    robot_entity = env.scene.entities["robot"]
    resolve_model_indices(robot_entity)

    # 基座状态: 起点 (0, 0), 高度 0.06, 直行朝向 (+X)
    root_state = torch.zeros(n, 13, device=env.device)
    root_state[:, 2] = 0.06             # z
    root_state[:, 4] = -0.70710678      # quat x
    root_state[:, 5] = -0.70710678      # quat y
    robot_entity.write_root_state_to_sim(root_state, env_ids=env_ids)

    # 关节状态 (与 Slalom 相同的初始关节角, 含闭链机构)
    joint_pos = torch.zeros(n, robot_entity.num_joints, device=env.device)
    joint_vel = torch.zeros(n, robot_entity.num_joints, device=env.device)

    joint_pos = torch.zeros(n, robot_entity.num_joints, device=env.device)
    joint_vel = torch.zeros(n, robot_entity.num_joints, device=env.device)
    for i, idx in enumerate(JOINT_INDICES):
        if idx < robot_entity.num_joints:
            joint_pos[:, idx] = JOINT_POSITIONS[i]
    robot_entity.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)