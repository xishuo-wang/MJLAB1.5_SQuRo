from __future__ import annotations
import torch

from .config import NECK_REF_POS
from .indices import (
    _ACT_F_BODY_ID,
    _ACT_F_SPINE1_ID,
    _ACT_FL_ELBOW_ID,
    _ACT_FL_SHOULDER_ID,
    _ACT_FR_ELBOW_ID,
    _ACT_FR_SHOULDER_ID,
    _ACT_H_BODY_ID,
    _ACT_H_SPINE1_ID,
    _ACT_HL_HIP_ID,
    _ACT_HL_KNEE_ID,
    _ACT_HR_HIP_ID,
    _ACT_HR_KNEE_ID,
    _ACT_NECK_IDS,
    _MODEL_INDICES,
    resolve_model_indices,
)


# 复位姿态: 12 个驱动腿/脊柱关节 (执行器序), 索引与取值同序
_LEG_ACTUATOR_IDS = (
    _ACT_FL_SHOULDER_ID, _ACT_FL_ELBOW_ID,
    _ACT_FR_SHOULDER_ID, _ACT_FR_ELBOW_ID,
    _ACT_HL_HIP_ID, _ACT_HL_KNEE_ID,
    _ACT_HR_HIP_ID, _ACT_HR_KNEE_ID,
    _ACT_F_SPINE1_ID, _ACT_F_BODY_ID,
    _ACT_H_SPINE1_ID, _ACT_H_BODY_ID,
)
_LEG_RESET_POS = (0.1, -0.3, 0.1, -0.3, -0.1, 0.3, -0.1, 0.3, 0.0, 0.0, 0.0, 0.0)

# 四连杆被动关节 (非驱动), 与模型关节序一一对应
_LINKAGE_JOINT_IDS = (7, 9, 10, 11, 13, 15, 16, 17, 25, 27, 28, 29, 31, 33, 34, 35)
_LINKAGE_RESET_POS = (
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

    # 驱动关节: 前/后腿 8 + 脊柱 4, 索引由 indices.py 解析 (执行器序)
    joint_ids = _MODEL_INDICES.joint_ids
    driver_ids = [joint_ids[i] for i in _LEG_ACTUATOR_IDS]
    neck_ids = [joint_ids[i] for i in _ACT_NECK_IDS]
    # 顺序与改动前一致: 12 驱动关节 + 16 四连杆被动关节 + 头颈 2
    joint_indices = driver_ids + list(_LINKAGE_JOINT_IDS) + neck_ids
    joint_positions = (list(_LEG_RESET_POS) + list(_LINKAGE_RESET_POS)
                       + [NECK_REF_POS[0], NECK_REF_POS[1]])

    for i, idx in enumerate(joint_indices):
        if idx < robot_entity.num_joints:
            joint_pos[:, idx] = joint_positions[i]

    robot_entity.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)