from __future__ import annotations
import math as _m
import torch
from .path import _INIT_DIST, get_approach_start, get_effective_pole_spacing
from .curriculums import get_training_phase, get_curriculum_pole_spacing


# 重置模型
def reset_model(env, env_ids):
    n = len(env_ids)
    if n == 0:
      return
    
    # 获取机器人实体
    robot_entity = env.scene.entities["robot"]
    
    # 重置基座状态: Phase 1 用接近段圆弧起点, Phase 0 用直行起点 (与期望轨迹一致)
    root_state = torch.zeros(n, 13, device=env.device)
    if get_training_phase(env.common_step_counter) == 1:
        # 新几何: 起点右移杆间距 (接近段终点 = 第一根杆正上方 (spacing, 0))
        spacing = get_effective_pole_spacing(get_curriculum_pole_spacing(env.common_step_counter))
        ax, ay, ah = get_approach_start(spacing=spacing)
        c, s = _m.cos(ah / 2), _m.sin(ah / 2)
        qx, qy = 0.70710678 * (s - c), -0.70710678 * (c + s)
    else:
        ax, ay = -_INIT_DIST, 0.0
        qx, qy = -0.70710678, -0.70710678
    root_state[:, 0] = ax         # x
    root_state[:, 1] = ay         # y
    root_state[:, 2] = 0.06       # z
    root_state[:, 3] = 0          # quat w
    root_state[:, 4] = qx        # quat x
    root_state[:, 5] = qy        # quat y
    root_state[:, 6] = 0.0          # quat z
    
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
