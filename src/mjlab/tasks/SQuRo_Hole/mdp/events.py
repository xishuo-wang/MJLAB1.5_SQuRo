from __future__ import annotations
import torch
from .config import NECK_REF_POS
from .indices import (
    LINKAGE_JOINT_NAMES,
    _ACT_JOINT_NAMES,
    _MODEL_INDICES,
    resolve_model_indices,
)



# 复位角顺序与 indices._ACT_JOINT_NAMES 的执行器顺序一致 (头颈取 NECK_REF_POS)
_ACT_JOINT_POS = (
    0.0, 0.0,                   # F_spine1, F_body
    *NECK_REF_POS,              # Neck_yaw, Neck_pitch
    0.1, -0.3, 0.1, -0.3,       # FL/FR shoulder, elbow
    0.0, 0.0,                   # H_spine1, H_body
    -0.1, 0.3, -0.1, 0.3,       # HL/HR hip, knee
)


# 闭链被动关节随动角, 顺序与 indices.LINKAGE_JOINT_NAMES 一致
_UNACT_JOINT_POS = (
    -0.0943, 0.3867, 0.0943, 0.3862, -0.0943, 0.3867, 0.0942, -0.3862,
    0.0978, -0.3905, -0.0978, -0.3905, 0.0978, -0.3905, -0.0978, -0.3905,
)

assert len(_ACT_JOINT_POS) == len(_ACT_JOINT_NAMES), "复位角数量与执行器数不一致"
assert len(_UNACT_JOINT_POS) == len(LINKAGE_JOINT_NAMES), "闭链复位角数量与闭链关节数不一致"



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

    joint_positions = list(_ACT_JOINT_POS) + list(_UNACT_JOINT_POS)
    joint_indices = tuple(_MODEL_INDICES.joint_ids) + tuple(_MODEL_INDICES.linkage_joint_ids)

    for i, idx in enumerate(joint_indices):
        if idx < robot_entity.num_joints:
            joint_pos[:, idx] = joint_positions[i]

    robot_entity.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
