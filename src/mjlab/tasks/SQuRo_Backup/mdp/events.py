from __future__ import annotations
import torch
from .indices import _ACTUATED_JOINT_NAMES, resolve_model_indices


# 跌倒爬起初始关节角: 腿=站立角, 脊柱/颈=0 (与参考脚本 Loco_Backup 一致, 闭链关节保持 0)
_INIT_JOINT_VALUES = [
    0.0, 0.0,        # F_spine1, F_body
    0.0, 0.0,        # Neck_yaw, Neck_pitch
    0.1, -0.3,       # FL_shoulder, FL_elbow
    0.1, -0.3,       # FR_shoulder, FR_elbow
    0.0, 0.0,        # H_spine1, H_body
    -0.1, 0.3,       # HL_hip, HL_knee
    -0.1, 0.3,       # HR_hip, HR_knee
]


# 重置模型 — 仰面跌倒初始状态 (identity quat, 站立初始关节角)
def reset_model(env, env_ids):
    n = len(env_ids)
    if n == 0:
        return

    robot_entity = env.scene.entities["robot"]
    resolve_model_indices(robot_entity)

    # 基座: identity (仰面跌倒), 位置 (0,0,0.06)
    root_state = torch.zeros(n, 13, device=env.device)
    root_state[:, 0] = 0.0
    root_state[:, 1] = 0.0
    root_state[:, 2] = 0.06
    root_state[:, 3] = 1.0  # quat w
    robot_entity.write_root_state_to_sim(root_state, env_ids=env_ids)

    # 驱动关节: 站立初始角; 其余(含闭链)关节保持 0
    joint_pos = torch.zeros(n, robot_entity.num_joints, device=env.device)
    joint_vel = torch.zeros(n, robot_entity.num_joints, device=env.device)
    joint_ids, _ = robot_entity.find_joints(_ACTUATED_JOINT_NAMES, preserve_order=True)
    for i, jid in enumerate(joint_ids):
        joint_pos[:, jid] = _INIT_JOINT_VALUES[i]

    robot_entity.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
