from __future__ import annotations
import torch

from .config import NECK_REF_POS
from .indices import resolve_model_indices


# 复位关节角: 三段分别对应 _RESET_JOINT_INDICES 的三段 (被控 12 关节 / 非被控 16 关节 / 头颈 2 关节)
# 注意 顺序既不是执行器序也不是参考表序, 而是"腿+脊柱 12 关节 -> 非被控 16 -> 头颈 2",
# 与 _RESET_JOINT_INDICES 逐项绑定; 改任一侧都必须同步改另一侧
_ACT_JOINT_POS = (0.1, -0.3, 0.1, -0.3, -0.1, 0.3, -0.1, 0.3, 0, 0, 0, 0)
_UNACT_JOINT_POS = (
    -0.0943, 0.3867, 0.0943, 0.3862, -0.0943, 0.3867, 0.0942, -0.3862,
    0.0978, -0.3905, -0.0978, -0.3905, 0.0978, -0.3905, -0.0978, -0.3905,
)
_NECK_JOINT_POS = (NECK_REF_POS[0], NECK_REF_POS[1])

# 复位关节的模型关节下标, 与上面三段关节角逐项对应 (末尾两项为 Neck_yaw / Neck_pitch)
_RESET_JOINT_INDICES = ([6, 8, 12, 14, 24, 26, 30, 32, 1, 3, 21, 23]
                        + [7, 9, 10, 11, 13, 15, 16, 17, 25, 27, 28, 29, 31, 33, 34, 35]
                        + [4, 5])


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

    joint_positions = (list(_ACT_JOINT_POS) + list(_UNACT_JOINT_POS) + list(_NECK_JOINT_POS))

    for i, idx in enumerate(_RESET_JOINT_INDICES):
        if idx < robot_entity.num_joints:
            joint_pos[:, idx] = joint_positions[i]

    robot_entity.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)