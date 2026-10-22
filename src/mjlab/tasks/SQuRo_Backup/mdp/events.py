from __future__ import annotations
import torch
from .config import LEG_INIT
from .indices import _MODEL_INDICES, resolve_model_indices


# 重置所用关节位置
JOINT_POSITIONS = (
    list(LEG_INIT)
    + [0, 0, 0, 0, 0, -0.3]
    + [
        -0.0943,
        0.3867,
        0.0943,
        0.3862,
        -0.0943,
        0.3867,
        0.0942,
        -0.3862,
        0.0978,
        -0.3905,
        -0.0978,
        -0.3905,
        0.0978,
        -0.3905,
        -0.0978,
        -0.3905,
    ]
)


# 把指定环境的机器人写回仰卧初态
def apply_fallen_state(env, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if n == 0:
        return

    robot_entity = env.scene.entities["robot"]
    resolve_model_indices(robot_entity)

    # 基座: identity (仰面跌倒), 位置贴地, 避免悬空落地阶段
    root_state = torch.zeros(n, 13, device=env.device)
    root_state[:, 2] = 0.024
    root_state[:, 3] = 1.0  # quat w
    robot_entity.write_root_state_to_sim(root_state, env_ids=env_ids)

    # 关节: 零速度 + 站立初始角
    joint_pos = torch.zeros(n, robot_entity.num_joints, device=env.device)
    joint_vel = torch.zeros(n, robot_entity.num_joints, device=env.device)
    indices = _MODEL_INDICES
    joint_ids = indices.joint_leg_ids + indices.joint_spn_ids + indices.joint_neck_ids + indices.linkage_joint_ids
    for idx, position in zip(joint_ids, JOINT_POSITIONS, strict=True):
        joint_pos[:, idx] = position
    robot_entity.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


# 重置模型
def reset_model(env, env_ids) -> None:
    n = len(env_ids)
    if n == 0:
        return
    apply_fallen_state(env, env_ids)


# 初始化受限空间状态
def init_restricted_space(env, env_ids) -> None:
    entity = env.scene.entities.get("restricted_space")
    if entity is None:
        return
    print(
        f"[INFO] 受限空间: 墙中心 x_neg={entity.cfg.wall_x_neg:+.4f} "
        f"x_pos={entity.cfg.wall_x_pos:+.4f} m, "
        f"实际内侧净宽 {entity.clear_width:.4f} m, "
        f"碰撞 {'开' if entity.collision_enabled else '关'}"
    )
