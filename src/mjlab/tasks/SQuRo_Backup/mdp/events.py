from __future__ import annotations
import torch
from .config import LEG_INIT
from .indices import (
    _ACT_LEG_IDS,
    _ACT_NECK_PITCH_ID,
    _ACT_NUM,
    _MODEL_INDICES,
    resolve_model_indices,
)


# 按执行器列序组织驱动关节角，腿角沿用统一配置。
def _act_joint_positions() -> tuple[float, ...]:
    positions = [0.0] * _ACT_NUM
    for column, angle in zip(_ACT_LEG_IDS, LEG_INIT, strict=True):
        positions[column] = angle
    positions[_ACT_NECK_PITCH_ID] = -0.3
    return tuple(positions)


# 驱动关节的复位角度。
_ACT_JOINT_POS = _act_joint_positions()

# 非驱动关节的复位角度。
_UNACT_JOINT_POS = (
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
    joint_ids = indices.joint_ids + indices.linkage_joint_ids
    positions = _ACT_JOINT_POS + _UNACT_JOINT_POS
    for idx, position in zip(joint_ids, positions, strict=True):
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
