from __future__ import annotations
import torch
from mjlab.entity import Entity
from typing import TYPE_CHECKING
from .indices import _MODEL_INDICES
from .reference import get_reference_joint_state
from .curriculums import get_curriculum_reward_weight
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv



# =========================================================================================
# 关节位置模仿奖励
def compute_mimic_pos_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    # 计算关节角度误差
    joint_pos = asset.data.joint_pos[:, _MODEL_INDICES.joint_ids]
    ref_pos, _ = get_reference_joint_state(env)
    error = joint_pos - ref_pos
    # 获取课程学习量
    weight = get_curriculum_reward_weight(env, "weight_mimic_pos")
    sigma = get_curriculum_reward_weight(env, "sigma_mimic_pos")
    # 计算奖励
    mse = torch.mean(error ** 2, dim=1)
    reward = torch.exp(-sigma * mse)
    return reward * weight



# =========================================================================================
# 关节速度模仿奖励
def compute_mimic_vel_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    # 计算关节速度误差
    joint_vel = asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
    _, ref_vel = get_reference_joint_state(env)
    error = joint_vel - ref_vel
    # 获取课程学习量
    weight = get_curriculum_reward_weight(env, "weight_mimic_vel")
    sigma = get_curriculum_reward_weight(env, "sigma_mimic_vel")
    # 计算奖励
    mse = torch.mean(error ** 2, dim=1)
    reward = torch.exp(-sigma * mse)
    return reward * weight



# =========================================================================================
# 线速度跟踪奖励
def compute_vel_track_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    # 计算F_body朝向
    f_body_quat = asset.data.body_link_quat_w[:, _MODEL_INDICES.f_body_id]
    w, x, y, z = f_body_quat[:, 0], f_body_quat[:, 1], f_body_quat[:, 2], f_body_quat[:, 3]  # type: ignore[misc]
    sin_cosp = 2.0 * (w * z + x * y)
    cos_cosp = 1.0 - 2.0 * (y * y + z * z)
    f_body_heading = torch.atan2(sin_cosp, cos_cosp)
    # 计算F_body朝向速度误差
    vel_w = asset.data.root_link_lin_vel_w
    forward_speed = vel_w[:, 0] * torch.cos(f_body_heading) + vel_w[:, 1] * torch.sin(f_body_heading)
    cmd_term = env.command_manager._terms["slalom_cmd"]
    v_cmd = cmd_term.command[:, 0]
    error = forward_speed - v_cmd
    # 获取课程学习量
    weight = get_curriculum_reward_weight(env, "weight_track_vel")
    sigma = get_curriculum_reward_weight(env, "sigma_track_vel")
    # 计算奖励
    reward = torch.exp(-sigma * error ** 2)
    # 记录日志
    env.extras["log"]["Data/vel_actual"] = forward_speed.mean().item()
    return reward * weight



# =========================================================================================
# 角速度跟踪奖励（ω_cmd = κ_cmd × v_cmd）
def compute_omg_track_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    actual_omega_z = asset.data.root_link_ang_vel_w[:, 2]
    cmd_term = env.command_manager._terms["slalom_cmd"]
    # 从曲率和速度计算期望角速度: ω = κ × v
    curvature_cmd = cmd_term.command[:, 4]
    vel_cmd = cmd_term.command[:, 0]
    omega_cmd = curvature_cmd * vel_cmd
    error = actual_omega_z - omega_cmd
    sigma = get_curriculum_reward_weight(env, "sigma_track_omg")
    weight = get_curriculum_reward_weight(env, "weight_track_omg")
    reward = torch.exp(-sigma * error ** 2)
    env.extras["log"]["Data/omg_actual"] = actual_omega_z.mean().item()
    env.extras["log"]["Data/omg_cmd"] = omega_cmd.mean().item()
    return reward * weight


# =========================================================================================
# 脊柱转弯奖励（ω_cmd = κ_cmd × v_cmd）
def compute_spine_turn_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """R_spine = |ω_cmd| * (w_lat*|F_spine1| + w_twist*(|F_body|+|H_body|))"""
    asset: Entity = env.scene["robot"]
    cmd_term = env.command_manager._terms["slalom_cmd"]
    omega_cmd_abs = torch.abs(cmd_term.command[:, 4] * cmd_term.command[:, 0])

    joint_pos = asset.data.joint_pos[:, _MODEL_INDICES.joint_ids]
    lateral_pos = joint_pos[:, _MODEL_INDICES.actuator_spn_lateral_id]
    body_ids = _MODEL_INDICES.actuator_spn_body_ids
    twist_pos = joint_pos[:, body_ids[0]] + joint_pos[:, body_ids[1]]

    spine_activity = 2.5 * torch.abs(lateral_pos) + 1.0 * torch.abs(twist_pos)
    reward = omega_cmd_abs * spine_activity
    weight = get_curriculum_reward_weight(env, "weight_spine_turn")
    env.extras["log"]["Data/spine_lateral_abs"] = torch.abs(lateral_pos).mean().item()
    return reward * weight


# =========================================================================================
# 稳定性惩罚
def compute_stability_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """R_stab = -(roll^2 + pitch^2)"""
    asset: Entity = env.scene["robot"]
    gravity_b = asset.data.projected_gravity_b
    tilt_sq = torch.sum(torch.square(gravity_b[:, :2]), dim=1)
    weight = get_curriculum_reward_weight(env, "weight_stability")
    return -tilt_sq * weight



# =========================================================================================
# L1 动作平滑惩罚
def compute_action_L1_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 计算动作变化
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    abs_diff = torch.abs(current_action - prev_action)
    leg_cost = torch.sum(abs_diff[:, _MODEL_INDICES.actuator_leg_ids], dim=1)
    spn_cost = torch.sum(abs_diff[:, _MODEL_INDICES.actuator_spn_ids], dim=1)
    # 获取课程学习量
    w_leg = get_curriculum_reward_weight(env, "weight_smooth_L1_leg")
    w_spn = get_curriculum_reward_weight(env, "weight_smooth_L1_spn")
    # 计算奖励
    penalty = -w_leg * leg_cost - w_spn * spn_cost
    return penalty



# =========================================================================================
# L2 动作平滑惩罚
def compute_action_L2_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 计算动作变化
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    sq_diff = torch.square(current_action - prev_action)
    leg_cost = torch.sum(sq_diff[:, _MODEL_INDICES.actuator_leg_ids], dim=1)
    spn_cost = torch.sum(sq_diff[:, _MODEL_INDICES.actuator_spn_ids], dim=1)
    # 获取课程学习量
    w_leg = get_curriculum_reward_weight(env, "weight_smooth_L2_leg")
    w_spn = get_curriculum_reward_weight(env, "weight_smooth_L2_spn")
    # 计算奖励
    penalty = -w_leg * leg_cost - w_spn * spn_cost
    return penalty



# =========================================================================================
# 能耗惩罚
def compute_energy_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    # 计算能量消耗
    actuator_vel = asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
    actuator_torque = asset.data.actuator_force
    cost = torch.sum(torch.abs(actuator_vel * actuator_torque), dim=1)
    # 获取课程学习量
    weight = get_curriculum_reward_weight(env, "weight_energy")
    # 计算奖励
    penalty = -weight * cost
    return penalty