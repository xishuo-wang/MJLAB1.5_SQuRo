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
    joint_pos = asset.data.joint_pos[:, _MODEL_INDICES.joint_ids]
    ref_pos, _ = get_reference_joint_state(env)
    error = joint_pos - ref_pos
    mse = torch.mean(error ** 2, dim=1)
    sigma = get_curriculum_reward_weight(env, "sigma_mimic_pos")
    reward = torch.exp(-sigma * mse)
    weight = get_curriculum_reward_weight(env, "weight_mimic_pos")
    env.extras["log"]["Reward/mimic_pos"] = reward.mean().item()
    return reward * weight



# =========================================================================================
# 关节速度模仿奖励
def compute_mimic_vel_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    joint_vel = asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
    _, ref_vel = get_reference_joint_state(env)
    error = joint_vel - ref_vel
    mse = torch.mean(error ** 2, dim=1)
    sigma = get_curriculum_reward_weight(env, "sigma_mimic_vel")
    reward = torch.exp(-sigma * mse)
    weight = get_curriculum_reward_weight(env, "weight_mimic_vel")
    env.extras["log"]["Reward/mimic_vel"] = reward.mean().item()
    return reward * weight



# =========================================================================================
# 线速度跟踪奖励（机身前进方向投影，转弯时不受朝向影响）
def compute_vel_track_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """R_vel = exp(-sigma * (v_forward - v_cmd)^2)
    v_forward = vel_w · [cos(heading), sin(heading), 0] — 世界速度投影到机身前进方向"""
    asset: Entity = env.scene["robot"]
    vel_w = asset.data.root_link_lin_vel_w  # [N, 3] 世界系
    heading = asset.data.heading_w           # [N] 机身偏航角

    # 投影到机身前进方向（heading=0 → +X, heading=π/2 → +Y）
    forward_speed = vel_w[:, 0] * torch.cos(heading) + vel_w[:, 1] * torch.sin(heading)

    cmd_term = env.command_manager._terms["slalom_cmd"]
    v_cmd = cmd_term.command[:, 0]

    error_vel = forward_speed - v_cmd
    sigma_under = 100.0
    sigma_over = 50.0
    sigma = torch.where(error_vel < 0, sigma_under, sigma_over)
    reward = torch.exp(-sigma * error_vel ** 2)
    weight = get_curriculum_reward_weight(env, "weight_track_vel")

    env.extras["log"]["Data/forward_speed"] = forward_speed.mean().item()
    env.extras["log"]["Data/vel_x_cmd"] = v_cmd.mean().item()
    env.extras["log"]["Reward/vel_track"] = reward.mean().item()
    return reward * weight



# =========================================================================================
# 角速度跟踪奖励
def compute_omega_track_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    actual_omega_z = asset.data.root_link_ang_vel_w[:, 2]
    cmd_term = env.command_manager._terms["slalom_cmd"]
    omega_cmd = cmd_term.command[:, 4]

    error_omega = actual_omega_z - omega_cmd
    reward = torch.exp(-50.0 * error_omega ** 2)
    weight = get_curriculum_reward_weight(env, "weight_track_omega")

    env.extras["log"]["Data/omega_z_actual"] = actual_omega_z.mean().item()
    env.extras["log"]["Reward/omega_track"] = reward.mean().item()
    return reward * weight



# =========================================================================================
# 脊柱转弯奖励
def compute_spine_turn_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    cmd_term = env.command_manager._terms["slalom_cmd"]
    omega_cmd_abs = torch.abs(cmd_term.command[:, 4])

    joint_pos = asset.data.joint_pos[:, _MODEL_INDICES.joint_ids]
    lateral_pos = joint_pos[:, _MODEL_INDICES.actuator_spn_lateral_id]
    body_ids = _MODEL_INDICES.actuator_spn_body_ids
    twist_pos = joint_pos[:, body_ids[0]] + joint_pos[:, body_ids[1]]

    spine_activity = 2.5 * torch.abs(lateral_pos) + 1.0 * torch.abs(twist_pos)
    reward = omega_cmd_abs * spine_activity
    weight = get_curriculum_reward_weight(env, "weight_spine_turn")

    env.extras["log"]["Data/spine_lateral_abs"] = torch.abs(lateral_pos).mean().item()
    env.extras["log"]["Reward/spine_turn"] = reward.mean().item()
    return reward * weight



# =========================================================================================
# 稳定性惩罚
def compute_stability_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    gravity_b = asset.data.projected_gravity_b
    tilt_sq = torch.sum(torch.square(gravity_b[:, :2]), dim=1)
    weight = get_curriculum_reward_weight(env, "weight_stability")
    return -tilt_sq * weight



# =========================================================================================
# L1 动作平滑惩罚（腿/脊柱分离权重）
def compute_action_L1_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    abs_diff = torch.abs(current_action - prev_action)

    leg_cost = torch.sum(abs_diff[:, _MODEL_INDICES.actuator_leg_ids], dim=1)
    spn_cost = torch.sum(abs_diff[:, _MODEL_INDICES.actuator_spn_ids], dim=1)

    w_leg = get_curriculum_reward_weight(env, "weight_smooth_L1_leg")
    w_spn = get_curriculum_reward_weight(env, "weight_smooth_L1_spn")
    return -w_leg * leg_cost - w_spn * spn_cost



# =========================================================================================
# L2 动作平滑惩罚
def compute_action_L2_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    sq_diff = torch.square(current_action - prev_action)

    leg_cost = torch.sum(sq_diff[:, _MODEL_INDICES.actuator_leg_ids], dim=1)
    spn_cost = torch.sum(sq_diff[:, _MODEL_INDICES.actuator_spn_ids], dim=1)

    w_leg = get_curriculum_reward_weight(env, "weight_smooth_L2_leg")
    w_spn = get_curriculum_reward_weight(env, "weight_smooth_L2_spn")
    return -w_leg * leg_cost - w_spn * spn_cost



# =========================================================================================
# 能耗惩罚
def compute_energy_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    actuator_vel = asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
    actuator_torque = asset.data.actuator_force
    power = torch.sum(torch.abs(actuator_vel * actuator_torque), dim=1)
    weight = get_curriculum_reward_weight(env, "weight_energy")
    return -weight * power