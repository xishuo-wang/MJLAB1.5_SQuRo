from __future__ import annotations
import torch
from mjlab.entity import Entity
from typing import TYPE_CHECKING
from .indices import _MODEL_INDICES
from .path import (
    compute_corridor_front_excess,
    compute_corridor_rear_excess,
    get_f_body_physical_heading,
)
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
    error_leg = error[:, _MODEL_INDICES.actuator_leg_ids]
    error_spn = error[:, _MODEL_INDICES.actuator_spn_ids]
    error_neck = error[:, _MODEL_INDICES.actuator_neck_ids]
    weight = get_curriculum_reward_weight(env, "weight_mimic_pos")
    sigma_leg = get_curriculum_reward_weight(env, "sigma_leg_pos")
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_pos")
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    mse_neck = torch.mean(error_neck ** 2, dim=1)
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward_neck = torch.exp(-sigma_spn * mse_neck)
    reward = (reward_leg + reward_spn) / 2 + 0.3 * reward_neck
    return reward * weight


# =========================================================================================
# 关节速度模仿奖励
def compute_mimic_vel_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    joint_vel = asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
    _, ref_vel = get_reference_joint_state(env)
    error = joint_vel - ref_vel
    error_leg = error[:, _MODEL_INDICES.actuator_leg_ids]
    error_spn = error[:, _MODEL_INDICES.actuator_spn_ids]
    error_neck = error[:, _MODEL_INDICES.actuator_neck_ids]
    weight = get_curriculum_reward_weight(env, "weight_mimic_vel")
    sigma_leg = get_curriculum_reward_weight(env, "sigma_leg_vel")
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_vel")
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    mse_neck = torch.mean(error_neck ** 2, dim=1)
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward_neck = torch.exp(-sigma_spn * mse_neck)
    reward = (reward_leg + reward_spn) / 2 + 0.3 * reward_neck
    return reward * weight


# =========================================================================================
# 线速度跟踪奖励 — F_body 局部坐标系 (body-frame 前进/侧向/垂向)
def compute_vel_track_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    f_body_heading = get_f_body_physical_heading(env)
    vel_w = asset.data.body_link_lin_vel_w[:, _MODEL_INDICES.f_body_id, :]  # [N, 3]
    forward_speed = vel_w[:, 0] * torch.cos(f_body_heading) + vel_w[:, 1] * torch.sin(f_body_heading)
    lateral_speed = -vel_w[:, 0] * torch.sin(f_body_heading) + vel_w[:, 1] * torch.cos(f_body_heading)
    vertical_speed = vel_w[:, 2]
    cmd_term = env.command_manager._terms["tunnel_cmd"]
    v_cmd = cmd_term.command[:, 0]
    error_vx = forward_speed - v_cmd
    error_vy = lateral_speed
    error_vz = vertical_speed
    weight = get_curriculum_reward_weight(env, "weight_track_vel")
    weight_yz = get_curriculum_reward_weight(env, "weight_track_vyz")
    sigma = get_curriculum_reward_weight(env, "sigma_track_vel")
    sigma_yz = get_curriculum_reward_weight(env, "sigma_track_vyz")
    r_vel_x = torch.exp(-sigma * error_vx ** 2)
    r_vel_y = torch.exp(-sigma_yz * error_vy ** 2)
    r_vel_z = torch.exp(-sigma_yz * error_vz ** 2)
    env.extras["log"]["Data/vel_actual"] = forward_speed.mean().item()
    env.extras["log"]["Data/vel_des"] = v_cmd.mean().item()
    return r_vel_x * weight + (r_vel_y + r_vel_z) * weight_yz


# =========================================================================================
# 身体高度跟踪奖励 (前/后肢, 命令高度 = 正常高度)
def compute_height_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    body_pos_w = asset.data.body_link_pos_w
    F_body_height = body_pos_w[:, _MODEL_INDICES.f_body_id, 2]
    H_body_height = body_pos_w[:, _MODEL_INDICES.h_body_id, 2]
    cmd_term = env.command_manager._terms["tunnel_cmd"]
    desired_height_F = cmd_term.command[:, 1]
    desired_height_H = cmd_term.command[:, 2]
    height_F_error = torch.abs(desired_height_F - F_body_height)
    height_H_error = torch.abs(desired_height_H - H_body_height)
    sigma_height = get_curriculum_reward_weight(env, "sigma_height")
    w_height = get_curriculum_reward_weight(env, "weight_height")
    r_height_F = torch.exp(-sigma_height * height_F_error ** 2)
    r_height_H = torch.exp(-sigma_height * height_H_error ** 2)
    Reward_height = w_height * (0.5 * r_height_F + 0.5 * r_height_H)
    env.extras["log"]["Data/height_actual"] = (0.5 * F_body_height + 0.5 * H_body_height).mean().item()
    return Reward_height


# =========================================================================================
# 朝向跟踪奖励 — F_body 物理前向对齐直行方向 (0°)
def compute_head_track_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    actual_heading = get_f_body_physical_heading(env)
    error = torch.atan2(torch.sin(actual_heading), torch.cos(actual_heading))
    sigma = get_curriculum_reward_weight(env, "sigma_track_head")
    weight = get_curriculum_reward_weight(env, "weight_track_head")
    reward = torch.exp(-sigma * error ** 2)
    env.extras["log"]["Data/head_error"] = error.abs().mean().item()
    return reward * weight


# =========================================================================================
# 走廊一致性奖励 — 前肢走廊(头部+前躯干) + 后肢走廊(后躯干)
def compute_corridor_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    v_f = compute_corridor_front_excess(env)   # 前肢组超额
    v_h = compute_corridor_rear_excess(env)    # 后肢组超额
    sigma = get_curriculum_reward_weight(env, "sigma_corridor")
    weight = get_curriculum_reward_weight(env, "weight_corridor")
    r_f = torch.exp(-sigma * v_f ** 2)
    r_h = torch.exp(-sigma * v_h ** 2)
    reward = (r_f + r_h) / 2
    env.extras["log"]["Data/corridor_front_excess"] = v_f.mean().item()
    env.extras["log"]["Data/corridor_rear_excess"] = v_h.mean().item()
    return reward * weight


# =========================================================================================
# L1 动作平滑惩罚
def compute_action_L1_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    abs_diff = torch.abs(current_action - prev_action)
    leg_cost = torch.sum(abs_diff[:, _MODEL_INDICES.actuator_leg_ids], dim=1)
    spn_cost = torch.sum(abs_diff[:, _MODEL_INDICES.actuator_spn_ids], dim=1)
    neck_cost = torch.sum(abs_diff[:, _MODEL_INDICES.actuator_neck_ids], dim=1)
    w_leg = get_curriculum_reward_weight(env, "weight_smooth_L1_leg")
    w_spn = get_curriculum_reward_weight(env, "weight_smooth_L1_spn")
    return -w_leg * leg_cost - w_spn * spn_cost - w_spn * neck_cost


# =========================================================================================
# L2 动作平滑惩罚
def compute_action_L2_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    sq_diff = torch.square(current_action - prev_action)
    leg_cost = torch.sum(sq_diff[:, _MODEL_INDICES.actuator_leg_ids], dim=1)
    spn_cost = torch.sum(sq_diff[:, _MODEL_INDICES.actuator_spn_ids], dim=1)
    neck_cost = torch.sum(sq_diff[:, _MODEL_INDICES.actuator_neck_ids], dim=1)
    w_leg = get_curriculum_reward_weight(env, "weight_smooth_L2_leg")
    w_spn = get_curriculum_reward_weight(env, "weight_smooth_L2_spn")
    return -w_leg * leg_cost - w_spn * spn_cost - w_spn * neck_cost


# =========================================================================================
# 能耗惩罚
def compute_energy_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    actuator_vel = asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
    actuator_torque = asset.data.actuator_force
    cost = torch.sum(torch.abs(actuator_vel * actuator_torque), dim=1)
    weight = get_curriculum_reward_weight(env, "weight_energy")
    return -weight * cost
