from __future__ import annotations
import torch
from mjlab.entity import Entity
from typing import TYPE_CHECKING
from .indices import _MODEL_INDICES
from .observations import _compute_path_ref, _get_f_body_heading
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
    error_leg = error[:, _MODEL_INDICES.actuator_leg_ids]
    error_spn = error[:, _MODEL_INDICES.actuator_spn_ids]
    # 获取课程学习量
    weight = get_curriculum_reward_weight(env, "weight_mimic_pos")
    sigma_leg = get_curriculum_reward_weight(env, "sigma_leg_pos")
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_pos")
    # 计算奖励
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward = (reward_leg + reward_spn) / 2
    return reward * weight



# =========================================================================================
# 关节速度模仿奖励
def compute_mimic_vel_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    # 计算关节速度误差
    joint_vel = asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
    _, ref_vel = get_reference_joint_state(env)
    error = joint_vel - ref_vel
    error_leg = error[:, _MODEL_INDICES.actuator_leg_ids]
    error_spn = error[:, _MODEL_INDICES.actuator_spn_ids]
    # 获取课程学习量
    weight = get_curriculum_reward_weight(env, "weight_mimic_vel")
    sigma_leg = get_curriculum_reward_weight(env, "sigma_leg_vel")
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_vel")
    # 计算奖励
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward = (reward_leg + reward_spn) / 2
    return reward * weight



# =========================================================================================
# 线速度跟踪奖励 — 世界坐标系，直接比较期望 vs 实际速度
def compute_vel_track_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    # 期望世界系速度（路径切线方向）
    _, _, vx_des, vy_des, _ = _compute_path_ref(env)
    # 实际世界系速度（F_body）
    vel_w = asset.data.body_link_lin_vel_w[:, _MODEL_INDICES.f_body_id, :]  # [N, 3]
    vx_actual = vel_w[:, 0]
    vy_actual = vel_w[:, 1]
    vz_actual = vel_w[:, 2]
    # 误差
    error_x = vx_actual - vx_des
    error_y = vy_actual - vy_des
    error_z = vz_actual
    weight = get_curriculum_reward_weight(env, "weight_track_vel")
    weight_yz = get_curriculum_reward_weight(env, "weight_track_vyz")
    sigma = get_curriculum_reward_weight(env, "sigma_track_vel")
    sigma_yz = get_curriculum_reward_weight(env, "sigma_track_vyz")
    r_vel_x = torch.exp(-sigma * error_x ** 2)
    r_vel_y = torch.exp(-sigma_yz * error_y ** 2)
    r_vel_z = torch.exp(-sigma_yz * error_z ** 2)
    env.extras["log"]["Data/vel_actual"] = vx_actual.mean().item()
    env.extras["log"]["Data/vel_des"] = vx_des.mean().item()
    return r_vel_x * weight + (r_vel_y + r_vel_z) * weight_yz



# =========================================================================================
# 角速度跟踪奖励（ω_cmd = κ_cmd × v_cmd, 用 F_body 角速度避免万向节中心震荡）
def compute_omg_track_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    # F_body_Link 世界系角速度 Z 分量 → 前体偏航率
    actual_omega_z = asset.data.body_link_ang_vel_w[:, _MODEL_INDICES.f_body_id, 2]  # type: ignore[call-overload]
    cmd_term = env.command_manager._terms["slalom_cmd"]
    curvature_cmd = cmd_term.command[:, 4]
    vel_cmd = cmd_term.command[:, 0]
    omega_cmd = curvature_cmd * vel_cmd
    error = actual_omega_z - omega_cmd
    # 获取课程学习量
    sigma = get_curriculum_reward_weight(env, "sigma_track_omg")
    weight = get_curriculum_reward_weight(env, "weight_track_omg")
    # 计算奖励
    reward = torch.exp(-sigma * error ** 2)
    # 记录日志
    env.extras["log"]["Data/omg_actual"] = actual_omega_z.mean().item()
    env.extras["log"]["Data/omg_cmd"] = omega_cmd.mean().item()
    return reward * weight



# =========================================================================================
# F_body 朝向跟踪奖励 — 实际 heading 对齐期望路径切线方向
def compute_head_track_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 计算朝向误差
    _, _, _, _, path_heading = _compute_path_ref(env)  # 期望朝向
    actual_heading = _get_f_body_heading(env) - (torch.pi / 2)  # type: ignore[call-arg]  # 实际物理前向
    error = actual_heading - path_heading
    error = torch.atan2(torch.sin(error), torch.cos(error))
    # 获取课程学习量
    sigma = get_curriculum_reward_weight(env, "sigma_track_head")
    weight = get_curriculum_reward_weight(env, "weight_track_head")
    # 计算奖励
    reward = torch.exp(-sigma * error ** 2)
    # 记录日志
    env.extras["log"]["Data/head_error"] = error.abs().mean().item()
    return reward * weight



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



# =========================================================================================
# 路径跟踪奖励
def compute_path_track_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]

    # 复用共享路径计算
    x_ref, y_ref, _, _, path_heading = _compute_path_ref(env)

    ref_xy = torch.stack([x_ref, y_ref], dim=1)  # [N, 2]
    tangent = torch.stack([torch.cos(path_heading), torch.sin(path_heading)], dim=1)

    body_offset = 0.065
    ref_f_body = ref_xy + body_offset * tangent
    ref_h_body = ref_xy - body_offset * tangent

    # 当前身体环节位置
    base_xy = asset.data.root_link_pos_w[:, :2]
    f_body_xy = asset.data.body_link_pos_w[:, _MODEL_INDICES.f_body_id, :2]
    h_body_xy = asset.data.body_link_pos_w[:, _MODEL_INDICES.h_body_id, :2]
    # 跟踪误差
    error_base = torch.norm(base_xy - ref_xy, dim=1)
    error_fbody = torch.norm(f_body_xy - ref_f_body, dim=1)
    error_hbody = torch.norm(h_body_xy - ref_h_body, dim=1)
    # 课程权重
    weight = get_curriculum_reward_weight(env, "weight_track_path")
    sigma  = get_curriculum_reward_weight(env, "sigma_track_path")
    # 计算奖励
    r_base  = torch.exp(-sigma * error_base ** 2)
    r_fbody = torch.exp(-sigma * error_fbody ** 2)
    r_hbody = torch.exp(-sigma * error_hbody ** 2)
    reward  = (r_base + r_fbody + r_hbody)/3
    # 记录日志
    env.extras["log"]["Data/path_error"] = ((error_base+error_fbody+error_hbody)/3).mean().item()
    return reward * weight

