from __future__ import annotations
import torch
from mjlab.entity import Entity
from typing import TYPE_CHECKING
from .indices import _MODEL_INDICES
from .path import (
    BODY_REF_OFFSET,
    CORRIDOR_HALF_WIDTH,
    F_BODY_HALF_LENGTH,
    F_BODY_HALF_WIDTH,
    H_BODY_HALF_LENGTH,
    H_BODY_HALF_WIDTH,
    compute_body_pole_dist,
    compute_corridor_excess,
    compute_leg_pole_dist,
    compute_path_ref,
    get_f_body_physical_heading,
    get_h_body_physical_heading,
)
from .pole import POLE_Y
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
# 线速度跟踪奖励 — F_body 局部坐标系（body-frame 前进/侧向/垂向）
def compute_vel_track_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    # F_body 物理前向 heading（body+X - π/2 → world+X = 0°）
    f_body_heading = get_f_body_physical_heading(env)
    # F_body_Link 世界系速度
    vel_w = asset.data.body_link_lin_vel_w[:, _MODEL_INDICES.f_body_id, :]  # [N, 3]
    # 投影到 F_body 局部坐标系：前进=物理前向，侧向=物理左向
    forward_speed = vel_w[:, 0] * torch.cos(f_body_heading) + vel_w[:, 1] * torch.sin(f_body_heading)
    lateral_speed = -vel_w[:, 0] * torch.sin(f_body_heading) + vel_w[:, 1] * torch.cos(f_body_heading)
    vertical_speed = vel_w[:, 2]
    # 期望前进速度
    cmd_term = env.command_manager._terms["slalom_cmd"]
    v_cmd = cmd_term.command[:, 0]
    # 误差
    error_vx = forward_speed - v_cmd
    error_vy = lateral_speed
    error_vz = vertical_speed
    # 获取课程学习量
    weight = get_curriculum_reward_weight(env, "weight_track_vel")
    weight_yz = get_curriculum_reward_weight(env, "weight_track_vyz")
    sigma = get_curriculum_reward_weight(env, "sigma_track_vel")
    sigma_yz = get_curriculum_reward_weight(env, "sigma_track_vyz")
    # 计算奖励
    r_vel_x = torch.exp(-sigma * error_vx ** 2)
    r_vel_y = torch.exp(-sigma_yz * error_vy ** 2)
    r_vel_z = torch.exp(-sigma_yz * error_vz ** 2)
    # 记录日志
    env.extras["log"]["Data/vel_actual"] = forward_speed.mean().item()
    env.extras["log"]["Data/vel_des"] = v_cmd.mean().item()
    return r_vel_x * weight + (r_vel_y + r_vel_z) * weight_yz



# =========================================================================================
# 身体高度跟踪奖励 
def compute_height_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    # 计算高度跟踪误差
    body_pos_w = asset.data.body_link_pos_w
    F_body_height = body_pos_w[:, _MODEL_INDICES.f_body_id, 2]
    H_body_height = body_pos_w[:, _MODEL_INDICES.h_body_id, 2]          
    cmd_term = env.command_manager._terms["slalom_cmd"]
    desired_height_F = cmd_term.command[:, 1]    
    desired_height_H = cmd_term.command[:, 2]    
    height_F_error = torch.abs(desired_height_F - F_body_height)
    height_H_error = torch.abs(desired_height_H - H_body_height)
    # 获取课程学习量
    sigma_height = get_curriculum_reward_weight(env, "sigma_height")
    w_height = get_curriculum_reward_weight(env, "weight_height")
    # 计算奖励
    r_height_F = torch.exp(-sigma_height * height_F_error ** 2)
    r_height_H = torch.exp(-sigma_height * height_H_error ** 2)   
    Reward_height = w_height * (0.5 * r_height_F + 0.5 * r_height_H)
    # 记录日志
    env.extras["log"]["Data/height_actual"] = (0.5 * F_body_height + 0.5 * H_body_height).mean().item()
    return Reward_height



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
    _, _, _, _, path_heading = compute_path_ref(env)  # 期望朝向
    actual_heading = get_f_body_physical_heading(env)  # 实际物理前向
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
# 走廊一致性奖励 — 身体包络不超出参考路径周围的允许走廊
# e_i = |d_lat| + |L·sin(Δθ)| + |W·cos(Δθ)|, v_i = max(0, e_i - C)
# 同时惩罚位置偏差和朝向偏差，死区 = 走廊半宽 C
def compute_corridor_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]

    # 路径参考
    x_ref, y_ref, _, _, path_heading = compute_path_ref(env)
    ref_xy = torch.stack([x_ref, y_ref], dim=1)  # [N, 2]
    tangent = torch.stack([torch.cos(path_heading), torch.sin(path_heading)], dim=1)

    # F_body/H_body 参考位置（偏移 ±6.5cm 沿切线）
    ref_f = ref_xy + BODY_REF_OFFSET * tangent
    ref_h = ref_xy - BODY_REF_OFFSET * tangent

    # 当前身体 XY 位置
    f_xy = asset.data.body_link_pos_w[:, _MODEL_INDICES.f_body_id, :2]
    h_xy = asset.data.body_link_pos_w[:, _MODEL_INDICES.h_body_id, :2]

    # 当前物理前向 heading
    f_heading = get_f_body_physical_heading(env)
    h_heading = get_h_body_physical_heading(env)

    # 走廊超额 e_i
    e_f = compute_corridor_excess(f_xy, f_heading, ref_f, path_heading,
                                   F_BODY_HALF_LENGTH, F_BODY_HALF_WIDTH)
    e_h = compute_corridor_excess(h_xy, h_heading, ref_h, path_heading,
                                   H_BODY_HALF_LENGTH, H_BODY_HALF_WIDTH)

    # 超出走廊的量
    v_f = torch.clamp(e_f - CORRIDOR_HALF_WIDTH, min=0.0)
    v_h = torch.clamp(e_h - CORRIDOR_HALF_WIDTH, min=0.0)

    # 课程权重
    sigma = get_curriculum_reward_weight(env, "sigma_corridor")
    weight = get_curriculum_reward_weight(env, "weight_corridor")
    r_f = torch.exp(-sigma * v_f ** 2)
    r_h = torch.exp(-sigma * v_h ** 2)
    reward = (r_f + r_h) / 2

    # 日志
    env.extras["log"]["Data/corridor_excess"] = ((v_f + v_h) / 2).mean().item()
    env.extras["log"]["Data/corridor_e_f"] = e_f.mean().item()
    env.extras["log"]["Data/corridor_e_h"] = e_h.mean().item()
    return reward * weight


# =========================================================================================
# 虚拟碰撞惩罚 — 躯干(矩形 vs 杆圆) + 腿(线段 vs 杆圆)
# 平滑指数惩罚: 碰撞区强惩罚, 接近区软梯度, 安全区零影响
def compute_virtual_collision_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    cmd_term = env.command_manager._terms["slalom_cmd"]
    spacing = cmd_term.active_pole_spacing  # type: ignore[union-attr]

    # 杆位: 6 根杆沿 +X, Y = POLE_Y
    pole_pos = torch.stack([
        torch.tensor([i * spacing, POLE_Y], device=env.device)
        for i in range(6)
    ])  # [6, 2]

    # F_body / H_body XY
    f_xy = asset.data.body_link_pos_w[:, _MODEL_INDICES.f_body_id, :2]
    h_xy = asset.data.body_link_pos_w[:, _MODEL_INDICES.h_body_id, :2]
    f_heading = get_f_body_physical_heading(env)
    h_heading = get_h_body_physical_heading(env)

    # 躯干碰撞距离 (到最近杆)
    d_body_f = compute_body_pole_dist(f_xy, f_heading, F_BODY_HALF_LENGTH, F_BODY_HALF_WIDTH, pole_pos)
    d_body_h = compute_body_pole_dist(h_xy, h_heading, H_BODY_HALF_LENGTH, H_BODY_HALF_WIDTH, pole_pos)

    # 腿碰撞距离 (4条腿)
    foot_xy = asset.data.site_pos_w[:, _MODEL_INDICES.foot_site_ids, :2]  # [N, 4, 2]
    # foot_site_ids order: FL_elbow, FR_elbow, HL_knee, HR_knee → 0=FL,1=FR,2=HL,3=HR
    d_leg_fl = compute_leg_pole_dist(f_xy, f_heading, F_BODY_HALF_WIDTH, -1, foot_xy[:, 0, :], pole_pos)
    d_leg_fr = compute_leg_pole_dist(f_xy, f_heading, F_BODY_HALF_WIDTH, +1, foot_xy[:, 1, :], pole_pos)
    d_leg_hl = compute_leg_pole_dist(h_xy, h_heading, H_BODY_HALF_WIDTH, -1, foot_xy[:, 2, :], pole_pos)
    d_leg_hr = compute_leg_pole_dist(h_xy, h_heading, H_BODY_HALF_WIDTH, +1, foot_xy[:, 3, :], pole_pos)

    # 渗透量: max(0, -dist) — 仅在碰撞时非零
    body_pen = torch.clamp(-d_body_f, min=0) + torch.clamp(-d_body_h, min=0)
    leg_pen = (torch.clamp(-d_leg_fl, min=0) + torch.clamp(-d_leg_fr, min=0) +
               torch.clamp(-d_leg_hl, min=0) + torch.clamp(-d_leg_hr, min=0))

    sigma = get_curriculum_reward_weight(env, "sigma_collision")
    w_body = get_curriculum_reward_weight(env, "weight_collision_body")
    w_leg  = get_curriculum_reward_weight(env, "weight_collision_leg")
    penalty = -w_body * torch.expm1(-sigma * body_pen) - w_leg * torch.expm1(-sigma * leg_pen)

    env.extras["log"]["Data/coll_body"] = body_pen.mean().item()
    env.extras["log"]["Data/coll_leg"] = leg_pen.mean().item()
    return penalty

