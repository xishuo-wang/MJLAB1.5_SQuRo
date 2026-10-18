from __future__ import annotations
import torch
from mjlab.entity import Entity
from typing import TYPE_CHECKING
from .indices import (
    _MODEL_INDICES, _ACT_JOINT_NAMES, _ACT_LEG_IDS, _ACT_SPN_IDS, _ACT_NECK_IDS,
    _ACT_SHOULDER_IDS, _ACT_HIP_IDS,
)
from .path import (
    CORRIDOR_HALF_WIDTH,
    compute_path_ref,
    get_f_body_physical_heading,
    get_path_curvature,
)
from .reference import get_reference_joint_state
from .curriculums import get_curriculum_reward_weight
from .corridor import compute_phase0_corridor_excess, compute_phase1_corridor_excess
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
    error_leg = error[:, _ACT_LEG_IDS]
    error_spn = error[:, _ACT_SPN_IDS]
    error_neck = error[:, _ACT_NECK_IDS]
    # 获取课程学习量
    weight = get_curriculum_reward_weight(env, "weight_mimic_pos")
    sigma_leg = get_curriculum_reward_weight(env, "sigma_leg_pos")
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_pos")
    # 计算奖励
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
    # 计算关节速度误差
    joint_vel = asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
    _, ref_vel = get_reference_joint_state(env)
    error = joint_vel - ref_vel
    error_leg = error[:, _ACT_LEG_IDS]
    error_spn = error[:, _ACT_SPN_IDS]
    error_neck = error[:, _ACT_NECK_IDS]
    # 获取课程学习量
    weight = get_curriculum_reward_weight(env, "weight_mimic_vel")
    sigma_leg = get_curriculum_reward_weight(env, "sigma_leg_vel")
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_vel")
    # 计算奖励
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    mse_neck = torch.mean(error_neck ** 2, dim=1)
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward_neck = torch.exp(-sigma_spn * mse_neck)
    reward = (reward_leg + reward_spn) / 2 + 0.3 * reward_neck
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
    # 计算速度跟踪误差
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
    leg_cost = torch.sum(abs_diff[:, _ACT_LEG_IDS], dim=1)
    spn_cost = torch.sum(abs_diff[:, _ACT_SPN_IDS], dim=1)
    error_cost = torch.sum(abs_diff[:, _ACT_NECK_IDS], dim=1)
    # 获取课程学习量
    w_leg = get_curriculum_reward_weight(env, "weight_smooth_L1_leg")
    w_spn = get_curriculum_reward_weight(env, "weight_smooth_L1_spn")
    # 计算奖励
    penalty = -w_leg * leg_cost - w_spn * spn_cost - w_spn * error_cost
    return penalty



# =========================================================================================
# L2 动作平滑惩罚
def compute_action_L2_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 计算动作变化
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    sq_diff = torch.square(current_action - prev_action)
    leg_cost = torch.sum(sq_diff[:, _ACT_LEG_IDS], dim=1)
    spn_cost = torch.sum(sq_diff[:, _ACT_SPN_IDS], dim=1)
    error_cost = torch.sum(sq_diff[:, _ACT_NECK_IDS], dim=1)
    # 获取课程学习量
    w_leg = get_curriculum_reward_weight(env, "weight_smooth_L2_leg")
    w_spn = get_curriculum_reward_weight(env, "weight_smooth_L2_spn")
    # 计算奖励
    penalty = -w_leg * leg_cost - w_spn * spn_cost - w_spn * error_cost
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
def compute_corridor_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 一阶段沿参考圆环，二阶段沿实际空间轨迹检查完整躯干包络。
    x_ref, y_ref, _, _, path_heading = compute_path_ref(env)
    ref_xy = torch.stack([x_ref, y_ref], dim=1)
    curvature = get_path_curvature(env)
    inner, outer = compute_phase0_corridor_excess(env, ref_xy, path_heading, curvature)
    cmd = env.command_manager._terms["slalom_cmd"]
    phase1_ids = cmd.phase1_mask.nonzero(as_tuple=False).flatten()
    if phase1_ids.numel():
        inner[phase1_ids], outer[phase1_ids] = compute_phase1_corridor_excess(env, phase1_ids)
    inner_f, inner_h = inner.unbind(dim=1)
    outer_f, outer_h = outer.unbind(dim=1)
    v_f, v_h = torch.maximum(inner, outer).unbind(dim=1)
    e_f, e_h = CORRIDOR_HALF_WIDTH + v_f, CORRIDOR_HALF_WIDTH + v_h
    # 获取课程学习量
    sigma = get_curriculum_reward_weight(env, "sigma_corridor")
    weight = get_curriculum_reward_weight(env, "weight_corridor")
    inner_multiplier = get_curriculum_reward_weight(env, "corridor_inner_multiplier")
    # 计算奖励
    r_f = torch.exp(-sigma * (outer_f ** 2 + inner_multiplier * inner_f ** 2))
    r_h = torch.exp(-sigma * (outer_h ** 2 + inner_multiplier * inner_h ** 2))
    reward = (r_f + r_h) / 2
    # 记录日志
    env.extras["log"]["Data/corridor_excess"] = ((v_f + v_h) / 2).mean().item()
    env.extras["log"]["Data/corridor_e_f"] = e_f.mean().item()
    env.extras["log"]["Data/corridor_e_h"] = e_h.mean().item()
    env.extras["log"]["Data/走廊内侧越界"] = ((inner_f + inner_h) / 2).mean().item()
    env.extras["log"]["Data/走廊外侧越界"] = ((outer_f + outer_h) / 2).mean().item()
    env.extras["log"]["Data/走廊误差尺度"] = sigma ** -0.5
    return reward * weight


# 腿部执行器饱和软惩罚：接近力矩上限的持续占用才计罚, 详见 docs/SQuRo_Slalom_碰撞课程.md
def compute_leg_saturation_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    force = asset.data.actuator_force
    limit = getattr(env, "_leg_saturation_limit", None)
    if limit is None:
        ctrl = asset.indexing.ctrl_ids.cpu().numpy()
        values = env.sim.mj_model.actuator_forcerange[ctrl, 1]
        limit = torch.tensor(values, device=env.device, dtype=torch.float32)
        env._leg_saturation_limit = limit
    ratio = (force.abs() / limit.clamp_min(1e-6)).clamp(max=1.0)
    threshold = get_curriculum_reward_weight(env, "leg_saturation_threshold")
    excess = ((ratio - threshold) / max(1.0 - threshold, 1e-6)).clamp(0.0, 1.0)
    worst = excess[:, _ACT_LEG_IDS].max(dim=1).values
    weight = get_curriculum_reward_weight(env, "weight_leg_saturation")
    env.extras["log"]["Data/腿部饱和占比"] = worst.mean().item()
    return -weight * worst


# 动作越界软惩罚：原始动作超出软限的部分计罚, 详见 docs/SQuRo_Slalom_碰撞课程.md
def compute_action_soft_limit_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    actions = env.action_manager.action
    soft = get_curriculum_reward_weight(env, "action_soft_limit")
    excess = (actions.abs() - soft).clamp(min=0.0).mean(dim=-1)
    weight = get_curriculum_reward_weight(env, "weight_action_soft_limit")
    env.extras["log"]["Data/动作越界均值"] = excess.mean().item()
    return -weight * excess


# 按参考相对角度检测前肩过抬、后髋过折，左右两侧使用同一容差。
def _abnormal_leg_pose_excess(position, reference, tolerance):
    error = position - reference
    deviation = torch.cat((error[:, _ACT_SHOULDER_IDS], -error[:, _ACT_HIP_IDS]), dim=1)
    return (deviation - tolerance).clamp_min(0.0).amax(dim=1)


# 同时处罚实际异常姿态与限幅前请求，标定依据见 docs/SQuRo_Slalom_碰撞课程.md。
def compute_abnormal_leg_pose_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    ref_pos, _ = get_reference_joint_state(env)
    asset = env.scene["robot"]
    action_term = env.action_manager.get_term("joint_pos")
    columns = getattr(env, "_abnormal_pose_action_columns", None)
    if columns is None:
        columns = tuple(action_term.target_names.index(name) for name in _ACT_JOINT_NAMES)
        env._abnormal_pose_action_columns = columns
    requested = action_term.raw_action * action_term.scale + action_term.offset
    requested = (requested - asset.data.encoder_bias[:, action_term.target_ids])[:, columns]
    actual = asset.data.joint_pos[:, _MODEL_INDICES.joint_ids]
    tolerance = get_curriculum_reward_weight(env, "abnormal_leg_pose_tolerance")
    mask = env.command_manager._terms["slalom_cmd"].phase1_mask
    actual_excess = _abnormal_leg_pose_excess(actual, ref_pos, tolerance) * mask
    request_excess = _abnormal_leg_pose_excess(requested, ref_pos, tolerance) * mask
    weight = get_curriculum_reward_weight(env, "weight_abnormal_leg_pose")
    env.extras["log"]["Data/异常腿部实际越界"] = actual_excess.mean().item()
    env.extras["log"]["Data/异常腿部请求越界"] = request_excess.mean().item()
    env.extras["log"]["Data/异常腿部姿态占比"] = (actual_excess > 0).float().mean().item()
    return -weight * (actual_excess.square() + request_excess.square())

