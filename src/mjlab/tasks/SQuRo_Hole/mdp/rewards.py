from __future__ import annotations
import re
import math
import torch
from mjlab.entity import Entity
from typing import TYPE_CHECKING
from .entity import hole_geometry
from .config import (
    CMD_VEL_X_IDS,
    CMD_ANGLE_IDS,
    CMD_HEIGHT_F_IDS,
    CMD_HEIGHT_H_IDS,
    THRESHOLD_HEIGHT,
)
from .curriculums import SPN_AXIS_SCALE, get_curriculum_reward_weight
from .indices import _MODEL_INDICES, _ACT_LEG_IDS, _ACT_NECK_IDS, _ACT_SPN_IDS
from .reference import (
    resolve_joint_ids,
    get_reference_joint_state,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv



# =========================================================================================
# 关节位置模仿奖励
def compute_mimic_pos_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取实际和期望关节位置
    asset: Entity = env.scene["robot"]
    joint_pos = asset.data.joint_pos[:, _MODEL_INDICES.joint_ids]
    ref_pos, _ = get_reference_joint_state(env)
    # 计算关节位置误差
    error = joint_pos - ref_pos
    error_leg = error[:, _ACT_LEG_IDS]
    error_spn = error[:, _ACT_SPN_IDS] * _error_spn_scale(env, error.device, error.dtype)
    error_neck = error[:, _ACT_NECK_IDS]
    # 获取课程学习量
    sigma_leg = get_curriculum_reward_weight(env, "sigma_leg_pos")
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_pos")
    sigma_neck = get_curriculum_reward_weight(env, "sigma_neck_pos")
    weight = get_curriculum_reward_weight(env, "weight_mimic_pos")
    # 计算 MSE 误差
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    mse_neck = torch.mean(error_neck ** 2, dim=1)
    # 计算奖励
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward_neck = torch.exp(-sigma_neck * mse_neck)
    reward = (reward_leg + reward_spn + reward_neck) / 3
    return reward * weight



# =========================================================================================
# 关节速度模仿奖励
def compute_mimic_vel_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取实际和期望关节速度
    asset: Entity = env.scene["robot"]
    joint_vel = asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
    _, ref_vel = get_reference_joint_state(env)
    # 计算关节速度误差
    error = joint_vel - ref_vel
    error_leg = error[:, _ACT_LEG_IDS]
    error_spn = error[:, _ACT_SPN_IDS] * _error_spn_scale(env, error.device, error.dtype)
    error_neck = error[:, _ACT_NECK_IDS]
    # 获取课程学习量
    sigma_leg = get_curriculum_reward_weight(env, "sigma_leg_vel")
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_vel")
    sigma_neck = get_curriculum_reward_weight(env, "sigma_neck_vel")
    weight = get_curriculum_reward_weight(env, "weight_mimic_vel")
    # 计算 MSE 误差
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    mse_neck = torch.mean(error_neck ** 2, dim=1)
    # 计算奖励
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward_neck = torch.exp(-sigma_neck * mse_neck)
    reward = (reward_leg + reward_spn + reward_neck) / 3
    return reward * weight



# =========================================================================================
# 速度跟踪奖励
def compute_vel_reward(env: ManagerBasedRlEnv) -> torch.Tensor:    
    # 获取实际和期望速度
    asset: Entity = env.scene["robot"]
    actual_vel_x = asset.data.root_link_lin_vel_w[:, 0]
    cmd_term = env.command_manager._terms["hole_cmd"]
    desired_vel_x = cmd_term.command[:, CMD_VEL_X_IDS]
    # 计算速度误差
    vel_error = torch.abs(desired_vel_x - actual_vel_x)
    # 获取课程学习量
    sigma_vel = get_curriculum_reward_weight(env, "sigma_vel")
    weight = get_curriculum_reward_weight(env, "weight_vel")  
    # 计算奖励
    reward = torch.exp(-sigma_vel * vel_error ** 2)
    # 记录日志
    env.extras["log"]["Data/vel_act"] = actual_vel_x.mean().item()
    env.extras["log"]["Data/vel_cmd"] = desired_vel_x.mean().item()
    return reward * weight



# =========================================================================================
# 身体高度跟踪奖励
def compute_height_reward1(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取实际和期望高度
    asset: Entity = env.scene["robot"]
    body_pos_w = asset.data.body_link_pos_w
    F_body_height = body_pos_w[:, _MODEL_INDICES.f_body_id, 2]
    H_body_height = body_pos_w[:, _MODEL_INDICES.h_body_id, 2]
    cmd_term = env.command_manager._terms["hole_cmd"]
    z_ref_F = cmd_term.command[:, CMD_HEIGHT_F_IDS]
    z_ref_H = cmd_term.command[:, CMD_HEIGHT_H_IDS]
    # 计算高度误差
    height_F_error = torch.abs(z_ref_F - F_body_height)
    height_H_error = torch.abs(z_ref_H - H_body_height)
    # 获取课程学习量
    sigma_height = get_curriculum_reward_weight(env, "sigma_height")
    weight = get_curriculum_reward_weight(env, "weight_height")
    # 计算奖励
    r_height_F = torch.exp(-sigma_height * height_F_error ** 2)
    r_height_H = torch.exp(-sigma_height * height_H_error ** 2)
    reward = 0.5 * r_height_F + 0.5 * r_height_H
    # 记录日志
    env.extras["log"]["Data/height_error"] = ((height_F_error + height_H_error) / 2).mean().item()
    return reward * weight






# 计算朝向奖励
def compute_orientation_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    quat = asset.data.root_link_quat_w 
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp) - math.pi/2
    yaw_error = torch.abs(yaw)
    yaw_error_alt = torch.min(yaw_error, 2 * math.pi - yaw_error)
    reward = torch.exp(-5 * yaw_error_alt ** 2)
    weight = get_curriculum_reward_weight(env, "orientation")
    return reward * weight



# 身体角度奖励
def compute_angle_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    cmd_term = env.command_manager._terms["hole_cmd"]
    desired_heightF = cmd_term.command[:, CMD_HEIGHT_F_IDS]  # 前肢高度命令
    desired_heightH = cmd_term.command[:, CMD_HEIGHT_H_IDS]  # 后肢高度命令
    desired_angle = cmd_term.command[:, CMD_ANGLE_IDS]       # 侧倾角度命令
    roll_error = torch.zeros(env.num_envs, device=env.device)
    sigma = 50

    # 情况1: 前肢高度低
    low_heightF_mask = desired_heightF < THRESHOLD_HEIGHT
    if low_heightF_mask.any():
        f_body_quat = asset.data.body_link_quat_w[low_heightF_mask, 4]
        f_body_roll = _quaternion_to_roll(f_body_quat) + math.pi/2  # +90度转换为弧度
        f_roll_error = torch.abs(f_body_roll - desired_angle[low_heightF_mask])
        f_roll_error = torch.min(f_roll_error, 2 * math.pi - f_roll_error)
        roll_error[low_heightF_mask] = f_roll_error
        sigma = 50
    
    # 情况2: 后肢高度低
    low_heightH_mask = desired_heightH < THRESHOLD_HEIGHT
    if low_heightH_mask.any():
        h_body_quat = asset.data.body_link_quat_w[low_heightH_mask, 24]
        h_body_roll = _quaternion_to_roll(h_body_quat) - math.pi/2  # -90度转换为弧度
        h_roll_error = torch.abs(h_body_roll - desired_angle[low_heightH_mask])
        h_roll_error = torch.min(h_roll_error, 2 * math.pi - h_roll_error)
        roll_error[low_heightH_mask] = h_roll_error
        sigma = 50
    
    # 情况3: 两个高度都不低于阈值
    high_height_mask = ~(low_heightF_mask | low_heightH_mask)
    if high_height_mask.any():
        f_body_quat = asset.data.body_link_quat_w[high_height_mask, 4]
        f_roll_error = torch.abs(_quaternion_to_roll(f_body_quat) + math.pi/2)  
        h_body_quat = asset.data.body_link_quat_w[high_height_mask, 24]
        h_roll_error = torch.abs(_quaternion_to_roll(h_body_quat) - math.pi/2)  
        roll_error[high_height_mask] = (f_roll_error + h_roll_error)
        sigma = 100
    
    reward = torch.exp(-sigma * roll_error ** 2)
    weight = get_curriculum_reward_weight(env, "angle")    
    return reward * weight




# 能量消耗惩罚函数
def compute_energy_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    actuator_vel = asset.data.joint_vel[:, resolve_joint_ids(asset)]
    actuator_torque = asset.data.actuator_force
    power = actuator_vel * actuator_torque
    total_power = torch.sum(torch.abs(power), dim=1)
    penalty = -0.05 * total_power
    weight = get_curriculum_reward_weight(env, "energy")
    
    return penalty * weight
    


# 计算COT惩罚函数
def compute_cot_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    total_mass = 2.4525
    actuator_vel = asset.data.joint_vel[:, resolve_joint_ids(asset)]
    actuator_torque = asset.data.actuator_force
    power = torch.sum(torch.abs(actuator_vel * actuator_torque), dim=1)
    base_lin_vel_w = asset.data.root_link_lin_vel_w
    forward_vel = base_lin_vel_w[:, 0]
    epsilon = 1e-9
    speed = torch.abs(forward_vel) + epsilon
    cot = power / (total_mass * speed + epsilon)
    penalty = -0.1 * cot  
    weight = get_curriculum_reward_weight(env, "cot")
    
    if env.common_step_counter % 100 == 0:
        print(f"COT Penalty - Mean COT: {cot.mean().item():.4f}, "
              f"Mean Energy: {power.mean().item():.4f}, "
              f"Mean Penalty: {penalty.mean().item():.4f}")
    
    return penalty * weight


    
# 动作平滑性惩罚
def compute_smoothness_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:    
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    action_diff = current_action - prev_action
    penalty = -torch.sum(torch.square(action_diff), dim=1)
    weight = get_curriculum_reward_weight(env, "smoothness")

    return penalty * weight



# 关节加速度惩罚函数
def compute_joint_acc_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    joint_acc = asset.data.joint_acc
    actuator_acc = joint_acc[:, resolve_joint_ids(asset)]
    penalty = -torch.mean(0.005 * torch.abs(actuator_acc), dim=1)
    weight = get_curriculum_reward_weight(env, "joint_acc")
    
    return penalty * weight



# Y轴基座偏移惩罚函数
def compute_base_y_offset_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    y_offset = asset.data.root_link_pos_w[:, 1]
    penalty = -torch.abs(y_offset) * 100
    weight = get_curriculum_reward_weight(env, "y_offset")
    
    return penalty * weight



# 腿部关节限位惩罚
def compute_limits_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    joint_pos = env.scene.entities["robot"].data.joint_pos
    joint_names = env.scene.entities["robot"].joint_names
    
    penalty = torch.zeros(env.num_envs, device=env.device)
    
    joint_limits = {}
    shoulder_pattern = re.compile(r".*_(shoulder_joint)")
    elbow_pattern = re.compile(r".*_(elbow_joint)")
    hip_pattern = re.compile(r".*_(hip_joint)")
    knee_pattern = re.compile(r".*_(knee_joint)")
    
    for i, joint_name in enumerate(joint_names):
        if shoulder_pattern.match(joint_name):
            joint_limits[i] = (-0.5, 1.2)
        elif elbow_pattern.match(joint_name):
            joint_limits[i] = (-0.8, 0.3)
        elif hip_pattern.match(joint_name):
            joint_limits[i] = (-0.9, 0.8)
        elif knee_pattern.match(joint_name):
            joint_limits[i] = (-0.2, 0.8)
        else:
            joint_limits[i] = (-2, 2)
    
    for joint_idx, (lower, upper) in joint_limits.items():
        joint_angles = joint_pos[:, joint_idx]
        lower_violation = torch.clamp(lower - joint_angles, min=0.0)
        upper_violation = torch.clamp(joint_angles - upper, min=0.0)
        joint_penalty = lower_violation + upper_violation
        penalty += joint_penalty
    
    weight = -get_curriculum_reward_weight(env, "joint_limits")
    return penalty * weight



# 动作加速度惩罚
def compute_action_acc(env: ManagerBasedRlEnv) -> torch.Tensor:
    policy_obs = env.observation_manager.compute_group("actor", update_history=False)
    if isinstance(policy_obs, dict):
        if "actions" in policy_obs:
            action_history = policy_obs["actions"]
        else:
            return torch.zeros(env.num_envs, device=env.device)
    elif isinstance(policy_obs, torch.Tensor):
        action_dim = 36 
        if policy_obs.shape[1] < action_dim:
            return torch.zeros(env.num_envs, device=env.device)
        action_history = policy_obs[:, :action_dim] 
    else:
        return torch.zeros(env.num_envs, device=env.device)
    if action_history.dim() != 2 or action_history.shape[1] != 36:
        return torch.zeros(env.num_envs, device=env.device)  
    action_history_3d = action_history.reshape(env.num_envs, 3, 12)
    prev_prev_action = action_history_3d[:, 0, :]  # t-2 (最旧)
    prev_action = action_history_3d[:, 1, :]       # t-1
    current_action = action_history_3d[:, 2, :]    # t (最新)
    action_acceleration = current_action - 2 * prev_action + prev_prev_action
    
    penalty = -torch.sum(torch.abs(action_acceleration), dim=1)
    weight = get_curriculum_reward_weight(env, "action_acc")
    
    return penalty * weight



# 虚拟碰撞奖励
def compute_body_contact_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset = env.scene["robot"]
    device = env.device
    num_envs = env.num_envs

    # 板几何 (x 覆盖区间 / 虚拟阈值) 读**实际场景实体**, 与实体碰撞共用同一份定义
    geo = hole_geometry(env)
    obs_x_min = torch.tensor([g["x_min"] for g in geo], device=device)
    obs_x_max = torch.tensor([g["x_max"] for g in geo], device=device)
    obs_z_thresh = torch.tensor([g["virtual_z_threshold"] for g in geo], device=device)

    # 前肢/后肢躯干中心的 X 坐标 [num_envs, 2], 索引由 indices.py 解析
    body_x = asset.data.body_link_pos_w[
        :, [_MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id], 0]

    # 判断是否在障碍物X范围内 [num_envs, 2, 3]
    in_obs_matrix = (body_x.unsqueeze(-1) >= obs_x_min) & (body_x.unsqueeze(-1) <= obs_x_max)
    # 不在任何板区内的躯干无高度约束, 阈值抬到 +inf, 使其恒判为"未超出"
    active_z_thresh = torch.where(
        in_obs_matrix, obs_z_thresh.expand_as(in_obs_matrix),
        torch.full_like(in_obs_matrix, float("inf"), dtype=obs_z_thresh.dtype).expand_as(in_obs_matrix)
    ).min(dim=-1).values                                    # [num_envs, 2]

    # 采样点: 前段 9 点 + 后段 9 点 (索引由 indices.py 按名字解析)
    seg_site_ids = list(_MODEL_INDICES.f_body_site_ids) + list(_MODEL_INDICES.h_body_site_ids)
    all_sites_z = asset.data.site_pos_w[:, seg_site_ids, 2]
    all_sites_z = all_sites_z.view(num_envs, 2, 9)
    # 每段取最高的采样点与阈值比较, 超出量 v = max(surface - z_thresh, 0)
    v = torch.clamp(all_sites_z.max(dim=-1).values - active_z_thresh, min=0.0)

    sigma = get_curriculum_reward_weight(env, "sigma_body_contact")
    weight = get_curriculum_reward_weight(env, "body_contact")
    # 在范围内给奖励, 超出按超出量指数衰减 (形态对齐 Slalom 走廊奖励)
    r_f = torch.exp(-sigma * v[:, 0] ** 2)
    r_h = torch.exp(-sigma * v[:, 1] ** 2)
    reward = (r_f + r_h) / 2

    # 记录 metrics
    env.extras["log"]["Metrics/body_contact_excess"] = v.mean().item()
    env.extras["log"]["Metrics/rear_surface_excess"] = v[:, 1].mean().item()

    return reward * weight



# Mode 2 前腿运动奖励（防止前腿不动）
def compute_stop_reward(env: ManagerBasedRlEnv, min_velocity: float = 0.5) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    cmd_term = env.command_manager._terms["hole_cmd"]
    desired_heightF = cmd_term.command[:, CMD_HEIGHT_F_IDS]
    desired_heightH = cmd_term.command[:, CMD_HEIGHT_H_IDS]
    
    # Mode 2 判断：前肢高度 >= 阈值，后肢高度 < 阈值
    mode2_mask = (desired_heightF >= THRESHOLD_HEIGHT) & (desired_heightH < THRESHOLD_HEIGHT)
    reward = torch.zeros(env.num_envs, device=env.device)
    
    if not mode2_mask.any():
        return reward
    
    # 获取前肢关节速度 (参考表列序前 4 项 = FL/FR shoulder+elbow)
    joint_vel = asset.data.joint_vel
    front_leg_vel = joint_vel[:, resolve_joint_ids(asset)[:4]]  # [num_envs, 4]
    vel_abs = torch.abs(front_leg_vel)
    vel_deficit = torch.clamp(min_velocity - vel_abs, min=0.0)
    vel_deficit_sum = vel_deficit.sum(dim=1)
    mode2_reward = -vel_deficit_sum * 2.0
    reward[mode2_mask] = mode2_reward[mode2_mask]
    
    if env.common_step_counter % 10 == 0 and mode2_mask.any():
        mean_deficit = vel_deficit_sum[mode2_mask].mean().item()
        mean_reward = reward[mode2_mask].mean().item()
        # 可选：打印每个关节的单独统计
        mean_joint_vel = vel_abs[mode2_mask].mean(dim=0)
        print(f"Stop Reward (Mode 2) - Envs: {mode2_mask.sum().item()}, "
              f"Mean Vel Deficit Sum: {mean_deficit:.3f}, "
              f"Mean Reward: {mean_reward:.3f}")
        print(f"  Joint vel means - FL_sh: {mean_joint_vel[0]:.3f}, "
              f"FL_el: {mean_joint_vel[1]:.3f}, "
              f"FR_sh: {mean_joint_vel[2]:.3f}, "
              f"FR_el: {mean_joint_vel[3]:.3f}")
    
    return reward




_SPN_SCALE_CACHE: dict = {}



# 缓存脊柱逐轴误差缩放 (顺序 = F_spine1, F_body, H_spine1, H_body)
def _error_spn_scale(env, device, dtype) -> torch.Tensor:
    scale = SPN_AXIS_SCALE
    key = (str(device), str(dtype), scale)
    s = _SPN_SCALE_CACHE.get(key)
    if s is None:
        s = torch.tensor(scale, device=device, dtype=dtype)
        _SPN_SCALE_CACHE[key] = s
    return s



# 辅助函数：四元数转偏航角（绕Z轴旋转）
def _quaternion_to_roll(quat: torch.Tensor) -> torch.Tensor:
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
    
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # 计算侧倾角（绕X轴）
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    return roll.squeeze()
