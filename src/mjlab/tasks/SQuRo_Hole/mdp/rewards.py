from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

from .config import (
    BASE_HEIGHT,
    CMD_HEIGHT_F_IDS,
    CMD_HEIGHT_H_IDS,
    CMD_VEL_X_IDS,
    THRESHOLD_HEIGHT,
    BODY_HEIGHT_OFFSET,
)
from .curriculums import (
    SPN_AXIS_SCALE,
    get_curriculum_reward_weight,
)
from .indices import (
    _ACT_F_LEG_IDS,
    _ACT_LEG_IDS,
    _ACT_NECK_IDS,
    _ACT_SPN_IDS,
    _MODEL_INDICES,
    resolve_actuator_joint_ids,
)
from .reference import (
    get_reference_joint_state,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv



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
    # 获取 sigma
    sigma_leg = get_curriculum_reward_weight(env, "sigma_leg_pos")
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_pos")
    sigma_neck = get_curriculum_reward_weight(env, "sigma_neck_pos")
    # 计算 MSE 误差
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    mse_neck = torch.mean(error_neck ** 2, dim=1)
    # 计算奖励
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward_neck = torch.exp(-sigma_neck * mse_neck)
    reward = reward_leg + reward_spn + reward_neck
    weight = get_curriculum_reward_weight(env, "mimic_pos")
    return reward * weight



# 关节速度模仿奖励
def compute_mimic_vel_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取实际和期望关节位置
    asset: Entity = env.scene["robot"]
    joint_vel = asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
    _, ref_vel = get_reference_joint_state(env)
    # 计算关节位置误差
    error = joint_vel - ref_vel
    error_leg = error[:, _ACT_LEG_IDS]
    error_spn = error[:, _ACT_SPN_IDS] * _error_spn_scale(env, error.device, error.dtype)
    error_neck = error[:, _ACT_NECK_IDS]
    # 获取 sigma
    sigma_leg = get_curriculum_reward_weight(env, "sigma_leg_vel")
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_vel")
    sigma_neck = get_curriculum_reward_weight(env, "sigma_neck_vel")
    # 计算 MSE 误差
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    mse_neck = torch.mean(error_neck ** 2, dim=1)
    # 计算奖励
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward_neck = torch.exp(-sigma_neck * mse_neck)
    reward = reward_leg + reward_spn + reward_neck
    weight = get_curriculum_reward_weight(env, "mimic_vel")
    return reward * weight



# X线速度跟踪奖励
def compute_linear_velocity_reward(env: ManagerBasedRlEnv) -> torch.Tensor:    
    # 获取实际速度
    asset: Entity = env.scene["robot"]
    actual_vel_x = asset.data.root_link_lin_vel_w[:, 0]
    # 获取期望速度
    cmd_term = env.command_manager._terms["hole_cmd"]
    desired_vel_x = cmd_term.command[:, CMD_VEL_X_IDS]
    has_velocity_cmd = torch.abs(desired_vel_x) > 0.01
    min_speed_mask = (torch.abs(actual_vel_x) < 0.01) & has_velocity_cmd
    # 计算速度误差
    vel_error = torch.abs(desired_vel_x - actual_vel_x)
    # 记录 metrics 到 extras["log"]
    env.extras["log"]["Metrics/vel_error_mean"] = vel_error.mean().item()
    env.extras["log"]["Metrics/actual_vel_x_mean"] = actual_vel_x.mean().item()
    env.extras["log"]["Metrics/cmd_vel_x_mean"] = desired_vel_x.mean().item()
    # 计算奖励
    reward = torch.exp(-100.0 * vel_error ** 2)
    reward = torch.where(min_speed_mask, torch.full_like(reward, -1.0), reward)
    weight = get_curriculum_reward_weight(env, "vel")  
    return reward * weight



# 身体高度跟踪奖励
def compute_height_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    body_pos_w = asset.data.body_link_pos_w
    F_body_height = body_pos_w[:, _MODEL_INDICES.f_body_id, 2]
    H_body_height = body_pos_w[:, _MODEL_INDICES.h_body_id, 2]
    cmd_term = env.command_manager._terms["hole_cmd"]
    z_ref_F = cmd_term.command[:, CMD_HEIGHT_F_IDS]
    z_ref_H = cmd_term.command[:, CMD_HEIGHT_H_IDS]
    height_F_error = torch.abs(z_ref_F - F_body_height)
    height_H_error = torch.abs(z_ref_H - H_body_height)
    # 记录 metrics
    env.extras["log"]["Metrics/height_F_error_mean"] = height_F_error.mean().item()
    env.extras["log"]["Metrics/height_H_error_mean"] = height_H_error.mean().item()
    sigma_height = get_curriculum_reward_weight(env, "height_sigma")
    w_height = get_curriculum_reward_weight(env, "height")
    r_height_F = torch.exp(-sigma_height * height_F_error ** 2)
    r_height_H = torch.exp(-sigma_height * height_H_error ** 2)
    Reward_height = w_height * (0.5 * r_height_F + 0.5 * r_height_H)
    return Reward_height



# 抬脚高度奖励（仅 Mode 0 触发）
def compute_foot_clearance_reward(env: ManagerBasedRlEnv, target_base_height: float = 0.01) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    cmd_term = env.command_manager._terms["hole_cmd"]
    desired_height_F = cmd_term.command[:, CMD_HEIGHT_F_IDS]
    desired_height_H = cmd_term.command[:, CMD_HEIGHT_H_IDS]
    
    # Mode 0: 前后肢高度都 >= 阈值
    mode0_mask = (desired_height_F >= THRESHOLD_HEIGHT) & (desired_height_H >= THRESHOLD_HEIGHT)
    
    # 非 Mode 0 直接返回零奖励
    if not mode0_mask.any():
        return torch.zeros(env.num_envs, device=env.device)
    
    foot_z = asset.data.site_pos_w[:, list(_MODEL_INDICES.foot_site_ids), 2]
    
    height_scale_F = desired_height_F / BASE_HEIGHT
    height_scale_H = desired_height_H / BASE_HEIGHT
    
    target_height_F = target_base_height * height_scale_F
    target_height_H = target_base_height * height_scale_H
    target_heights = torch.stack([target_height_F, target_height_F, target_height_H, target_height_H], dim=1)
    
    height_error = torch.abs(foot_z - target_heights)
    reward_per_foot = torch.exp(-100.0 * torch.square(height_error))
    reward = torch.mean(reward_per_foot, dim=1)
    
    # 只保留 Mode 0 的奖励，其他为 0
    reward = reward * mode0_mask.float()
    weight = get_curriculum_reward_weight(env, "foot_clearance")
    
    
    return reward * weight



# 目标位置接近奖励（距离x=1.5越近奖励越大）
def compute_reached_reward(env: ManagerBasedRlEnv, target_x: float = 1.5, sigma: float = 10.0) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    # 获取baselink的当前位置（世界坐标系）
    current_x = asset.data.root_link_pos_w[:, 0]
    
    # 计算x方向距离误差
    x_error = current_x - target_x
    distance = torch.abs(x_error)
    
    # 使用高斯函数计算奖励：距离越近奖励越大
    # exp(-sigma * distance^2)，距离为0时奖励为1
    reward = torch.exp(-sigma * distance ** 2)
    
    # 获取课程学习权重（如果有配置的话）
    weight = get_curriculum_reward_weight(env, "reached")
    
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



# 身体角度奖励，修正坐标系后的目标角度固定为零。
def compute_angle_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    cmd_term = env.command_manager._terms["hole_cmd"]
    desired_heightF = cmd_term.command[:, CMD_HEIGHT_F_IDS]
    desired_heightH = cmd_term.command[:, CMD_HEIGHT_H_IDS]
    roll_error = torch.zeros(env.num_envs, device=env.device)
    sigma = torch.full_like(roll_error, get_curriculum_reward_weight(env, "sigma_angle_low"))

    # 情况1: 前肢高度低
    low_heightF_mask = desired_heightF < THRESHOLD_HEIGHT
    if low_heightF_mask.any():
        f_body_quat = asset.data.body_link_quat_w[low_heightF_mask, _MODEL_INDICES.f_body_id]
        f_body_roll = _quaternion_to_roll(f_body_quat) + math.pi/2  # +90度转换为弧度
        f_roll_error = torch.abs(f_body_roll)
        f_roll_error = torch.min(f_roll_error, 2 * math.pi - f_roll_error)
        roll_error[low_heightF_mask] = f_roll_error
    
    # 情况2: 后肢高度低
    low_heightH_mask = desired_heightH < THRESHOLD_HEIGHT
    if low_heightH_mask.any():
        h_body_quat = asset.data.body_link_quat_w[low_heightH_mask, _MODEL_INDICES.h_body_id]
        h_body_roll = _quaternion_to_roll(h_body_quat) - math.pi/2  # -90度转换为弧度
        h_roll_error = torch.abs(h_body_roll)
        h_roll_error = torch.min(h_roll_error, 2 * math.pi - h_roll_error)
        roll_error[low_heightH_mask] = h_roll_error
    
    # 情况3: 两个高度都不低于阈值
    high_height_mask = ~(low_heightF_mask | low_heightH_mask)
    if high_height_mask.any():
        f_body_quat = asset.data.body_link_quat_w[high_height_mask, _MODEL_INDICES.f_body_id]
        f_roll_error = torch.abs(_quaternion_to_roll(f_body_quat) + math.pi/2)  
        h_body_quat = asset.data.body_link_quat_w[high_height_mask, _MODEL_INDICES.h_body_id]
        h_roll_error = torch.abs(_quaternion_to_roll(h_body_quat) - math.pi/2)  
        roll_error[high_height_mask] = (f_roll_error + h_roll_error)
        sigma[high_height_mask] = get_curriculum_reward_weight(env, "sigma_angle_high")
    
    reward = torch.exp(-sigma * roll_error ** 2)
    weight = get_curriculum_reward_weight(env, "angle")
    
    return reward * weight




# 动作平滑性惩罚
def compute_smoothness_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:    
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    action_diff = current_action - prev_action
    penalty = -torch.sum(torch.square(action_diff), dim=1)
    weight = get_curriculum_reward_weight(env, "smoothness")

    return penalty * weight



# 全位置走廊只惩罚两段最高采样点超出各自目标上限。
def compute_corridor_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    weight = get_curriculum_reward_weight(env, "corridor")
    log = env.extras["log"]
    if weight == 0.0:
        zero = torch.zeros(env.num_envs, device=env.device)
        log["Metrics/走廊/前段超限均值"] = zero.mean().detach()
        log["Metrics/走廊/后段超限均值"] = zero.mean().detach()
        return zero

    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command("hole_cmd")
    surface_z = torch.stack((
        asset.data.site_pos_w[:, _MODEL_INDICES.f_body_site_ids, 2].max(dim=1).values,
        asset.data.site_pos_w[:, _MODEL_INDICES.h_body_site_ids, 2].max(dim=1).values,
    ), dim=1)
    ceiling = command[:, [CMD_HEIGHT_F_IDS, CMD_HEIGHT_H_IDS]] + BODY_HEIGHT_OFFSET  # type: ignore
    excess = (surface_z - ceiling).clamp_min(0.0)
    sigma = get_curriculum_reward_weight(env, "sigma_corridor")
    log["Metrics/走廊/前段超限均值"] = excess[:, 0].mean().detach()
    log["Metrics/走廊/后段超限均值"] = excess[:, 1].mean().detach()
    return -weight * (1.0 - torch.exp(-sigma * excess.square())).mean(dim=1)



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
    
    # 按前腿执行器分组读取模型关节速度。
    joint_vel = asset.data.joint_vel
    front_leg_vel = joint_vel[:, resolve_actuator_joint_ids(asset, _ACT_F_LEG_IDS)]
    vel_abs = torch.abs(front_leg_vel)
    vel_deficit = torch.clamp(min_velocity - vel_abs, min=0.0)
    vel_deficit_sum = vel_deficit.sum(dim=1)
    mode2_reward = -vel_deficit_sum * 2.0
    reward[mode2_mask] = mode2_reward[mode2_mask]
    
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
