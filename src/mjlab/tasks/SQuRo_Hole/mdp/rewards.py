from __future__ import annotations
import math
import torch
from mjlab.entity import Entity
from typing import TYPE_CHECKING
from mjlab.utils.lab_api.math import euler_xyz_from_quat
from .config import (
    BASE_HEIGHT,
    CMD_VEL_X_IDS,
    THRESHOLD_HEIGHT,
    CMD_HEIGHT_F_IDS,
    CMD_HEIGHT_H_IDS,
    BODY_HEIGHT_OFFSET,
)
from .indices import (
    _ACT_LEG_IDS,
    _ACT_SPN_IDS,
    _ACT_NECK_IDS,
    _ACT_LEG_F_IDS,
    _MODEL_INDICES,
    resolve_actuator_joint_ids,
)
from .curriculums import (
    SPN_AXIS_SCALE,
    get_curriculum_reward_weight,
)
from .reference import get_reference_joint_state

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
    reward = (reward_leg + reward_spn + reward_neck) * weight
    return reward



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
    reward = (reward_leg + reward_spn + reward_neck) * weight
    return reward



# X 线速度跟踪奖励
def compute_vel_reward(env: ManagerBasedRlEnv) -> torch.Tensor:    
    # 获取实际与期望速度
    asset: Entity = env.scene["robot"]
    actual_vel_x = asset.data.root_link_lin_vel_w[:, 0]
    cmd_term = env.command_manager._terms["hole_cmd"]
    desired_vel_x = cmd_term.command[:, CMD_VEL_X_IDS]
    has_velocity_cmd = torch.abs(desired_vel_x) > 0.01
    min_speed_mask = (torch.abs(actual_vel_x) < 0.01) & has_velocity_cmd
    # 计算速度误差
    vel_error = torch.abs(desired_vel_x - actual_vel_x)
    # 获取课程学习量
    sigma = get_curriculum_reward_weight(env, "sigma_vel")
    weight = get_curriculum_reward_weight(env, "weight_vel")
    # 计算奖励
    r = torch.exp(-sigma * vel_error ** 2) * weight
    reward = torch.where(min_speed_mask, torch.full_like(r, -1.0), r)
    # 记录日志
    env.extras["log"]["Data/act_vel_x"] = actual_vel_x.mean().item()
    env.extras["log"]["Data/cmd_vel_x"] = desired_vel_x.mean().item()
    return reward



# 身体高度跟踪奖励
def compute_height_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取实际与期望前后肢高度
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
    sigma = get_curriculum_reward_weight(env, "sigma_height")
    weight = get_curriculum_reward_weight(env, "weight_height")
    # 计算奖励
    r_height_F = torch.exp(-sigma * height_F_error ** 2)
    r_height_H = torch.exp(-sigma * height_H_error ** 2)
    reward = (0.5 * r_height_F + 0.5 * r_height_H) * weight
    # 记录日志
    env.extras["log"]["Data/error_height_F"] = height_F_error.mean().item()
    env.extras["log"]["Data/error_height_H"] = height_H_error.mean().item()
    return reward



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



# 身体角度奖励，修正坐标系后的目标角度固定为零
def compute_angle_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    cmd_term = env.command_manager._terms["hole_cmd"]
    desired_height_F = cmd_term.command[:, CMD_HEIGHT_F_IDS]
    desired_height_H = cmd_term.command[:, CMD_HEIGHT_H_IDS]
    roll_error = torch.zeros(env.num_envs, device=env.device)
    sigma = torch.full_like(roll_error, get_curriculum_reward_weight(env, "sigma_angle_low"))

    # 情况1: 前肢高度低
    low_height_F_mask = desired_height_F < THRESHOLD_HEIGHT
    if low_height_F_mask.any():
        f_body_quat = asset.data.body_link_quat_w[low_height_F_mask, _MODEL_INDICES.f_body_id]
        f_body_roll = euler_xyz_from_quat(f_body_quat)[0] + math.pi/2  # +90度转换为弧度
        f_roll_error = torch.abs(f_body_roll)
        f_roll_error = torch.min(f_roll_error, 2 * math.pi - f_roll_error)
        roll_error[low_height_F_mask] = f_roll_error
    
    # 情况2: 后肢高度低
    low_height_H_mask = desired_height_H < THRESHOLD_HEIGHT
    if low_height_H_mask.any():
        h_body_quat = asset.data.body_link_quat_w[low_height_H_mask, _MODEL_INDICES.h_body_id]
        h_body_roll = euler_xyz_from_quat(h_body_quat)[0] - math.pi/2  # -90度转换为弧度
        h_roll_error = torch.abs(h_body_roll)
        h_roll_error = torch.min(h_roll_error, 2 * math.pi - h_roll_error)
        roll_error[low_height_H_mask] = h_roll_error
    
    # 情况3: 两个高度都不低于阈值
    high_height_mask = ~(low_height_F_mask | low_height_H_mask)
    if high_height_mask.any():
        f_body_quat = asset.data.body_link_quat_w[high_height_mask, _MODEL_INDICES.f_body_id]
        f_roll_error = torch.abs(euler_xyz_from_quat(f_body_quat)[0] + math.pi/2)
        h_body_quat = asset.data.body_link_quat_w[high_height_mask, _MODEL_INDICES.h_body_id]
        h_roll_error = torch.abs(euler_xyz_from_quat(h_body_quat)[0] - math.pi/2)
        roll_error[high_height_mask] = (f_roll_error + h_roll_error)
        sigma[high_height_mask] = get_curriculum_reward_weight(env, "sigma_angle_high")
    
    reward = torch.exp(-sigma * roll_error ** 2)
    weight = get_curriculum_reward_weight(env, "angle")
    
    return reward * weight



# L2 动作平滑惩罚
def compute_action_L2_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取当前和上一步动作   
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    # 计算动作差
    action_diff = current_action - prev_action
    # 获取课程学习量
    weight = get_curriculum_reward_weight(env, "weight_action_L2")
    # 计算惩罚
    p = -torch.sum(torch.square(action_diff), dim=1)
    penalty = p * weight
    return penalty




# 全位置走廊只惩罚两段最高采样点超出各自目标上限
def compute_corridor_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    weight = get_curriculum_reward_weight(env, "corridor")
    if weight == 0.0:
        return torch.zeros(env.num_envs, device=env.device)

    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command("hole_cmd")
    surface_z = torch.stack((
        asset.data.site_pos_w[:, _MODEL_INDICES.f_body_site_ids, 2].max(dim=1).values,
        asset.data.site_pos_w[:, _MODEL_INDICES.h_body_site_ids, 2].max(dim=1).values,
    ), dim=1)
    ceiling = command[:, [CMD_HEIGHT_F_IDS, CMD_HEIGHT_H_IDS]] + BODY_HEIGHT_OFFSET  # type: ignore
    excess = (surface_z - ceiling).clamp_min(0.0)
    sigma = get_curriculum_reward_weight(env, "sigma_corridor")
    return -weight * (1.0 - torch.exp(-sigma * excess.square())).mean(dim=1)



# Mode 2 前腿运动奖励（防止前腿不动）
def compute_stop_reward(env: ManagerBasedRlEnv, min_velocity: float = 0.5) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    cmd_term = env.command_manager._terms["hole_cmd"]
    desired_height_F = cmd_term.command[:, CMD_HEIGHT_F_IDS]
    desired_height_H = cmd_term.command[:, CMD_HEIGHT_H_IDS]
    
    # Mode 2 判断：前肢高度 >= 阈值，后肢高度 < 阈值
    mode2_mask = (desired_height_F >= THRESHOLD_HEIGHT) & (desired_height_H < THRESHOLD_HEIGHT)
    reward = torch.zeros(env.num_envs, device=env.device)
    
    if not mode2_mask.any():
        return reward
    
    # 按前腿执行器分组读取模型关节速度
    joint_vel = asset.data.joint_vel
    front_leg_vel = joint_vel[:, resolve_actuator_joint_ids(asset, _ACT_LEG_F_IDS)]
    vel_abs = torch.abs(front_leg_vel)
    vel_deficit = torch.clamp(min_velocity - vel_abs, min=0.0)
    vel_deficit_sum = vel_deficit.sum(dim=1)
    mode2_reward = -vel_deficit_sum * 2.0
    reward[mode2_mask] = mode2_reward[mode2_mask]
    
    return reward



_SPN_SCALE_CACHE: dict = {}



# 缓存脊柱逐轴误差缩放
def _error_spn_scale(env, device, dtype) -> torch.Tensor:
    scale = SPN_AXIS_SCALE
    key = (str(device), str(dtype), scale)
    s = _SPN_SCALE_CACHE.get(key)
    if s is None:
        s = torch.tensor(scale, device=device, dtype=dtype)
        _SPN_SCALE_CACHE[key] = s
    return s