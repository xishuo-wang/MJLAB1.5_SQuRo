from __future__ import annotations
import torch
from mjlab.entity import Entity
from typing import TYPE_CHECKING
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.asset_zoo.robots.SQuRo.SQuRo_constants import (
    SPINE_HIND_FRAME,
    SPINE_FRONT_FRAME,
)
from .indices import (
    _ACT_LEG_IDS,
    _ACT_SPN_IDS,
    _ACT_NECK_IDS,
    _ACT_LEG_F_IDS,
    _ACT_LEG_H_IDS,
    _MODEL_INDICES,
    resolve_model_indices,
    resolve_actuator_joint_ids,
)
from .config import (
    BASE_HEIGHT,
    CMD_VEL_X_IDS,
    THRESHOLD_HEIGHT,
    CMD_HEIGHT_F_IDS,
    CMD_HEIGHT_H_IDS,
    VIRTUAL_CLEARANCE_MARGIN,
)
from .curriculums import (
    SPN_AXIS_SCALE,
    get_curriculum_value,
)
from .entity import read_obstacle_bottom_z
from .command import get_low_pose_intensity
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
    error_spn = error[:, _ACT_SPN_IDS] * _error_spn_scale(error.device, error.dtype)
    error_neck = error[:, _ACT_NECK_IDS]
    # 获取课程学习量
    sigma_leg = get_curriculum_value(env, "sigma_leg_pos")
    sigma_spn = get_curriculum_value(env, "sigma_spn_pos")
    sigma_neck = get_curriculum_value(env, "sigma_neck_pos")
    weight = get_curriculum_value(env, "weight_mimic_pos")
    # 计算 MSE 误差
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    mse_neck = torch.mean(error_neck ** 2, dim=1)
    # 计算奖励
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward_neck = torch.exp(-sigma_neck * mse_neck)
    reward = (reward_leg + reward_spn + reward_neck)/2 * weight
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
    error_spn = error[:, _ACT_SPN_IDS] * _error_spn_scale(error.device, error.dtype)
    error_neck = error[:, _ACT_NECK_IDS]
    # 获取课程学习量
    sigma_leg = get_curriculum_value(env, "sigma_leg_vel")
    sigma_spn = get_curriculum_value(env, "sigma_spn_vel")
    sigma_neck = get_curriculum_value(env, "sigma_neck_vel")
    weight = get_curriculum_value(env, "weight_mimic_vel")
    # 计算 MSE 误差
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    mse_neck = torch.mean(error_neck ** 2, dim=1)
    # 计算奖励
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward_neck = torch.exp(-sigma_neck * mse_neck)
    reward = (reward_leg + reward_spn + reward_neck)/2 * weight
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
    sigma = get_curriculum_value(env, "sigma_vel")
    weight = get_curriculum_value(env, "weight_vel")
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
    sigma = get_curriculum_value(env, "sigma_height")
    weight = get_curriculum_value(env, "weight_height")
    # 计算奖励
    r_height_F = torch.exp(-sigma * height_F_error ** 2)
    r_height_H = torch.exp(-sigma * height_H_error ** 2)
    reward = (0.5 * r_height_F + 0.5 * r_height_H) * weight
    # 记录日志
    env.extras["log"]["Data/error_height_F"] = height_F_error.mean().item()
    env.extras["log"]["Data/error_height_H"] = height_H_error.mean().item()
    return reward



# 偏航角奖励
def compute_yaw_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取实际 body 偏航角
    yaw, _ = _get_body_orientation(env)
    # 计算偏航角误差
    yaw_error = torch.atan2(torch.sin(yaw).mean(dim=1), torch.cos(yaw).mean(dim=1))
    # 获取课程学习量
    sigma = get_curriculum_value(env, "sigma_yaw")
    weight = get_curriculum_value(env, "weight_yaw")
    # 计算奖励
    r = torch.exp(-sigma * yaw_error ** 2)
    reward = r * weight
    return reward



# 滚转角奖励
def compute_roll_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取当前高度 Mode
    cmd_term = env.command_manager._terms["hole_cmd"]
    desired_height_F = cmd_term.command[:, CMD_HEIGHT_F_IDS]
    desired_height_H = cmd_term.command[:, CMD_HEIGHT_H_IDS]
    low_height_F_mask = desired_height_F < THRESHOLD_HEIGHT
    low_height_H_mask = desired_height_H < THRESHOLD_HEIGHT
    high_height_mask = ~(low_height_F_mask | low_height_H_mask)
    # 获取期望和实际滚转
    _, roll = _get_body_orientation(env)
    f_roll_error, h_roll_error = torch.abs(roll).unbind(dim=1)
    # 计算滚转角误差
    roll_error = torch.where(high_height_mask | low_height_F_mask, f_roll_error, 0.0)
    roll_error += torch.where(high_height_mask | low_height_H_mask, h_roll_error, 0.0)
    # 获取课程学习量
    sigma = torch.where(
        high_height_mask,
        get_curriculum_value(env, "sigma_roll_high"),
        get_curriculum_value(env, "sigma_roll_low"),
    )
    weight = get_curriculum_value(env, "weight_roll")
    # 计算奖励
    r = torch.exp(-sigma * roll_error ** 2)
    reward = r * weight
    return reward



# 抬脚高度奖励：仅双高的 Mode 0触发
def compute_foot_reward(env: ManagerBasedRlEnv, target_base_height: float = 0.01) -> torch.Tensor:
    # 获取当前高度 Mode
    asset: Entity = env.scene["robot"]
    cmd_term = env.command_manager._terms["hole_cmd"]
    desired_height_F = cmd_term.command[:, CMD_HEIGHT_F_IDS]
    desired_height_H = cmd_term.command[:, CMD_HEIGHT_H_IDS]
    mode0_mask = (desired_height_F >= THRESHOLD_HEIGHT) & (desired_height_H >= THRESHOLD_HEIGHT)
    # 获取期望和实际足端 site 点高度
    foot_z = asset.data.site_pos_w[:, list(_MODEL_INDICES.foot_site_ids), 2]
    height_scale_F = desired_height_F / BASE_HEIGHT
    height_scale_H = desired_height_H / BASE_HEIGHT
    target_height_F = target_base_height * height_scale_F
    target_height_H = target_base_height * height_scale_H
    target_heights = torch.stack([target_height_F, target_height_F, target_height_H, target_height_H], dim=1)
    # 计算高度误差
    height_error = torch.abs(foot_z - target_heights)
    # 获取课程学习量
    sigma = get_curriculum_value(env, "sigma_foot")
    weight = get_curriculum_value(env, "weight_foot")
    # 计算奖励
    r_per_foot = torch.exp(-sigma * torch.square(height_error))
    r = torch.mean(r_per_foot, dim=1) * mode0_mask.float()
    reward = r * weight
    return reward



# 走廊惩罚
def compute_corridor_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取期望身体高度
    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command("hole_cmd")
    surface_z = torch.stack((
        asset.data.site_pos_w[:, _MODEL_INDICES.f_body_site_ids, 2].max(dim=1).values,
        asset.data.site_pos_w[:, _MODEL_INDICES.h_body_site_ids, 2].max(dim=1).values,
    ), dim=1)
    # 获取课程学习量
    offset = get_curriculum_value(env, "offset_corridor")
    sigma = get_curriculum_value(env, "sigma_corridor")
    weight = get_curriculum_value(env, "weight_corridor")
    # 计算超出高度
    ceiling = command[:, [CMD_HEIGHT_F_IDS, CMD_HEIGHT_H_IDS]] + offset  # type: ignore
    excess = (surface_z - ceiling).clamp_min(0.0)
    # 计算惩罚
    p = (torch.exp(-sigma * excess.square()) - 1).mean(dim=1)
    penalty = p * weight
    return penalty



# 虚拟碰撞惩罚
def compute_contact_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取障碍物信息
    asset: Entity = env.scene["robot"]
    resolve_model_indices(asset)
    obstacle = env.scene.entities.get("obstacle")
    if obstacle is None:
        return torch.zeros(env.num_envs, device=env.device)
    # 获取障碍物对应区域
    threshold = read_obstacle_bottom_z(obstacle) - VIRTUAL_CLEARANCE_MARGIN
    center_x = float(obstacle.cfg.position[0]) # type: ignore
    half_length = float(obstacle.cfg.size[0]) # type: ignore
    x_min, x_max = center_x - half_length, center_x + half_length
    # 获取机器人躯干 x 坐标
    body_x = asset.data.body_link_pos_w[:, (_MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id), 0]
    # 计算位于障碍物区间内的身体超出量
    inside = (body_x >= x_min) & (body_x <= x_max)
    front_ids = list(_MODEL_INDICES.f_body_site_ids)
    hind_ids = list(_MODEL_INDICES.h_body_site_ids)
    assert len(front_ids) == len(hind_ids), "前后段躯干采样点数量必须一致"
    surface_z = asset.data.site_pos_w[:, front_ids + hind_ids, 2].view(env.num_envs, 2, -1).max(dim=-1).values
    active_threshold = torch.where(inside, threshold.unsqueeze(1), torch.full_like(surface_z, float("inf")))
    excess = (surface_z - active_threshold).clamp_min(0.0)
    # 获取课程学习量
    sigma = get_curriculum_value(env, "sigma_contact")
    weight = get_curriculum_value(env, "weight_contact")
    # 计算惩罚
    p = (torch.exp(-sigma * excess.square()) - 1).mean(dim=1)
    penalty = p * weight
    return penalty



# 前腿静止惩罚：仅前高后低的 Mode 2 触发
def compute_leg_stop_penalty(env: ManagerBasedRlEnv, min_velocity: float = 0.5) -> torch.Tensor:
    # 获取当前高度 Mode
    asset: Entity = env.scene["robot"]
    cmd_term = env.command_manager._terms["hole_cmd"]
    desired_height_F = cmd_term.command[:, CMD_HEIGHT_F_IDS]
    desired_height_H = cmd_term.command[:, CMD_HEIGHT_H_IDS]
    mode2_mask = (desired_height_F >= THRESHOLD_HEIGHT) & (desired_height_H < THRESHOLD_HEIGHT)
    # 获取前腿关节速度
    joint_vel = asset.data.joint_vel
    leg_F_vel = joint_vel[:, resolve_actuator_joint_ids(asset, _ACT_LEG_F_IDS)]
    # 计算前腿速度误差
    vel_deficit = torch.clamp(min_velocity - torch.abs(leg_F_vel), min=0.0)
    vel_deficit_sum = vel_deficit.sum(dim=1)
    # 获取课程学习量
    weight = get_curriculum_value(env, "weight_leg_stop")
    # 计算惩罚
    p = -vel_deficit_sum * weight
    penalty = torch.where(mode2_mask, p, torch.zeros_like(p))
    return penalty



# 低侧腿运动惩罚
def compute_leg_motion_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取当前高度 Mode
    asset: Entity = env.scene["robot"]
    command = env.command_manager._terms["hole_cmd"].command
    height_f = command[:, CMD_HEIGHT_F_IDS]
    height_h = command[:, CMD_HEIGHT_H_IDS]
    front_low = height_f < THRESHOLD_HEIGHT
    hind_low = height_h < THRESHOLD_HEIGHT
    # 获取腿部速度
    front_vel = asset.data.joint_vel[:, resolve_actuator_joint_ids(asset, _ACT_LEG_F_IDS)]
    hind_vel = asset.data.joint_vel[:, resolve_actuator_joint_ids(asset, _ACT_LEG_H_IDS)]
    # 计算腿部速度误差
    low_mse = torch.where(front_low, front_vel.square().mean(dim=1), hind_vel.square().mean(dim=1))
    strength = get_low_pose_intensity(torch.minimum(height_f, height_h))
    # 获取课程学习量
    weight = get_curriculum_value(env, "weight_leg_motion")
    # 计算惩罚
    p = -strength * low_mse * (front_low ^ hind_low)
    penalty = p * weight
    return penalty



# L2 动作平滑惩罚
def compute_action_L2_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取当前和上一步动作   
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    # 计算动作差
    action_diff = current_action - prev_action
    # 获取课程学习量
    weight = get_curriculum_value(env, "weight_action_L2")
    # 计算惩罚
    p = -torch.sum(torch.square(action_diff), dim=1)
    penalty = p * weight
    return penalty



# 脊柱误差缩放缓存
_SPN_SCALE_CACHE: dict = {}



# 缓存脊柱逐轴误差缩放
def _error_spn_scale(device, dtype) -> torch.Tensor:
    scale = SPN_AXIS_SCALE
    key = (str(device), str(dtype), scale)
    s = _SPN_SCALE_CACHE.get(key)
    if s is None:
        s = torch.tensor(scale, device=device, dtype=dtype)
        _SPN_SCALE_CACHE[key] = s
    return s



# 获取前后躯干的物理偏航角与滚转角
def _get_body_orientation(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    asset: Entity = env.scene["robot"]
    body_ids = (_MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id)
    quat = asset.data.body_link_quat_w[:, body_ids]
    rotation = matrix_from_quat(quat) @ quat.new_tensor((SPINE_FRONT_FRAME, SPINE_HIND_FRAME))
    yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    roll = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    return yaw, roll
