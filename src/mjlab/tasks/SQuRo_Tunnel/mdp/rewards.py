from __future__ import annotations
import re
import math
import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .curriculums import get_curriculum_reward_weight
from .reference import (
    ACTUATOR_IDS,
    JOINT_IDS,
    ACTUATOR_NUM,
    get_reference_joint_pos,
    get_reference_joint_vel
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# 计算位置模仿奖励
def compute_mimic_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    joint_pos = asset.data.joint_pos
    target_joint_pos = get_reference_joint_pos(env)
    current_pos = joint_pos[:, JOINT_IDS]
    pos_errors = current_pos - target_joint_pos
    mse_errors = torch.mean(pos_errors ** 2, dim=1)
    sigma = get_curriculum_reward_weight(env, "mimic_pos_sigma")
    reward = torch.exp(-sigma * mse_errors)
    weight = get_curriculum_reward_weight(env, "mimic_pos")
    
    # 调试信息
    if env.common_step_counter % 1000 == 0:
        leg_mse = torch.mean(pos_errors[:, :8] ** 2, dim=1).mean().item()
        print(f"Position Imitation Reward - Mean: {reward.mean().item():.3f}, "
              f"Leg MSE: {leg_mse:.3f}")
    
    return reward * weight


# 计算速度模仿奖励
def compute_mimic_velocity_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    joint_vel = asset.data.joint_vel
    target_joint_vel = get_reference_joint_vel(env)
    current_vel = joint_vel[:, JOINT_IDS]
    vel_errors = current_vel - target_joint_vel
    mse_errors = torch.mean(vel_errors ** 2, dim=1)
    
    sigma = get_curriculum_reward_weight(env, "mimic_vel_sigma")
    reward = torch.exp(-sigma * mse_errors)
    weight = get_curriculum_reward_weight(env, "mimic_vel")
    
    # 调试信息
    if env.common_step_counter % 1000 == 0:
        leg_mse = torch.mean(vel_errors[:, :8] ** 2, dim=1).mean().item()
        print(f"Velocity Imitation Reward - Mean: {reward.mean().item():.3f}, "
              f"Leg MSE: {leg_mse:.3f}")

    return reward * weight


# X线速度跟踪奖励
def compute_linear_velocity_reward(env: ManagerBasedRlEnv) -> torch.Tensor:    
    # 获取实际速度
    asset: Entity = env.scene["robot"]
    actual_vel_x = asset.data.root_link_lin_vel_w[:, 0]
    # 获取期望速度
    cmd_term = env.command_manager._terms["mouse_cmd"]
    desired_vel_x = cmd_term.command[:, 0]
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


# 高度奖励 
def compute_height_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取实际高度
    asset: Entity = env.scene["robot"]
    body_pos_w = asset.data.body_link_pos_w
    f_body_height = body_pos_w[:, 4, 2]             # 前肢身体高度
    h_body_height = body_pos_w[:, 24, 2]            # 后肢身体高度
    base_height = asset.data.root_link_pos_w[:, 2]
    # 获取期望高度
    cmd_term = env.command_manager._terms["mouse_cmd"]
    desired_height_F = cmd_term.command[:, 3]       # 前肢高度命令
    desired_height_H = cmd_term.command[:, 4]       # 后肢高度命令
    desired_height = (desired_height_F + desired_height_H) / 2
    # 计算高度误差
    height_F_error = torch.abs(desired_height_F - f_body_height)
    height_H_error = torch.abs(desired_height_H - h_body_height)
    height_base_error = torch.abs(desired_height - base_height)
    # 记录 metrics
    env.extras["log"]["Metrics/height_F_error_mean"] = height_F_error.mean().item()
    env.extras["log"]["Metrics/height_H_error_mean"] = height_H_error.mean().item()
    env.extras["log"]["Metrics/height_base_error_mean"] = height_base_error.mean().item()    
    # 计算奖励
    sigma = get_curriculum_reward_weight(env, "height_sigma")
    reward_F = torch.exp(-sigma * height_F_error ** 2)
    reward_H = torch.exp(-sigma * height_H_error ** 2)
    reward_Base = torch.exp(-sigma * height_base_error ** 2)
    reward = 0.4 * reward_F + 0.4 * reward_H + 0.2 * reward_Base
    weight = get_curriculum_reward_weight(env, "height")    
    return reward * weight


# 抬脚高度奖励（仅 Mode 0 触发）
def compute_foot_clearance_reward(env: ManagerBasedRlEnv, target_base_height: float = 0.01) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    cmd_term = env.command_manager._terms["mouse_cmd"]
    desired_height_F = cmd_term.command[:, 3]
    desired_height_H = cmd_term.command[:, 4]
    
    # Mode 0: 前后肢高度都 >= 0.04
    height_threshold = 0.04
    mode0_mask = (desired_height_F >= height_threshold) & (desired_height_H >= height_threshold)
    
    # 非 Mode 0 直接返回零奖励
    if not mode0_mask.any():
        return torch.zeros(env.num_envs, device=env.device)
    
    foot_site_ids = [10, 11, 21, 22]
    foot_z = asset.data.site_pos_w[:, foot_site_ids, 2]
    
    base_height = 0.06
    height_scale_F = desired_height_F / base_height
    height_scale_H = desired_height_H / base_height
    
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
    
    # 调试打印（每100步打印一次）
    if env.common_step_counter % 100 == 0:
        print(f"Target Proximity Reward - Target X: {target_x}, "
              f"Current X Mean: {current_x.mean().item():.3f}, "
              f"Distance Mean: {distance.mean().item():.3f}, "
              f"Reward Mean: {reward.mean().item():.3f}")
    
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
    
    # 调试信息
    if env.common_step_counter % 1000 == 0:
        mean_yaw = torch.rad2deg(yaw).mean().item()
        mean_error = torch.rad2deg(yaw_error_alt).mean().item()
        print(f"Orientation Reward - Mean Yaw: {mean_yaw:.1f}°, "
              f"Mean Error: {mean_error:.1f}°, "
              f"Reward Mean: {reward.mean().item():.3f}")
    
    return reward * weight


# 身体角度奖励
def compute_angle_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    cmd_term = env.command_manager._terms["mouse_cmd"]
    desired_heightF = cmd_term.command[:, 3]  # 前肢高度命令
    desired_heightH = cmd_term.command[:, 4]  # 后肢高度命令
    desired_angle = cmd_term.command[:, 5]    # 侧倾角度命令
    roll_error = torch.zeros(env.num_envs, device=env.device)
    sigma = 50

    # 情况1: 前肢高度低 (<0.035)
    low_heightF_mask = desired_heightF < 0.04
    if low_heightF_mask.any():
        f_body_quat = asset.data.body_link_quat_w[low_heightF_mask, 4]
        f_body_roll = _quaternion_to_roll(f_body_quat) + math.pi/2  # +90度转换为弧度
        f_roll_error = torch.abs(f_body_roll - desired_angle[low_heightF_mask])
        f_roll_error = torch.min(f_roll_error, 2 * math.pi - f_roll_error)
        roll_error[low_heightF_mask] = f_roll_error
        sigma = 50
    
    # 情况2: 后肢高度低 (<0.035)
    low_heightH_mask = desired_heightH < 0.04
    if low_heightH_mask.any():
        h_body_quat = asset.data.body_link_quat_w[low_heightH_mask, 24]
        h_body_roll = _quaternion_to_roll(h_body_quat) - math.pi/2  # -90度转换为弧度
        h_roll_error = torch.abs(h_body_roll - desired_angle[low_heightH_mask])
        h_roll_error = torch.min(h_roll_error, 2 * math.pi - h_roll_error)
        roll_error[low_heightH_mask] = h_roll_error
        sigma = 50
    
    # 情况3: 两个高度都大于等于0.035
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
    
    # 调试信息
    if env.common_step_counter % 100 == 0:
        low_F_count = low_heightF_mask.sum().item()
        low_H_count = low_heightH_mask.sum().item()
        high_count = high_height_mask.sum().item()
        
        if low_F_count > 0:
            low_F_mean_error = roll_error[low_heightF_mask].mean().item()
            low_F_mean_reward = reward[low_heightF_mask].mean().item()
            low_F_mean_roll = _quaternion_to_roll(asset.data.body_link_quat_w[low_heightF_mask, 4]).mean().item()
            print(f"Body Angle Reward (F low) - Envs: {low_F_count}, "
                  f"F Body Roll: {torch.rad2deg(torch.tensor(low_F_mean_roll + math.pi/2)).item():.1f}°, "
                  f"Cmd Angle: {torch.rad2deg(desired_angle[low_heightF_mask].mean()).item():.1f}°, "
                  f"Mean Error: {torch.rad2deg(torch.tensor(low_F_mean_error)).item():.1f}°, "
                  f"Mean Reward: {low_F_mean_reward:.3f}")
        
        if low_H_count > 0:
            low_H_mean_error = roll_error[low_heightH_mask].mean().item()
            low_H_mean_reward = reward[low_heightH_mask].mean().item()
            low_H_mean_roll = _quaternion_to_roll(asset.data.body_link_quat_w[low_heightH_mask, 24]).mean().item()
            print(f"Body Angle Reward (H low) - Envs: {low_H_count}, "
                  f"H Body Roll: {torch.rad2deg(torch.tensor(low_H_mean_roll - math.pi/2)).item():.1f}°, "
                  f"Cmd Angle: {torch.rad2deg(desired_angle[low_heightH_mask].mean()).item():.1f}°, "
                  f"Mean Error: {torch.rad2deg(torch.tensor(low_H_mean_error)).item():.1f}°, "
                  f"Mean Reward: {low_H_mean_reward:.3f}")
        
        if high_count > 0:
            high_mean_error = roll_error[high_height_mask].mean().item()
            high_mean_reward = reward[high_height_mask].mean().item()
            print(f"Body Angle Reward (Both high) - Envs: {high_count}, "
                  f"Mean Error: {torch.rad2deg(torch.tensor(high_mean_error)).item():.1f}°, "
                  f"Mean Reward: {high_mean_reward:.3f}")
    
    return reward * weight


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


# 能量消耗惩罚函数
def compute_energy_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    actuator_vel = asset.data.joint_vel[:, JOINT_IDS]
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
    actuator_vel = asset.data.joint_vel[:, JOINT_IDS]
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
    actuator_acc = joint_acc[:, JOINT_IDS]
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
    policy_obs = env.observation_manager.compute_group("policy", update_history=False)
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


def compute_body_contact_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset = env.scene["robot"]
    device = env.device
    num_envs = env.num_envs

    obs_x_min = torch.tensor([0.185, 0.5, 1.185], device=device)
    obs_x_max = torch.tensor([0.215, 0.7, 1.215], device=device)
    obs_z_thresh = torch.tensor([0.045, 0.07, 0.045], device=device)
    
    # 前肢(body 4)和后肢(body 24)的X坐标 [num_envs, 2]
    body_x = asset.data.body_link_pos_w[:, [4, 24], 0]
    
    # 判断是否在障碍物X范围内 [num_envs, 2, 3]
    in_obs_matrix = (body_x.unsqueeze(-1) >= obs_x_min) & (body_x.unsqueeze(-1) <= obs_x_max)
    
    if not in_obs_matrix.any():
        return torch.zeros(num_envs, device=device)
    
    active_z_thresh = (in_obs_matrix.float() @ obs_z_thresh)
    in_any_obs = in_obs_matrix.any(dim=-1)
    combined_site_ids = list(range(9)) + list(range(12, 21))
    all_sites_z = asset.data.site_pos_w[:, combined_site_ids, 2]
    all_sites_z = all_sites_z.view(num_envs, 2, 9)
    diff = all_sites_z - active_z_thresh.unsqueeze(-1)
    excess = torch.clamp(diff, min=0.0)
    excess = excess * in_any_obs.unsqueeze(-1)
    total_penalty = excess.sum(dim=(1, 2))  # [num_envs]
    weight = get_curriculum_reward_weight(env, "body_contact")

    return -total_penalty * weight * 10


# 更新课程学习状态（障碍物碰撞等）
def update_curriculum(env: ManagerBasedRlEnv) -> torch.Tensor:
    from .curriculums import update_curriculum_holes
    update_curriculum_holes(env)
    return torch.zeros(env.num_envs, device=env.device)


# Mode 2 前腿运动奖励（防止前腿不动）
def compute_stop_reward(env: ManagerBasedRlEnv, min_velocity: float = 0.5) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    cmd_term = env.command_manager._terms["mouse_cmd"]
    desired_heightF = cmd_term.command[:, 3]
    desired_heightH = cmd_term.command[:, 4]
    
    # Mode 2 判断：前肢高度 >= 0.04，后肢高度 < 0.04
    height_threshold = 0.04
    mode2_mask = (desired_heightF >= height_threshold) & (desired_heightH < height_threshold)
    reward = torch.zeros(env.num_envs, device=env.device)
    
    if not mode2_mask.any():
        return reward
    
    # 获取前肢关节速度（JOINT_IDS 中前4个是 FL shoulder, FL elbow, FR shoulder, FR elbow）
    joint_vel = asset.data.joint_vel
    front_leg_vel = joint_vel[:, JOINT_IDS[:4]]  # [num_envs, 4]
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