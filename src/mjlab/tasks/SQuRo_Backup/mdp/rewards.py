from __future__ import annotations
import torch
from mjlab.entity import Entity
from typing import TYPE_CHECKING, cast
from .config import STAND_STILL_FULL_SPEED, T4
from .command import BackupCommand, _GROUND_TH_S1
from .indices import _ACTUATED_JOINT_NAMES, _ACTUATOR_CTRL_RANGE, _MODEL_INDICES
from .curriculums import get_curriculum_reward_weight, get_spn_axis_scale
from .reference import get_reference_joint_state, get_body_reference, get_reference_body_attitude

if TYPE_CHECKING:
    from mjlab.envs.mdp.actions import JointPositionAction
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv



# s1 里程碑奖励
def compute_s1_milestone_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    pulse = command.s1_milestone_pulse
    weight = get_curriculum_reward_weight(env, "weight_milestone_s1")
    quality = _milestone_time_quality(command.s1_dev_early, command.s1_dev_late, command.time_scale_command)
    reward = weight * pulse.float() * quality / env.step_dt
    return reward



# s2 里程碑奖励
def compute_s2_milestone_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    pulse = command.s2_milestone_pulse
    weight = get_curriculum_reward_weight(env, "weight_milestone_s2")
    quality = _milestone_time_quality(command.s2_dev_early, command.s2_dev_late, command.time_scale_command)
    reward = weight * pulse.float() * quality / env.step_dt
    return reward



# 成功站立里程碑奖励
def compute_success_milestone_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    command.stand_reward_and_pulse()
    pulse = command.consume_cycle_end_pulse()
    weight = get_curriculum_reward_weight(env, "weight_milestone_success")
    reward = weight * pulse.float() / env.step_dt
    return reward



# s1 区间奖励
def compute_s1_progress_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    weight = get_curriculum_reward_weight(env, "weight_progress_s1")
    reward = weight * command.progress_s1 * _spine_track_kernel(env)
    return reward



# s2 区间奖励：P2 及以后
def compute_s2_progress_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    weight = get_curriculum_reward_weight(env, "weight_progress_s2")
    reward = weight * command.progress_s2 * (command.phase >= 1) * _spine_track_kernel(env)
    return reward



# s3 区间奖励：仅 P3
def compute_s3_progress_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    _, progress = command.standing_state()
    weight = get_curriculum_reward_weight(env, "weight_progress_s3")
    reward = weight * progress * (command.phase == 2)
    return reward



# S1 姿态塑形奖励：仅 P1
def compute_s1_shape_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    asset: Entity = env.scene["robot"]
    u = command._get_pose_cos()
    front_inverted = ((1.0 - u[:, 0]) / 2.0).clamp(0.0, 1.0)
    back_upright = ((1.0 + u[:, 1]) / 2.0).clamp(0.0, 1.0)
    heights = asset.data.body_link_pos_w
    z_front = heights[:, _MODEL_INDICES.f_body_id, 2]
    z_back = heights[:, _MODEL_INDICES.h_body_id, 2]
    f_front = (1.0 - (z_front - _GROUND_TH_S1).clamp(min=0.0) / 0.02).clamp(0.0, 1.0)
    f_back = (1.0 - (z_back - _GROUND_TH_S1).clamp(min=0.0) / 0.02).clamp(0.0, 1.0)
    ground = f_front * f_back
    shape = front_inverted * back_upright * ground
    weight = get_curriculum_reward_weight(env, "weight_s1_shape")
    reward =  weight * shape * (command.phase == 0) * _spine_track_kernel(env)
    return reward



# 关节位置模仿奖励
def compute_mimic_pos_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    joint_pos = asset.data.joint_pos[:, _MODEL_INDICES.joint_ids]
    ref_pos, _ = get_reference_joint_state(env)
    error = joint_pos - ref_pos
    error_leg = error[:, _MODEL_INDICES.actuator_leg_ids]
    error_spn = error[:, _MODEL_INDICES.actuator_spn_ids]
    error_spn = error_spn * _spn_axis_scale(env, error_spn.device, error_spn.dtype)
    error_neck = error[:, _MODEL_INDICES.actuator_neck_ids]
    weight = get_curriculum_reward_weight(env, "weight_mimic_pos")
    sigma_leg = get_curriculum_reward_weight(env, "sigma_leg_pos")
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_pos")
    sigma_neck = get_curriculum_reward_weight(env, "sigma_neck_pos")
    alpha_neck = get_curriculum_reward_weight(env, "alpha_neck_pos")
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    mse_neck = torch.mean(error_neck ** 2, dim=1)
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward_neck = torch.exp(-sigma_neck * mse_neck)
    reward = (reward_leg + reward_spn) / 2 + alpha_neck * reward_neck
    return reward * weight



# 关节速度模仿奖励
def compute_mimic_vel_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    joint_vel = asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
    _, ref_vel = get_reference_joint_state(env)
    error = joint_vel - ref_vel
    error_leg = error[:, _MODEL_INDICES.actuator_leg_ids]
    error_spn = error[:, _MODEL_INDICES.actuator_spn_ids]
    error_spn = error_spn * _spn_axis_scale(env, error_spn.device, error_spn.dtype)
    error_neck = error[:, _MODEL_INDICES.actuator_neck_ids]
    weight = get_curriculum_reward_weight(env, "weight_mimic_vel")
    sigma_leg = get_curriculum_reward_weight(env, "sigma_leg_vel")
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_vel")
    sigma_neck = get_curriculum_reward_weight(env, "sigma_neck_vel")
    alpha_neck = get_curriculum_reward_weight(env, "alpha_neck_vel")
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    mse_neck = torch.mean(error_neck ** 2, dim=1)
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward_neck = torch.exp(-sigma_neck * mse_neck)
    reward = (reward_leg + reward_spn) / 2 + alpha_neck * reward_neck
    return reward * weight



# 身体高度跟踪奖励
def compute_height_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    body_pos_w = asset.data.body_link_pos_w
    F_body_height = body_pos_w[:, _MODEL_INDICES.f_body_id, 2]
    H_body_height = body_pos_w[:, _MODEL_INDICES.h_body_id, 2]
    _, z_ref_F, _, z_ref_H = get_body_reference(env)
    height_F_error = torch.abs(z_ref_F - F_body_height)
    height_H_error = torch.abs(z_ref_H - H_body_height)
    sigma_height = get_curriculum_reward_weight(env, "sigma_height")
    w_height = get_curriculum_reward_weight(env, "weight_height")
    r_height_F = torch.exp(-sigma_height * height_F_error ** 2)
    r_height_H = torch.exp(-sigma_height * height_H_error ** 2)   
    Reward_height = w_height * (0.5 * r_height_F + 0.5 * r_height_H)
    return Reward_height



# 关节跟踪惩罚
def compute_joint_track_penalty(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    joint_pos = asset.data.joint_pos[:, _MODEL_INDICES.joint_ids]
    ref_pos, _ = get_reference_joint_state(env)
    err = joint_pos - ref_pos
    err = err * _joint_error_scale(env, err.device, err.dtype)
    w = _joint_group_weights(err.device, err.dtype)
    cost = (w * err.square()).mean(dim=1) / 3.0
    weight = get_curriculum_reward_weight(env, "weight_track_joint")
    return -weight * cost



# 脊柱动作跟踪惩罚
def compute_spn_track_penalty(env: "ManagerBasedRlEnv") -> torch.Tensor:
    spn = _MODEL_INDICES.actuator_spn_ids
    ref_pos, _ = get_reference_joint_state(env)
    weight = get_curriculum_reward_weight(env, "weight_spn_track")
    penalty = weight * _joint_target_cost(env, spn, _spn_axis_scale(env, ref_pos.device, ref_pos.dtype))
    return penalty



# 躯干朝向跟踪惩罚
def compute_body_track_penalty(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    ref_u = get_reference_body_attitude(env)
    actual_u = command._pose_cos()
    valid = torch.isfinite(actual_u)
    err = torch.where(valid, actual_u - ref_u, torch.zeros_like(ref_u))
    weight = get_curriculum_reward_weight(env, "weight_body_track")
    penalty = -weight * err.square().mean(dim=1)
    return penalty



# 腿部动作跟踪惩罚：仅 P3
def compute_leg_action_penalty(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    cost = _joint_target_cost(env, _MODEL_INDICES.actuator_leg_ids)
    weight = get_curriculum_reward_weight(env, "weight_leg_action")
    penalty = weight * cost * (command.phase == 2)
    return penalty



# 腿部姿态跟踪惩罚：仅 P3
def compute_leg_pos_penalty(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    asset: Entity = env.scene["robot"]
    leg_ids = _MODEL_INDICES.actuator_leg_ids
    actual = asset.data.joint_pos[:, _MODEL_INDICES.joint_ids][:, leg_ids]
    ref_pos, _ = get_reference_joint_state(env)
    error = actual - ref_pos[:, leg_ids]
    mse = torch.mean(error.square(), dim=1)
    weight = get_curriculum_reward_weight(env, "weight_leg_pose")
    penalty = -weight * mse * (command.phase == 2)
    # 记录日志
    log = getattr(env, "extras", {}).get("log") if hasattr(env, "extras") else None
    if log is not None:
        rmse = mse.sqrt()
        in_p3 = command.phase == 2
        lam = command.time_scale_command.clamp(min=0.1)
        in_hold = in_p3 & (command.t_phase >= lam * T4)
        n_p3 = int(in_p3.sum())
        n_hold = int(in_hold.sum())
        if n_p3 > 0:
            log["Data/leg_pose_rmse_p3"] = float(rmse[in_p3].mean().item())
            if n_hold > 0:
                log["Data/leg_pose_rmse_hold"] = float(rmse[in_hold].mean().item())
    return penalty



# 站立静止奖励：仅 P3
def compute_stand_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    hold, _ = command.stand_gate()
    mean_vel = command.windowed_mean_vel()
    speed = torch.nan_to_num(mean_vel, nan=STAND_STILL_FULL_SPEED, posinf=STAND_STILL_FULL_SPEED, neginf=STAND_STILL_FULL_SPEED)
    still = (1.0 - speed / STAND_STILL_FULL_SPEED).clamp(0.0, 1.0)
    weight = get_curriculum_reward_weight(env, "weight_stand")
    reward = weight * (hold & (command.phase == 2)).float() * still
    return reward



# 动作超出执行器范围惩罚
def compute_action_excess_penalty(env: "ManagerBasedRlEnv") -> torch.Tensor:
    action_term = cast("JointPositionAction", env.action_manager.get_term("joint_pos"))
    target = action_term.raw_action * action_term.scale + action_term.offset
    lo, hi = _ctrl_range_tensors(action_term.target_names, target.device, target.dtype)
    cost = (lo - target).clamp(min=0.0) + (target - hi).clamp(min=0.0)
    weight = get_curriculum_reward_weight(env, "weight_action_excess")
    penalty = -weight * cost
    return penalty



# L1 动作平滑惩罚
def compute_action_L1_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    abs_diff = torch.abs(current_action - prev_action)
    leg_cost = torch.sum(abs_diff[:, _MODEL_INDICES.actuator_leg_ids], dim=1)
    spn_cost = torch.sum(abs_diff[:, _MODEL_INDICES.actuator_spn_ids], dim=1)
    error_cost = torch.sum(abs_diff[:, _MODEL_INDICES.actuator_neck_ids], dim=1)
    w_leg = get_curriculum_reward_weight(env, "weight_smooth_L1_leg")
    w_spn = get_curriculum_reward_weight(env, "weight_smooth_L1_spn")
    penalty = -w_leg * leg_cost - w_spn * spn_cost - w_spn * error_cost
    return penalty



# L2 动作平滑惩罚
def compute_action_L2_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    sq_diff = torch.square(current_action - prev_action)
    leg_cost = torch.sum(sq_diff[:, _MODEL_INDICES.actuator_leg_ids], dim=1)
    spn_cost = torch.sum(sq_diff[:, _MODEL_INDICES.actuator_spn_ids], dim=1)
    error_cost = torch.sum(sq_diff[:, _MODEL_INDICES.actuator_neck_ids], dim=1)
    w_leg = get_curriculum_reward_weight(env, "weight_smooth_L2_leg")
    w_spn = get_curriculum_reward_weight(env, "weight_smooth_L2_spn")
    penalty = -w_leg * leg_cost - w_spn * spn_cost - w_spn * error_cost
    return penalty



# 能耗惩罚
def compute_energy_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    actuator_vel = asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
    actuator_torque = asset.data.actuator_force
    cost = torch.sum(torch.abs(actuator_vel * actuator_torque), dim=1)
    weight = get_curriculum_reward_weight(env, "weight_energy")
    penalty = -weight * cost
    return penalty



# ===================================================================================================================
# 缓存用变量
_SPN_SCALE_CACHE: dict = {}
_ERR_SCALE_CACHE: dict = {}
_CTRL_RANGE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_JOINT_W_CACHE: dict = {}
_NECK_NAMES = {_ACTUATED_JOINT_NAMES[i] for i in (2, 3)}
_SPN_NAMES = {_ACTUATED_JOINT_NAMES[i] for i in (0, 1, 8, 9)}



# 缓存脊柱四关节误差缩放
def _spn_axis_scale(env, device, dtype) -> torch.Tensor:
    scale = get_spn_axis_scale(int(env.common_step_counter))
    key = (str(device), str(dtype), scale)
    s = _SPN_SCALE_CACHE.get(key)
    if s is None:
        s = torch.tensor(scale, device=device, dtype=dtype)
        _SPN_SCALE_CACHE[key] = s
    return s



# 缓存执行器关节误差缩放
def _joint_error_scale(env, device, dtype) -> torch.Tensor:
    scale = get_spn_axis_scale(int(env.common_step_counter))
    key = (str(device), str(dtype), scale)
    s = _ERR_SCALE_CACHE.get(key)
    if s is None:
        s = torch.ones(len(_ACTUATED_JOINT_NAMES), device=device, dtype=dtype)
        s[list(_MODEL_INDICES.actuator_spn_ids)] = torch.tensor(scale, device=device, dtype=dtype)
        _ERR_SCALE_CACHE[key] = s
    return s



# 缓存执行器范围
def _ctrl_range_tensors(names, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    key = (tuple(names), str(device), str(dtype))
    cached = _CTRL_RANGE_CACHE.get(key)
    if cached is None:
        lo = [0.0] * len(names)
        hi = [0.0] * len(names)
        for i, name in enumerate(names):
            lo[i], hi[i] = _ACTUATOR_CTRL_RANGE[name]
        cached = (torch.tensor(lo, device=device, dtype=dtype), torch.tensor(hi, device=device, dtype=dtype))
        _CTRL_RANGE_CACHE[key] = cached
    return cached



# 缓存关节权重
def _joint_group_weights(device, dtype) -> torch.Tensor:
    key = (str(device), str(dtype))
    w = _JOINT_W_CACHE.get(key)
    if w is None:
        w = torch.tensor([1.57 if n in _SPN_NAMES else 0.30 if n in _NECK_NAMES
                          else 1.00 for n in _ACTUATED_JOINT_NAMES], device=device, dtype=dtype)
        _JOINT_W_CACHE[key] = w
    return w



# 按名称对齐关节目标成本
def _joint_target_cost(env: "ManagerBasedRlEnv", ref_columns: tuple[int, ...], scales: torch.Tensor | None = None) -> torch.Tensor:
    action_term = cast("JointPositionAction", env.action_manager.get_term("joint_pos"))
    target = action_term.raw_action * action_term.scale + action_term.offset
    ref_pos, _ = get_reference_joint_state(env)
    target_columns = tuple(action_term.target_names.index(_ACTUATED_JOINT_NAMES[i]) for i in ref_columns)
    error = target[:, target_columns] - ref_pos[:, ref_columns]
    if scales is not None:
        error = error * scales
    return -torch.mean(error.square(), dim=1)



# 里程碑奖励时间门控
def _milestone_time_quality(dev_early: torch.Tensor, dev_late: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(dev_early) & torch.isfinite(dev_late)
    e = torch.nan_to_num(dev_early, nan=0.0)
    l = torch.nan_to_num(dev_late, nan=0.0)
    late_positive = l.clamp(min=0.0)
    early_negative = (-e).clamp(min=0.0)
    sigma_early = (0.40 * lam).clamp_min(1e-6)
    q_early = torch.exp(-(early_negative / sigma_early) ** 2)
    q_late = torch.exp(-(late_positive / 0.35) ** 2)
    return torch.where(valid, q_early * q_late, torch.zeros_like(q_early))



# 脊柱跟踪门控
def _spine_track_kernel(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    joint_pos = asset.data.joint_pos[:, _MODEL_INDICES.joint_ids]
    ref_pos, _ = get_reference_joint_state(env)
    error_spn = (joint_pos - ref_pos)[:, _MODEL_INDICES.actuator_spn_ids]
    error_spn = error_spn * _spn_axis_scale(env, error_spn.device, error_spn.dtype)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_pos")
    return torch.exp(-sigma_spn * mse_spn)