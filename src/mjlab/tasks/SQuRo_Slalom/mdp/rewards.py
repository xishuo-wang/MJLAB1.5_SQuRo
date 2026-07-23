from __future__ import annotations
import torch
from mjlab.entity import Entity
from typing import TYPE_CHECKING
from .indices import _MODEL_INDICES
from .observations import _get_f_body_heading
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
    # 获取课程学习量
    weight = get_curriculum_reward_weight(env, "weight_mimic_pos")
    sigma = get_curriculum_reward_weight(env, "sigma_mimic_pos")
    # 计算奖励
    mse = torch.mean(error ** 2, dim=1)
    reward = torch.exp(-sigma * mse)
    return reward * weight



# =========================================================================================
# 关节速度模仿奖励
def compute_mimic_vel_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    # 计算关节速度误差
    joint_vel = asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
    _, ref_vel = get_reference_joint_state(env)
    error = joint_vel - ref_vel
    # 获取课程学习量
    weight = get_curriculum_reward_weight(env, "weight_mimic_vel")
    sigma = get_curriculum_reward_weight(env, "sigma_mimic_vel")
    # 计算奖励
    mse = torch.mean(error ** 2, dim=1)
    reward = torch.exp(-sigma * mse)
    return reward * weight



# =========================================================================================
# 线速度跟踪奖励
def compute_vel_track_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    # F_body 物理前向 = body +Y (不是 body +X)
    # body +X → heading=90°(world +Y), body +Y → heading=0°(world +X=前向)
    # 所以 forward_heading = f_body_heading - π/2
    f_body_heading = _get_f_body_heading(env) - (torch.pi / 2)
    vel_w = asset.data.body_link_lin_vel_w[:, _MODEL_INDICES.f_body_id, :]  # type: ignore[call-overload]  # [N,3]
    forward_speed = vel_w[:, 0] * torch.cos(f_body_heading) + vel_w[:, 1] * torch.sin(f_body_heading)
    lateral_speed = -vel_w[:, 0] * torch.sin(f_body_heading) + vel_w[:, 1] * torch.cos(f_body_heading)
    vertical_speed = vel_w[:, 2]
    cmd_term = env.command_manager._terms["slalom_cmd"]
    v_cmd = cmd_term.command[:, 0]
    error = forward_speed - v_cmd
    error_vy = lateral_speed
    error_vz = vertical_speed
    # 获取课程学习量
    weight = get_curriculum_reward_weight(env, "weight_track_vel")
    weight_yz = get_curriculum_reward_weight(env, "weight_track_vyz")
    sigma = get_curriculum_reward_weight(env, "sigma_track_vel")
    sigma_yz = get_curriculum_reward_weight(env, "sigma_track_vyz")
    # 计算奖励
    reward = torch.exp(-sigma * error ** 2)
    r_vel_y = torch.exp(-sigma_yz * error_vy ** 2)
    r_vel_z = torch.exp(-sigma_yz * error_vz ** 2)
    # 记录日志
    env.extras["log"]["Data/vel_actual"] = forward_speed.mean().item()
    return reward * weight + (r_vel_y + r_vel_z) * weight_yz



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

    # 初始化/更新路径起始状态
    if getattr(env, "_path_state", None) is None:
        env._path_state = {  # type: ignore[attr-defined]
            "start_pos": asset.data.root_link_pos_w.clone(),
            "start_heading": (_get_f_body_heading(env) - (torch.pi / 2)).clone(),
        }

    state = env._path_state  # type: ignore[attr-defined]
    start_pos = state["start_pos"]       # [N, 3]
    start_heading = state["start_heading"]  # [N] 物理前向 (= body+X - π/2)

    # 重置已终止环境的状态
    reset_ids = getattr(env, "reset_terminated", None)
    if reset_ids is not None:
        ids = reset_ids.nonzero(as_tuple=False).flatten()
        if len(ids) > 0:
            start_pos[ids] = asset.data.root_link_pos_w[ids]
            start_heading[ids] = _get_f_body_heading(env)[ids] - (torch.pi / 2)

    # 获取命令参数及时间
    cmd_term = env.command_manager._terms["slalom_cmd"]
    curvature = cmd_term.command[:, 4]
    vel_cmd = cmd_term.command[:, 0]
    t = env.episode_length_buf.float() * env.step_dt

    # 参考路径生成 (Base)
    omega = curvature * vel_cmd
    delta_theta = omega * t

    # chord = 2R·sin(Δθ/2) = v·t·sin(Δθ/2)/(Δθ/2) = v·t·sinc(Δθ/(2π))
    chord_length = vel_cmd * t * torch.sinc(delta_theta / (2 * torch.pi))

    # 参考点位置：起始点 + 弦长 × 方向向量（方向为 θ₀ + Δθ/2）
    ref_heading = start_heading + delta_theta / 2.0
    x_ref = start_pos[:, 0] + chord_length * torch.cos(ref_heading)
    y_ref = start_pos[:, 1] + chord_length * torch.sin(ref_heading)

    ref_xy = torch.stack([x_ref, y_ref], dim=1)  # [N, 2]

    # ---- 为 F_body 和 H_body 生成独立参考点 ----
    body_offset  = 0.065  # 身体节距 Base 的前后距离 (米)
    path_heading = start_heading + delta_theta                # 当前路径切线方向
    tangent = torch.stack([torch.cos(path_heading), torch.sin(path_heading)], dim=1)   # [N, 2]

    ref_f_body = ref_xy + body_offset * tangent
    ref_h_body = ref_xy - body_offset * tangent

    # 获取当前身体环节的 XY 位置
    base_xy   = asset.data.root_link_pos_w[:, :2]             # [N, 2]
    f_body_xy = asset.data.body_link_pos_w[:, _MODEL_INDICES.f_body_id, :2]  # [N, 2]
    h_body_xy = asset.data.body_link_pos_w[:, _MODEL_INDICES.h_body_id, :2]  # [N, 2]
    # 计算跟踪误差
    error_base  = torch.norm(base_xy - ref_xy, dim=1)
    error_fbody = torch.norm(f_body_xy - ref_f_body, dim=1)
    error_hbody = torch.norm(h_body_xy - ref_h_body, dim=1)
    # 获取课程学习量
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