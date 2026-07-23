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
# 路径跟踪奖励 — 基于曲率命令生成期望圆弧/直线，跟踪 base+F_body+H_body
def compute_path_track_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]

    # 初始化路径状态（episode 开始时记录 base 初始位置和 F_body 朝向）
    if getattr(env, "_path_state", None) is None:
        env._path_state = {  # type: ignore[attr-defined]
            "start_pos": asset.data.root_link_pos_w.clone(),
            "start_heading": _get_f_body_heading(env).clone(),
        }

    state = env._path_state  # type: ignore[attr-defined]
    start_pos = state["start_pos"]   # [N, 3]
    start_heading = state["start_heading"]  # [N]

    # 重置已终止环境
    reset_ids = getattr(env, "reset_terminated", None)
    if reset_ids is not None:
        ids = reset_ids.nonzero(as_tuple=False).flatten()
        if len(ids) > 0:
            start_pos[ids] = asset.data.root_link_pos_w[ids]
            start_heading[ids] = _get_f_body_heading(env)[ids]

    # 从命令获取路径参数
    cmd_term = env.command_manager._terms["slalom_cmd"]
    curvature = cmd_term.command[:, 4]    # κ [N]
    vel_cmd = cmd_term.command[:, 0]      # v [N]

    # 时间
    t = env.episode_length_buf.float() * env.step_dt  # [N]

    # 圆弧参数
    omega = curvature * vel_cmd                                    # ω = κ×v [N]
    delta_theta = omega * t                                       # Δθ [N]
    R = torch.where(torch.abs(curvature) > 1e-6, 1.0 / curvature,
                    torch.full_like(curvature, 1e6))              # [N]

    # 参考位置: 直行 vs 圆弧
    is_straight = torch.abs(curvature) < 1e-6
    # 直行: x = x0 + v*t*cos(θ0),  y = y0 + v*t*sin(θ0)
    x_s = start_pos[:, 0] + vel_cmd * t * torch.cos(start_heading)
    y_s = start_pos[:, 1] + vel_cmd * t * torch.sin(start_heading)
    # 圆弧: x = x0 + R*(sin(θ0+Δθ)-sin(θ0)),  y = y0 - R*(cos(θ0+Δθ)-cos(θ0))
    x_c = start_pos[:, 0] + R * (torch.sin(start_heading + delta_theta) - torch.sin(start_heading))
    y_c = start_pos[:, 1] - R * (torch.cos(start_heading + delta_theta) - torch.cos(start_heading))

    x_ref = torch.where(is_straight, x_s, x_c)
    y_ref = torch.where(is_straight, y_s, y_c)

    # 三个身体环节的当前位置
    base_xy = asset.data.root_link_pos_w[:, :2]                                     # [N, 2]
    f_body_xy = asset.data.body_link_pos_w[:, _MODEL_INDICES.f_body_id, :2]         # [N, 2]
    h_body_xy = asset.data.body_link_pos_w[:, _MODEL_INDICES.h_body_id, :2]         # [N, 2]
    ref_xy = torch.stack([x_ref, y_ref], dim=1)                                      # [N, 2]

    # 跟踪误差
    base_err = torch.norm(base_xy - ref_xy, dim=1)
    f_body_err = torch.norm(f_body_xy - ref_xy, dim=1)
    h_body_err = torch.norm(h_body_xy - ref_xy, dim=1)

    # 奖励
    sigma = get_curriculum_reward_weight(env, "sigma_path_track")
    w_base = get_curriculum_reward_weight(env, "weight_path_base")
    w_fbody = get_curriculum_reward_weight(env, "weight_path_fbody")
    w_hbody = get_curriculum_reward_weight(env, "weight_path_hbody")

    r_base = torch.exp(-sigma * base_err ** 2)
    r_fbody = torch.exp(-sigma * f_body_err ** 2)
    r_hbody = torch.exp(-sigma * h_body_err ** 2)

    env.extras["log"]["Data/path_base_err"] = base_err.mean().item()
    env.extras["log"]["Data/path_fbody_err"] = f_body_err.mean().item()
    return w_base * r_base + w_fbody * r_fbody + w_hbody * r_hbody