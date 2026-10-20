from __future__ import annotations
import torch
from mjlab.entity import Entity
from typing import TYPE_CHECKING
from .indices import (
    _ACT_HIP_IDS,
    _ACT_LEG_IDS,
    _ACT_SPN_IDS,
    _ACT_NECK_IDS,
    _MODEL_INDICES,
    _ACT_JOINT_NAMES,
    _ACT_SHOULDER_IDS,
)
from .reference import (
    _BOX_EDGES,
    compute_path_ref,
    PROJECTION_STRIDE,
    get_path_curvature,
    sample_body_envelopes,
    project_body_envelopes,
    get_spatial_slalom_tables,
    get_reference_joint_state,
    get_f_body_physical_heading,
    _PHASE0_ENVELOPE_SAMPLE_STEP,
)
from .entity import POLE_RADIUS
from .config import CORRIDOR_HALF_WIDTH
from .curriculums import get_curriculum_reward_weight

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv



# 关节位置模仿奖励
def compute_mimic_pos_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取实际和期望关节位置
    asset: Entity = env.scene["robot"]
    joint_pos = asset.data.joint_pos[:, _MODEL_INDICES.joint_ids]
    ref_pos, _ = get_reference_joint_state(env)
    # 计算关节角度误差
    error = joint_pos - ref_pos
    error_leg = error[:, _ACT_LEG_IDS]
    error_spn = error[:, _ACT_SPN_IDS]
    error_neck = error[:, _ACT_NECK_IDS]
    # 获取课程学习量
    sigma_leg = get_curriculum_reward_weight(env, "sigma_leg_pos")
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_pos")
    weight = get_curriculum_reward_weight(env, "weight_mimic_pos")
    # 计算 MSE 误差
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    mse_neck = torch.mean(error_neck ** 2, dim=1)
    # 计算奖励
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward_neck = torch.exp(-sigma_spn * mse_neck)
    reward = (0.5 * reward_leg + 0.5 * reward_spn + 0.3 * reward_neck) * weight
    return reward



# 关节速度模仿奖励
def compute_mimic_vel_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取实际和期望关节位置
    asset: Entity = env.scene["robot"]
    joint_vel = asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
    _, ref_vel = get_reference_joint_state(env)
    # 计算关节速度误差
    error = joint_vel - ref_vel
    error_leg = error[:, _ACT_LEG_IDS]
    error_spn = error[:, _ACT_SPN_IDS]
    error_neck = error[:, _ACT_NECK_IDS]
    # 获取课程学习量
    sigma_leg = get_curriculum_reward_weight(env, "sigma_leg_vel")
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_vel")
    weight = get_curriculum_reward_weight(env, "weight_mimic_vel")
    # 计算 MSE 误差
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    mse_neck = torch.mean(error_neck ** 2, dim=1)
    # 计算奖励
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward_neck = torch.exp(-sigma_spn * mse_neck)
    reward = (0.5 * reward_leg + 0.5 * reward_spn + 0.3 * reward_neck) * weight
    return reward



# 线速度跟踪奖励
def compute_vel_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取实际与期望速度
    asset: Entity = env.scene["robot"]
    f_body_heading = get_f_body_physical_heading(env)
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



# 身体高度跟踪奖励 
def compute_height_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取实际与期望前后肢高度
    asset: Entity = env.scene["robot"]
    body_pos_w = asset.data.body_link_pos_w
    F_body_height = body_pos_w[:, _MODEL_INDICES.f_body_id, 2]
    H_body_height = body_pos_w[:, _MODEL_INDICES.h_body_id, 2]          
    cmd_term = env.command_manager._terms["slalom_cmd"]
    desired_height_F = cmd_term.command[:, 1]    
    desired_height_H = cmd_term.command[:, 2]
    # 计算高度跟踪误差    
    height_F_error = torch.abs(desired_height_F - F_body_height)
    height_H_error = torch.abs(desired_height_H - H_body_height)
    # 获取课程学习量
    sigma = get_curriculum_reward_weight(env, "sigma_height")
    weight = get_curriculum_reward_weight(env, "weight_height")
    # 计算奖励
    r_height_F = torch.exp(-sigma * height_F_error ** 2)
    r_height_H = torch.exp(-sigma * height_H_error ** 2)   
    reward = weight * (0.5 * r_height_F + 0.5 * r_height_H)
    # 记录日志
    env.extras["log"]["Data/height_actual"] = (0.5 * F_body_height + 0.5 * H_body_height).mean().item()
    return reward



# =========================================================================================
# 角速度跟踪奖励（ω_cmd = κ_cmd × v_cmd, 用 F_body 角速度避免万向节中心震荡）
def compute_omg_track_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    # F_body_Link 世界系角速度 Z 分量 → 前体偏航率
    actual_omega_z = asset.data.body_link_ang_vel_w[:, _MODEL_INDICES.f_body_id, 2]
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



# L1 动作平滑惩罚
def compute_action_L1_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取当前和上一步动作   
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    # 计算动作差
    action_diff = torch.abs(current_action - prev_action)
    # 获取课程学习量
    w_leg = get_curriculum_reward_weight(env, "weight_smooth_L1_leg")
    w_spn = get_curriculum_reward_weight(env, "weight_smooth_L1_spn")
    # 计算惩罚
    leg_cost = torch.sum(action_diff[:, _ACT_LEG_IDS], dim=1)
    spn_cost = torch.sum(action_diff[:, _ACT_SPN_IDS], dim=1)
    error_cost = torch.sum(action_diff[:, _ACT_NECK_IDS], dim=1)
    penalty = -w_leg * leg_cost - w_spn * spn_cost - w_spn * error_cost
    return penalty



# L2 动作平滑惩罚
def compute_action_L2_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取当前和上一步动作 
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    # 计算动作差
    action_diff = torch.square(current_action - prev_action)
    # 获取课程学习量
    w_leg = get_curriculum_reward_weight(env, "weight_smooth_L2_leg")
    w_spn = get_curriculum_reward_weight(env, "weight_smooth_L2_spn")
    # 计算惩罚
    leg_cost = torch.sum(action_diff[:, _ACT_LEG_IDS], dim=1)
    spn_cost = torch.sum(action_diff[:, _ACT_SPN_IDS], dim=1)
    error_cost = torch.sum(action_diff[:, _ACT_NECK_IDS], dim=1)
    penalty = -w_leg * leg_cost - w_spn * spn_cost - w_spn * error_cost
    return penalty



# =========================================================================================
# 能耗惩罚
def compute_energy_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取实际关节速度和力矩
    asset: Entity = env.scene["robot"]
    actuator_vel = asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
    actuator_torque = asset.data.actuator_force
    # 获取课程学习量
    weight = get_curriculum_reward_weight(env, "weight_energy")
    # 计算能量消耗
    cost = torch.sum(torch.abs(actuator_vel * actuator_torque), dim=1)
    # 计算惩罚
    penalty = -weight * cost
    return penalty



# 走廊一致性奖励 — 身体包络不超出参考路径周围的允许走廊
def compute_corridor_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 一阶段沿参考圆环，二阶段沿实际空间轨迹检查完整躯干包络。
    x_ref, y_ref, _, _, path_heading = compute_path_ref(env)
    ref_xy = torch.stack([x_ref, y_ref], dim=1)
    curvature = get_path_curvature(env)
    inner, outer = compute_phase0_corridor_excess(env, ref_xy, path_heading, curvature)
    cmd = env.command_manager._terms["slalom_cmd"]
    phase1_ids = cmd.phase1_mask.nonzero(as_tuple=False).flatten()  # type: ignore
    if phase1_ids.numel():
        inner[phase1_ids], outer[phase1_ids] = compute_phase1_corridor_excess(env, phase1_ids)
    # 动态一阶段回合按本回合实际路径投影, 恒定一阶段保留真实圆环
    schedule = getattr(cmd, "phase0_schedule", None)
    if schedule is not None:
        dynamic_ids = ((~cmd.phase1_mask) & schedule["dynamic"]).nonzero(as_tuple=False).flatten()  # type: ignore
        if dynamic_ids.numel():
            inner[dynamic_ids], outer[dynamic_ids] = compute_phase0_dynamic_corridor_excess(
                env, dynamic_ids)
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
        env._leg_saturation_limit = limit  # type: ignore
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
        columns = tuple(action_term.target_names.index(name) for name in _ACT_JOINT_NAMES)  # type: ignore
        env._abnormal_pose_action_columns = columns  # type: ignore
    requested = action_term.raw_action * action_term.scale + action_term.offset  # type: ignore
    requested = (requested - asset.data.encoder_bias[:, action_term.target_ids])[:, columns]  # type: ignore
    actual = asset.data.joint_pos[:, _MODEL_INDICES.joint_ids]
    tolerance = get_curriculum_reward_weight(env, "abnormal_leg_pose_tolerance")
    mask = env.command_manager._terms["slalom_cmd"].phase1_mask  # type: ignore
    actual_excess = _abnormal_leg_pose_excess(actual, ref_pos, tolerance) * mask
    request_excess = _abnormal_leg_pose_excess(requested, ref_pos, tolerance) * mask
    weight = get_curriculum_reward_weight(env, "weight_abnormal_leg_pose")
    env.extras["log"]["Data/异常腿部实际越界"] = actual_excess.mean().item()
    env.extras["log"]["Data/异常腿部请求越界"] = request_excess.mean().item()
    env.extras["log"]["Data/异常腿部姿态占比"] = (actual_excess > 0).float().mean().item()
    return -weight * (actual_excess.square() + request_excess.square())



# 走廊投影与越界计算

_PROJECTION_ENV_BLOCK = 64
_PHASE0_PROJECTION_STRIDE = 8




# 走廊超额计算
def _compute_corridor_projection(
    body_pos_xy: torch.Tensor,     # [N, 2] 身体中心在投影平面上的位置
    body_heading: torch.Tensor,    # [N]    身体物理前向朝向
    ref_pos_xy: torch.Tensor,      # [N, 2] 参考点位置
    path_heading: torch.Tensor,    # [N]    路径切线方向
    half_length: float,            # 身体半长
    half_width: float,             # 身体半宽
) -> tuple[torch.Tensor, torch.Tensor]:
    # 路径法向量（指向路径左侧）
    n_x = -torch.sin(path_heading)
    n_y = torch.cos(path_heading)

    # 侧向距离（带符号）
    dx = body_pos_xy[:, 0] - ref_pos_xy[:, 0]
    dy = body_pos_xy[:, 1] - ref_pos_xy[:, 1]
    d_lat = dx * n_x + dy * n_y  # [N]

    # 朝向误差导致的半宽膨胀
    delta_theta = body_heading - path_heading
    eff_hw = torch.abs(half_length * torch.sin(delta_theta)) + torch.abs(half_width * torch.cos(delta_theta))

    return d_lat, eff_hw


# 返回沿路径法向的最大身体包络距离，保留原有诊断口径。
def compute_corridor_excess(
    body_pos_xy: torch.Tensor,
    body_heading: torch.Tensor,
    ref_pos_xy: torch.Tensor,
    path_heading: torch.Tensor,
    half_length: float,
    half_width: float,
) -> torch.Tensor:
    d_lat, eff_hw = _compute_corridor_projection(
        body_pos_xy, body_heading, ref_pos_xy, path_heading, half_length, half_width
    )
    return d_lat.abs() + eff_hw


# 正曲率以内侧左法向计量，负曲率相反；直行使用对称走廊。
def compute_corridor_side_excess(
    body_pos_xy: torch.Tensor,
    body_heading: torch.Tensor,
    ref_pos_xy: torch.Tensor,
    path_heading: torch.Tensor,
    half_length: float,
    half_width: float,
    curvature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_lat, eff_hw = _compute_corridor_projection(
        body_pos_xy, body_heading, ref_pos_xy, path_heading, half_length, half_width
    )
    signed_lat = curvature.sign() * d_lat
    turning = curvature.abs() > 1e-6
    inner = torch.clamp(signed_lat + eff_hw - CORRIDOR_HALF_WIDTH, min=0.0)
    outer = torch.clamp(-signed_lat + eff_hw - CORRIDOR_HALF_WIDTH, min=0.0)
    straight = torch.clamp(d_lat.abs() + eff_hw - CORRIDOR_HALF_WIDTH, min=0.0)
    return torch.where(turning, inner, 0.0), torch.where(turning, outer, straight)




# 稳定计算包络点相对参考圆半径的径向偏差。
def _radial_offset(points: torch.Tensor, normal: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    scaled = curvature[:, None, None] * points - normal[:, None]
    numerator = (curvature.abs()[:, None] * points.square().sum(dim=-1)
                 - 2 * curvature.sign()[:, None] * (points * normal[:, None]).sum(dim=-1))
    return numerator / (torch.linalg.vector_norm(scaled, dim=-1) + 1)




# 检查整个投影盒到圆心的最近边界和最远角点，详见一阶段训练文档。
def compute_annulus_side_excess(
    vertices_xy: torch.Tensor,
    ref_xy: torch.Tensor,
    path_heading: torch.Tensor,
    curvature: torch.Tensor,
    pole_clearance: float = POLE_RADIUS,
) -> tuple[torch.Tensor, torch.Tensor]:
    normal = torch.stack((-path_heading.sin(), path_heading.cos()), dim=-1)
    relative = vertices_xy - ref_xy[:, None]
    turning = curvature.abs() > 1e-6
    safe_curvature = torch.where(turning, curvature, torch.ones_like(curvature))
    radius = safe_curvature.abs().reciprocal()

    # 圆心到十二条投影边的最近点包含多边形边界的最近点。
    starts = relative[:, [edge[0] for edge in _BOX_EDGES]]
    edges = relative[:, [edge[1] for edge in _BOX_EDGES]] - starts
    scaled_center_delta = normal[:, None] - safe_curvature[:, None, None] * starts
    denominator = safe_curvature[:, None] * edges.square().sum(dim=-1).clamp_min(1e-20)
    fraction = ((scaled_center_delta * edges).sum(dim=-1) / denominator).clamp(0, 1)
    nearest_points = starts + fraction[..., None] * edges
    inner_offset = _radial_offset(nearest_points, normal, safe_curvature).amin(dim=-1)
    outer_offset = _radial_offset(relative, normal, safe_curvature).amax(dim=-1)

    # 圆心落入完整投影多边形时，最近距离为零。
    generators = (relative[:, (4, 2, 1)] - relative[:, :1]) / 2
    axes = torch.stack((-generators[..., 1], generators[..., 0]), dim=-1)
    center_delta = normal - safe_curvature[:, None] * relative.mean(dim=1)
    projections = torch.einsum("nai,ni->na", axes, center_delta).abs()
    support = torch.einsum("nai,nbi->nab", axes, generators).abs().sum(dim=-1)
    inside = (projections <= safe_curvature.abs()[:, None] * support + 1e-12).all(dim=-1)
    inner_offset = torch.where(inside, -radius, inner_offset)
    min_distance = torch.where(inside, torch.zeros_like(radius), radius + inner_offset).clamp_min(0)
    inner = torch.maximum((-CORRIDOR_HALF_WIDTH - inner_offset).clamp_min(0),
                          (pole_clearance - min_distance).clamp_min(0))
    outer = (outer_offset - CORRIDOR_HALF_WIDTH).clamp_min(0)

    lateral = (relative * normal[:, None]).sum(dim=-1)
    straight = (lateral.abs().amax(dim=-1) - CORRIDOR_HALF_WIDTH).clamp_min(0)
    return torch.where(turning, inner, 0.0), torch.where(turning, outer, straight)




# 按实际前后体包络计算一阶段圆环越界，直行退化为对称带。
def compute_phase0_corridor_excess(
    env: ManagerBasedRlEnv,
    ref_xy: torch.Tensor,
    path_heading: torch.Tensor,
    curvature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    vertices = project_body_envelopes(env)
    batch_size = vertices.shape[0]
    inner, outer = compute_annulus_side_excess(
        vertices.reshape(-1, 8, 2),
        ref_xy[:, None].expand(-1, 2, -1).reshape(-1, 2),
        path_heading[:, None].expand(-1, 2).reshape(-1),
        curvature[:, None].expand(-1, 2).reshape(-1),
    )
    return inner.reshape(batch_size, 2), outer.reshape(batch_size, 2)




# 以固定形状分块查询，避免阶段切换时不同大批量形状累积显存预留。
def project_slalom_points(
    points: torch.Tensor, tables: torch.Tensor, rows: torch.Tensor, spacing: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = points.shape[0]
    outputs = (
        points.new_empty(points.shape[:2]),
        points.new_empty(points.shape[:2]),
        points.new_empty(points.shape[:2]),
    )
    if count == 0:
        return outputs
    block_size = min(_PROJECTION_ENV_BLOCK, 1 << (count - 1).bit_length())
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        block_points, block_rows, block_spacing = points[start:stop], rows[start:stop], spacing[start:stop]
        padding = block_size - (stop - start)
        if padding:
            block_points = torch.cat((block_points, block_points[-1:].expand(padding, -1, -1)))
            block_rows = torch.cat((block_rows, block_rows[-1:].expand(padding)))
            block_spacing = torch.cat((block_spacing, block_spacing[-1:].expand(padding)))
        values = _project_slalom_points_block(block_points, tables, block_rows, block_spacing)
        for output, value in zip(outputs, values, strict=True):
            output[start:stop].copy_(value[:stop - start])
    return outputs




# 粗定位后在线段上细化投影，接近段起点向后延伸为切线射线。
def _project_slalom_points_block(
    points: torch.Tensor, tables: torch.Tensor, rows: torch.Tensor, spacing: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points = points.clone()
    period_x = 2 * spacing[:, None]
    cycles = torch.floor((points[..., 0] - period_x) / period_x).clamp_min(0)
    points[..., 0] -= cycles * period_x
    coarse = tables[rows, ::PROJECTION_STRIDE, :2]
    distance_sq = (points.square().sum(-1, keepdim=True) + coarse.square().sum(-1)[:, None]
                   - 2 * torch.bmm(points, coarse.transpose(1, 2)))
    nearest = distance_sq.argmin(dim=-1) * PROJECTION_STRIDE
    offsets = torch.arange(-PROJECTION_STRIDE, PROJECTION_STRIDE, device=points.device)
    indices = (nearest[..., None] + offsets).clamp(0, tables.shape[1] - 2)
    start = tables[rows[:, None, None], indices]
    end = tables[rows[:, None, None], indices + 1]
    edges = end[..., :2] - start[..., :2]
    delta = points[:, :, None] - start[..., :2]
    fraction = ((delta * edges).sum(-1) / edges.square().sum(-1).clamp_min(1e-20)).clamp(0, 1)
    residual = delta - fraction[..., None] * edges
    best = residual.square().sum(-1).argmin(-1, keepdim=True)
    chosen_residual = residual.gather(2, best[..., None].expand(-1, -1, -1, 2)).squeeze(2)
    chosen_edge = edges.gather(2, best[..., None].expand(-1, -1, -1, 2)).squeeze(2)
    curvature = torch.lerp(start[..., 3], end[..., 3], fraction).gather(2, best).squeeze(2)
    distance = torch.linalg.vector_norm(chosen_residual, dim=-1)
    lateral = chosen_edge[..., 0] * chosen_residual[..., 1] - chosen_edge[..., 1] * chosen_residual[..., 0]

    origin = tables[rows, 0]
    tangent = torch.stack((origin[:, 2].cos(), origin[:, 2].sin()), dim=-1)
    ray_delta = points - origin[:, None, :2]
    ray_residual = ray_delta - (ray_delta * tangent[:, None]).sum(-1).clamp_max(0)[..., None] * tangent[:, None]
    ray_distance = torch.linalg.vector_norm(ray_residual, dim=-1)
    use_ray = ray_distance < distance
    ray_lateral = tangent[:, None, 0] * ray_residual[..., 1] - tangent[:, None, 1] * ray_residual[..., 0]
    return (torch.minimum(distance, ray_distance), torch.where(use_ray, ray_lateral, lateral),
            torch.where(use_ray, 0.0, curvature))




# 二阶段按各包络点的最近实际轨迹计算内外越界，不依赖参考时钟的位置。
def compute_phase1_corridor_excess(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    vertices = project_body_envelopes(env)[env_ids]
    points = sample_body_envelopes(env, vertices)
    origins = getattr(env.scene, "env_origins", None)
    if origins is not None:
        points = points - origins[env_ids, None, None, :2]
    tables = getattr(env, "_slalom_spatial_tables", None)
    if tables is None:
        tables = torch.as_tensor(get_spatial_slalom_tables(), device=env.device, dtype=points.dtype)
        env._slalom_spatial_tables = tables  # type: ignore
    cmd = env.command_manager._terms["slalom_cmd"]
    distance, lateral, curvature = project_slalom_points(
        points.flatten(1, 2), tables, cmd.slalom_curvature_index[env_ids], cmd.pole_spacing[env_ids])  # type: ignore
    excess = (distance - CORRIDOR_HALF_WIDTH).clamp_min(0)
    inner_side = curvature * lateral > 0
    inner = torch.where(inner_side, excess, 0.0).reshape(points.shape[:-1]).amax(-1)
    outer = torch.where(inner_side, 0.0, excess).reshape(points.shape[:-1]).amax(-1)
    return inner, outer




# 一阶段路径投影: 返回包络点到本回合实际路径 (含起终点切线延伸) 的距离、侧向叉积与局部曲率
def project_phase0_path_points(points: torch.Tensor, path_xy: torch.Tensor,
                               path_heading: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stride = _PHASE0_PROJECTION_STRIDE
    coarse = path_xy[:, ::stride]
    distance_sq = (points.square().sum(-1, keepdim=True) + coarse.square().sum(-1)[:, None]
                   - 2.0 * torch.bmm(points, coarse.transpose(1, 2)))
    nearest = distance_sq.argmin(dim=-1) * stride
    offsets = torch.arange(-stride, stride, device=points.device)
    indices = (nearest[..., None] + offsets).clamp(0, path_xy.shape[1] - 2)
    rows = torch.arange(points.shape[0], device=points.device)[:, None, None]
    start = path_xy[rows, indices]
    end = path_xy[rows, indices + 1]
    edges = end - start
    delta = points[:, :, None] - start
    fraction = ((delta * edges).sum(-1) / edges.square().sum(-1).clamp_min(1e-20)).clamp(0.0, 1.0)
    residual = delta - fraction[..., None] * edges
    best = residual.square().sum(-1).argmin(-1, keepdim=True)
    chosen_residual = residual.gather(2, best[..., None].expand(-1, -1, -1, 2)).squeeze(2)
    chosen_edge = edges.gather(2, best[..., None].expand(-1, -1, -1, 2)).squeeze(2)
    distance = torch.linalg.vector_norm(chosen_residual, dim=-1)
    lateral = chosen_edge[..., 0] * chosen_residual[..., 1] - chosen_edge[..., 1] * chosen_residual[..., 0]
    head_start = path_heading[rows, indices]
    head_end = path_heading[rows, indices + 1]
    segment_length = torch.linalg.vector_norm(chosen_edge, dim=-1).clamp_min(1e-9)
    curvature = (head_end - head_start).gather(2, best).squeeze(2) / segment_length
    # 起终点切线延伸: 起点向后、终点向前各作一条射线, 身体跨出路径端点时按射线量距
    for position, extend_back in ((0, True), (-1, False)):
        tangent = torch.stack((torch.cos(path_heading[:, position]),
                               torch.sin(path_heading[:, position])), dim=-1)
        offset = points - path_xy[:, position][:, None, :]
        along = (offset * tangent[:, None]).sum(-1)
        along = along.clamp_max(0.0) if extend_back else along.clamp_min(0.0)
        ray_residual = offset - along[..., None] * tangent[:, None]
        ray_distance = torch.linalg.vector_norm(ray_residual, dim=-1)
        ray_lateral = tangent[:, None, 0] * ray_residual[..., 1] - tangent[:, None, 1] * ray_residual[..., 0]
        use_ray = ray_distance < distance
        distance = torch.where(use_ray, ray_distance, distance)
        lateral = torch.where(use_ray, ray_lateral, lateral)
        curvature = torch.where(use_ray, torch.zeros_like(curvature), curvature)
    return distance, lateral, curvature




# 一阶段动态回合: 按各包络采样点到本回合实际路径的最近距离计算内外越界
def compute_phase0_dynamic_corridor_excess(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    vertices = project_body_envelopes(env)[env_ids]
    points = sample_body_envelopes(env, vertices, _PHASE0_ENVELOPE_SAMPLE_STEP)
    origins = getattr(env.scene, "env_origins", None)
    if origins is not None:
        points = points - origins[env_ids, None, None, :2]
    path_xy = getattr(env, "_phase0_path_xy")[env_ids]
    path_heading = getattr(env, "_phase0_path_heading")[env_ids]
    flat = points.flatten(1, 2)
    # 分块投影, 限制大中间张量的峰值显存
    distances, laterals, curvatures = [], [], []
    for block in range(0, len(env_ids), _PROJECTION_ENV_BLOCK):
        section = slice(block, min(block + _PROJECTION_ENV_BLOCK, len(env_ids)))
        distance, lateral, curvature = project_phase0_path_points(
            flat[section], path_xy[section], path_heading[section])
        distances.append(distance)
        laterals.append(lateral)
        curvatures.append(curvature)
    distance = torch.cat(distances)
    lateral = torch.cat(laterals)
    curvature = torch.cat(curvatures)
    excess = (distance - CORRIDOR_HALF_WIDTH).clamp_min(0.0)
    inner_side = curvature * lateral > 0.0
    inner = torch.where(inner_side, excess, torch.zeros_like(excess)).reshape(points.shape[:-1]).amax(-1)
    outer = torch.where(inner_side, torch.zeros_like(excess), excess).reshape(points.shape[:-1]).amax(-1)
    return inner, outer
