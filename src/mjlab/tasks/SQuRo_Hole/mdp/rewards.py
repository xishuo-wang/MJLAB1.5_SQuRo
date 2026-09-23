from __future__ import annotations
import math
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from typing import TYPE_CHECKING
from .curriculums import get_curriculum_reward_weight
from .command import HEIGHT_THRESHOLD, BASE_HEIGHT
from .reference import (
    F_BODY_ID,
    FRONT_SEG_SITE_NAMES,
    H_BODY_ID,
    REAR_SEG_SITE_NAMES,
    REF_FRONT_IDS,
    REF_HIND_IDS,
    REF_SPINE_IDS,
    get_reference_joint_pos,
    get_reference_joint_vel,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_FOOT_SITE_NAMES = ["FL_elbow_site", "FR_elbow_site", "HL_knee_site", "HR_knee_site"]

# 参考表列口径: 前腿 4 + 后腿 4 + 脊柱 4 = 12 个被控关节
_REF_COL_FRONT = (0, 1, 2, 3)
_REF_COL_HIND = (4, 5, 6, 7)
_REF_COL_SPINE = (8, 9, 10, 11)
_JOINT_ORDER = REF_FRONT_IDS + REF_HIND_IDS + REF_SPINE_IDS

# 虚拟碰撞采样 site (前段 9 + 后段 9), 首次使用时按名字解析
_SEG_SITE_IDS: tuple[int, ...] | None = None


# 解析前后段的采样 site 索引 (9 + 9)
def _resolve_seg_sites(asset: Entity) -> tuple[int, ...]:
    global _SEG_SITE_IDS
    if _SEG_SITE_IDS is None:
        site_ids, _ = asset.find_sites(
            list(FRONT_SEG_SITE_NAMES) + list(REAR_SEG_SITE_NAMES), preserve_order=True)
        _SEG_SITE_IDS = tuple(site_ids)
    return _SEG_SITE_IDS


# 取 12 个被控关节的实际位置
def _controlled_joint_pos(asset: Entity) -> torch.Tensor:
    return asset.data.joint_pos[:, list(_JOINT_ORDER)]


# 取 12 个被控关节的实际速度
def _controlled_joint_vel(asset: Entity) -> torch.Tensor:
    return asset.data.joint_vel[:, list(_JOINT_ORDER)]


# 位置模仿奖励: 关节位置对参考表的 MSE 取 exp 核
def compute_mimic_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    current_pos = _controlled_joint_pos(asset)
    target_joint_pos = get_reference_joint_pos(env)
    pos_errors = current_pos - target_joint_pos
    mse_errors = torch.mean(pos_errors ** 2, dim=1)
    sigma = get_curriculum_reward_weight(env, "mimic_pos_sigma")
    weight = get_curriculum_reward_weight(env, "mimic_pos")
    return torch.exp(-sigma * mse_errors) * weight


# 速度模仿奖励: 关节速度对参考表的 MSE 取 exp 核
def compute_mimic_velocity_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    current_vel = _controlled_joint_vel(asset)
    target_joint_vel = get_reference_joint_vel(env)
    vel_errors = current_vel - target_joint_vel
    mse_errors = torch.mean(vel_errors ** 2, dim=1)
    sigma = get_curriculum_reward_weight(env, "mimic_vel_sigma")
    weight = get_curriculum_reward_weight(env, "mimic_vel")
    return torch.exp(-sigma * mse_errors) * weight


# X 线速度跟踪奖励 (世界系 x 速度)
def compute_linear_velocity_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    actual_vel_x = asset.data.root_link_lin_vel_w[:, 0]
    cmd_term = env.command_manager._terms["hole_cmd"]  # type: ignore[union-attr]
    desired_vel_x = cmd_term.command[:, 0]
    vel_error = actual_vel_x - desired_vel_x
    weight = get_curriculum_reward_weight(env, "velocity")

    reward = torch.exp(-100.0 * vel_error ** 2)
    # 有速度指令却几乎不动 → 该项置 -1
    stalled = (desired_vel_x.abs() > 0.01) & (actual_vel_x.abs() < 0.01)
    reward = torch.where(stalled, torch.full_like(reward, -1.0), reward)

    env.extras["log"]["Metrics/actual_vel_x_mean"] = actual_vel_x.mean().item()
    env.extras["log"]["Metrics/cmd_vel_x_mean"] = desired_vel_x.mean().item()
    env.extras["log"]["Metrics/vel_error_mean"] = vel_error.abs().mean().item()
    return reward * weight


# 高度奖励: 前/后躯干 + 基准躯干 (0.4/0.4/0.2)
def compute_height_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    body_pos_w = asset.data.body_link_pos_w
    f_body_height = body_pos_w[:, F_BODY_ID, 2]
    h_body_height = body_pos_w[:, H_BODY_ID, 2]
    base_height = asset.data.root_link_pos_w[:, 2]

    cmd_term = env.command_manager._terms["hole_cmd"]  # type: ignore[union-attr]
    desired_height_F = cmd_term.command[:, 3]
    desired_height_H = cmd_term.command[:, 4]
    desired_height = (desired_height_F + desired_height_H) / 2

    height_F_error = torch.abs(desired_height_F - f_body_height)
    height_H_error = torch.abs(desired_height_H - h_body_height)
    height_base_error = torch.abs(desired_height - base_height)

    sigma = get_curriculum_reward_weight(env, "height_sigma")
    weight = get_curriculum_reward_weight(env, "height")
    reward = (0.4 * torch.exp(-sigma * height_F_error ** 2)
              + 0.4 * torch.exp(-sigma * height_H_error ** 2)
              + 0.2 * torch.exp(-sigma * height_base_error ** 2))

    env.extras["log"]["Metrics/height_F_error_mean"] = height_F_error.mean().item()
    env.extras["log"]["Metrics/height_H_error_mean"] = height_H_error.mean().item()
    env.extras["log"]["Metrics/height_base_error_mean"] = height_base_error.mean().item()
    return reward * weight


# 抬脚高度奖励: 仅 Mode0 (前后肢都高) 触发
def compute_foot_clearance_reward(env: ManagerBasedRlEnv,
                                  target_base_height: float = 0.01) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    cmd_term = env.command_manager._terms["hole_cmd"]  # type: ignore[union-attr]
    height_F = cmd_term.command[:, 3]
    height_H = cmd_term.command[:, 4]
    mode0 = (height_F >= HEIGHT_THRESHOLD) & (height_H >= HEIGHT_THRESHOLD)

    foot_site_ids, _ = asset.find_sites(_FOOT_SITE_NAMES, preserve_order=True)
    foot_z = asset.data.site_pos_w[:, list(foot_site_ids), 2]
    base_height = asset.data.root_link_pos_w[:, 2]
    swing_target = base_height / BASE_HEIGHT * target_base_height
    err = torch.abs(foot_z - swing_target.unsqueeze(1))
    reward = torch.exp(-100.0 * err ** 2).mean(dim=1)
    weight = get_curriculum_reward_weight(env, "foot_clearance")
    return torch.where(mode0, reward, torch.zeros_like(reward)) * weight


# 目标位置接近奖励: 稠密高斯, 距 x=1.5 越近越大
def compute_reached_reward(env: ManagerBasedRlEnv, target_x: float = 1.5,
                           sigma: float = 10.0) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    current_x = asset.data.root_link_pos_w[:, 0]
    distance = current_x - target_x
    weight = get_curriculum_reward_weight(env, "reached")
    return torch.exp(-sigma * distance ** 2) * weight


# 由四元数解 yaw
def _quaternion_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# 由四元数解 roll
def _quaternion_to_roll(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


# 朝向奖励: root yaw 对齐前进方向 (body frame 有 90 度偏置)
def compute_orientation_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    quat = asset.data.root_link_quat_w
    yaw = _quaternion_to_yaw(quat) - math.pi / 2
    error = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    weight = get_curriculum_reward_weight(env, "orientation")
    return torch.exp(-5.0 * error ** 2) * weight


# 身体角度奖励: 按前后肢高低分情况约束躯干 roll
def compute_angle_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    cmd_term = env.command_manager._terms["hole_cmd"]  # type: ignore[union-attr]
    height_F = cmd_term.command[:, 3]
    height_H = cmd_term.command[:, 4]

    f_body_quat = asset.data.body_link_quat_w[:, F_BODY_ID]
    h_body_quat = asset.data.body_link_quat_w[:, H_BODY_ID]
    f_roll = _quaternion_to_roll(f_body_quat)
    h_roll = _quaternion_to_roll(h_body_quat)

    low_F = height_F < HEIGHT_THRESHOLD
    low_H = height_H < HEIGHT_THRESHOLD
    # 低高度那一侧需要侧倾; 都高时都保持竖直
    target_f = torch.where(low_F, torch.full_like(f_roll, -math.pi / 2), torch.zeros_like(f_roll))
    target_h = torch.where(low_H, torch.full_like(h_roll, math.pi / 2), torch.zeros_like(h_roll))
    err_f = torch.atan2(torch.sin(f_roll - target_f), torch.cos(f_roll - target_f))
    err_h = torch.atan2(torch.sin(h_roll - target_h), torch.cos(h_roll - target_h))

    both_high = ~low_F & ~low_H
    sigma = torch.where(both_high, torch.full_like(err_f, 100.0), torch.full_like(err_f, 50.0))
    reward = (torch.exp(-sigma * err_f ** 2) + torch.exp(-sigma * err_h ** 2)) / 2
    weight = get_curriculum_reward_weight(env, "angle")
    return reward * weight


# 动作平滑惩罚 (一阶差分平方和)
def compute_smoothness_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    weight = get_curriculum_reward_weight(env, "smoothness")
    return -weight * torch.sum((current_action - prev_action) ** 2, dim=1)


# 能耗惩罚: sum |tau * qdot|
def compute_energy_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    actuator_vel = _controlled_joint_vel(asset)
    actuator_torque = asset.data.actuator_force
    return -0.05 * torch.sum(torch.abs(actuator_vel * actuator_torque), dim=1)


# 运输代价 (COT) 惩罚
def compute_cot_penalty(env: ManagerBasedRlEnv, total_mass: float = 2.4525) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    actuator_vel = _controlled_joint_vel(asset)
    actuator_torque = asset.data.actuator_force
    vel_x = asset.data.root_link_lin_vel_w[:, 0]
    power = torch.sum(torch.abs(actuator_vel * actuator_torque), dim=1)
    cot = power / (total_mass * vel_x.abs() + 1e-9)
    return -0.1 * cot


# 虚拟碰撞惩罚 (软限高): 躯干进入限高板 x 区间时, 统计采样 site 超出板底阈值的量
def compute_body_contact_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    device = env.device
    num_envs = env.num_envs
    seg_site_ids = _resolve_seg_sites(asset)

    # 三块限高板的 x 区间与板底高度 (与 env_cfg 的 Hole1/2/3 及 docs 对齐)
    obs_x_min = torch.tensor([0.185, 0.5, 1.185], device=device)
    obs_x_max = torch.tensor([0.215, 0.7, 1.215], device=device)
    obs_z_thresh = torch.tensor([0.045, 0.07, 0.045], device=device)

    body_x = asset.data.body_link_pos_w[:, [F_BODY_ID, H_BODY_ID], 0]
    in_obs = (body_x.unsqueeze(-1) >= obs_x_min) & (body_x.unsqueeze(-1) <= obs_x_max)
    assert in_obs.shape[0] == num_envs, (
        f"[Hole] body_contact 形状不一致: env.num_envs={num_envs}, "
        f"asset batch={in_obs.shape[0]}, site batch={asset.data.site_pos_w.shape[0]}")
    if not in_obs.any():
        return torch.zeros(num_envs, device=device)

    active_z_thresh = in_obs.float() @ obs_z_thresh
    # in_any_obs: [N, 2] 每段是否进入任一块板; reshape 成 [N, 2, 1] 以广播到 9 个 site
    in_any_obs = in_obs.any(dim=-1).reshape(num_envs, 2).unsqueeze(-1)
    # 每段 9 个采样 site: 按名字解析 (旧版硬编码 index 已随模型变化, 见 docs §5)
    all_sites_z = asset.data.site_pos_w[:, list(seg_site_ids), 2].view(num_envs, 2, 9)

    excess = torch.clamp(all_sites_z - active_z_thresh.unsqueeze(-1), min=0.0)
    excess = excess * in_any_obs
    total_penalty = excess.sum(dim=(1, 2))

    weight = get_curriculum_reward_weight(env, "body_contact")
    env.extras["log"]["Metrics/body_contact_penalty"] = total_penalty.mean().item()
    return -total_penalty * weight * 10.0


# 防前腿不动: Mode2 (前高后低) 时惩罚前肢关节速度不足
def compute_stop_reward(env: ManagerBasedRlEnv, min_velocity: float = 0.5) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    cmd_term = env.command_manager._terms["hole_cmd"]  # type: ignore[union-attr]
    height_F = cmd_term.command[:, 3]
    height_H = cmd_term.command[:, 4]
    mode2 = (height_F >= HEIGHT_THRESHOLD) & (height_H < HEIGHT_THRESHOLD)

    front_vel = asset.data.joint_vel[:, REF_FRONT_IDS].abs()
    deficit = torch.clamp(min_velocity - front_vel, min=0.0).sum(dim=1)
    return torch.where(mode2, -deficit * 2.0, torch.zeros_like(deficit))
