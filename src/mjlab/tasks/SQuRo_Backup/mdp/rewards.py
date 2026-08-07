from __future__ import annotations
import torch
from mjlab.entity import Entity
from typing import TYPE_CHECKING
from .indices import _MODEL_INDICES
from .reference import get_reference_joint_state
from .curriculums import get_curriculum_reward_weight

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# 奖励权重/σ 由 curriculums.RewardWeightCurriculum 按训练阶段提供
_TARGET_HEIGHT = 0.055   # 站立时 F/H body 期望高度 (m, 不参与课程)


# =========================================================================================
# 关节位置模仿奖励
def compute_mimic_pos_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    joint_pos = asset.data.joint_pos[:, _MODEL_INDICES.joint_ids]
    ref_pos, _ = get_reference_joint_state(env)
    error = joint_pos - ref_pos
    error_leg = error[:, _MODEL_INDICES.actuator_leg_ids]
    error_spn = error[:, _MODEL_INDICES.actuator_spn_ids]
    error_neck = error[:, _MODEL_INDICES.actuator_neck_ids]
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    mse_neck = torch.mean(error_neck ** 2, dim=1)
    weight = get_curriculum_reward_weight(env, "weight_mimic_pos")
    sigma_leg = get_curriculum_reward_weight(env, "sigma_leg_pos")
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_pos")
    sigma_neck = get_curriculum_reward_weight(env, "sigma_neck_pos")
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward_neck = torch.exp(-sigma_neck * mse_neck)
    reward = (reward_leg + reward_spn) / 2 + 0.3 * reward_neck
    return reward * weight


# =========================================================================================
# 关节速度模仿奖励
def compute_mimic_vel_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    joint_vel = asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
    _, ref_vel = get_reference_joint_state(env)
    error = joint_vel - ref_vel
    error_leg = error[:, _MODEL_INDICES.actuator_leg_ids]
    error_spn = error[:, _MODEL_INDICES.actuator_spn_ids]
    error_neck = error[:, _MODEL_INDICES.actuator_neck_ids]
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    mse_neck = torch.mean(error_neck ** 2, dim=1)
    weight = get_curriculum_reward_weight(env, "weight_mimic_vel")
    sigma_leg = get_curriculum_reward_weight(env, "sigma_leg_vel")
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_vel")
    sigma_neck = get_curriculum_reward_weight(env, "sigma_neck_vel")
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward_neck = torch.exp(-sigma_neck * mse_neck)
    reward = (reward_leg + reward_spn) / 2 + 0.3 * reward_neck
    return reward * weight


# =========================================================================================
# 身体竖直奖励 — root 的 body+Z 与重力反方向对齐程度 (站立≈+1, 仰面≈-1, 侧躺≈0)
def compute_upright_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    uprightness = asset.data.projected_gravity_b[:, 2]  # [N]
    sigma = get_curriculum_reward_weight(env, "sigma_upright")
    weight = get_curriculum_reward_weight(env, "weight_upright")
    reward = torch.exp(-sigma * (1.0 - uprightness) ** 2)
    env.extras["log"]["Data/uprightness"] = uprightness.mean().item()
    return reward * weight


# =========================================================================================
# 身体高度奖励 — F/H body 高度跟踪站立高度 (躺地≈0.02, 站起≈0.055)
def compute_height_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    body_pos_w = asset.data.body_link_pos_w
    h_f = body_pos_w[:, _MODEL_INDICES.f_body_id, 2]
    h_h = body_pos_w[:, _MODEL_INDICES.h_body_id, 2]
    sigma = get_curriculum_reward_weight(env, "sigma_height")
    weight = get_curriculum_reward_weight(env, "weight_height")
    r_f = torch.exp(-sigma * (h_f - _TARGET_HEIGHT) ** 2)
    r_h = torch.exp(-sigma * (h_h - _TARGET_HEIGHT) ** 2)
    env.extras["log"]["Data/height_actual"] = (0.5 * h_f + 0.5 * h_h).mean().item()
    return weight * (0.5 * r_f + 0.5 * r_h)


# =========================================================================================
# 站起成功奖励 — 身体竖直 (uprightness>0.9) 且高度达标 (height>0.05) 时每步 +1.0
# 与站起成功终止配合, 直接激励"尽快完成复位"
def compute_stand_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    up = asset.data.projected_gravity_b[:, 2]  # [N]
    body_pos_w = asset.data.body_link_pos_w
    h = 0.5 * (body_pos_w[:, _MODEL_INDICES.f_body_id, 2] + body_pos_w[:, _MODEL_INDICES.h_body_id, 2])
    standing = (up > 0.9) & (h > 0.05)  # [N] bool
    weight = get_curriculum_reward_weight(env, "weight_stand")
    env.extras["log"]["Data/stand_success"] = standing.float().mean().item()
    return standing.float() * weight
