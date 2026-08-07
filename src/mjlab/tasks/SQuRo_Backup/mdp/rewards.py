from __future__ import annotations
import torch
from mjlab.entity import Entity
from typing import TYPE_CHECKING
from .indices import _MODEL_INDICES
from .reference import get_reference_joint_state

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# 奖励超参数（跌倒爬起第一阶段: 以模仿为主, 辅以竖直/高度引导）
_WEIGHT_MIMIC_POS = 1.0
_WEIGHT_MIMIC_VEL = 1.0
_WEIGHT_UPRIGHT = 1.0
_WEIGHT_HEIGHT = 1.0
_SIGMA_LEG_POS = 5.0
_SIGMA_SPN_POS = 10.0
_SIGMA_NECK_POS = 5.0
_SIGMA_LEG_VEL = 0.1
_SIGMA_SPN_VEL = 0.1
_SIGMA_NECK_VEL = 0.1
_SIGMA_UPRIGHT = 5.0
_SIGMA_HEIGHT = 500.0
_TARGET_HEIGHT = 0.055   # 站立时 F/H body 期望高度 (m)


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
    reward_leg = torch.exp(-_SIGMA_LEG_POS * mse_leg)
    reward_spn = torch.exp(-_SIGMA_SPN_POS * mse_spn)
    reward_neck = torch.exp(-_SIGMA_NECK_POS * mse_neck)
    reward = (reward_leg + reward_spn) / 2 + 0.3 * reward_neck
    return reward * _WEIGHT_MIMIC_POS


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
    reward_leg = torch.exp(-_SIGMA_LEG_VEL * mse_leg)
    reward_spn = torch.exp(-_SIGMA_SPN_VEL * mse_spn)
    reward_neck = torch.exp(-_SIGMA_NECK_VEL * mse_neck)
    reward = (reward_leg + reward_spn) / 2 + 0.3 * reward_neck
    return reward * _WEIGHT_MIMIC_VEL


# =========================================================================================
# 身体竖直奖励 — root 的 body+Z 与重力反方向对齐程度 (站立≈+1, 仰面≈-1, 侧躺≈0)
def compute_upright_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    uprightness = asset.data.projected_gravity_b[:, 2]  # [N]
    reward = torch.exp(-_SIGMA_UPRIGHT * (1.0 - uprightness) ** 2)
    env.extras["log"]["Data/uprightness"] = uprightness.mean().item()
    return reward * _WEIGHT_UPRIGHT


# =========================================================================================
# 身体高度奖励 — F/H body 高度跟踪站立高度 (躺地≈0.02, 站起≈0.055)
def compute_height_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    body_pos_w = asset.data.body_link_pos_w
    h_f = body_pos_w[:, _MODEL_INDICES.f_body_id, 2]
    h_h = body_pos_w[:, _MODEL_INDICES.h_body_id, 2]
    r_f = torch.exp(-_SIGMA_HEIGHT * (h_f - _TARGET_HEIGHT) ** 2)
    r_h = torch.exp(-_SIGMA_HEIGHT * (h_h - _TARGET_HEIGHT) ** 2)
    env.extras["log"]["Data/height_actual"] = (0.5 * h_f + 0.5 * h_h).mean().item()
    return _WEIGHT_HEIGHT * (0.5 * r_f + 0.5 * r_h)
