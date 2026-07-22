"""SQuRo绕杆任务第一阶段 — 机动基元奖励函数

奖励构成:
  R = R_vel + R_omega + R_spine + R_stability + R_energy + R_smoothness

脊柱关节（万向节结构）:
  F_spine1 (joint[1]): 侧摆, Z轴, ±0.6rad  — 转弯主驱动
  F_body    (joint[3]): 扭转, X轴, ±1.57rad — 转弯辅助
  H_spine1  (joint[21]): 俯仰, Y轴, ±0.6rad  — 姿态调控
  H_body    (joint[23]): 扭转, X轴, ±1.57rad — 转弯辅助
"""

from __future__ import annotations
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .curriculums import get_curriculum_reward_weight

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# 脊柱关节全局索引（从XML关节顺序，0-based）
_F_SPINE1_IDX = 1   # 侧摆（primary turning）
_F_BODY_IDX = 3     # 前体扭转
_H_SPINE1_IDX = 21  # 俯仰
_H_BODY_IDX = 23    # 后体扭转
_SPINE_INDICES = [_F_SPINE1_IDX, _F_BODY_IDX, _H_BODY_IDX]


# 线速度跟踪奖励
def compute_vel_track_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """R_vel = exp(-100 * (v_x_actual - v_cmd)²)"""
    asset: Entity = env.scene["robot"]
    actual_vel_x = asset.data.root_link_lin_vel_w[:, 0]
    cmd_term = env.command_manager._terms["slalom_cmd"]
    v_cmd = cmd_term.command[:, 0]

    vel_error = actual_vel_x - v_cmd
    reward = torch.exp(-100.0 * vel_error ** 2)
    weight = get_curriculum_reward_weight(env, "track_vel")

    # 日志
    env.extras["log"]["Metrics/vel_error_mean"] = torch.abs(vel_error).mean().item()
    env.extras["log"]["Metrics/actual_vel_x_mean"] = actual_vel_x.mean().item()
    env.extras["log"]["Metrics/cmd_vel_x_mean"] = v_cmd.mean().item()

    return reward * weight


# 角速度跟踪奖励
def compute_omega_track_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """R_omega = exp(-50 * (ω_z_actual - ω_cmd)²)"""
    asset: Entity = env.scene["robot"]
    actual_omega_z = asset.data.root_link_ang_vel_w[:, 2]
    cmd_term = env.command_manager._terms["slalom_cmd"]
    omega_cmd = cmd_term.command[:, 1]

    omega_error = actual_omega_z - omega_cmd
    reward = torch.exp(-50.0 * omega_error ** 2)
    weight = get_curriculum_reward_weight(env, "track_omega")

    env.extras["log"]["Metrics/omega_error_mean"] = torch.abs(omega_error).mean().item()
    env.extras["log"]["Metrics/actual_omega_mean"] = actual_omega_z.mean().item()
    env.extras["log"]["Metrics/cmd_omega_mean"] = omega_cmd.mean().item()

    return reward * weight


# 脊柱转弯奖励：引导agent在转弯时使用脊柱侧摆+扭转
def compute_spine_turn_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """R_spine = |ω_cmd| * (2.5*|F_spine1| + |F_body| + |H_body|)
    转弯越大 → 鼓励脊柱弯曲幅度越大；直行(ω=0) → 奖励为0，不弯曲"""
    asset: Entity = env.scene["robot"]
    cmd_term = env.command_manager._terms["slalom_cmd"]
    omega_cmd_abs = torch.abs(cmd_term.command[:, 1])

    # 获取脊柱关节位置（全局索引）
    joint_pos = asset.data.joint_pos
    f_spine1_pos = joint_pos[:, _F_SPINE1_IDX]  # 侧摆
    f_body_pos = joint_pos[:, _F_BODY_IDX]      # 前体扭转
    h_body_pos = joint_pos[:, _H_BODY_IDX]      # 后体扭转

    spine_activity = (
        2.5 * torch.abs(f_spine1_pos)   # 侧摆权重最高（主驱动）
        + torch.abs(f_body_pos)          # 前体扭转辅助
        + torch.abs(h_body_pos)          # 后体扭转辅助
    )
    reward = omega_cmd_abs * spine_activity
    weight = get_curriculum_reward_weight(env, "spine_turn")

    env.extras["log"]["Metrics/spine_activity_mean"] = spine_activity.mean().item()
    env.extras["log"]["Metrics/f_spine1_abs_mean"] = torch.abs(f_spine1_pos).mean().item()

    return reward * weight


# 稳定性惩罚
def compute_stability_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """R_stab = -(roll² + pitch²)，惩罚机身大幅倾斜"""
    asset: Entity = env.scene["robot"]
    gravity_b = asset.data.projected_gravity_b
    tilt_sq = torch.sum(torch.square(gravity_b[:, :2]), dim=1)
    penalty = tilt_sq
    weight = get_curriculum_reward_weight(env, "stability")
    return -penalty * weight


# 能耗惩罚
def compute_energy_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """R_energy = -Σ(τ_i²)"""
    asset: Entity = env.scene["robot"]
    torques = asset.data.actuator_force
    penalty = torch.sum(torch.square(torques), dim=1)
    weight = get_curriculum_reward_weight(env, "energy")
    return -penalty * weight


# 动作平滑惩罚
def compute_smoothness_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """R_smooth = -Σ(Δa_i²)，惩罚动作突变"""
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    penalty = torch.sum(torch.square(current_action - prev_action), dim=1)
    weight = get_curriculum_reward_weight(env, "smoothness")
    return -penalty * weight


# 课程更新（零奖励，仅触发权重调度）
def update_curriculum(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)
