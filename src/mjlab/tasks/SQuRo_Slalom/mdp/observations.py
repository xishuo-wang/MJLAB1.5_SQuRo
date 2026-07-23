from __future__ import annotations
import torch
from mjlab.entity import Entity
from typing import TYPE_CHECKING
from mjlab.managers.scene_entity_config import SceneEntityCfg
from .indices import ACTUATED_JOINT_CFG, _MODEL_INDICES
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# 获取F_body_Link 偏航角
def _get_f_body_heading(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    quat = asset.data.body_link_quat_w[:, _MODEL_INDICES.f_body_id]  # [N, 4]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]  # type: ignore[misc]
    sin_h = 2.0 * (w * z + x * y)
    cos_h = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(sin_h, cos_h)


# 基座世界线速度
def base_lin_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_w


# 基座世界角速度
def base_ang_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_w


# 机身重力投影
def projected_gravity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


# 执行器关节位置（相对默认值，仅12个主动关节）
def actuator_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = ACTUATED_JOINT_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    return asset.data.joint_pos[:, _MODEL_INDICES.joint_ids] - default_joint_pos[:, _MODEL_INDICES.joint_ids]


# 执行器关节速度（相对默认值）
def actuator_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = ACTUATED_JOINT_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    default_joint_vel = asset.data.default_joint_vel
    assert default_joint_vel is not None
    return asset.data.joint_vel[:, _MODEL_INDICES.joint_ids] - default_joint_vel[:, _MODEL_INDICES.joint_ids]


# 执行器力矩
def actuator_force(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = ACTUATED_JOINT_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.actuator_force


# F_body_Link 物理前向 heading (body+X - π/2 → world+X方向 = 0°)
# 与 vel_track 奖励的 forward_heading 一致
def heading(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    return (_get_f_body_heading(env) - (torch.pi / 2)).unsqueeze(-1)


# 基座位置
def base_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w


# 参考关节位置
def ref_joint_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    from .reference import get_reference_joint_state
    pos, _ = get_reference_joint_state(env)
    return pos


# 参考关节速度
def ref_joint_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    from .reference import get_reference_joint_state
    _, vel = get_reference_joint_state(env)
    return vel


# 期望路径参考点 + 速度（共享，供观测和奖励使用）
def _compute_path_ref(env: "ManagerBasedRlEnv"):
    """返回 (x_ref, y_ref, vx_des, vy_des) — 世界系期望位置和速度"""
    asset: Entity = env.scene["robot"]
    cmd_term = env.command_manager._terms["slalom_cmd"]
    curvature = cmd_term.command[:, 4]          # [N]
    vel_cmd = cmd_term.command[:, 0]            # [N]
    t = env.episode_length_buf.float() * env.step_dt  # [N]

    # 初始化路径状态
    if getattr(env, "_path_obs_state", None) is None:
        env._path_obs_state = {  # type: ignore[attr-defined]
            "start_pos": asset.data.root_link_pos_w.clone(),
            "start_heading": (_get_f_body_heading(env) - (torch.pi / 2)).clone(),
        }
    state = env._path_obs_state  # type: ignore[attr-defined]

    # 重置已终止环境
    reset_ids = getattr(env, "reset_terminated", None)
    if reset_ids is not None:
        ids = reset_ids.nonzero(as_tuple=False).flatten()
        if len(ids) > 0:
            state["start_pos"][ids] = asset.data.root_link_pos_w[ids]
            state["start_heading"][ids] = _get_f_body_heading(env)[ids] - (torch.pi / 2)

    start_pos = state["start_pos"]
    start_heading = state["start_heading"]
    omega = curvature * vel_cmd
    dtheta = omega * t
    path_heading = start_heading + dtheta                    # 当前路径切线方向

    # 位置参考（与 path_track_reward 公式一致）
    chord = vel_cmd * t * torch.sinc(dtheta / (2 * torch.pi))
    x_ref = start_pos[:, 0] + chord * torch.cos(start_heading + dtheta / 2)
    y_ref = start_pos[:, 1] + chord * torch.sin(start_heading + dtheta / 2)

    # 世界系期望速度（路径切线方向）
    vx_des = vel_cmd * torch.cos(path_heading)
    vy_des = vel_cmd * torch.sin(path_heading)

    return x_ref, y_ref, vx_des, vy_des


# 期望路径误差 [Δx, Δy] — 相对基座的参考点偏移
def path_ref(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    x_ref, y_ref, vx_des, vy_des = _compute_path_ref(env)
    base_pos = env.scene[asset_cfg.name].data.root_link_pos_w
    dx = x_ref - base_pos[:, 0]
    dy = y_ref - base_pos[:, 1]
    return torch.stack([dx, dy, vx_des, vy_des], dim=1)  # [N, 4]
