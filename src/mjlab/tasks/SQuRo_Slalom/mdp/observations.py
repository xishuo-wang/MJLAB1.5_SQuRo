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


# F_body_Link 朝向角（与 vel_track 奖励一致）
def heading(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    return _get_f_body_heading(env).unsqueeze(-1)


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
