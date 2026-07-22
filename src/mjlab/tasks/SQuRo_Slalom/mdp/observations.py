"""SQuRo绕杆任务第一阶段 — 本体感受观测函数

仅观测执行器关节（12维），滤除被动闭链关节（16个）。
使用 ACTUATED_JOINT_CFG 通过 SceneEntityCfg 自动过滤。
"""

from __future__ import annotations
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from .indices import ACTUATED_JOINT_CFG, _MODEL_INDICES, resolve_model_indices
from .reference import get_reference_joint_state

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _ensure_indices(env: "ManagerBasedRlEnv") -> None:
    resolve_model_indices(env.scene["robot"])


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
    _ensure_indices(env)
    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    return asset.data.joint_pos[:, _MODEL_INDICES.joint_ids] - default_joint_pos[:, _MODEL_INDICES.joint_ids]


# 执行器关节速度（相对默认值）
def actuator_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = ACTUATED_JOINT_CFG) -> torch.Tensor:
    _ensure_indices(env)
    asset: Entity = env.scene[asset_cfg.name]
    default_joint_vel = asset.data.default_joint_vel
    assert default_joint_vel is not None
    return asset.data.joint_vel[:, _MODEL_INDICES.joint_ids] - default_joint_vel[:, _MODEL_INDICES.joint_ids]


# 执行器力矩
def actuator_force(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = ACTUATED_JOINT_CFG) -> torch.Tensor:
    _ensure_indices(env)
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.actuator_force


# 朝向角
def heading(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.heading_w.unsqueeze(-1)


# 基座位置
def base_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w


# Trot 参考关节位置
def ref_joint_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    pos, _ = get_reference_joint_state(env)
    return pos


# Trot 参考关节速度
def ref_joint_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    _, vel = get_reference_joint_state(env)
    return vel
