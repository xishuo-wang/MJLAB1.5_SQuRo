"""SQuRo绕杆任务第一阶段 — 本体感受观测函数"""

from __future__ import annotations
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# 基座世界线速度
def base_lin_vel_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_w


# 基座世界位置
def base_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w


# 关节加速度
def joint_acc(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.joint_acc


# 执行器力矩
def actuator_force(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.actuator_force


# 朝向角（机身坐标系下重力向量方向）
def heading(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.heading_w.unsqueeze(-1)
