from __future__ import annotations
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from typing import TYPE_CHECKING, List, Dict, Optional 

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

from .reference import get_reference_joint_pos, get_reference_joint_vel

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_HOLE_NAMES = ["Hole1", "Hole2", "Hole3"] 

# -静态数据缓存 
_HOLE_CACHE: Dict[str, Optional[torch.Tensor]] = {
    "positions": None,  # 缓存的 Hole 位置张量
    "heights": None,    # 缓存的 Hole 高度张量
}

def _ensure_hole_cache(env: ManagerBasedRlEnv, hole_names: List[str]):
    # 显式使用 global 声明，确保 Pylance 知道我们在修改全局变量
    global _HOLE_CACHE
    
    if _HOLE_CACHE["positions"] is not None:
        return

    positions = []
    heights = []
    
    for hole_name in hole_names:
        if hole_name in env.scene.entities:
            hole = env.scene.entities[hole_name]
            
            # 1. 提取位置
            pos = getattr(hole, '_position', None)
            if pos is None:
                cfg = getattr(hole, 'cfg', None)
                if cfg is not None:
                    pos_tuple = getattr(cfg, 'position', None)
                    if pos_tuple is not None:
                        pos = torch.tensor(pos_tuple, device=env.device, dtype=torch.float)
            if pos is None:
                pos = torch.zeros(3, device=env.device, dtype=torch.float)
            positions.append(pos)
            
            # 2. 提取高度
            height = None
            get_height_func = getattr(hole, 'get_height', None)
            if get_height_func is not None:
                height = get_height_func()
            else:
                cfg = getattr(hole, 'cfg', None)
                if cfg is not None:
                    size = getattr(cfg, 'size', None)
                    if size is not None and len(size) >= 3:
                        height = float(size[2] * 2)
            if height is None:
                height = 0.01
            heights.append(height)
        else:
            # 找不到实体时填充默认值
            positions.append(torch.zeros(3, device=env.device, dtype=torch.float))
            heights.append(0.0)
            
    # 转换为常驻 GPU 的 Tensor [num_holes, 3] 和 [num_holes]
    _HOLE_CACHE["positions"] = torch.stack(positions, dim=0)
    _HOLE_CACHE["heights"] = torch.tensor(heights, device=env.device, dtype=torch.float)
# -------------------------------------------------------------


# 获取基座位置 (保持高效，无须改动)
def base_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w


# 获取基座线速度
def base_lin_vel_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_w


# 获取关节加速度
def joint_acc(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.joint_acc


# 4. 获取执行器力矩
def actuator_force(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.actuator_force


# 获取朝向角
def heading(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.heading_w.unsqueeze(-1)  


# 获取当前时间步的参考关节位置
def ref_joint_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    return get_reference_joint_pos(env)


# 获取当前时间步的参考关节速度
def ref_joint_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    return get_reference_joint_vel(env)
