from __future__ import annotations
import torch
import mujoco
from dataclasses import dataclass, replace
from mjlab.entity import Entity, EntityCfg
from .reference import get_slalom_geometry
from .config import OBSTACLE_HALF_HEIGHT, OBSTACLE_NUM, OBSTACLE_RADIUS, OBSTACLE_SPACING_START, SLALOM_CURVATURE_MAX



# 平滑后的杆心 Y 坐标
def _smooth_obstacle_y() -> float:
    return get_slalom_geometry(SLALOM_CURVATURE_MAX)["obstacle_y"]



OBSTACLE_Y = _smooth_obstacle_y()   # 编译占位和默认绘图使用，运行时逐环境更新



# 仅改变独立考核场景的杆碰撞掩码，保留 mocap 与实际尺寸
def configure_obstacle_collision(env_cfg, enable_collision: bool) -> None:
    mask = int(enable_collision)
    env_cfg.scene.entities = {
        name: replace(entity, contype=mask, conaffinity=mask) if isinstance(entity, ObstacleEntityCfg) else entity
        for name, entity in env_cfg.scene.entities.items()
    }


# 从编译模型核对整排杆的实际碰撞开关
def obstacle_collision_enabled(env) -> bool:
    model = env.sim.mj_model
    states = []
    for index in range(OBSTACLE_NUM):
        entity = env.scene.entities[f"obstacle{index}"]
        local, _ = entity.find_geoms([f"obstacle{index}_geom"], preserve_order=True)
        geom_id = int(entity.indexing.geom_ids[local[0]])
        states.append(bool(model.geom_contype[geom_id] and model.geom_conaffinity[geom_id]))
    if any(states) and not all(states):
        raise ValueError("整排杆的实际碰撞开关不一致")
    return all(states)



@dataclass
class ObstacleEntityCfg(EntityCfg):
    name: str = "obstacle"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = OBSTACLE_RADIUS
    half_height: float = OBSTACLE_HALF_HEIGHT
    rgba: tuple[float, float, float, float] = (0.9, 0.35, 0.2, 1.0)  # 橙红色
    contype: int = 1
    conaffinity: int = 1

    def build(self) -> "ObstacleEntity":
        return ObstacleEntity(cfg=self)



class ObstacleEntity(Entity):
    cfg: ObstacleEntityCfg
    def __init__(self, cfg: ObstacleEntityCfg):
        super().__init__(cfg)
        self.cfg = cfg
        self._geom_ref = None
        self._build_geometry()


    def _build_geometry(self):
        body = self._spec.worldbody.add_body(name=self.cfg.name, pos=self.cfg.position, mocap=True)
        self._geom_ref = body.add_geom(
            name=f"{self.cfg.name}_geom",
            pos=(0, 0, self.cfg.half_height),  # 圆柱中心离地半个高度
            size=(self.cfg.radius, self.cfg.half_height),
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            rgba=self.cfg.rgba,
            contype=self.cfg.contype,
            conaffinity=self.cfg.conaffinity,
        )


    @property
    def spec(self) -> mujoco.MjSpec:
        return self._spec



# 构建整排障碍物的实体配置，训练默认关闭实际碰撞
def build_obstacle_entities(enable_collision: bool = False) -> dict[str, ObstacleEntityCfg]:
    mask = int(enable_collision)
    return {
        f"obstacle{i}": ObstacleEntityCfg(
            name=f"obstacle{i}",
            position=((i + 1) * OBSTACLE_SPACING_START, OBSTACLE_Y, 0.0),
            rgba=(0.9, 0.35, 0.2, 1.0),
            contype=mask,
            conaffinity=mask,
        )
        for i in range(OBSTACLE_NUM)
    }



# 按名称解析杆体的 mocap 编号，避免依赖模型中的排列顺序
def obstacle_mocap_id(entity) -> int:
    cached = getattr(entity, "_obstacle_mocap_id", None)
    if cached is None:
        local, _ = entity.find_bodies([entity.cfg.name], preserve_order=True)
        if len(local) != 1:
            raise ValueError(f"找不到唯一杆体：{entity.cfg.name}")
        body_id = int(entity.indexing.body_ids[local[0]])
        cached = int(entity.data.model.body_mocapid[body_id])
        if cached < 0:
            raise ValueError(f"杆体不是 mocap：{entity.cfg.name}")
        entity._obstacle_mocap_id = cached
    return cached



# 仅更新重置环境的整排杆，一阶段将杆移出活动区域
def write_obstacle_mocap(env, env_ids: torch.Tensor, spacing: torch.Tensor, obstacle_y: torch.Tensor, phase1: bool) -> None:
    origins = getattr(env.scene, "env_origins", None)
    origin = torch.zeros((len(env_ids), 3), device=env.device) if origins is None else origins[env_ids]
    for i in range(OBSTACLE_NUM):
        entity = env.scene.entities.get(f"obstacle{i}")
        if entity is None:
            continue
        mocap_id = obstacle_mocap_id(entity)
        position = origin.clone()
        position[:, 0] += (i + 1) * spacing
        position[:, 1] += obstacle_y if phase1 else 100.0
        entity.data.data.mocap_pos[env_ids, mocap_id] = position
        entity.data.data.mocap_quat[env_ids, mocap_id] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)



def generate_obstacle_positions(
    spacing: float = 0.35,
    num_obstacles: int = OBSTACLE_NUM,
    start_x: float = 0.0,
    start_y: float = OBSTACLE_Y,
) -> list[tuple[float, float, float]]:
    positions: list[tuple[float, float, float]] = []
    for i in range(num_obstacles):
        x = start_x + i * spacing
        y = start_y
        positions.append((x, y, 0.0))
    return positions
