from __future__ import annotations
import torch
import mujoco
from dataclasses import dataclass
from mjlab.entity import Entity, EntityCfg
from .curriculums import SLALOM_CURVATURE_MAX
from .slalom_geometry import get_slalom_geometry


# 杆几何常量
POLE_RADIUS = 0.01          # 半径 1cm
POLE_HALF_HEIGHT = 0.05     # 高 10cm（半高 5cm）
POLE_NUM = 12               # 杆阵列数量，实际间距按回合几何更新


# 平滑后的杆心 Y 坐标
def _smooth_pole_y() -> float:
    return get_slalom_geometry(SLALOM_CURVATURE_MAX)["pole_y"]


POLE_Y = _smooth_pole_y()   # 编译占位和默认绘图使用，运行时逐环境更新


@dataclass
class PoleEntityCfg(EntityCfg):
    name: str = "pole"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = POLE_RADIUS
    half_height: float = POLE_HALF_HEIGHT
    rgba: tuple[float, float, float, float] = (0.9, 0.35, 0.2, 1.0)  # 橙红色
    contype: int = 1
    conaffinity: int = 1

    def build(self) -> "PoleEntity":
        return PoleEntity(cfg=self)


class PoleEntity(Entity):
    def __init__(self, cfg: PoleEntityCfg):
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


# 按名称解析杆体的 mocap 编号，避免依赖模型中的排列顺序。
def pole_mocap_id(entity) -> int:
    cached = getattr(entity, "_pole_mocap_id", None)
    if cached is None:
        local, _ = entity.find_bodies([entity.cfg.name], preserve_order=True)
        if len(local) != 1:
            raise ValueError(f"找不到唯一杆体：{entity.cfg.name}")
        body_id = int(entity.indexing.body_ids[local[0]])
        cached = int(entity.data.model.body_mocapid[body_id])
        if cached < 0:
            raise ValueError(f"杆体不是 mocap：{entity.cfg.name}")
        entity._pole_mocap_id = cached
    return cached


# 仅更新重置环境的整排杆，一阶段将杆移出活动区域。
def write_pole_mocap(env, env_ids: torch.Tensor, spacing: torch.Tensor, pole_y: torch.Tensor, phase1: bool) -> None:
    origins = getattr(env.scene, "env_origins", None)
    origin = torch.zeros((len(env_ids), 3), device=env.device) if origins is None else origins[env_ids]
    for i in range(POLE_NUM):
        entity = env.scene.entities.get(f"pole{i}")
        if entity is None:
            continue
        mocap_id = pole_mocap_id(entity)
        position = origin.clone()
        position[:, 0] += (i + 1) * spacing
        position[:, 1] += pole_y if phase1 else 100.0
        entity.data.data.mocap_pos[env_ids, mocap_id] = position
        entity.data.data.mocap_quat[env_ids, mocap_id] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=env.device)


def generate_pole_positions(
    spacing: float = 0.35,
    num_poles: int = POLE_NUM,
    start_x: float = 0.0,
    start_y: float = POLE_Y,
) -> list[tuple[float, float, float]]:
    positions: list[tuple[float, float, float]] = []
    for i in range(num_poles):
        x = start_x + i * spacing
        y = start_y
        positions.append((x, y, 0.0))
    return positions
