from __future__ import annotations
import torch
import mujoco
from dataclasses import dataclass
from mjlab.entity import Entity, EntityCfg
from .curriculums import CURVATURE_TARGET_MAX


# 杆几何常量
POLE_RADIUS = 0.01                      # 直径 1cm
POLE_HALF_HEIGHT = 0.05                 # 高 10cm（半高 5cm）
POLE_Y = -1.0 / CURVATURE_TARGET_MAX    # 杆心 Y 坐标 (= -Rmin = -0.0667, 路径第一弧底部)


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
        body = self._spec.worldbody.add_body(
            name=self.cfg.name,
            pos=self.cfg.position,
        )
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


# 根据训练阶段更新所有杆的可见性（按 geom 类型匹配圆柱体)
def update_pole_visibility(env, phase: int) -> None:
    alpha = 0.0 if phase == 0 else 1.0
    model = env.sim.model
    rgba = torch.tensor([0.9, 0.35, 0.2, alpha], device=env.device, dtype=torch.float32)
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            model.geom_rgba[0, i] = rgba


def generate_pole_positions(
    spacing: float = 0.35,
    num_poles: int = 6,
    start_x: float = 0.0,
    start_y: float = POLE_Y,
) -> list[tuple[float, float, float]]:
    positions: list[tuple[float, float, float]] = []
    for i in range(num_poles):
        x = start_x + i * spacing
        y = start_y
        positions.append((x, y, 0.0))
    return positions
