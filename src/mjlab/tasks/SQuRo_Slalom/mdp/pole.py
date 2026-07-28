from __future__ import annotations
import numpy as np
import mujoco
from dataclasses import dataclass
from mjlab.entity import Entity, EntityCfg


# 杆几何常量
POLE_RADIUS = 0.01          # 直径 1cm
POLE_HALF_HEIGHT = 0.05     # 高 10cm（半高 5cm）


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


# 根据训练阶段更新所有杆的可见性（遍历编译后模型 geom 名称，兼容 WarpBridge)
def update_pole_visibility(env, phase: int) -> None:
    alpha = 0.0 if phase == 0 else 1.0
    model = env.sim.model
    r, g, b = 0.9, 0.35, 0.2
    for i in range(model.ngeom):
        try:
            name = model.geom(i).name  # type: ignore[union-attr]
        except Exception:
            continue
        if name and name.startswith("pole") and name.endswith("_geom"):
            model.geom_rgba[i] = np.array([r, g, b, alpha], dtype=np.float32)


def generate_pole_positions(
    spacing: float = 0.35,
    num_poles: int = 6,
    start_x: float = 0.25,
    start_y: float = -0.1,
) -> list[tuple[float, float, float]]:
    positions: list[tuple[float, float, float]] = []
    for i in range(num_poles):
        x = start_x + i * spacing
        y = start_y
        positions.append((x, y, 0.0))
    return positions
