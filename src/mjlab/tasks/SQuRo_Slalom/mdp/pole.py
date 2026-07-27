"""绕杆实体 — 圆柱形杆，作为 SQuRo 绕杆任务的物理障碍物

训练阶段杆的碰撞属性设为 0（仅视觉提示，约束由奖励函数提供），
play 阶段启用物理碰撞。

杆位生成逻辑与后续绕杆路径(compute_slalom_path_ref)对齐：
  沿世界 +X 方向等间距排列，奇偶杆 Y 方向交错偏移。
"""

from __future__ import annotations
from dataclasses import dataclass
import mujoco
from mjlab.entity import Entity, EntityCfg


# 杆几何常量
POLE_RADIUS = 0.005     # 直径 1cm
POLE_HALF_HEIGHT = 0.05  # 高 10cm（半高 5cm）


@dataclass
class PoleEntityCfg(EntityCfg):
    """单根杆的 MuJoCo 实体配置"""
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
    """圆柱杆实体 — 使用 MuJoCo mjGEOM_CYLINDER"""

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


def generate_pole_positions(
    spacing: float = 0.35,
    stagger: float = 0.08,
    num_poles: int = 6,
    start_x: float = 0.25,
) -> list[tuple[float, float, float]]:
    """生成交错排列的杆位坐标

    杆沿世界 +X 方向等间距排列，奇偶杆在 Y 方向交错偏移：
      杆 0: (start_x, +stagger),  杆 1: (start_x+spacing, -stagger),
      杆 2: (start_x+2*spacing, +stagger), ...

    参数:
      spacing:    相邻杆的 X 方向间距
      stagger:    Y 方向交错偏移量
      num_poles:  杆的数量
      start_x:    第一根杆的 X 坐标
    """
    positions: list[tuple[float, float, float]] = []
    for i in range(num_poles):
        x = start_x + i * spacing
        y = stagger if i % 2 == 0 else -stagger
        positions.append((x, y, 0.0))
    return positions
