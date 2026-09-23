from dataclasses import dataclass
import mujoco
from mjlab.entity import Entity, EntityCfg


@dataclass
class HoleEntityCfg(EntityCfg):
    name: str = "hole"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: tuple[float, float, float] = (0.01, 0.1, 0.01)
    rgba: tuple[float, float, float, float] = (0.95, 0.55, 0.22, 0.3)
    mass: float = 1.0
    contype: int = 1 
    conaffinity: int = 1 
    
    def build(self) -> "HoleEntity":
        return HoleEntity(cfg=self)


class HoleEntity(Entity):    
    def __init__(self, cfg: HoleEntityCfg):
        super().__init__(cfg)  
        self.cfg = cfg
        self._geom = None
        self._build_geometry()
    
    def _build_geometry(self):
        body = self._spec.worldbody.add_body(
            name=self.cfg.name,
            pos=self.cfg.position
        )
        
        self._geom = body.add_geom(
            name=f"{self.cfg.name}_geom",
            pos=(0, 0, self.cfg.size[2] / 2),
            size=self.cfg.size,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            rgba=self.cfg.rgba,
            contype=self.cfg.contype,
            conaffinity=self.cfg.conaffinity,
            mass=self.cfg.mass
        )
    
    # 启用碰撞属性
    def enable_collision(self):
        if self._geom is not None:
            self._geom.contype = self.cfg.contype
            self._geom.conaffinity = self.cfg.conaffinity
    
    # 禁用碰撞属性
    def disable_collision(self):
        if self._geom is not None:
            self._geom.contype = 0
            self._geom.conaffinity = 0
    
    # 设置碰撞属性
    def set_collision(self, enabled: bool):
        if enabled:
            self.enable_collision()
        else:
            self.disable_collision()
    
    # 碰撞是否开启 (编译期决定, 运行期不可改)
    @property
    def collision_enabled(self) -> bool:
        return int(self.cfg.contype) > 0

    @property
    def spec(self) -> mujoco.MjSpec:
        return self._spec