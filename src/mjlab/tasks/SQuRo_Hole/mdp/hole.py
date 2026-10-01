from dataclasses import dataclass, replace
import mujoco
from mjlab.entity import Entity, EntityCfg


# 三块限高板的几何: 板底 z 即板体 position.z, 板厚 2*half_thickness 向上长
# 板底 = position.z, 板顶 = position.z + 2*half_thickness
# 实测板底 / 板顶 (重构前口径保持一致):
#   hole1 / hole3: 板底 0.0475, 板顶 0.0575 (position.z=0.05, 半厚 0.005)
#   hole2:         板底 0.0725, 板顶 0.0825 (position.z=0.075, 半厚 0.005)
# 虚拟净空阈值 = 板底 − 2.5 mm (见 rewards.compute_body_contact_penalty)
HOLE_LAYOUT = (
    ("Hole1", 0.2, 0.0, 0.0475, 0.015, 0.005),
    ("Hole2", 0.6, 0.0, 0.0725, 0.100, 0.005),
    ("Hole3", 1.2, 0.0, 0.0475, 0.015, 0.005),
)

# 虚拟净空阈值相对板底的余量 (板底 − 该值 = 虚拟阈值)
VIRTUAL_CLEARANCE_MARGIN = 0.0025


@dataclass
class HoleEntityCfg(EntityCfg):
    name: str = "hole"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)   # 板底中心 (x, y, 板底 z)
    size: tuple[float, float, float] = (0.01, 0.1, 0.005)     # 半长, 半宽, 半厚
    rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5)
    mass: float = 1.0
    contype: int = 1
    conaffinity: int = 1
    solref: tuple[float, ...] | None = None
    solimp: tuple[float, ...] | None = None

    def build(self) -> "HoleEntity":
        return HoleEntity(cfg=self)


class HoleEntity(Entity):
    def __init__(self, cfg: HoleEntityCfg):
        super().__init__(cfg)
        self.cfg = cfg
        self._geom = None
        self._build_geometry()

    def _build_geometry(self):
        # 板体中心高度 = 板底 + 半厚 (统一由此计算, 不再混用 position.z 与 size)
        half_t = self.cfg.size[2]
        body = self._spec.worldbody.add_body(
            name=self.cfg.name,
            pos=self.cfg.position,
        )
        kwargs: dict = {}
        if self.cfg.solref is not None:
            kwargs["solref"] = self.cfg.solref
        if self.cfg.solimp is not None:
            kwargs["solimp"] = self.cfg.solimp
        self._geom = body.add_geom(
            name=f"{self.cfg.name}_geom",
            pos=(0, 0, half_t),
            size=self.cfg.size,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            rgba=self.cfg.rgba,
            contype=self.cfg.contype,
            conaffinity=self.cfg.conaffinity,
            mass=self.cfg.mass,
            **kwargs,
        )

    # 板底高度 (m): 板体配置的 z 即板底
    @property
    def bottom_z(self) -> float:
        return float(self.cfg.position[2])

    # 板顶高度 (m)
    @property
    def top_z(self) -> float:
        return self.bottom_z + 2.0 * float(self.cfg.size[2])

    # 虚拟净空阈值 (m): 板底减去余量
    @property
    def virtual_z_threshold(self) -> float:
        return self.bottom_z - VIRTUAL_CLEARANCE_MARGIN

    # 板体在 x 方向的覆盖区间 (m)
    @property
    def x_range(self) -> tuple[float, float]:
        hx = float(self.cfg.size[0])
        return (float(self.cfg.position[0]) - hx, float(self.cfg.position[0]) + hx)

    # 碰撞是否开启 (编译期决定, 运行期不可改)
    @property
    def collision_enabled(self) -> bool:
        return int(self.cfg.contype) > 0

    @property
    def spec(self) -> mujoco.MjSpec:
        return self._spec


# 按几何表构建三块板 (碰撞在编译期固化, 换开关必须重建环境)
def build_hole_entities(enable_collision: bool,
                        solref: tuple[float, ...] | None = None,
                        solimp: tuple[float, ...] | None = None) -> dict:
    mask = 1 if enable_collision else 0
    entities: dict = {}
    for name, x, y, bottom_z, half_len, half_t in HOLE_LAYOUT:
        kwargs: dict = {}
        if solref is not None:
            kwargs["solref"] = solref
        if solimp is not None:
            kwargs["solimp"] = solimp
        entities[name.lower()] = HoleEntityCfg(
            name=name,
            position=(x, y, bottom_z),
            size=(half_len, 0.1, half_t),
            contype=mask,
            conaffinity=mask,
            **kwargs,
        )
    return entities


# 改写场景里的三块板: 保留已有实体的位置/尺寸/接触参数, 只替换碰撞掩码。
# 重建时不能整体替换为默认几何, 否则会丢掉命令行或脚本自定义的板位与 solref/solimp
def configure_hole_entities(env_cfg, enable_collision: bool,
                            solref: tuple[float, ...] | None = None,
                            solimp: tuple[float, ...] | None = None) -> dict:
    mask = 1 if enable_collision else 0
    entities = dict(env_cfg.scene.entities)
    defaults = build_hole_entities(enable_collision=enable_collision,
                                   solref=solref, solimp=solimp)
    for key, fresh in defaults.items():
        existing = entities.get(key)
        if isinstance(existing, HoleEntityCfg):
            keep: dict = {"contype": mask, "conaffinity": mask}
            if solref is not None:
                keep["solref"] = fresh.solref
            if solimp is not None:
                keep["solimp"] = fresh.solimp
            entities[key] = replace(existing, **keep)
        else:
            entities[key] = fresh
    env_cfg.scene.entities = entities
    return entities


# 从环境里读实际编译进仿真的碰撞开关 (不缓存"以为改成了什么")
def hole_collision_enabled(env) -> bool:
    entity = env.scene.entities.get("hole1")
    return bool(entity is not None and int(entity.cfg.contype) > 0)


# 统一几何查询: 返回三块板的 x 区间 / 板底 / 虚拟阈值 (奖励与诊断共用同一来源)
def hole_geometry(env=None) -> list[dict]:
    out = []
    for name, x, y, bottom_z, half_len, half_t in HOLE_LAYOUT:
        out.append({
            "name": name,
            "x_min": x - half_len,
            "x_max": x + half_len,
            "bottom_z": bottom_z,
            "top_z": bottom_z + 2.0 * half_t,
            "virtual_z_threshold": bottom_z - VIRTUAL_CLEARANCE_MARGIN,
        })
    return out
