import torch
import mujoco
from mjlab.entity import Entity, EntityCfg
from dataclasses import dataclass, replace
from .config import HEIGHT_INITIAL_MIN

# 三块限高板的几何: 板底 z 即板体 position.z, 板厚 2*half_thickness 向上长
# 板底 = position.z, 板顶 = position.z + 2*half_thickness
# 当前板底 / 板顶:
#   hole1 / hole3: 板底 0.0500, 板顶 0.0600 (position.z=0.05, 半厚 0.005)
#   hole2:         板底 0.0700, 板顶 0.0800 (position.z=0.07, 半厚 0.005)
# 虚拟净空阈值 = 板底 − 2.5 mm (见 rewards.compute_body_contact_reward)
HOLE_LAYOUT = (
    ("Hole1", 0.2, 0.0, 0.05, 0.015, 0.005),
    ("Hole2", 0.6, 0.0, 0.07, 0.100, 0.005),
    ("Hole3", 1.2, 0.0, 0.05, 0.015, 0.005),
)

# 新课程训练使用的单板几何；三板表保留给旧配置和预览。
SINGLE_HOLE_LAYOUT = HOLE_LAYOUT[:1]

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
        half_t = self.cfg.size[2]
        body = self._spec.worldbody.add_body(
            name=self.cfg.name,
            pos=self.cfg.position,
            mocap=True,
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
def build_hole_entities(enable_collision: bool, solref: tuple[float, ...] | None = None,solimp: tuple[float, ...] | None = None) -> dict:
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



# 构建单板课程实体。
def build_single_hole_entities(enable_collision: bool,
                               bottom_z: float = HEIGHT_INITIAL_MIN,
                               solref: tuple[float, ...] | None = None,
                               solimp: tuple[float, ...] | None = None) -> dict:
    mask = 1 if enable_collision else 0
    name, x, y, _, half_len, half_t = SINGLE_HOLE_LAYOUT[0]
    kwargs: dict = {}
    if solref is not None:
        kwargs["solref"] = solref
    if solimp is not None:
        kwargs["solimp"] = solimp
    return {
        name.lower(): HoleEntityCfg(
            name=name,
            position=(x, y, float(bottom_z)),
            size=(half_len, 0.1, half_t),
            contype=mask,
            conaffinity=mask,
            **kwargs,
        )
    }



# 改写场景里的三块板: 保留已有实体的位置/尺寸/接触参数, 只替换碰撞掩码
def configure_hole_entities(env_cfg, enable_collision: bool, solref: tuple[float, ...] | None = None, solimp: tuple[float, ...] | None = None) -> dict:
    mask = 1 if enable_collision else 0
    entities = dict(env_cfg.scene.entities)
    for key, existing in tuple(entities.items()):
        if not isinstance(existing, HoleEntityCfg):
            continue
        keep: dict = {"contype": mask, "conaffinity": mask}
        if solref is not None:
            keep["solref"] = solref
        if solimp is not None:
            keep["solimp"] = solimp
        entities[key] = replace(existing, **keep)
    env_cfg.scene.entities = entities
    return entities



# 将训练配置切换为单板几何并保留显式碰撞开关。
def configure_single_hole_entities(env_cfg, enable_collision: bool, bottom_z: float = HEIGHT_INITIAL_MIN) -> dict:
    entities = dict(env_cfg.scene.entities)
    key = SINGLE_HOLE_LAYOUT[0][0].lower()
    existing = entities.get(key)
    fresh = build_single_hole_entities(enable_collision, bottom_z)[key]
    if isinstance(existing, HoleEntityCfg):
        fresh = replace(
            existing,
            position=fresh.position,
            size=fresh.size,
            contype=fresh.contype,
            conaffinity=fresh.conaffinity,
        )
    entities = {
        name: entity for name, entity in entities.items()
        if not isinstance(entity, HoleEntityCfg) or name == key
    }
    entities[key] = fresh
    env_cfg.scene.entities = entities
    return entities



# 从环境里读实际编译进仿真的碰撞开关
def hole_collision_enabled(env) -> bool:
    entity = env.scene.entities.get("hole1")
    return bool(entity is not None and int(entity.cfg.contype) > 0)



# 板体自身的 mocap 下标; entity.indexing.mocap_id 指向 root 的空 mocap_base, 不能用于板
def hole_mocap_id(entity) -> int:
    local, _ = entity.find_bodies([entity.cfg.name], preserve_order=True)
    if not local:
        raise ValueError(f"{entity.cfg.name} 实体里找不到同名 body")
    body_id = int(entity.indexing.body_ids[local[0]])
    mocap_id = int(entity.data.model.body_mocapid[body_id])
    if mocap_id < 0:
        raise ValueError(f"{entity.cfg.name} 板体不是 mocap 体")
    return mocap_id



# 将板底高度写入板体 mocap，支持全量或部分环境更新
def write_hole_bottom_z(entity, bottom_z, env_ids=None) -> None:
    mocap_pos = entity.data.data.mocap_pos
    mocap_id = hole_mocap_id(entity)
    mocap_poses = mocap_pos[:, mocap_id, :]
    all_env_ids = torch.arange(mocap_poses.shape[0], device=mocap_poses.device)
    if env_ids is None:
        target_ids = all_env_ids
    elif isinstance(env_ids, slice):
        target_ids = all_env_ids[env_ids]
    else:
        target_ids = env_ids.to(device=mocap_poses.device, dtype=torch.long).flatten()
    heights = torch.as_tensor(bottom_z, device=mocap_pos.device, dtype=mocap_pos.dtype)
    if heights.ndim == 0:
        heights = heights.expand(target_ids.shape[0])
    if heights.shape[0] != target_ids.shape[0]:
        raise ValueError("板底高度数量必须与目标环境数量一致")
    mocap_pos[target_ids, mocap_id, 0] = float(entity.cfg.position[0])
    mocap_pos[target_ids, mocap_id, 1] = float(entity.cfg.position[1])
    mocap_pos[target_ids, mocap_id, 2] = heights



# 读取板体当前实际 mocap 板底高度
def read_bottom_z(entity):
    return entity.data.data.mocap_pos[:, hole_mocap_id(entity), 2]



# 单块板的几何字典
def _entity_geometry(cfg) -> dict:
    x, y, bottom_z = (float(v) for v in cfg.position)
    half_len, _, half_t = (float(v) for v in cfg.size)
    return {
        "name": cfg.name,
        "x_min": x - half_len,
        "x_max": x + half_len,
        "bottom_z": bottom_z,
        "top_z": bottom_z + 2.0 * half_t,
        "virtual_z_threshold": bottom_z - VIRTUAL_CLEARANCE_MARGIN,
    }



# 统一几何查询: 返回三块板的 x 区间 / 板底 / 虚拟阈值
def hole_geometry(env=None) -> list[dict]:
    if env is not None:
        scene = getattr(env, "scene", env)
        entities = getattr(scene, "entities", None) or {}
        out = []
        for name, *_ in HOLE_LAYOUT:
            ent = entities.get(name.lower())
            if ent is not None and isinstance(getattr(ent, "cfg", None), HoleEntityCfg):
                out.append(_entity_geometry(ent.cfg))
        if out:
            return out
    return [_entity_geometry(HoleEntityCfg(name=n, position=(x, y, bz), size=(hl, 0.1, ht))) for n, x, y, bz, hl, ht in HOLE_LAYOUT]



# 统一设置障碍物板底高度
def set_obstacle_bottom_z(env_cfg, bottom_z: float) -> int:
    entities = dict(env_cfg.scene.entities)
    n = 0
    for key, ent in entities.items():
        if isinstance(ent, HoleEntityCfg):
            entities[key] = replace(ent, position=(float(ent.position[0]), float(ent.position[1]), float(bottom_z)),)
            n += 1
    env_cfg.scene.entities = entities
    return n



# 统一设置三块板的可见性: 仅改 rgba, 不影响碰撞 (碰撞只由 contype/conaffinity 决定)
def set_obstacle_visibility(env_cfg, visible: bool, hidden_rgba: tuple = (0.0, 0.0, 0.0, 0.0), shown_rgba: tuple = (0.5, 0.5, 0.5, 0.5)) -> int:
    entities = dict(env_cfg.scene.entities)
    rgba = tuple(shown_rgba) if visible else tuple(hidden_rgba)
    n = 0
    for name, *_ in HOLE_LAYOUT:
        key = name.lower()
        ent = entities.get(key)
        if isinstance(ent, HoleEntityCfg):
            entities[key] = replace(ent, rgba=rgba)
            n += 1
    env_cfg.scene.entities = entities
    return n



# 把 checkpoint 里记录的板几何/接触参数应用到 env_cfg
def apply_saved_layout(env_cfg, layout: list) -> int:
    if not layout:
        return 0
    entities = dict(env_cfg.scene.entities)
    collision_template = next((ent for ent in entities.values() if isinstance(ent, HoleEntityCfg)), None,)
    saved_keys = {str(d.get("name", "")).lower() for d in layout}
    entities = {key: ent for key, ent in entities.items() if not isinstance(ent, HoleEntityCfg) or key in saved_keys}
    applied = 0
    for d in layout:
        key = str(d.get("name", "")).lower()
        existing = entities.get(key)
        if not isinstance(existing, HoleEntityCfg):
            position = d.get("position")
            size = d.get("size")
            if position is None or size is None or not key.startswith("hole"):
                continue
            existing = HoleEntityCfg(
                name=str(d.get("name", key)),
                position=(float(position[0]), float(position[1]), float(position[2])),
                size=(float(size[0]), float(size[1]), float(size[2])),
                contype=(int(collision_template.contype) if collision_template else 1),
                conaffinity=(int(collision_template.conaffinity) if collision_template else 1),
            )
        keep: dict = {}
        if d.get("position") is not None:
            keep["position"] = tuple(float(v) for v in d["position"])
        if d.get("size") is not None:
            keep["size"] = tuple(float(v) for v in d["size"])
        for f in ("solref", "solimp"):
            if d.get(f) is not None:
                keep[f] = tuple(float(v) for v in d[f])
        entities[key] = replace(existing, **keep)
        applied += 1
    env_cfg.scene.entities = entities
    return applied
