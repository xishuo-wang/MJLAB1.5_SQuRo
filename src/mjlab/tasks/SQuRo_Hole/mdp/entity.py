import torch
import mujoco
from mjlab.entity import Entity, EntityCfg
from dataclasses import dataclass, replace
from .config import HEIGHT_INITIAL_MIN, OBSTACLE_POSITION, OBSTACLE_SIZE, VIRTUAL_CLEARANCE_MARGIN



@dataclass
class ObstacleEntityCfg(EntityCfg):
    name: str = "obstacle"
    position: tuple[float, float, float] = OBSTACLE_POSITION            # 板底中心 x, y, 板底 z
    size: tuple[float, float, float] = OBSTACLE_SIZE                    # 半长，半宽，半厚
    rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5)
    mass: float = 1.0
    contype: int = 1
    conaffinity: int = 1
    solref: tuple[float, ...] | None = None
    solimp: tuple[float, ...] | None = None


    def build(self) -> "ObstacleEntity":
        return ObstacleEntity(cfg=self)



class ObstacleEntity(Entity):
    cfg: ObstacleEntityCfg
    def __init__(self, cfg: ObstacleEntityCfg):
        super().__init__(cfg)
        self._build_geometry()


    def _build_geometry(self) -> None:
        half_thickness = self.cfg.size[2]
        body = self._spec.worldbody.add_body(name=self.cfg.name, pos=self.cfg.position, mocap=True,)
        kwargs: dict = {}
        if self.cfg.solref is not None:
            kwargs["solref"] = self.cfg.solref
        if self.cfg.solimp is not None:
            kwargs["solimp"] = self.cfg.solimp
        body.add_geom(
            name=f"{self.cfg.name}_geom",
            pos=(0, 0, half_thickness),
            size=self.cfg.size,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            rgba=self.cfg.rgba,
            contype=self.cfg.contype,
            conaffinity=self.cfg.conaffinity,
            mass=self.cfg.mass,
            **kwargs,
        )



# 构建真实碰撞课程使用的受限空间实体
def build_obstacle_entity(enable_collision: bool, bottom_z: float = HEIGHT_INITIAL_MIN, solref: tuple[float, ...] | None = None, solimp: tuple[float, ...] | None = None) -> dict:
    mask = 1 if enable_collision else 0
    kwargs: dict = {}
    if solref is not None:
        kwargs["solref"] = solref
    if solimp is not None:
        kwargs["solimp"] = solimp
    entity = ObstacleEntityCfg(
        position=(OBSTACLE_POSITION[0], OBSTACLE_POSITION[1], float(bottom_z)),
        contype=mask,
        conaffinity=mask,
        **kwargs,
    )
    return {entity.name.lower(): entity}



# 保留实际板体的位置、尺寸和接触参数，只替换碰撞掩码
def configure_obstacle_collision(env_cfg, enable_collision: bool, solref: tuple[float, ...] | None = None, solimp: tuple[float, ...] | None = None) -> dict:
    mask = 1 if enable_collision else 0
    entities = dict(env_cfg.scene.entities)
    existing = entities.get("obstacle")
    if isinstance(existing, ObstacleEntityCfg):
        keep: dict = {"contype": mask, "conaffinity": mask}
        if solref is not None:
            keep["solref"] = solref
        if solimp is not None:
            keep["solimp"] = solimp
        entities["obstacle"] = replace(existing, **keep)
    env_cfg.scene.entities = entities
    return entities



# 将训练配置切换为单板几何并保留显式碰撞开关
def configure_obstacle_entity(env_cfg, enable_collision: bool, bottom_z: float = HEIGHT_INITIAL_MIN) -> dict:
    entities = dict(env_cfg.scene.entities)
    key, fresh = next(iter(build_obstacle_entity(enable_collision, bottom_z).items()))
    existing = entities.get(key)
    if isinstance(existing, ObstacleEntityCfg):
        fresh = replace(
            existing,
            position=fresh.position,
            size=fresh.size,
            contype=fresh.contype,
            conaffinity=fresh.conaffinity,
        )
    entities = {
        name: entity for name, entity in entities.items()
        if not isinstance(entity, ObstacleEntityCfg) or name == key
    }
    entities[key] = fresh
    env_cfg.scene.entities = entities
    return entities



# 从环境里读实际编译进仿真的碰撞开关
def obstacle_collision_enabled(env) -> bool:
    entity = env.scene.entities.get("obstacle")
    return bool(entity is not None and int(entity.cfg.contype) > 0)



# 板体自身的 mocap 下标
def obstacle_mocap_id(entity) -> int:
    local, _ = entity.find_bodies([entity.cfg.name], preserve_order=True)
    if not local:
        raise ValueError(f"{entity.cfg.name} 里找不到同名 body")
    body_id = int(entity.indexing.body_ids[local[0]])
    mocap_id = int(entity.data.model.body_mocapid[body_id])
    if mocap_id < 0:
        raise ValueError(f"{entity.cfg.name} 不是 mocap 体")
    return mocap_id



# 把板体的完整 mocap 位姿写入: x/y 取配置位置, z 取板底高度
def write_obstacle_mocap(entity, bottom_z, env_ids=None) -> None:
    mocap_pos = entity.data.data.mocap_pos
    mocap_id = obstacle_mocap_id(entity)
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
def read_obstacle_bottom_z(entity):
    return entity.data.data.mocap_pos[:, obstacle_mocap_id(entity), 2]



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



# 查询实际场景板几何，无环境参数时返回默认单板配置
def obstacle_geometry(env=None) -> list[dict]:
    if env is None:
        return [_entity_geometry(ObstacleEntityCfg())]
    scene = getattr(env, "scene", env)
    entity = (getattr(scene, "entities", None) or {}).get("obstacle")
    cfg = getattr(entity, "cfg", None)
    return [_entity_geometry(cfg)] if isinstance(cfg, ObstacleEntityCfg) else []



# 设置障碍物板底高度
def set_obstacle_bottom_z(env_cfg, bottom_z: float) -> int:
    entities = dict(env_cfg.scene.entities)
    existing = entities.get("obstacle")
    if not isinstance(existing, ObstacleEntityCfg):
        return 0
    entities["obstacle"] = replace(
        existing,
        position=(float(existing.position[0]), float(existing.position[1]), float(bottom_z)),
    )
    env_cfg.scene.entities = entities
    return 1



# 设置实际板体的可见性，仅改外观
def set_obstacle_visibility(env_cfg, visible: bool, hidden_rgba: tuple = (0.0, 0.0, 0.0, 0.0), shown_rgba: tuple = (0.5, 0.5, 0.5, 0.5)) -> int:
    entities = dict(env_cfg.scene.entities)
    existing = entities.get("obstacle")
    if not isinstance(existing, ObstacleEntityCfg):
        return 0
    entities["obstacle"] = replace(existing, rgba=(tuple(shown_rgba) if visible else tuple(hidden_rgba)))
    env_cfg.scene.entities = entities
    return 1



# 把 checkpoint 里记录的板几何/接触参数应用到 env_cfg
def apply_saved_layout(env_cfg, layout: list) -> int:
    if not layout:
        return 0
    record = layout[0]
    entities = dict(env_cfg.scene.entities)
    existing = entities.get("obstacle")
    if not isinstance(existing, ObstacleEntityCfg):
        return 0
    keep: dict = {}
    if record.get("position") is not None:
        keep["position"] = tuple(float(v) for v in record["position"])
    if record.get("size") is not None:
        keep["size"] = tuple(float(v) for v in record["size"])
    for f in ("solref", "solimp"):
        if record.get(f) is not None:
            keep[f] = tuple(float(v) for v in record[f])
    entities["obstacle"] = replace(existing, **keep)
    env_cfg.scene.entities = entities
    return 1
