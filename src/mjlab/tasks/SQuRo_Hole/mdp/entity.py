from dataclasses import dataclass, replace
import mujoco
import torch
from mjlab.entity import Entity, EntityCfg


# 三块限高板的几何: 板底 z 即板体 position.z, 板厚 2*half_thickness 向上长
# 板底 = position.z (编译期默认值, 也是每次复位的回落值), 板顶 = position.z + 2*half_thickness
# 运行期板底 = mocap_pos.z, 由 mode="reset" 事件逐环境写入 (见 mdp/events.reset_obstacle_height)
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
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)      # 板底中心 (x, y, 板底 z)
    size: tuple[float, float, float] = (0.01, 0.1, 0.005)       # 半长, 半宽, 半厚
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
        # 板体自身建成 mocap body: 只有这样 mujoco_warp 才会逐帧更新 geom_xpos
        # (见 docs/SQuRo_Hole_受限空间随机化.md §2)
        body = self._spec.worldbody.add_body(name=self.cfg.name, pos=self.cfg.position, mocap=True,)
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


    # 板体全局 body 下标
    @property
    def board_body_id(self) -> int:
        local, _ = self.find_bodies([self.cfg.name], preserve_order=True)
        return int(self.indexing.body_ids[local[0]])


    # 板体 mocap 下标 (逐环境摆位用; 取板体自身, 不是 root 的空 mocap_base)
    @property
    def board_mocap_id(self) -> int:
        return int(self.data.model.body_mocapid[self.board_body_id])


    # 逐环境写板底高度 (m); 无仿真时静默跳过 (单测替身)
    def write_bottom_z(self, env, bottom_z, env_ids=None) -> None:
        sim = getattr(env, "sim", None)
        pos = getattr(getattr(sim, "data", None), "mocap_pos", None)
        if pos is None:
            return
        ids = slice(None) if env_ids is None else env_ids
        if not torch.is_tensor(bottom_z):
            bottom_z = torch.as_tensor(bottom_z, device=pos.device, dtype=pos.dtype)
        pos[ids, self.board_mocap_id, 2] = bottom_z


    # 读回逐环境板底高度 (m); 无仿真时返回全 0
    def read_bottom_z(self, env):
        sim = getattr(env, "sim", None)
        pos = getattr(getattr(sim, "data", None), "mocap_pos", None)
        if pos is None:
            return torch.zeros(1)
        return pos[:, self.board_mocap_id, 2].clone()


    # 板底高度 (m)
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
def build_hole_entities(enable_collision: bool, solref: tuple[float, ...] | None = None, solimp: tuple[float, ...] | None = None) -> dict:
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
def configure_hole_entities(env_cfg, enable_collision: bool, solref: tuple[float, ...] | None = None, solimp: tuple[float, ...] | None = None) -> dict:
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



# 从环境里读实际编译进仿真的碰撞开关
def hole_collision_enabled(env) -> bool:
    entity = env.scene.entities.get("hole1")
    return bool(entity is not None and int(entity.cfg.contype) > 0)



# 单块板的几何字典 (实体口径: position.z 是板底, 半厚在 size[2])
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



# 统一几何查询: 返回三块板的 x 区间 / 板底 / 虚拟阈值。
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



# 设置障碍物的可见性
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



# 回放时应用保存的配置
def apply_saved_layout(env_cfg, layout: list) -> int:
    if not layout:
        return 0
    entities = dict(env_cfg.scene.entities)
    applied = 0
    for d in layout:
        key = str(d.get("name", "")).lower()
        existing = entities.get(key)
        if not isinstance(existing, HoleEntityCfg):
            continue
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
