from __future__ import annotations
import torch
import mujoco
from dataclasses import dataclass, replace
from mjlab.entity import Entity, EntityCfg



WALL_HALF_THICKNESS = 0.01          # 墙厚半值 (m)
WALL_HALF_LENGTH = 0.30             # 墙沿 Y 方向的半长 (m)
DEFAULT_WALL_HEIGHT = 0.10          # 走廊开口高度 (m)
DEFAULT_WALL_X_NEG = -0.20          # −X 墙中心默认值 (m)
DEFAULT_WALL_X_POS = 0.20           # +X 墙中心默认值 (m)
DEFAULT_CORRIDOR_WIDTH = DEFAULT_WALL_X_POS - DEFAULT_WALL_X_NEG   # 对称简写 = 0.40



@dataclass
class RestrictedSpaceEntityCfg(EntityCfg):
    name: str = "restricted_space"
    wall_x_neg: float = DEFAULT_WALL_X_NEG
    wall_x_pos: float = DEFAULT_WALL_X_POS
    wall_height: float = DEFAULT_WALL_HEIGHT
    wall_half_thickness: float = WALL_HALF_THICKNESS
    wall_half_length: float = WALL_HALF_LENGTH
    rgba: tuple[float, float, float, float] = (0.55, 0.62, 0.72, 0.45)
    # rgba: tuple[float, float, float, float] = (0.55, 0.62, 0.72, 0.0)       # 备用无色透明版本
    contype: int = 1
    conaffinity: int = 1
    solref: tuple[float, float] = (0.005, 1.0)
    solimp: tuple[float, float, float, float, float] = (0.99, 0.999, 0.001, 0.5, 2.0)
    fixed_width: bool = False


    # 两墙中心间距 (m)
    @property
    def corridor_width(self) -> float:
        return self.wall_x_pos - self.wall_x_neg


    # 实际内侧净宽 (m)
    @property
    def clear_width(self) -> float:
        return self.corridor_width - 2.0 * self.wall_half_thickness


    def build(self) -> "RestrictedSpaceEntity":
        return RestrictedSpaceEntity(cfg=self)



class RestrictedSpaceEntity(Entity):
    def __init__(self, cfg: RestrictedSpaceEntityCfg):
        super().__init__(cfg)
        self.cfg = cfg
        self._wall_geoms: list = []
        self._wall_bodies: list = []
        self._build_geometry()


    # 创建几何体
    def _build_geometry(self) -> None:
        half_t = self.cfg.wall_half_thickness
        half_l = self.cfg.wall_half_length
        half_h = 0.5 * self.cfg.wall_height
        for center, tag in ((self.cfg.wall_x_neg, "n"), (self.cfg.wall_x_pos, "p")):
            body = self._spec.worldbody.add_body(name=f"{self.cfg.name}_wall_{tag}", pos=(center, 0.0, 0.0), mocap=True,)
            self._wall_bodies.append(body)
            self._wall_geoms.append(body.add_geom(
                name=f"{self.cfg.name}_wall_{tag}_geom",
                pos=(0.0, 0.0, half_h),
                size=(half_t, half_l, half_h),
                type=mujoco.mjtGeom.mjGEOM_BOX,
                rgba=self.cfg.rgba,
                contype=self.cfg.contype,
                conaffinity=self.cfg.conaffinity,
                solref=self.cfg.solref,
                solimp=self.cfg.solimp,
            ))


    @property
    def wall_geom_ids(self) -> list[int]:
        local, _ = self.find_geoms([f"{self.cfg.name}_wall_.*_geom"], preserve_order=True)
        return list(self.indexing.geom_ids[torch.tensor(local, dtype=torch.long)].cpu().tolist())


    @property
    def wall_body_ids(self) -> list[int]:
        local, _ = self.find_bodies([f"{self.cfg.name}_wall_.*"], preserve_order=True)
        return list(self.indexing.body_ids[torch.tensor(local, dtype=torch.long)].cpu().tolist())


    @property
    def wall_mocap_ids(self) -> list[int]:
        return [int(self.data.model.body_mocapid[b]) for b in self.wall_body_ids]


    # 逐环境写墙位
    def write_wall_x(self, env, x_neg, x_pos, env_ids=None) -> None:
        sim = getattr(env, "sim", None)
        pos = getattr(getattr(sim, "data", None), "mocap_pos", None)
        if pos is None:
            return
        ids = slice(None) if env_ids is None else env_ids
        mn, mp = self.wall_mocap_ids
        for mocap_id, value in ((mn, x_neg), (mp, x_pos)):
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, device=pos.device, dtype=pos.dtype)
            pos[ids, mocap_id, 0] = value


    # 读回逐环境墙位 (世界 x); 无仿真时返回全 0 (单测替身)
    def read_wall_x(self, env) -> tuple:
        sim = getattr(env, "sim", None)
        pos = getattr(getattr(sim, "data", None), "mocap_pos", None)
        if pos is None:
            z = torch.zeros(1)
            return z.clone(), z.clone()
        mn, mp = self.wall_mocap_ids
        return pos[:, mn, 0].clone(), pos[:, mp, 0].clone()


    @property
    def clear_width(self) -> float:
        return self.cfg.clear_width


    @property
    def collision_enabled(self) -> bool:
        return int(self.cfg.contype) > 0


    @property
    def spec(self) -> mujoco.MjSpec:
        return self._spec



# 受限空间位置配置 (wall_x_neg, wall_x_pos)
def _resolve_wall_pair(corridor_width: float | None, wall_x_neg: float | None, wall_x_pos: float | None) -> tuple[float, float]:
    if wall_x_neg is not None or wall_x_pos is not None:
        if wall_x_neg is None or wall_x_pos is None:
            raise ValueError("wall_x_neg 与 wall_x_pos 必须成对给出")
        if corridor_width is not None:
            raise ValueError("corridor_width 与 wall_x_* 不能同时给出 (口径混用)")
        if not wall_x_neg < wall_x_pos:
            raise ValueError(f"wall_x_neg 必须小于 wall_x_pos, 实际 {wall_x_neg} / {wall_x_pos}")
        return float(wall_x_neg), float(wall_x_pos)
    if corridor_width is None:
        raise ValueError("必须给出 corridor_width (对称) 或 wall_x_neg/wall_x_pos (显式)")
    half = 0.5 * float(corridor_width)
    return -half, half



# 创建受限空间实体配置
def build_restricted_space_cfg(enable_collision: bool,
                               corridor_width: float | None = None,
                               fixed_width: bool = False, *,
                               wall_x_neg: float | None = None,
                               wall_x_pos: float | None = None,
                               solref: tuple[float, ...] | None = None,
                               solimp: tuple[float, ...] | None = None) -> RestrictedSpaceEntityCfg:
    neg, pos = _resolve_wall_pair(corridor_width, wall_x_neg, wall_x_pos)
    value = 1 if enable_collision else 0
    extra: dict = {}
    if solref is not None:
        extra["solref"] = tuple(float(v) for v in solref)
    if solimp is not None:
        extra["solimp"] = tuple(float(v) for v in solimp)
    return RestrictedSpaceEntityCfg(
        name="restricted_space",
        wall_x_neg=neg,
        wall_x_pos=pos,
        contype=value,
        conaffinity=value,
        fixed_width=bool(fixed_width),
        **extra,
    )



# 改写场景里的受限空间实体
def configure_restricted_space(env_cfg, corridor_width: float | None = None,
                               enable_collision: bool = True, fixed_width: bool = False, *,
                               wall_x_neg: float | None = None,
                               wall_x_pos: float | None = None,
                               solref: tuple[float, ...] | None = None,
                               solimp: tuple[float, ...] | None = None) -> RestrictedSpaceEntityCfg:
    neg, pos = _resolve_wall_pair(corridor_width, wall_x_neg, wall_x_pos)
    cfg = build_restricted_space_cfg(enable_collision=enable_collision,
                                     wall_x_neg=neg, wall_x_pos=pos,
                                     fixed_width=fixed_width,
                                     solref=solref, solimp=solimp)
    existing = dict(env_cfg.scene.entities).get("restricted_space")
    if isinstance(existing, RestrictedSpaceEntityCfg):
        keep: dict = dict(wall_x_neg=cfg.wall_x_neg, wall_x_pos=cfg.wall_x_pos,
                          contype=cfg.contype, conaffinity=cfg.conaffinity,
                          fixed_width=cfg.fixed_width)
        if solref is not None:
            keep["solref"] = cfg.solref
        if solimp is not None:
            keep["solimp"] = cfg.solimp
        cfg = replace(existing, **keep)
    entities = dict(env_cfg.scene.entities)
    entities["restricted_space"] = cfg
    env_cfg.scene.entities = entities
    return cfg
