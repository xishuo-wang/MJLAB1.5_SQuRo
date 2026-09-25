from __future__ import annotations

import mujoco
import torch
from dataclasses import dataclass, replace
from mjlab.entity import Entity, EntityCfg


# 受限空间: 走廊沿世界 Y 方向延伸, 两侧墙分别位于 wall_x_neg / wall_x_pos, 约束横向位移。
# **碰撞开关与几何位置都必须在编译期决定** —— 依据见 build_restricted_space_cfg 的说明。
# 墙位在单个 episode 内固定; 换墙位需要重建环境 (与 Slalom 的杆实体同构)。
WALL_HALF_THICKNESS = 0.01          # 墙厚半值 (m)
WALL_HALF_LENGTH = 0.30             # 墙沿 Y 方向的半长 (m)
DEFAULT_WALL_HEIGHT = 0.10          # 走廊开口高度 (m)
DEFAULT_WALL_X_NEG = -0.20          # −X 墙中心默认值 (m)
DEFAULT_WALL_X_POS = 0.20           # +X 墙中心默认值 (m)
DEFAULT_CORRIDOR_WIDTH = DEFAULT_WALL_X_POS - DEFAULT_WALL_X_NEG   # 对称简写 = 0.40


@dataclass
class RestrictedSpaceEntityCfg(EntityCfg):
    name: str = "restricted_space"
    # 墙位是**原语**: 两墙中心的世界 x。可以不对称 (翻正只朝 −X 走, +X 侧几乎不用)。
    # 中心间距与净宽一律由这两个值派生, 不再单独存储, 避免两套口径混用。
    wall_x_neg: float = DEFAULT_WALL_X_NEG
    wall_x_pos: float = DEFAULT_WALL_X_POS
    wall_height: float = DEFAULT_WALL_HEIGHT         # 开口高度 (m)
    wall_half_thickness: float = WALL_HALF_THICKNESS
    wall_half_length: float = WALL_HALF_LENGTH
    rgba: tuple[float, float, float, float] = (0.55, 0.62, 0.72, 0.45)
    # contype/conaffinity 在 put_model 时被固化进碰撞对列表, 运行期改它无效 (实测),
    # 所以这两个值就是"这面墙到底碰不碰"的唯一开关, 必须在编译前定好。
    contype: int = 1
    conaffinity: int = 1
    # True = 本次训练的墙位全程锁死 (不跟随课程), runner 不会为课程推进重建环境
    fixed_width: bool = False

    # 两墙中心间距 (m) —— 派生量, 只为日志与既有验收口径保留
    @property
    def corridor_width(self) -> float:
        return self.wall_x_pos - self.wall_x_neg

    # 实际内侧净宽 (m) = 两内壁之间
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
        self._build_geometry()

    # 两面侧墙: −x 与 +x 各一面, 沿 Y 延伸、沿 Z 立在 0~wall_height。
    # 位置直接取 cfg 的两个墙心 (可不对称), 不再由间距的一半推导。
    def _build_geometry(self) -> None:
        half_t = self.cfg.wall_half_thickness
        half_l = self.cfg.wall_half_length
        half_h = 0.5 * self.cfg.wall_height
        for center, tag in ((self.cfg.wall_x_neg, "n"), (self.cfg.wall_x_pos, "p")):
            body = self._spec.worldbody.add_body(
                name=f"{self.cfg.name}_wall_{tag}",
                pos=(center, 0.0, 0.0),
            )
            self._wall_geoms.append(body.add_geom(
                name=f"{self.cfg.name}_wall_{tag}_geom",
                pos=(0.0, 0.0, half_h),
                size=(half_t, half_l, half_h),
                type=mujoco.mjtGeom.mjGEOM_BOX,
                rgba=self.cfg.rgba,
                contype=self.cfg.contype,
                conaffinity=self.cfg.conaffinity,
            ))

    # 编译后的全局 geom 下标 (顺序同 _build_geometry 的 neg, pos)。
    # 必须换算: find_geoms 返回的是实体内的局部下标, 直接索引 model.geom_contype 会写错几何。
    @property
    def wall_geom_ids(self) -> list[int]:
        local, _ = self.find_geoms([f"{self.cfg.name}_wall_.*_geom"], preserve_order=True)
        return list(self.indexing.geom_ids[torch.tensor(local, dtype=torch.long)].cpu().tolist())

    @property
    def clear_width(self) -> float:
        return self.cfg.clear_width

    # 碰撞是否开启 (编译期决定, 运行期不可改)
    @property
    def collision_enabled(self) -> bool:
        return int(self.cfg.contype) > 0

    @property
    def spec(self) -> mujoco.MjSpec:
        return self._spec


# 把"对称间距"或"显式墙位对"统一成 (wall_x_neg, wall_x_pos)。
# 两种写法只能给一种: 同时给出时静默取其一, 正是"以为改了、其实没改"的来源。
def _resolve_wall_pair(corridor_width: float | None,
                       wall_x_neg: float | None,
                       wall_x_pos: float | None) -> tuple[float, float]:
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


# 建受限空间实体配置 (env_cfg 与回放脚本共用)。
# 两条不可绕过的约束, 都来自 mjwarp 在 put_model 时固化碰撞信息这一事实:
#  1. contype/conaffinity 运行期修改无效 —— 实测编译开启后把两个掩码置 0, 墙接触从
#     93 个只降到 68 个 (仍在阻挡), 机器人在墙内不会被推出。所以"阶段一关碰撞"只能
#     在编译期用 contype=0 实现。
#  2. 几何位置运行期修改无效 —— 实测把墙的 geom_pos 从 ±0.20 平移到 ±0.10 (并 expand +
#     重建 CUDA graph) 后, 机器人放到新墙位接触数为 0。所以"移动墙来收紧走廊"不可行,
#     换墙位必须重建环境。
# 结论: 墙位与碰撞开关都是 env_cfg 的一部分, 与 Slalom 的 PoleEntity 用法一致。
def build_restricted_space_cfg(enable_collision: bool,
                               corridor_width: float | None = None,
                               fixed_width: bool = False, *,
                               wall_x_neg: float | None = None,
                               wall_x_pos: float | None = None) -> RestrictedSpaceEntityCfg:
    neg, pos = _resolve_wall_pair(corridor_width, wall_x_neg, wall_x_pos)
    value = 1 if enable_collision else 0
    return RestrictedSpaceEntityCfg(
        name="restricted_space",
        wall_x_neg=neg,
        wall_x_pos=pos,
        contype=value,
        conaffinity=value,
        fixed_width=bool(fixed_width),
    )


# 按目标墙位改写场景里的受限空间实体 (必须在建环境之前调用; 课程推进用)
def configure_restricted_space(env_cfg, corridor_width: float | None = None,
                               enable_collision: bool = True, fixed_width: bool = False, *,
                               wall_x_neg: float | None = None,
                               wall_x_pos: float | None = None) -> RestrictedSpaceEntityCfg:
    neg, pos = _resolve_wall_pair(corridor_width, wall_x_neg, wall_x_pos)
    cfg = build_restricted_space_cfg(enable_collision=enable_collision,
                                     wall_x_neg=neg, wall_x_pos=pos,
                                     fixed_width=fixed_width)
    existing = dict(env_cfg.scene.entities).get("restricted_space")
    # 只改课程控制的两个墙位与碰撞开关, 其余尺寸 (墙高/半长/半厚/配色) 必须沿用启动配置 ——
    # 直接新建默认 cfg 会把自定义墙体尺寸悄悄改回默认值。见技术细节 §7.11 第 9 条。
    if isinstance(existing, RestrictedSpaceEntityCfg):
        cfg = replace(existing, wall_x_neg=cfg.wall_x_neg, wall_x_pos=cfg.wall_x_pos,
                      contype=cfg.contype, conaffinity=cfg.conaffinity,
                      fixed_width=cfg.fixed_width)
    entities = dict(env_cfg.scene.entities)
    entities["restricted_space"] = cfg
    env_cfg.scene.entities = entities
    return cfg
