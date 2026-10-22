from __future__ import annotations
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.asset_zoo.robots.SQuRo.SQuRo_constants import (
    SPINE_FRONT_FRAME as SPINE_FRONT_FRAME,
    SPINE_HIND_FRAME as SPINE_HIND_FRAME,
    SPINE_DEFORMATION_BODY_NAMES,
)


# 执行器关节名称
_ACT_JOINT_NAMES = [
    "F_spine1_joint",
    "F_body_joint",
    "Neck_yaw_joint",
    "Neck_pitch_joint",
    "FL_shoulder_joint",
    "FL_elbow_joint",
    "FR_shoulder_joint",
    "FR_elbow_joint",
    "H_spine1_joint",
    "H_body_joint",
    "HL_hip_joint",
    "HL_knee_joint",
    "HR_hip_joint",
    "HR_knee_joint",
]


# 从关节名称推导执行器列号，保持现有十四维动作顺序。
_ACT_IDX = {name: i for i, name in enumerate(_ACT_JOINT_NAMES)}
_ACT_NUM = len(_ACT_JOINT_NAMES)
_ACT_F_SPINE1_ID = _ACT_IDX["F_spine1_joint"]
_ACT_F_BODY_ID = _ACT_IDX["F_body_joint"]
_ACT_NECK_YAW_ID = _ACT_IDX["Neck_yaw_joint"]
_ACT_NECK_PITCH_ID = _ACT_IDX["Neck_pitch_joint"]
_ACT_H_SPINE1_ID = _ACT_IDX["H_spine1_joint"]
_ACT_H_BODY_ID = _ACT_IDX["H_body_joint"]
_ACT_FL_LEG_IDS = tuple(_ACT_IDX[name] for name in ("FL_shoulder_joint", "FL_elbow_joint"))
_ACT_FR_LEG_IDS = tuple(_ACT_IDX[name] for name in ("FR_shoulder_joint", "FR_elbow_joint"))
_ACT_HL_LEG_IDS = tuple(_ACT_IDX[name] for name in ("HL_hip_joint", "HL_knee_joint"))
_ACT_HR_LEG_IDS = tuple(_ACT_IDX[name] for name in ("HR_hip_joint", "HR_knee_joint"))
_ACT_LEG_F_IDS = _ACT_FL_LEG_IDS + _ACT_FR_LEG_IDS
_ACT_LEG_H_IDS = _ACT_HL_LEG_IDS + _ACT_HR_LEG_IDS
_ACT_LEG_IDS = _ACT_LEG_F_IDS + _ACT_LEG_H_IDS
_ACT_SPN_IDS = (_ACT_F_SPINE1_ID, _ACT_F_BODY_ID, _ACT_H_SPINE1_ID, _ACT_H_BODY_ID)
_ACT_NECK_IDS = (_ACT_NECK_YAW_ID, _ACT_NECK_PITCH_ID)
_ACT_SPN_BODY_IDS = (_ACT_F_BODY_ID, _ACT_H_BODY_ID)

# 保留已有诊断脚本的导入接口。
_ACTUATED_JOINT_NAMES = _ACT_JOINT_NAMES


# 闭链被动关节与足端名称。
LINKAGE_JOINT_NAMES = (
    "FL_shoulder1_joint",
    "FL_elbow1_joint",
    "FL_elbow2_joint",
    "FL_elbow3_joint",
    "FR_shoulder1_joint",
    "FR_elbow1_joint",
    "FR_elbow2_joint",
    "FR_elbow3_joint",
    "HL_hip1_joint",
    "HL_knee1_joint",
    "HL_knee2_joint",
    "HL_knee3_joint",
    "HR_hip1_joint",
    "HR_knee1_joint",
    "HR_knee2_joint",
    "HR_knee3_joint",
)
BODY_NAMES = ("F_body_Link", "H_body_Link")
FOOT_SITE_NAMES = ("FL_elbow_site", "FR_elbow_site", "HL_knee_site", "HR_knee_site")
FOOT_NAMES = tuple(name.split("_", 1)[0] for name in FOOT_SITE_NAMES)
FOOT_GEOM_NAMES = tuple(f"{name}_foot_collision" for name in FOOT_NAMES)


# 观测用配置
ACTUATED_JOINT_CFG = SceneEntityCfg("robot", joint_names=tuple(_ACT_JOINT_NAMES))


# 躯干朝向标记: 见 docs/SQuRo_Backup_技术细节.md §4
SEGMENT_BELLY_BACK_SITES = (
    ("F_body_belly_site", "F_body_back_site"),
    ("H_body_belly_site", "H_body_back_site"),
)


# 按名称解析当前实体的三段索引，不复用训练任务的全局缓存。
def resolve_spine_deformation_indices(entity) -> tuple[int, int, int]:
    ids, names = entity.find_bodies(list(SPINE_DEFORMATION_BODY_NAMES), preserve_order=True)
    if tuple(names) != SPINE_DEFORMATION_BODY_NAMES:
        raise ValueError(f"脊柱形变所需刚体不完整或顺序不符：{names}")
    front, center, hind = ids
    return int(front), int(center), int(hind)


# action 张量中的腿/脊柱/颈部列索引（与 entity actuator 顺序一致）
_ACTION_LEG_IDS = _ACT_LEG_IDS
_ACTION_SPN_IDS = _ACT_SPN_IDS
_ACTION_NECK_IDS = _ACT_NECK_IDS
_ACTION_SPN_LATERAL_ID = _ACT_F_SPINE1_ID
_ACTION_SPN_BODY_IDS = _ACT_SPN_BODY_IDS


# 执行器控制范围
_ACTUATOR_CTRL_RANGE: dict[str, tuple[float, float]] = {
    "F_spine1_joint": (-0.6, 0.6),
    "F_body_joint": (-1.57, 1.57),
    "Neck_yaw_joint": (-0.8, 0.8),
    "Neck_pitch_joint": (-0.9, 0.9),
    "FL_shoulder_joint": (-1.5, 1.9),
    "FL_elbow_joint": (-1.8, 2.5),
    "FR_shoulder_joint": (-1.5, 1.9),
    "FR_elbow_joint": (-1.8, 2.5),
    "H_spine1_joint": (-0.6, 0.6),
    "H_body_joint": (-1.57, 1.57),
    "HL_hip_joint": (-1.5, 0.8),
    "HL_knee_joint": (-0.5, 1.9),
    "HR_hip_joint": (-1.5, 0.8),
    "HR_knee_joint": (-0.5, 1.9),
}


class ModelIndices:
    __slots__ = (
        "f_body_id",
        "h_body_id",
        "foot_site_ids",
        "segment_belly_back_ids",
        "joint_ids",
        "linkage_joint_ids",
        "joint_leg_ids",
        "joint_spn_ids",
        "joint_neck_ids",
        "actuator_leg_ids",
        "actuator_spn_ids",
        "actuator_neck_ids",
        "actuator_spn_lateral_id",
        "actuator_spn_body_ids",
    )

    # 初始化待解析的模型索引。
    def __init__(self):
        self.f_body_id: int = -1
        self.h_body_id: int = -1
        self.foot_site_ids: tuple[int, ...] = ()
        self.segment_belly_back_ids: tuple[tuple[int, int], tuple[int, int]] | None = None
        self.joint_ids: tuple[int, ...] = ()
        self.linkage_joint_ids: tuple[int, ...] = ()
        self.joint_leg_ids: tuple[int, ...] = ()
        self.joint_spn_ids: tuple[int, ...] = ()
        self.joint_neck_ids: tuple[int, ...] = ()
        self.actuator_leg_ids: tuple[int, ...] = _ACTION_LEG_IDS
        self.actuator_spn_ids: tuple[int, ...] = _ACTION_SPN_IDS
        self.actuator_neck_ids: tuple[int, ...] = _ACTION_NECK_IDS
        self.actuator_spn_lateral_id: int = _ACTION_SPN_LATERAL_ID
        self.actuator_spn_body_ids: tuple[int, ...] = _ACTION_SPN_BODY_IDS


_MODEL_INDICES = ModelIndices()


# 完整解析身体、足端、背腹标记、驱动与闭链关节后一次写入缓存。
def resolve_model_indices(entity) -> None:
    if (
        _MODEL_INDICES.f_body_id >= 0
        and len(_MODEL_INDICES.joint_ids) == _ACT_NUM
        and len(_MODEL_INDICES.linkage_joint_ids) == len(LINKAGE_JOINT_NAMES)
    ):
        return

    body_ids, body_names = entity.find_bodies(BODY_NAMES, preserve_order=True)
    site_ids, site_names = entity.find_sites(FOOT_SITE_NAMES, preserve_order=True)
    joint_ids, joint_names = entity.find_joints(_ACT_JOINT_NAMES, preserve_order=True)
    linkage_ids, linkage_names = entity.find_joints(LINKAGE_JOINT_NAMES, preserve_order=True)
    if (
        tuple(body_names) != BODY_NAMES
        or tuple(site_names) != FOOT_SITE_NAMES
        or list(joint_names) != _ACT_JOINT_NAMES
        or tuple(linkage_names) != LINKAGE_JOINT_NAMES
    ):
        raise ValueError("身体、足端或关节名称解析不完整或顺序不符")

    belly_back = []
    for belly_name, back_name in SEGMENT_BELLY_BACK_SITES:
        ids, names = entity.find_sites([belly_name, back_name], preserve_order=True)
        if tuple(names) != (belly_name, back_name):
            raise ValueError("躯干背腹标记解析不完整或顺序不符")
        belly_back.append((ids[0], ids[1]))

    _MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id = body_ids
    _MODEL_INDICES.foot_site_ids = tuple(site_ids)
    _MODEL_INDICES.segment_belly_back_ids = (belly_back[0], belly_back[1])
    _MODEL_INDICES.joint_ids = tuple(joint_ids)
    _MODEL_INDICES.linkage_joint_ids = tuple(linkage_ids)
    _MODEL_INDICES.joint_leg_ids = tuple(joint_ids[i] for i in _ACT_LEG_IDS)
    _MODEL_INDICES.joint_spn_ids = tuple(joint_ids[i] for i in _ACT_SPN_IDS)
    _MODEL_INDICES.joint_neck_ids = tuple(joint_ids[i] for i in _ACT_NECK_IDS)

    print("\n[SQuRo] 模型索引解析完成:")
    print(f"  F_body_Link: {body_names[0]} -> ID {body_ids[0]}")
    print(f"  H_body_Link: {body_names[1]} -> ID {body_ids[1]}")
    print(f"  足端 site IDs: {_MODEL_INDICES.foot_site_ids}")
    print(f"  躯干腹/背 site IDs: {_MODEL_INDICES.segment_belly_back_ids}")
    print(f"  actuator 顺序: {list(joint_names)}")


# 按传感器公开名称解析足端接触力的列顺序。
def resolve_foot_force_ids(sensor) -> tuple[int, ...]:
    return tuple(sensor.primary_names.index(name) for name in FOOT_GEOM_NAMES)
