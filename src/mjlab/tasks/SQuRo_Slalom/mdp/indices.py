from __future__ import annotations

from mjlab.managers.scene_entity_config import SceneEntityCfg


# 执行器列序与 Hole 一致，所有部位分组由名称派生。
_ACT_JOINT_NAMES = (
    "F_spine1_joint", "F_body_joint", "Neck_yaw_joint", "Neck_pitch_joint",
    "FL_shoulder_joint", "FL_elbow_joint", "FR_shoulder_joint", "FR_elbow_joint",
    "H_spine1_joint", "H_body_joint", "HL_hip_joint", "HL_knee_joint",
    "HR_hip_joint", "HR_knee_joint",
)
_ACT_IDX = {name: i for i, name in enumerate(_ACT_JOINT_NAMES)}
_ACT_F_SPINE1_ID = _ACT_IDX["F_spine1_joint"]
_ACT_F_BODY_ID = _ACT_IDX["F_body_joint"]
_ACT_NECK_YAW_ID = _ACT_IDX["Neck_yaw_joint"]
_ACT_NECK_PITCH_ID = _ACT_IDX["Neck_pitch_joint"]
_ACT_FL_SHOULDER_ID = _ACT_IDX["FL_shoulder_joint"]
_ACT_FL_ELBOW_ID = _ACT_IDX["FL_elbow_joint"]
_ACT_FR_SHOULDER_ID = _ACT_IDX["FR_shoulder_joint"]
_ACT_FR_ELBOW_ID = _ACT_IDX["FR_elbow_joint"]
_ACT_H_SPINE1_ID = _ACT_IDX["H_spine1_joint"]
_ACT_H_BODY_ID = _ACT_IDX["H_body_joint"]
_ACT_HL_HIP_ID = _ACT_IDX["HL_hip_joint"]
_ACT_HL_KNEE_ID = _ACT_IDX["HL_knee_joint"]
_ACT_HR_HIP_ID = _ACT_IDX["HR_hip_joint"]
_ACT_HR_KNEE_ID = _ACT_IDX["HR_knee_joint"]

_ACT_FL_LEG_IDS = (_ACT_FL_SHOULDER_ID, _ACT_FL_ELBOW_ID)
_ACT_FR_LEG_IDS = (_ACT_FR_SHOULDER_ID, _ACT_FR_ELBOW_ID)
_ACT_HL_LEG_IDS = (_ACT_HL_HIP_ID, _ACT_HL_KNEE_ID)
_ACT_HR_LEG_IDS = (_ACT_HR_HIP_ID, _ACT_HR_KNEE_ID)
_ACT_F_LEG_IDS = _ACT_FL_LEG_IDS + _ACT_FR_LEG_IDS
_ACT_H_LEG_IDS = _ACT_HL_LEG_IDS + _ACT_HR_LEG_IDS
_ACT_LEG_IDS = _ACT_F_LEG_IDS + _ACT_H_LEG_IDS
_ACT_SPN_IDS = (_ACT_F_SPINE1_ID, _ACT_F_BODY_ID, _ACT_H_SPINE1_ID, _ACT_H_BODY_ID)
_ACT_NECK_IDS = (_ACT_NECK_YAW_ID, _ACT_NECK_PITCH_ID)

LINKAGE_JOINT_NAMES = (
    "FL_shoulder1_joint", "FL_elbow1_joint", "FL_elbow2_joint", "FL_elbow3_joint",
    "FR_shoulder1_joint", "FR_elbow1_joint", "FR_elbow2_joint", "FR_elbow3_joint",
    "HL_hip1_joint", "HL_knee1_joint", "HL_knee2_joint", "HL_knee3_joint",
    "HR_hip1_joint", "HR_knee1_joint", "HR_knee2_joint", "HR_knee3_joint",
)
FOOT_SITE_NAMES = ("FL_elbow_site", "FR_elbow_site", "HL_knee_site", "HR_knee_site")
FOOT_NAMES = tuple(name.split("_", 1)[0] for name in FOOT_SITE_NAMES)
FOOT_GEOM_NAMES = tuple(f"{name}_foot_collision" for name in FOOT_NAMES)
ACTUATED_JOINT_CFG = SceneEntityCfg("robot", joint_names=_ACT_JOINT_NAMES)


# 只保存模型编号，执行器列索引使用上方常量。
class ModelIndices:
    __slots__ = ("f_body_id", "h_body_id", "foot_site_ids", "joint_ids", "linkage_joint_ids")

    # 初始化尚未解析的模型编号。
    def __init__(self):
        self.f_body_id: int = -1
        self.h_body_id: int = -1
        self.foot_site_ids: tuple[int, ...] = ()
        self.joint_ids: tuple[int, ...] = ()
        self.linkage_joint_ids: tuple[int, ...] = ()


_MODEL_INDICES = ModelIndices()


# 按名称解析身体、足端、驱动关节与闭链关节。
def resolve_model_indices(entity) -> None:
    if _MODEL_INDICES.joint_ids and _MODEL_INDICES.linkage_joint_ids:
        return
    body_ids, _ = entity.find_bodies(("F_body_Link", "H_body_Link"), preserve_order=True)
    foot_ids, _ = entity.find_sites(FOOT_SITE_NAMES, preserve_order=True)
    joint_ids, joint_names = entity.find_joints(_ACT_JOINT_NAMES, preserve_order=True)
    linkage_ids, _ = entity.find_joints(LINKAGE_JOINT_NAMES, preserve_order=True)
    if len(body_ids) != 2 or len(foot_ids) != len(FOOT_SITE_NAMES):
        raise ValueError("身体或足端索引解析不完整")
    if len(joint_ids) != len(_ACT_JOINT_NAMES) or len(linkage_ids) != len(LINKAGE_JOINT_NAMES):
        raise ValueError("驱动或闭链关节索引解析不完整")
    _MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id = body_ids
    _MODEL_INDICES.foot_site_ids = tuple(foot_ids)
    _MODEL_INDICES.joint_ids = tuple(joint_ids)
    _MODEL_INDICES.linkage_joint_ids = tuple(linkage_ids)
    print(f"[SQuRo Slalom] 模型索引解析完成，执行器顺序：{list(joint_names)}")


# 将执行器列分组转换成模型关节编号。
def resolve_actuator_joint_ids(entity, actuator_ids: tuple[int, ...]) -> tuple[int, ...]:
    resolve_model_indices(entity)
    return tuple(_MODEL_INDICES.joint_ids[i] for i in actuator_ids)


# 足端力列按传感器名称排序。
def resolve_foot_force_ids(sensor) -> tuple[int, ...]:
    return tuple(sensor.primary_names.index(name) for name in FOOT_GEOM_NAMES)
