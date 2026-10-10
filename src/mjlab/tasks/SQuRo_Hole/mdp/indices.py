from __future__ import annotations
from mjlab.managers.scene_entity_config import SceneEntityCfg



# 执行器顺序
_ACT_JOINT_NAMES = [
    "F_spine1_joint",       # 脊柱侧摆: 0
    "F_body_joint",         # 前脊柱扭转: 1
    "Neck_yaw_joint",       # 头部侧摆: 2
    "Neck_pitch_joint",     # 头部俯仰: 3
    "FL_shoulder_joint",    # 左前肩: 4
    "FL_elbow_joint",       # 左前肘: 5
    "FR_shoulder_joint",    # 右前肩: 6
    "FR_elbow_joint",       # 右前肘: 7
    "H_spine1_joint",       # 脊柱俯仰: 8
    "H_body_joint",         # 后脊柱扭转: 9
    "HL_hip_joint",         # 左后髋: 10
    "HL_knee_joint",        # 左后膝: 11
    "HR_hip_joint",         # 右后髋: 12
    "HR_knee_joint",        # 右后膝: 13
]
_ACT_NUM = len(_ACT_JOINT_NAMES)                                # 驱动关节数量
_ACT_JOINT_IDS = tuple(range(len(_ACT_JOINT_NAMES)))            # 驱动关节索引


# 执行器对应索引
_ACT_IDX = {name: i for i, name in enumerate(_ACT_JOINT_NAMES)}
_ACT_F_SPINE1_ID    = _ACT_IDX["F_spine1_joint"]      # 脊柱侧摆: 0
_ACT_F_BODY_ID      = _ACT_IDX["F_body_joint"]        # 前脊柱扭转: 1
_ACT_NECK_YAW_ID    = _ACT_IDX["Neck_yaw_joint"]      # 头部侧摆: 2
_ACT_NECK_PITCH_ID  = _ACT_IDX["Neck_pitch_joint"]    # 头部俯仰: 3
_ACT_FL_SHOULDER_ID = _ACT_IDX["FL_shoulder_joint"]   # 左前肩: 4
_ACT_FL_ELBOW_ID    = _ACT_IDX["FL_elbow_joint"]      # 左前肘: 5
_ACT_FR_SHOULDER_ID = _ACT_IDX["FR_shoulder_joint"]   # 右前肩: 6
_ACT_FR_ELBOW_ID    = _ACT_IDX["FR_elbow_joint"]      # 右前肘: 7
_ACT_H_SPINE1_ID    = _ACT_IDX["H_spine1_joint"]      # 脊柱俯仰: 8
_ACT_H_BODY_ID      = _ACT_IDX["H_body_joint"]        # 后脊柱扭转: 9
_ACT_HL_HIP_ID      = _ACT_IDX["HL_hip_joint"]        # 左后髋: 10
_ACT_HL_KNEE_ID     = _ACT_IDX["HL_knee_joint"]       # 左后膝: 11
_ACT_HR_HIP_ID      = _ACT_IDX["HR_hip_joint"]        # 右后髋: 12
_ACT_HR_KNEE_ID     = _ACT_IDX["HR_knee_joint"]       # 右后膝: 13


# 腿部索引
_ACT_FL_LEG_IDS = (_ACT_FL_SHOULDER_ID, _ACT_FL_ELBOW_ID)       # 左前腿: (4, 5)
_ACT_FR_LEG_IDS = (_ACT_FR_SHOULDER_ID, _ACT_FR_ELBOW_ID)       # 右前腿: (6, 7)
_ACT_HL_LEG_IDS = (_ACT_HL_HIP_ID,      _ACT_HL_KNEE_ID)        # 左后腿: (10, 11)
_ACT_HR_LEG_IDS = (_ACT_HR_HIP_ID,      _ACT_HR_KNEE_ID)        # 右后腿: (12, 13)
_ACT_F_LEG_IDS  = _ACT_FL_LEG_IDS + _ACT_FR_LEG_IDS             # 前腿合并: (4, 5, 6, 7)
_ACT_H_LEG_IDS  = _ACT_HL_LEG_IDS + _ACT_HR_LEG_IDS             # 后腿合并: (10, 11, 12, 13)
_ACT_LEG_IDS    = _ACT_F_LEG_IDS  + _ACT_H_LEG_IDS              # 全腿合并: (4, 5, 6, 7, 10, 11, 12, 13)


# 脊柱索引
_ACT_SPN_F_YAW_ID   = _ACT_F_SPINE1_ID                          # 前脊柱侧摆: 0
_ACT_SPN_F_ROLL_ID  = _ACT_F_BODY_ID                            # 前脊柱扭转: 1
_ACT_SPN_H_PITCH_ID = _ACT_H_SPINE1_ID                          # 后脊柱俯仰: 8
_ACT_SPN_H_ROLL_ID  = _ACT_H_BODY_ID                            # 后脊柱扭转: 9
_ACT_SPN_F_IDS = (_ACT_SPN_F_YAW_ID,   _ACT_SPN_F_ROLL_ID)      # 前脊柱合并: (0, 1)
_ACT_SPN_H_IDS = (_ACT_SPN_H_PITCH_ID, _ACT_SPN_H_ROLL_ID)      # 后脊柱合并: (8, 9)
_ACT_SPN_ROLL_IDS = (_ACT_SPN_F_ROLL_ID, _ACT_SPN_H_ROLL_ID)    # 脊柱扭转合并: (1, 9)
_ACT_SPN_IDS   = _ACT_SPN_F_IDS + _ACT_SPN_H_IDS                # 脊柱合并: (0, 1, 8, 9)


# 头颈索引
_ACT_NECK_IDS = (_ACT_NECK_YAW_ID, _ACT_NECK_PITCH_ID)          # 头颈两列: (2, 3)


# 执行器控制范围
_ACT_CTRL_RANGE: dict[str, tuple[float, float]] = {
    "F_spine1_joint":    (-0.6, 0.6),
    "F_body_joint":      (-1.57, 1.57),
    "Neck_yaw_joint":    (-0.8, 0.8),
    "Neck_pitch_joint":  (-0.9, 0.9),
    "FL_shoulder_joint": (-1.5, 1.9),
    "FL_elbow_joint":    (-1.8, 2.5),
    "FR_shoulder_joint": (-1.5, 1.9),
    "FR_elbow_joint":    (-1.8, 2.5),
    "H_spine1_joint":    (-0.6, 0.6),
    "H_body_joint":      (-1.57, 1.57),
    "HL_hip_joint":      (-1.5, 0.8),
    "HL_knee_joint":     (-0.5, 1.9),
    "HR_hip_joint":      (-1.5, 0.8),
    "HR_knee_joint":     (-0.5, 1.9),
}


# 虚拟碰撞采样点
FRONT_SEG_SITE_NAMES = tuple(f"F_body_{i}_site" for i in range(1, 10))
REAR_SEG_SITE_NAMES  = tuple(f"H_body_{i}_site" for i in range(1, 10))
FOOT_SITE_NAMES      = ("FL_elbow_site", "FR_elbow_site", "HL_knee_site", "HR_knee_site")


# 观测用配置
ACT_JOINT_CFG = SceneEntityCfg("robot", joint_names=tuple(_ACT_JOINT_NAMES))



class ModelIndices:
    __slots__ = (
        "f_body_id", 
        "h_body_id",
        "foot_site_ids",
        "f_body_site_ids", 
        "h_body_site_ids",
        "joint_ids",
    )


    def __init__(self):
        self.f_body_id: int = -1
        self.h_body_id: int = -1
        self.f_body_site_ids: tuple[int, ...] = ()
        self.h_body_site_ids: tuple[int, ...] = ()
        self.foot_site_ids: tuple[int, ...] = ()
        self.joint_ids: tuple[int, ...] = ()

_MODEL_INDICES = ModelIndices()



def resolve_model_indices(entity) -> None:
    if _MODEL_INDICES.f_body_id >= 0 and len(_MODEL_INDICES.joint_ids) > 0:
        return

    body_ids, _ = entity.find_bodies(["F_body_Link", "H_body_Link"], preserve_order=True)
    _MODEL_INDICES.f_body_id = body_ids[0]
    _MODEL_INDICES.h_body_id = body_ids[1]

    front_ids, _ = entity.find_sites(list(FRONT_SEG_SITE_NAMES), preserve_order=True)
    rear_ids, _ = entity.find_sites(list(REAR_SEG_SITE_NAMES), preserve_order=True)
    foot_ids, _ = entity.find_sites(list(FOOT_SITE_NAMES), preserve_order=True)
    _MODEL_INDICES.f_body_site_ids = tuple(front_ids)
    _MODEL_INDICES.h_body_site_ids = tuple(rear_ids)
    _MODEL_INDICES.foot_site_ids = tuple(foot_ids)

    joint_ids, joint_names = entity.find_joints(_ACT_JOINT_NAMES, preserve_order=True)
    assert len(joint_ids) == len(_ACT_JOINT_NAMES), (
        f"只解析到 {len(joint_ids)}/{len(_ACT_JOINT_NAMES)} 个被控关节: "
        f"{list(joint_names)}")
    _MODEL_INDICES.joint_ids = tuple(joint_ids)

    print("\n[SQuRo Hole] 模型索引解析完成:")
    print(f"  F_body_Link -> {_MODEL_INDICES.f_body_id}, H_body_Link -> {_MODEL_INDICES.h_body_id}")
    print(f"  前段 site ids: {_MODEL_INDICES.f_body_site_ids}")
    print(f"  后段 site ids: {_MODEL_INDICES.h_body_site_ids}")
    print(f"  足端 site ids: {_MODEL_INDICES.foot_site_ids}")
    print(f"  actuator 顺序: {list(joint_names)}")
