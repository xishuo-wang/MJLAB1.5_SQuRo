from __future__ import annotations
from mjlab.managers.scene_entity_config import SceneEntityCfg



# 执行器顺序
_ACTUATED_JOINT_NAMES = [
    "F_spine1_joint",       # 0  脊柱侧摆
    "F_body_joint",         # 1  前脊柱扭转
    "Neck_yaw_joint",       # 2  头部侧摆
    "Neck_pitch_joint",     # 3  头部俯仰
    "FL_shoulder_joint",    # 4  左前肩
    "FL_elbow_joint",       # 5  左前肘
    "FR_shoulder_joint",    # 6  右前肩
    "FR_elbow_joint",       # 7  右前肘
    "H_spine1_joint",       # 8  脊柱俯仰
    "H_body_joint",         # 9  后脊柱扭转
    "HL_hip_joint",         # 10 左后髋
    "HL_knee_joint",        # 11 左后膝
    "HR_hip_joint",         # 12 右后髋
    "HR_knee_joint",        # 13 右后膝
]



# 执行器对应索引
_ACTUATED_IDX = {name: i for i, name in enumerate(_ACTUATED_JOINT_NAMES)}
_ACTUATED_F_SPINE1_ID    = _ACTUATED_IDX["F_spine1_joint"]
_ACTUATED_F_BODY_ID      = _ACTUATED_IDX["F_body_joint"]
_ACTUATED_NECK_YAW_ID    = _ACTUATED_IDX["Neck_yaw_joint"]
_ACTUATED_NECK_PITCH_ID  = _ACTUATED_IDX["Neck_pitch_joint"]
_ACTUATED_FL_SHOULDER_ID = _ACTUATED_IDX["FL_shoulder_joint"]
_ACTUATED_FL_ELBOW_ID    = _ACTUATED_IDX["FL_elbow_joint"]
_ACTUATED_FR_SHOULDER_ID = _ACTUATED_IDX["FR_shoulder_joint"]
_ACTUATED_FR_ELBOW_ID    = _ACTUATED_IDX["FR_elbow_joint"]
_ACTUATED_H_SPINE1_ID    = _ACTUATED_IDX["H_spine1_joint"]
_ACTUATED_H_BODY_ID      = _ACTUATED_IDX["H_body_joint"]
_ACTUATED_HL_HIP_ID      = _ACTUATED_IDX["HL_hip_joint"]
_ACTUATED_HL_KNEE_ID     = _ACTUATED_IDX["HL_knee_joint"]
_ACTUATED_HR_HIP_ID      = _ACTUATED_IDX["HR_hip_joint"]
_ACTUATED_HR_KNEE_ID     = _ACTUATED_IDX["HR_knee_joint"]


# ==================== 逐条腿（前后 × 左右）====================
_ACTUATED_FL_LEG_IDS = (_ACTUATED_FL_SHOULDER_ID, _ACTUATED_FL_ELBOW_ID)   # (4, 5)  左前腿
_ACTUATED_FR_LEG_IDS = (_ACTUATED_FR_SHOULDER_ID, _ACTUATED_FR_ELBOW_ID)   # (6, 7)  右前腿
_ACTUATED_HL_LEG_IDS = (_ACTUATED_HL_HIP_ID,      _ACTUATED_HL_KNEE_ID)    # (10, 11) 左后腿
_ACTUATED_HR_LEG_IDS = (_ACTUATED_HR_HIP_ID,      _ACTUATED_HR_KNEE_ID)    # (12, 13) 右后腿


# ==================== 腿合并组 ====================
_ACTUATED_FRONT_LEG_IDS = _ACTUATED_FL_LEG_IDS + _ACTUATED_FR_LEG_IDS       # (4, 5, 6, 7)
_ACTUATED_HIND_LEG_IDS  = _ACTUATED_HL_LEG_IDS + _ACTUATED_HR_LEG_IDS       # (10, 11, 12, 13)
_ACTUATED_LEG_IDS       = _ACTUATED_FRONT_LEG_IDS + _ACTUATED_HIND_LEG_IDS


# ==================== 脊柱（单下标）====================
_ACTUATED_SPN_LAT_ID        = _ACTUATED_F_SPINE1_ID   # 0   前脊柱侧摆（yaw）
_ACTUATED_SPN_TWIST_ID      = _ACTUATED_F_BODY_ID     # 1   脊柱扭转
_ACTUATED_SPN_PITCH_ID      = _ACTUATED_H_SPINE1_ID   # 8   脊柱俯仰
_ACTUATED_SPN_TWIST_HIND_ID = _ACTUATED_H_BODY_ID     # 9   后脊柱扭转

# 兼容旧名字（原文件里叫 _ACTUATED_SPN_LATERAL_ID）
_ACTUATED_SPN_LATERAL_ID    = _ACTUATED_SPN_LAT_ID    # 0   前脊柱侧摆（yaw）


# ==================== 脊柱合并组 ====================
# 前脊柱两列：侧摆 + 扭转
_ACTUATED_SPN_FRONT_IDS = (_ACTUATED_SPN_LAT_ID,   _ACTUATED_SPN_TWIST_ID)        # (0, 1)
# 后脊柱两列：俯仰 + 扭转
_ACTUATED_SPN_HIND_IDS  = (_ACTUATED_SPN_PITCH_ID, _ACTUATED_SPN_TWIST_HIND_ID)   # (8, 9)
# 脊柱 4 列（参考表顺序：F_spine1, F_body, H_spine1, H_body）
_ACTUATED_SPN_IDS       = _ACTUATED_SPN_FRONT_IDS + _ACTUATED_SPN_HIND_IDS        # (0, 1, 8, 9)

# 两根"扭转"列（用于只看扭转的观测/奖励）
_ACTUATED_SPN_TWIST_IDS = (_ACTUATED_SPN_TWIST_ID, _ACTUATED_SPN_TWIST_HIND_ID)   # (1, 9)
# 兼容旧名字（原文件里叫 _ACTUATED_SPN_BODY_IDS）
_ACTUATED_SPN_BODY_IDS  = _ACTUATED_SPN_TWIST_IDS


# ==================== 头颈 ====================
_ACTUATED_NECK_IDS = (_ACTUATED_NECK_YAW_ID, _ACTUATED_NECK_PITCH_ID)   # (2, 3)


# ==================== 参考表 14 列序 ====================
# 顺序: 前腿 4 + 后腿 4 + 脊柱 4 + 头颈 2
REF_TABLE_ORDER = (
    _ACTUATED_FRONT_LEG_IDS          # (4, 5, 6, 7)
    + _ACTUATED_HIND_LEG_IDS         # (10, 11, 12, 13)
    + _ACTUATED_SPN_IDS              # (0, 1, 8, 9)
    + _ACTUATED_NECK_IDS             # (2, 3)
)   # (4, 5, 6, 7, 10, 11, 12, 13, 0, 1, 8, 9, 2, 3)


# ==================== 执行器控制范围 ====================
_ACTUATOR_CTRL_RANGE: dict[str, tuple[float, float]] = {
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


# ==================== 头颈初值 ====================
NECK_JOINT_NAMES = ("Neck_yaw_joint", "Neck_pitch_joint")
NECK_INIT_POS    = (0.0, -0.3)


# ==================== 虚拟碰撞采样点 ====================
FRONT_SEG_SITE_NAMES = tuple(f"F_body_{i}_site" for i in range(1, 10))
REAR_SEG_SITE_NAMES  = tuple(f"H_body_{i}_site" for i in range(1, 10))
FOOT_SITE_NAMES      = ("FL_elbow_site", "FR_elbow_site", "HL_knee_site", "HR_knee_site")


# ==================== 观测用配置 ====================
ACTUATED_JOINT_CFG = SceneEntityCfg("robot", joint_names=tuple(_ACTUATED_JOINT_NAMES))


class ModelIndices:
    __slots__ = (
        "f_body_id", "h_body_id",
        "foot_site_ids",
        "f_body_site_ids", "h_body_site_ids",
        "joint_ids", 
        "joint_leg_ids", "joint_spn_ids", "joint_neck_ids",
        "actuator_leg_ids", "actuator_spn_ids", "actuator_neck_ids",
        "actuator_spn_lateral_id", "actuator_spn_body_ids",
    )


    def __init__(self):
        self.f_body_id: int = -1
        self.h_body_id: int = -1
        self.f_body_site_ids: tuple[int, ...] = ()
        self.h_body_site_ids: tuple[int, ...] = ()
        self.foot_site_ids: tuple[int, ...] = ()
        self.joint_ids: tuple[int, ...] = ()
        self.joint_leg_ids: tuple[int, ...] = ()
        self.joint_spn_ids: tuple[int, ...] = ()
        self.joint_neck_ids: tuple[int, ...] = ()
        self.actuator_leg_ids: tuple[int, ...] = _ACTUATED_LEG_IDS
        self.actuator_spn_ids: tuple[int, ...] = _ACTUATED_SPN_IDS
        self.actuator_neck_ids: tuple[int, ...] = _ACTUATED_NECK_IDS
        self.actuator_spn_lateral_id: int = _ACTUATED_SPN_LATERAL_ID
        self.actuator_spn_body_ids: tuple[int, ...] = _ACTUATED_SPN_BODY_IDS


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

    joint_ids, joint_names = entity.find_joints(_ACTUATED_JOINT_NAMES, preserve_order=True)
    assert len(joint_ids) == len(_ACTUATED_JOINT_NAMES), (
        f"只解析到 {len(joint_ids)}/{len(_ACTUATED_JOINT_NAMES)} 个被控关节: "
        f"{list(joint_names)}")
    _MODEL_INDICES.joint_ids = tuple(joint_ids)
    # 前腿 4-7 / 后腿 10-13 / 脊柱 0-1,8-9 / 头颈 2-3
    name_to_id = {
        n: i for n, i in zip(_ACTUATED_JOINT_NAMES, joint_ids, strict=True)}
    _MODEL_INDICES.joint_leg_ids = tuple(
        name_to_id[n] for n in _ACTUATED_JOINT_NAMES[4:8] + _ACTUATED_JOINT_NAMES[10:14])
    _MODEL_INDICES.joint_spn_ids = tuple(
        name_to_id[n] for n in _ACTUATED_JOINT_NAMES[0:2] + _ACTUATED_JOINT_NAMES[8:10])
    _MODEL_INDICES.joint_neck_ids = tuple(
        name_to_id[n] for n in _ACTUATED_JOINT_NAMES[2:4])

    print("\n[SQuRo Hole] 模型索引解析完成:")
    print(f"  F_body_Link -> {_MODEL_INDICES.f_body_id}, H_body_Link -> {_MODEL_INDICES.h_body_id}")
    print(f"  前段 site ids: {_MODEL_INDICES.f_body_site_ids}")
    print(f"  后段 site ids: {_MODEL_INDICES.h_body_site_ids}")
    print(f"  足端 site ids: {_MODEL_INDICES.foot_site_ids}")
    print(f"  actuator 顺序: {list(joint_names)}")
