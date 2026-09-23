from mjlab.managers.scene_entity_config import SceneEntityCfg



# 执行器关节名称
_ACTUATED_JOINT_NAMES = [
    "F_spine1_joint", "F_body_joint",
    "Neck_yaw_joint", "Neck_pitch_joint",
    "FL_shoulder_joint", "FL_elbow_joint",
    "FR_shoulder_joint", "FR_elbow_joint",
    "H_spine1_joint", "H_body_joint",
    "HL_hip_joint", "HL_knee_joint",
    "HR_hip_joint", "HR_knee_joint",
]



# 观测用配置
ACTUATED_JOINT_CFG = SceneEntityCfg("robot", joint_names=tuple(_ACTUATED_JOINT_NAMES))



# action 张量中的腿/脊柱/颈部列索引
_ACTION_LEG_IDS  = (4, 5, 6, 7, 10, 11, 12, 13)
_ACTION_SPN_IDS  = (0, 1, 8, 9)
_ACTION_NECK_IDS = (2, 3)           # Neck_yaw, Neck_pitch
_ACTION_SPN_LATERAL_ID = 0          # F_spine1
_ACTION_SPN_BODY_IDS = (1, 9)       # F_body, H_body



class ModelIndices:
    __slots__ = (
        "head_body_id", "f_body_id", "h_body_id",
        "foot_site_ids",
        "joint_ids",
        "joint_leg_ids", "joint_spn_ids", "joint_neck_ids",
        "actuator_leg_ids", "actuator_spn_ids", "actuator_neck_ids",
        "actuator_spn_lateral_id", "actuator_spn_body_ids",
    )


    def __init__(self):
        self.head_body_id: int = -1     # 头部 (Neck_pitch_Link)
        self.f_body_id: int = -1
        self.h_body_id: int = -1
        self.foot_site_ids: tuple[int, ...] = ()
        self.joint_ids: tuple[int, ...] = ()
        self.joint_leg_ids: tuple[int, ...] = ()
        self.joint_spn_ids: tuple[int, ...] = ()
        self.joint_neck_ids: tuple[int, ...] = ()
        self.actuator_leg_ids: tuple[int, ...] = _ACTION_LEG_IDS
        self.actuator_spn_ids: tuple[int, ...] = _ACTION_SPN_IDS
        self.actuator_neck_ids: tuple[int, ...] = _ACTION_NECK_IDS
        self.actuator_spn_lateral_id: int = _ACTION_SPN_LATERAL_ID
        self.actuator_spn_body_ids: tuple[int, ...] = _ACTION_SPN_BODY_IDS



_MODEL_INDICES = ModelIndices()



def resolve_model_indices(entity) -> None:
    if _MODEL_INDICES.f_body_id >= 0:
        return

    body_ids, body_names = entity.find_bodies(["F_body_Link", "H_body_Link", "Neck_pitch_Link"], preserve_order=True)
    _MODEL_INDICES.f_body_id = body_ids[0]
    _MODEL_INDICES.h_body_id = body_ids[1]
    _MODEL_INDICES.head_body_id = body_ids[2]

    site_ids, _ = entity.find_sites(["FL_elbow_site", "FR_elbow_site", "HL_knee_site", "HR_knee_site"])
    _MODEL_INDICES.foot_site_ids = tuple(site_ids)

    joint_ids, joint_names = entity.find_joints(_ACTUATED_JOINT_NAMES, preserve_order=True)
    _MODEL_INDICES.joint_ids = tuple(joint_ids)
    # 腿: 位置 4-7, 10-13; 脊柱: 0-1, 8-9; 颈: 2-3
    _MODEL_INDICES.joint_leg_ids  = tuple(joint_ids[4:8]) + tuple(joint_ids[10:14])
    _MODEL_INDICES.joint_spn_ids  = tuple(joint_ids[0:2]) + tuple(joint_ids[8:10])
    _MODEL_INDICES.joint_neck_ids = tuple(joint_ids[2:4])

    print("\n[SQuRo] 模型索引解析完成:")
    print(f"  F_body_Link: {body_names[0]} -> ID {body_ids[0]}")
    print(f"  H_body_Link: {body_names[1]} -> ID {body_ids[1]}")
    print(f"  Neck_pitch_Link (头部): {body_names[2]} -> ID {body_ids[2]}")
    print(f"  足端 site IDs: {_MODEL_INDICES.foot_site_ids}")
    print(f"  actuator 顺序: {list(joint_names)}")
    print(f"  leg action idx: {_ACTION_LEG_IDS}")
    print(f"  spn action idx: {_ACTION_SPN_IDS}")
    print(f"  neck action idx: {_ACTION_NECK_IDS}")
