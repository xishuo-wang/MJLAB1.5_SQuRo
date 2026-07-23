"""SQuRo 模型索引 — 统一管理所有关节/身体/足端 ID

顺序遵循 entity actuator 顺序（= MuJoCo 关节树深度优先遍历）:
  F_spine1, F_body, FL_shoulder, FL_elbow, FR_shoulder, FR_elbow,
  H_spine1, H_body, HL_hip, HL_knee, HR_hip, HR_knee

与 entity.find_joints_by_actuator_names 返回的顺序一致。
_action_leg_ids / _action_spn_ids 用于索引 policy 输出的 action 张量。
"""

from mjlab.managers.scene_entity_config import SceneEntityCfg

# 执行器关节名称 — 必须与 entity actuator 顺序一致
_ACTUATED_JOINT_NAMES = [
    "F_spine1_joint", "F_body_joint",
    "FL_shoulder_joint", "FL_elbow_joint",
    "FR_shoulder_joint", "FR_elbow_joint",
    "H_spine1_joint", "H_body_joint",
    "HL_hip_joint", "HL_knee_joint",
    "HR_hip_joint", "HR_knee_joint",
]

# 观测用配置
ACTUATED_JOINT_CFG = SceneEntityCfg("robot", joint_names=tuple(_ACTUATED_JOINT_NAMES))

# action 张量中的腿/脊柱列索引（与 entity actuator 顺序一致）
_ACTION_LEG_IDS = (2, 3, 4, 5, 8, 9, 10, 11)
_ACTION_SPN_IDS = (0, 1, 6, 7)
_ACTION_SPN_LATERAL_ID = 0   # F_spine1
_ACTION_SPN_BODY_IDS = (1, 7)  # F_body, H_body


class ModelIndices:
    __slots__ = (
        "f_body_id", "h_body_id",
        "foot_site_ids",
        "joint_ids",             # entity 级别关节ID（按 _ACTUATED_JOINT_NAMES 序）
        "joint_leg_ids",         # 腿关节 entity ID
        "joint_spn_ids",         # 脊柱关节 entity ID
        "actuator_leg_ids",      # action 张量腿列索引
        "actuator_spn_ids",      # action 张量脊柱列索引
        "actuator_spn_lateral_id",  # F_spine1 action 列索引
        "actuator_spn_body_ids",    # F_body+H_body action 列索引
    )

    def __init__(self):
        self.f_body_id: int = -1
        self.h_body_id: int = -1
        self.foot_site_ids: tuple[int, ...] = ()
        self.joint_ids: tuple[int, ...] = ()
        self.joint_leg_ids: tuple[int, ...] = ()
        self.joint_spn_ids: tuple[int, ...] = ()
        self.actuator_leg_ids: tuple[int, ...] = _ACTION_LEG_IDS
        self.actuator_spn_ids: tuple[int, ...] = _ACTION_SPN_IDS
        self.actuator_spn_lateral_id: int = _ACTION_SPN_LATERAL_ID
        self.actuator_spn_body_ids: tuple[int, ...] = _ACTION_SPN_BODY_IDS


_MODEL_INDICES = ModelIndices()


def resolve_model_indices(entity) -> None:
    if _MODEL_INDICES.f_body_id >= 0:
        return

    body_ids, body_names = entity.find_bodies(
        ["F_body_Link", "H_body_Link"], preserve_order=True
    )
    _MODEL_INDICES.f_body_id = body_ids[0]
    _MODEL_INDICES.h_body_id = body_ids[1]

    site_ids, _ = entity.find_sites(
        ["FL_elbow_site", "FR_elbow_site", "HL_knee_site", "HR_knee_site"]
    )
    _MODEL_INDICES.foot_site_ids = tuple(site_ids)

    joint_ids, joint_names = entity.find_joints(
        _ACTUATED_JOINT_NAMES, preserve_order=True
    )
    _MODEL_INDICES.joint_ids = tuple(joint_ids)
    # 腿关节在 _ACTUATED_JOINT_NAMES 中的位置：2-5, 8-11
    _MODEL_INDICES.joint_leg_ids = tuple(joint_ids[2:6]) + tuple(joint_ids[8:12])
    _MODEL_INDICES.joint_spn_ids = tuple(joint_ids[0:2]) + tuple(joint_ids[6:8])

    print("\n[SQuRo] 模型索引解析完成:")
    print(f"  F_body_Link: {body_names[0]} -> ID {body_ids[0]}")
    print(f"  H_body_Link: {body_names[1]} -> ID {body_ids[1]}")
    print(f"  足端 site IDs: {_MODEL_INDICES.foot_site_ids}")
    print(f"  actuator 顺序: {list(joint_names)}")
    print(f"  leg action idx: {_ACTION_LEG_IDS}")
    print(f"  spn action idx: {_ACTION_SPN_IDS}")
