from mjlab.managers.scene_entity_config import SceneEntityCfg


# 执行器关节名称 — 必须与 entity actuator 顺序一致
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


# 躯干朝向标记: 每段一对 (腹面 site, 背面 site), 世界坐标差即该段背腹轴。
# 见 docs/SQuRo_Backup_技术细节.md §4 —— 不依赖四元数约定, 是姿态判据的唯一可信来源。
SEGMENT_BELLY_BACK_SITES = (
    ("F_body_belly_site", "F_body_back_site"),
    ("H_body_belly_site", "H_body_back_site"),
)


# action 张量中的腿/脊柱/颈部列索引（与 entity actuator 顺序一致）
_ACTION_LEG_IDS  = (4, 5, 6, 7, 10, 11, 12, 13)
_ACTION_SPN_IDS  = (0, 1, 8, 9)
_ACTION_NECK_IDS = (2, 3)           # Neck_yaw, Neck_pitch
_ACTION_SPN_LATERAL_ID = 0          # F_spine1
_ACTION_SPN_BODY_IDS = (1, 9)       # F_body, H_body


# 执行器控制范围 (SQuRo.xml 中 <position ... ctrlrange=...>)，顺序与 _ACTUATED_JOINT_NAMES 一致。
# ctrllimited="true" 时 MuJoCo 会把 ctrl 直接裁到该区间: 目标指令超出部分被完全丢弃,
# 命令再大也不会产生额外力矩。动作幅值惩罚只对"被丢弃的那一段"计成本, 因此依赖这张表。
# 如改 XML 的 ctrlrange, 必须同步改这里; verify_backup_config.py 会与 XML 对拍。
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


class ModelIndices:
    __slots__ = (
        "f_body_id", "h_body_id",
        "foot_site_ids",
        "segment_belly_back_ids",
        "joint_ids",
        "joint_leg_ids", "joint_spn_ids", "joint_neck_ids",
        "actuator_leg_ids", "actuator_spn_ids", "actuator_neck_ids",
        "actuator_spn_lateral_id", "actuator_spn_body_ids",
    )

    def __init__(self):
        self.f_body_id: int = -1
        self.h_body_id: int = -1
        self.foot_site_ids: tuple[int, ...] = ()
        # ((F 腹面, F 背面), (H 腹面, H 背面))
        self.segment_belly_back_ids: tuple[tuple[int, int], tuple[int, int]] | None = None
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

    body_ids, body_names = entity.find_bodies(["F_body_Link", "H_body_Link"], preserve_order=True)
    _MODEL_INDICES.f_body_id = body_ids[0]
    _MODEL_INDICES.h_body_id = body_ids[1]

    site_ids, _ = entity.find_sites(["FL_elbow_site", "FR_elbow_site", "HL_knee_site", "HR_knee_site"])
    _MODEL_INDICES.foot_site_ids = tuple(site_ids)

    belly_back = []
    for belly_name, back_name in SEGMENT_BELLY_BACK_SITES:
        ids, _ = entity.find_sites([belly_name, back_name], preserve_order=True)
        belly_back.append((ids[0], ids[1]))
    _MODEL_INDICES.segment_belly_back_ids = (belly_back[0], belly_back[1])

    joint_ids, joint_names = entity.find_joints(_ACTUATED_JOINT_NAMES, preserve_order=True)
    _MODEL_INDICES.joint_ids = tuple(joint_ids)
    # 腿: 位置 4-7, 10-13; 脊柱: 0-1, 8-9; 颈: 2-3
    _MODEL_INDICES.joint_leg_ids  = tuple(joint_ids[4:8]) + tuple(joint_ids[10:14])
    _MODEL_INDICES.joint_spn_ids  = tuple(joint_ids[0:2]) + tuple(joint_ids[8:10])
    _MODEL_INDICES.joint_neck_ids = tuple(joint_ids[2:4])

    print("\n[SQuRo] 模型索引解析完成:")
    print(f"  F_body_Link: {body_names[0]} -> ID {body_ids[0]}")
    print(f"  H_body_Link: {body_names[1]} -> ID {body_ids[1]}")
    print(f"  足端 site IDs: {_MODEL_INDICES.foot_site_ids}")
    print(f"  躯干腹/背 site IDs: {_MODEL_INDICES.segment_belly_back_ids}")
    print(f"  actuator 顺序: {list(joint_names)}")
    print(f"  leg action idx: {_ACTION_LEG_IDS}")
    print(f"  spn action idx: {_ACTION_SPN_IDS}")
    print(f"  neck action idx: {_ACTION_NECK_IDS}")