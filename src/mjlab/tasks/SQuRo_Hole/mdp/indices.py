from __future__ import annotations
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

# 执行器顺序下的关节索引 (执行器序: 脊柱2 + 头颈2 + 前腿4 + 后腿4)
ACTUATOR_LEG_IDS = (
    (4, 5, 6, 7, 10, 11, 12, 13)
)
REF_SPINE_IDS = (0, 1, 8, 9)      # F_spine1, F_body, H_spine1, H_body
REF_NECK_IDS = (2, 3)             # Neck_yaw, Neck_pitch (参考期望恒 0)
REF_FRONT_IDS = (4, 5, 6, 7)      # FL/FR shoulder, elbow
REF_HIND_IDS = (10, 11, 12, 13)   # HL/HR hip, knee

# 参考表 14 列的列序: 前腿 4 + 后腿 4 + 脊柱 4 + 头颈 2
REF_TABLE_ORDER = REF_FRONT_IDS + REF_HIND_IDS + REF_SPINE_IDS + REF_NECK_IDS

# 模型 joint_names 里的头颈索引与初值姿态
NECK_JOINT_NAMES = ("Neck_yaw_joint", "Neck_pitch_joint")
NECK_INIT_POS = (0.0, -0.3)

# 虚拟碰撞采样点: 前段 F_body_1..9 + 后段 H_body_1..9 (世界系上表面)
# 这 18 个点按 body 局部 z 分布在上/下表面, 世界系下 +0.008 的表面朝上, 用于
# 判断"body 上表面是否高于板底"; 见 docs/SQuRo_Hole_技术细节.md §5
FRONT_SEG_SITE_NAMES = tuple(f"F_body_{i}_site" for i in range(1, 10))
REAR_SEG_SITE_NAMES = tuple(f"H_body_{i}_site" for i in range(1, 10))

# 足端 site 名称 (抬脚高度奖励用)
FOOT_SITE_NAMES = ("FL_elbow_site", "FR_elbow_site", "HL_knee_site", "HR_knee_site")

# 观测用配置
ACTUATED_JOINT_CFG = SceneEntityCfg("robot", joint_names=tuple(_ACTUATED_JOINT_NAMES))


class ModelIndices:
    __slots__ = (
        "f_body_id", "h_body_id",
        "front_seg_site_ids", "rear_seg_site_ids", "foot_site_ids",
        "joint_ids", "joint_leg_ids", "joint_spn_ids", "joint_neck_ids",
    )

    def __init__(self):
        self.f_body_id: int = -1
        self.h_body_id: int = -1
        self.front_seg_site_ids: tuple[int, ...] = ()
        self.rear_seg_site_ids: tuple[int, ...] = ()
        self.foot_site_ids: tuple[int, ...] = ()
        self.joint_ids: tuple[int, ...] = ()
        self.joint_leg_ids: tuple[int, ...] = ()
        self.joint_spn_ids: tuple[int, ...] = ()
        self.joint_neck_ids: tuple[int, ...] = ()


_MODEL_INDICES = ModelIndices()


# 解析 body / site / 关节索引。
# 判据必须同时看 body 与 joint: 旧写法只查 f_body_id, 一旦之前被部分解析过
# (例如上一轮解析在 find_joints 之前中断), 再调用会直接 return,
# joint_ids 永远留空 —— 观测项 actuator_pos/actuator_vel 会退化成 0 维。
def resolve_model_indices(entity) -> None:
    if _MODEL_INDICES.f_body_id >= 0 and len(_MODEL_INDICES.joint_ids) > 0:
        return

    body_ids, _ = entity.find_bodies(["F_body_Link", "H_body_Link"], preserve_order=True)
    _MODEL_INDICES.f_body_id = body_ids[0]
    _MODEL_INDICES.h_body_id = body_ids[1]

    front_ids, _ = entity.find_sites(list(FRONT_SEG_SITE_NAMES), preserve_order=True)
    rear_ids, _ = entity.find_sites(list(REAR_SEG_SITE_NAMES), preserve_order=True)
    foot_ids, _ = entity.find_sites(list(FOOT_SITE_NAMES), preserve_order=True)
    _MODEL_INDICES.front_seg_site_ids = tuple(front_ids)
    _MODEL_INDICES.rear_seg_site_ids = tuple(rear_ids)
    _MODEL_INDICES.foot_site_ids = tuple(foot_ids)

    joint_ids, joint_names = entity.find_joints(_ACTUATED_JOINT_NAMES, preserve_order=True)
    # 索引表按执行器序写死, 少一个就会整表错位/0 维; 这里直接拦住
    assert len(joint_ids) == len(_ACTUATED_JOINT_NAMES), (
        f"只解析到 {len(joint_ids)}/{len(_ACTUATED_JOINT_NAMES)} 个被控关节: "
        f"{list(joint_names)}")
    _MODEL_INDICES.joint_ids = tuple(joint_ids)
    # 按执行器名分组: 前腿 4-7 / 后腿 10-13 / 脊柱 0-1,8-9 / 头颈 2-3
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
    print(f"  前段 site ids: {_MODEL_INDICES.front_seg_site_ids}")
    print(f"  后段 site ids: {_MODEL_INDICES.rear_seg_site_ids}")
    print(f"  足端 site ids: {_MODEL_INDICES.foot_site_ids}")
    print(f"  actuator 顺序: {list(joint_names)}")
