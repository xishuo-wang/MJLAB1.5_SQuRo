from mjlab.managers.scene_entity_config import SceneEntityCfg


# 执行器关节名称
_ACTUATED_JOINT_NAMES = [
    "FL_shoulder_joint", "FL_elbow_joint",
    "FR_shoulder_joint", "FR_elbow_joint",
    "HL_hip_joint", "HL_knee_joint",
    "HR_hip_joint", "HR_knee_joint",
    "F_spine1_joint", "F_body_joint",
    "H_spine1_joint", "H_body_joint",
]


# 观测用配置 — 仅取执行器关节，滤除被动闭链关节
ACTUATED_JOINT_CFG = SceneEntityCfg("robot", joint_names=tuple(_ACTUATED_JOINT_NAMES))


class ModelIndices:
    __slots__ = (
        "f_body_id", "h_body_id",
        "foot_site_ids",
        "joint_ids",                # entity 级别的关节ID（按 _ACTUATED_JOINT_NAMES 顺序）
        "joint_leg_ids",            # 腿关节 entity ID（前8个）
        "joint_spn_ids",            # 脊柱关节 entity ID（后4个）
        "actuator_leg_ids",         # 执行器张量中腿的列索引 (0-7)
        "actuator_spn_ids",         # 执行器张量中脊柱的列索引 (8-11)
        "actuator_spn_lateral_id",  # F_spine1 侧摆在执行器张量中的列索引 (8)
        "actuator_spn_body_ids",    # F_body(9)+H_body(11) 扭转在执行器张量中的列索引
    )

    def __init__(self):
        self.f_body_id: int = -1
        self.h_body_id: int = -1
        self.foot_site_ids: tuple[int, ...] = ()
        self.joint_ids: tuple[int, ...] = ()
        self.joint_leg_ids: tuple[int, ...] = ()
        self.joint_spn_ids: tuple[int, ...] = ()
        self.actuator_leg_ids: tuple[int, ...] = ()
        self.actuator_spn_ids: tuple[int, ...] = ()
        self.actuator_spn_lateral_id: int = 8
        self.actuator_spn_body_ids: tuple[int, ...] = (9, 11)


# 全局单例
_MODEL_INDICES = ModelIndices()


# 懒加载解析 body/site/joint 索引
def resolve_model_indices(entity) -> None:
    if _MODEL_INDICES.f_body_id >= 0:
        return

    # body 索引
    body_ids, body_names = entity.find_bodies(["F_body_Link", "H_body_Link"], preserve_order=True)
    _MODEL_INDICES.f_body_id = body_ids[0]
    _MODEL_INDICES.h_body_id = body_ids[1]

    # site 索引（四足足端）
    site_ids, _ = entity.find_sites(["FL_elbow_site", "FR_elbow_site", "HL_knee_site", "HR_knee_site"])
    _MODEL_INDICES.foot_site_ids = tuple(site_ids)

    # joint entity 级别索引（按 _ACTUATED_JOINT_NAMES 顺序）
    joint_ids, joint_names = entity.find_joints(_ACTUATED_JOINT_NAMES, preserve_order=True)
    _MODEL_INDICES.joint_ids = tuple(joint_ids)
    _MODEL_INDICES.joint_leg_ids = tuple(joint_ids[:8])
    _MODEL_INDICES.joint_spn_ids = tuple(joint_ids[8:])

    # 执行器张量列索引（约定：前8=腿，后4=脊柱）
    _MODEL_INDICES.actuator_leg_ids = tuple(range(8))
    _MODEL_INDICES.actuator_spn_ids = tuple(range(8, 12))
    _MODEL_INDICES.actuator_spn_lateral_id = 8   # F_spine1
    _MODEL_INDICES.actuator_spn_body_ids = (9, 11)  # F_body, H_body

    # 打印解析结果
    print("\n[SQuRo] 模型索引解析完成:")
    print(f"  F_body_Link: {body_names[0]} -> ID {body_ids[0]}")
    print(f"  H_body_Link: {body_names[1]} -> ID {body_ids[1]}")
    print(f"  足端 site IDs: {_MODEL_INDICES.foot_site_ids}")
    print(f"  执行器 joint entity IDs: {_MODEL_INDICES.joint_ids}")
    print(f"  腿 joint IDs: {_MODEL_INDICES.joint_leg_ids}")
    print(f"  脊柱 joint IDs: {_MODEL_INDICES.joint_spn_ids}")
