import mujoco
from pathlib import Path
from mjlab import MJLAB_SRC_PATH
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


# SQuRo 跌倒爬起专用模型 XML（基于参考仿真 Mouse_Pos_Backup.xml 适配）
# 与 SQuRo.xml 差异:
#   - base_Link 初始姿态为 identity（仰面跌倒, 现代 MuJoCo 不接受 quat=0 0 0 0）
#   - F_spine1/H_spine1 range ±0.65, Neck_yaw ±0.3, Neck_pitch -0.3~0.5
#   - 身体 F/H_body conaffinity=1（可与地板碰撞, 爬起借力）
#   - actuator 顺序与 MJLAB 统一: 脊柱+颈在前, 腿在后
SQURO_BACKUP_XML: Path = (
    MJLAB_SRC_PATH
    / "asset_zoo"
    / "robots"
    / "SQuRo_Backup"
    / "xmls"
    / "SQuRo_Backup.xml"
)
assert SQURO_BACKUP_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    assets_path = SQURO_BACKUP_XML.parent / "assets"
    if assets_path.exists():
        for f in assets_path.iterdir():
            if f.is_file():
                assets[f"{meshdir}/{f.name}" if meshdir else f.name] = f.read_bytes()
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(SQURO_BACKUP_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


# 执行器配置 — 顺序与 XML actuator 顺序一致（脊柱+颈在前, 腿在后）
SQURO_BACKUP_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        XmlActuatorCfg(
            target_names_expr=(
                "F_spine1_joint", "F_body_joint",
                "Neck_yaw_joint", "Neck_pitch_joint",
                "FL_shoulder_joint", "FL_elbow_joint",
                "FR_shoulder_joint", "FR_elbow_joint",
                "H_spine1_joint", "H_body_joint",
                "HL_hip_joint", "HL_knee_joint",
                "HR_hip_joint", "HR_knee_joint",
            )
        ),
    ),
)


# 初始状态: 仰面跌倒 (identity) + 站立初始关节角（与参考脚本 Loco_Backup 一致）
# 闭链关节保持 0（与参考脚本 set_initial_position 行为一致）
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.06),
    rot=(1.0, 0.0, 0.0, 0.0),
    lin_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
    joint_pos={
        "F_spine1_joint": 0.0,
        "F_body_joint": 0.0,
        "Neck_yaw_joint": 0.0,
        "Neck_pitch_joint": 0.0,
        "FL_shoulder_joint": 0.1,
        "FL_elbow_joint": -0.3,
        "FR_shoulder_joint": 0.1,
        "FR_elbow_joint": -0.3,
        "H_spine1_joint": 0.0,
        "H_body_joint": 0.0,
        "HL_hip_joint": -0.1,
        "HL_knee_joint": 0.3,
        "HR_hip_joint": -0.1,
        "HR_knee_joint": 0.3,
        ".*": 0.0,
    },
    joint_vel={".*": 0.0},
)


# 机器人配置
def get_squro_backup_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=INIT_STATE,
        spec_fn=get_spec,
        articulation=SQURO_BACKUP_ARTICULATION,
    )


# mujoco 可视化界面
if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_squro_backup_robot_cfg())
    viewer.launch(robot.spec.compile())
