import mujoco
from pathlib import Path
from mjlab import MJLAB_SRC_PATH
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


# XML文件路径
SENGIBOT_XML: Path = (MJLAB_SRC_PATH / "asset_zoo" / "robots" / "Sengibot" / "xmls" / "Sengibot.xml")
assert SENGIBOT_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    assets_path = SENGIBOT_XML.parent / "assets"
    if assets_path.exists():
        for f in assets_path.iterdir():
            if f.is_file():
                assets[f"{meshdir}/{f.name}" if meshdir else f.name] = f.read_bytes()
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(SENGIBOT_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


# 执行器配置
# 注意：XML 中只有这 7 个 <position>，其余关节由 18 个 connect 等式约束驱动
# <position> 会被 XmlActuator 识别为 command_field='position',
# 动作即目标关节角(rad), 任务侧用 JointPositionActionCfg 的 scale/offset 映射
SENGIBOT_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        XmlActuatorCfg(
            target_names_expr=(
                "joint_waist_drive",
                "joint_LF_drive",
                "joint_RF_drive",
                "joint_LH_drive",
                "joint_LH_tibia",
                "joint_RH_drive",
                "joint_RH_tibia",
            )
        ),
    ),
)


# 初始状态配置
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.06),
    rot=(1.0, 0.0, 0.0, 0.0),
    lin_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
    joint_pos={
        # 7个执行器对应的关节
        "joint_waist_drive": 0.0,
        "joint_LF_drive": 0.0,
        "joint_RF_drive": 0.0,
        "joint_LH_drive": 0.0,
        "joint_LH_tibia": 0.0,
        "joint_RH_drive": 0.0,
        "joint_RH_tibia": 0.0,

        # 其他关节设为0
        ".*": 0.0
    },
    joint_vel={".*": 0.0}
) 


# 机器人配置
def get_sengibot_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    spec_fn=get_spec,
    articulation=SENGIBOT_ARTICULATION,
  )


# mujoco可视化界面
if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_sengibot_robot_cfg())
    viewer.launch(robot.spec.compile())
