import mujoco
from pathlib import Path
from mjlab import MJLAB_SRC_PATH
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


# XML文件路径
SQuRO_XML: Path = (MJLAB_SRC_PATH / "asset_zoo" / "robots" / "SQuRo" / "xmls" / "SQuRo.xml")
assert SQuRO_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    assets_path = SQuRO_XML.parent / "assets"
    if assets_path.exists():
        for f in assets_path.iterdir():
            if f.is_file():
                assets[f"{meshdir}/{f.name}" if meshdir else f.name] = f.read_bytes()
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(SQuRO_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


# 执行器配置
SQURO_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        XmlActuatorCfg(
            target_names_expr=(
                "FL_shoulder_joint", "FL_elbow_joint",
                "FR_shoulder_joint", "FR_elbow_joint", 
                "HL_hip_joint", "HL_knee_joint",
                "HR_hip_joint", "HR_knee_joint",
                "F_spine1_joint", "F_body_joint",
                "H_spine1_joint", "H_body_joint"
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
        # 12个执行器对应的关节
        "FL_shoulder_joint": 0.1,
        "FL_elbow_joint": -0.3,
        "FR_shoulder_joint": 0.1,
        "FR_elbow_joint": -0.3,
        "HL_hip_joint": -0.1,
        "HL_knee_joint": 0.3,
        "HR_hip_joint": -0.1,
        "HR_knee_joint": 0.3,
        "F_spine1_joint": 0.0,
        "F_body_joint": 0.0,
        "H_spine1_joint": 0.0,
        "H_body_joint": 0.0,
        
        # 其他需要设置的关节（闭链机构相关）
        "FL_shoulder1_joint": -0.0943,
        "FL_elbow1_joint": 0.3867,
        "FL_elbow2_joint": 0.0943,
        "FL_elbow3_joint": 0.3862,
        "FR_shoulder1_joint": -0.0943,
        "FR_elbow1_joint": 0.3867,
        "FR_elbow2_joint": 0.0942,
        "FR_elbow3_joint": -0.3862,
        "HL_hip1_joint": 0.0978,
        "HL_knee1_joint": -0.3905,
        "HL_knee2_joint": -0.0978,
        "HL_knee3_joint": -0.3905,
        "HR_hip1_joint": 0.0978,
        "HR_knee1_joint": -0.3905,
        "HR_knee2_joint": -0.0978,
        "HR_knee3_joint": -0.3905,
        
        # 其他关节设为0
        ".*": 0.0
    },
    joint_vel={".*": 0.0}
) 


# 机器人配置
def get_squro_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    spec_fn=get_spec,
    articulation=SQURO_ARTICULATION,
  )


# mujoco可视化界面
if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_squro_robot_cfg())
    viewer.launch(robot.spec.compile())