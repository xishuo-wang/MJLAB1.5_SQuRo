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


# 交互式 F_body 朝向角测试
#   uv run python src/mjlab/asset_zoo/robots/SQuRo/SQuRo_constants.py
#   拖拽滑块改变脊柱侧摆角，实时观察朝向差异与转弯方向
if __name__ == "__main__":
    import math
    import mujoco
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    def _heading(quat_xyzw):
        """从四元数 [w,x,y,z] 提取偏航角 (deg), 0°=+X, +90°=+Y
        R[0:2,0] = body X轴在world XY平面的投影方向"""
        w, x, y, z = quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]
        sin_h = 2.0 * (w * z + x * y)       # R[1,0]
        cos_h = 1.0 - 2.0 * (y * y + z * z)  # R[0,0]
        return math.degrees(math.atan2(sin_h, cos_h))

    robot = Entity(get_squro_robot_cfg())
    model = robot.spec.compile()
    data = mujoco.MjData(model)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")
    f_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "F_body_Link")
    h_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "H_body_Link")
    f_spine1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "F_spine1_joint")

    with (viewer.launch_passive(model, data) as v):
        print("\n=== SQuRo 朝向角 & 转弯方向测试 ===")
        print("公式: F_body_heading = base_heading - F_spine1")
        print("  正F_spine1 → F_body右偏 → 机器人右转")
        print("  负F_spine1 → F_body左偏 → 机器人左转")
        print("  拖拽右侧 F_spine1 滑块验证")
        print("=" * 70)

        last_print = 0.0
        while v.is_running():
            mujoco.mj_step(model, data)

            if data.time - last_print >= 0.1:
                last_print = data.time

                base_h = _heading(data.xquat[base_id])
                f_body_h = _heading(data.xquat[f_body_id])
                h_body_h = _heading(data.xquat[h_body_id])
                f_spine1 = math.degrees(data.qpos[model.jnt_qposadr[f_spine1_id]])
                delta = f_body_h - base_h

                if abs(f_spine1) < 0.5:
                    turn = "直行"
                elif f_spine1 > 0:
                    turn = "右转→"
                else:
                    turn = "←左转"

                print(f"\rt={data.time:5.2f}s | "
                      f"F_spine1={f_spine1:+6.1f}° | "
                      f"base={base_h:+7.1f}° | "
                      f"F_body={f_body_h:+7.1f}° | "
                      f"H_body={h_body_h:+7.1f}° | "
                      f"Δ={delta:+6.1f}° | "
                      f"{turn}",
                      end="", flush=True)

            v.sync()

    print()  # 换行
