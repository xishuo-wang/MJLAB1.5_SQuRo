import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from mjlab.scripts._diagnostics import HOLE_DIAGNOSTICS_ROOT
from mjlab.asset_zoo.robots.SQuRo.SQuRo_constants import get_squro_robot_cfg
from mjlab.entity import Entity


ROOT = Path(__file__).resolve().parents[4]
OUT = HOLE_DIAGNOSTICS_ROOT / "head_collision"
XML = ROOT / "src/mjlab/asset_zoo/robots/SQuRo/xmls/SQuRo.xml"


# 通过刚体名称获取该刚体唯一的碰撞几何。
def body_geom(model, name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert body_id >= 0 and model.body_geomnum[body_id] == 1
    return int(model.body_geomadr[body_id])


# 统计指定几何对的实际接触。
def contacts(data, first, second):
    return sum(set(contact.geom) == {first, second} for contact in data.contact)


# 获取头部网格的世界坐标高度范围。
def head_bounds(model, data, geom):
    mesh = int(model.geom_dataid[geom])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    vertices = model.mesh_vert[start:start + count]
    return vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


# 对比修改前后实际重叠接触，并核对实体装配后的排除条目。
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    text = XML.read_text(encoding="utf-8")
    exclusion = '<exclude body1="Neck_pitch_Link" body2="F_body_Link"/>'
    assert text.count(exclusion) == 1
    old = mujoco.MjModel.from_xml_string(text.replace(exclusion, ""))
    new = mujoco.MjModel.from_xml_string(text)
    before, after = mujoco.MjData(old), mujoco.MjData(new)
    for field in ("body_mass", "body_ipos", "body_iquat", "body_inertia", "geom_type", "geom_dataid",
                  "geom_contype", "geom_conaffinity", "actuator_forcerange", "actuator_ctrlrange"):
        assert np.array_equal(getattr(old, field), getattr(new, field)), field
    assert (old.nq, old.nv, old.nu, old.ngeom) == (new.nq, new.nv, new.nu, new.ngeom)
    head = body_geom(new, "Neck_pitch_Link")
    front = body_geom(new, "F_body_Link")
    yaw = int(new.jnt_qposadr[mujoco.mj_name2id(new, mujoco.mjtObj.mjOBJ_JOINT, "Neck_yaw_joint")])
    pitch = int(new.jnt_qposadr[mujoco.mj_name2id(new, mujoco.mjtObj.mjOBJ_JOINT, "Neck_pitch_joint")])
    found = None
    for yaw_angle in np.linspace(-1.5, 1.5, 31):
        for pitch_angle in np.linspace(-1.5, 1.5, 31):
            before.qpos[:] = old.qpos0
            before.qpos[[yaw, pitch]] = [yaw_angle, pitch_angle]
            mujoco.mj_forward(old, before)
            count = contacts(before, head, front)
            if count:
                after.qpos[:] = before.qpos
                mujoco.mj_forward(new, after)
                assert contacts(after, head, front) == 0
                found = {"yaw": float(yaw_angle), "pitch": float(pitch_angle), "before": count, "after": 0}
                break
        if found:
            break
    assert found is not None, "未找到排除前发生头部自碰撞的验证姿态"

    after.qpos[:] = new.qpos0
    mujoco.mj_forward(new, after)
    vertices = head_bounds(new, after, head)
    free_joint = int(np.flatnonzero(new.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
    root_z = int(new.jnt_qposadr[free_joint]) + 2
    after.qpos[root_z] -= float(vertices[:, 2].min()) + .001
    mujoco.mj_forward(new, after)
    floor = mujoco.mj_name2id(new, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    ground_count = contacts(after, head, floor)
    assert ground_count > 0, "头部与地面的接触被意外禁用"

    after.qpos[:] = new.qpos0
    mujoco.mj_forward(new, after)
    top = float(head_bounds(new, after, head)[:, 2].max())
    tree = ET.fromstring(text)
    ET.SubElement(tree.find("worldbody"), "geom", name="audit_obstacle", type="box",
                  size="0.5 0.5 0.005", pos=f"0 0 {top + .004}", contype="1", conaffinity="1")
    board_model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    board_data = mujoco.MjData(board_model)
    mujoco.mj_forward(board_model, board_data)
    board = mujoco.mj_name2id(board_model, mujoco.mjtObj.mjOBJ_GEOM, "audit_obstacle")
    board_count = contacts(board_data, body_geom(board_model, "Neck_pitch_Link"), board)
    assert board_count > 0, "头部与障碍板的接触被意外禁用"

    assembled = Entity(get_squro_robot_cfg()).spec.compile()
    head_body = mujoco.mj_name2id(assembled, mujoco.mjtObj.mjOBJ_BODY, "Neck_pitch_Link")
    front_body = mujoco.mj_name2id(assembled, mujoco.mjtObj.mjOBJ_BODY, "F_body_Link")
    first, second = sorted((head_body, front_body))
    assert (first << 16) + second in assembled.exclude_signature
    result = {"头部与前躯干实际接触": found, "头部与地面接触数": ground_count,
              "头部与障碍板接触数": board_count, "MJLAB实体装配保留排除关系": True,
              "质量惯量与执行器参数": "修改前后完全一致", "动作维度": new.nu}
    (OUT / "排除碰撞验证.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
