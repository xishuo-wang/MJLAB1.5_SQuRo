import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares

from mjlab.asset_zoo.robots.SQuRo.SQuRo_constants import (
    INIT_STATE, SQuRO_XML, SPINE_DEFORMATION_BODY_NAMES, SPINE_FRONT_FRAME, SPINE_HIND_FRAME, get_spec,
)
from mjlab.tasks.SQuRo_Hole.mdp.indices import (
    _ACT_JOINT_NAMES, _ACT_FL_LEG_IDS, _ACT_FR_LEG_IDS, _ACT_HL_LEG_IDS,
    _ACT_HR_LEG_IDS, FOOT_SITE_NAMES, FOOT_GEOM_NAMES,
)
from mjlab.tasks.SQuRo_Hole.mdp.reference import inverse_kinematics
from mjlab.scripts._diagnostics import HOLE_DIAGNOSTICS_ROOT

LEG_GROUPS = (_ACT_FL_LEG_IDS, _ACT_FR_LEG_IDS, _ACT_HL_LEG_IDS, _ACT_HR_LEG_IDS)


# 解析实际XML，所有关节、site和碰撞几何均按名称定位。
class Probe:
    # 固定根姿态进行闭链几何检查，动力学阶段恢复全部真实接触。
    def __init__(self):
        self.model = get_spec().compile()
        self.model.opt.timestep = 0.002
        self.model.opt.iterations = 100
        self.model.opt.ls_iterations = 50
        self.data = mujoco.MjData(self.model)
        self.act_joints = np.array([self.model.joint(name).id for name in _ACT_JOINT_NAMES])
        self.act_qpos = self.model.jnt_qposadr[self.act_joints]
        self.passive = np.array([index for index in range(self.model.njnt)
                                 if self.model.jnt_type[index] == mujoco.mjtJoint.mjJNT_HINGE
                                 and index not in self.act_joints])
        self.passive_qpos = self.model.jnt_qposadr[self.passive]
        self.body_ids = np.array([self.model.body(name).id for name in
                                  (SPINE_DEFORMATION_BODY_NAMES[0], SPINE_DEFORMATION_BODY_NAMES[2])])
        self.site_ids = np.array([self.model.site(name).id for name in FOOT_SITE_NAMES])
        self.geom_ids = np.array([self.model.geom(name).id for name in FOOT_GEOM_NAMES])
        self.ctrl_ids = np.array([next(i for i in range(self.model.nu) if self.model.actuator_trnid[i, 0] == joint)
                                  for joint in self.act_joints])
        self.vertices = []
        for geom in self.geom_ids:
            mesh = self.model.geom_dataid[geom]
            start = self.model.mesh_vertadr[mesh]
            self.vertices.append(self.model.mesh_vert[start:start + self.model.mesh_vertnum[mesh]].astype(float))
        self.eq_first = self.model.eq_obj1id
        self.eq_second = self.model.eq_obj2id
        assert np.all(self.model.eq_type == mujoco.mjtEq.mjEQ_CONNECT)
        self.seed = self.model.qpos0.copy()
        for index in range(self.model.njnt):
            if self.model.jnt_type[index] == mujoco.mjtJoint.mjJNT_HINGE:
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, index)
                self.seed[self.model.jnt_qposadr[index]] = INIT_STATE.joint_pos.get(name, 0.0)
        self.seed[2] = 0.15
        self.seed[self.act_qpos] = 0.0
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    # 根据当前状态计算十二个闭链连接的世界坐标残差。
    def closure(self):
        first = self.data.xpos[self.eq_first] + np.einsum("nij,nj->ni",
            self.data.xmat[self.eq_first].reshape(-1, 3, 3), self.model.eq_data[:, :3])
        second = self.data.xpos[self.eq_second] + np.einsum("nij,nj->ni",
            self.data.xmat[self.eq_second].reshape(-1, 3, 3), self.model.eq_data[:, 3:6])
        return (first - second).ravel()

    # 保留驱动关节给定角度，仅求被动关节的闭链一致状态。
    def close_linkages(self):
        def residual(values):
            self.data.qpos[self.passive_qpos] = values
            mujoco.mj_kinematics(self.model, self.data)
            return self.closure()
        result = least_squares(residual, self.data.qpos[self.passive_qpos],
            bounds=self.model.jnt_range[self.passive].T, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=150)
        residual(result.x)
        if np.max(np.abs(self.closure())) > 2e-6:
            raise AssertionError(f"闭链几何误差超限：{np.max(np.abs(self.closure()))}")

    # 从真实足部网格最低点和site分别读取几何高度。
    def measure(self):
        mujoco.mj_kinematics(self.model, self.data)
        body_z = self.data.xpos[self.body_ids, 2]
        bottoms = np.array([(vertices @ self.data.geom_xmat[geom].reshape(3, 3).T
                             + self.data.geom_xpos[geom])[:, 2].min()
                            for vertices, geom in zip(self.vertices, self.geom_ids)])
        centers = self.data.xpos[self.body_ids[np.arange(4) // 2]]
        feet = self.data.site_xpos[self.site_ids]
        rotation = self.data.xmat[self.body_ids].reshape(-1, 3, 3) @ np.array((SPINE_FRONT_FRAME, SPINE_HIND_FRAME))
        return {"center_world_z_mm": (body_z * 1000).tolist(),
                "foot_bottom_world_z_mm": (bottoms * 1000).tolist(),
                "center_above_foot_mesh_mm": ((centers[:, 2] - bottoms) * 1000).tolist(),
                "center_above_foot_site_mm": ((centers[:, 2] - feet[:, 2]) * 1000).tolist(),
                "foot_site_forward_offset_mm": ((feet[:, 0] - centers[:, 0]) * 1000).tolist(),
                "physical_pitch_deg": np.rad2deg(np.arctan2(-rotation[:, 2, 0], np.hypot(rotation[:, 0, 0], rotation[:, 1, 0]))).tolist(),
                "closure_max_mm": float(np.max(np.abs(self.closure())) * 1000)}

    # 求原简化IK并用真实闭链模型测量对应几何。
    def set_ik(self, height):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.seed
        for ids, front in zip(LEG_GROUPS, (True, True, False, False)):
            angles = torch.stack(inverse_kinematics(torch.tensor(0., dtype=torch.float64),
                torch.tensor(-height, dtype=torch.float64), front)).numpy()
            self.data.qpos[self.act_qpos[list(ids)]] = angles
        self.close_linkages()
        mujoco.mj_forward(self.model, self.data)
        equality_rows = self.data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        assert np.max(np.abs(self.data.efc_pos[equality_rows])) < 2e-6
        return self.data.qpos.copy()

    # 联合检索四个对称驱动角和被动角，使真实足部落地点满足高度与前后位置。
    def search(self, height, fix_x=False):
        actuator_slots = np.array([*LEG_GROUPS[0], *LEG_GROUPS[2]])
        variable_joints = np.r_[self.passive, self.act_joints[actuator_slots]]
        variable_qpos = self.model.jnt_qposadr[variable_joints]
        bounds = self.model.jnt_range[variable_joints].T.copy()
        for offset, slot in enumerate(actuator_slots):
            bounds[:, len(self.passive) + offset] = np.maximum(bounds[0, len(self.passive) + offset], self.model.actuator_ctrlrange[self.ctrl_ids[slot], 0]), np.minimum(bounds[1, len(self.passive) + offset], self.model.actuator_ctrlrange[self.ctrl_ids[slot], 1])
        initial = self.data.qpos[variable_qpos].copy()
        def residual(values):
            self.data.qpos[variable_qpos] = values
            self.data.qpos[self.act_qpos[list(LEG_GROUPS[1])]] = values[-4:-2]
            self.data.qpos[self.act_qpos[list(LEG_GROUPS[3])]] = values[-2:]
            values_measured = self.measure()
            targets = np.repeat(np.atleast_1d(height), 2) if np.ndim(height) else height
            height_error = np.array(values_measured["center_above_foot_mesh_mm"]) / 1000 - targets
            x_error = np.array(values_measured["foot_site_forward_offset_mm"]) / 1000
            if fix_x:
                return np.r_[self.closure() * 20, height_error, x_error]
            return np.r_[self.closure() * 20, height_error, (values[-4:] - initial[-4:]) * 1e-5]
        result = least_squares(residual, initial, bounds=bounds, xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=200)
        residual(result.x)
        return self.data.qpos.copy(), {"cost": float(result.cost), "evaluations": int(result.nfev)}

    # 释放根部并保留实际PD、重力和全部原始碰撞，统计末半秒稳态。
    def settle(self, qpos, height, control_qpos=None):
        self.model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        values = self.measure()
        self.data.qpos[2] -= min(values["foot_bottom_world_z_mm"]) / 1000 - 0.001
        control_qpos = qpos if control_qpos is None else control_qpos
        self.data.ctrl[self.ctrl_ids] = control_qpos[self.act_qpos]
        rows = []
        for step in range(2000):
            mujoco.mj_step(self.model, self.data)
            if step >= 1750 and step % 5 == 0:
                rows.append(self.measure())
        errors = self.data.qpos[self.act_qpos] - control_qpos[self.act_qpos]
        floor_id = self.model.geom("floor").id
        ground_contacts = []
        for contact in self.data.contact:
            if floor_id in contact.geom:
                other = int(contact.geom[1] if contact.geom[0] == floor_id else contact.geom[0])
                ground_contacts.append({"body": self.model.body(int(self.model.geom_bodyid[other])).name,
                    "geom": self.model.geom(other).name, "distance_mm": float(contact.dist * 1000)})
        result = {"mean_center_world_z_mm": np.mean([row["center_world_z_mm"] for row in rows], axis=0).tolist(),
                  "min_max_center_world_z_mm": np.stack((np.min([row["center_world_z_mm"] for row in rows], axis=0),
                      np.max([row["center_world_z_mm"] for row in rows], axis=0))).tolist(),
                  "actuator_max_tracking_error_rad": float(np.abs(errors).max()),
                  "actuator_errors_rad": dict(zip(_ACT_JOINT_NAMES, errors.tolist())),
                  "mean_physical_pitch_deg": np.mean([row["physical_pitch_deg"] for row in rows], axis=0).tolist(),
                  "ground_contacts": ground_contacts,
                  "closure_max_mm": max(row["closure_max_mm"] for row in rows)}
        result["height_error_mm"] = (np.array(result["mean_center_world_z_mm"]) - height * 1000).tolist()
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        return result


# 比较简化IK、真实闭链几何与高度优先搜索的落地结果。
def main():
    parser = argparse.ArgumentParser(description="核对同高、零脊柱姿态的实际落地高度")
    parser.add_argument("--heights", type=float, nargs="+", default=[0.025, 0.030, 0.035])
    parser.add_argument("--output-dir", type=Path, default=HOLE_DIAGNOSTICS_ROOT / "ik_accuracy")
    args = parser.parse_args()
    if any(not np.isfinite(height) or not 0.020 <= height <= 0.040 for height in args.heights):
        parser.error("目标高度必须在20～40毫米之间，单位为米")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe = Probe()
    results = {"xml_sha256": hashlib.sha256(SQuRO_XML.read_bytes()).hexdigest(),
               "mesh_geometry_sha256": hashlib.sha256(probe.model.mesh_vert.tobytes() + probe.model.mesh_face.tobytes()).hexdigest(),
               "mujoco_version": mujoco.__version__, "heights": []}
    for height in args.heights:
        qpos = probe.set_ik(height)
        row = {"height_mm": height * 1000, "ik_angles_rad": qpos[probe.act_qpos].tolist(),
               "geometry": probe.measure(), "dynamics": probe.settle(qpos, height)}
        probe.data.qpos[:] = qpos
        searched, info = probe.search(height, fix_x=False)
        row["height_only_search"] = {"angles_rad": searched[probe.act_qpos].tolist(),
            "geometry": probe.measure(), "optimization": info, "dynamics": probe.settle(searched, height)}
        results["heights"].append(row)
        print(f"目标{height * 1000:g}毫米：IK落地{row['dynamics']['mean_center_world_z_mm']}，高度优先搜索{row['height_only_search']['dynamics']['mean_center_world_z_mm']}", flush=True)
    (args.output_dir / "ik_accuracy.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
