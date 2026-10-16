import hashlib
import argparse
import json
from pathlib import Path

import numpy as np

from mjlab.tasks.SQuRo_Hole.mdp.config import NECK_REF_POS
from mjlab.tasks.SQuRo_Hole.mdp.indices import _ACT_NECK_IDS, _ACT_FL_LEG_IDS, _ACT_FR_LEG_IDS, _ACT_HL_LEG_IDS, _ACT_HR_LEG_IDS
from mjlab.scripts.Hole.probe_ik_accuracy import Probe, FOOT_GEOM_NAMES
from mjlab.scripts._diagnostics import HOLE_DIAGNOSTICS_ROOT

# 保持当前颈部参考、脊柱归零，按实际落地高度标定低姿态固定角。
def main():
    parser = argparse.ArgumentParser(description="按真实闭链和落地接触重新标定低姿态，结果仅输出到JSON")
    parser.add_argument("--heights", type=float, nargs="+", default=[0.025, 0.030, 0.035, 0.040])
    parser.add_argument("--output-dir", type=Path, default=HOLE_DIAGNOSTICS_ROOT / "pose_calibration")
    args = parser.parse_args()
    if any(not np.isfinite(height) or not 0.020 < height <= 0.040 for height in args.heights):
        parser.error("标定高度必须大于20毫米且不超过40毫米，20毫米历史角度不重新标定")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe = Probe()
    probe.seed[probe.act_qpos[list(_ACT_NECK_IDS)]] = NECK_REF_POS
    result = {"说明": "脊柱目标为零，颈部保持当前参考；同侧两腿角度相同，使用真实闭链、碰撞与PD落地标定。20毫米角度保持原值。",
              "neck_reference_rad": list(NECK_REF_POS), "search_method": "优先实际高度，角度接近原姿态，不强制足端X归零",
              "mesh_sha256": hashlib.sha256(probe.model.mesh_vert.tobytes()).hexdigest(), "anchors": []}
    for height in args.heights:
        probe.set_ik(height)
        calibrated, _ = probe.search(height, fix_x=False)
        targets = np.full(2, height)
        iterations = []
        for _ in range(5):
            measured = probe.settle(calibrated, height)
            iterations.append({"geometric_targets_mm": (targets * 1000).tolist(), "dynamics": measured})
            error = height - np.array(measured["mean_center_world_z_mm"]) / 1000
            if np.max(np.abs(error)) < 0.00002:
                break
            targets += error
            probe.data.qpos[:] = calibrated
            calibrated, _ = probe.search(targets, fix_x=False)
        calibrated[probe.act_qpos] = np.round(calibrated[probe.act_qpos], 6)
        measured = probe.settle(calibrated, height)
        if max(abs(value) for value in measured["height_error_mm"]) > 0.05:
            raise AssertionError(f"{height}米落地误差超过0.05毫米：{measured['height_error_mm']}")
        support = {contact["geom"] for contact in measured["ground_contacts"]}
        if not support.issubset(set(FOOT_GEOM_NAMES)) or len(support) != 4:
            raise AssertionError(f"标定姿态出现非足部支撑或缺少足部接触：{support}")
        row = {"height": height,
               "front_angles": calibrated[probe.act_qpos[list(_ACT_FL_LEG_IDS)]].tolist(),
               "hind_angles": calibrated[probe.act_qpos[list(_ACT_HL_LEG_IDS)]].tolist(),
               "dynamics": measured, "iterations": iterations}
        assert np.array_equal(calibrated[probe.act_qpos[list(_ACT_FL_LEG_IDS)]], calibrated[probe.act_qpos[list(_ACT_FR_LEG_IDS)]])
        assert np.array_equal(calibrated[probe.act_qpos[list(_ACT_HL_LEG_IDS)]], calibrated[probe.act_qpos[list(_ACT_HR_LEG_IDS)]])
        result["anchors"].append(row)
        (args.output_dir / "calibrated_anchors.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: row[key] for key in ("height", "front_angles", "hind_angles")}
                         | {"actual_heights_mm": measured["mean_center_world_z_mm"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
