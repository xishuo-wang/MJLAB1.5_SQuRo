import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from mjlab.asset_zoo.robots.SQuRo.SQuRo_constants import INIT_STATE
from mjlab.tasks.SQuRo_Hole.mdp.config import COMMAND_DIM, CMD_HEIGHT_F_IDS, CMD_HEIGHT_H_IDS, CMD_GAIT_FREQ_IDS, BASE_FREQ, LOW_POSE_CALIBRATION
from mjlab.tasks.SQuRo_Hole.mdp.indices import _ACT_JOINT_NAMES
from mjlab.tasks.SQuRo_Hole.mdp.reference import get_reference_joint_state, initialize_tables

from mjlab.scripts.Hole.probe_ik_accuracy import Probe
from mjlab.scripts._diagnostics import HOLE_DIAGNOSTICS_ROOT


# 从默认复位测量两位小数固定角的实际落地高度。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heights", type=float, nargs="+", default=[0.020, 0.021, 0.025, 0.030, 0.035, 0.039999])
    parser.add_argument("--output-dir", type=Path, default=HOLE_DIAGNOSTICS_ROOT / "pose_physics")
    args = parser.parse_args()
    if any(not 0.020 <= height < 0.040 for height in args.heights):
        parser.error("低姿态测量高度必须在[20,40)毫米内，单位为米")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe = Probe()
    initialize_tables("cpu")
    results = []
    for height in args.heights:
        command = torch.zeros(1, COMMAND_DIM)
        command[:, CMD_HEIGHT_F_IDS] = command[:, CMD_HEIGHT_H_IDS] = height
        command[:, CMD_GAIT_FREQ_IDS] = BASE_FREQ
        env = SimpleNamespace(device="cpu", step_dt=0.01, episode_length_buf=torch.zeros(1),
            command_manager=SimpleNamespace(_terms={"hole_cmd": SimpleNamespace(command=command)}))
        pos, vel = get_reference_joint_state(env)
        assert torch.isfinite(pos).all() and (vel == 0).all()
        start = probe.seed.copy()
        start[probe.act_qpos] = [INIT_STATE.joint_pos.get(name, 0.0) for name in _ACT_JOINT_NAMES]
        probe.data.qpos[:] = start
        probe.close_linkages()
        start = probe.data.qpos.copy()
        target = start.copy()
        target[probe.act_qpos] = pos[0].numpy()
        actual = probe.settle(start, height, target)
        row = {"height_mm": height * 1000, "reference": pos[0].tolist(), "dynamics": actual}
        results.append(row)
        print(json.dumps({"height_mm": height * 1000, "actual_mm": actual["mean_center_world_z_mm"],
                          "error_mm": actual["height_error_mm"]}, ensure_ascii=False), flush=True)
    for height, front, hind in LOW_POSE_CALIBRATION:
        if height in args.heights:
            assert all(abs(value - round(value, 2)) < 1e-12 for value in (*front, *hind))
    (args.output_dir / "rounded_poses_physics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
