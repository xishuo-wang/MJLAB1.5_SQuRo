from __future__ import annotations
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from mjlab.tasks.SQuRo_Backup.mdp.reference import _generate_reference_table

# 检查 action_scale 统一为 0.3 后, 动作是否能覆盖期望轨迹的极值并留余量
#   joint_delta = scale * action   →   needed_action = (target - default) / scale
#   clip_actions = 6.0 (见 config/rl_cfg.py)

CLIP = 6.0
JOINTS = ["F_spine1", "F_body", "Neck_yaw", "Neck_pitch",
          "FL_shoulder", "FL_elbow", "FR_shoulder", "FR_elbow",
          "H_spine1", "H_body", "HL_hip", "HL_knee", "HR_hip", "HR_knee"]

# 默认关节角 (events.py reset_model 写入的站立姿态)
DEFAULT = np.array([0.0, 0.0, 0.0, 0.0,
                    0.1, -0.3, 0.1, -0.3,
                    0.0, 0.0,
                    -0.1, 0.3, -0.1, 0.3])

# 关节软限位 (SQuRo.xml)
LIMITS = {
    "F_spine1": (-0.6, 0.6), "F_body": (-1.57, 1.57),
    "Neck_yaw": (-0.8, 0.8), "Neck_pitch": (-0.9, 0.9),
    "FL_shoulder": (-1.5, 1.9), "FL_elbow": (-1.8, 2.5),
    "FR_shoulder": (-1.5, 1.9), "FR_elbow": (-1.8, 2.5),
    "H_spine1": (-0.6, 0.6), "H_body": (-1.57, 1.57),
    "HL_hip": (-1.5, 0.8), "HL_knee": (-0.5, 1.9),
    "HR_hip": (-1.5, 0.8), "HR_knee": (-0.5, 1.9),
}


def main() -> None:
    _, ref = _generate_reference_table()   # [T, 14] 期望关节角
    print(f"参考表: {ref.shape[0]} 帧 x {ref.shape[1]} 关节")
    print()
    hdr = f"{'关节':<12} {'默认':>7} {'期望min':>8} {'期望max':>8} | "
    print(hdr + " | ".join(f"{'a@'+str(s):>10}" for s in (0.3, 0.5)) + " |  余量@0.3 | 结论")
    print("-" * 108)

    worst = 0.0
    worst_j = ""
    for j, name in enumerate(JOINTS):
        d = DEFAULT[j]
        lo, hi = float(ref[:, j].min()), float(ref[:, j].max())
        # 需要的动作幅度 = 期望角与默认角之差 / scale
        need = np.abs(np.array([lo, hi]) - d)
        row = f"{name:<12} {d:+7.3f} {lo:+8.3f} {hi:+8.3f} | "
        cells = []
        for s in (0.3, 0.5):
            a = need / s
            cells.append(f"{a.max():10.3f}")
        a03 = need.max() / 0.3
        margin = CLIP / a03 if a03 > 0 else float("inf")
        ok = "OK" if a03 <= CLIP else "** 超限 **"
        # 是否触及关节限位
        jl = LIMITS[name]
        touch = "限位内" if (lo >= jl[0] - 1e-9 and hi <= jl[1] + 1e-9) else "** 触限 **"
        print(row + " | ".join(cells) + f" | {margin:7.2f}x | {ok} {touch}")
        if a03 > worst:
            worst, worst_j = a03, name

    print("-" * 108)
    print(f"@scale=0.3: 最大所需动作 {worst:.3f} ({worst_j}), clip={CLIP} → 余量 {CLIP / worst:.2f}x")
    print(f"@scale=0.5: 最大所需动作 {worst * 0.3 / 0.5:.3f} → 余量 {CLIP / (worst * 0.3 / 0.5):.2f}x")
    print()
    print("说明: 若某关节动作达到 clip, 则该关节无法跟踪到期望极值。")
    print("      余量 = clip / 所需动作, >1 即有余量。")


if __name__ == "__main__":
    main()
