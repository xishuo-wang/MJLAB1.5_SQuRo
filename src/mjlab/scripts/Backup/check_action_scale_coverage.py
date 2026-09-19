# 检查 action_scale 统一为 0.3 后, 动作是否能覆盖期望轨迹的极值并留余量
#   joint_delta = scale * action   →   needed_action = (target - default) / scale
#
# 2026-09-19 校正: 本脚本此前硬编码 CLIP = 6.0, 但 config/rl_cfg.py 的 clip_actions
# 实际为 None, 即**没有任何外层裁剪**。于是"余量 1.15x"是对着不存在的边界算的, 掩盖了
# 真实危险: 策略必须输出 ~5.2 才能到达 ctrlrange 极值, 实测已到 5.4~5.6。
# 现在 CLIP 从 RL 配置读取, 并在 clip 为 None 时改报"距真限位还有多少"。
from __future__ import annotations
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from mjlab.tasks.registry import load_rl_cfg
from mjlab.tasks.SQuRo_Backup.mdp.reference import _generate_reference_table

# 从任务 RL 配置读取真实裁剪值; None 表示不裁剪
try:
    CLIP: float | None = load_rl_cfg("Mjlab-SQuRo-Backup").clip_actions  # type: ignore[assignment]
except Exception:
    CLIP = None

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
        if CLIP is None:
            # 无裁剪: 唯一约束是 XML ctrlrange, 用"所需动作(=达到 ctrlrange 极值所需的动作量)"本身作标尺
            margin = float("nan")
            ok = "无裁剪"
        else:
            margin = CLIP / a03 if a03 > 0 else float("inf")
            ok = "OK" if a03 <= CLIP else "** 超限 **"
        # 是否触及关节限位
        jl = LIMITS[name]
        touch = "限位内" if (lo >= jl[0] - 1e-9 and hi <= jl[1] + 1e-9) else "** 触限 **"
        print(row + " | ".join(cells) + f" | {margin:7.2f}x | {ok} {touch}")
        if a03 > worst:
            worst, worst_j = a03, name

    print("-" * 108)
    if CLIP is None:
        print(f"@scale=0.3: 最大所需动作 {worst:.3f} ({worst_j}) —— 但 clip_actions=None, **没有任何外层裁剪**")
        print(f"            该值 = 把 {worst_j} 推到 ctrlrange 极值所需的动作量; 策略实测可达 5.4~5.6,")
        print(f"            即它长期工作在 |action|≈5, 是标称 ±1 的 5 倍。")
        print(f"            把 scale 提到 1.6 后同一目标只需 {worst * 0.3 / 1.6:.3f}。")
    else:
        print(f"@scale=0.3: 最大所需动作 {worst:.3f} ({worst_j}), clip={CLIP} → 余量 {CLIP / worst:.2f}x")
    print(f"@scale=0.5: 最大所需动作 {worst * 0.3 / 0.5:.3f}"
          + (f" → 余量 {CLIP / (worst * 0.3 / 0.5):.2f}x" if CLIP is not None else ""))
    print()
    print("说明: '所需动作'= (参考极值 - 默认角) / scale, 即跟踪到极值所需的动作幅度。")
    if CLIP is None:
        print("      clip_actions=None 时该值没有上界约束, 唯一约束是 XML ctrlrange —— 动作再大, ")
        print("      关节角到 ctrlrange 就停住。故'动作幅度'只反映策略被逼出的内部增益, 不设限。")
    else:
        print("      若某关节动作达到 clip, 则该关节无法跟踪到期望极值。余量 = clip / 所需动作。")


if __name__ == "__main__":
    main()
