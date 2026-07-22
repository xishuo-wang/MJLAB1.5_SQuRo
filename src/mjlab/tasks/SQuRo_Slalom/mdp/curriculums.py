"""SQuRo绕杆任务第一阶段 — 课程调度（数据驱动模式）

3阶段课程:
  阶段1 (0~500 iter): 直行 — ω=0, v逐步提升, 脊柱奖励关闭
  阶段2 (500~1500 iter): 转弯引入 — ω范围渐进扩展, 脊柱奖励逐步开启
  阶段3 (1500+ iter): 全范围 — 随机[v,ω]混合, 全奖励权重

课程配置集中在 _STAGES 和 _CURVES 中，通过阶段索引插值查表。
"""

from __future__ import annotations
from typing import Any

_STEPS_PER_ITER = 24
_STAGES = (0, 500, 1500)

_CURVES: dict[str, tuple[float, ...]] = {
    # 主奖励权重
    "weight_track_vel":     (2.0, 1.0, 1.0),
    "weight_track_omega":   (0.0, 1.0, 1.0),
    "weight_spine_turn":    (0.0, 0.3, 1.0),
    "weight_stability":     (0.5, 0.3, 0.3),
    "weight_energy":        (0.01, 0.01, 0.01),
    "weight_smooth_L1_leg": (0.1, 0.2, 0.3),
    "weight_smooth_L1_spn": (0.0, 0.1, 0.2),

    # 腿/脊柱 L2 平滑权重
    "weight_smooth_L2_leg": (0.1, 0.2, 0.3),
    "weight_smooth_L2_spn": (0.0, 0.1, 0.2),
}


class RewardWeightCurriculum:
    """数据驱动课程调度器"""

    def get_reward_weights(self, current_step: int) -> dict[str, float]:
        current_iter = current_step // _STEPS_PER_ITER
        result: dict[str, float] = {}
        for name, values in _CURVES.items():
            # 找到 ≤ current_iter 的最大阶段下标
            idx = 0
            for i, t in enumerate(_STAGES):
                if current_iter >= t:
                    idx = i
            safe_idx = idx if idx < len(values) else len(values) - 1
            result[name] = values[safe_idx]
        return result

    def get_current_stage_info(self, current_step: int) -> dict[str, Any]:
        current_iter = current_step // _STEPS_PER_ITER
        stage = 0
        for t in _STAGES:
            if current_iter >= t:
                stage = t
        return {
            "current_stage": stage,
            "current_iter": current_iter,
            "current_step": current_step,
            "reward_weights": self.get_reward_weights(current_step),
        }


reward_weight_curriculum = RewardWeightCurriculum()


def get_curriculum_reward_weight(env, reward_name: str) -> float:
    return reward_weight_curriculum.get_reward_weights(
        env.common_step_counter
    ).get(reward_name, 1.0)
