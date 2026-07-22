from __future__ import annotations
from typing import Any


_STEPS_PER_ITER = 24
_STAGES = (0, 1000, 2000)


_CURVES: dict[str, tuple[float, ...]] = {
    "weight_mimic_pos":         (5.0, 5.0, 5.0),
    "weight_mimic_vel":         (2.5, 2.5, 2.5),
    "weight_track_vel":         (4.0, 4.0, 4.0),
    "weight_track_omg":         (2.0, 2.0, 2.0),
    "weight_smooth_L1_leg":     (0.1, 0.2, 0.2),
    "weight_smooth_L1_spn":     (0.1, 0.2, 0.2),
    "weight_smooth_L2_leg":     (0.1, 0.2, 0.2),
    "weight_smooth_L2_spn":     (0.1, 0.2, 0.2),
    "weight_energy":            (0.1, 0.2, 0.2),

    "sigma_mimic_pos":          (5.0, 5.0, 5.0),
    "sigma_mimic_vel":          (0.1, 0.1, 0.1),
    "sigma_track_vel":          (50, 50, 50),
    "sigma_track_omg":          (20, 20, 20),
}


class RewardWeightCurriculum:
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
    return reward_weight_curriculum.get_reward_weights(env.common_step_counter).get(reward_name, 1.0)
