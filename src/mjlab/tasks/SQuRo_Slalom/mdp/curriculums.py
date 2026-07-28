from __future__ import annotations
from typing import Any

_STEPS_PER_ITER = 24

# 两阶段训练
PHASE1_END_ITER = 4000       # iter 0-4000: 转弯基元
PHASE2_END_ITER = 6000       # iter 4000-6000: 绕杆训练

# 绕杆阶段杆间距课程
POLE_SPACING_START = 0.20    # 绕杆起始杆间距 (宽)
POLE_SPACING_MIN = 0.15      # 绕杆最小杆间距 (= 2.5×Rmin)

# 奖励权重阶段 (与训练阶段对齐)
_STAGES = (0, 2000, 4000)

_CURVES: dict[str, tuple[float, ...]] = {
    "weight_mimic_pos":         (5.0, 5.0, 5.0),
    "weight_mimic_vel":         (2.5, 2.5, 2.5),
    "weight_height":            (2.5, 2.5, 2.5),
    "weight_track_vel":         (4.0, 4.0, 4.0),
    "weight_track_vyz":         (1.0, 1.0, 1.0),
    "weight_track_omg":         (5.0, 5.0, 5.0),
    "weight_corridor":          (5.0, 5.0, 8.0),
    "weight_track_head":        (5.0, 5.0, 5.0),
    "weight_smooth_L1_leg":     (0.1, 0.5, 0.5),
    "weight_smooth_L1_spn":     (0.1, 0.5, 0.5),
    "weight_smooth_L2_leg":     (0.1, 0.5, 0.5),
    "weight_smooth_L2_spn":     (0.1, 0.5, 0.5),
    "weight_energy":            (0.1, 0.5, 0.5),

    "sigma_leg_pos":            (5.0,),
    "sigma_spn_pos":            (10.0, 10.0, 20.0),
    "sigma_leg_vel":            (0.1,),
    "sigma_spn_vel":            (0.1,),
    "sigma_height":             (1000,),
    "sigma_track_vel":          (50,),
    "sigma_track_vyz":          (50,),
    "sigma_track_omg":          (20,),
    "sigma_corridor":           (10, 20),
    "sigma_track_head":         (20,),
}


# ========== 训练阶段查询 ==========
def get_training_phase(step_counter: int) -> int:
    return 0 if step_counter // _STEPS_PER_ITER < PHASE1_END_ITER else 1


def get_curriculum_pole_spacing(step_counter: int) -> float:
    iter_num = step_counter // _STEPS_PER_ITER
    if iter_num < PHASE1_END_ITER:
        return POLE_SPACING_START
    progress = min(1.0, (iter_num - PHASE1_END_ITER) / (PHASE2_END_ITER - PHASE1_END_ITER))
    return POLE_SPACING_START - progress * (POLE_SPACING_START - POLE_SPACING_MIN)


# ========== 奖励权重 ==========
class RewardWeightCurriculum:
    def get_reward_weights(self, current_step: int) -> dict[str, float]:
        current_iter = current_step // _STEPS_PER_ITER
        result: dict[str, float] = {}
        for name, values in _CURVES.items():
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
