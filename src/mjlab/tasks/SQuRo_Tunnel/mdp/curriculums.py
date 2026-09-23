from __future__ import annotations
from typing import Any


_STEPS_PER_ITER = 24       # 每 iter 步数 (对齐 Slalom)
CURVATURE_TARGET_MAX = 25.0   # 曲率归一化上界 (Tunnel 曲率恒 0, 仅供 reference 表接口)

PHASE1_START_ITER = 3000   # iter < 3000: Phase0 (无障碍物随机高度); 3000~6000: Phase1 (受限空间)


# 固定奖励权重 (Tunnel 暂不设阶段课程, 后续可按需扩展)
_FIXED_WEIGHTS: dict[str, float] = {
    "weight_mimic_pos": 5.0,
    "weight_mimic_vel": 2.5,
    "weight_height": 2.5,
    "weight_track_vel": 4.0,
    "weight_track_vyz": 1.0,
    "weight_corridor": 8.0,
    "weight_track_head": 5.0,
    "weight_smooth_L1_leg": 0.5,
    "weight_smooth_L1_spn": 0.5,
    "weight_smooth_L2_leg": 0.5,
    "weight_smooth_L2_spn": 0.5,
    "weight_energy": 0.5,
    "sigma_leg_pos": 5.0,
    "sigma_spn_pos": 10.0,
    "sigma_leg_vel": 0.1,
    "sigma_spn_vel": 0.1,
    "sigma_height": 1000.0,
    "sigma_track_vel": 50.0,
    "sigma_track_vyz": 50.0,
    "sigma_corridor": 50.0,
    "sigma_track_head": 20.0,
}


# 获取训练阶段: 0 = Phase0 (无障碍物随机高度), 1 = Phase1 (受限空间高度轨迹)
def get_training_phase(step_counter: int) -> int:
    return 0 if step_counter // _STEPS_PER_ITER < PHASE1_START_ITER else 1


# 奖励权重课程 (固定)
class RewardWeightCurriculum:
    def get_reward_weights(self, current_step: int) -> dict[str, float]:
        return dict(_FIXED_WEIGHTS)

    def get_current_stage_info(self, current_step: int) -> dict[str, Any]:
        return {
            "current_stage": 0,
            "current_iter": current_step // _STEPS_PER_ITER,
            "current_step": current_step,
            "reward_weights": self.get_reward_weights(current_step),
        }


reward_weight_curriculum = RewardWeightCurriculum()


# 获取课程奖励权重
def get_curriculum_reward_weight(env, reward_name: str) -> float:
    return _FIXED_WEIGHTS.get(reward_name, 1.0)
