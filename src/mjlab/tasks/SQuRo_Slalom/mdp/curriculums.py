from __future__ import annotations
from typing import Any

_STEPS_PER_ITER = 24

# 曲率常量 — CURVATURE_TARGET_MAX 用于 Phase0 采样范围, CURVATURE_TARGET 用于绕杆弧
CURVATURE_MIN = 0.5            # Phase 0 课程曲率起始值
CURVATURE_TARGET_MAX = 20.0    # Phase 0 课程曲率最大值
CURVATURE_TARGET = 20.0        # Phase 1 绕杆弧曲率 (= 1/Rmin)

# 两阶段训练
PHASE1_MID_ITER = 2000          # iter 0-2000: 转弯曲率增大
PHASE1_END_ITER = 4000          # iter 2000-4000: 转弯基元
PHASE2_MID_ITER = 6000          # iter 4000-6000: 绕杆间距缩小阶段
PHASE2_END_ITER = 8000          # iter 6000-8000: 绕杆训练

# 绕杆阶段杆间距课程 — Phase 1 每 episode 在范围内随机采样 (同 Phase 0 κ 机制)
Rmin = 1/CURVATURE_TARGET          # 最小转弯半径 (= 1/κ_arc)
POLE_SPACING_MAX   = 0.25          # 间距上限 (固定)
POLE_SPACING_START = 0.20          # 间距下限起始值 (PHASE1_END_ITER)
POLE_SPACING_MIN   = 2 * Rmin      # 间距下限最小值 (PHASE2_MID_ITER)

# 奖励权重阶段
_STAGES = (0, 2000, 4000, 6000)

_CURVES: dict[str, tuple[float, ...]] = {
    "weight_mimic_pos":         (4.0, 4.0, 4.0),
    "weight_mimic_vel":         (2.0, 2.0, 2.0),
    "weight_height":            (2.0, 2.0, 2.0),
    "weight_track_vel":         (4.0, 4.0, 4.0),
    "weight_track_vyz":         (0.5, 0.5, 0.5),
    "weight_track_omg":         (5.0, 5.0, 5.0),
    "weight_corridor":          (5.0, 8.0, 8.0),
    "weight_track_head":        (5.0, 5.0, 5.0),
    "weight_smooth_L1_leg":     (0.1, 0.5, 0.5),
    "weight_smooth_L1_spn":     (0.1, 0.5, 0.5),
    "weight_smooth_L2_leg":     (0.1, 0.5, 0.5),
    "weight_smooth_L2_spn":     (0.1, 0.5, 0.5),
    "weight_energy":            (0.1, 0.5, 0.5),
    "weight_collision_body":    (0.0, 0.0, 1.0, 3.0),
    "weight_collision_leg":     (0.0, 1.0, 3.0),

    "sigma_leg_pos":            (5.0,),
    "sigma_spn_pos":            (10.0, 20.0, 20.0),
    "sigma_leg_vel":            (0.1,),
    "sigma_spn_vel":            (0.1,),
    "sigma_height":             (500,),
    "sigma_track_vel":          (50,),
    "sigma_track_vyz":          (50,),
    "sigma_track_omg":          (20,),
    "sigma_corridor":           (10, 20, 30),
    "sigma_collision":          (200,),
    "sigma_track_head":         (20,),
}


# 获取训练阶段
def get_training_phase(step_counter: int) -> int:
    return 0 if step_counter // _STEPS_PER_ITER < PHASE1_END_ITER else 1


# 获取杆间距采样范围 — 下限线性缩小到 POLE_SPACING_MIN, 上限固定 POLE_SPACING_MAX
def get_pole_spacing_range(step_counter: int) -> tuple[float, float]:
    iter_num = step_counter // _STEPS_PER_ITER
    if iter_num < PHASE1_END_ITER:
        return (POLE_SPACING_START, POLE_SPACING_MAX)
    progress = min(1.0, (iter_num - PHASE1_END_ITER) / (PHASE2_MID_ITER - PHASE1_END_ITER))
    low = POLE_SPACING_START - progress * (POLE_SPACING_START - POLE_SPACING_MIN)
    return (low, POLE_SPACING_MAX)


# 获取杆间距 (确定性中值, 用于回放/可视化的默认值)
def get_curriculum_pole_spacing(step_counter: int) -> float:
    low, high = get_pole_spacing_range(step_counter)
    return (low + high) / 2


# 奖励权重课程
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


# 获取课程奖励权重
def get_curriculum_reward_weight(env, reward_name: str) -> float:
    return reward_weight_curriculum.get_reward_weights(env.common_step_counter).get(reward_name, 1.0)
