from __future__ import annotations
from typing import Any

import torch


_STEPS_PER_ITER = 48                # 与 Hole 的每轮策略步数一致

PHASE1_MID_ITER = 2000              # iter 0-2000: 转弯曲率增大
PHASE1_END_ITER = 3000              # 转圈结束后，新回合进入绕杆
PHASE2_MID_ITER = 6000              # 绕杆奖励课程的后段边界
PHASE2_END_ITER = 8000              # 完整训练预算

CURVATURE_TARGET_MAX = 20.0         # 最大曲率同时用于命令采样与最大形变归一化
SLALOM_CURVATURE_MIN = 5
SLALOM_CURVATURE_MAX = 15
SLALOM_CURVATURES = tuple(range(SLALOM_CURVATURE_MIN, SLALOM_CURVATURE_MAX + 1))
CURVATURE_TARGET = float(SLALOM_CURVATURE_MAX)  # 默认绕杆峰值，形变仍按二十归一化
Rmin = 1.0 / CURVATURE_TARGET       # 最小转弯半径

SMOOTH_TIME = 1.0                   # 仅用于定义固定平滑弧长，不代表实际过渡耗时
SMOOTH_VEL = 0.025                  # 固定平滑弧长 = SMOOTH_VEL × SMOOTH_TIME

POLE_SPACING_START = 0.20           # 编译占位；运行时杆距由本回合曲率决定

GAIT_FREQ_MIN = 1.0                 # Phase 0 步频采样下限
GAIT_FREQ_MAX = 2.0                 # Phase 0 步频采样上限

# 两阶段共用速度曲线。
BASE_VEL = 0.1                      # 基础速度 (m/s, 1Hz 时)
VELOCITY_CURVATURE_POWER = 1.5
VEL_MIN = 0.15                      # 最大曲率下速度缩放比例 (转弯低速)
REFERENCE_CURVATURE_BLEND = 2.0     # 二阶段后脊柱在零曲率附近的平滑范围


# 奖励权重阶段
_STAGES = (0, PHASE1_MID_ITER, PHASE1_END_ITER, PHASE2_MID_ITER)


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
    "sigma_corridor":           (400.0, 2500.0, 2500.0, 2500.0),
    "corridor_inner_multiplier": (2.0,),
    "sigma_track_head":         (20,),
}


# 获取训练阶段
def get_training_phase(step_counter: int, phase0_only: bool = False) -> int:
    return 0 if phase0_only or step_counter // _STEPS_PER_ITER < PHASE1_END_ITER else 1


# 批量获取训练阶段 (逐环境, 用于按回合起点锁定的阶段判定)
def get_training_phase_batch(step_counter: torch.Tensor, phase0_only: bool = False) -> torch.Tensor:
    if phase0_only:
        return torch.zeros_like(step_counter, dtype=torch.bool)
    return (step_counter // _STEPS_PER_ITER) >= PHASE1_END_ITER


# 两阶段共用曲率速度曲线，保留直行和最大曲率的端点。
def get_velocity_scale(curvature: Any) -> Any:
    remaining = 1.0 - abs(curvature) / CURVATURE_TARGET_MAX
    remaining = max(0.0, min(1.0, remaining)) if isinstance(remaining, (int, float)) else remaining.clip(0.0, 1.0)
    return VEL_MIN + (1.0 - VEL_MIN) * remaining ** VELOCITY_CURVATURE_POWER


# 无直行路径族的最小和最大杆心距。
def get_pole_spacing_bounds() -> tuple[float, float]:
    from .slalom_geometry import get_slalom_geometry
    return (get_slalom_geometry(SLALOM_CURVATURE_MAX)["spacing"],
            get_slalom_geometry(SLALOM_CURVATURE_MIN)["spacing"])


# 回放未指定条件时采用峰值十五；训练按回合采样整数峰值。
def get_curriculum_pole_spacing(step_counter: int) -> float:
    del step_counter
    return get_pole_spacing_bounds()[0]


# 奖励权重课程
class RewardWeightCurriculum:
    def get_reward_weights(self, current_step: int, phase0_only: bool = False) -> dict[str, float]:
        current_iter = current_step // _STEPS_PER_ITER
        if phase0_only:
            current_iter = min(current_iter, PHASE1_END_ITER - 1)
        result: dict[str, float] = {}
        for name, values in _CURVES.items():
            idx = 0
            for i, t in enumerate(_STAGES):
                if current_iter >= t:
                    idx = i
            safe_idx = idx if idx < len(values) else len(values) - 1
            result[name] = values[safe_idx]
            if name == "sigma_corridor":
                progress = min(max(current_iter / PHASE1_MID_ITER, 0.0), 1.0)
                result[name] = values[0] + progress * (values[1] - values[0])
        return result

    def get_current_stage_info(self, current_step: int, phase0_only: bool = False) -> dict[str, Any]:
        current_iter = current_step // _STEPS_PER_ITER
        stage_iter = min(current_iter, PHASE1_END_ITER - 1) if phase0_only else current_iter
        stage = 0
        for t in _STAGES:
            if stage_iter >= t:
                stage = t
        return {
            "current_stage": stage,
            "current_iter": current_iter,
            "current_step": current_step,
            "reward_weights": self.get_reward_weights(current_step, phase0_only),
        }


reward_weight_curriculum = RewardWeightCurriculum()


# 获取课程奖励权重
def get_curriculum_reward_weight(env, reward_name: str) -> float:
    command = env.command_manager._terms["slalom_cmd"]
    return reward_weight_curriculum.get_reward_weights(
        env.common_step_counter, command.cfg.phase0_only
    )[reward_name]
