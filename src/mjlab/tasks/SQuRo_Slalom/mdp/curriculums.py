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
PHASE0_EPISODE_SECONDS = 20.0
SLALOM_EPISODE_SECONDS = 30.0
PASSAGE_REQUIRED_POLES = 3
SPACING_INITIAL_CURVATURE_MAX = 10
SPACING_HARDEST_PROBABILITY = 0.30
SPACING_SUCCESS_BATCH = 256
SPACING_SUCCESS_RATE = 0.60
SPACING_REQUIRED_BATCHES = 2
SPACING_MIN_DWELL_ITERATIONS = 100
SPACING_PRETRAIN_ITERATIONS = 500
SPACING_EVALUATION_INTERVAL = 100
SPACING_EVALUATION_ENVS = 64


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


# 仅用独立真实碰撞考核推进无碰撞训练的采样范围。
class SlalomSpacingCurriculum:
    # 初始化可随检查点保存的档位和连续考核统计。
    def __init__(self):
        self.max_curvature = SPACING_INITIAL_CURVATURE_MAX
        self.level_start_iteration = PHASE1_END_ITER
        self.last_evaluation_iteration = -1
        self.consecutive_success_batches = 0
        self.final_level_passed = False
        self.history: list[dict] = []

    # 先完成无碰撞预训练，后续按固定间隔考核当前最难档。
    def should_evaluate(self, iteration: int) -> bool:
        return (
            iteration >= PHASE1_END_ITER + SPACING_PRETRAIN_ITERATIONS
            and iteration - self.level_start_iteration >= SPACING_MIN_DWELL_ITERATIONS
            and iteration - self.last_evaluation_iteration >= SPACING_EVALUATION_INTERVAL
        )

    # 每次必须完整提交一批真实碰撞回合，重复轮次不能重复计分。
    def record_evaluation(self, results: list[dict], iteration: int, collision_enabled: bool) -> bool:
        if not collision_enabled:
            raise ValueError("杆距课程只接受开启真实杆碰撞的独立考核")
        if len(results) != SPACING_SUCCESS_BATCH:
            raise ValueError(f"碰撞考核需要完整的 {SPACING_SUCCESS_BATCH} 个回合")
        if any(int(item["curvature"]) != self.max_curvature for item in results):
            raise ValueError("碰撞考核必须全部属于当前最小杆距档")
        if not self.should_evaluate(iteration):
            return False
        self.last_evaluation_iteration = int(iteration)
        success_rate = sum(bool(item["success"]) for item in results) / len(results)
        self.consecutive_success_batches = (
            self.consecutive_success_batches + 1 if success_rate >= SPACING_SUCCESS_RATE else 0
        )
        failures: dict[str, int] = {}
        for item in results:
            reason = item.get("failure", "")
            if reason:
                failures[reason] = failures.get(reason, 0) + 1
        self.history.append({
            "iteration": int(iteration), "max_curvature": self.max_curvature,
            "count": len(results), "success_rate": success_rate,
            "consecutive_batches": self.consecutive_success_batches, "failures": failures,
        })
        ready = self.consecutive_success_batches >= SPACING_REQUIRED_BATCHES
        self.final_level_passed = ready and self.max_curvature == SLALOM_CURVATURE_MAX
        if not ready or self.max_curvature >= SLALOM_CURVATURE_MAX:
            return False
        self.max_curvature += 1
        self.level_start_iteration = int(iteration)
        self.consecutive_success_batches = 0
        return True

    # 序列化档位、上次考核时间与连续批次，续训不会重新起课。
    def state_dict(self) -> dict:
        return {
            "max_curvature": self.max_curvature,
            "level_start_iteration": self.level_start_iteration,
            "last_evaluation_iteration": self.last_evaluation_iteration,
            "consecutive_success_batches": self.consecutive_success_batches,
            "final_level_passed": self.final_level_passed,
            "history": list(self.history),
        }

    # 恢复课程并拒绝超出当前整数路径族的档位。
    def load_state_dict(self, state: dict) -> None:
        curvature = int(state.get("max_curvature", SPACING_INITIAL_CURVATURE_MAX))
        if not SPACING_INITIAL_CURVATURE_MAX <= curvature <= SLALOM_CURVATURE_MAX:
            raise ValueError("检查点杆距课程的曲率上限越界")
        self.max_curvature = curvature
        for name in ("level_start_iteration", "last_evaluation_iteration", "consecutive_success_batches"):
            if name in state:
                setattr(self, name, int(state[name]))
        self.final_level_passed = bool(state.get("final_level_passed", False))
        self.history = list(state.get("history", []))


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
