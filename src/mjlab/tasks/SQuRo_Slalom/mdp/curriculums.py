from __future__ import annotations
import torch
from typing import Any
from .config import (
    VEL_MIN,
    STEPS_PER_ITER,
    PHASE1_END_ITER,
    PHASE1_MID_ITER,
    PHASE2_MID_ITER,
    FINAL_ACCEPT_RATE,
    PHASE0_MAX_SEGMENTS,
    PHASE0_MIN_SEGMENTS,
    CURVATURE_TARGET_MAX,
    SPACING_SUCCESS_RATE,
    SLALOM_CURVATURE_MAX,
    SLALOM_CURVATURE_MIN,
    FINAL_BIN_WEIGHT_GAIN,
    PHASE0_SEGMENT_JITTER,
    PHASE0_EPISODE_SECONDS,
    NORMAL_BIN_TARGET_RATE,
    SPACING_REQUIRED_BATCHES,
    VELOCITY_CURVATURE_POWER,
    PHASE0_TRANSITION_SECONDS,
    PHASE0_DYNAMIC_START_ITER,
    PHASE0_DYNAMIC_PROBABILITY,
    SPACING_EVALUATION_INTERVAL,
    SPACING_PRETRAIN_ITERATIONS,
    SPACING_MIN_DWELL_ITERATIONS,
    SPACING_INITIAL_CURVATURE_MAX,
    SPACING_EVALUATION_MIN_SAMPLES,
)

# 回放、绘图和验证脚本使用的配置再导出
from .config import (
    BASE_VEL as BASE_VEL,
    PHASE2_END_ITER as PHASE2_END_ITER,
    OBSTACLE_SPACING_START as OBSTACLE_SPACING_START,
    REFERENCE_CURVATURE_BLEND as REFERENCE_CURVATURE_BLEND,
    SPINE_REFERENCE_TIME_CONSTANT as SPINE_REFERENCE_TIME_CONSTANT,
)


_STEPS_PER_ITER = STEPS_PER_ITER      # 兼容旧导入名

# 由 config 派生的整数档位与曲率目标
SLALOM_CURVATURES = tuple(range(SLALOM_CURVATURE_MIN, SLALOM_CURVATURE_MAX + 1))
SLALOM_BIN_COUNT = len(SLALOM_CURVATURES)   # 整数峰值档位数，最终档按档位等概率采样
CURVATURE_TARGET = float(SLALOM_CURVATURE_MAX)  # 默认绕杆峰值，形变仍按二十归一化
Rmin = 1.0 / CURVATURE_TARGET       # 最小转弯半径

# 奖励权重阶段
_STAGES = (0, PHASE1_MID_ITER, PHASE1_END_ITER, PHASE2_MID_ITER)


_CURVES: dict[str, tuple[float, ...]] = {
    # 权重项
    "weight_mimic_pos":         (5.0, 5.0, 5.0),
    "weight_mimic_vel":         (2.5, 2.5, 2.5),
    "weight_height":            (2.5, 2.5, 2.5),
    "weight_track_vel":         (4.0, 4.0, 4.0),
    "weight_track_vyz":         (1.0, 1.0, 1.0),
    "weight_corridor":          (5.0, 5.0, 8.0),
    "weight_track_head":        (5.0, 5.0, 5.0),
    "weight_smooth_L1_leg":     (0.1, 0.5, 0.5),
    "weight_smooth_L1_spn":     (0.1, 0.5, 0.5),
    "weight_smooth_L2_leg":     (0.1, 0.5, 0.5),
    "weight_smooth_L2_spn":     (0.1, 0.5, 0.5),
    "weight_energy":            (0.1, 0.5, 0.5),
    "weight_leg_saturation":    (0.0, 0.0, 0.0),
    # 非权重项
    "sigma_leg_pos":            (5.0,),
    "sigma_spn_pos":            (10.0, 10.0, 20.0),
    "sigma_leg_vel":            (0.1,),
    "sigma_spn_vel":            (0.1,),
    "sigma_height":             (1000,),
    "sigma_track_vel":          (50,),
    "sigma_track_vyz":          (50,),
    "sigma_corridor":           (400.0, 2500.0, 2500.0, 2500.0),
    "corridor_inner_multiplier": (2.0,),
    "sigma_track_head":         (20,),
    "leg_saturation_threshold": (0.9,),
    "action_soft_limit":        (3.0,),
    "weight_action_soft_limit": (10.0, 10.0, 10.0),
    "weight_abnormal_leg_pose": (0.0, 0.0, 5.0),
    "abnormal_leg_pose_tolerance": (0.7,),
}



# 获取训练阶段
def get_training_phase(step_counter: int, phase0_only: bool = False) -> int:
    return 0 if phase0_only or step_counter // _STEPS_PER_ITER < PHASE1_END_ITER else 1



# 批量获取训练阶段 (逐环境, 用于按回合起点锁定的阶段判定)
def get_training_phase_batch(step_counter: torch.Tensor, phase0_only: bool = False) -> torch.Tensor:
    if phase0_only:
        return torch.zeros_like(step_counter, dtype=torch.bool)
    return (step_counter // _STEPS_PER_ITER) >= PHASE1_END_ITER



# 两阶段共用曲率速度曲线，保留直行和最大曲率的端点
def get_velocity_scale(curvature: Any) -> Any:
    remaining = 1.0 - abs(curvature) / CURVATURE_TARGET_MAX
    remaining = max(0.0, min(1.0, remaining)) if isinstance(remaining, (int, float)) else remaining.clip(0.0, 1.0)
    return VEL_MIN + (1.0 - VEL_MIN) * remaining ** VELOCITY_CURVATURE_POWER



# 一阶段曲率范围: 前两千轮从 0.5 扩到 20, 之后保持完整范围
def get_phase0_curvature_range(step_counter: int) -> tuple[float, float]:
    progress = min((step_counter // _STEPS_PER_ITER) / PHASE1_MID_ITER, 1.0)
    upper = 0.5 + progress * (CURVATURE_TARGET_MAX - 0.5)
    return -upper, upper



# 采样一阶段曲率日程: 恒定回合单段, 动态回合 4~6 段并在段间平滑过渡
def sample_phase0_schedule(iteration: int, count: int, device) -> dict:
    lower, upper = get_phase0_curvature_range(iteration * _STEPS_PER_ITER)
    dynamic = torch.rand(count, device=device) < (
        PHASE0_DYNAMIC_PROBABILITY if iteration >= PHASE0_DYNAMIC_START_ITER else 0.0)
    positions = torch.arange(PHASE0_MAX_SEGMENTS, device=device)
    segments = torch.empty(count, PHASE0_MAX_SEGMENTS, device=device).uniform_(lower, upper)
    counts = torch.randint(PHASE0_MIN_SEGMENTS, PHASE0_MAX_SEGMENTS + 1, (count,), device=device)
    counts = torch.where(dynamic, counts, torch.ones_like(counts))
    valid = positions[None, :] < counts[:, None]
    # 段时长 = 回合时长/段数 + 抖动, 归一后累积成段边界时间
    base = PHASE0_EPISODE_SECONDS / counts.float()
    durations = base[:, None] + (torch.rand(count, PHASE0_MAX_SEGMENTS, device=device) - 0.5) * 2.0 * PHASE0_SEGMENT_JITTER
    durations = torch.where(valid, durations.clamp(min=PHASE0_TRANSITION_SECONDS * 1.5), torch.zeros_like(durations))
    durations = durations * (PHASE0_EPISODE_SECONDS / durations.sum(dim=1, keepdim=True).clamp_min(1e-6))
    bounds = durations.cumsum(dim=1)
    bounds = torch.where(valid & (positions[None, :] < counts[:, None] - 1), bounds, torch.full_like(bounds, PHASE0_EPISODE_SECONDS + 1.0))
    segments = torch.where(valid, segments, segments[:, :1])
    return {"dynamic": dynamic, "segments": segments, "bounds": bounds, "peak": segments.abs().amax(dim=1)}



# 按日程求 t 时刻的一阶段目标曲率, 段间用 smootherstep 过渡 (支持逐环境时间网格)
def phase0_kappa_at(t: torch.Tensor, schedule: dict) -> torch.Tensor:
    segments = schedule["segments"]
    bounds = schedule["bounds"]
    expand = t.dim() > 1
    kappa = segments[:, 0]
    if expand:
        kappa = kappa[:, None].expand_as(t)
    for index in range(1, segments.shape[1]):
        start = bounds[:, index - 1] - 0.5 * PHASE0_TRANSITION_SECONDS
        previous = segments[:, index - 1]
        target = segments[:, index]
        if expand:
            start, previous = start[:, None], previous[:, None]
            target = target[:, None]
        ratio = ((t - start) / PHASE0_TRANSITION_SECONDS).clamp(0.0, 1.0)
        blend = ratio * ratio * (3.0 - 2.0 * ratio)
        kappa = torch.where(t >= start, previous + (target - previous) * blend, kappa)
    return kappa



# 峰值曲率对应的整数档位索引，越界时夹到端点
def curvature_bin_index(curvature: Any) -> int:
    index = int(round(float(curvature))) - SLALOM_CURVATURE_MIN
    return min(max(index, 0), SLALOM_BIN_COUNT - 1)



# 按档位权重采样绕杆峰值索引，upper 为当前课程允许的最高档索引
def sample_curvature_indices(count: int, weights, upper: int, device) -> torch.Tensor:
    weight_tensor = torch.as_tensor(weights, device=device, dtype=torch.float32)[: int(upper) + 1]
    return torch.multinomial(weight_tensor, int(count), replacement=True)



# 仅用独立真实碰撞考核推进无碰撞训练的采样范围
class SpacingCurriculumState:
    def __init__(self):
        self.max_curvature = SPACING_INITIAL_CURVATURE_MAX
        self.level_start_iteration = PHASE1_END_ITER
        self.last_evaluation_iteration = -1
        self.consecutive_success_batches = 0
        self.final_level_passed = False
        self.final_accepted = False
        self.bin_weights = [1.0] * SLALOM_BIN_COUNT
        self.history: list[dict] = []


    # 先完成无碰撞预训练，后续按固定间隔考核当前最难档。
    def should_evaluate(self, iteration: int) -> bool:
        return (
            iteration >= PHASE1_END_ITER + SPACING_PRETRAIN_ITERATIONS
            and iteration - self.level_start_iteration >= SPACING_MIN_DWELL_ITERATIONS
            and iteration - self.last_evaluation_iteration >= SPACING_EVALUATION_INTERVAL
        )


    # 统计一批考核的最低档回合数、通过率、分档统计与失败原因。
    def measure_batch(self, results: list[dict]) -> dict:
        eligible = [item for item in results if item.get("valid", True)]
        minimum = [item for item in eligible if int(round(float(item["curvature"]))) == self.max_curvature]
        count = len(minimum)
        successes = sum(int(item["success"]) for item in minimum)
        rate = successes / max(1, count)
        failures: dict[str, int] = {}
        for item in eligible:
            if not item["success"]:
                reason = item.get("failure", "")
                if reason:
                    failures[reason] = failures.get(reason, 0) + 1
        return {
            "count": count,
            "successes": successes,
            "success_rate": rate,
            "passed": count >= SPACING_EVALUATION_MIN_SAMPLES and rate >= SPACING_SUCCESS_RATE,
            "eligible": len(eligible),
            "eligible_successes": sum(int(item["success"]) for item in eligible),
            "failure_counts": failures,
            **self._bin_statistics(eligible),
        }


    # 按整数峰值档位统计回合数与通过率
    def _bin_statistics(self, eligible: list[dict]) -> dict:
        bin_count = [0] * SLALOM_BIN_COUNT
        bin_successes = [0] * SLALOM_BIN_COUNT
        for item in eligible:
            index = curvature_bin_index(item["curvature"])
            bin_count[index] += 1
            bin_successes[index] += int(item["success"])
        return {
            "bin_count": bin_count,
            "bin_successes": bin_successes,
            "bin_rate": [bin_successes[i] / bin_count[i] if bin_count[i] else 0.0 for i in range(SLALOM_BIN_COUNT)],
        }


    # 按档位通过率缺口更新训练采样权重，无样本档位保持 1.0
    def _update_bin_weights(self, bin_count: list, bin_successes: list, target: float) -> list[float]:
        return [
            1.0 + FINAL_BIN_WEIGHT_GAIN * max(0.0, target - bin_successes[i] / bin_count[i])
            if bin_count[i] else 1.0
            for i in range(SLALOM_BIN_COUNT)
        ]


    # 统计最终档一批考核的整体通过率与分档统计
    def measure_final_batch(self, results: list[dict]) -> dict:
        eligible = [item for item in results if item.get("valid", True)]
        stats = self._bin_statistics(eligible)
        failures: dict[str, int] = {}
        for item in eligible:
            if not item["success"]:
                reason = item.get("failure", "")
                if reason:
                    failures[reason] = failures.get(reason, 0) + 1
        count = sum(stats["bin_count"])
        successes = sum(stats["bin_successes"])
        rate = successes / max(1, count)
        return {
            "count": count,
            "successes": successes,
            "success_rate": rate,
            "passed": rate >= FINAL_ACCEPT_RATE,
            "eligible": len(eligible),
            "eligible_successes": successes,
            "failure_counts": failures,
            **stats,
        }


    # 记录一次完整考核：普通模式推进/进入最终档，最终模式只验收并更新档位采样权重
    def record_evaluation(self, batch_stats: list[dict], iteration: int, collision_enabled: bool) -> dict:
        if not collision_enabled:
            raise ValueError("杆距课程只接受开启真实杆碰撞的独立考核")
        if len(batch_stats) != SPACING_REQUIRED_BATCHES:
            raise ValueError(f"考核需要完整的 {SPACING_REQUIRED_BATCHES} 批回合")
        if not self.should_evaluate(iteration):
            return {"counted": False, "passed": False, "advanced": False,
                    "entered_final_mode": False, "mode": "final" if self.final_level_passed else "normal",
                    "success_rate": 0.0, "sample_count": 0,
                    "bin_count": [0] * SLALOM_BIN_COUNT, "bin_rate": [0.0] * SLALOM_BIN_COUNT,
                    "bin_weights": list(self.bin_weights)}
        self.last_evaluation_iteration = int(iteration)
        if self.final_level_passed:
            return self._record_final_evaluation(batch_stats, iteration)
        return self._record_normal_evaluation(batch_stats, iteration)


    # 普通模式：两批最低档都达标就推进一档，曲率十五通过后进入最终档
    def _record_normal_evaluation(self, batch_stats: list[dict], iteration: int) -> dict:
        passed = all(bool(batch["passed"]) for batch in batch_stats)
        sample_count = sum(int(batch["count"]) for batch in batch_stats)
        sample_successes = sum(int(batch["successes"]) for batch in batch_stats)
        rate = sample_successes / max(1, sample_count)
        bin_count = [sum(int(batch["bin_count"][i]) for batch in batch_stats) for i in range(SLALOM_BIN_COUNT)]
        bin_successes = [sum(int(batch["bin_successes"][i]) for batch in batch_stats) for i in range(SLALOM_BIN_COUNT)]
        bin_rate = [bin_successes[i] / bin_count[i] if bin_count[i] else 0.0 for i in range(SLALOM_BIN_COUNT)]
        self.bin_weights = self._update_bin_weights(bin_count, bin_successes, NORMAL_BIN_TARGET_RATE)
        self.consecutive_success_batches = (self.consecutive_success_batches + 1 if passed else 0)
        self.history.append({
            "iteration": int(iteration), "max_curvature": self.max_curvature,
            "count": sample_count, "successes": sample_successes, "success_rate": rate,
            "consecutive_batches": self.consecutive_success_batches, "evaluation": True,
            "passed": passed, "mode": "normal",
            "bin_count": bin_count, "bin_rate": bin_rate, "bin_weights": list(self.bin_weights),
            "batch_passed": [bool(batch["passed"]) for batch in batch_stats],
        })
        result = {
            "counted": True, "passed": passed, "mode": "normal", "advanced": False,
            "entered_final_mode": False, "success_rate": rate, "sample_count": sample_count,
            "previous_curvature": self.max_curvature, "max_curvature": self.max_curvature,
            "bin_count": bin_count, "bin_rate": bin_rate, "bin_weights": list(self.bin_weights),
            "batch_passed": [bool(batch["passed"]) for batch in batch_stats],
        }
        if not passed:
            return result
        if self.max_curvature < SLALOM_CURVATURE_MAX:
            self.max_curvature += 1
            self.level_start_iteration = int(iteration)
            self.consecutive_success_batches = 0
            result.update({"advanced": True, "max_curvature": self.max_curvature})
        else:
            self._enter_final_mode()
            result["entered_final_mode"] = True
        return result


    # 进入最终档：训练与考核切换为全档位等概率（训练随后按缺口加权）
    def _enter_final_mode(self) -> None:
        self.final_level_passed = True
        self.consecutive_success_batches = 0
        self.bin_weights = [1.0] * SLALOM_BIN_COUNT


    # 最终档：两批整体通过率都达到 80% 才算最终验收，并按档位缺口更新训练权重。
    def _record_final_evaluation(self, batch_stats: list[dict], iteration: int) -> dict:
        passed = all(bool(batch["passed"]) for batch in batch_stats)
        sample_count = sum(int(batch["count"]) for batch in batch_stats)
        sample_successes = sum(int(batch["successes"]) for batch in batch_stats)
        rate = sample_successes / max(1, sample_count)
        bin_count = [sum(int(batch["bin_count"][i]) for batch in batch_stats) for i in range(SLALOM_BIN_COUNT)]
        bin_successes = [sum(int(batch["bin_successes"][i]) for batch in batch_stats) for i in range(SLALOM_BIN_COUNT)]
        bin_rate = [bin_successes[i] / bin_count[i] if bin_count[i] else 0.0 for i in range(SLALOM_BIN_COUNT)]
        if passed:
            self.final_accepted = True
        self.bin_weights = self._update_bin_weights(bin_count, bin_successes, FINAL_ACCEPT_RATE)
        self.history.append({
            "iteration": int(iteration), "max_curvature": self.max_curvature,
            "count": sample_count, "successes": sample_successes, "success_rate": rate,
            "consecutive_batches": 0, "evaluation": True,
            "passed": passed, "mode": "final", "accepted": self.final_accepted,
            "bin_count": bin_count, "bin_rate": bin_rate, "bin_weights": list(self.bin_weights),
            "batch_passed": [bool(batch["passed"]) for batch in batch_stats],
        })
        return {
            "counted": True, "passed": passed, "mode": "final", "advanced": False,
            "entered_final_mode": False, "accepted": self.final_accepted,
            "success_rate": rate, "sample_count": sample_count,
            "previous_curvature": self.max_curvature, "max_curvature": self.max_curvature,
            "bin_count": bin_count, "bin_rate": bin_rate, "bin_weights": list(self.bin_weights),
            "batch_passed": [bool(batch["passed"]) for batch in batch_stats],
        }


    # 序列化档位、最终档状态、分档权重与历史，续训不会重新起课
    def state_dict(self) -> dict:
        return {
            "max_curvature": self.max_curvature,
            "level_start_iteration": self.level_start_iteration,
            "last_evaluation_iteration": self.last_evaluation_iteration,
            "consecutive_success_batches": self.consecutive_success_batches,
            "final_level_passed": self.final_level_passed,
            "final_accepted": self.final_accepted,
            "bin_weights": list(self.bin_weights),
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
        self.final_accepted = bool(state.get("final_accepted", False))
        weights = list(state.get("bin_weights", [1.0] * SLALOM_BIN_COUNT))
        self.bin_weights = ([float(value) for value in weights] if len(weights) == SLALOM_BIN_COUNT else [1.0] * SLALOM_BIN_COUNT)
        self.history = list(state.get("history", []))



# 奖励权重课程
class RewardCurriculum:
    def get_curriculum_values(self, current_step: int, phase0_only: bool = False) -> dict[str, float]:
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



reward_weight_curriculum = RewardCurriculum()



# 获取课程奖励权重
def get_curriculum_value(env, reward_name: str) -> float:
    command = env.command_manager._terms["slalom_cmd"]
    return reward_weight_curriculum.get_curriculum_values(env.common_step_counter, command.cfg.phase0_only)[reward_name]



