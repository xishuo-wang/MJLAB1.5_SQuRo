from __future__ import annotations
import torch
from typing import Any
from .config import (
    HEIGHT_MAX,
    HEIGHT_MIN,
    HEIGHT_STEP,
    BASE_HEIGHT,
    EXIT_MARGIN,
    ENTRY_MARGIN,
    HEIGHT_FINE_STEP,
    BODY_HEIGHT_OFFSET,
    HEIGHT_LIST,
    OBSTACLE_HEIGHT_EDGES,
    FULL_HEIGHT_VALUES,
    HEIGHT_INITIAL_MIN,
    RANDOM_HEIGHT_VALUES,
    OBSTACLE_HEIGHT_SPLIT,
    HEIGHT_MIN_SUCCESS_RATE,
    HEIGHT_REQUIRED_BATCHES,
    HEIGHT_MIN_SUCCESS_BATCH,
    HEIGHT_FINE_STEP_THRESHOLD,
    HEIGHT_MIN_DWELL_ITERATIONS,
    HEIGHT_MIN_SAMPLE_PROBABILITY,
    FINAL_ACCEPT_RATE,
    NORMAL_BIN_TARGET_RATE,
    FINAL_BIN_WEIGHT_GAIN,
    RETREAT_FAILURE_LIMIT,
    RETREAT_REQUIRED_PASSES,
)



# 训练阶段与课程步数配置
STEPS_PER_ITER = 48
NUM_STAGES = 4
STAGE0_ITER = 0
STAGE1_ITER = 1000
STAGE2_ITER = 2000
STAGE3_ITER = 3000
HEIGHT_RULE_STAGE = 4
STAGE1_STEP = STAGE1_ITER * STEPS_PER_ITER
STAGE2_STEP = STAGE2_ITER * STEPS_PER_ITER
STAGE3_STEP = STAGE3_ITER * STEPS_PER_ITER
_STAGES = (0, STAGE1_ITER, STAGE2_ITER, STAGE3_ITER)


# 独立真实碰撞考核的课程推进配置
HEIGHT_EVALUATION_INTERVAL = 100          # 每 100 iter 评估一次
HEIGHT_EVALUATION_ENVS = 1024             # 考核环境数量, 与训练环境数一致
HEIGHT_EVALUATION_PRETRAIN_ITERATIONS = 500
HEIGHT_EVALUATION_MIN_SAMPLES = 256       # 每批最低档有效回合下限
HEIGHT_EVALUATION_FINAL_INTERVAL = 500    # 到达最终档后降低考核频率, 不停止考核


# 最终档全挡位采样: 8 个命令挡位左闭右开, 末档含上限
HEIGHT_BIN_EDGES = (HEIGHT_MIN,) + OBSTACLE_HEIGHT_EDGES + (HEIGHT_MAX,)
HEIGHT_BIN_COUNT = len(HEIGHT_BIN_EDGES) - 1


# 按挡位权重采样板底高度: 先选挡位再在挡内均匀取值
def sample_bin_heights(count: int, weights, device) -> torch.Tensor:
    edges = torch.tensor(HEIGHT_BIN_EDGES, device=device, dtype=torch.float32)
    weight_tensor = torch.as_tensor(weights, device=device, dtype=torch.float32)
    indices = torch.multinomial(weight_tensor, int(count), replacement=True)
    lower = edges[indices]
    upper = edges[indices + 1]
    return lower + torch.rand(int(count), device=device) * (upper - lower)


# 在 [lower,upper] 范围内按挡位权重采样板底高度, 用于普通模式 70% 部分
def sample_range_bin_heights(count: int, weights, lower: float, upper: float, device) -> torch.Tensor:
    edges = torch.tensor(HEIGHT_BIN_EDGES, device=device, dtype=torch.float32)
    bin_weights = [
        float(weights[index]) if (edges[index + 1] > lower and edges[index] < upper) else 0.0
        for index in range(HEIGHT_BIN_COUNT)
    ]
    weight_tensor = torch.as_tensor(bin_weights, device=device, dtype=torch.float32)
    indices = torch.multinomial(weight_tensor, int(count), replacement=True)
    low = torch.maximum(edges[indices], torch.full((len(indices),), lower, device=device))
    high = torch.minimum(edges[indices + 1], torch.full((len(indices),), upper, device=device))
    return low + torch.rand(int(count), device=device) * (high - low).clamp_min(0.0)



# 奖励权重与 σ 课程曲线
_CURVES: dict[str, tuple[float, ...]] = {
    # 奖励项
    "weight_mimic_pos":         (10.0, 10.0, 10.0, 10.0),
    "weight_mimic_vel":         (5.0, 5.0, 5.0, 5.0),
    "weight_vel":               (5.0, 5.0, 5.0, 10.0),
    "weight_height":            (2.5, 2.5, 2.5, 5.0),
    "foot_clearance":           (1.0, 1.0, 1.0, 1.0),
    "orientation":              (2.0, 2.0, 2.0, 4.0),
    "angle":                    (1.0, 1.0, 1.0, 1.0),
    "smoothness":               (0.1, 0.2, 0.2, 1.0),
    "weight_action_L2":         (0.1, 0.2, 0.2, 1.0),
    "weight_low_leg_hold":      (0.2, 0.2, 0.2, 0.2),
    "corridor":                 (0.0, 0.0, 5.0, 5.0),
    "virtual_contact":          (0.0, 0.0, 0.0, 5.0),
    # 非奖励项
    "sigma_leg_pos":            (5.0, 10.0, 10.0, 5.0),
    "sigma_spn_pos":            (5.0, 10.0, 10.0, 5.0),
    "sigma_neck_pos":           (5.0, 10.0, 10.0, 5.0),
    "sigma_leg_vel":            (0.1, 0.1, 0.1, 0.1),
    "sigma_spn_vel":            (0.1, 0.1, 0.1, 0.1),
    "sigma_neck_vel":           (0.1, 0.1, 0.1, 0.1),
    "sigma_vel":                (100, 100, 100, 100),
    "sigma_height":             (500, 1000, 1000, 1000),

    "sigma_angle_low":          (50.0, 50.0, 50.0, 50.0),
    "sigma_angle_high":         (100.0, 100.0, 100.0, 100.0),
    "sigma_orientation":        (5.0, 5.0, 5.0, 5.0),
    "sigma_corridor":           (2000.0, 2000.0, 2000.0, 2000.0),
    "sigma_virtual_contact":    (2000.0, 2000.0, 2000.0, 2000.0),
    "corridor_height_offset":   (0.030, 0.030, 0.030, BODY_HEIGHT_OFFSET),
}



SPN_AXIS_SCALE: tuple[float, float, float, float] = (1.5, 0.7, 1.5, 0.7)    # 脊柱误差缩放 (F_spine1, F_body, H_spine1, H_body)
STAGE_COLLISION_ENABLED = (False, False, False, False)      # 训练关闭实体碰撞, 仅独立考核开启



# 根据全局步数获取当前课程阶段
def get_current_stage(step_counter: int) -> int:
    if step_counter < STAGE1_STEP:
        return 1
    if step_counter < STAGE2_STEP:
        return 2
    if step_counter < STAGE3_STEP:
        return 3
    return HEIGHT_RULE_STAGE



# 判断是否已进入真实碰撞高度课程
def stage_uses_height_course(stage: int) -> bool:
    return stage >= HEIGHT_RULE_STAGE



# 判断该阶段是否要求实体碰撞
def stage_requires_collision(stage: int) -> bool:
    idx = min(max(int(stage), 1), NUM_STAGES) - 1
    return STAGE_COLLISION_ENABLED[idx]



# 按阶段获取高度跟踪采样范围
def heights_for_stage(stage: int) -> list | None:
    if stage == 1:
        return RANDOM_HEIGHT_VALUES
    if stage in (2, 3):
        return FULL_HEIGHT_VALUES
    return None



# 奖励权重课程
class RewardWeightCurriculum:
    def get_reward_weights(self, current_step: int) -> dict[str, float]:
        current_iter = current_step // STEPS_PER_ITER
        result: dict[str, float] = {}
        for name, values in _CURVES.items():
            idx = 0
            for i, t in enumerate(_STAGES):
                if current_iter >= t:
                    idx = i
            safe_idx = idx if idx < len(values) else len(values) - 1
            result[name] = values[safe_idx]
        # 走廊余量在2000～3000轮从30毫米线性减至最终值。
        progress = min(1.0, max(0.0, (current_step - STAGE2_STEP) / (STAGE3_STEP - STAGE2_STEP)))
        start, end = _CURVES["corridor_height_offset"][-2:]
        result["corridor_height_offset"] = start + (end - start) * progress
        return result


    # 判断是否要求开启限高板碰撞
    def should_enable_obstacle(self, current_step: int) -> bool:
        return stage_requires_collision(get_current_stage(current_step))



reward_weight_curriculum = RewardWeightCurriculum()



# 获取课程参数
def get_curriculum_reward_weight(env, reward_name: str) -> float:
    weights = reward_weight_curriculum.get_reward_weights(env.common_step_counter)
    try:
        return weights[reward_name]
    except KeyError:
        raise KeyError(f"课程参数 {reward_name!r} 未在 _CURVES 定义") from None



# 从单板几何计算查表边界
def obstacle_rule_bounds(center_x: float, half_length: float) -> tuple[float, float, float]:
    x1 = center_x - half_length
    x2 = center_x + half_length
    left = x1 - ENTRY_MARGIN
    middle = (x1 + x2) / 2.0
    right = x2 + EXIT_MARGIN
    return left, middle, right



# 按 base_Link 的世界 X 统一决定前后端高度
def compute_base_heights(
    base_x: torch.Tensor,
    center_x: float,
    half_length: float,
    bottom_z: torch.Tensor | float,
    equal_height: torch.Tensor | bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    left, middle, right = obstacle_rule_bounds(center_x, half_length)
    bottom = torch.as_tensor(bottom_z, device=base_x.device, dtype=base_x.dtype)
    high_obstacle = bottom >= OBSTACLE_HEIGHT_SPLIT
    use_equal = high_obstacle & torch.as_tensor(equal_height, device=base_x.device, dtype=torch.bool)
    front = torch.full_like(base_x, BASE_HEIGHT)
    hind = torch.full_like(base_x, BASE_HEIGHT)
    edges = bottom.new_tensor(OBSTACLE_HEIGHT_EDGES)
    heights = bottom.new_tensor(HEIGHT_LIST)
    target = heights[torch.bucketize(bottom.contiguous(), edges, right=True)]
    in_range = (base_x >= left) & (base_x <= right)
    front_low = in_range & (base_x < middle)
    hind_low = in_range & (base_x >= middle)
    front = torch.where(use_equal & in_range, target, front)
    hind = torch.where(use_equal & in_range, target, hind)
    front = torch.where((~use_equal) & front_low, target, front)
    hind = torch.where((~use_equal) & hind_low, target, hind)
    return front, hind



# 高度课程状态
class HeightCurriculumState:
    def __init__(self, level_start_iteration: int = STAGE3_ITER):
        self.z_min = HEIGHT_INITIAL_MIN
        self.z_max = HEIGHT_MAX
        self.level_start_iteration = int(level_start_iteration)
        self.batch_count = 0
        self.batch_success_count = 0
        self.consecutive_success_batches = 0
        self.level_episode_count = 0
        self.level_success_count = 0
        self.all_height_episode_count = 0
        self.all_height_success_count = 0
        self.failure_counts = {"未过板": 0, "侧绕": 0, "高度超限": 0, "跌倒": 0, "超时": 0}
        self.history: list[dict] = []
        self.final_level_passed = False
        self.final_accepted = False
        self.consecutive_failures = 0
        self.retreats = 0
        self.retreat_pending_advance = False
        self.bin_weights = [1.0] * HEIGHT_BIN_COUNT


    # 判断本轮是否执行独立真实碰撞考核, 最终档降低频率但不停止
    def should_evaluate(self, iteration: int) -> bool:
        if int(iteration) < STAGE3_ITER + HEIGHT_EVALUATION_PRETRAIN_ITERATIONS:
            return False
        interval = (HEIGHT_EVALUATION_FINAL_INTERVAL if self.final_level_passed
                    else HEIGHT_EVALUATION_INTERVAL)
        return (int(iteration) - STAGE3_ITER) % interval == 0


    # 记录一条真实碰撞回合并判断是否推进课程
    def record_episode(
        self,
        bottom_z: float,
        success: bool,
        iteration: int,
        stage: int,
        collision_enabled: bool,
        eligible: bool = True,
        failure_code: int = 0,
    ) -> dict:
        result = {
            "counted": False,
            "lowest_counted": False,
            "batch_closed": False,
            "advanced": False,
            "previous_z_min": self.z_min,
            "success_rate": self.batch_success_count / max(1, self.batch_count),
            "sample_count": self.batch_count,
        }
        if not eligible or stage != HEIGHT_RULE_STAGE or not collision_enabled:
            return result

        height = float(bottom_z)
        passed = bool(success)
        self.all_height_episode_count += 1
        self.all_height_success_count += int(passed)
        if not passed:
            failure_name = {1: "未过板", 2: "侧绕", 3: "高度超限", 4: "跌倒", 5: "超时"}.get(failure_code)
            if failure_name is not None:
                self.failure_counts[failure_name] += 1
        result["counted"] = True
        if abs(height - self.z_min) > 1e-6:
            return result

        self.level_episode_count += 1
        self.level_success_count += int(passed)
        self.batch_count += 1
        self.batch_success_count += int(passed)
        result["lowest_counted"] = True
        if self.batch_count < HEIGHT_MIN_SUCCESS_BATCH:
            if self._advance_if_ready(iteration):
                self._set_advance_result(result)
            return result

        rate = self.batch_success_count / self.batch_count
        self.consecutive_success_batches = (self.consecutive_success_batches + 1 if rate >= HEIGHT_MIN_SUCCESS_RATE else 0)
        self.history.append({
            "z_min": self.z_min,
            "iteration": int(iteration),
            "count": self.batch_count,
            "successes": self.batch_success_count,
            "success_rate": rate,
            "consecutive_batches": self.consecutive_success_batches,
        })
        self.batch_count = 0
        self.batch_success_count = 0
        result["batch_closed"] = True

        if self._advance_if_ready(iteration):
            self._set_advance_result(result)
        return result


    # 统计一批考核的最低档回合数、通过率、失败原因与分挡位统计
    def measure_batch(self, results: list) -> dict:
        eligible = [item for item in results if item.get("valid", True)]
        minimum = [item for item in eligible if abs(float(item["height"]) - self.z_min) <= 1e-6]
        count = len(minimum)
        successes = sum(int(item["success"]) for item in minimum)
        failure_names = {1: "未过板", 2: "侧绕", 3: "高度超限", 4: "跌倒", 5: "超时"}
        failure_counts: dict[str, int] = {}
        for item in eligible:
            if not item["success"]:
                name = failure_names.get(item.get("failure_code"))
                if name is not None:
                    failure_counts[name] = failure_counts.get(name, 0) + 1
        rate = successes / max(1, count)
        return {
            "count": count,
            "successes": successes,
            "success_rate": rate,
            "passed": count >= HEIGHT_EVALUATION_MIN_SAMPLES and rate >= HEIGHT_MIN_SUCCESS_RATE,
            "eligible": len(eligible),
            "eligible_successes": sum(int(item["success"]) for item in eligible),
            "failure_counts": failure_counts,
            **self._bin_statistics(eligible),
        }


    # 按命令挡位统计回合数与通过率
    def _bin_statistics(self, eligible: list) -> dict:
        heights = torch.tensor([float(item["height"]) for item in eligible], dtype=torch.float64)
        edges = torch.tensor(HEIGHT_BIN_EDGES[1:-1], dtype=torch.float64)
        indices = torch.bucketize(heights, edges, right=True).tolist() if len(eligible) else []
        bin_count = [0] * HEIGHT_BIN_COUNT
        bin_successes = [0] * HEIGHT_BIN_COUNT
        for index, item in zip(indices, eligible, strict=True):
            bin_count[index] += 1
            bin_successes[index] += int(item["success"])
        return {
            "bin_count": bin_count,
            "bin_successes": bin_successes,
            "bin_rate": [bin_successes[i] / bin_count[i] if bin_count[i] else 0.0
                         for i in range(HEIGHT_BIN_COUNT)],
        }


    # 按挡位通过率缺口更新训练采样权重, 无样本挡位保持 1.0
    def _update_bin_weights(self, bin_count: list, bin_successes: list, target: float) -> list[float]:
        return [
            1.0 + FINAL_BIN_WEIGHT_GAIN * max(0.0, target - bin_successes[i] / bin_count[i])
            if bin_count[i] else 1.0
            for i in range(HEIGHT_BIN_COUNT)
        ]


    # 统计 final 模式一批考核的各挡位通过率与整体通过率
    def measure_final_batch(self, results: list) -> dict:
        eligible = [item for item in results if item.get("valid", True)]
        stats = self._bin_statistics(eligible)
        failure_names = {1: "未过板", 2: "侧绕", 3: "高度超限", 4: "跌倒", 5: "超时"}
        failure_counts: dict[str, int] = {}
        for item in eligible:
            if not item["success"]:
                name = failure_names.get(item.get("failure_code"))
                if name is not None:
                    failure_counts[name] = failure_counts.get(name, 0) + 1
        count = sum(stats["bin_count"])
        successes = sum(stats["bin_successes"])
        rate = successes / max(1, count)
        return {
            "count": count,
            "successes": successes,
            "success_rate": rate,
            "passed": rate >= FINAL_ACCEPT_RATE,
            **stats,
            "eligible": len(eligible),
            "eligible_successes": successes,
            "failure_counts": failure_counts,
        }


    # 记录一次完整考核: 普通模式推进/退档, 最终模式只验收并更新挡位采样权重
    def record_evaluation(self, batch_stats: list, iteration: int) -> dict:
        if self.final_level_passed:
            return self._record_final_evaluation(batch_stats, iteration)

        passed = bool(batch_stats) and all(batch["passed"] for batch in batch_stats)
        sample_count = sum(int(batch["count"]) for batch in batch_stats)
        sample_successes = sum(int(batch["successes"]) for batch in batch_stats)
        rate = sample_successes / max(1, sample_count)
        self._add_episode_totals(batch_stats)

        # 普通模式: 按各挡位通过率更新 70% 部分的采样权重
        bin_count = [sum(int(batch["bin_count"][i]) for batch in batch_stats)
                     for i in range(HEIGHT_BIN_COUNT)]
        bin_successes = [sum(int(batch["bin_successes"][i]) for batch in batch_stats)
                         for i in range(HEIGHT_BIN_COUNT)]
        bin_rate = [bin_successes[i] / bin_count[i] if bin_count[i] else 0.0
                    for i in range(HEIGHT_BIN_COUNT)]
        self.bin_weights = self._update_bin_weights(bin_count, bin_successes, NORMAL_BIN_TARGET_RATE)

        self.level_episode_count = sample_count
        self.level_success_count = sample_successes
        self.batch_count = sample_count
        self.batch_success_count = sample_successes
        self.consecutive_success_batches = (self.consecutive_success_batches + 1 if passed else 0)
        self.consecutive_failures = 0 if passed else self.consecutive_failures + 1
        self.history.append({
            "z_min": self.z_min,
            "iteration": int(iteration),
            "count": sample_count,
            "successes": sample_successes,
            "success_rate": rate,
            "consecutive_batches": self.consecutive_success_batches,
            "consecutive_failures": self.consecutive_failures,
            "retreats": self.retreats,
            "evaluation": True,
            "passed": passed,
            "mode": "normal",
            "bin_count": bin_count,
            "bin_rate": bin_rate,
            "bin_weights": list(self.bin_weights),
            "batch_passed": [bool(batch["passed"]) for batch in batch_stats],
        })
        result = {
            "counted": True, "batch_closed": True, "advanced": False, "passed": passed,
            "success_rate": rate, "sample_count": sample_count,
            "previous_z_min": self.z_min, "mode": "normal", "retreated": False,
            "entered_final_mode": False, "bin_count": bin_count, "bin_rate": bin_rate,
            "bin_weights": list(self.bin_weights),
        }
        # 连续失败达到上限: 退一档, 之后重新推进需要连续达标
        if (not passed and self.consecutive_failures >= RETREAT_FAILURE_LIMIT
                and self.z_min < HEIGHT_INITIAL_MIN):
            previous_z_min = self.z_min
            step = HEIGHT_FINE_STEP if self.z_min <= HEIGHT_FINE_STEP_THRESHOLD else HEIGHT_STEP
            self.z_min = min(HEIGHT_INITIAL_MIN, round(self.z_min + step, 6))
            self.retreats += 1
            self.consecutive_failures = 0
            self.consecutive_success_batches = 0
            self.level_start_iteration = int(iteration)
            self.retreat_pending_advance = True
            result.update({"retreated": True, "previous_z_min": previous_z_min, "z_min": self.z_min})
            return result
        # 驻留与连续达标次数满足后降档; 退档后需要连续两轮达标
        required_passes = RETREAT_REQUIRED_PASSES if self.retreat_pending_advance else 1
        if (passed and self.consecutive_success_batches >= required_passes
                and int(iteration) - self.level_start_iteration >= HEIGHT_MIN_DWELL_ITERATIONS):
            if self.z_min > HEIGHT_MIN:
                previous_z_min = self.z_min
                step = HEIGHT_FINE_STEP if self.z_min <= HEIGHT_FINE_STEP_THRESHOLD else HEIGHT_STEP
                self.z_min = max(HEIGHT_MIN, round(self.z_min - step, 6))
                self.level_start_iteration = int(iteration)
                self.consecutive_success_batches = 0
                self.retreat_pending_advance = False
                result.update({"advanced": True, "z_min": self.z_min, "previous_z_min": previous_z_min})
            else:
                self._enter_final_mode()
                result["entered_final_mode"] = True
        return result


    # 进入最终档: 训练与考核切换为全挡位加权均匀采样
    def _enter_final_mode(self) -> None:
        self.final_level_passed = True
        self.consecutive_success_batches = 0
        self.consecutive_failures = 0
        self.bin_weights = [1.0] * HEIGHT_BIN_COUNT


    # 记录最终档一次考核: 80% 整体验收并按挡位通过率更新训练采样权重
    def _record_final_evaluation(self, batch_stats: list, iteration: int) -> dict:
        passed = bool(batch_stats) and all(batch["passed"] for batch in batch_stats)
        sample_count = sum(int(batch["count"]) for batch in batch_stats)
        sample_successes = sum(int(batch["successes"]) for batch in batch_stats)
        rate = sample_successes / max(1, sample_count)
        self._add_episode_totals(batch_stats)

        bin_count = [sum(int(batch["bin_count"][i]) for batch in batch_stats)
                     for i in range(HEIGHT_BIN_COUNT)]
        bin_successes = [sum(int(batch["bin_successes"][i]) for batch in batch_stats)
                         for i in range(HEIGHT_BIN_COUNT)]
        bin_rate = [bin_successes[i] / bin_count[i] if bin_count[i] else 0.0
                    for i in range(HEIGHT_BIN_COUNT)]
        self.level_episode_count = sample_count
        self.level_success_count = sample_successes
        self.batch_count = sample_count
        self.batch_success_count = sample_successes
        if passed:
            self.final_accepted = True
        self.bin_weights = self._update_bin_weights(bin_count, bin_successes, FINAL_ACCEPT_RATE)
        self.history.append({
            "z_min": self.z_min,
            "iteration": int(iteration),
            "count": sample_count,
            "successes": sample_successes,
            "success_rate": rate,
            "consecutive_batches": 0,
            "consecutive_failures": 0,
            "retreats": self.retreats,
            "evaluation": True,
            "passed": passed,
            "mode": "final",
            "accepted": self.final_accepted,
            "bin_count": bin_count,
            "bin_rate": bin_rate,
            "bin_weights": list(self.bin_weights),
            "batch_passed": [bool(batch["passed"]) for batch in batch_stats],
        })
        return {
            "counted": True, "batch_closed": True, "advanced": False, "passed": passed,
            "success_rate": rate, "sample_count": sample_count,
            "previous_z_min": self.z_min, "mode": "final", "retreated": False,
            "entered_final_mode": False, "accepted": self.final_accepted,
            "bin_count": bin_count, "bin_rate": bin_rate, "bin_weights": list(self.bin_weights),
        }


    # 累加全体回合统计与失败原因
    def _add_episode_totals(self, batch_stats: list) -> None:
        self.all_height_episode_count += sum(int(batch["eligible"]) for batch in batch_stats)
        self.all_height_success_count += sum(int(batch["eligible_successes"]) for batch in batch_stats)
        for batch in batch_stats:
            for name, count in batch.get("failure_counts", {}).items():
                self.failure_counts[name] = self.failure_counts.get(name, 0) + int(count)



    # 为所有降档路径填充统一日志数据
    def _set_advance_result(self, result: dict) -> None:
        previous_z_min = self.history[-1]["z_min"]
        previous_batch = self.history[-1]
        result.update({
            "advanced": True,
            "previous_z_min": previous_z_min,
            "z_min": self.z_min,
            "success_rate": previous_batch["success_rate"],
            "sample_count": previous_batch["count"],
        })


    # 在成功批数与驻留时间都达标时降低未来采样下界
    def _advance_if_ready(self, iteration: int) -> bool:
        dwell = int(iteration) - self.level_start_iteration
        if (
            self.consecutive_success_batches < HEIGHT_REQUIRED_BATCHES
            or dwell < HEIGHT_MIN_DWELL_ITERATIONS
            or self.z_min <= HEIGHT_MIN
        ):
            return False
        step = HEIGHT_FINE_STEP if self.z_min <= HEIGHT_FINE_STEP_THRESHOLD else HEIGHT_STEP
        self.z_min = max(HEIGHT_MIN, round(self.z_min - step, 6))
        self.level_start_iteration = int(iteration)
        self.batch_count = 0
        self.batch_success_count = 0
        self.consecutive_success_batches = 0
        self.level_episode_count = 0
        self.level_success_count = 0
        return True


    # 生成可保存到 checkpoint 的课程状态
    def state_dict(self) -> dict:
        return {
            "z_min": self.z_min,
            "z_max": self.z_max,
            "level_start_iteration": self.level_start_iteration,
            "batch_count": self.batch_count,
            "batch_success_count": self.batch_success_count,
            "consecutive_success_batches": self.consecutive_success_batches,
            "level_episode_count": self.level_episode_count,
            "level_success_count": self.level_success_count,
            "all_height_episode_count": self.all_height_episode_count,
            "minimum_sample_probability": HEIGHT_MIN_SAMPLE_PROBABILITY,
            "all_height_success_count": self.all_height_success_count,
            "failure_counts": dict(self.failure_counts),
            "history": list(self.history),
            "final_level_passed": self.final_level_passed,
            "final_accepted": self.final_accepted,
            "consecutive_failures": self.consecutive_failures,
            "retreats": self.retreats,
            "retreat_pending_advance": self.retreat_pending_advance,
            "bin_weights": list(self.bin_weights),
            "height_bin_edges": list(HEIGHT_BIN_EDGES),
        }


    # 从 checkpoint 状态恢复课程统计
    def load_state_dict(self, state: dict) -> None:
        for name in (
            "z_min", "z_max", "level_start_iteration", "batch_count",
            "batch_success_count", "consecutive_success_batches",
            "level_episode_count", "level_success_count",
            "all_height_episode_count", "all_height_success_count",
        ):
            if name in state:
                setattr(self, name, state[name])
        self.history = list(state.get("history", []))
        self.failure_counts = dict(state.get("failure_counts", self.failure_counts))
        if "final_level_passed" in state:
            self.final_level_passed = bool(state["final_level_passed"])
        if "final_accepted" in state:
            self.final_accepted = bool(state["final_accepted"])
        if "consecutive_failures" in state:
            self.consecutive_failures = int(state["consecutive_failures"])
        if "retreats" in state:
            self.retreats = int(state["retreats"])
        if "retreat_pending_advance" in state:
            self.retreat_pending_advance = bool(state["retreat_pending_advance"])
        weights = list(state.get("bin_weights", [1.0] * HEIGHT_BIN_COUNT))
        self.bin_weights = ([float(value) for value in weights]
                            if len(weights) == HEIGHT_BIN_COUNT else [1.0] * HEIGHT_BIN_COUNT)
        # 旧检查点 46mm 下限迁移到当前下限
        self.z_min = max(HEIGHT_MIN, min(float(self.z_min), float(self.z_max)))


    # 汇总最近一次考核结果, 避免降档后计数清零造成日志失真
    def metrics(self) -> dict[str, Any]:
        if self.history:
            last = self.history[-1]
            success_rate = float(last.get("success_rate", 0.0))
            passed = bool(last.get("passed", False))
        else:
            success_rate = self.level_success_count / max(1, self.level_episode_count)
            passed = False
        min_bin_rate = 0.0
        if self.history:
            bin_rate = self.history[-1].get("bin_rate")
            bin_count = self.history[-1].get("bin_count")
            if bin_rate and bin_count:
                rates = [rate for rate, count in zip(bin_rate, bin_count, strict=True) if count]
                min_bin_rate = float(min(rates)) if rates else 0.0
        return {
            "z_min": self.z_min,
            "success_rate": success_rate,
            "failure_rate": 1.0 - success_rate,
            "level_progress": 1.0 if passed else 0.0,
            "final_mode": int(self.final_level_passed),
            "final_accepted": int(self.final_accepted),
            "min_bin_rate": min_bin_rate,
        }
