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
    LOW_TARGET_HEIGHT,
    BODY_HEIGHT_OFFSET,
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
)



STEPS_PER_ITER = 48
NUM_STAGES = 4
STAGE0_ITER = 0
STAGE1_ITER = 1000
STAGE2_ITER = 2000
STAGE3_ITER = 3000
HEIGHT_COURSE_STAGE = 4
STAGE1_STEP = STAGE1_ITER * STEPS_PER_ITER
STAGE2_STEP = STAGE2_ITER * STEPS_PER_ITER
STAGE3_STEP = STAGE3_ITER * STEPS_PER_ITER



# 奖励曲线直接共用阶段边界
_STAGES = (0, STAGE1_ITER, STAGE2_ITER, STAGE3_ITER)



# 奖励权重课程曲线
_CURVES: dict[str, tuple[float, ...]] = {
    # 奖励项
    "mimic_pos":                (10.0, 10.0, 10.0, 10.0),
    "mimic_vel":                (5.0, 5.0, 5.0, 5.0),
    "vel":                      (5.0, 5.0, 5.0, 10.0),
    "height":                   (2.5, 2.5, 2.5, 5.0),
    "foot_clearance":           (1.0, 1.0, 1.0, 1.0),
    "reached":                  (1.0, 1.0, 1.0, 1.0),
    "orientation":              (2.0, 2.0, 2.0, 4.0),
    "angle":                    (1.0, 1.0, 1.0, 1.0),
    "smoothness":               (0.1, 0.2, 0.2, 1.0),
    "body_contact":             (0.0, 0.0, 0.0, 5.0),
    "corridor":                 (0.0, 0.0, 5.0, 5.0),
    # 角度 σ 按各环境高度模式选择。
    "sigma_angle_low":          (50.0, 50.0, 50.0, 50.0),
    "sigma_angle_high":         (100.0, 100.0, 100.0, 100.0),
    # 虚拟净空 σ: 超出量的指数衰减系数 (超出 5mm 时按 exp(-sigma*0.005^2) 计)
    "sigma_body_contact":       (2000.0, 2000.0, 2000.0, 2000.0),
    "sigma_corridor":           (2000.0, 2000.0, 2000.0, 2000.0),
    # 位置/速度模仿的腿部 σ
    "sigma_leg_pos":            (5.0, 10.0, 10.0, 5.0),
    "sigma_leg_vel":            (0.1, 0.1, 0.1, 0.1),
    # 位置/速度模仿的脊柱 σ
    "sigma_spn_pos":            (5.0, 10.0, 10.0, 5.0),
    "sigma_spn_vel":            (0.1, 0.1, 0.1, 0.1),
    # 头颈 σ: 曲线与脊柱一致, 保证替换模仿奖励前后行为不变
    "sigma_neck_pos":           (5.0, 10.0, 10.0, 5.0),
    "sigma_neck_vel":           (0.1, 0.1, 0.1, 0.1),
    # 高度 σ
    "height_sigma":             (500.0, 1000.0, 1000.0, 1000.0),
}



# 脊柱误差的逐轴缩放, 顺序 = 参考表脊柱 4 列 (F_spine1, F_body, H_spine1, H_body)
# 即 (侧摆, 扭转, 俯仰, 扭转)
SPN_AXIS_SCALE: tuple[float, float, float, float] = (1.5, 0.7, 1.5, 0.7)

# 各阶段是否要求限高板实体碰撞 (编译期固化; 切换由 runner 重建环境完成)
STAGE_COLLISION = (False, False, False, True)


# 按全局步数取当前阶段，统一命令、奖励课程与碰撞边界。
def get_current_stage(step_counter: int) -> int:
    if step_counter < STAGE1_STEP:
        return 1
    if step_counter < STAGE2_STEP:
        return 2
    if step_counter < STAGE3_STEP:
        return 3
    return HEIGHT_COURSE_STAGE



# 判断是否已进入真实碰撞高度课程。
def stage_uses_height_course(stage: int) -> bool:
    return stage >= HEIGHT_COURSE_STAGE



# 该阶段是否要求实体碰撞
def stage_requires_collision(stage: int) -> bool:
    idx = min(max(int(stage), 1), NUM_STAGES) - 1
    return STAGE_COLLISION[idx]



# 按阶段获取热身高度池; 课程阶段不使用随机高度命令池。
def heights_for_stage(stage: int):
    if stage == 1:
        return RANDOM_HEIGHT_VALUES
    if stage in (2, 3):
        return FULL_HEIGHT_VALUES
    return None



# 奖励权重课程: 按当前训练 iter 返回各奖励项权重/σ
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
        return result


    # 关卡是否要求开启限高板碰撞 (只读视图, 供基线脚本展示课程意图)
    def should_enable_holes(self, current_step: int) -> bool:
        return stage_requires_collision(get_current_stage(current_step))


reward_weight_curriculum = RewardWeightCurriculum()



# 获取课程参数，缺失定义时明确报错。
def get_curriculum_reward_weight(env, reward_name: str) -> float:
    weights = reward_weight_curriculum.get_reward_weights(env.common_step_counter)
    try:
        return weights[reward_name]
    except KeyError:
        raise KeyError(f"课程参数 {reward_name!r} 未在 _CURVES 定义") from None


# 从单板几何计算查表边界。
def rule_bounds(center_x: float, half_length: float) -> tuple[float, float, float]:
    x1 = center_x - half_length
    x2 = center_x + half_length
    left = x1 - ENTRY_MARGIN
    middle = (x1 + x2) / 2.0
    right = x2 + EXIT_MARGIN
    return left, middle, right


# 按 base_Link 的世界 X 统一决定前后端高度，低板两段互斥。
def sample_base_heights(
    base_x: torch.Tensor,
    center_x: float,
    half_length: float,
    bottom_z: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    left, middle, right = rule_bounds(center_x, half_length)
    bottom = torch.as_tensor(bottom_z, device=base_x.device, dtype=base_x.dtype)
    high_board = bottom >= OBSTACLE_HEIGHT_SPLIT
    front = torch.full_like(base_x, BASE_HEIGHT)
    hind = torch.full_like(base_x, BASE_HEIGHT)
    high_target = bottom - BODY_HEIGHT_OFFSET
    in_range = (base_x >= left) & (base_x <= right)
    front_low = in_range & (base_x < middle)
    hind_low = in_range & (base_x >= middle)
    front = torch.where(high_board & in_range, high_target, front)
    hind = torch.where(high_board & in_range, high_target, hind)
    front = torch.where((~high_board) & front_low, LOW_TARGET_HEIGHT, front)
    hind = torch.where((~high_board) & hind_low, LOW_TARGET_HEIGHT, hind)
    return front, hind



# 逐环境回合结果聚合为最低板高课程状态。
class HeightCurriculumState:
    def __init__(self, start_iteration: int = STAGE3_ITER):
        self.z_min = HEIGHT_INITIAL_MIN
        self.z_max = HEIGHT_MAX
        self.level_start_iteration = int(start_iteration)
        self.batch_count = 0
        self.batch_success_count = 0
        self.consecutive_success_batches = 0
        self.level_episode_count = 0
        self.level_success_count = 0
        self.all_height_episode_count = 0
        self.all_height_success_count = 0
        self.failure_counts = {"未过板": 0, "侧绕": 0, "高度超限": 0, "跌倒": 0, "超时": 0}
        self.history: list[dict] = []


    # 记录一条真实碰撞回合并判断是否应降档。
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
        if not eligible or stage != HEIGHT_COURSE_STAGE or not collision_enabled:
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
        self.consecutive_success_batches = (
            self.consecutive_success_batches + 1
            if rate >= HEIGHT_MIN_SUCCESS_RATE
            else 0
        )
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


    # 为所有降档路径填充统一日志数据。
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


    # 在成功批数与驻留时间都达标时降低未来采样下界。
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


    # 生成可保存到 checkpoint 的课程状态。
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
        }


    # 从 checkpoint 状态恢复课程统计。
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


    # 汇总当前档日志: 最低档成功率/失败率与挡位推进进度。
    def metrics(self) -> dict[str, Any]:
        success_rate = self.level_success_count / max(1, self.level_episode_count)
        return {
            "z_min": self.z_min,
            "success_rate": success_rate,
            "failure_rate": 1.0 - success_rate,
            "level_progress": self.consecutive_success_batches / max(1, HEIGHT_REQUIRED_BATCHES),
        }
