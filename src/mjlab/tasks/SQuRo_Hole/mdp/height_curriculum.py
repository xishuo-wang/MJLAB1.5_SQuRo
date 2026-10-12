from __future__ import annotations

from .config import COURSE_START_ITER, HEIGHT_COURSE_STAGE
from .height_schedule import (
    HEIGHT_INITIAL_MIN,
    HEIGHT_MAX,
    HEIGHT_MIN,
    HEIGHT_MIN_DWELL_ITERATIONS,
    HEIGHT_MIN_SAMPLE_PROBABILITY,
    HEIGHT_MIN_SUCCESS_BATCH,
    HEIGHT_MIN_SUCCESS_RATE,
    HEIGHT_REQUIRED_BATCHES,
    HEIGHT_STEP,
)


# 逐环境回合结果聚合为最低板高课程状态。
class HeightCurriculumState:
    VERSION = 1

    def __init__(self, start_iteration: int = COURSE_START_ITER):
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
        self.z_min = max(HEIGHT_MIN, round(self.z_min - HEIGHT_STEP, 6))
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
            "version": self.VERSION,
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
        if int(state.get("version", 0)) != self.VERSION:
            raise ValueError(f"不支持的高度课程状态版本: {state.get('version')}")
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

    # 汇总当前档和全高度的通过率。
    def metrics(self) -> dict[str, float | int]:
        return {
            "z_min": self.z_min,
            "z_max": self.z_max,
            "level_success_rate": self.level_success_count / max(1, self.level_episode_count),
            "level_episode_count": self.level_episode_count,
            "batch_count": self.batch_count,
            "consecutive_success_batches": self.consecutive_success_batches,
            "all_height_success_rate": self.all_height_success_count / max(1, self.all_height_episode_count),
            "all_height_episode_count": self.all_height_episode_count,
            "failure_counts": dict(self.failure_counts),
        }
