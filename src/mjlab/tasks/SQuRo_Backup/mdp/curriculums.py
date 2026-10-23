from __future__ import annotations
from copy import deepcopy
from math import isfinite
import torch
from typing import Any


_STEPS_PER_ITER = 96  # 控制dt 0.01 s, 96 步覆盖 0.96 s


# 阶段相关定义
STAGE1_1_ITER = 1000  # iter    0-1000: 纯模仿
STAGE1_2_ITER = 2000  # iter 1000-2000: 时间缩放课程 (λ 下限 2.0 -> 1.0)
STAGE1_3_ITER = 3000  # iter 2000-3000: 动作优化 (平滑权重第 2/3 档)
STAGE2_1_ITER = 5000  # 旧阶段标记, 墙位改由独立考核推进
STAGE2_2_ITER = 6000  # 旧阶段标记, 总训练预算由 CURRICULUM_MAX_ITER 管理
_STAGES = (0, STAGE1_1_ITER, STAGE1_2_ITER)


# P1 冗余时间课程
P1_SETTLE_MAX = 0.50  # P1 阶段最大冗余时间
P1_SETTLE_MIN = 0.00  # P1 阶段最小冗余时间
P1_SETTLE_START_ITER = 2000  # P1 阶段冗余时间课程开始轮数
P1_SETTLE_END_ITER = 3000  # P1 阶段冗余时间课程结束轮数
P1_CLOSE_FOLLOWS_SETTLE = False  # 重试截止不跟随 P1 冗余时间一起收缩


# 受限空间课程
WALL_D_MAX = 0.20  # 采样上限
WALL_D_MIN_END = 0.06  # 课程下限
WALL_D_MIN_STEP = 0.01  # 课程推进参数
WALL_D_MIN_FRAC = 0.3  # 取得最小间距的固定概率
CURRICULUM_BATCH_EPISODES = 256  # 每批考核最低档有效回合下限
CURRICULUM_BATCHES_REQUIRED = 2  # 同次考核两批均达标后推进课程
WALL_X_POS = 0.05  # +X 墙中心, 全程固定 (内侧 +0.040)
WALL_X_NEG_START = -0.08  # −X 墙中心起点 (= 旧课程的末档, 便于续训对齐)
WALL_X_NEG_END = -WALL_D_MIN_END  # −X 墙中心终点: 末档内侧净宽 0.090 m
WALL_X_NEG_STEP = 0.005  # -X 墙课程推进步
CURRICULUM_START_ITER = STAGE1_3_ITER  # 受限空间课程开始轮数
WALL_MIN_SUCCESS_RATE = 0.60  # 独立考核最低档翻正率门槛
CURRICULUM_GATE_P_STOOD = WALL_MIN_SUCCESS_RATE  # 旧名兼容
CURRICULUM_WINDOW_EPISODES = 3  # 每环境保留最近几个有效回合
CURRICULUM_MIN_DWELL_ITER = 100  # 每档最短驻留轮数
CURRICULUM_MAX_ITER = 9000  # 总预算, 与 STAGE2_2_ITER 解耦

# 独立真实碰撞考核: 协议与 Hole 对齐, 训练仍保留真实墙碰撞
WALL_EVALUATION_ENVS = 1024
WALL_EVALUATION_INTERVAL = 100
WALL_EVALUATION_PRETRAIN_ITERATIONS = 500
WALL_EVALUATION_FINAL_INTERVAL = 500
WALL_FINAL_ACCEPT_RATE = 0.80
WALL_RETREAT_FAILURE_LIMIT = 5
WALL_RETREAT_REQUIRED_PASSES = 2
WALL_BIN_WEIGHT_GAIN = 2.0
WALL_CURRICULUM_STATE_VERSION = 2  # 首次尝试翻正考核, 不继承旧版通过成绩
WALL_D_LEVELS = tuple(
    round(WALL_D_MAX - i * WALL_D_MIN_STEP, 6)
    for i in range(round((WALL_D_MAX - WALL_D_MIN_END) / WALL_D_MIN_STEP) + 1)
)
WALL_BIN_EDGES = tuple(reversed(WALL_D_LEVELS))
WALL_BIN_COUNT = len(WALL_BIN_EDGES) - 1


# 时间缩放因子课程
TIME_SCALE_MAX = 3.0  # 时间缩放因子采样最大值
TIME_SCALE_MIN_START = 2.0  # 时间缩放因子采样最小值开始
TIME_SCALE_MIN_END = 1.0  # 时间缩放因子采样最小值结束


# 脊柱误差缩放课程
SPN_AXIS_RELAX_ITER = 4000
SPN_AXIS_SCALE_STRICT: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
SPN_AXIS_SCALE_RELAXED: tuple[float, float, float, float] = (0.5, 1.0, 0.5, 1.0)


# 奖励权重课程曲线
_CURVES: dict[str, tuple[float, ...]] = {
    # 奖励项
    "weight_milestone_s1": (10.0,),
    "weight_milestone_s2": (15.0,),
    "weight_milestone_success": (2.0,),
    "weight_progress_s1": (3.0,),
    "weight_progress_s2": (3.0,),
    "weight_progress_s3": (3.0,),
    "weight_s1_shape": (2.0,),
    "weight_mimic_pos": (10.0,),
    "weight_mimic_vel": (5.0,),
    "weight_height": (5.0,),
    # 惩罚项
    "weight_track_joint": (1.0,),
    "weight_spn_track": (2.0,),
    "weight_body_track": (1.0,),
    "weight_leg_action": (1.0,),
    "weight_leg_pose": (6.0,),
    "weight_stand": (3.0,),
    "weight_action_excess": (0.5,),
    "weight_smooth_L1_leg": (0.1, 0.2, 0.4),
    "weight_smooth_L1_spn": (0.2, 0.2, 0.4),
    "weight_smooth_L2_leg": (0.1, 0.2, 0.4),
    "weight_smooth_L2_spn": (0.2, 0.2, 0.4),
    "weight_energy": (0.1,),
    # 非权重项
    "sigma_leg_pos": (10.0,),
    "sigma_spn_pos": (20.0,),
    "sigma_neck_pos": (10.0,),
    "sigma_leg_vel": (0.5,),
    "sigma_spn_vel": (0.5,),
    "sigma_neck_vel": (0.5,),
    "sigma_height": (500.0,),
    "alpha_neck_pos": (0.3,),
    "alpha_neck_vel": (0.3,),
}


# 奖励权重课程: 按当前训练 iter 返回各奖励项权重/σ
class RewardWeightCurriculum:
    # 读取当前训练步对应的奖励权重与尺度。
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

    # 汇总当前奖励课程阶段与权重。
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


# 获取时间缩放因子
def get_curriculum_time_scale(step_counter: int, n: int, device: str) -> torch.Tensor:
    iter_num = step_counter // _STEPS_PER_ITER
    span = max(1, STAGE1_2_ITER - STAGE1_1_ITER)
    progress = min(1.0, max(0.0, (iter_num - STAGE1_1_ITER) / span))
    lam_min = TIME_SCALE_MIN_START - progress * (TIME_SCALE_MIN_START - TIME_SCALE_MIN_END)
    return lam_min + torch.rand(n, device=device) * (TIME_SCALE_MAX - lam_min)


# 获取脊柱误差缩放
def get_spn_axis_scale(step_counter: int) -> tuple[float, float, float, float]:
    if step_counter // _STEPS_PER_ITER < SPN_AXIS_RELAX_ITER:
        return SPN_AXIS_SCALE_STRICT
    return SPN_AXIS_SCALE_RELAXED


# 获取训练阶段
def get_training_phase(step_counter: int) -> int:
    return 0 if step_counter // _STEPS_PER_ITER < STAGE1_3_ITER else 1


# 档位表显式枚举
WALL_X_NEG_LEVELS: tuple[float, ...] = tuple(
    round(WALL_X_NEG_START + i * WALL_X_NEG_STEP, 6)
    for i in range(int(round((WALL_X_NEG_END - WALL_X_NEG_START) / WALL_X_NEG_STEP)) + 1)
)
assert abs(WALL_X_NEG_LEVELS[-1] - WALL_X_NEG_END) < 1e-9, WALL_X_NEG_LEVELS
CURRICULUM_LEVELS = len(WALL_X_NEG_LEVELS)


# 获取受限空间实体位置
def get_wall_positions(level: int) -> tuple[float, float]:
    idx = min(max(int(level), 0), CURRICULUM_LEVELS - 1)
    return WALL_X_NEG_LEVELS[idx], WALL_X_POS


# 获取对应的课程阶段
def get_level_for_wall_x_neg(x_neg: float) -> int:
    return min(range(CURRICULUM_LEVELS), key=lambda i: abs(WALL_X_NEG_LEVELS[i] - float(x_neg)))


# 获取 P1 余量
def get_p1_settle_margin(step_counter: int) -> float:
    it = int(step_counter) // _STEPS_PER_ITER
    if it <= P1_SETTLE_START_ITER:
        return P1_SETTLE_MAX
    if it >= P1_SETTLE_END_ITER:
        return P1_SETTLE_MIN
    frac = (it - P1_SETTLE_START_ITER) / float(P1_SETTLE_END_ITER - P1_SETTLE_START_ITER)
    return P1_SETTLE_MAX + frac * (P1_SETTLE_MIN - P1_SETTLE_MAX)


# 按固定间距区间统计考核结果, 末区间包含采样上限
def wall_bin_indices(distances: torch.Tensor) -> torch.Tensor:
    edges = distances.new_tensor(WALL_BIN_EDGES[1:-1])
    return torch.bucketize(distances.contiguous(), edges, right=True)


# 按区间权重采样连续墙位, 普通模式保留下界定点采样
def sample_wall_distances(
    count: int,
    lower: float,
    upper: float,
    minimum_fraction: float,
    weights,
    device,
    final_mode: bool = False,
) -> torch.Tensor:
    if upper <= lower:
        return torch.full((int(count),), lower, device=device)
    edges = torch.tensor(WALL_BIN_EDGES, device=device, dtype=torch.float32)
    low = edges[:-1].clamp_min(lower)
    high = edges[1:].clamp_max(upper)
    active = high > low
    weight_tensor = torch.as_tensor(weights, device=device, dtype=torch.float32)
    indices = torch.multinomial(weight_tensor * active, int(count), replacement=True)
    distances = low[indices] + torch.rand(int(count), device=device) * (high - low)[indices]
    if not final_mode:
        pick = torch.rand(int(count), device=device) < minimum_fraction
        distances = torch.where(pick, torch.full_like(distances, lower), distances)
    return distances


# 墙间距课程状态: 只消费独立真实碰撞考核, 不消费训练回合成绩
class WallCurriculumState:
    # 初始化墙位范围与独立考核计数
    def __init__(self, level_start_iteration: int = CURRICULUM_START_ITER):
        self.d_min = WALL_D_MAX
        self.d_max = WALL_D_MAX
        self.level_start_iteration = int(level_start_iteration)
        self.consecutive_success_evaluations = 0
        self.consecutive_failures = 0
        self.retreats = 0
        self.retreat_pending_advance = False
        self.final_level_passed = False
        self.final_accepted = False
        self.bin_weights = [1.0] * WALL_BIN_COUNT
        self.history: list[dict] = []

    # 普通模式每百轮考核, 最终模式每五百轮继续验收
    def should_evaluate(self, iteration: int) -> bool:
        if int(iteration) < CURRICULUM_START_ITER + WALL_EVALUATION_PRETRAIN_ITERATIONS:
            return False
        if self.history and int(iteration) <= int(self.history[-1]["iteration"]):
            return False
        interval = WALL_EVALUATION_FINAL_INTERVAL if self.final_level_passed else WALL_EVALUATION_INTERVAL
        return (int(iteration) - CURRICULUM_START_ITER) % interval == 0

    # 汇总每环境的首次尝试成绩, 普通模式按最低档翻正率判定。
    def measure_batch(self, results: list[dict]) -> dict:
        eligible = [item for item in results if item.get("valid", True)]
        if any(
            not isfinite(float(item["wall_d"])) or not self.d_min - 1e-6 <= float(item["wall_d"]) <= self.d_max + 1e-6
            for item in eligible
        ):
            raise ValueError("考核回合墙位超出当前课程范围")
        if len({item["env_id"] for item in eligible}) != len(eligible):
            raise ValueError("同一环境的考核回合不得重复计分")
        selected = (
            eligible
            if self.final_level_passed
            else [item for item in eligible if abs(float(item["wall_d"]) - self.d_min) <= 1e-6]
        )
        count = len(selected)
        successes = sum(int(item["success"]) for item in selected)
        bin_count = [0] * WALL_BIN_COUNT
        bin_successes = [0] * WALL_BIN_COUNT
        distances = torch.tensor([float(item["wall_d"]) for item in eligible])
        for index, item in zip(wall_bin_indices(distances).tolist(), eligible, strict=True):
            bin_count[index] += 1
            bin_successes[index] += int(item["success"])
        rate = successes / max(1, count)
        threshold = WALL_FINAL_ACCEPT_RATE if self.final_level_passed else WALL_MIN_SUCCESS_RATE
        enough = count > 0 if self.final_level_passed else count >= CURRICULUM_BATCH_EPISODES
        return {
            "mode": "final" if self.final_level_passed else "normal",
            "wall_d_min": self.d_min,
            "wall_d_max": self.d_max,
            "count": count,
            "successes": successes,
            "success_rate": rate,
            "passed": enough and rate >= threshold,
            "eligible": len(eligible),
            "eligible_successes": sum(int(item["success"]) for item in eligible),
            "bin_count": bin_count,
            "bin_successes": bin_successes,
            "bin_rate": [bin_successes[i] / value if value else 0.0 for i, value in enumerate(bin_count)],
        }

    # 两批在同一档位完成后一次结算, 缺样本或跨档位结果拒绝写入。
    def record_evaluation(self, batches: list[dict], iteration: int) -> dict:
        mode = "final" if self.final_level_passed else "normal"
        if len(batches) != CURRICULUM_BATCHES_REQUIRED:
            raise ValueError("独立考核必须收齐同次的两批结果")
        if self.history and int(iteration) <= int(self.history[-1]["iteration"]):
            raise ValueError("同一轮考核不得重复结算")
        for batch in batches:
            if (
                batch["mode"] != mode
                or abs(float(batch["wall_d_min"]) - self.d_min) > 1e-9
                or abs(float(batch["wall_d_max"]) - self.d_max) > 1e-9
                or int(batch["eligible"]) != WALL_EVALUATION_ENVS
                or len(batch["bin_count"]) != WALL_BIN_COUNT
                or len(batch["bin_successes"]) != WALL_BIN_COUNT
                or sum(batch["bin_count"]) != int(batch["eligible"])
                or not 0 <= int(batch["successes"]) <= int(batch["count"]) <= int(batch["eligible"])
                or sum(batch["bin_successes"]) != int(batch["eligible_successes"])
                or any(
                    not 0 <= ok <= count for ok, count in zip(batch["bin_successes"], batch["bin_count"], strict=True)
                )
            ):
                raise ValueError("考核档位不一致或首次尝试结果不足, 拒绝推进课程")
        passed = all(bool(batch["passed"]) for batch in batches)
        count = sum(int(batch["count"]) for batch in batches)
        successes = sum(int(batch["successes"]) for batch in batches)
        bin_count = [sum(int(batch["bin_count"][i]) for batch in batches) for i in range(WALL_BIN_COUNT)]
        bin_successes = [sum(int(batch["bin_successes"][i]) for batch in batches) for i in range(WALL_BIN_COUNT)]
        target = WALL_FINAL_ACCEPT_RATE if self.final_level_passed else WALL_MIN_SUCCESS_RATE
        self.bin_weights = [
            1.0 + WALL_BIN_WEIGHT_GAIN * max(0.0, target - bin_successes[i] / value) if value else 1.0
            for i, value in enumerate(bin_count)
        ]
        result = {
            "iteration": int(iteration),
            "mode": mode,
            "passed": passed,
            "sample_count": count,
            "successes": successes,
            "success_rate": successes / max(1, count),
            "previous_d_min": self.d_min,
            "advanced": False,
            "retreated": False,
            "entered_final_mode": False,
            "bin_count": bin_count,
            "bin_rate": [bin_successes[i] / value if value else 0.0 for i, value in enumerate(bin_count)],
            "batch_passed": [bool(batch["passed"]) for batch in batches],
        }
        if self.final_level_passed:
            self.final_accepted |= passed
        else:
            self.consecutive_success_evaluations = self.consecutive_success_evaluations + 1 if passed else 0
            self.consecutive_failures = 0 if passed else self.consecutive_failures + 1
            previous = [value for value in WALL_D_LEVELS if value > self.d_min + 1e-9]
            if not passed and self.consecutive_failures >= WALL_RETREAT_FAILURE_LIMIT and previous:
                self.d_min = min(previous)
                self.retreats += 1
                self.retreat_pending_advance = True
                self.consecutive_failures = 0
                self.consecutive_success_evaluations = 0
                self.level_start_iteration = int(iteration)
                result["retreated"] = True
            else:
                required = WALL_RETREAT_REQUIRED_PASSES if self.retreat_pending_advance else 1
                ready = int(iteration) - self.level_start_iteration >= CURRICULUM_MIN_DWELL_ITER
                if passed and ready and self.consecutive_success_evaluations >= required:
                    if self.d_min > WALL_D_MIN_END + 1e-9:
                        self.d_min = max(WALL_D_MIN_END, round(self.d_min - WALL_D_MIN_STEP, 6))
                        self.level_start_iteration = int(iteration)
                        self.retreat_pending_advance = False
                        result["advanced"] = True
                    else:
                        self.final_level_passed = True
                        self.consecutive_failures = 0
                        self.bin_weights = [1.0] * WALL_BIN_COUNT
                        result["entered_final_mode"] = True
                    self.consecutive_success_evaluations = 0
        result.update(
            {
                "d_min": self.d_min,
                "bin_weights": list(self.bin_weights),
                "consecutive_failures": self.consecutive_failures,
                "consecutive_success_evaluations": self.consecutive_success_evaluations,
                "retreats": self.retreats,
                "final_accepted": self.final_accepted,
            }
        )
        self.history.append(deepcopy(result))
        return result

    # 保存课程状态, 不保存未完成的考核批次
    def state_dict(self) -> dict:
        return {
            "version": WALL_CURRICULUM_STATE_VERSION,
            "d_min": self.d_min,
            "d_max": self.d_max,
            "level_start_iteration": self.level_start_iteration,
            "consecutive_success_evaluations": self.consecutive_success_evaluations,
            "consecutive_failures": self.consecutive_failures,
            "retreats": self.retreats,
            "retreat_pending_advance": self.retreat_pending_advance,
            "final_level_passed": self.final_level_passed,
            "final_accepted": self.final_accepted,
            "bin_weights": list(self.bin_weights),
            "bin_edges": list(WALL_BIN_EDGES),
            "history": deepcopy(self.history),
        }

    # 恢复课程并检查几何范围与采样权重
    def load_state_dict(self, state: dict) -> None:
        version = int(state["version"])
        lower, upper = float(state["d_min"]), float(state["d_max"])
        weights = [float(value) for value in state["bin_weights"]]
        if (
            version not in (1, WALL_CURRICULUM_STATE_VERSION)
            or not isfinite(lower)
            or not isfinite(upper)
            or not WALL_D_MIN_END <= lower <= upper == WALL_D_MAX
            or len(weights) != WALL_BIN_COUNT
            or any(not isfinite(value) or value <= 0.0 for value in weights)
            or list(state["bin_edges"]) != list(WALL_BIN_EDGES)
            or (bool(state["final_level_passed"]) and abs(lower - WALL_D_MIN_END) > 1e-9)
            or (bool(state["final_accepted"]) and not bool(state["final_level_passed"]))
            or any(
                int(state[name]) < 0
                for name in (
                    "level_start_iteration",
                    "consecutive_success_evaluations",
                    "consecutive_failures",
                    "retreats",
                )
            )
        ):
            raise ValueError("检查点墙位课程范围或分档权重不兼容")
        if version == 1:
            state = WallCurriculumState(int(state["level_start_iteration"])).state_dict()
            state.update(d_min=lower, d_max=upper)
            weights = state["bin_weights"]
        self.d_min, self.d_max = lower, upper
        self.bin_weights = weights
        for name in (
            "level_start_iteration",
            "consecutive_success_evaluations",
            "consecutive_failures",
            "retreats",
        ):
            setattr(self, name, int(state[name]))
        for name in ("retreat_pending_advance", "final_level_passed", "final_accepted"):
            setattr(self, name, bool(state[name]))
        self.history = deepcopy(state["history"])

    # 保留最近完成的考核成绩, 换档后日志不清零
    def metrics(self) -> dict[str, Any]:
        last = self.history[-1] if self.history else {}
        return {
            "wall_d_min": self.d_min,
            "success_rate": float(last.get("success_rate", 0.0)),
            "sample_count": int(last.get("sample_count", 0)),
            "passed": bool(last.get("passed", False)),
            "final_mode": self.final_level_passed,
            "final_accepted": self.final_accepted,
            "retreats": self.retreats,
        }
