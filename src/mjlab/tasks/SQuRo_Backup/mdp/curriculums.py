from __future__ import annotations
import torch
from typing import Any



_STEPS_PER_ITER = 96            # 控制dt 0.01 s, 96 步覆盖 0.96 s



# 阶段相关定义
STAGE1_1_ITER = 1000                # iter    0-1000: 纯模仿
STAGE1_2_ITER = 2000                # iter 1000-2000: 时间缩放课程 (λ 下限 2.0 -> 1.0)
STAGE1_3_ITER = 3000                # iter 2000-3000: 动作优化 (平滑权重第 2/3 档)
STAGE2_1_ITER = 5000                # iter 3000-5000: 受限空间课程 (开碰撞, a 0.40 -> 0.20)
STAGE2_2_ITER = 6000                # iter 5000-6000: 受限空间动作优化 (= 总训练轮数)
_STAGES = (0, STAGE1_1_ITER, STAGE1_2_ITER)



# P1 冗余时间课程
P1_SETTLE_MAX = 0.50                # P1 阶段最大冗余时间
P1_SETTLE_MIN = 0.00                # P1 阶段最小冗余时间
P1_SETTLE_START_ITER = 2000         # P1 阶段冗余时间课程开始轮数
P1_SETTLE_END_ITER = 3000           # P1 阶段冗余时间课程结束轮数
P1_CLOSE_FOLLOWS_SETTLE = False     # 重试截止不跟随 P1 冗余时间一起收缩



# 受限空间课程
WALL_D_MAX = 0.20                       # 采样上限
WALL_D_MIN_END = 0.06                   # 课程下限
WALL_D_MIN_STEP = 0.01                  # 课程推进参数
WALL_D_MIN_FRAC = 0.3                   # 取得最小间距的固定概率
CURRICULUM_BATCH_EPISODES = 256         # 累计 256 个最小间距回合后结算
CURRICULUM_BATCHES_REQUIRED = 2         # 连续 2 次达标后推进课程
WALL_X_POS = 0.05                       # +X 墙中心, 全程固定 (内侧 +0.040)
WALL_X_NEG_START = -0.08                # −X 墙中心起点 (= 旧课程的末档, 便于续训对齐)
WALL_X_NEG_END = -0.05                  # −X 墙中心终点 = 课程下界 (末档净宽 0.080, 两侧对称)
WALL_X_NEG_STEP = 0.005                 # -X 墙课程推进步
CURRICULUM_START_ITER = STAGE1_3_ITER   # 受限空间课程开始轮数
CURRICULUM_GATE_P_STOOD = 0.60          # 推进课程所需环境达标率
CURRICULUM_WINDOW_EPISODES = 3          # 每环境保留最近几个有效回合
CURRICULUM_MIN_DWELL_ITER = 100         # 每档最短驻留轮数
CURRICULUM_MAX_ITER = 9000              # 总预算, 与 STAGE2_2_ITER 解耦



# 时间缩放因子课程
TIME_SCALE_MAX = 3.0                    # 时间缩放因子采样最大值
TIME_SCALE_MIN_START = 2.0              # 时间缩放因子采样最小值开始
TIME_SCALE_MIN_END = 1.0                # 时间缩放因子采样最小值结束



# 脊柱误差缩放课程
SPN_AXIS_RELAX_ITER = 4000
SPN_AXIS_SCALE_STRICT: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
SPN_AXIS_SCALE_RELAXED: tuple[float, float, float, float] = (0.5, 1.0, 0.5, 1.0)



# 奖励权重课程曲线
_CURVES: dict[str, tuple[float, ...]] = {
    # 奖励项
    "weight_milestone_s1":      (10.0,),
    "weight_milestone_s2":      (15.0,),
    "weight_milestone_success": (2.0,),
    "weight_progress_s1":       (3.0,),
    "weight_progress_s2":       (3.0,),
    "weight_progress_s3":       (3.0,),
    "weight_s1_shape":          (2.0,),
    "weight_mimic_pos":         (10.0,),
    "weight_mimic_vel":         (5.0,),
    "weight_height":            (5.0,),
    # 惩罚项
    "weight_track_joint":       (1.0,),
    "weight_spn_track":         (2.0,),
    "weight_body_track":        (1.0,),
    "weight_leg_action":        (1.0,),
    "weight_leg_pose":          (6.0,),
    "weight_stand":             (3.0,),
    "weight_action_excess":     (0.5,),
    "weight_smooth_L1_leg":     (0.1, 0.2, 0.4),
    "weight_smooth_L1_spn":     (0.2, 0.2, 0.4),
    "weight_smooth_L2_leg":     (0.1, 0.2, 0.4),
    "weight_smooth_L2_spn":     (0.2, 0.2, 0.4),
    "weight_energy":            (0.1,),
    # 非权重项
    "sigma_leg_pos":            (10.0,),
    "sigma_spn_pos":            (20.0,),
    "sigma_neck_pos":           (10.0,),
    "sigma_leg_vel":            (0.5,),
    "sigma_spn_vel":            (0.5,),
    "sigma_neck_vel":           (0.5,),
    "sigma_height":             (500.0,),
    "alpha_neck_pos":           (0.3,),
    "alpha_neck_vel":           (0.3,),
}



# 奖励权重课程: 按当前训练 iter 返回各奖励项权重/σ
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
