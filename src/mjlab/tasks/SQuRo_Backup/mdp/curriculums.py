from __future__ import annotations
import torch
from typing import Any


# 100 Hz 初筛配置：控制周期 0.002 × 5 = 0.01 s，96 步采样覆盖 0.96 s。
# RL 配置与课程轮数换算共用此值；采样段末不重置环境，后续采样继续当前回合。
# 短采样段更依赖价值估计，结果确定后再恢复更高物理精度和更长采样段验证。
_STEPS_PER_ITER = 96


# 训练阶段边界 (iter)
STAGE1_MID_ITER = 1000             # iter 0-1000:   纯模仿阶段
STAGE2_MID_ITER = 2000             # iter 1000-2000: 强化竖直/高度引导
STAGE2_END_ITER = 4000             # iter 2000-4000: 收敛/泛化 (预留走廊/命令)


# 阶段边界表 (RewardWeightCurriculum 按 iter 取段)
_STAGES = (0, 2000, 4000)


# 命令课程: time_scale λ (放慢倍数) 采样区间
TIME_SCALE_MAX = 4.0
TIME_SCALE_MIN_START = 2.0
TIME_SCALE_MIN_END = 1.0


# 奖励权重课程曲线 — 每阶段一个值, 值数量不足时取末值 (对齐 Slalom curriculums 风格)
#
# 权重唯一管理处: env_cfg 中所有奖励项的 cfg.weight 一律为 1.0,
# 生效权重 = cfg.weight(1.0) × 本表的值, 调权重只改这里。
# 本表同时容纳 weight_*(权重) / sigma_*(误差系数) / alpha_*(复合项配比) 三类量。
_CURVES: dict[str, tuple[float, ...]] = {
    "weight_mimic_pos":         (10.0,),
    "weight_mimic_vel":         (5.0,),
    "weight_spine_target":      (2.0,),
    "weight_leg_target":        (1.0,),
    "weight_height":            (5.0,),
    "weight_milestone_s1":      (10.0,),
    "weight_milestone_s2":      (15.0,),
    "weight_milestone_success": (35.0,),
    "weight_progress_s1":       (3.0,),
    "weight_progress_s2":       (3.0,),
    "weight_progress_s3":       (3.0,),
    "weight_smooth_L1_leg":     (0.1,),
    "weight_smooth_L1_spn":     (0.1,),
    "weight_smooth_L2_leg":     (0.1,),
    "weight_smooth_L2_spn":     (0.1,),
    "weight_energy":            (0.1,),
    "sigma_leg_pos":      (10.0,),
    "sigma_spn_pos":      (20.0,),
    "sigma_neck_pos":     (10.0,),
    "sigma_leg_vel":      (0.5,),
    "sigma_spn_vel":      (0.5,),
    "sigma_neck_vel":     (0.5,),
    "sigma_height":       (500.0,),
    "alpha_neck_pos":     (0.3,),
    "alpha_neck_vel":     (0.3,),
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


# 采样命令 time_scale λ (episode 内固定): 返回 [n] 张量
def get_curriculum_time_scale(step_counter: int, n: int, device: str) -> torch.Tensor:
    iter_num = step_counter // _STEPS_PER_ITER
    progress = min(1.0, max(0.0, (iter_num - STAGE1_MID_ITER) / (STAGE2_MID_ITER - STAGE1_MID_ITER)))
    lam_min = TIME_SCALE_MIN_START - progress * (TIME_SCALE_MIN_START - TIME_SCALE_MIN_END)
    return lam_min + torch.rand(n, device=device) * (TIME_SCALE_MAX - lam_min)
