from __future__ import annotations
from typing import Any


_STEPS_PER_ITER = 24               # 每 iter 步数 (与 Slalom 一致)

# 训练阶段边界 (iter)
STAGE1_MID_ITER = 1000             # iter 0-1000:   纯模仿阶段
STAGE2_MID_ITER = 2000             # iter 1000-2000: 强化竖直/高度引导
STAGE2_END_ITER = 4000             # iter 2000-4000: 收敛/泛化 (预留走廊/命令)


# 阶段边界表 (RewardWeightCurriculum 按 iter 取段)
_STAGES = (0, 1000, 2000, 4000)


# 奖励权重课程曲线 — 每阶段一个值, 值数量不足时取末值 (对齐 Slalom curriculums 风格)
_CURVES: dict[str, tuple[float, ...]] = {
    # 模仿 (核心, 全程保持较高权重)
    "weight_mimic_pos":   (5.0, 5.0, 5.0, 5.0),
    "weight_mimic_vel":   (2.5, 2.5, 2.5, 2.5),
    # 竖直/高度 (先轻后重, 引导策略在模仿翻身之后稳定站直)
    "weight_upright":     (2.0, 3.0, 5.0, 5.0),
    "weight_height":      (2.0, 3.0, 5.0, 5.0),
    # 关节位置/速度 σ
    "sigma_leg_pos":      (5.0, 5.0, 5.0, 5.0),
    "sigma_spn_pos":      (10.0, 10.0, 20.0, 20.0),
    "sigma_neck_pos":     (5.0, 5.0, 5.0, 5.0),
    "sigma_leg_vel":      (0.1, 0.1, 0.1, 0.1),
    "sigma_spn_vel":      (0.1, 0.1, 0.1, 0.1),
    "sigma_neck_vel":     (0.1, 0.1, 0.1, 0.1),
    "sigma_upright":      (5.0, 5.0, 10.0, 10.0),
    "sigma_height":       (500.0, 500.0, 1000.0, 1000.0),
    # 预留: 走廊奖励课程 (后续加入走廊奖励后启用; 当前 rewards.py 未引用)
    "weight_corridor":    (0.0, 0.0, 4.0, 8.0),
    "sigma_corridor":     (50.0, 50.0, 50.0, 50.0),
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
