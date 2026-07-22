"""SQuRo绕杆任务第一阶段 — 课程调度

3阶段课程:
  阶段1 (0~500 iter): 直行 — ω=0, v逐步提升, 脊柱奖励关闭
  阶段2 (500~1500 iter): 转弯引入 — ω范围渐进扩展, 脊柱奖励逐步开启
  阶段3 (1500+ iter): 全范围 — 随机[v, ω]混合, 全奖励权重
"""

from __future__ import annotations
from typing import Dict, Any
from .command import STAGE1_END, STAGE2_END


class RewardWeightCurriculum:
    """奖励权重课程调度器"""

    def __init__(self):
        self.weight_stages = {
            0: {  # 阶段1: 直行优先，关闭转弯
                "track_vel": 2.0,
                "track_omega": 0.0,
                "spine_turn": 0.0,
                "stability": 0.5,
                "energy": 0.01,
                "smoothness": 0.1,
            },
            STAGE1_END * 24: {  # 阶段2: 引入转弯
                "track_vel": 1.0,
                "track_omega": 1.0,
                "spine_turn": 0.3,
                "stability": 0.3,
                "energy": 0.01,
                "smoothness": 0.2,
            },
            STAGE2_END * 24: {  # 阶段3: 全权重
                "track_vel": 1.0,
                "track_omega": 1.0,
                "spine_turn": 1.0,
                "stability": 0.3,
                "energy": 0.01,
                "smoothness": 0.5,
            },
        }

    def get_reward_weights(self, current_step: int) -> Dict[str, float]:
        """根据当前步数返回奖励权重"""
        weights = self.weight_stages[0]
        for step_threshold in sorted(self.weight_stages.keys()):
            if current_step >= step_threshold:
                weights = self.weight_stages[step_threshold]
        return weights

    def get_current_stage_info(self, current_step: int) -> Dict[str, Any]:
        weights = self.get_reward_weights(current_step)
        current_stage = 0
        for step_threshold in sorted(self.weight_stages.keys()):
            if current_step >= step_threshold:
                current_stage = step_threshold
        return {
            "current_stage": current_stage,
            "reward_weights": weights,
            "current_step": current_step,
            "total_stages": len(self.weight_stages),
        }


# 全局单例
reward_weight_curriculum = RewardWeightCurriculum()


def get_curriculum_reward_weight(env, reward_name: str) -> float:
    """根据当前训练步数查询奖励权重"""
    current_weights = reward_weight_curriculum.get_reward_weights(env.common_step_counter)
    return current_weights.get(reward_name, 1.0)
