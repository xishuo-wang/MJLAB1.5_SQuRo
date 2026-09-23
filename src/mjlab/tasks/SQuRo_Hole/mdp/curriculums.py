from __future__ import annotations
from typing import Any, Dict

from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


# 奖励权重课程 (4 段, 阈值单位 = 全局步数; 96 步以上全是第 4 段)
# 键名与 SQuRo_Hole_env_cfg 的奖励项名一致
class RewardWeightCurriculum:
    def __init__(self):
        self.weight_stages = {
            0: {
                "mimic_pos": 10.0,
                "mimic_vel": 5.0,
                "velocity": 5.0,
                "height": 2.5,
                "foot_clearance": 1.0,
                "reached": 1.0,
                "orientation": 2.0,
                "angle": 1.0,
                "smoothness": 0.1,
                "body_contact": 0.0,
                "mimic_pos_sigma": 5.0,
                "mimic_vel_sigma": 0.1,
                "height_sigma": 500.0,
            },
            1000 * 24: {
                "mimic_pos": 10.0,
                "mimic_vel": 5.0,
                "velocity": 5.0,
                "height": 2.5,
                "foot_clearance": 1.0,
                "reached": 1.0,
                "orientation": 2.0,
                "angle": 1.0,
                "smoothness": 0.2,
                "body_contact": 0.0,
                "mimic_pos_sigma": 10.0,
                "mimic_vel_sigma": 0.1,
                "height_sigma": 1000.0,
            },
            2000 * 24: {
                "mimic_pos": 10.0,
                "mimic_vel": 5.0,
                "velocity": 5.0,
                "height": 5.0,
                "foot_clearance": 1.0,
                "reached": 1.0,
                "orientation": 4.0,
                "angle": 1.0,
                "smoothness": 0.5,
                "body_contact": 1.0,
                "mimic_pos_sigma": 10.0,
                "mimic_vel_sigma": 0.1,
                "height_sigma": 1000.0,
            },
            3000 * 24: {
                "mimic_pos": 10.0,
                "mimic_vel": 5.0,
                "velocity": 10.0,
                "height": 10.0,
                "foot_clearance": 1.0,
                "reached": 1.0,
                "orientation": 4.0,
                "angle": 1.0,
                "smoothness": 1.0,
                "body_contact": 1.0,
                "mimic_pos_sigma": 5.0,
                "mimic_vel_sigma": 0.1,
                "height_sigma": 1000.0,
            },
        }

    # 取当前步数对应的权重表
    def get_reward_weights(self, current_step: int) -> Dict[str, float]:
        weights = self.weight_stages[0]
        for step_threshold in sorted(self.weight_stages.keys()):
            if current_step >= step_threshold:
                weights = self.weight_stages[step_threshold]
        return weights

    def add_weight_stage(self, step_threshold: int, weights: Dict[str, float]) -> None:
        self.weight_stages[step_threshold] = weights

    # 取当前课程阶段信息
    def get_current_stage_info(self, current_step: int) -> Dict[str, Any]:
        current_stage = 0
        for step_threshold in sorted(self.weight_stages.keys()):
            if current_step >= step_threshold:
                current_stage = step_threshold
        return {
            "current_stage": current_stage,
            "reward_weights": self.get_reward_weights(current_step),
            "current_step": current_step,
            "total_stages": len(self.weight_stages),
        }


reward_weight_curriculum = RewardWeightCurriculum()


# 取当前步数的奖励权重
def get_curriculum_reward_weight(env, reward_name: str) -> float:
    current_weights = reward_weight_curriculum.get_reward_weights(env.common_step_counter)
    return current_weights.get(reward_name, 1.0)
