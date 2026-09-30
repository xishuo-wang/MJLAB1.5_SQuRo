from __future__ import annotations
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from typing import TYPE_CHECKING, Dict, List, Any
from typing import Callable, Optional

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


# 奖励权重课程
class RewardWeightCurriculum:
    def __init__(self):
        self.weight_stages = {
            0: {  
                "mimic_pos": 10.0,
                "mimic_vel": 5.0,
                "vel": 5.0,
                "height": 2.5,
                "foot_clearance": 1.0,
                "reached": 1.0,
                "orientation": 2.0,
                "angle": 1.0,
                "smoothness": 0.1,
                "body_contact": 0,

                "mimic_pos_sigma": 5.0,
                "mimic_vel_sigma": 0.1,
                "height_sigma": 500,

                "enable_holes": False,
            },

            1000 * 24: {  
                "mimic_pos": 10.0,
                "mimic_vel": 5.0,
                "vel": 5.0,
                "height": 2.5,
                "foot_clearance": 1.0,
                "reached": 1.0,
                "orientation": 2.0,
                "angle": 1.0,
                "smoothness": 0.2,
                "body_contact": 0,

                "mimic_pos_sigma": 10.0,
                "mimic_vel_sigma": 0.1,
                "height_sigma": 1000,

                "enable_holes": False,
            },

            2000 * 24: {  
                "mimic_pos": 10.0,
                "mimic_vel": 5.0,
                "vel": 5.0,
                "height": 5.0,
                "reached": 1.0,
                "foot_clearance": 1.0,
                "orientation": 4.0,
                "angle": 1.0,
                "smoothness": 0.5,
                "body_contact": 1,

                "mimic_pos_sigma": 10.0,
                "mimic_vel_sigma": 0.1,
                "height_sigma": 1000,

                "enable_holes": False,
            },

            3000 * 24: {  
                "mimic_pos": 10.0,
                "mimic_vel": 5.0,
                "vel": 10.0,
                "height": 10.0,
                "foot_clearance": 1.0,
                "reached": 1.0,
                "orientation": 4.0,
                "angle": 1.0,
                "smoothness": 1,
                "body_contact": 1,

                "mimic_pos_sigma": 5.0,
                "mimic_vel_sigma": 0.1,
                "height_sigma": 1000,
                
                "enable_holes": True,  # 启用障碍物碰撞
            },
        }
        self._holes_enabled = False

    # 获取奖励权重
    def get_reward_weights(self, current_step: int) -> Dict[str, float]:
        weights = self.weight_stages[0]  # 默认第一阶段权重

        # 找到当前阶段对应的权重配置
        for step_threshold in sorted(self.weight_stages.keys()):
            if current_step >= step_threshold:
                weights = self.weight_stages[step_threshold]
        
        return weights
    
    # 添加课程阶段
    def add_weight_stage(self, step_threshold: int, weights: Dict[str, float]):
        self.weight_stages[step_threshold] = weights
    
    # 获取当前课程阶段信息
    def get_current_stage_info(self, current_step: int) -> Dict[str, Any]:
        weights = self.get_reward_weights(current_step)
        current_stage = 0
        
        # 找到当前阶段
        for step_threshold in sorted(self.weight_stages.keys()):
            if current_step >= step_threshold:
                current_stage = step_threshold
        
        return {
            "current_stage": current_stage,
            "reward_weights": weights,
            "current_step": current_step,
            "total_stages": len(self.weight_stages)
        }
    
    def should_enable_holes(self, current_step: int) -> bool:
        weights = self.get_reward_weights(current_step)
        enable_value = weights.get("enable_holes", 0.0)
        return bool(enable_value) 
    
    # 更新障碍物状态（在环境 step 中调用）
    def update_holes(self, env, current_step: int):
        """根据课程阶段更新障碍物碰撞状态"""
        should_enable = self.should_enable_holes(current_step)
        
        if should_enable and not self._holes_enabled:
            # 启用障碍物碰撞
            hole_names = ["Hole1", "Hole2", "Hole3"]
            for hole_name in hole_names:
                if hole_name in env.scene.entities:
                    hole = env.scene.entities[hole_name]
                    if hasattr(hole, 'enable_collision'):
                        hole.enable_collision()
                        print(f"[Curriculum] Enabled collision for {hole_name} at step {current_step}")
            self._holes_enabled = True
        elif not should_enable and self._holes_enabled:
            # 禁用障碍物碰撞（如果需要）
            hole_names = ["Hole1", "Hole2", "Hole3"]
            for hole_name in hole_names:
                if hole_name in env.scene.entities:
                    hole = env.scene.entities[hole_name]
                    if hasattr(hole, 'disable_collision'):
                        hole.disable_collision()
            self._holes_enabled = False

# 奖励权重实例
reward_weight_curriculum = RewardWeightCurriculum()

# 获取当前步数的奖励权重
def get_curriculum_reward_weight(env, reward_name: str) -> float:
    current_weights = reward_weight_curriculum.get_reward_weights(env.common_step_counter)
    return current_weights.get(reward_name, 1.0)

# 更新障碍物状态
def update_curriculum_holes(env) -> None:
    current_step = env.common_step_counter
    reward_weight_curriculum.update_holes(env, current_step)