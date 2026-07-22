from __future__ import annotations
import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# 终止条件
def check_fallen(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取机器人实体
    robot_entity = env.scene.entities["robot"]
    
    # 使用重力向量检测姿态
    gravity_vec = robot_entity.data.projected_gravity_b
    uprightness = gravity_vec[:, 2]  
    
    # 跌倒判定：姿态太倾斜
    fallen = uprightness < 0.2
    
    return fallen


# 完成任务：达到目标距离
def check_reach_goal(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取机器人实体
    robot_entity = env.scene.entities["robot"]
    base_pos_x = robot_entity.data.root_link_pos_w[:, 0]
    reached_goal = base_pos_x >= 1.2
    
    return reached_goal