from __future__ import annotations
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# 终止条件
def check_fallen(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot_entity = env.scene.entities["robot"]
    # 使用重力向量检测姿态
    gravity_vec = robot_entity.data.projected_gravity_b
    uprightness = gravity_vec[:, 2]  
    # 跌倒判定：姿态太倾斜
    fallen = uprightness < 0.2
    return fallen