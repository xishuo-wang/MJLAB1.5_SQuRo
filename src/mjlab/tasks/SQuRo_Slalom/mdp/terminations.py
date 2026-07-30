from __future__ import annotations
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_FALLEN_UPRIGHTNESS_THRESHOLD = 0.2  # 跌倒判定阈值 (重力投影z分量)


def check_fallen(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot_entity = env.scene.entities["robot"]
    gravity_vec = robot_entity.data.projected_gravity_b
    uprightness = gravity_vec[:, 2]
    fallen = uprightness < _FALLEN_UPRIGHTNESS_THRESHOLD
    return fallen