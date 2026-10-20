from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from .curriculums import PHASE0_EPISODE_SECONDS
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



# 一阶段保持二十秒，二阶段使用配置中的完整回合时长。
def check_episode_timeout(env: ManagerBasedRlEnv) -> torch.Tensor:
    phase1 = env.command_manager._terms["slalom_cmd"].phase1_mask
    phase0_steps = min(env.max_episode_length, round(PHASE0_EPISODE_SECONDS / env.step_dt))
    limit = torch.where(phase1, env.max_episode_length, phase0_steps)
    return env.episode_length_buf >= limit
