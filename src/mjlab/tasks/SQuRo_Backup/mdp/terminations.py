from __future__ import annotations
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# 终止条件: 跌倒爬起任务初始即跌倒, 因此仅用超时, 不做跌倒终止
def check_fallen(env: "ManagerBasedRlEnv") -> torch.Tensor:
    # 保留接口但默认不终止（爬起任务中"跌倒"是初始状态而非失败）
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
