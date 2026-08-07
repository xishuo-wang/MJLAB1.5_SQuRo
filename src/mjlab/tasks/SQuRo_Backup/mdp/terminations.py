from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from .indices import _MODEL_INDICES

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# 终止条件: 跌倒爬起任务初始即跌倒, 因此仅用超时, 不做跌倒终止
def check_fallen(env: "ManagerBasedRlEnv") -> torch.Tensor:
    # 保留接口但默认不终止（爬起任务中"跌倒"是初始状态而非失败）
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


# 站起成功终止 — 身体竖直且高度达标持续 0.5s, 视为复位完成 (episode length = 复位时间)
def check_stand_success(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset = env.scene.entities["robot"]
    up = asset.data.projected_gravity_b[:, 2]  # [N]
    body_pos_w = asset.data.body_link_pos_w
    h = 0.5 * (body_pos_w[:, _MODEL_INDICES.f_body_id, 2] + body_pos_w[:, _MODEL_INDICES.h_body_id, 2])
    standing = (up > 0.9) & (h > 0.05)
    buf = getattr(env, "_stand_steps", None)
    if buf is None:
        buf = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._stand_steps = buf  # type: ignore[attr-defined]
    buf[:] = torch.where(standing, buf + 1, torch.zeros_like(buf))
    return buf >= int(0.5 / env.step_dt)
