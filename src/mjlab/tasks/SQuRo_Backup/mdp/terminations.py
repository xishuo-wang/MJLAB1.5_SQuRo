from __future__ import annotations
import torch
from typing import TYPE_CHECKING, cast
from .command import BackupCommand
from .timing import STAND_CONFIRM_DURATION

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def check_fallen(env: "ManagerBasedRlEnv") -> torch.Tensor:
    # 爬起任务中"跌倒"是初始状态而非失败, 保留接口不终止
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


# P3 中两段分别正置且高度达标，连续 STAND_CONFIRM_DURATION 实际秒后结束。
# 该终止在 env_cfg 中标记为 time_out=True: 提前结束会截断后续密集奖励,
# 按真终止处理会把"站起来"算成亏分, 标记为截断后价值估计会 bootstrap 到后续状态。
def check_stand_success(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast(BackupCommand, env.command_manager.get_term("backup_cmd"))
    standing, _ = command.standing_state()
    candidate = standing & (command.phase == 2)
    elapsed = getattr(env, "_stand_elapsed", None)
    if elapsed is None:
        elapsed = torch.zeros(env.num_envs, device=env.device)
    elapsed, confirmed = BackupCommand._update_confirmation(
        elapsed, candidate, env.episode_length_buf > 0, env.step_dt, STAND_CONFIRM_DURATION
    )
    env._stand_elapsed = elapsed  # type: ignore[attr-defined]
    env.extras["log"]["Data/stand_candidate"] = candidate.float().mean().item()
    env.extras["log"]["Data/stand_confirm_elapsed"] = elapsed.mean().item()
    return confirmed
