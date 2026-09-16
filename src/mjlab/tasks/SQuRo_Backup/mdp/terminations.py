from __future__ import annotations
import torch
from typing import TYPE_CHECKING, cast
from .command import BackupCommand
from .timing import STAND_CONFIRM_DURATION

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv



# 检测是否成功站起
def check_stand_success(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast(BackupCommand, env.command_manager.get_term("backup_cmd"))
    standing, _ = command.standing_state()
    candidate = standing & (command.phase == 2)
    elapsed = getattr(env, "_stand_elapsed", None)
    if elapsed is None:
        elapsed = torch.zeros(env.num_envs, device=env.device)
    elapsed, confirmed = BackupCommand._update_confirmation(elapsed, candidate, env.episode_length_buf > 0, env.step_dt, STAND_CONFIRM_DURATION)
    env._stand_elapsed = elapsed  # type: ignore[attr-defined]
    env.extras["log"]["Data/stand_candidate"] = candidate.float().mean().item()
    env.extras["log"]["Data/stand_confirm_elapsed"] = elapsed.mean().item()
    return confirmed
