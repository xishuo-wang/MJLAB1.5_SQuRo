from __future__ import annotations
import torch
from typing import TYPE_CHECKING, cast
from .command import BackupCommand
from .config import STAND_CONFIRM_DURATION, STAND_VEL_MEAN_MAX

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv



# 只读判定: 站立窗口是否已达标 (不写任何状态, 不推进窗口)。
def check_stand_success(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast(BackupCommand, env.command_manager.get_term("backup_cmd"))
    hold, strict = command.stand_gate()
    in_p3 = command.phase == 2
    elapsed = command._stand_elapsed
    vel_integral = command._stand_vel_integral
    steps = max(1.0, abs(STAND_CONFIRM_DURATION) / max(abs(env.step_dt), 1e-9))
    eps = torch.finfo(elapsed.dtype).eps * steps * 4.0
    mean_vel = vel_integral / elapsed.clamp_min(torch.finfo(elapsed.dtype).tiny)
    return hold & in_p3 & strict & (elapsed >= (STAND_CONFIRM_DURATION - eps)) & (mean_vel <= STAND_VEL_MEAN_MAX)
