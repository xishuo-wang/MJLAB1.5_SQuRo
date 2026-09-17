from __future__ import annotations
import torch
from typing import TYPE_CHECKING, cast
from .command import BackupCommand
from .timing import STAND_CONFIRM_DURATION, STAND_VEL_MEAN_MAX

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv



# 检测是否成功站起：P3 内站立窗口的时长与窗口平均关节速度同时达标。设计与标定见技术细节 §7.2.1。
def check_stand_success(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast(BackupCommand, env.command_manager.get_term("backup_cmd"))
    hold, strict = command.stand_gate()
    in_p3 = command.phase == 2
    running = env.episode_length_buf > 0
    active = hold & in_p3
    elapsed = getattr(env, "_stand_elapsed", None)
    if elapsed is None:
        elapsed = torch.zeros(env.num_envs, device=env.device)
    vel_integral = getattr(env, "_stand_vel_integral", None)
    if vel_integral is None:
        vel_integral = torch.zeros(env.num_envs, device=env.device)
    elapsed, vel_integral = BackupCommand._update_stand_window(
        elapsed, vel_integral, active, running, command.standing_metrics()[2], env.step_dt)
    env._stand_elapsed = elapsed  # type: ignore[attr-defined]
    env._stand_vel_integral = vel_integral  # type: ignore[attr-defined]
    # float32 逐步累加的舍入容差（约 7e-5 秒，不改变实际确认秒数）。
    steps = max(1.0, abs(STAND_CONFIRM_DURATION) / max(abs(env.step_dt), 1e-9))
    eps = torch.finfo(elapsed.dtype).eps * steps * 4.0
    mean_vel = vel_integral / elapsed.clamp_min(torch.finfo(elapsed.dtype).tiny)
    # 结算当步仍须在 P3 且满足严格几何。
    confirmed = active & running & strict & (elapsed >= (STAND_CONFIRM_DURATION - eps)) & (mean_vel <= STAND_VEL_MEAN_MAX)
    env.extras["log"]["Progress/standing"] = (strict & in_p3).float().mean().item()
    # 两个标定用日志，不参与判据：窗口时长与窗口平均速度（只对已建立窗口的环境取条件均值）。
    window = active & running & (elapsed > 0.0)
    count = window.sum().clamp_min(1).to(elapsed.dtype)
    env.extras["log"]["Progress/stand_hold"] = elapsed.mean().item()
    env.extras["log"]["Progress/stand_mean_vel"] = (torch.where(window, mean_vel, torch.zeros_like(mean_vel)).sum() / count).item()
    return confirmed
