from __future__ import annotations
import torch
from typing import TYPE_CHECKING, cast
from .command import BackupCommand
from .timing import STAND_CONFIRM_DURATION, STAND_VEL_MEAN_MAX

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv



# 检测是否成功站起
# 成功定义 = 在 P3 内保持"背部朝上 + 抬高"的站立几何达到 STAND_CONFIRM_DURATION,
# 且这段窗口内的**平均**关节速度不超过 STAND_VEL_MEAN_MAX。
# 加入速度条件是为了让"冲得快但抖动"直接失败: 里程奖励按达成时刻发放, 越早越赚,
# 于是策略会赶时间硬翻; 只在 P3 额外加平滑惩罚会被"早到"的收益盖过去, 写进成功判据则无法绕过。
# 用窗口均值而不是"连续达标时长": 均值骗不过去(窗口内任何一次剧烈抖动都会抬高它),
# 且没有占空比悬崖 —— 后者会让判据在某个人为的占空比之下变成永远不可达。
def check_stand_success(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast(BackupCommand, env.command_manager.get_term("backup_cmd"))
    hold, strict = command.stand_gate()
    in_p3 = command.phase == 2
    running = env.episode_length_buf > 0
    elapsed = getattr(env, "_stand_elapsed", None)
    if elapsed is None:
        elapsed = torch.zeros(env.num_envs, device=env.device)
    vel_integral = getattr(env, "_stand_vel_integral", None)
    if vel_integral is None:
        vel_integral = torch.zeros(env.num_envs, device=env.device)
    # 窗口只在 P3 且宽松几何成立时推进; 中断即 T 与 V 一起清零重来。
    active = hold & in_p3
    elapsed, vel_integral = BackupCommand._update_stand_window(
        elapsed, vel_integral, active, running, command.standing_metrics()[2], env.step_dt)
    env._stand_elapsed = elapsed  # type: ignore[attr-defined]
    env._stand_vel_integral = vel_integral  # type: ignore[attr-defined]
    # 时长按 duration/dt 步累加, float32 的逐步舍入随步数增长, 容差同步放大
    # (eps*150*4 ≈ 7e-5 秒, 仅为确认时长的 0.005%, 不改变实际确认秒数)。
    steps = max(1.0, abs(STAND_CONFIRM_DURATION) / max(abs(env.step_dt), 1e-9))
    eps = torch.finfo(elapsed.dtype).eps * steps * 4.0
    # T=0 时 V 也为 0, 相除得 0 而不是 NaN。
    mean_vel = vel_integral / elapsed.clamp_min(torch.finfo(elapsed.dtype).tiny)
    # 结算当步仍须在 P3 且满足严格几何, 窗口时长与窗口平均速度同时达标。
    confirmed = active & running & strict & (elapsed >= (STAND_CONFIRM_DURATION - eps)) & (mean_vel <= STAND_VEL_MEAN_MAX)
    # Progress/standing: 严格几何达标的瞬时占比(跨版本可比, 语义未变)。
    env.extras["log"]["Progress/standing"] = (strict & in_p3).float().mean().item()
    # 下面两项是标定用日志, 不参与判据: 分别回答"离 1.5 s 还有多远"和"离 3.5 rad/s 还有多远"。
    # 两者都只对"窗口已建立"的环境取条件均值, 否则无窗口时的 0 会把均值稀释成无意义的小数。
    window = active & running & (elapsed > 0.0)
    count = window.sum().clamp_min(1).to(elapsed.dtype)
    env.extras["log"]["Progress/stand_hold"] = elapsed.mean().item()
    env.extras["log"]["Progress/stand_mean_vel"] = (torch.where(window, mean_vel, torch.zeros_like(mean_vel)).sum() / count).item()
    return confirmed
