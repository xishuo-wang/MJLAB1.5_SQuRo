from __future__ import annotations
import torch
from typing import TYPE_CHECKING, cast
from .command import BackupCommand
from .timing import STAND_CONFIRM_DECAY, STAND_CONFIRM_DURATION, STAND_MIN_HEIGHT, STAND_UPRIGHT_COS

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv



# 检测是否成功站起
# 成功定义 = "背部朝上 + 抬高 + 关节基本静止" 三者同时成立并维持 STAND_CONFIRM_DURATION。
# 加入速度条件是为了让"冲得快但抖动"直接失败: 里程奖励按达成时刻发放, 越早越赚,
# 于是策略会赶时间硬翻; 只在 P3 额外加平滑惩罚会被"早到"的收益盖过去, 写进成功判据则无法绕过。
def check_stand_success(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast(BackupCommand, env.command_manager.get_term("backup_cmd"))
    enter, stay = command.stand_gate()
    in_p3 = command.phase == 2
    elapsed = getattr(env, "_stand_elapsed", None)
    if elapsed is None:
        elapsed = torch.zeros(env.num_envs, device=env.device)
    elapsed, confirmed = BackupCommand._update_stand_confirmation(
        elapsed, enter & in_p3, stay & in_p3, env.episode_length_buf > 0, env.step_dt,
        STAND_CONFIRM_DURATION, STAND_CONFIRM_DECAY)
    env._stand_elapsed = elapsed  # type: ignore[attr-defined]
    # Progress/standing 语义已改为"严格达标瞬时占比"(原先只含几何, 现在含速度条件)。
    env.extras["log"]["Progress/standing"] = enter.float().mean().item()
    # 站立几何成立(不含速度条件)时的关节速度 RMS 均值: 用来标定 STAND_VEL_RMS_ENTER 的松紧, 不参与判据。
    u_floor, h_floor, vel_rms = command.standing_metrics()
    geometric = in_p3 & torch.isfinite(vel_rms) & (u_floor > STAND_UPRIGHT_COS) & (h_floor > STAND_MIN_HEIGHT)
    count = geometric.sum().clamp_min(1).to(vel_rms.dtype)
    env.extras["log"]["Progress/stand_vel"] = (torch.where(geometric, vel_rms, torch.zeros_like(vel_rms)).sum() / count).item()
    # 已累计的连续达标时长均值 (非 P3 环境恒为 0): 用来区分"速度阈值"还是"1.5 s 保持时长"是瓶颈。
    # 若长期停在 0.2~0.4 s, 说明连不成片 -> 该放松保持时长或加大侵蚀容忍;
    # 若接近 1.5 s 却仍不结算, 说明几何/速度门限才是瓶颈 -> 该放松 STAND_VEL_RMS_ENTER。
    env.extras["log"]["Progress/stand_hold"] = elapsed.mean().item()
    return confirmed
