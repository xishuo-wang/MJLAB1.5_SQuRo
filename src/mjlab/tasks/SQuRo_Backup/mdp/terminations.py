from __future__ import annotations
import torch
from typing import TYPE_CHECKING, cast
from .command import BackupCommand
from .timing import STAND_CONFIRM_DURATION, STAND_VEL_MEAN_MAX

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# 本模块已不再提供终止项。
# "站稳"原先是一个 termination, 现改为非终止的"循环完成"事件: 完成即部分复位、重新开始
# 下一次翻正, 回合只在 episode_length_s 上限处截断。
# 原因见 docs/SQuRo_Backup_技术细节.md §7.2.7: 作为终止项时成功步 done=True, 价值要么截断
# 要么弃值, 这正是此前 value 发散、σ 回升的来源之一; 另外它会把"一回合一次翻正"写死。
# 站立窗口的累积与结算现在在 BackupCommand.stand_reward_and_pulse() 里, 由奖励项每步调用
# —— 必须在复位之前结算, 而 _update_command 位于 reward 之后。
#
# 本模块保留文件是为了兼容既有的 `from .mdp import terminations` 引用与工具脚本。


# 只读判定: 站立窗口是否已达标 (不写任何状态, 不推进窗口)。
# 训练侧的成功结算请用 command.stand_reward_and_pulse(); 此函数仅供诊断/回放读取。
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
