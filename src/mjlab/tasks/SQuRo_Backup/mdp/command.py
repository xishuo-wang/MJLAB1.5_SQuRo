from __future__ import annotations
import torch
from dataclasses import dataclass, field
from mjlab.managers import CommandTermCfg
from typing import TYPE_CHECKING, Tuple
from mjlab.managers.command_manager import CommandTerm
from .curriculums import get_curriculum_time_scale

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


# 命令系统 — 6D 命令 [vel_x, height_f, height_h, gait_freq, curvature, time_scale]
# 前 5 维与 Slalom/Tunnel 对齐 (跌倒爬起中 vel/gait/curvature 占位, 高度命令与 height 奖励一致)
# 第 6 维 time_scale λ: 参考轨迹时间缩放 (放慢倍数), 驱动爬起快慢
#   λ=1.0 原始 fast1 速度 (~1s 复位), λ=2.0 放慢 2 倍 (学习起点); 由 curriculums 按课程采样
class BackupCommand(CommandTerm):
    cfg: "BackupCommandCfg"

    def __init__(self, cfg: "BackupCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.command_tensor = torch.zeros(self.num_envs, 6, device=self.device)
        self.vel_command = self.command_tensor[:, 0]
        self.height_f_command = self.command_tensor[:, 1]
        self.height_h_command = self.command_tensor[:, 2]
        self.gait_freq_command = self.command_tensor[:, 3]
        self.curvature_command = self.command_tensor[:, 4]
        self.time_scale_command = self.command_tensor[:, 5]

        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(env_ids)
        t_range = self.cfg.resampling_time_range
        self.time_left[env_ids] = torch.rand(len(env_ids), device=self.device) * (t_range[1] - t_range[0]) + t_range[0]

    @property
    def command(self) -> torch.Tensor:
        return self.command_tensor

    # 按课程采样 time_scale λ (episode 内固定); 其余字段与 Slalom/Tunnel 语义对齐
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        self.vel_command[env_ids] = 0.0                # 跌倒爬起无速度跟踪 (占位)
        self.height_f_command[env_ids] = 0.055         # 期望前体高度
        self.height_h_command[env_ids] = 0.055         # 期望后体高度
        self.gait_freq_command[env_ids] = 1.0          # 占位
        self.curvature_command[env_ids] = 0.0          # 无转向
        lam = get_curriculum_time_scale(self._env.common_step_counter, n, self.device)
        self.time_scale_command[env_ids] = lam         # 参考时间缩放

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
            self._resample_command(env_ids)
        return extras

    def _update_command(self) -> None:
        # episode 内固定, 无需定期重采样
        pass

    def _update_metrics(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        pass


@dataclass(kw_only=True)
class BackupCommandCfg(CommandTermCfg):
    asset_name: str = "robot"
    resampling_time_range: Tuple[float, float] = (1000.0, 1000.0)   # 不重采样 (episode 内固定)
    debug_vis: bool = False

    @dataclass
    class VizCfg:
        z_offset: float = 0.1
        scale: float = 1.0

    viz: VizCfg = field(default_factory=VizCfg)
    class_type: type[CommandTerm] = BackupCommand

    def build(self, env: "ManagerBasedRlEnv") -> CommandTerm:
        return self.class_type(self, env)
