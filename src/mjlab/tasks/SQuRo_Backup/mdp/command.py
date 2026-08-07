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


# 命令系统 — 1D 命令 [time_scale λ]
# λ = 参考轨迹时间缩放 (放慢倍数): λ=1.0 原始 fast1 速度 (~1s 复位),
#     λ=1.5 放慢 1.5 倍 (学习起点); 课程从慢到快, 由 curriculums 采样
class BackupCommand(CommandTerm):
    cfg: "BackupCommandCfg"

    def __init__(self, cfg: "BackupCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.command_tensor = torch.zeros(self.num_envs, 1, device=self.device)

        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(env_ids)
        t_range = self.cfg.resampling_time_range
        self.time_left[env_ids] = torch.rand(len(env_ids), device=self.device) * (t_range[1] - t_range[0]) + t_range[0]

    @property
    def command(self) -> torch.Tensor:
        return self.command_tensor

    # 按课程采样 time_scale λ (episode 内固定)
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        lam = get_curriculum_time_scale(self._env.common_step_counter, n, self.device)
        self.command_tensor[env_ids, 0] = lam

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
