from __future__ import annotations
import torch
import numpy as np
from mjlab.entity import Entity
from dataclasses import dataclass, field
from mjlab.managers import CommandTermCfg
from typing import TYPE_CHECKING, Optional, Tuple
from mjlab.managers.command_manager import CommandTerm
from .path import (
    FRONT_DOWN,
    FRONT_UP,
    REAR_DOWN,
    REAR_UP,
    HEIGHT_NORMAL,
    HEIGHT_HOLE,
    compute_point_height,
)
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


# 命令配置
BASE_VEL = 0.1            # 基础速度 (m/s)
FIXED_HEIGHT_F = 0.06     # 前肢高度命令 (= HEIGHT_NORMAL)
FIXED_HEIGHT_H = 0.06     # 后肢高度命令 (= HEIGHT_NORMAL)
GAIT_FREQ = 1.0           # 步频 (Hz)


# 5D 命令 [vel_x, height_f, height_h, gait_freq, curvature]
# Tunnel 为直行钻洞: 曲率固定为 0, 高度/速度/步频固定
class TunnelCommand(CommandTerm):
    cfg: "TunnelCommandCfg"

    def __init__(self, cfg: "TunnelCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.asset_name]

        self.command_tensor = torch.zeros(self.num_envs, 5, device=self.device)
        self.vel_command = self.command_tensor[:, 0]
        self.height_f_command = self.command_tensor[:, 1]
        self.height_h_command = self.command_tensor[:, 2]
        self.gait_freq_command = self.command_tensor[:, 3]
        self.curvature_command = self.command_tensor[:, 4]

        # 固定值配置 (优先级高于默认)
        self.fixed_velocity = cfg.fixed_velocity
        self.fixed_height_f = cfg.fixed_height_f
        self.fixed_height_h = cfg.fixed_height_h
        self.fixed_gait_freq = cfg.fixed_gait_freq

        self._start_positions = torch.zeros(self.num_envs, 3, device=self.device)
        self._start_recorded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(env_ids)
        t_range = self.cfg.resampling_time_range
        self.time_left[env_ids] = torch.rand(len(env_ids), device=self.device) * (t_range[1] - t_range[0]) + t_range[0]

    @property
    def command(self) -> torch.Tensor:
        return self.command_tensor

    def _get_velocity(self, n: int) -> torch.Tensor:
        if self.fixed_velocity is not None:
            return torch.full((n,), float(self.fixed_velocity), device=self.device)
        return torch.full((n,), BASE_VEL, device=self.device)

    def _get_height_f(self, n: int) -> torch.Tensor:
        if self.fixed_height_f is not None:
            return torch.full((n,), float(self.fixed_height_f), device=self.device)
        return torch.full((n,), FIXED_HEIGHT_F, device=self.device)

    def _get_height_h(self, n: int) -> torch.Tensor:
        if self.fixed_height_h is not None:
            return torch.full((n,), float(self.fixed_height_h), device=self.device)
        return torch.full((n,), FIXED_HEIGHT_H, device=self.device)

    def _get_gait_freq(self, n: int) -> torch.Tensor:
        if self.fixed_gait_freq is not None:
            return torch.full((n,), float(self.fixed_gait_freq), device=self.device)
        return torch.full((n,), GAIT_FREQ, device=self.device)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        self.vel_command[env_ids] = self._get_velocity(n)
        self.height_f_command[env_ids] = self._get_height_f(n)
        self.height_h_command[env_ids] = self._get_height_h(n)
        self.gait_freq_command[env_ids] = self._get_gait_freq(n)
        self.curvature_command[env_ids] = 0.0   # 曲率固定 0 (直行)
        self._start_recorded[env_ids] = False

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
            self._resample_command(env_ids)
        return extras

    def _update_command(self) -> None:
        # Tunnel 命令全固定, 仅按时间重采样 (仍是固定值)
        env_ids = (self.time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._resample_command(env_ids)
            t_range = self.cfg.resampling_time_range
            self.time_left[env_ids] = torch.rand(len(env_ids), device=self.device) * (t_range[1] - t_range[0]) + t_range[0]
        self.time_left -= self._env.step_dt

    def _update_metrics(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        if not self.cfg.debug_vis:
            return
        batch = visualizer.env_idx
        if batch >= self.num_envs:
            return
        if torch.norm(self.robot.data.root_link_pos_w[batch]) < 1e-6:
            return
        if not self._start_recorded[batch]:
            self._start_positions[batch] = self.robot.data.root_link_pos_w[batch]
            self._start_recorded[batch] = True
        z_off = self.cfg.viz.z_offset
        start_x = float(self._start_positions[batch, 0])
        # 绘制前肢中心 / 后肢中心期望高度轨迹 (从起点起 0.5m)
        xs = np.linspace(start_x, start_x + 0.5, 60)
        z_front = compute_point_height(torch.tensor(xs, device=self.device), FRONT_DOWN, FRONT_UP).cpu().numpy()
        z_rear = compute_point_height(torch.tensor(xs, device=self.device), REAR_DOWN, REAR_UP).cpu().numpy()
        for x_i, z_f, z_r in zip(xs, z_front, z_rear):
            visualizer.add_sphere(center=np.array([x_i, 0.0, z_f + z_off]), radius=0.004,
                                  color=(0.2, 0.6, 1.0, 0.6), label=f"fref_{batch}_{x_i:.3f}")
            visualizer.add_sphere(center=np.array([x_i, 0.0, z_r + z_off]), radius=0.004,
                                  color=(1.0, 0.3, 0.3, 0.6), label=f"rref_{batch}_{x_i:.3f}")


@dataclass(kw_only=True)
class TunnelCommandCfg(CommandTermCfg):
    asset_name: str = "robot"
    resampling_time_range: Tuple[float, float] = (20.0, 30.0)
    debug_vis: bool = False
    fixed_velocity: Optional[float] = None
    fixed_height_f: Optional[float] = None
    fixed_height_h: Optional[float] = None
    fixed_gait_freq: Optional[float] = None

    @dataclass
    class VizCfg:
        z_offset: float = 0.1
        scale: float = 1.0

    viz: VizCfg = field(default_factory=VizCfg)
    class_type: type[CommandTerm] = TunnelCommand

    def build(self, env: "ManagerBasedRlEnv") -> CommandTerm:
        return self.class_type(self, env)
