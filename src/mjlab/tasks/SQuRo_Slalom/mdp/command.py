from __future__ import annotations
import math
import torch
from mjlab.entity import Entity
from mjlab.managers import CommandTermCfg
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple
from mjlab.managers.command_manager import CommandTerm
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


# 阶段阈值（iterations）
STAGE1_END = 1000
STAGE2_END = 3000


# 命令配置
FIXED_VEL = 0.1
FIXED_HEIGHT_F = 0.055
FIXED_HEIGHT_H = 0.055
FIXED_GAIT_FREQ = 1.0
CURVATURE_TARGET_MAX = 5.0


# 获取当前阶段
def get_current_stage(step_counter: int) -> int:
    iter_num = step_counter // 24
    if iter_num < STAGE1_END:
        return 1
    elif iter_num < STAGE2_END:
        return 2
    else:
        return 3


# 获取曲率采样范围
def get_curvature_range(stage: int, step_counter: int) -> Tuple[float, float]:
    if stage == 1:
        return (0.0, 0.0)  # 直行
    elif stage == 2:
        iter_num = step_counter // 24
        progress = (iter_num - STAGE1_END) / (STAGE2_END - STAGE1_END)
        kappa_max = 0.5 + progress * (CURVATURE_TARGET_MAX - 0.5)
        return (-kappa_max, kappa_max)
    else:
        return (-CURVATURE_TARGET_MAX, CURVATURE_TARGET_MAX)


# 5D命令 [vel_x, height_f, height_h, gait_freq, curvature]
class SlalomCommand(CommandTerm):
    cfg: "SlalomCommandCfg"
    def __init__(self, cfg: "SlalomCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.asset_name]

        # 命令张量: [vel_x, height_f, height_h, gait_freq, curvature]
        self.command_tensor = torch.zeros(self.num_envs, 5, device=self.device)
        self.vel_command = self.command_tensor[:, 0]
        self.height_f_command = self.command_tensor[:, 1]
        self.height_h_command = self.command_tensor[:, 2]
        self.gait_freq_command = self.command_tensor[:, 3]
        self.curvature_command = self.command_tensor[:, 4]

        # 固定值配置（优先级高于采样）
        self.fixed_velocity = cfg.fixed_velocity
        self.fixed_height_f = cfg.fixed_height_f
        self.fixed_height_h = cfg.fixed_height_h
        self.fixed_gait_freq = cfg.fixed_gait_freq
        self.fixed_curvature = cfg.fixed_curvature

        # 可视化：episode 起始位置 + 是否已记录
        self._start_positions = torch.zeros(self.num_envs, 3, device=self.device)
        self._start_recorded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 初始化
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(env_ids)
        self._resample_curvature(env_ids)

        t_range = self.cfg.resampling_time_range
        self.time_left[env_ids] = torch.rand(len(env_ids), device=self.device) * (t_range[1] - t_range[0]) + t_range[0]

    @property
    def command(self) -> torch.Tensor:
        return self.command_tensor

    def _get_velocity(self, n: int) -> torch.Tensor:
        if self.fixed_velocity is not None:
            return torch.full((n,), float(self.fixed_velocity), device=self.device)
        return torch.full((n,), FIXED_VEL, device=self.device)

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
        return torch.full((n,), FIXED_GAIT_FREQ, device=self.device)

    def _get_curvature(self, n: int, step_counter: int) -> torch.Tensor:
        if self.fixed_curvature is not None:
            return torch.full((n,), float(self.fixed_curvature), device=self.device)
        stage = get_current_stage(step_counter)
        kappa_range = get_curvature_range(stage, step_counter)
        return torch.rand(n, device=self.device) * (kappa_range[1] - kappa_range[0]) + kappa_range[0]

    # 仅在 reset 时调用，每个 episode 固定曲率不变
    def _resample_curvature(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        current_step = self._env.common_step_counter
        self.curvature_command[env_ids] = self._get_curvature(n, current_step)
        # 标记需重新记录起始位置（在 _debug_vis_impl 首次调用时延迟读取）
        self._start_recorded[env_ids] = False

    # 定期重采样：仅更新固定值，曲率不参与（由 reset 单独触发）
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        self.vel_command[env_ids] = self._get_velocity(n)
        self.height_f_command[env_ids] = self._get_height_f(n)
        self.height_h_command[env_ids] = self._get_height_h(n)
        self.gait_freq_command[env_ids] = self._get_gait_freq(n)

    # 重置时额外采样曲率 + 记录轨迹起始位置
    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
            self._resample_curvature(env_ids)
        return extras

    def _update_command(self) -> None:
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

        # 延迟记录起始位置（reset后entity data已同步，避免读到摔倒时的旧位置）
        if not self._start_recorded[batch]:
            self._start_positions[batch] = self.robot.data.root_link_pos_w[batch]
            self._start_recorded[batch] = True

        # 期望轨迹
        self._draw_path(visualizer, batch, self.cfg.viz.z_offset)

    # 绘制期望轨迹：直行=射线, 转弯=圆弧
    def _draw_path(self, visualizer: "DebugVisualizer", batch: int, z_offset: float) -> None:
        curvature = self.curvature_command[batch].item()
        vel = self.vel_command[batch].item()
        start = self._start_positions[batch].cpu().numpy()
        heading_0 = 0.0

        n_pts = 50
        t_max = 20.0  # 显示未来5秒的轨迹
        dt_path = t_max / n_pts
        radius = 0.008  # 轨迹点小球半径

        for i in range(n_pts + 1):
            t_i = i * dt_path
            if abs(curvature) < 1e-6:
                # 直行
                x_i = start[0] + vel * t_i * math.cos(heading_0)
                y_i = start[1] + vel * t_i * math.sin(heading_0)
            else:
                # 圆弧: R=1/κ, ω=κ·v
                R = 1.0 / curvature
                omega = curvature * vel
                dtheta = omega * t_i
                x_i = start[0] + R * (math.sin(heading_0 + dtheta) - math.sin(heading_0))
                y_i = start[1] - R * (math.cos(heading_0 + dtheta) - math.cos(heading_0))

            import numpy as np
            pt = np.array([x_i, y_i, start[2] + z_offset])
            visualizer.add_sphere(
                center=pt, radius=radius,
                color=(1.0, 0.6, 0.0, 0.6),  # 橙色半透明
                label=f"path_{i}",
            )


@dataclass(kw_only=True)
class SlalomCommandCfg(CommandTermCfg):
    asset_name: str = "robot"
    resampling_time_range: Tuple[float, float] = (20.0, 30.0)
    debug_vis: bool = False

    # 固定值（None=使用课程采样，设值可覆盖）
    fixed_velocity: Optional[float] = None
    fixed_height_f: Optional[float] = None
    fixed_height_h: Optional[float] = None
    fixed_gait_freq: Optional[float] = None
    fixed_curvature: Optional[float] = None

    @dataclass
    class VizCfg:
        z_offset: float = 0.1
        scale: float = 1.0

    viz: VizCfg = field(default_factory=VizCfg)
    class_type: type[CommandTerm] = SlalomCommand

    def build(self, env: "ManagerBasedRlEnv") -> CommandTerm:
        return self.class_type(self, env)
