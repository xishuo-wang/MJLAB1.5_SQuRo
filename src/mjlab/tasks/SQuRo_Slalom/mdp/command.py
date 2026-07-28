from __future__ import annotations
import math
import torch
from mjlab.entity import Entity
from dataclasses import dataclass, field
from mjlab.managers import CommandTermCfg
from typing import TYPE_CHECKING, Optional, Tuple
from mjlab.managers.command_manager import CommandTerm
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


# 阶段阈值（iterations）— 无纯直行阶段，从 iter 0 直接开始曲率线性增长
STAGE2_END = 2000


# 命令配置
FIXED_VEL = 0.1
FIXED_HEIGHT_F = 0.055
FIXED_HEIGHT_H = 0.055
FIXED_GAIT_FREQ = 1.0
CURVATURE_TARGET_MAX = 20.0
VEL_MIN = 0.25


# 获取当前阶段（2=曲率线性增长, 3=全范围）
def get_current_stage(step_counter: int) -> int:
    iter_num = step_counter // 24
    if iter_num < STAGE2_END:
        return 2
    else:
        return 3


# 获取曲率采样范围 — 从 iter 0 开始线性增长至 CURVATURE_TARGET_MAX
def get_curvature_range(stage: int, step_counter: int) -> Tuple[float, float]:
    if stage == 2:
        iter_num = step_counter // 24
        progress = iter_num / STAGE2_END  # 0 → 1
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

    # 绕杆模式 (由 curriculums.get_training_phase 自动控制)
    @property
    def slalom_mode_active(self) -> bool:
        from .curriculums import get_training_phase
        return get_training_phase(self._env.common_step_counter) == 1

    # 当前杆间距 (由 curriculums.get_curriculum_pole_spacing 自动控制)
    @property
    def active_pole_spacing(self) -> float:
        from .curriculums import get_curriculum_pole_spacing
        return get_curriculum_pole_spacing(self._env.common_step_counter)

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
        from .curriculums import get_training_phase
        from .pole import update_pole_visibility
        n = len(env_ids)
        current_step = self._env.common_step_counter
        phase = get_training_phase(current_step)

        # 根据阶段自动切换杆可见性 (Phase 0 透明, Phase 1 正常)
        update_pole_visibility(self._env, phase)

        if phase == 0:
            # Phase 0: 转弯基元 — 采样曲率, 速度按曲率缩放
            self.curvature_command[env_ids] = self._get_curvature(n, current_step)
            base_vel = float(self.fixed_velocity) if self.fixed_velocity is not None else FIXED_VEL
            scale = 1.0 - (1.0 - VEL_MIN) * self.curvature_command[env_ids].abs() / CURVATURE_TARGET_MAX
            self.vel_command[env_ids] = base_vel * scale
        else:
            # Phase 1: 绕杆训练 — 曲率取 ±max (脊柱参考用), 速度减半
            sign = torch.where(
                torch.rand(n, device=self.device) > 0.5,
                torch.tensor(1.0, device=self.device),
                torch.tensor(-1.0, device=self.device),
            )
            self.curvature_command[env_ids] = sign * CURVATURE_TARGET_MAX
            base_vel = float(self.fixed_velocity) if self.fixed_velocity is not None else FIXED_VEL
            self.vel_command[env_ids] = torch.full((n,), base_vel * 0.5, device=self.device)

        self._start_recorded[env_ids] = False

    # 定期重采样：仅更新固定值（速度由 _resample_curvature 按曲率缩放）
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
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

        # 根据阶段选择可视化路径
        if self.slalom_mode_active:
            self._draw_slalom_path(visualizer, batch, self.cfg.viz.z_offset)
        else:
            self._draw_arc_path(visualizer, batch, self.cfg.viz.z_offset)

    # 绘制转弯基元轨迹：直行=射线, 转弯=圆弧
    def _draw_arc_path(self, visualizer: "DebugVisualizer", batch: int, z_offset: float) -> None:
        curvature = self.curvature_command[batch].item()
        vel = self.vel_command[batch].item()
        start = self._start_positions[batch].cpu().numpy()
        heading_0 = 0.0

        n_pts = 50
        t_max = 20.0
        dt_path = t_max / n_pts
        radius = 0.008

        import numpy as np
        for i in range(n_pts + 1):
            t_i = i * dt_path
            if abs(curvature) < 1e-6:
                x_i = start[0] + vel * t_i * math.cos(heading_0)
                y_i = start[1] + vel * t_i * math.sin(heading_0)
            else:
                R = 1.0 / curvature
                omega = curvature * vel
                dtheta = omega * t_i
                x_i = start[0] + R * (math.sin(heading_0 + dtheta) - math.sin(heading_0))
                y_i = start[1] - R * (math.cos(heading_0 + dtheta) - math.cos(heading_0))
            pt = np.array([x_i, y_i, start[2] + z_offset])
            visualizer.add_sphere(
                center=pt, radius=radius,
                color=(1.0, 0.6, 0.0, 0.6),
                label=f"arc_{i}",
            )

    # 绘制绕杆轨迹：圆弧拼接路径
    def _draw_slalom_path(self, visualizer: "DebugVisualizer", batch: int, z_offset: float) -> None:
        import numpy as np
        from .path import _generate_slalom_lut_one_period

        spacing = self.active_pole_spacing
        start = self._start_positions[batch].cpu().numpy()

        _, xs, ys, _ = _generate_slalom_lut_one_period(spacing, n_arc_pts=10)
        period_len = np.array(xs[-1])  # 一个周期的 X 跨度 = 2*spacing
        n_periods = 3

        radius = 0.006   # 略小于弧线的轨迹点
        for k in range(n_periods):
            offset_x = k * period_len
            for i in range(len(xs)):
                pt = np.array([
                    start[0] + xs[i] + offset_x,
                    start[1] + ys[i],
                    start[2] + z_offset,
                ])
                # 交替颜色区分周期
                color = (0.2, 0.7, 1.0, 0.5) if k % 2 == 0 else (1.0, 0.5, 0.2, 0.5)
                visualizer.add_sphere(
                    center=pt, radius=radius,
                    color=color,
                    label=f"slalom_{k}_{i}",
                )

        # 杆位置标记（红色小球）
        num_poles = int(n_periods * 2) + 1
        for pi in range(num_poles):
            px = start[0] + pi * spacing
            py = start[1] - 0.1  # Y = -Rmin
            pt = np.array([px, py, start[2] + z_offset + 0.05])
            visualizer.add_sphere(
                center=pt, radius=0.007,
                color=(0.9, 0.2, 0.2, 0.8),
                label=f"pole_{pi}",
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
