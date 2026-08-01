from __future__ import annotations
import math
import torch
import numpy as np
from mjlab.entity import Entity
from dataclasses import dataclass, field
from mjlab.managers import CommandTermCfg
from typing import TYPE_CHECKING, Optional, Tuple
from mjlab.managers.command_manager import CommandTerm
from .curriculums import (
    PHASE1_MID_ITER,
    _STEPS_PER_ITER,
    CURVATURE_TARGET,
    CURVATURE_TARGET_MAX,
    CURVATURE_MIN,
    POLE_SPACING,
    GAIT_FREQ_MIN,
    GAIT_FREQ_MAX,
    GAIT_FREQ_PHASE1,
    get_training_phase,
    get_pole_spacing_range,
    get_curriculum_pole_spacing,
)
from .pole import POLE_Y, POLE_HALF_HEIGHT, POLE_RADIUS, update_pole_visibility
from .path import _INIT_DIST, get_path_curvature, _generate_slalom_lut_one_period
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer



# =========================================================================================
# 命令配置
BASE_VEL = 0.1            # 基准速度 (对应 gait=1Hz 时的最大直行速度)
FIXED_HEIGHT_F = 0.055    # 默认前身体高度
FIXED_HEIGHT_H = 0.055    # 默认后身体高度
FIXED_GAIT_FREQ = 1.0     # 默认步频 (Hz)
VEL_MIN = 0.25            # 最大曲率时的速度比例（降至基准速度的 25%）



# 根据课程进度返回曲率的采样范围 [-κ_max, κ_max]
def get_curvature_range(step_counter: int) -> Tuple[float, float]:
    iter_num = step_counter // _STEPS_PER_ITER
    if iter_num < PHASE1_MID_ITER:
        progress = iter_num / PHASE1_MID_ITER
        kappa_max = CURVATURE_MIN + progress * (CURVATURE_TARGET_MAX - CURVATURE_MIN)
        return (-kappa_max, kappa_max)
    return (-CURVATURE_TARGET_MAX, CURVATURE_TARGET_MAX)



# =========================================================================================
# 5 维命令：[vel_x, height_f, height_h, gait_freq, curvature]
class SlalomCommand(CommandTerm):
    cfg: "SlalomCommandCfg"
    def __init__(self, cfg: "SlalomCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.asset_name]

        # 命令张量
        self.command_tensor = torch.zeros(self.num_envs, 5, device=self.device)
        self.vel_command = self.command_tensor[:, 0]
        self.height_f_command = self.command_tensor[:, 1]
        self.height_h_command = self.command_tensor[:, 2]
        self.gait_freq_command = self.command_tensor[:, 3]
        self.curvature_command = self.command_tensor[:, 4]

        # 固定值覆盖（若不为 None 则替代采样）
        self.fixed_velocity = cfg.fixed_velocity
        self.fixed_height_f = cfg.fixed_height_f
        self.fixed_height_h = cfg.fixed_height_h
        self.fixed_gait_freq = cfg.fixed_gait_freq
        self.fixed_curvature = cfg.fixed_curvature

        # 每 episode 共享值
        self._shared_pole_spacing = POLE_SPACING
        self._shared_gait_freq = FIXED_GAIT_FREQ

        # 可视化起始位置记录
        self._start_positions = torch.zeros(self.num_envs, 3, device=self.device)
        self._start_recorded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 初始化命令
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(env_ids)
        self._resample_curvature(env_ids)

        t_range = self.cfg.resampling_time_range
        self.time_left[env_ids] = torch.rand(len(env_ids), device=self.device) * (t_range[1] - t_range[0]) + t_range[0]


    @property
    def command(self) -> torch.Tensor:
        return self.command_tensor


    @property
    def slalom_mode_active(self) -> bool:
        return get_training_phase(self._env.common_step_counter) == 1


    @property
    def active_pole_spacing(self) -> float:
        override = getattr(self.cfg, "fixed_pole_spacing", None)
        if override is not None:
            return float(override)
        if self.slalom_mode_active:
            return self._shared_pole_spacing
        return get_curriculum_pole_spacing(self._env.common_step_counter)


    # 返回基准速度 BASE_VEL (1Hz 时的最大直行速度)
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
        return torch.full((n,), self._shared_gait_freq, device=self.device)


    def _get_curvature(self, n: int, step_counter: int) -> torch.Tensor:
        if self.fixed_curvature is not None:
            return torch.full((n,), float(self.fixed_curvature), device=self.device)
        kappa_range = get_curvature_range(step_counter)
        return torch.rand(n, device=self.device) * (kappa_range[1] - kappa_range[0]) + kappa_range[0]


    # Episode 开始时采样曲率、步频和速度
    def _resample_curvature(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        current_step = self._env.common_step_counter
        phase = get_training_phase(current_step)

        update_pole_visibility(self._env, phase)

        if phase == 0:
            # Phase 0: 转弯基元
            self.curvature_command[env_ids] = self._get_curvature(n, current_step)
            if self.fixed_gait_freq is not None:
                self._shared_gait_freq = float(self.fixed_gait_freq)
            else:
                self._shared_gait_freq = float(GAIT_FREQ_MIN + torch.rand(1).item() * (GAIT_FREQ_MAX - GAIT_FREQ_MIN))
            self.gait_freq_command[env_ids] = self._shared_gait_freq

            base_vel = float(self.fixed_velocity) if self.fixed_velocity is not None else BASE_VEL
            scale_kappa = 1.0 - (1.0 - VEL_MIN) * self.curvature_command[env_ids].abs() / CURVATURE_TARGET_MAX
            self.vel_command[env_ids] = base_vel * self.gait_freq_command[env_ids] * scale_kappa

        else:
            # Phase 1: 绕杆训练
            if self.cfg.fixed_pole_spacing is not None:
                self._shared_pole_spacing = float(self.cfg.fixed_pole_spacing)
            else:
                sp_range = get_pole_spacing_range(current_step)
                self._shared_pole_spacing = float(sp_range[0] + torch.rand(1).item() * (sp_range[1] - sp_range[0]))
            if self.fixed_gait_freq is not None:
                self._shared_gait_freq = float(self.fixed_gait_freq)
            else:
                self._shared_gait_freq = GAIT_FREQ_PHASE1
            self.gait_freq_command[env_ids] = self._shared_gait_freq
            self.curvature_command[env_ids] = torch.full((n,), -CURVATURE_TARGET, device=self.device)

            base_vel = float(self.fixed_velocity) if self.fixed_velocity is not None else BASE_VEL
            scale_kappa = 1.0 - (1.0 - VEL_MIN) * CURVATURE_TARGET / CURVATURE_TARGET_MAX
            self.vel_command[env_ids] = base_vel * self.gait_freq_command[env_ids] * scale_kappa

        self._start_recorded[env_ids] = False


    # 定期重采样高度和步频
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        self.height_f_command[env_ids] = self._get_height_f(n)
        self.height_h_command[env_ids] = self._get_height_h(n)
        self.gait_freq_command[env_ids] = self._get_gait_freq(n)


    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
            self._resample_curvature(env_ids)
        return extras

    
    def _update_command(self) -> None:
        if self.slalom_mode_active:
            # Phase 1: 仅每步动态更新曲率 (LUT 瞬时值)
            # 速度固定为 reset 时按弧段曲率计算的常量 (不再随 κ 动态缩放)
            # 原因: 变速命令 × 固定步幅步态表会导致参考轨迹超前于实际执行能力 (7/31 崩溃分析)
            kappa = get_path_curvature(self._env)
            self.curvature_command[:] = kappa

        env_ids = (self.time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._resample_command(env_ids)
            t_range = self.cfg.resampling_time_range
            self.time_left[env_ids] = torch.rand(len(env_ids), device=self.device) * (t_range[1] - t_range[0]) + t_range[0]
        self.time_left -= self._env.step_dt


    def _update_metrics(self) -> None:
        pass


    # 可视化期望轨迹
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

        if self.slalom_mode_active:
            self._draw_slalom_path(visualizer, batch, self.cfg.viz.z_offset)
        else:
            self._draw_arc_path(visualizer, batch, self.cfg.viz.z_offset)


    # 绘制转弯基元路径
    def _draw_arc_path(self, visualizer: "DebugVisualizer", batch: int, z_offset: float) -> None:
        curvature = self.curvature_command[batch].item()
        vel = self.vel_command[batch].item()
        start = self._start_positions[batch].cpu().numpy()
        radius = 0.008

        # 接近段
        n_app = 5
        for i in range(n_app + 1):
            frac = i / n_app
            pt = np.array([-_INIT_DIST + frac * _INIT_DIST, 0.0, start[2] + z_offset])
            visualizer.add_sphere(center=pt, radius=radius, color=(0.5, 0.8, 0.5, 0.5), label=f"approach_{i}")

        # 圆弧段
        n_pts = 50
        t_max = 20.0
        dt_path = t_max / n_pts
        heading_0 = 0.0
        for i in range(n_pts + 1):
            t_i = i * dt_path
            if abs(curvature) < 1e-6:
                x_i = vel * t_i * math.cos(heading_0)
                y_i = vel * t_i * math.sin(heading_0)
            else:
                R = 1.0 / curvature
                omega = curvature * vel
                dtheta = omega * t_i
                x_i = R * (math.sin(heading_0 + dtheta) - math.sin(heading_0))
                y_i = -R * (math.cos(heading_0 + dtheta) - math.cos(heading_0))
            pt = np.array([x_i, y_i, start[2] + z_offset])
            visualizer.add_sphere(center=pt, radius=radius, color=(1.0, 0.6, 0.0, 0.6), label=f"arc_{i}")

    # 绘制绕杆路径：接近段 + LUT 周期性路径 + 杆柱
    def _draw_slalom_path(self, visualizer: "DebugVisualizer", batch: int, z_offset: float) -> None:
        spacing = self.active_pole_spacing
        start = self._start_positions[batch].cpu().numpy()
        z = start[2] + z_offset

        _, xs, ys, _, _ = _generate_slalom_lut_one_period(spacing, n_arc_pts=15)
        period_len = np.array(xs[-1])
        n_periods = 3

        radius = 0.006
        # 接近段 (长度 = _INIT_DIST, 与 path.py / events.py 一致)
        n_app = 5
        for i in range(n_app + 1):
            frac = i / n_app
            pt = np.array([-_INIT_DIST + frac * _INIT_DIST, 0.0, z])
            visualizer.add_sphere(center=pt, radius=radius, color=(0.5, 0.8, 0.5, 0.5), label=f"sl_ap_{i}")

        # 周期路径
        for k in range(n_periods):
            offset_x = k * period_len
            for i in range(len(xs)):
                pt = np.array([xs[i] + offset_x, ys[i], z])
                color = (0.2, 0.7, 1.0, 0.5) if k % 2 == 0 else (1.0, 0.5, 0.2, 0.5)
                visualizer.add_sphere(center=pt, radius=radius, color=color, label=f"sl_{k}_{i}")

        # 杆柱
        pole_h = POLE_HALF_HEIGHT * 2
        for pi in range(6):
            px, py = pi * spacing, POLE_Y
            visualizer.add_cylinder(
                start=np.array([px, py, 0.0]),
                end=np.array([px, py, pole_h]),
                radius=POLE_RADIUS,
                color=(0.9, 0.35, 0.2, 0.6),
                label=f"pole_{pi}",
            )


@dataclass(kw_only=True)
class SlalomCommandCfg(CommandTermCfg):
    asset_name: str = "robot"
    resampling_time_range: Tuple[float, float] = (20.0, 30.0)
    debug_vis: bool = False

    # 固定值覆盖（None 表示使用课程采样）
    fixed_velocity: Optional[float] = None
    fixed_height_f: Optional[float] = None
    fixed_height_h: Optional[float] = None
    fixed_gait_freq: Optional[float] = None
    fixed_curvature: Optional[float] = None
    fixed_pole_spacing: Optional[float] = None

    @dataclass
    class VizCfg:
        z_offset: float = 0.1
        scale: float = 1.0

    viz: VizCfg = field(default_factory=VizCfg)
    class_type: type[CommandTerm] = SlalomCommand

    def build(self, env: "ManagerBasedRlEnv") -> CommandTerm:
        return self.class_type(self, env)
