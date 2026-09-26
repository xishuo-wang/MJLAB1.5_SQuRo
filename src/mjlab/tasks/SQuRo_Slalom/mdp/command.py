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
    BASE_VEL,
    CURVATURE_TARGET,
    CURVATURE_TARGET_MAX,
    GAIT_FREQ_MIN,
    GAIT_FREQ_MAX,
    GAIT_FREQ_PHASE1,
    PHASE1_MID_ITER,
    STRAIGHT_VEL_SCALE,
    VEL_MIN,
    _STEPS_PER_ITER,
    SMOOTH_TIME,
    SMOOTH_VEL,
    get_curriculum_pole_spacing,
    get_training_phase,
)
from .pole import update_pole_visibility
from .path import (
    _INIT_DIST,
    _approach_rev_table,
    get_path_curvature,
    get_phase0_approach,
    get_effective_pole_spacing,
    _generate_slalom_lut_smooth_period,
)
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer



# 命令配置
FIXED_HEIGHT_F = 0.055      # 前肢高度
FIXED_HEIGHT_H = 0.055      # 后肢高度



# 获取曲率采样范围
def get_curvature_range(step_counter: int) -> Tuple[float, float]:
    iter_num = step_counter // _STEPS_PER_ITER
    if iter_num < PHASE1_MID_ITER:
        progress = iter_num / PHASE1_MID_ITER  # 0 → 1
        kappa_max = 0.5 + progress * (CURVATURE_TARGET_MAX - 0.5)
        return (-kappa_max, kappa_max)
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
        self.fixed_pole_spacing = cfg.fixed_pole_spacing

        # 每 episode 共享步频 (Phase 0 随机采样, Phase 1 固定)
        self._shared_gait_freq = GAIT_FREQ_PHASE1

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
        return get_training_phase(self._env.common_step_counter) == 1


    # 当前杆间距 (cfg.fixed_pole_spacing 优先, 否则从课程自动读取)
    @property
    def active_pole_spacing(self) -> float:
        override = getattr(self.cfg, "fixed_pole_spacing", None)
        if override is not None:
            return float(override)
        raw = get_curriculum_pole_spacing(self._env.common_step_counter)
        # Phase 1: 平滑有效间距 (不兼容区间 → 无直行最小间距, 与 vel 解耦, 保证周期位移匹配)
        if self.slalom_mode_active:
            return get_effective_pole_spacing(raw)
        return raw


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


    # 仅在 reset 时调用，每个 episode 固定曲率不变
    def _resample_curvature(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        current_step = self._env.common_step_counter
        phase = get_training_phase(current_step)

        # 根据阶段自动切换杆可见性 (Phase 0 透明, Phase 1 正常)
        update_pole_visibility(self._env, phase)

        if phase == 0:
            # Phase 0: 转弯基元 — 采样曲率 + 步频(1~2Hz 随机), 速度 = 基础速度×步频×曲率缩放
            self.curvature_command[env_ids] = self._get_curvature(n, current_step)
            if self.fixed_gait_freq is not None:
                self._shared_gait_freq = float(self.fixed_gait_freq)
            else:
                self._shared_gait_freq = float(GAIT_FREQ_MIN + torch.rand(1).item() * (GAIT_FREQ_MAX - GAIT_FREQ_MIN))
            self.gait_freq_command[env_ids] = self._shared_gait_freq
            base_vel = float(self.fixed_velocity) if self.fixed_velocity is not None else BASE_VEL
            scale = 1.0 - (1.0 - VEL_MIN) * self.curvature_command[env_ids].abs() / CURVATURE_TARGET_MAX
            self.vel_command[env_ids] = base_vel * self.gait_freq_command[env_ids] * scale
            # 更新机器人初始位置/姿态: 接近段圆弧起点 (匹配本 episode 曲率)
            if n > 0:
                ax, ay, ah = get_phase0_approach(self.curvature_command[env_ids])
                root = torch.zeros(n, 13, device=self.device)
                root[:, 0] = ax
                root[:, 1] = ay
                root[:, 2] = 0.06
                c = torch.cos(ah / 2)
                s = torch.sin(ah / 2)
                root[:, 4] = 0.70710678 * (s - c)
                root[:, 5] = -0.70710678 * (c + s)
                self._env.scene.entities["robot"].write_root_state_to_sim(root, env_ids=env_ids)
        else:
            # Phase 1: 绕杆训练 — 曲率 -CURVATURE_TARGET (LUT 第一段 CW 弧 = 负), 步频 1~2Hz 随机
            # 速度: 变速 (直行段 STRAIGHT_VEL_SCALE 快, 转弯段 VEL_MIN 慢), 每步由 _update_command 动态更新
            self.curvature_command[env_ids] = torch.full((n,), -CURVATURE_TARGET, device=self.device)
            if self.fixed_gait_freq is not None:
                self._shared_gait_freq = float(self.fixed_gait_freq)
            else:
                self._shared_gait_freq = float(GAIT_FREQ_MIN + torch.rand(1).item() * (GAIT_FREQ_MAX - GAIT_FREQ_MIN))
            self.gait_freq_command[env_ids] = self._shared_gait_freq
            base_vel = float(self.fixed_velocity) if self.fixed_velocity is not None else BASE_VEL
            # 接近段起点 κ=-CURVATURE_TARGET → 初始速度为弯道低速 (之后每步变速)
            self.vel_command[env_ids] = torch.full((n,), base_vel * self._shared_gait_freq * VEL_MIN, device=self.device)

        # 缓存基础速度标量 (cfg 级全局量), 供 path 模块生成名义 τ(s) 表 (避免每步 GPU-CPU 同步)
        if n > 0:
            self._env._slalom_base_vel_scalar = base_vel  # type: ignore[attr-defined]
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
        # 计时器与到期重采样由基类 CommandTerm.compute 负责, 此处不得再扣减或重采样
        # Phase 1: 每步动态更新曲率 + 期望速度 (变速: 直行快, 弯道慢; fixed_velocity 作为基础速度同样变速)
        if self.slalom_mode_active:
            kappa = get_path_curvature(self._env)
            self.curvature_command[:] = kappa
            base_vel = float(self.fixed_velocity) if self.fixed_velocity is not None else BASE_VEL
            scale = STRAIGHT_VEL_SCALE - (STRAIGHT_VEL_SCALE - VEL_MIN) * kappa.abs() / CURVATURE_TARGET
            self.vel_command[:] = base_vel * self.gait_freq_command * scale


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


    # 绘制转弯基元轨迹：直行=射线, 转弯=圆弧 (与 compute_arc_path_ref 期望轨迹一致)
    def _draw_arc_path(self, visualizer: "DebugVisualizer", batch: int, z_offset: float) -> None:
        curvature = self.curvature_command[batch].item()
        vel = self.vel_command[batch].item()
        start = self._start_positions[batch].cpu().numpy()
        z = start[2] + z_offset

        n_pts = 50
        t_max = 20.0
        dt_path = t_max / n_pts
        radius = 0.008

        if abs(curvature) < 1e-6:
            # 直行: 从起点沿 +X (期望轨迹为直线)
            for i in range(n_pts + 1):
                t_i = i * dt_path
                pt = np.array([start[0] + vel * t_i, start[1], z])
                visualizer.add_sphere(center=pt, radius=radius, color=(1.0, 0.6, 0.0, 0.6), label=f"arc_{i}")
        else:
            # 转弯: 期望轨迹 = 圆心 (0, 1/κ) 半径 |1/κ| 的固定圆弧 (与 _INIT_DIST 无关)
            # θ = κ·(vel·t − _INIT_DIST), 起点 θ0 = −κ·_INIT_DIST (start 位于圆弧上)
            theta0 = -curvature * _INIT_DIST
            for i in range(n_pts + 1):
                t_i = i * dt_path
                theta = theta0 + curvature * vel * t_i
                x_i = math.sin(theta) / curvature
                y_i = (1.0 - math.cos(theta)) / curvature
                pt = np.array([x_i, y_i, z])
                visualizer.add_sphere(center=pt, radius=radius, color=(1.0, 0.6, 0.0, 0.6), label=f"arc_{i}")


    # 绘制绕杆轨迹：平滑 LUT 路径 (与实际期望轨迹一致)
    def _draw_slalom_path(self, visualizer: "DebugVisualizer", batch: int, z_offset: float) -> None:
        spacing = self.active_pole_spacing
        start = self._start_positions[batch].cpu().numpy()
        z = start[2] + z_offset

        _, xs, ys, _, _ = _generate_slalom_lut_smooth_period(spacing)
        period_len = float(xs[-1])
        n_periods = 3
        x_off = spacing   # 新几何: 周期起点 = 第一根杆正上方 (x=spacing)

        radius = 0.006
        # 接近段 (圆弧: 平台-K + 过渡-K→0, 与实际期望轨迹一致)
        app_tbl = _approach_rev_table(_INIT_DIST, SMOOTH_VEL * SMOOTH_TIME)
        n_app = len(app_tbl["x"])
        step = max(1, n_app // 5)
        for i in range(0, n_app, step):
            pt = np.array([x_off + app_tbl["x"][i], app_tbl["y"][i], z])
            visualizer.add_sphere(center=pt, radius=radius, color=(0.5, 0.8, 0.5, 0.5), label=f"sl_ap_{i}")

        # 周期路径 (世界坐标, 与期望轨迹一致) — 平滑 LUT 逐点积分点数过多, 每周期降采样约 60 点
        n_pts = len(xs)
        draw_idx = np.unique(np.linspace(0, n_pts - 1, 60).astype(int))
        for k in range(n_periods):
            offset_x = k * period_len
            for i in draw_idx:
                pt = np.array([x_off + xs[i] + offset_x, ys[i], z])
                color = (0.2, 0.7, 1.0, 0.5) if k % 2 == 0 else (1.0, 0.5, 0.2, 0.5)
                visualizer.add_sphere(center=pt, radius=radius, color=color, label=f"sl_{k}_{i}",)



@dataclass(kw_only=True)
class SlalomCommandCfg(CommandTermCfg):
    asset_name: str = "robot"
    resampling_time_range: Tuple[float, float] = (20.0, 30.0)
    debug_vis: bool = False
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
