from __future__ import annotations
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
STAGE1_END = 500
STAGE2_END = 1500

# 目标最大曲率: κ = 1/R_min, R_min=0.1m → κ_max=10 m⁻¹
# ω = κ × v, 在 v=0.1m/s 时 ω_max = 1.0 rad/s
CURVATURE_TARGET_MAX = 5.0

# 固定值
FIXED_VEL = 0.1
FIXED_HEIGHT_F = 0.06
FIXED_HEIGHT_H = 0.06
FIXED_GAIT_FREQ = 1.0


def get_current_stage(step_counter: int) -> int:
    iter_num = step_counter // 24
    if iter_num < STAGE1_END:
        return 1
    elif iter_num < STAGE2_END:
        return 2
    else:
        return 3


def get_curvature_range(stage: int, step_counter: int) -> Tuple[float, float]:
    """曲率采样范围 κ = 1/R, ω = κ × v"""
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
    """curvature = 1/R (signed): κ>0=左转, κ<0=右转, κ=0=直行; ω = κ × v"""
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

        # 初始化
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

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        current_step = self._env.common_step_counter

        self.vel_command[env_ids] = self._get_velocity(n)
        self.height_f_command[env_ids] = self._get_height_f(n)
        self.height_h_command[env_ids] = self._get_height_h(n)
        self.gait_freq_command[env_ids] = self._get_gait_freq(n)
        self.curvature_command[env_ids] = self._get_curvature(n, current_step)

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
        base_pos = self.robot.data.root_link_pos_w[batch].cpu().numpy()
        if torch.norm(self.robot.data.root_link_pos_w[batch]) < 1e-6:
            return

        scale = self.cfg.viz.scale
        z_offset = self.cfg.viz.z_offset
        cmd_start = base_pos + [0, 0, z_offset]

        # 线速度命令箭头（蓝色）
        cmd_vel = torch.tensor([self.vel_command[batch].item(), 0.0, 0.0])
        visualizer.add_arrow(
            cmd_start, cmd_start + cmd_vel.cpu().numpy() * scale,
            color=(0.2, 0.2, 0.8, 0.8), width=0.01,
        )
        # 实际速度箭头（绿色）
        actual_vel = self.robot.data.root_link_lin_vel_w[batch].cpu().numpy()
        visualizer.add_arrow(
            cmd_start, cmd_start + actual_vel * scale,
            color=(0.2, 0.8, 0.2, 0.8), width=0.01,
        )


@dataclass(kw_only=True)
class SlalomCommandCfg(CommandTermCfg):
    asset_name: str = "robot"
    resampling_time_range: Tuple[float, float] = (4.0, 6.0)
    debug_vis: bool = False

    # 固定值（None=使用课程采样，设值可覆盖）
    fixed_velocity: Optional[float] = None
    fixed_height_f: Optional[float] = None
    fixed_height_h: Optional[float] = None
    fixed_gait_freq: Optional[float] = None
    fixed_curvature: Optional[float] = None  # 固定曲率 κ=1/R (m⁻¹)，用于测试

    @dataclass
    class VizCfg:
        z_offset: float = 0.1
        scale: float = 1.0

    viz: VizCfg = field(default_factory=VizCfg)
    class_type: type[CommandTerm] = SlalomCommand

    def build(self, env: "ManagerBasedRlEnv") -> CommandTerm:
        return self.class_type(self, env)
