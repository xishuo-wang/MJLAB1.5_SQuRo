from __future__ import annotations
import torch
import numpy as np
from mjlab.entity import Entity
from dataclasses import dataclass, field
from mjlab.managers import CommandTermCfg
from typing import TYPE_CHECKING, Optional, Tuple
from mjlab.managers.command_manager import CommandTerm
from .curriculums import get_training_phase
from .indices import _MODEL_INDICES, resolve_model_indices
from .path import (
    HEIGHT_NORMAL,
    HEIGHT_HOLE,
    HOLE_NUM,
    get_front_center_height,
    get_holes,
    get_rear_center_height,
    sample_hole_positions,
)
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


# 命令配置
BASE_VEL = 0.1            # 正常高度基础速度 (m/s)
VEL_LOW = 0.05            # 单低速度 (m/s): 前肢或后肢任一低高度
VEL_STOP = 0.0            # 双低速度 (m/s): 前后肢均低高度
GAIT_FREQ = 1.0           # 步频 (Hz)
HEIGHT_THRESHOLD = (HEIGHT_NORMAL + HEIGHT_HOLE) / 2  # 高度状态判定阈值 = 0.0375

# Phase0 高度档 (重构前接近) — 随机采样前后肢高度命令
PHASE0_HEIGHTS = [0.02, 0.04, 0.045, 0.05, 0.055, 0.06]
PHASE0_BASE_SPEED = 0.25  # Phase0 基础速度 (m/s, 重构前)
PHASE0_BASE_HEIGHT = 0.06 # Phase0 速度缩放基准高度 (m, 重构前)


# 5D 命令 [vel_x, height_f, height_h, gait_freq, curvature]
# Tunnel: 曲率固定 0; Phase0 随机高度命令; Phase1 由高度轨迹动态生成高度与速度
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

        # 洞位置缓存 (单 episode 内固定, Phase1 使用)
        env._tunnel_hole_xs = sample_hole_positions(env, self.num_envs)  # type: ignore[attr-defined]

        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(env_ids)
        t_range = self.cfg.resampling_time_range
        self.time_left[env_ids] = torch.rand(len(env_ids), device=self.device) * (t_range[1] - t_range[0]) + t_range[0]

    @property
    def command(self) -> torch.Tensor:
        return self.command_tensor

    def _get_velocity(self, n: int, base_vel: float = BASE_VEL) -> torch.Tensor:
        if self.fixed_velocity is not None:
            return torch.full((n,), float(self.fixed_velocity), device=self.device)
        return torch.full((n,), base_vel, device=self.device)

    def _get_height_f(self, n: int) -> torch.Tensor:
        if self.fixed_height_f is not None:
            return torch.full((n,), float(self.fixed_height_f), device=self.device)
        return torch.full((n,), HEIGHT_NORMAL, device=self.device)

    def _get_height_h(self, n: int) -> torch.Tensor:
        if self.fixed_height_h is not None:
            return torch.full((n,), float(self.fixed_height_h), device=self.device)
        return torch.full((n,), HEIGHT_NORMAL, device=self.device)

    def _get_gait_freq(self, n: int) -> torch.Tensor:
        if self.fixed_gait_freq is not None:
            return torch.full((n,), float(self.fixed_gait_freq), device=self.device)
        return torch.full((n,), GAIT_FREQ, device=self.device)

    # Phase0: 随机采样前后肢高度命令 + 连续速度缩放 (重构前接近)
    def _resample_phase0(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        device = self.device
        heights = torch.tensor(PHASE0_HEIGHTS, device=device)
        idx_f = torch.randint(0, len(heights), (n,), device=device)
        idx_h = torch.randint(0, len(heights), (n,), device=device)
        h_f = heights[idx_f]
        h_h = heights[idx_h]
        self.height_f_command[env_ids] = h_f
        self.height_h_command[env_ids] = h_h
        # 速度 = 基础速度 × min(hF,hH)/基准高度 (重构前连续缩放)
        eff_h = torch.minimum(h_f, h_h)
        vel = PHASE0_BASE_SPEED * eff_h / PHASE0_BASE_HEIGHT
        self.vel_command[env_ids] = self._get_velocity(n, base_vel=0.0) if self.fixed_velocity is not None else vel * self.gait_freq_command[env_ids]

    # Phase1: 初始化高度命令为正常 (每步由轨迹动态覆盖), 速度 = 基础
    def _resample_phase1(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        self.height_f_command[env_ids] = self._get_height_f(n)
        self.height_h_command[env_ids] = self._get_height_h(n)
        vel = self._get_velocity(n)
        if self.fixed_velocity is None:
            vel = vel * self.gait_freq_command[env_ids]
        self.vel_command[env_ids] = vel

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        self.gait_freq_command[env_ids] = self._get_gait_freq(n)
        self.curvature_command[env_ids] = 0.0   # 曲率固定 0 (直行)
        self._start_recorded[env_ids] = False
        phase = get_training_phase(self._env.common_step_counter)
        if phase == 0:
            self._resample_phase0(env_ids)
        else:
            self._resample_phase1(env_ids)

    # Phase1 每步: 根据前/后肢当前位置的期望高度, 动态更新高度命令与速度
    def _update_phase1(self) -> None:
        resolve_model_indices(self.robot)
        body_pos = self.robot.data.body_link_pos_w
        x_f = body_pos[:, _MODEL_INDICES.f_body_id, 0]   # 前肢中心 (F_body) x
        x_h = body_pos[:, _MODEL_INDICES.h_body_id, 0]   # 后肢中心 (H_body) x
        z_f_ref = get_front_center_height(self._env, x_f)
        z_h_ref = get_rear_center_height(self._env, x_h)
        self.height_f_command[:] = z_f_ref
        self.height_h_command[:] = z_h_ref
        # 速度规则: 双正常 0.1 / 单低 0.05 / 双低 0
        if self.fixed_velocity is not None:
            self.vel_command[:] = float(self.fixed_velocity)
            return
        low_f = z_f_ref < HEIGHT_THRESHOLD
        low_h = z_h_ref < HEIGHT_THRESHOLD
        vel = torch.where(low_f | low_h, torch.full_like(z_f_ref, VEL_LOW),
                          torch.full_like(z_f_ref, BASE_VEL))
        vel = torch.where(low_f & low_h, torch.full_like(z_f_ref, VEL_STOP), vel)
        self.vel_command[:] = vel * self.gait_freq_command

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
            # 每 episode 重新采样洞位置 (单 episode 内固定)
            holes = get_holes(self._env)
            if holes.shape[1] != HOLE_NUM:
                holes = torch.zeros(self.num_envs, HOLE_NUM, device=self.device)
            holes[env_ids] = sample_hole_positions(self._env, len(env_ids))
            self._env._tunnel_hole_xs = holes  # type: ignore[attr-defined]
            self._resample_command(env_ids)
        return extras

    def _update_command(self) -> None:
        phase = get_training_phase(self._env.common_step_counter)
        if phase == 1:
            self._update_phase1()
        else:
            # Phase0: 定期重采样高度命令 (不需要 episode 内固定)
            env_ids = (self.time_left <= 0.0).nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self._resample_phase0(env_ids)
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
        # 绘制当前 episode 前/后肢期望高度轨迹 (从起点起 1.5m, 覆盖多个洞)
        xs = torch.linspace(start_x, start_x + 1.5, 120, device=self.device)
        holes = get_holes(self._env)[batch : batch + 1]
        from .path import _height_from_holes, FRONT_DOWN_OFF, FRONT_UP_OFF, REAR_DOWN_OFF, REAR_UP_OFF
        z_front = _height_from_holes(xs, holes.expand(len(xs), -1), FRONT_DOWN_OFF, FRONT_UP_OFF).cpu().numpy()
        z_rear = _height_from_holes(xs, holes.expand(len(xs), -1), REAR_DOWN_OFF, REAR_UP_OFF).cpu().numpy()
        xs_np = xs.cpu().numpy()
        for x_i, z_f, z_r in zip(xs_np, z_front, z_rear):
            visualizer.add_sphere(center=np.array([x_i, 0.0, z_f + z_off]), radius=0.004,
                                  color=(0.2, 0.6, 1.0, 0.6), label=f"fref_{batch}_{x_i:.3f}")
            visualizer.add_sphere(center=np.array([x_i, 0.0, z_r + z_off]), radius=0.004,
                                  color=(1.0, 0.3, 0.3, 0.6), label=f"rref_{batch}_{x_i:.3f}")


@dataclass(kw_only=True)
class TunnelCommandCfg(CommandTermCfg):
    asset_name: str = "robot"
    resampling_time_range: Tuple[float, float] = (2.0, 4.0)   # Phase0 高度重采样间隔
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
