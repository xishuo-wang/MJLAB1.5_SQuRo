from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.managers import CommandTermCfg
from mjlab.managers.command_manager import CommandTerm
from .curriculums import (
    BASE_VEL, CURVATURE_TARGET_MAX, GAIT_FREQ_MIN, GAIT_FREQ_MAX,
    PHASE1_MID_ITER, SLALOM_CURVATURE_MIN, SLALOM_CURVATURE_MAX, _STEPS_PER_ITER,
    SlalomSpacingCurriculum, SPACING_HARDEST_PROBABILITY,
    get_training_phase, get_training_phase_batch, get_velocity_scale,
)
from .path import _INIT_DIST, compute_path_ref, get_phase0_approach, get_slalom_tables
from .pole import write_pole_mocap
from .reference import invalidate_reference_cache
from .slalom_geometry import curvature_from_spacing, get_slalom_geometry

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


FIXED_HEIGHT_F = 0.055
FIXED_HEIGHT_H = 0.055


# 一阶段在前两千轮扩展曲率，随后保持完整范围。
def get_curvature_range(step_counter: int) -> tuple[float, float]:
    progress = min((step_counter // _STEPS_PER_ITER) / PHASE1_MID_ITER, 1.0)
    upper = 0.5 + progress * (CURVATURE_TARGET_MAX - 0.5)
    return -upper, upper


# 管理逐回合命令、整数绕杆曲率及其对应的实体杆阵列。
class SlalomCommand(CommandTerm):
    cfg: SlalomCommandCfg

    # 初始化命令视图和逐环境几何状态。
    def __init__(self, cfg: SlalomCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.spacing_curriculum = SlalomSpacingCurriculum()
        if cfg.spacing_curriculum_state is not None:
            self.spacing_curriculum.load_state_dict(cfg.spacing_curriculum_state)
        self.robot = env.scene[cfg.asset_name]
        self.command_tensor = torch.zeros(self.num_envs, 5, device=self.device)
        self.vel_command = self.command_tensor[:, 0]
        self.height_f_command = self.command_tensor[:, 1]
        self.height_h_command = self.command_tensor[:, 2]
        self.gait_freq_command = self.command_tensor[:, 3]
        self.curvature_command = self.command_tensor[:, 4]
        for name in ("fixed_velocity", "fixed_height_f", "fixed_height_h", "fixed_gait_freq", "fixed_curvature"):
            setattr(self, name, getattr(cfg, name))
        self.slalom_curvature_index = torch.full(
            (self.num_envs,), SLALOM_CURVATURE_MAX - SLALOM_CURVATURE_MIN, device=self.device, dtype=torch.long)
        self.pole_spacing = torch.zeros(self.num_envs, device=self.device)
        self.pole_y = torch.zeros_like(self.pole_spacing)
        self.metrics = {name: torch.zeros(self.num_envs, device=self.device)
                        for name in ("slalom_fraction", "slalom_peak_curvature", "pole_spacing")}
        self._start_positions = torch.zeros(self.num_envs, 3, device=self.device)
        self._start_recorded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(env_ids)
        self._resample_curvature(env_ids)
        self.time_left.fill_(cfg.resampling_time_range[0])

    @property
    def command(self) -> torch.Tensor:
        return self.command_tensor

    @property
    def slalom_mode_active(self) -> bool:
        return get_training_phase(self._env.common_step_counter, self.cfg.phase0_only) == 1

    @property
    def phase1_mask(self) -> torch.Tensor:
        episode_start = self._env.common_step_counter - self._env.episode_length_buf
        return get_training_phase_batch(episode_start, self.cfg.phase0_only)

    @property
    def active_pole_spacing(self) -> torch.Tensor:
        return self.pole_spacing

    @property
    def slalom_peak_curvature(self) -> torch.Tensor:
        return self.slalom_curvature_index + SLALOM_CURVATURE_MIN

    # 仅在回合开始时采样高度和自身步频。
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        self.height_f_command[env_ids] = FIXED_HEIGHT_F if self.fixed_height_f is None else self.fixed_height_f
        self.height_h_command[env_ids] = FIXED_HEIGHT_H if self.fixed_height_h is None else self.fixed_height_h
        if self.fixed_gait_freq is None:
            self.gait_freq_command[env_ids] = torch.empty(n, device=self.device).uniform_(GAIT_FREQ_MIN, GAIT_FREQ_MAX)
        else:
            self.gait_freq_command[env_ids] = self.fixed_gait_freq

    # 采样整数峰值；杆距覆盖映射到最近的整数曲率条件。
    def _sample_slalom_indices(self, n: int) -> torch.Tensor:
        fixed, spacing = self.cfg.fixed_slalom_curvature, self.cfg.fixed_pole_spacing
        if fixed is not None and spacing is not None:
            raise ValueError("固定绕杆峰值曲率和固定杆距只能指定一个")
        if spacing is not None:
            fixed = curvature_from_spacing(spacing)
        if fixed is not None:
            get_slalom_geometry(fixed)
            return torch.full((n,), fixed - SLALOM_CURVATURE_MIN, device=self.device, dtype=torch.long)
        if self.cfg.enable_spacing_curriculum:
            highest = self.spacing_curriculum.max_curvature - SLALOM_CURVATURE_MIN
            sampled = torch.randint(0, highest, (n,), device=self.device)
            hardest = torch.rand(n, device=self.device) < SPACING_HARDEST_PROBABILITY
            return torch.where(hardest, highest, sampled)
        return torch.randint(0, SLALOM_CURVATURE_MAX - SLALOM_CURVATURE_MIN + 1, (n,), device=self.device)

    # 重置曲率、机器人起点和实体杆，只作用于本次重置环境。
    def _resample_curvature(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        env, n = self._env, len(env_ids)
        phase1 = self.slalom_mode_active
        base_velocity = BASE_VEL if self.fixed_velocity is None else float(self.fixed_velocity)
        env._slalom_base_vel_scalar = base_velocity
        tables = get_slalom_tables(env, base_velocity)
        if phase1:
            self.slalom_curvature_index[env_ids] = self._sample_slalom_indices(n)
        rows = self.slalom_curvature_index[env_ids]
        self.pole_spacing[env_ids] = tables["spacing"][rows]
        self.pole_y[env_ids] = tables["pole_y"][rows]
        if phase1:
            self.curvature_command[env_ids] = 0.0
            ax, ay, ah = tables["start"][rows].unbind(dim=-1)
        else:
            if self.fixed_curvature is None:
                lower, upper = get_curvature_range(env.common_step_counter)
                self.curvature_command[env_ids] = torch.empty(n, device=self.device).uniform_(lower, upper)
            else:
                self.curvature_command[env_ids] = self.fixed_curvature
            ax, ay, ah = get_phase0_approach(self.curvature_command[env_ids])
        speed_curvature = self.slalom_peak_curvature[env_ids] if phase1 else self.curvature_command[env_ids]
        self.vel_command[env_ids] = base_velocity * self.gait_freq_command[env_ids] * get_velocity_scale(speed_curvature)
        root = torch.zeros(n, 13, device=self.device)
        root[:, 0], root[:, 1], root[:, 2] = ax, ay, 0.06
        origins = getattr(env.scene, "env_origins", None)
        if origins is not None:
            root[:, :3] += origins[env_ids]
        c, s = torch.cos(ah / 2), torch.sin(ah / 2)
        root[:, 4], root[:, 5] = 0.70710678 * (s - c), -0.70710678 * (c + s)
        self.robot.write_root_state_to_sim(root, env_ids=env_ids)
        write_pole_mocap(env, env_ids, self.pole_spacing[env_ids], self.pole_y[env_ids], phase1)
        self._start_recorded[env_ids] = False

    # 命令管理器完成重采样后同步几何，并使参考缓存失效。
    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        if env_ids is None or isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)
            env_ids = ids if env_ids is None else ids[env_ids]
        extras = super().reset(env_ids)
        self._resample_curvature(env_ids)
        invalidate_reference_cache(self._env)
        monitor = getattr(self._env, "_slalom_passage_monitor", None)
        if monitor is not None:
            monitor.reset(env_ids)
        return extras

    # 按逐环境阶段更新曲率，绕杆速度按本回合峰值保持稳定。
    def _update_command(self) -> None:
        compute_path_ref(self._env)
        kappa = self._env._path_kappa
        base_velocity = BASE_VEL if self.fixed_velocity is None else float(self.fixed_velocity)
        phase1 = self.phase1_mask
        self.curvature_command.copy_(torch.where(phase1, kappa, self.curvature_command))
        velocity = base_velocity * self.gait_freq_command * get_velocity_scale(self.slalom_peak_curvature)
        self.vel_command.copy_(torch.where(phase1, velocity, self.vel_command))

    # 在训练日志中标记绕杆比例和当前几何分布。
    def _update_metrics(self) -> None:
        self.metrics["slalom_fraction"].copy_(self.phase1_mask)
        self.metrics["slalom_peak_curvature"].copy_(self.slalom_peak_curvature)
        self.metrics["pole_spacing"].copy_(self.pole_spacing)

    # 绘制选中环境本回合的真实参考几何。
    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        if not self.cfg.debug_vis or visualizer.env_idx >= self.num_envs:
            return
        batch = visualizer.env_idx
        origin = getattr(self._env.scene, "env_origins", None)
        offset = np.zeros(3) if origin is None else origin[batch].cpu().numpy()
        z = float(self.robot.data.root_link_pos_w[batch, 2]) + self.cfg.viz.z_offset
        if bool(self.phase1_mask[batch]):
            geometry = get_slalom_geometry(int(self.slalom_peak_curvature[batch]))
            for part_name, path in (("approach", geometry["approach"]), ("period", geometry["period"])):
                for i in np.linspace(0, len(path["x"]) - 1, 60).astype(int):
                    x = path["x"][i] + (geometry["spacing"] if part_name == "period" else 0)
                    visualizer.add_sphere(center=np.array([x + offset[0], path["y"][i] + offset[1], z]),
                                          radius=0.004, color=(0.2, 0.7, 1.0, 0.5), label=f"{part_name}_{i}")
        else:
            k, velocity = float(self.curvature_command[batch]), float(self.vel_command[batch])
            for i, t in enumerate(np.linspace(0, self._env.cfg.episode_length_s, 60)):
                distance = velocity * t - _INIT_DIST
                x = distance if abs(k) < 1e-6 else math.sin(k * distance) / k
                y = 0.0 if abs(k) < 1e-6 else (1 - math.cos(k * distance)) / k
                visualizer.add_sphere(center=np.array([x + offset[0], y + offset[1], z]), radius=0.004,
                                      color=(1.0, 0.6, 0.0, 0.6), label=f"arc_{i}")


@dataclass(kw_only=True)
class SlalomCommandCfg(CommandTermCfg):
    asset_name: str = "robot"
    phase0_only: bool = False
    enable_spacing_curriculum: bool = True
    spacing_curriculum_state: dict | None = None
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = False
    fixed_velocity: float | None = None
    fixed_height_f: float | None = None
    fixed_height_h: float | None = None
    fixed_gait_freq: float | None = None
    fixed_curvature: float | None = None
    fixed_pole_spacing: float | None = None
    fixed_slalom_curvature: int | None = None

    @dataclass
    class VizCfg:
        z_offset: float = 0.1
        scale: float = 1.0

    viz: VizCfg = field(default_factory=VizCfg)
    class_type: type[CommandTerm] = SlalomCommand

    # 构造命令管理器实例。
    def build(self, env: ManagerBasedRlEnv) -> CommandTerm:
        return self.class_type(self, env)
