"""SQuRo绕杆任务第一阶段 — 速度+角速度命令系统

命令格式: [v_cmd (前向线速度), ω_cmd (偏航角速度)]
课程:
  阶段1 (直行): v ∈ [0, 0.2], ω = 0
  阶段2 (转弯引入): v ∈ [0, 0.2], ω 从 [-0.5, 0.5] 扩展到 [-ω_max, ω_max]
  阶段3 (全范围): v ∈ [0, 0.2], ω ∈ [-ω_max, ω_max]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Tuple
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers import CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


# 阶段阈值（iterations）
STAGE1_END = 500
STAGE2_END = 1500

# 目标最大角速度：v=0.2m/s ÷ r=0.1m = 2.0 rad/s
OMEGA_TARGET_MAX = 2.0

# 线速度范围
V_MIN, V_MAX = 0.0, 0.2


def get_current_stage(step_counter: int) -> int:
    """根据全局步数返回当前课程阶段"""
    iter_num = step_counter // 24
    if iter_num < STAGE1_END:
        return 1
    elif iter_num < STAGE2_END:
        return 2
    else:
        return 3


def get_v_range(stage: int) -> Tuple[float, float]:
    """线速度采样范围"""
    return (V_MIN, V_MAX)


def get_omega_range(stage: int, step_counter: int) -> Tuple[float, float]:
    """角速度采样范围，阶段2内线性插值"""
    if stage == 1:
        return (0.0, 0.0)
    elif stage == 2:
        iter_num = step_counter // 24
        progress = (iter_num - STAGE1_END) / (STAGE2_END - STAGE1_END)
        omega_max = 0.5 + progress * (OMEGA_TARGET_MAX - 0.5)
        return (-omega_max, omega_max)
    else:
        return (-OMEGA_TARGET_MAX, OMEGA_TARGET_MAX)


class SlalomCommand(CommandTerm):
    """2D命令 [v_cmd (前向线速度), ω_cmd (偏航角速度)]"""

    cfg: "SlalomCommandCfg"

    def __init__(self, cfg: "SlalomCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.asset_name]
        self.command_tensor = torch.zeros(self.num_envs, 2, device=self.device)
        self._v_command = self.command_tensor[:, 0]
        self._omega_command = self.command_tensor[:, 1]

        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(env_ids)

        t_range = self.cfg.resampling_time_range
        self.time_left[env_ids] = torch.rand(len(env_ids), device=self.device) * (
            t_range[1] - t_range[0]
        ) + t_range[0]

    @property
    def command(self) -> torch.Tensor:
        return self.command_tensor

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        stage = get_current_stage(self._env.common_step_counter)
        v_range = get_v_range(stage)
        omega_range = get_omega_range(stage, self._env.common_step_counter)

        v_cmd = torch.rand(n, device=self.device) * (v_range[1] - v_range[0]) + v_range[0]
        omega_cmd = torch.rand(n, device=self.device) * (omega_range[1] - omega_range[0]) + omega_range[0]

        self._v_command[env_ids] = v_cmd
        self._omega_command[env_ids] = omega_cmd

    def _update_command(self) -> None:
        env_ids = (self.time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._resample_command(env_ids)
            t_range = self.cfg.resampling_time_range
            self.time_left[env_ids] = torch.rand(len(env_ids), device=self.device) * (
                t_range[1] - t_range[0]
            ) + t_range[0]
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

        # 画线速度命令箭头（蓝色）
        cmd_vel = torch.tensor([self._v_command[batch].item(), 0.0, 0.0])
        cmd_start = base_pos + [0, 0, z_offset]
        visualizer.add_arrow(
            cmd_start, cmd_start + cmd_vel.cpu().numpy() * scale,
            color=(0.2, 0.2, 0.8, 0.8), width=0.01,
        )

        # 画实际速度箭头（绿色）
        actual_vel = self.robot.data.root_link_lin_vel_w[batch].cpu().numpy()
        visualizer.add_arrow(
            cmd_start, cmd_start + actual_vel * scale,
            color=(0.2, 0.8, 0.2, 0.8), width=0.01,
        )


@dataclass(kw_only=True)
class SlalomCommandCfg(CommandTermCfg):
    """SlalomCommand 配置"""

    asset_name: str = "robot"
    resampling_time_range: Tuple[float, float] = (4.0, 6.0)
    debug_vis: bool = False

    @dataclass
    class VizCfg:
        z_offset: float = 0.1
        scale: float = 1.0

    viz: VizCfg = field(default_factory=VizCfg)
    class_type: type[CommandTerm] = SlalomCommand

    def build(self, env: "ManagerBasedRlEnv") -> CommandTerm:
        return self.class_type(self, env)
