from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple
import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers import CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


# 阶段定义 (全局步数): 阶段 1/2 各占 1 个 iter, 其余全部走阶段 3 的位置表
STAGE1_END = 1 * 24
STAGE2_END = 1 * 24
STAGE3_END = 4000 * 24

BASE_HEIGHT = 0.06                     # 基准高度 (m)
BASE_SPEED = 0.25                      # 基准速度 (m/s, 对应基准高度)
HEIGHT_THRESHOLD = 0.04                # 高度高低判定阈值 (m)

STAGE1_HEIGHT_VALUES = [0.04, 0.045, 0.05, 0.055, 0.06]        # 阶段1: 中等高度
STAGE2_HEIGHT_VALUES = [0.02, 0.04, 0.045, 0.05, 0.055, 0.06]  # 阶段2: 全部高度
STAGE3_HEIGHT_VALUES = None            # 阶段3: 用位置表, 不随机采样

ANGLE_VALUES = [0.0]                   # 角度命令候选 (度)

# 阶段3固定位置表 (移动距离 m, 前肢高度, 后肢高度)
STAGE3_POSITION_SCHEDULE = [
    (0.0, 0.02, 0.05),
    (0.2, 0.06, 0.02),
    (0.32, 0.06, 0.06),
    (0.4, 0.04, 0.04),
    (0.8, 0.06, 0.06),
    (1.0, 0.02, 0.05),
    (1.2, 0.06, 0.02),
    (1.32, 0.06, 0.06),
]


# 高度缩放系数: 低于阈值统一取 0.1, 否则按高度比例
def get_height_scale_factor(target_height: float, base_height: float = BASE_HEIGHT) -> float:
    if target_height < HEIGHT_THRESHOLD:
        return 0.1
    return target_height / base_height


# 按全局步数取命令阶段
def get_current_stage(step_counter: int) -> int:
    if step_counter < STAGE1_END:
        return 1
    if step_counter < STAGE2_END:
        return 2
    return 3


# 6D 命令 [vel_x, vel_y, vel_z, height_F, height_H, angle]
class HoleCommand(CommandTerm):
    cfg: "HoleCommandCfg"

    def __init__(self, cfg: "HoleCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.asset_name]

        self.command_tensor = torch.zeros(self.num_envs, 6, device=self.device)
        self.vel_command_w = self.command_tensor[:, :3]
        self.height_F_command = self.command_tensor[:, 3]
        self.height_H_command = self.command_tensor[:, 4]
        self.angle_command = self.command_tensor[:, 5]

        # 位置/时间表配置 (cfg 优先, 否则用阶段3内置表)
        self.use_position_schedule = cfg.use_position_schedule
        self.position_schedule = cfg.position_schedule
        self.use_height_schedule = cfg.use_height_schedule
        self.height_schedule = cfg.height_schedule
        self.angle_values_tensor = torch.tensor(ANGLE_VALUES, device=self.device)
        self.start_positions = torch.zeros(self.num_envs, 3, device=self.device)
        self.has_printed_current_cmd = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(env_ids)
        t_range = self.cfg.resampling_time_range
        self.time_left[env_ids] = torch.rand(len(env_ids), device=self.device) * (
            t_range[1] - t_range[0]
        ) + t_range[0]

    @property
    def command(self) -> torch.Tensor:
        return self.command_tensor

    # 记录 episode 起点 (位置表按 x 位移查表)
    def _update_start_positions(self, env_ids: torch.Tensor) -> None:
        self.start_positions[env_ids] = self.robot.data.root_link_pos_w[env_ids]

    # 按阶段取可随机采样的高度集合
    def _get_available_heights(self, stage: int) -> Optional[list]:
        if stage == 1:
            return STAGE1_HEIGHT_VALUES
        if stage == 2:
            return STAGE2_HEIGHT_VALUES
        return None

    # 是否使用位置/时间表
    def _should_use_schedule(self, stage: int) -> bool:
        if self.use_position_schedule and self.position_schedule:
            return True
        if self.use_height_schedule and self.height_schedule:
            return True
        return stage == 3

    # 取位置/时间表 (cfg 位置表 > cfg 时间表 > 阶段3内置表)
    def _get_schedule(self, stage: int) -> List[Tuple[float, float, float]]:
        if self.use_position_schedule and self.position_schedule:
            return self.position_schedule
        if self.use_height_schedule and self.height_schedule:
            return self.height_schedule
        if stage == 3:
            return STAGE3_POSITION_SCHEDULE
        return []

    # 采样前后肢高度组合: 避免双低 (一侧低时另一侧必须 >= 阈值)
    def _sample_heights(self, available_heights: list) -> tuple:
        device = self.device
        values = torch.tensor(available_heights, device=device)
        for _ in range(10):
            h_f = values[torch.randint(0, len(available_heights), (1,), device=device)]
            h_h = values[torch.randint(0, len(available_heights), (1,), device=device)]
            if bool(h_f < HEIGHT_THRESHOLD) != bool(h_h < HEIGHT_THRESHOLD):
                return h_f, h_h
            if h_f >= HEIGHT_THRESHOLD and h_h >= HEIGHT_THRESHOLD:
                return h_f, h_f
        fallback = torch.tensor(BASE_HEIGHT, device=device)
        return fallback, fallback

    # 按有效高度定速度
    def _determine_velocity(self, height_F: torch.Tensor, height_H: torch.Tensor) -> float:
        effective_height = min(height_F.item(), height_H.item())
        return BASE_SPEED * get_height_scale_factor(effective_height, BASE_HEIGHT)

    # 按高度组合定角度命令
    def _determine_angle(self, height_F: torch.Tensor, height_H: torch.Tensor) -> float:
        if height_F >= HEIGHT_THRESHOLD and height_H >= HEIGHT_THRESHOLD:
            return 0.0
        idx = torch.randint(0, len(ANGLE_VALUES), (1,), device=self.device)
        return self.angle_values_tensor[idx].item()

    # 按位移查位置表批量更新命令
    def _update_command_from_schedule(self, env_ids: torch.Tensor,
                                      schedule: List[Tuple[float, float, float]]) -> None:
        if not schedule:
            return
        device = self.device
        current_positions = self.robot.data.root_link_pos_w[env_ids]
        start_positions = self.start_positions[env_ids]
        travel_distance = torch.clamp(current_positions[:, 0] - start_positions[:, 0], min=0.0)

        schedule_distances = torch.tensor([s[0] for s in schedule], device=device)
        schedule_hF = torch.tensor([s[1] for s in schedule], device=device)
        schedule_hH = torch.tensor([s[2] for s in schedule], device=device)

        schedule_indices = torch.searchsorted(schedule_distances, travel_distance, side="right") - 1
        schedule_indices = torch.clamp(schedule_indices, min=0, max=len(schedule) - 1)

        height_F = schedule_hF[schedule_indices]
        height_H = schedule_hH[schedule_indices]
        effective_height = torch.minimum(height_F, height_H)
        vel_x = BASE_SPEED * (effective_height / BASE_HEIGHT)

        self.vel_command_w[env_ids, 0] = vel_x
        self.vel_command_w[env_ids, 1] = 0.0
        self.vel_command_w[env_ids, 2] = 0.0
        self.height_F_command[env_ids] = height_F
        self.height_H_command[env_ids] = height_H
        self.angle_command[env_ids] = 0.0
        self.has_printed_current_cmd[env_ids] = False

    # 采样命令: 位置表模式走位移查表, 否则按阶段随机采样
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n_envs = len(env_ids)
        stage = get_current_stage(self._env.common_step_counter)
        use_schedule = self._should_use_schedule(stage)
        schedule = self._get_schedule(stage) if use_schedule else []

        if use_schedule and schedule:
            self._update_start_positions(env_ids)
            self._update_command_from_schedule(env_ids, schedule)
            return

        vel_x = torch.zeros(n_envs, device=self.device)
        height_F = torch.zeros(n_envs, device=self.device)
        height_H = torch.zeros(n_envs, device=self.device)
        angle = torch.zeros(n_envs, device=self.device)
        available_heights = self._get_available_heights(stage) or STAGE2_HEIGHT_VALUES

        for i in range(n_envs):
            if self.cfg.fixed_velocity is not None:
                vel_x[i] = self.cfg.fixed_velocity
            if self.cfg.fixed_height_F is not None:
                height_F[i] = self.cfg.fixed_height_F
            if self.cfg.fixed_height_H is not None:
                height_H[i] = self.cfg.fixed_height_H
            if self.cfg.fixed_height_F is None or self.cfg.fixed_height_H is None:
                sampled_F, sampled_H = self._sample_heights(available_heights)
                if self.cfg.fixed_height_F is None:
                    height_F[i] = sampled_F
                if self.cfg.fixed_height_H is None:
                    height_H[i] = sampled_H
            if self.cfg.fixed_velocity is None:
                vel_x[i] = self._determine_velocity(height_F[i], height_H[i])
            angle[i] = self._determine_angle(height_F[i], height_H[i])

        self.vel_command_w[env_ids, 0] = vel_x
        self.vel_command_w[env_ids, 1] = 0.0
        self.vel_command_w[env_ids, 2] = 0.0
        self.height_F_command[env_ids] = height_F
        self.height_H_command[env_ids] = height_H
        self.angle_command[env_ids] = angle
        self.has_printed_current_cmd[env_ids] = False
        self._update_start_positions(env_ids)

    # 每步更新: 位置表模式下持续按位移查表
    def _update_command(self) -> None:
        stage = get_current_stage(self._env.common_step_counter)
        use_schedule = self._should_use_schedule(stage)
        schedule = self._get_schedule(stage) if use_schedule else []

        if use_schedule and schedule:
            env_ids = torch.arange(self.num_envs, device=self.device)
            self._update_command_from_schedule(env_ids, schedule)
            return

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
        base_vel = self.robot.data.root_link_lin_vel_w[batch].cpu().numpy()
        vel_cmd = self.command_tensor[batch, :3].cpu().numpy()
        scale = self.cfg.viz.scale
        z_off = self.cfg.viz.z_offset
        visualizer.add_arrow(
            start=np.array([base_pos[0], base_pos[1], base_pos[2] + z_off]),
            end=np.array([base_pos[0] + vel_cmd[0] * scale, base_pos[1] + vel_cmd[1] * scale,
                          base_pos[2] + z_off + vel_cmd[2] * scale]),
            color=(1.0, 0.4, 0.2, 0.8),
            label=f"cmd_{batch}",
        )
        visualizer.add_arrow(
            start=np.array([base_pos[0], base_pos[1], base_pos[2] + z_off]),
            end=np.array([base_pos[0] + base_vel[0] * scale, base_pos[1] + base_vel[1] * scale,
                          base_pos[2] + z_off + base_vel[2] * scale]),
            color=(0.2, 0.6, 1.0, 0.8),
            label=f"act_{batch}",
        )


@dataclass(kw_only=True)
class HoleCommandCfg(CommandTermCfg):
    asset_name: str = "robot"
    resampling_time_range: Tuple[float, float] = (4.0, 6.0)
    debug_vis: bool = False
    use_position_schedule: bool = False
    position_schedule: Optional[List[Tuple[float, float, float]]] = None
    use_height_schedule: bool = False
    height_schedule: Optional[List[Tuple[float, float, float]]] = None
    fixed_velocity: Optional[float] = None
    fixed_height_F: Optional[float] = None
    fixed_height_H: Optional[float] = None

    @dataclass
    class VizCfg:
        z_offset: float = 0.1
        scale: float = 1.0

    viz: VizCfg = field(default_factory=VizCfg)
    class_type: type[CommandTerm] = HoleCommand

    def build(self, env: "ManagerBasedRlEnv") -> CommandTerm:
        return self.class_type(self, env)
