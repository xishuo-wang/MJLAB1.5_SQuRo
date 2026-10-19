from __future__ import annotations
import torch
from mjlab.entity import Entity
from dataclasses import dataclass, field
from mjlab.managers import CommandTermCfg
from mjlab.managers.command_manager import CommandTerm
from typing import TYPE_CHECKING, Optional, Tuple, overload
from .config import (
    BASE_FREQ,
    BASE_SPEED,
    BASE_HEIGHT,
    COMMAND_DIM,
    CMD_VEL_X_IDS,
    MODE_BOTH_LOW,
    MODE_HIND_LOW,
    BASE_CURVATURE,
    MODE_FRONT_LOW,
    MODE_BOTH_HIGH,
    CMD_HEIGHT_F_IDS,
    CMD_HEIGHT_H_IDS,
    THRESHOLD_HEIGHT,
    CMD_CURVATURE_IDS,
    CMD_GAIT_FREQ_IDS,
    FULL_HEIGHT_VALUES,
    LOW_POSE_CALIBRATION,
    LOW_POSE_SPEED_RATIO,
    WARMUP_OBSTACLE_HEIGHT,
    CMD_OBSTACLE_HEIGHT_IDS,
    HEIGHT_MIN_SAMPLE_PROBABILITY,
    HIGH_OBSTACLE_EQUAL_PROBABILITY,
)
from .curriculums import (
    get_current_stage,
    heights_for_stage,
    sample_bin_heights,
    HeightCurriculumState,
    compute_height_commands,
    sample_range_bin_heights,
    stage_uses_height_course,
)
from .entity import read_obstacle_bottom_z, write_obstacle_mocap

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv



# 按当前高度命令判定高度模式
@overload
def get_height_mode(height_f: float, height_h: float) -> int: ...



@overload
def get_height_mode(height_f: torch.Tensor, height_h: torch.Tensor) -> torch.Tensor: ...



# 标量可视化与批量参考查询共用四种模式判定
def get_height_mode(height_f: float | torch.Tensor, height_h: float | torch.Tensor) -> int | torch.Tensor:
    if isinstance(height_f, torch.Tensor) and isinstance(height_h, torch.Tensor):
        front_low = height_f < THRESHOLD_HEIGHT
        hind_low = height_h < THRESHOLD_HEIGHT
        return torch.where(
            front_low,
            torch.where(hind_low, MODE_BOTH_LOW, MODE_FRONT_LOW),
            torch.where(hind_low, MODE_HIND_LOW, MODE_BOTH_HIGH),
        )
    front_low = height_f < THRESHOLD_HEIGHT
    hind_low = height_h < THRESHOLD_HEIGHT
    if front_low and hind_low:
        return MODE_BOTH_LOW
    if front_low:
        return MODE_FRONT_LOW
    if hind_low:
        return MODE_HIND_LOW
    return MODE_BOTH_HIGH



# 计算从 40 毫米的零到最低标定高度的满强度
def get_low_pose_intensity(height: torch.Tensor) -> torch.Tensor:
    low_min = LOW_POSE_CALIBRATION[0][0]
    return ((THRESHOLD_HEIGHT - height) / (THRESHOLD_HEIGHT - low_min)).clamp(0.0, 1.0)



# 前后高度分别缩放取平均，并按最低高度连续减速
def compute_cmd_vel_x(height_f, height_h):
    heights = torch.stack(torch.broadcast_tensors(height_f, height_h), dim=-1)
    low_min = LOW_POSE_CALIBRATION[0][0]
    low_max = LOW_POSE_CALIBRATION[-2][0]
    low_scale = ((heights - low_min) / (low_max - low_min)).clamp(min=0.0) * (low_max / BASE_HEIGHT)
    scale = torch.where(heights < low_max, low_scale, heights / BASE_HEIGHT)
    low_factor = get_low_pose_intensity(heights.amin(dim=-1))
    speed_ratio = 1.0 - (1.0 - LOW_POSE_SPEED_RATIO) * low_factor
    return BASE_SPEED * scale.mean(dim=-1) * speed_ratio



class HoleCommand(CommandTerm):
    cfg: HoleCommandCfg 
    def __init__(self, cfg: HoleCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)  
        self.robot: Entity = env.scene[cfg.asset_name]
        self.command_buffer = torch.zeros(self.num_envs, COMMAND_DIM, device=self.device)
        self.command_vel_x = self.command_buffer[:, CMD_VEL_X_IDS]
        self.command_height_f = self.command_buffer[:, CMD_HEIGHT_F_IDS]
        self.command_height_h = self.command_buffer[:, CMD_HEIGHT_H_IDS]
        self.command_gait_freq = self.command_buffer[:, CMD_GAIT_FREQ_IDS]
        self.command_curvature = self.command_buffer[:, CMD_CURVATURE_IDS]
        self.command_obstacle_height = self.command_buffer[:, CMD_OBSTACLE_HEIGHT_IDS]
        self.command_gait_freq.fill_(BASE_FREQ)
        self.command_curvature.fill_(BASE_CURVATURE)
        self.enable_height_course = cfg.enable_height_course
        self.adaptive_sampling = True
        self.height_curriculum_state = HeightCurriculumState()
        if cfg.height_curriculum_state:
            self.height_curriculum_state.load_state_dict(cfg.height_curriculum_state)
        self.obstacle = env.scene.entities.get("obstacle")
        self.obstacle_sampled_bottom_z = torch.full((self.num_envs,), self.height_curriculum_state.z_min, device=self.device)
        self.obstacle_height_sampled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.obstacle_equal_height = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.course_episode_eligible = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._allow_obstacle_resample = True
        self.start_positions = torch.zeros(self.num_envs, 3, device=self.device)
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(env_ids)
        self._update_start_positions(env_ids)
        resampling_time_range = self.cfg.resampling_time_range
        self.time_left[env_ids] = torch.rand(len(env_ids), device=self.device) * (resampling_time_range[1] - resampling_time_range[0]) + resampling_time_range[0]
        self._allow_obstacle_resample = False


    # 覆写基类抽象 property
    @property
    def command(self) -> torch.Tensor:
        return self.command_buffer


    # 可视化使用三维速度箭头
    @property
    def viz_cmd_vel(self) -> torch.Tensor:
        zeros = torch.zeros_like(self.command_vel_x)
        return torch.stack((self.command_vel_x, zeros, zeros), dim=-1)


    # 有板时读取真实板底，无板时使用100毫米占位
    def _update_obstacle_height_command(self) -> None:
        if self.obstacle is None:
            self.command_obstacle_height.fill_(WARMUP_OBSTACLE_HEIGHT)
        else:
            self.command_obstacle_height[:] = read_obstacle_bottom_z(self.obstacle)


    # 从已写入的根位姿记录起点，避开复位前的缓存位置
    def _update_start_positions(self, env_ids: torch.Tensor) -> None:
        root_pose_ids = self.robot.indexing.free_joint_q_adr
        assert root_pose_ids is not None, "位移诊断要求机器人带浮动根关节"
        self.start_positions[env_ids] = self.robot.data.data.qpos[env_ids][:, root_pose_ids[:3]]


    # 重采样高度、速度和障碍高度命令
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        current_step = self._env.common_step_counter
        stage = get_current_stage(current_step)
        
        if self.enable_height_course and stage_uses_height_course(stage) and self.obstacle is not None:
            if self._allow_obstacle_resample:
                self._sample_obstacle_height(env_ids)
                self._set_neutral_command(env_ids)
                self._update_start_positions(env_ids)
            return
        
        # 使用随机采样
        available_heights = heights_for_stage(stage)
        if available_heights is None:
            # 固定和随机命令回放仍使用完整高度池
            available_heights = FULL_HEIGHT_VALUES

        # 整批采样: 未固定的一侧独立采样, 固定的一侧直接覆盖
        sampled_height_f, sampled_height_h = self._sample_heights(available_heights, len(env_ids))
        if self.cfg.fixed_height_f is not None:
            sampled_height_f.fill_(self.cfg.fixed_height_f)
        if self.cfg.fixed_height_h is not None:
            sampled_height_h.fill_(self.cfg.fixed_height_h)
        height_f, height_h = sampled_height_f, sampled_height_h

        # 速度: 固定值优先, 否则按高度命令整批计算
        if self.cfg.fixed_velocity is not None:
            vel_x = torch.full((len(env_ids),), self.cfg.fixed_velocity, device=self.device)
        else:
            vel_x = compute_cmd_vel_x(height_f, height_h)

        # 更新命令张量
        self.command_vel_x[env_ids] = vel_x
        self.command_height_f[env_ids] = height_f
        self.command_height_h[env_ids] = height_h
        self._update_obstacle_height_command()
        
        # 重置起始位置
        self._update_start_positions(env_ids)


    # 仅在回合重置时采样一次板底高度
    def _sample_obstacle_height(self, env_ids: torch.Tensor) -> None:
        if self.obstacle is None:
            raise RuntimeError("高度课程需要受限空间实体")
        state = self.height_curriculum_state
        lower = state.z_min
        upper = state.z_max
        count = len(env_ids)
        if lower >= upper - 1e-9:
            heights = torch.full((count,), lower, device=self.device)
        elif state.final_level_passed:
            # 最终档: 按挡位权重在全部高度挡位内采样
            heights = sample_bin_heights(count, state.bin_weights, self.device)
        else:
            # 普通模式: 30% 固定最低档, 其余 70% 按挡位成功率加权
            uniform = lower + torch.rand(count, device=self.device) * (upper - lower)
            at_minimum = torch.rand(count, device=self.device) < HEIGHT_MIN_SAMPLE_PROBABILITY
            adaptive = (sample_range_bin_heights(count, state.bin_weights, lower, upper, self.device) if self.adaptive_sampling else uniform)
            heights = torch.where(at_minimum, torch.full_like(uniform, lower), adaptive)
        self.obstacle_sampled_bottom_z[env_ids] = heights
        self.obstacle_height_sampled[env_ids] = True
        self._sample_high_obstacle_mode(env_ids)
        write_obstacle_mocap(self.obstacle, heights, env_ids)
        self._update_obstacle_height_command()


    # 高板姿态每回合独立采样一次，显式配置可固定等高或分段
    def _sample_high_obstacle_mode(self, env_ids: torch.Tensor) -> None:
        fixed = self.cfg.high_obstacle_equal_height
        if fixed is None:
            values = torch.rand(len(env_ids), device=self.device) < HIGH_OBSTACLE_EQUAL_PROBABILITY
        else:
            values = torch.full((len(env_ids),), fixed, device=self.device, dtype=torch.bool)
        self.obstacle_equal_height[env_ids] = values


    # 写入新回合的中性命令，等待 forward 后按躯干位置更新
    def _set_neutral_command(self, env_ids: torch.Tensor) -> None:
        heights = torch.full((len(env_ids),), BASE_HEIGHT, device=self.device)
        velocity = compute_cmd_vel_x(heights, heights)
        if self.cfg.fixed_velocity is not None:
            velocity.fill_(float(self.cfg.fixed_velocity))
        self.command_vel_x[env_ids] = velocity
        self.command_height_f[env_ids] = heights
        self.command_height_h[env_ids] = heights
        self._update_obstacle_height_command()


    # 使用 base_Link 的世界 X 统一查单板高度表
    def _update_command_by_height_rule(self) -> None:
        if self.obstacle is None:
            raise RuntimeError("高度课程需要受限空间实体")
        obstacle_cfg = self.obstacle.cfg
        base_x = self.robot.data.root_link_pos_w[:, 0]
        bottom_z = read_obstacle_bottom_z(self.obstacle)
        front, hind = compute_height_commands(base_x, float(obstacle_cfg.position[0]), float(obstacle_cfg.size[0]), bottom_z, self.obstacle_equal_height,) # type: ignore
        velocity = compute_cmd_vel_x(front, hind)
        if self.cfg.fixed_velocity is not None:
            velocity.fill_(float(self.cfg.fixed_velocity))
        self.command_vel_x[:] = velocity
        self.command_height_f[:] = front
        self.command_height_h[:] = hind
        self.command_obstacle_height[:] = bottom_z


    # 整批采样前后肢高度组合，两侧独立采样
    def _sample_heights(self, available_heights: list, count: int) -> tuple:
        values = torch.tensor(available_heights, device=self.device)
        idx_f = torch.randint(0, len(available_heights), (count,), device=self.device)
        idx_h = torch.randint(0, len(available_heights), (count,), device=self.device)
        return values[idx_f], values[idx_h]


    # 按当前课程阶段更新高度和速度命令
    def _update_command(self) -> None:
        current_step = self._env.common_step_counter
        stage = get_current_stage(current_step)
        
        if self.enable_height_course and stage_uses_height_course(stage) and self.obstacle is not None:
            self._update_command_by_height_rule()
        self._update_obstacle_height_command()


    # 回合重置时重新锚定位移起点
    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:  # type: ignore[override]
        self._allow_obstacle_resample = True
        extras = super().reset(env_ids)
        self._allow_obstacle_resample = False
        monitor = getattr(self._env, "_pass_monitor", None)
        if monitor is not None:
            monitor.reset(env_ids)
        self.course_episode_eligible[env_ids if env_ids is not None else slice(None)] = True
        if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
            self._update_start_positions(env_ids)
        return extras


    # 可视化命令速度和实际速度
    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        if not self.cfg.debug_vis:
            return
            
        batch = visualizer.env_idx
        if batch >= self.num_envs:
            return
        
        base_pos = self.robot.data.root_link_pos_w[batch].cpu().numpy()
        actual_vel = self.robot.data.root_link_lin_vel_w[batch].cpu().numpy()
        cmd_vel = self.viz_cmd_vel[batch].cpu().numpy()
        if torch.norm(self.robot.data.root_link_pos_w[batch]) < 1e-6:
            return
        
        scale = self.cfg.viz.scale
        z_offset = self.cfg.viz.z_offset
        arrow_width = 0.01
        cmd_start = base_pos + [0, 0, z_offset]
        cmd_end = cmd_start + cmd_vel * scale
        visualizer.add_arrow(cmd_start, cmd_end, color=(0.2, 0.2, 0.8, 0.8), width=arrow_width)
        actual_end = cmd_start + actual_vel * scale
        visualizer.add_arrow(cmd_start, actual_end, color=(0.2, 0.8, 0.2, 0.8), width=arrow_width)


    def _update_metrics(self) -> None:
        pass



@dataclass(kw_only=True)  
class HoleCommandCfg(CommandTermCfg):
    asset_name: str = "robot"
    resampling_time_range: Tuple[float, float] = (3.0, 4.0)
    debug_vis: bool = False
    fixed_velocity: Optional[float] = None
    fixed_height_f: Optional[float] = None
    fixed_height_h: Optional[float] = None
    enable_height_course: bool = False
    high_obstacle_equal_height: Optional[bool] = None
    height_curriculum_state: dict = field(default_factory=dict)
    
    @dataclass
    class VizCfg:
        z_offset: float = 0.1
        scale: float = 1.0
    
    viz: VizCfg = field(default_factory=VizCfg)
    class_type: type[CommandTerm] = HoleCommand

    def build(self, env: ManagerBasedRlEnv) -> CommandTerm:
        return self.class_type(self, env)
