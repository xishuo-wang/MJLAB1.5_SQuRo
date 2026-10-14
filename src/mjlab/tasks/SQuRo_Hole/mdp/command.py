from __future__ import annotations
import torch
from mjlab.entity import Entity
from dataclasses import dataclass, field
from mjlab.managers import CommandTermCfg
from typing import TYPE_CHECKING, Optional, Tuple
from mjlab.managers.command_manager import CommandTerm
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
    WARMUP_OBSTACLE_HEIGHT,
    CMD_OBSTACLE_HEIGHT_IDS,
    HEIGHT_MIN_SAMPLE_PROBABILITY,
)
from .curriculums import (
    sample_base_heights,
    HeightCurriculumState,
    get_current_stage,
    heights_for_stage,
    stage_uses_height_course,
)
from .entity import read_bottom_z, write_hole_bottom_z

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv



# 按当前高度命令判定高度模式
def get_height_mode(height_F: float, height_H: float) -> int:
    front_low = height_F < THRESHOLD_HEIGHT
    hind_low = height_H < THRESHOLD_HEIGHT
    if front_low and hind_low:
        return MODE_BOTH_LOW
    if front_low:
        return MODE_FRONT_LOW
    if hind_low:
        return MODE_HIND_LOW
    return MODE_BOTH_HIGH



# 根据高度命令生成前向速度：v = BASE_SPEED × min(h_F,h_H)/BASE_HEIGHT × (2 − 低侧数)/2
def compute_command_velocity(height_F, height_H):
    lower = torch.minimum(height_F, height_H)
    height_scale = lower / BASE_HEIGHT
    n_low = (height_F < THRESHOLD_HEIGHT).long() + (height_H < THRESHOLD_HEIGHT).long()
    speed_scale = (2.0 - n_low.float()) / 2.0
    return BASE_SPEED * height_scale * speed_scale



class HoleCommand(CommandTerm):
    cfg: HoleCommandCfg 
    def __init__(self, cfg: HoleCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)  
        self.robot: Entity = env.scene[cfg.asset_name]
        self.command_tensor = torch.zeros(self.num_envs, COMMAND_DIM, device=self.device)
        self.vel_command = self.command_tensor[:, CMD_VEL_X_IDS]
        self.height_f_command = self.command_tensor[:, CMD_HEIGHT_F_IDS]
        self.height_h_command = self.command_tensor[:, CMD_HEIGHT_H_IDS]
        self.gait_freq_command = self.command_tensor[:, CMD_GAIT_FREQ_IDS]
        self.curvature_command = self.command_tensor[:, CMD_CURVATURE_IDS]
        self.obstacle_height_command = self.command_tensor[:, CMD_OBSTACLE_HEIGHT_IDS]
        self.gait_freq_command.fill_(BASE_FREQ)
        self.curvature_command.fill_(BASE_CURVATURE)
        self.enable_height_course = cfg.enable_height_course
        self.enable_course_progress = cfg.enable_course_progress
        self.height_curriculum_state = HeightCurriculumState()
        if cfg.height_curriculum_state:
            self.height_curriculum_state.load_state_dict(cfg.height_curriculum_state)
        self.obstacle_entity = env.scene.entities.get("hole1")
        self.obstacle_bottom_z = torch.full((self.num_envs,), self.height_curriculum_state.z_min, device=self.device)
        self.obstacle_height_sampled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._allow_obstacle_resample = True
        self.start_positions = torch.zeros(self.num_envs, 3, device=self.device)
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(env_ids)
        self._update_start_positions(env_ids)
        resampling_time_range = self.cfg.resampling_time_range
        self.time_left[env_ids] = torch.rand(len(env_ids), device=self.device) * (resampling_time_range[1] - resampling_time_range[0]) + resampling_time_range[0]
        self._allow_obstacle_resample = False


    @property
    def command(self) -> torch.Tensor:
        return self.command_tensor


    # 可视化使用三维速度箭头
    @property
    def vel_command_w(self) -> torch.Tensor:
        zeros = torch.zeros_like(self.vel_command)
        return torch.stack((self.vel_command, zeros, zeros), dim=-1)


    # 有板时读取真实板底，无板时使用100毫米占位
    def _update_obstacle_height_command(self) -> None:
        if self.obstacle_entity is None:
            self.obstacle_height_command.fill_(WARMUP_OBSTACLE_HEIGHT)
        else:
            self.obstacle_height_command[:] = read_bottom_z(self.obstacle_entity)


    # 从已写入的根位姿记录起点，避开复位前的缓存位置
    def _update_start_positions(self, env_ids: torch.Tensor) -> None:
        root_pose_ids = self.robot.indexing.free_joint_q_adr
        assert root_pose_ids is not None, "位移诊断要求机器人带浮动根关节"
        self.start_positions[env_ids] = self.robot.data.data.qpos[env_ids][:, root_pose_ids[:3]]


    # 根据阶段获取随机高度采样池
    def _get_available_heights(self, stage: int) -> Optional[list]:
        return heights_for_stage(stage)


    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n_envs = len(env_ids)  
        current_step = self._env.common_step_counter
        stage = get_current_stage(current_step)
        
        if self.enable_height_course and stage_uses_height_course(stage) and self.obstacle_entity is not None:
            if self._allow_obstacle_resample:
                self._sample_obstacle_height(env_ids)
                self._set_neutral_command(env_ids)
                self._update_start_positions(env_ids)
            return
        
        # 使用随机采样
        available_heights = self._get_available_heights(stage)
        if available_heights is None:
            # 固定和随机命令回放仍使用完整高度池。
            available_heights = FULL_HEIGHT_VALUES
        
        # 初始化命令张量
        vel_x = torch.zeros(n_envs, device=self.device)
        height_F = torch.zeros(n_envs, device=self.device)
        height_H = torch.zeros(n_envs, device=self.device)
        
        # 为每个环境采样
        for i in range(n_envs):
            # 检查是否使用固定值
            if self.cfg.fixed_velocity is not None:
                vel_x[i] = self.cfg.fixed_velocity
            if self.cfg.fixed_height_F is not None:
                height_F[i] = self.cfg.fixed_height_F
            if self.cfg.fixed_height_H is not None:
                height_H[i] = self.cfg.fixed_height_H
            
            # 如果固定值未设置，则使用采样逻辑
            if self.cfg.fixed_height_F is None or self.cfg.fixed_height_H is None:
                sampled_height_F, sampled_height_H = self._sample_heights(available_heights)
                if self.cfg.fixed_height_F is None:
                    height_F[i] = sampled_height_F
                if self.cfg.fixed_height_H is None:
                    height_H[i] = sampled_height_H
            
            # 如果固定速度未设置，则根据高度计算速度
            if self.cfg.fixed_velocity is None:
                vel_x[i] = self._determine_velocity(height_F[i], height_H[i])

        # 更新命令张量
        self.vel_command[env_ids] = vel_x
        self.height_f_command[env_ids] = height_F
        self.height_h_command[env_ids] = height_H
        self._update_obstacle_height_command()
        
        # 重置起始位置
        self._update_start_positions(env_ids)


    # 仅在回合重置时采样一次板底高度
    def _sample_obstacle_height(self, env_ids: torch.Tensor) -> None:
        if self.obstacle_entity is None:
            raise RuntimeError("高度课程需要受限空间实体")
        lower = self.height_curriculum_state.z_min
        upper = self.height_curriculum_state.z_max
        uniform = lower + torch.rand(len(env_ids), device=self.device) * (upper - lower)
        at_minimum = torch.rand(len(env_ids), device=self.device) < HEIGHT_MIN_SAMPLE_PROBABILITY
        heights = torch.where(at_minimum, torch.full_like(uniform, lower), uniform)
        self.obstacle_bottom_z[env_ids] = heights
        self.obstacle_height_sampled[env_ids] = True
        write_hole_bottom_z(self.obstacle_entity, heights, env_ids)
        self._update_obstacle_height_command()


    # 写入新回合的中性命令，等待 forward 后按躯干位置更新
    def _set_neutral_command(self, env_ids: torch.Tensor) -> None:
        heights = torch.full((len(env_ids),), BASE_HEIGHT, device=self.device)
        velocity = compute_command_velocity(heights, heights)
        if self.cfg.fixed_velocity is not None:
            velocity.fill_(float(self.cfg.fixed_velocity))
        self.vel_command[env_ids] = velocity
        self.height_f_command[env_ids] = heights
        self.height_h_command[env_ids] = heights
        self._update_obstacle_height_command()


    # 使用 base_Link 的世界 X 统一查单板高度表
    def _update_command_from_height_rule(self) -> None:
        if self.obstacle_entity is None:
            raise RuntimeError("高度课程需要受限空间实体")
        board_cfg = self.obstacle_entity.cfg
        base_x = self.robot.data.root_link_pos_w[:, 0]
        bottom_z = read_bottom_z(self.obstacle_entity)
        front, hind = sample_base_heights(base_x, float(board_cfg.position[0]), float(board_cfg.size[0]), bottom_z,) # type: ignore
        velocity = compute_command_velocity(front, hind)
        if self.cfg.fixed_velocity is not None:
            velocity.fill_(float(self.cfg.fixed_velocity))
        self.vel_command[:] = velocity
        self.height_f_command[:] = front
        self.height_h_command[:] = hind
        self.obstacle_height_command[:] = bottom_z


    # 采样前后肢高度组合: 两侧独立采样
    def _sample_heights(self, available_heights: list) -> tuple:
        values = torch.tensor(available_heights, device=self.device)
        idx_F = torch.randint(0, len(available_heights), (1,), device=self.device)
        idx_H = torch.randint(0, len(available_heights), (1,), device=self.device)
        return values[idx_F], values[idx_H]


    # 绝对速度 (随机采样路径)
    def _determine_velocity(self, height_F: torch.Tensor, height_H: torch.Tensor) -> float:
        return float(compute_command_velocity(height_F, height_H))


    def _update_command(self) -> None:
        current_step = self._env.common_step_counter
        stage = get_current_stage(current_step)
        
        if self.enable_height_course and stage_uses_height_course(stage) and self.obstacle_entity is not None:
            self._update_command_from_height_rule()
        self._update_obstacle_height_command()


    # 回合重置: 重新锚定位移起点
    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:  # type: ignore[override]
        self._allow_obstacle_resample = True
        extras = super().reset(env_ids)
        self._allow_obstacle_resample = False
        monitor = getattr(self._env, "_hole_passage_monitor", None)
        if monitor is not None:
            monitor.reset(env_ids)
        if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
            self._update_start_positions(env_ids)
        return extras


    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        if not self.cfg.debug_vis:
            return
            
        batch = visualizer.env_idx
        if batch >= self.num_envs:
            return
        
        base_pos = self.robot.data.root_link_pos_w[batch].cpu().numpy()
        actual_vel = self.robot.data.root_link_lin_vel_w[batch].cpu().numpy()
        cmd_vel = self.vel_command_w[batch].cpu().numpy()
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
    fixed_height_F: Optional[float] = None
    fixed_height_H: Optional[float] = None
    enable_height_course: bool = False
    enable_course_progress: bool = True
    height_curriculum_state: dict = field(default_factory=dict)
    
    @dataclass
    class VizCfg:
        z_offset: float = 0.1
        scale: float = 1.0
    
    viz: VizCfg = field(default_factory=VizCfg)
    class_type: type[CommandTerm] = HoleCommand

    def build(self, env: ManagerBasedRlEnv) -> CommandTerm:
        return self.class_type(self, env)
