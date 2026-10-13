from __future__ import annotations
import torch
from mjlab.entity import Entity
from dataclasses import dataclass, field
from mjlab.managers import CommandTermCfg
from typing import TYPE_CHECKING, Optional, Tuple
from mjlab.managers.command_manager import CommandTerm
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi
from .config import (
    BASE_FREQ,
    BASE_HEIGHT,
    BASE_SPEED,
    COMMAND_DIM,
    ANGLE_VALUES,
    BASE_CURVATURE,
    CMD_CURVATURE_IDS,
    CMD_GAIT_FREQ_IDS,
    CMD_HEIGHT_F_IDS,
    CMD_HEIGHT_H_IDS,
    CMD_OBSTACLE_HEIGHT_IDS,
    CMD_VEL_X_IDS,
    FULL_HEIGHT_VALUES,
    MODE_BOTH_HIGH,
    MODE_BOTH_LOW,
    MODE_FRONT_LOW,
    MODE_HIND_LOW,
    THRESHOLD_HEIGHT,
    get_current_stage,
    heights_for_stage,
    stage_uses_height_course,
    HEIGHT_MIN_SAMPLE_PROBABILITY,
)
from .curriculums import (
    HeightCurriculumState,
    sample_base_heights,
)
from .entity import read_bottom_z, write_hole_bottom_z
from .indices import _MODEL_INDICES, resolve_model_indices

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
        self.height_F_command = self.height_f_command
        self.height_H_command = self.height_h_command
        self.angle_command = torch.zeros(self.num_envs, device=self.device)
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
        self.angle_values_tensor = torch.tensor(ANGLE_VALUES, device=self.device)
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


    # 可视化使用三维速度箭头，策略命令只保留前向速度。
    @property
    def vel_command_w(self) -> torch.Tensor:
        zeros = torch.zeros_like(self.vel_command)
        return torch.stack((self.vel_command, zeros, zeros), dim=-1)


    # 板高先验始终读取实际板体，无板场景使用零占位。
    def _update_obstacle_height_command(self) -> None:
        if self.obstacle_entity is None:
            self.obstacle_height_command.zero_()
        else:
            self.obstacle_height_command[:] = read_bottom_z(self.obstacle_entity)


    # 从已写入的根位姿记录起点，避开复位前的缓存位置；见 docs/SQuRo_Hole_课程复位.md。
    def _update_start_positions(self, env_ids: torch.Tensor) -> None:
        root_pose_ids = self.robot.indexing.free_joint_q_adr
        assert root_pose_ids is not None, "位移诊断要求机器人带浮动根关节"
        self.start_positions[env_ids] = self.robot.data.data.qpos[env_ids][:, root_pose_ids[:3]]


    # 根据阶段获取热身随机高度池。
    def _get_available_heights(self, stage: int) -> Optional[list]:
        return heights_for_stage(stage)


    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n_envs = len(env_ids)  
        current_step = self._env.common_step_counter
        stage = get_current_stage(current_step)
        
        if self.enable_height_course and stage_uses_height_course(stage):
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
        angle = torch.zeros(n_envs, device=self.device)
        
        # 为每个环境采样
        for i in range(n_envs):
            # 检查是否使用固定值
            if self.cfg.fixed_velocity is not None:
                vel_x[i] = self.cfg.fixed_velocity
            if self.cfg.fixed_height_F is not None:
                height_F[i] = self.cfg.fixed_height_F
            if self.cfg.fixed_height_H is not None:
                height_H[i] = self.cfg.fixed_height_H
            if self.cfg.fixed_angle is not None:
                angle[i] = self.cfg.fixed_angle
            
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

            if self.cfg.fixed_angle is None:
                angle[i] = self._determine_angle(height_F[i], height_H[i])
        
        # 更新命令张量
        self.vel_command[env_ids] = vel_x
        self.height_F_command[env_ids] = height_F
        self.height_H_command[env_ids] = height_H
        self.angle_command[env_ids] = angle
        self._update_obstacle_height_command()
        
        # 重置起始位置
        self._update_start_positions(env_ids)


    # 仅在回合重置时采样一次板底高度。
    def _sample_obstacle_height(self, env_ids: torch.Tensor) -> None:
        if self.obstacle_entity is None:
            raise RuntimeError("高度课程需要 Hole1 单板实体")
        lower = self.height_curriculum_state.z_min
        upper = self.height_curriculum_state.z_max
        uniform = lower + torch.rand(len(env_ids), device=self.device) * (upper - lower)
        at_minimum = torch.rand(len(env_ids), device=self.device) < HEIGHT_MIN_SAMPLE_PROBABILITY
        heights = torch.where(at_minimum, torch.full_like(uniform, lower), uniform)
        self.obstacle_bottom_z[env_ids] = heights
        self.obstacle_height_sampled[env_ids] = True
        write_hole_bottom_z(self.obstacle_entity, heights, env_ids)
        self._update_obstacle_height_command()


    # 写入新回合的中性命令，等待 forward 后按躯干位置更新。
    def _set_neutral_command(self, env_ids: torch.Tensor) -> None:
        heights = torch.full((len(env_ids),), BASE_HEIGHT, device=self.device)
        velocity = compute_command_velocity(heights, heights)
        angle = torch.full_like(heights, float(self.cfg.fixed_angle or 0.0))
        if self.cfg.fixed_velocity is not None:
            velocity.fill_(float(self.cfg.fixed_velocity))
        self.vel_command[env_ids] = velocity
        self.height_F_command[env_ids] = heights
        self.height_H_command[env_ids] = heights
        self.angle_command[env_ids] = angle
        self._update_obstacle_height_command()


    # 使用 base_Link 的世界 X 统一查单板高度表。
    def _update_command_from_height_rule(self) -> None:
        if self.obstacle_entity is None:
            raise RuntimeError("高度课程需要 Hole1 单板实体")
        board_cfg = self.obstacle_entity.cfg
        base_x = self.robot.data.root_link_pos_w[:, 0]
        bottom_z = read_bottom_z(self.obstacle_entity)
        front, hind = sample_base_heights(base_x, float(board_cfg.position[0]), float(board_cfg.size[0]), bottom_z,) # type: ignore
        velocity = compute_command_velocity(front, hind)
        angle = torch.zeros_like(front)
        if self.cfg.fixed_velocity is not None:
            velocity.fill_(float(self.cfg.fixed_velocity))
        if self.cfg.fixed_angle is not None:
            angle.fill_(float(self.cfg.fixed_angle))
        self.vel_command[:] = velocity
        self.height_F_command[:] = front
        self.height_H_command[:] = hind
        self.angle_command[:] = angle
        self.obstacle_height_command[:] = bottom_z


    # 采样前后肢高度组合: 两侧独立采样, 四种模式 (双高/前低/后低/双低) 全部覆盖
    def _sample_heights(self, available_heights: list) -> tuple:
        values = torch.tensor(available_heights, device=self.device)
        idx_F = torch.randint(0, len(available_heights), (1,), device=self.device)
        idx_H = torch.randint(0, len(available_heights), (1,), device=self.device)
        return values[idx_F], values[idx_H]


    # 绝对速度 (随机采样路径)
    def _determine_velocity(self, height_F: torch.Tensor, height_H: torch.Tensor) -> float:
        return float(compute_command_velocity(height_F, height_H))


    # 决定角度
    def _determine_angle(self, height_F: torch.Tensor, height_H: torch.Tensor) -> float:
        if height_F >= THRESHOLD_HEIGHT and height_H >= THRESHOLD_HEIGHT:
            return 0.0
        else:
            angle_idx = torch.randint(0, len(ANGLE_VALUES), (1,), device=self.device)
            return self.angle_values_tensor[angle_idx].item()


    def _update_command(self) -> None:
        current_step = self._env.common_step_counter
        stage = get_current_stage(current_step)
        
        if self.enable_height_course and stage_uses_height_course(stage):
            self._update_command_from_height_rule()
        self._update_obstacle_height_command()
        self._log_diagnostics()


    # 按更新后的命令记录模式覆盖与物理方向；口径见 docs/SQuRo_Hole_课程复位.md。
    def _log_diagnostics(self) -> None:
        resolve_model_indices(self.robot)
        front_low = self.height_F_command < THRESHOLD_HEIGHT
        hind_low = self.height_H_command < THRESHOLD_HEIGHT
        groups = (
            ("双高", ~front_low & ~hind_low),
            ("前低后高", front_low & ~hind_low),
            ("前高后低", ~front_low & hind_low),
            ("双低", front_low & hind_low),
        )
        positions = self.robot.data.root_link_pos_w
        relative_x = positions[:, 0] - self.start_positions[:, 0]
        velocity = self.robot.data.root_link_lin_vel_w
        body_ids = [_MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id]
        quaternions = self.robot.data.body_link_quat_w[:, body_ids]
        _, _, yaw = euler_xyz_from_quat(quaternions.reshape(-1, 4))
        yaw = yaw.reshape(self.num_envs, len(body_ids))
        offsets = yaw.new_tensor([-torch.pi / 2, torch.pi / 2])
        heading_error = torch.rad2deg(wrap_to_pi(yaw + offsets)).abs()
        log = self._env.extras["log"]
        if not self.cfg.log_detailed_metrics:
            log["Metrics/课程/阶段"] = positions.new_tensor(get_current_stage(self._env.common_step_counter))
            log["Metrics/运动/侧向速度绝对值均值"] = velocity[:, 1].abs().mean().detach()
            for name, mask in groups:
                log[f"Metrics/{name}/占比"] = mask.float().mean().detach()
                if name in ("前低后高", "前高后低"):
                    mean_heading = (heading_error[:, 0] * mask).sum() / mask.sum().clamp_min(1)
                    log[f"Metrics/{name}/前躯干方向误差均值_度"] = mean_heading.detach()
            return
        values = torch.stack(
            (relative_x, velocity[:, 0], velocity[:, 1], velocity[:, 1].abs(),
             heading_error[:, 0], heading_error[:, 1]), dim=1,
        )
        masks = torch.stack([mask for _, mask in groups]).to(values.dtype)
        counts = masks.sum(dim=1)
        means = (masks @ values) / counts.clamp_min(1).unsqueeze(1)
        fields = ("相对X均值", "前向速度均值", "侧向速度均值", "侧向速度绝对值均值", "前躯干方向误差均值_度", "后躯干方向误差均值_度")
        log["Metrics/位移/起点X均值"] = self.start_positions[:, 0].mean().detach()
        log["Metrics/位移/相对X均值"] = relative_x.mean().detach()
        log["Metrics/位移/相对X最大值"] = relative_x.max().detach()
        for i, (name, _) in enumerate(groups):
            prefix = f"Metrics/{name}"
            log[f"{prefix}/样本数"] = counts[i].detach()
            log[f"{prefix}/占比"] = (counts[i] / self.num_envs).detach()
            for j, field_name in enumerate(fields):
                log[f"{prefix}/{field_name}"] = means[i, j].detach()


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
    resampling_time_range: Tuple[float, float] = (3.0, 4.0)  # 默认值
    debug_vis: bool = False
    fixed_velocity: Optional[float] = None
    fixed_height_F: Optional[float] = None
    fixed_height_H: Optional[float] = None
    fixed_angle: Optional[float] = None  
    enable_height_course: bool = False
    enable_course_progress: bool = True
    log_detailed_metrics: bool = False
    height_curriculum_state: dict = field(default_factory=dict)
    
    @dataclass
    class VizCfg:
        z_offset: float = 0.1
        scale: float = 1.0
    
    viz: VizCfg = field(default_factory=VizCfg)
    class_type: type[CommandTerm] = HoleCommand

    def build(self, env: ManagerBasedRlEnv) -> CommandTerm:
        return self.class_type(self, env)
