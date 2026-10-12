from __future__ import annotations
import torch
from mjlab.entity import Entity
from dataclasses import dataclass, field
from mjlab.managers import CommandTermCfg
from mjlab.managers.command_manager import CommandTerm
from typing import TYPE_CHECKING, Optional, Tuple, List
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi
from .config import (
    BASE_SPEED,
    BASE_HEIGHT,
    ANGLE_VALUES,
    CMD_ANGLE_IDS,
    CMD_VEL_IDS,
    CMD_VEL_X_IDS,
    CMD_VEL_Y_IDS,
    CMD_VEL_Z_IDS,
    CMD_HEIGHT_F_IDS,
    CMD_HEIGHT_H_IDS,
    MODE_BOTH_HIGH,
    MODE_BOTH_LOW,
    MODE_FRONT_LOW,
    MODE_HIND_LOW,
    THRESHOLD_HEIGHT,
    FULL_HEIGHT_VALUES,
    get_current_stage,
    heights_for_stage,
    stage_uses_schedule,
)
from .indices import _MODEL_INDICES, resolve_model_indices

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv



# 第三阶段固定位置表（基于移动距离, 任务专属: 位移 → 前肢高度, 后肢高度）
STAGE3_POSITION_SCHEDULE = [
    (0.0, 0.02, 0.05),
    (0.2, 0.055, 0.02),
    (0.32, 0.055, 0.055),
    (0.4, 0.04, 0.04),
    (0.8, 0.055, 0.055),
    (1.0, 0.02, 0.05),
    (1.2, 0.055, 0.02),
    (1.32, 0.055, 0.055),
]



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
        self.command_tensor = torch.zeros(self.num_envs, 6, device=self.device)
        self.vel_command_w = self.command_tensor[:, CMD_VEL_IDS]
        self.height_F_command = self.command_tensor[:, CMD_HEIGHT_F_IDS]  # 前肢高度命令
        self.height_H_command = self.command_tensor[:, CMD_HEIGHT_H_IDS]  # 后肢高度命令
        self.angle_command = self.command_tensor[:, CMD_ANGLE_IDS]        # 角度命令
        self.has_printed_current_cmd = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.use_height_schedule = cfg.use_height_schedule
        self.height_schedule = cfg.height_schedule or [] 
        self.use_position_schedule = cfg.use_position_schedule
        self.position_schedule = cfg.position_schedule or []
        self.angle_values_tensor = torch.tensor(ANGLE_VALUES, device=self.device)
        self.start_positions = torch.zeros(self.num_envs, 3, device=self.device)
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(env_ids)
        self._update_start_positions(env_ids)
        resampling_time_range = self.cfg.resampling_time_range
        self.time_left[env_ids] = torch.rand(len(env_ids), device=self.device) * (resampling_time_range[1] - resampling_time_range[0]) + resampling_time_range[0]


    @property
    def command(self) -> torch.Tensor:
        return self.command_tensor


    # 从已写入的根位姿记录起点，避开复位前的缓存位置；见 docs/SQuRo_Hole_课程复位.md。
    def _update_start_positions(self, env_ids: torch.Tensor) -> None:
        root_pose_ids = self.robot.indexing.free_joint_q_adr
        assert root_pose_ids is not None, "位置表要求机器人带浮动根关节"
        self.start_positions[env_ids] = self.robot.data.data.qpos[env_ids][:, root_pose_ids[:3]]


    # 根据阶段获取可用的高度值列表 (阶段 1 正常档 / 阶段 2 含低高度全档 / 阶段 3+ 位置表)
    def _get_available_heights(self, stage: int) -> Optional[list]:
        return heights_for_stage(stage)

    
    def _should_use_schedule(self, stage: int) -> bool:
        if self.use_position_schedule and self.position_schedule:
            return True
        
        if self.use_height_schedule and self.height_schedule:
            return True
        
        if stage_uses_schedule(stage) and self.cfg.stage_schedule_fallback:
            return True
        
        return False

    
    def _get_schedule(self, stage: int) -> List[Tuple[float, float, float]]:
        # 最高优先级：cfg 中配置的位置表
        if self.use_position_schedule and self.position_schedule:
            return self.position_schedule
        
        # 次优先级：cfg 中配置的时间表
        if self.use_height_schedule and self.height_schedule:
            return self.height_schedule
        
        # 阶段 3/4：使用默认位置表
        if stage_uses_schedule(stage) and self.cfg.stage_schedule_fallback:
            return STAGE3_POSITION_SCHEDULE
        
        return []


    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n_envs = len(env_ids)  
        current_step = self._env.common_step_counter
        stage = get_current_stage(current_step)
        
        # 判断是否使用时间表
        use_schedule = self._should_use_schedule(stage)
        schedule = self._get_schedule(stage) if use_schedule else []
        
        if use_schedule and schedule:
            # 使用时间表更新命令
            self._update_command_from_schedule(env_ids, schedule)
            return
        
        # 使用随机采样
        available_heights = self._get_available_heights(stage)
        if available_heights is None:
            # 阶段 3/4 走位置表, 不该到这里; 兜底用全高度档
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
        self.vel_command_w[env_ids, CMD_VEL_X_IDS] = vel_x
        self.vel_command_w[env_ids, CMD_VEL_Y_IDS] = 0
        self.vel_command_w[env_ids, CMD_VEL_Z_IDS] = 0
        self.height_F_command[env_ids] = height_F
        self.height_H_command[env_ids] = height_H
        self.angle_command[env_ids] = angle
        self.command_tensor[env_ids, CMD_VEL_IDS] = self.vel_command_w[env_ids]
        self.command_tensor[env_ids, CMD_HEIGHT_F_IDS] = height_F
        self.command_tensor[env_ids, CMD_HEIGHT_H_IDS] = height_H
        self.command_tensor[env_ids, CMD_ANGLE_IDS] = angle
        self.has_printed_current_cmd[env_ids] = False
        
        # 重置起始位置
        self._update_start_positions(env_ids)


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


    # 根据位置表更新命令
    def _update_command_from_schedule(self, env_ids: torch.Tensor, schedule: List[Tuple[float, float, float]]) -> None:
        if not schedule:
            return
        
        n_envs = len(env_ids)
        device = self.device
        
        # 获取当前位置
        current_positions = self.robot.data.root_link_pos_w[env_ids]  # [n_envs, 3]
        start_positions = self.start_positions[env_ids]  # [n_envs, 3]
        
        # 计算移动距离（X方向，只考虑前进方向）
        travel_distance = current_positions[:, 0] - start_positions[:, 0]  # [n_envs]
        travel_distance = torch.clamp(travel_distance, min=0.0)  # 不允许负距离
        
        # 将位置表转换为张量
        schedule_distances = torch.tensor([s[0] for s in schedule], device=device)  # [len(schedule)]
        schedule_hF = torch.tensor([s[1] for s in schedule], device=device)         # [len(schedule)]
        schedule_hH = torch.tensor([s[2] for s in schedule], device=device)         # [len(schedule)]
        
        # 向量化查找：对于每个距离，找到最后一个 <= 当前距离的索引
        schedule_indices = torch.searchsorted(schedule_distances, travel_distance, side='right') - 1
        schedule_indices = torch.clamp(schedule_indices, min=0, max=len(schedule) - 1)
        
        # 批量获取高度值
        height_F = schedule_hF[schedule_indices]  # [n_envs]
        height_H = schedule_hH[schedule_indices]  # [n_envs]
        
        # 速度与随机采样路径共用同一公式 (见 compute_command_velocity)
        vel_x = compute_command_velocity(height_F, height_H)
        
        # 显式值优先于位置表: 显式指定的字段在位置表算完后覆盖
        if self.cfg.fixed_height_F is not None:
            height_F = torch.full_like(height_F, float(self.cfg.fixed_height_F))
        if self.cfg.fixed_height_H is not None:
            height_H = torch.full_like(height_H, float(self.cfg.fixed_height_H))
        if self.cfg.fixed_velocity is not None:
            vel_x = torch.full_like(vel_x, float(self.cfg.fixed_velocity))
        
        # 批量计算角度
        both_high = (height_F >= THRESHOLD_HEIGHT) & (height_H >= THRESHOLD_HEIGHT)
        angle = torch.zeros(n_envs, device=device)
        if self.cfg.fixed_angle is not None:
            angle = torch.full_like(angle, float(self.cfg.fixed_angle))
        
        # 批量更新命令张量
        self.vel_command_w[env_ids, CMD_VEL_X_IDS] = vel_x
        self.vel_command_w[env_ids, CMD_VEL_Y_IDS] = 0
        self.vel_command_w[env_ids, CMD_VEL_Z_IDS] = 0
        self.height_F_command[env_ids] = height_F
        self.height_H_command[env_ids] = height_H
        self.angle_command[env_ids] = angle
        self.command_tensor[env_ids, CMD_VEL_IDS] = self.vel_command_w[env_ids]
        self.command_tensor[env_ids, CMD_HEIGHT_F_IDS] = height_F
        self.command_tensor[env_ids, CMD_HEIGHT_H_IDS] = height_H
        self.command_tensor[env_ids, CMD_ANGLE_IDS] = angle
        self.has_printed_current_cmd[env_ids] = False


    def _update_command(self) -> None:
        current_step = self._env.common_step_counter
        stage = get_current_stage(current_step)
        
        # 判断是否使用时间表
        use_schedule = self._should_use_schedule(stage)
        schedule = self._get_schedule(stage) if use_schedule else []
        
        # 使用时间表模式，每个时间步都需要更新命令
        if use_schedule and schedule:
            env_ids = torch.arange(self.num_envs, device=self.device)
            self._update_command_from_schedule(env_ids, schedule)
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
        values = torch.stack(
            (relative_x, velocity[:, 0], velocity[:, 1], velocity[:, 1].abs(),
             heading_error[:, 0], heading_error[:, 1]), dim=1,
        )
        masks = torch.stack([mask for _, mask in groups]).to(values.dtype)
        counts = masks.sum(dim=1)
        means = (masks @ values) / counts.clamp_min(1).unsqueeze(1)
        fields = ("相对X均值", "前向速度均值", "侧向速度均值", "侧向速度绝对值均值", "前躯干方向误差均值_度", "后躯干方向误差均值_度")
        log = self._env.extras["log"]
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
        extras = super().reset(env_ids)
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
    use_height_schedule: bool = False  
    height_schedule: List[Tuple[float, float, float]] = field(default_factory=list)  # [(开始时间, 前肢高度, 后肢高度), ...]
    use_position_schedule: bool = False  # 是否使用位置表
    position_schedule: List[Tuple[float, float, float]] = field(default_factory=list)  # [(距离阈值, 前肢高度, 后肢高度), ...]
    stage_schedule_fallback: bool = True        # 阶段 3 是否兜底使用内置位置表; 回放侧选择 fixed/random 时置 False, 保证 play 优先
    
    @dataclass
    class VizCfg:
        z_offset: float = 0.1
        scale: float = 1.0
    
    viz: VizCfg = field(default_factory=VizCfg)
    class_type: type[CommandTerm] = HoleCommand

    def build(self, env: ManagerBasedRlEnv) -> CommandTerm:
        return self.class_type(self, env)
