from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple, List
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers import CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

from .config import (
    ANGLE_VALUES,
    BASE_HEIGHT,
    BASE_SPEED,
    FULL_HEIGHT_VALUES,
    THRESHOLD_HEIGHT,
    get_current_stage,
    heights_for_stage,
    stage_uses_schedule,
)


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


# 命令层的四种高度模式: 按前后肢各自是否低于阈值切分, 覆盖全部组合
# 注: 参考表的 mode 只有 3 个 (它的 0 同时覆盖"双高"与"双低", 见 reference.py),
# 两者是不同的分层, 不要互相替换
MODE_BOTH_HIGH = 0
MODE_FRONT_LOW = 1
MODE_HIND_LOW = 2
MODE_BOTH_LOW = 3


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


# 按高度命令生成前向速度: 先按较低一侧做腿长缩放, 再按低高度侧数降速
# v = BASE_SPEED × min(h_F,h_H)/BASE_HEIGHT × (2 − 低侧数)/2
# 双高满速; 单低减半; 双低归零 (趴地保持静止)
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
        self.vel_command_w = self.command_tensor[:, :3]         # 速度命令（x, y, z）
        self.height_F_command = self.command_tensor[:, 3]       # 前肢高度命令
        self.height_H_command = self.command_tensor[:, 4]       # 后肢高度命令
        self.angle_command = self.command_tensor[:, 5]          # 角度命令
        self.has_printed_current_cmd = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.use_height_schedule = cfg.use_height_schedule
        self.height_schedule = cfg.height_schedule or [] 
        self.use_position_schedule = cfg.use_position_schedule
        self.position_schedule = cfg.position_schedule or []
        self.angle_values_tensor = torch.tensor(ANGLE_VALUES, device=self.device)
        
        # 记录每个环境的起始位置（用于计算移动距离）
        self.start_positions = torch.zeros(self.num_envs, 3, device=self.device)
        
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(env_ids)
        
        # 记录起始位置
        self._update_start_positions(env_ids)
        
        resampling_time_range = self.cfg.resampling_time_range
        self.time_left[env_ids] = torch.rand(len(env_ids), device=self.device) * (
            resampling_time_range[1] - resampling_time_range[0]
        ) + resampling_time_range[0]

    @property
    def command(self) -> torch.Tensor:
        return self.command_tensor

    def _update_start_positions(self, env_ids: torch.Tensor) -> None:
        self.start_positions[env_ids] = self.robot.data.root_link_pos_w[env_ids]

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
        self.vel_command_w[env_ids, 0] = vel_x
        self.vel_command_w[env_ids, 1] = 0
        self.vel_command_w[env_ids, 2] = 0
        self.height_F_command[env_ids] = height_F
        self.height_H_command[env_ids] = height_H
        self.angle_command[env_ids] = angle
        self.command_tensor[env_ids, :3] = self.vel_command_w[env_ids]
        self.command_tensor[env_ids, 3] = height_F
        self.command_tensor[env_ids, 4] = height_H
        self.command_tensor[env_ids, 5] = angle
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
        self.vel_command_w[env_ids, 0] = vel_x
        self.vel_command_w[env_ids, 1] = 0
        self.vel_command_w[env_ids, 2] = 0
        self.height_F_command[env_ids] = height_F
        self.height_H_command[env_ids] = height_H
        self.angle_command[env_ids] = angle
        self.command_tensor[env_ids, :3] = self.vel_command_w[env_ids]
        self.command_tensor[env_ids, 3] = height_F
        self.command_tensor[env_ids, 4] = height_H
        self.command_tensor[env_ids, 5] = angle
        self.has_printed_current_cmd[env_ids] = False

    def _update_command(self) -> None:
        current_step = self._env.common_step_counter
        stage = get_current_stage(current_step)
        
        # 判断是否使用时间表
        use_schedule = self._should_use_schedule(stage)
        schedule = self._get_schedule(stage) if use_schedule else []
        
        # 使用时间表模式，每个时间步都需要更新命令（因为距离在变化）
        if use_schedule and schedule:
            env_ids = torch.arange(self.num_envs, device=self.device)
            self._update_command_from_schedule(env_ids, schedule)
        else:
            # 定时重采样: 计时器与重采样都由基类 compute() 负责 (基类已做 time_left -= dt
            # 与到点 _resample), 这里不再重复扣减, 否则实际间隔会减半
            pass

    # 回合重置: 重新锚定位移起点 (基类 reset 已重采样命令)
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
        visualizer.add_arrow(
            cmd_start, cmd_end, color=(0.2, 0.2, 0.8, 0.8), width=arrow_width
        )
        
        actual_end = cmd_start + actual_vel * scale
        visualizer.add_arrow(
            cmd_start, actual_end, color=(0.2, 0.8, 0.2, 0.8), width=arrow_width
        )

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
    # 阶段 3 是否兜底使用内置位置表; 回放侧选择 fixed/random 时置 False, 保证 play 优先
    stage_schedule_fallback: bool = True
    
    @dataclass
    class VizCfg:
        z_offset: float = 0.1
        scale: float = 1.0
    
    viz: VizCfg = field(default_factory=VizCfg)
    class_type: type[CommandTerm] = HoleCommand

    # 1.5 把 CommandTermCfg.build 变成抽象方法, 子类必须自己实现
    def build(self, env: ManagerBasedRlEnv) -> CommandTerm:
        return self.class_type(self, env)