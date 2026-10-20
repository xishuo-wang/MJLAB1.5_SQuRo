from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from .config import CMD_GAIT_FREQ_IDS, PASSAGE_REQUIRED_POLES, PHASE0_EPISODE_SECONDS, POLE_RADIUS
from .indices import _MODEL_INDICES
from .reference import project_body_envelopes

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv



# 终止条件
def check_fallen(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot_entity = env.scene.entities["robot"]
    # 使用重力向量检测姿态
    gravity_vec = robot_entity.data.projected_gravity_b
    uprightness = gravity_vec[:, 2]  
    # 跌倒判定：姿态太倾斜
    fallen = uprightness < 0.2
    return fallen



# 一阶段保持二十秒，二阶段使用配置中的完整回合时长
def check_episode_timeout(env: ManagerBasedRlEnv) -> torch.Tensor:
    phase1 = env.command_manager._terms["slalom_cmd"].phase1_mask  # type: ignore
    phase0_steps = min(env.max_episode_length, round(PHASE0_EPISODE_SECONDS / env.step_dt))
    limit = torch.where(phase1, env.max_episode_length, phase0_steps)
    return env.episode_length_buf >= limit



# 按真实躯干与基座越过杆平面的时刻检查交替侧别，不依赖参考时钟
class SlalomPassageMonitor:
    def __init__(self, env):
        self.env = env
        shape = (env.num_envs, PASSAGE_REQUIRED_POLES, 3)
        self.crossed = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.passed = torch.zeros(shape[:2], dtype=torch.bool, device=env.device)
        self.wrong_side = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.initialized = torch.zeros_like(self.wrong_side)
        self.previous_centers = torch.zeros(env.num_envs, 3, 2, device=env.device)
        self.last_done_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.completed: list[dict] = []


    # 仅清理重置环境的通过状态，已发布的结果继续由 runner 消费
    def reset(self, env_ids) -> None:
        for buffer in (self.crossed, self.passed, self.wrong_side, self.initialized):
            buffer[env_ids] = False
        self.last_done_step[env_ids] = -1


    # 检查实际位置；足端与两段躯干均过杆后才累计通过
    def update(self, centers: torch.Tensor, rear_extent: torch.Tensor) -> None:
        env = self.env
        command = env.command_manager._terms["slalom_cmd"]
        origins = env.scene.env_origins[:, :2]
        numbers = torch.arange(1, PASSAGE_REQUIRED_POLES + 1, device=env.device)
        pole_x = origins[:, :1] + command.pole_spacing[:, None] * numbers
        pole_y = origins[:, 1:2] + command.pole_y[:, None]
        x, y = centers[..., 0], centers[..., 1]
        previous_x = self.previous_centers[..., 0]
        previous_y = self.previous_centers[..., 1]
        initial = ~self.initialized
        crossed_now = (x[:, None] >= pole_x[:, :, None]) & ((previous_x[:, None] < pole_x[:, :, None]) | initial[:, None, None])
        fraction = ((pole_x[:, :, None] - previous_x[:, None]) / (x - previous_x)[:, None].clamp_min(1e-12)).clamp(0, 1)
        crossing_y = previous_y[:, None] + fraction * (y - previous_y)[:, None]
        crossing_y = torch.where(initial[:, None, None], y[:, None], crossing_y)
        expected_side = torch.where(numbers % 2 == 1, 1.0, -1.0)
        correct = (crossing_y - pole_y[:, :, None]) * expected_side[None, :, None] > 0
        active = command.phase1_mask
        self.wrong_side |= (crossed_now & ~correct).any(dim=(1, 2)) & active
        self.crossed |= crossed_now & correct & active[:, None, None]
        cleared = rear_extent[:, None] > pole_x + POLE_RADIUS
        self.passed |= self.crossed.all(dim=-1) & cleared & active[:, None]
        self.previous_centers.copy_(centers)
        self.initialized[:] = True


    # 在真实回合结束时发布成绩；超时本身不能作为成功
    def compute(self) -> None:
        env = self.env
        robot = env.scene["robot"]
        body_ids = (_MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id)
        centers = torch.cat((robot.data.root_link_pos_w[:, None, :2], robot.data.body_link_pos_w[:, body_ids, :2]), dim=1)
        vertices = project_body_envelopes(env)
        feet = robot.data.site_pos_w[:, _MODEL_INDICES.foot_site_ids]
        rear_extent = torch.minimum(vertices[..., 0].amin(dim=(1, 2)), feet[..., 0].amin(dim=1))
        self.update(centers, rear_extent)
        fallen = check_fallen(env)
        done = fallen | check_episode_timeout(env)
        command = env.command_manager._terms["slalom_cmd"]
        publish = done & command.phase1_mask & (self.last_done_step != env.common_step_counter)
        for env_id in publish.nonzero(as_tuple=False).flatten().tolist():
            self.last_done_step[env_id] = env.common_step_counter
            passed = int(self.passed[env_id].cumprod(dim=0).sum())
            wrong = bool(self.wrong_side[env_id])
            failure = "跌倒" if fallen[env_id] else "错侧" if wrong else "未过三杆" if passed < PASSAGE_REQUIRED_POLES else ""
            self.completed.append({
                "success": failure == "", "failure": failure, "passed_poles": passed,
                "curvature": int(command.slalom_peak_curvature[env_id]),
                "gait": float(command.command[env_id, CMD_GAIT_FREQ_IDS]),
                "env_id": env_id,
            })


    # 取走已结束回合，避免重复计数或无限积累
    def consume(self) -> list[dict]:
        results, self.completed = self.completed, []
        return results



# 记录绕杆成绩，不增加额外终止条件或奖励
def track_slalom_passage(env) -> torch.Tensor:
    monitor = getattr(env, "_slalom_passage_monitor", None)
    if monitor is None:
        monitor = env._slalom_passage_monitor = SlalomPassageMonitor(env)
    if bool(env.command_manager._terms["slalom_cmd"].phase1_mask.any()):
        monitor.compute()
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
