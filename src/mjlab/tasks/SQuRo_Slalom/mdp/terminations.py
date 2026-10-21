from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from .config import (
    OBSTACLE_RADIUS,
    CMD_GAIT_FREQ_IDS, 
    PHASE0_EPISODE_SECONDS, 
    PASSAGE_REQUIRED_OBSTACLES, 
)
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



# 确认 Episode 时长：一阶段 20s，二阶段 30s
def check_episode_timeout(env: ManagerBasedRlEnv) -> torch.Tensor:
    phase1 = env.command_manager._terms["slalom_cmd"].phase1_mask  # type: ignore
    phase0_steps = min(env.max_episode_length, round(PHASE0_EPISODE_SECONDS / env.step_dt))
    limit = torch.where(phase1, env.max_episode_length, phase0_steps)
    return env.episode_length_buf >= limit



# Passage state detector for the slalom course.
_FAILURE_NAMES = {
    1: "not_passed",
    2: "wrong_side",
    3: "height_violation",
    4: "fallen",
    5: "timeout",
}


class PassMonitor:
    def __init__(self, env):
        self.env = env
        shape = (env.num_envs, PASSAGE_REQUIRED_OBSTACLES, 3)
        self.seq = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.crossed = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.passed = torch.zeros(shape[:2], dtype=torch.bool, device=env.device)
        self.wrong_side = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.initialized = torch.zeros_like(self.wrong_side)
        self.previous_centers = torch.zeros(env.num_envs, 3, 2, device=env.device)
        self.last_done_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.completed: list[dict] = []


    # Clear passage state only for reset envs; published results stay for the runner.
    def reset(self, env_ids) -> None:
        for buffer in (self.crossed, self.passed, self.wrong_side, self.initialized):
            buffer[env_ids] = False
        self.last_done_step[env_ids] = -1


    # Accumulate a pass once the rear extent clears each obstacle plane on the correct side.
    def update(self, centers: torch.Tensor, rear_extent: torch.Tensor) -> None:
        env = self.env
        command = env.command_manager._terms["slalom_cmd"]
        origins = env.scene.env_origins[:, :2]
        numbers = torch.arange(1, PASSAGE_REQUIRED_OBSTACLES + 1, device=env.device)
        obstacle_x = origins[:, :1] + command.obstacle_spacing[:, None] * numbers
        obstacle_y = origins[:, 1:2] + command.obstacle_y[:, None]
        x, y = centers[..., 0], centers[..., 1]
        previous_x = self.previous_centers[..., 0]
        previous_y = self.previous_centers[..., 1]
        initial = ~self.initialized
        crossed_now = (x[:, None] >= obstacle_x[:, :, None]) & ((previous_x[:, None] < obstacle_x[:, :, None]) | initial[:, None, None])
        fraction = ((obstacle_x[:, :, None] - previous_x[:, None]) / (x - previous_x)[:, None].clamp_min(1e-12)).clamp(0, 1)
        crossing_y = previous_y[:, None] + fraction * (y - previous_y)[:, None]
        crossing_y = torch.where(initial[:, None, None], y[:, None], crossing_y)
        expected_side = torch.where(numbers % 2 == 1, 1.0, -1.0)
        correct = (crossing_y - obstacle_y[:, :, None]) * expected_side[None, :, None] > 0
        active = command.phase1_mask
        self.wrong_side |= (crossed_now & ~correct).any(dim=(1, 2)) & active
        self.crossed |= crossed_now & correct & active[:, None, None]
        cleared = rear_extent[:, None] > obstacle_x + OBSTACLE_RADIUS
        self.passed |= self.crossed.all(dim=-1) & cleared & active[:, None]
        self.previous_centers.copy_(centers)
        self.initialized[:] = True


    # Publish one result per completed episode; timeout alone is not a success.
    def compute(self) -> None:
        env = self.env
        robot = env.scene["robot"]
        body_ids = (_MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id)
        centers = torch.cat((robot.data.root_link_pos_w[:, None, :2], robot.data.body_link_pos_w[:, body_ids, :2]), dim=1)
        vertices = project_body_envelopes(env)
        feet = robot.data.site_pos_w[:, _MODEL_INDICES.foot_site_ids]
        rear_extent = torch.minimum(vertices[..., 0].amin(dim=(1, 2)), feet[..., 0].amin(dim=1))
        self.update(centers, rear_extent)
        timeout = check_episode_timeout(env)
        fallen = check_fallen(env)
        done = timeout | fallen
        command = env.command_manager._terms["slalom_cmd"]
        publish = done & command.phase1_mask & (self.last_done_step != env.common_step_counter)
        for env_id in publish.nonzero(as_tuple=False).flatten().tolist():
            self.last_done_step[env_id] = env.common_step_counter
            passed = int(self.passed[env_id].cumprod(dim=0).sum())
            wrong = bool(self.wrong_side[env_id])
            if fallen[env_id]:
                failure_code = 4
            elif wrong:
                failure_code = 2
            elif passed < PASSAGE_REQUIRED_OBSTACLES:
                failure_code = 5 if timeout[env_id] else 1
            else:
                failure_code = 0
            self.seq[env_id] += 1
            self.completed.append({
                "seq": int(self.seq[env_id]),
                "env_id": env_id,
                "valid": True,  # dedicated evaluation envs are not episode-length filtered
                "success": failure_code == 0,
                "failure_code": failure_code,
                "failure": _FAILURE_NAMES.get(failure_code, ""),
                "passed_obstacles": passed,
                "curvature": int(command.slalom_peak_curvature[env_id]),
                "gait": float(command.command[env_id, CMD_GAIT_FREQ_IDS]),
            })


    # Drain completed results so the runner can consume each one once.
    def consume(self) -> list[dict]:
        results, self.completed = self.completed, []
        return results



# Record slalom passage results as a side effect without adding termination or reward.
def collect_passage_state(env) -> torch.Tensor:
    monitor = getattr(env, "_slalom_passage_monitor", None)
    if monitor is None:
        monitor = env._slalom_passage_monitor = PassMonitor(env)
    if bool(env.command_manager._terms["slalom_cmd"].phase1_mask.any()):
        monitor.compute()
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
