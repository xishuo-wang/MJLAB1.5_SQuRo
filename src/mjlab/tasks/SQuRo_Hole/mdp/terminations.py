from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .config import STEPS_PER_ITER, get_current_stage, stage_uses_height_course
from .entity import hole_collision_enabled, read_bottom_z
from .indices import _MODEL_INDICES, resolve_model_indices

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv



# 终止条件
def check_fallen(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 获取机器人实体
    robot_entity = env.scene.entities["robot"]
    
    # 使用重力向量检测姿态
    gravity_vec = robot_entity.data.projected_gravity_b
    uprightness = gravity_vec[:, 2]  
    
    # 跌倒判定
    fallen = uprightness < 0.2
    return fallen


# 持久化单回合穿板状态，并为 runner 提供无覆盖的完成快照。
class HolePassageMonitor:
    def __init__(self, env: ManagerBasedRlEnv):
        self.env = env
        n = env.num_envs
        self.seq = torch.zeros(n, dtype=torch.long, device=env.device)
        self.approached = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.entered = torch.zeros_like(self.approached)
        self.front_passed = torch.zeros_like(self.approached)
        self.hind_passed = torch.zeros_like(self.approached)
        self.fully_passed = torch.zeros_like(self.approached)
        self.front_inside = torch.zeros_like(self.approached)
        self.hind_inside = torch.zeros_like(self.approached)
        self.height_violation = torch.zeros_like(self.approached)
        self.side_violation = torch.zeros_like(self.approached)
        self.initialized = torch.zeros_like(self.approached)
        self.last_done_step = torch.full((n,), -1, dtype=torch.long, device=env.device)
        self.completed: list[dict] = []

    # 清空指定环境的回合跟踪状态。
    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        for name in (
            "approached", "entered", "front_passed", "hind_passed", "fully_passed",
            "front_inside", "hind_inside", "height_violation", "side_violation", "initialized",
        ):
            getattr(self, name)[env_ids] = False
        self.last_done_step[env_ids] = -1

    # 更新穿板判据；此 term 永远返回 False，不影响原终止语义。
    def compute(self) -> torch.Tensor:
        env = self.env
        command = env.command_manager._terms.get("hole_cmd")
        valid_course = (
            stage_uses_height_course(get_current_stage(env.common_step_counter))
            and hole_collision_enabled(env)
            and command is not None
            and command.enable_height_course  # type: ignore
        )
        if not valid_course:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        robot = env.scene["robot"]
        resolve_model_indices(robot)
        front = robot.data.site_pos_w[:, _MODEL_INDICES.f_body_site_ids]
        hind = robot.data.site_pos_w[:, _MODEL_INDICES.h_body_site_ids]
        board = env.scene.entities.get("hole1")
        if board is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        x1 = float(board.cfg.position[0]) - float(board.cfg.size[0])  # type: ignore
        x2 = float(board.cfg.position[0]) + float(board.cfg.size[0])  # type: ignore
        y0 = float(board.cfg.position[1])  # type: ignore
        half_width = float(board.cfg.size[1])  # type: ignore
        bottom = read_bottom_z(board)
        front_min = front[..., 0].min(dim=1).values
        front_max = front[..., 0].max(dim=1).values
        hind_min = hind[..., 0].min(dim=1).values
        front_in_x = (front[..., 0] >= x1) & (front[..., 0] <= x2)
        hind_in_x = (hind[..., 0] >= x1) & (hind[..., 0] <= x2)
        front_in_y = (front[..., 1] - y0).abs() <= half_width
        hind_in_y = (hind[..., 1] - y0).abs() <= half_width
        front_crossing = front_in_x.any(dim=1)
        hind_crossing = hind_in_x.any(dim=1)
        front_lane_ok = front_in_y.all(dim=1)
        hind_lane_ok = hind_in_y.all(dim=1)
        front_inside_now = front_crossing & front_lane_ok
        hind_inside_now = hind_crossing & hind_lane_ok

        initial = ~self.initialized
        self.approached |= initial & (front_max < x1)
        self.entered |= self.approached & (front_max >= x1)
        self.front_passed |= front_min > x2
        self.hind_passed |= hind_min > x2
        self.front_inside |= front_inside_now
        self.hind_inside |= hind_inside_now
        self.side_violation |= (front_crossing & ~front_lane_ok) | (hind_crossing & ~hind_lane_ok)
        front_over = (front_in_x & front_in_y & (front[..., 2] > bottom[:, None] + 1e-3)).any(dim=1)
        hind_over = (hind_in_x & hind_in_y & (hind[..., 2] > bottom[:, None] + 1e-3)).any(dim=1)
        self.height_violation |= front_over | hind_over
        self.fully_passed |= (
            self.entered
            & (front_min > x2)
            & (hind_min > x2)
            & self.front_inside
            & self.hind_inside
            & ~self.side_violation
            & ~self.height_violation
        )
        self.initialized[:] = True

        timeout = env.episode_length_buf >= env.max_episode_length
        fallen = check_fallen(env)
        done = timeout | fallen
        step = int(env.common_step_counter)
        publish = done & (self.last_done_step != step)
        for env_id in publish.nonzero(as_tuple=False).flatten().tolist():
            self.last_done_step[env_id] = step
            success = bool(
                valid_course
                and self.fully_passed[env_id]
                and not self.side_violation[env_id]
                and not self.height_violation[env_id]
                and not fallen[env_id]
            )
            if fallen[env_id]:
                failure_code = 4
            elif self.side_violation[env_id]:
                failure_code = 2
            elif self.height_violation[env_id]:
                failure_code = 3
            elif not success:
                failure_code = 5 if timeout[env_id] else 1
            else:
                failure_code = 0
            self.seq[env_id] += 1
            valid = bool(
                valid_course
                and command.obstacle_height_sampled[env_id]  # type: ignore
            )
            course_result = {"advanced": False}
            if command is not None and command.enable_course_progress:  # type: ignore
                course_result = command.height_curriculum_state.record_episode(  # type: ignore
                    float(bottom[env_id]),
                    success,
                    int(env.common_step_counter // STEPS_PER_ITER),
                    get_current_stage(env.common_step_counter),
                    hole_collision_enabled(env),
                    eligible=valid,
                    failure_code=failure_code,
                )
                if course_result["advanced"]:
                    print(
                        f"高度课程降档：Zmin {course_result['previous_z_min']:.3f}→"
                        f"{course_result['z_min']:.3f} m，当前档通过率 "
                        f"{course_result['success_rate']:.1%}，样本数 {course_result['sample_count']}"
                    )
            self.completed.append({
                "seq": int(self.seq[env_id]),
                "env_id": env_id,
                "height": float(bottom[env_id]),
                "valid": valid,
                "success": success,
                "failure_code": failure_code,
                "advanced": course_result["advanced"],
            })
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 读取本步新完成的回合快照。
    def consume(self) -> list[dict]:
        completed = self.completed
        self.completed = []
        return completed


# 执行永不触发的穿板监测项。
def monitor_hole_passage(env: ManagerBasedRlEnv) -> torch.Tensor:
    monitor = getattr(env, "_hole_passage_monitor", None)
    if monitor is None:
        with torch.inference_mode(False):
            monitor = HolePassageMonitor(env)
        env._hole_passage_monitor = monitor  # type: ignore
    return monitor.compute()
