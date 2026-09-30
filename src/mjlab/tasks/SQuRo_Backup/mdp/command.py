from __future__ import annotations
import torch
from math import cos, isfinite, radians
from typing import TYPE_CHECKING, Tuple
from dataclasses import dataclass, field
from mjlab.managers import CommandTermCfg
from mjlab.managers.command_manager import CommandTerm
from .events import apply_fallen_state
from .indices import _MODEL_INDICES, resolve_model_indices
from .config import (
    T3,
    P1_END,
    STAND_GROUND_HEIGHT,
    STAND_TARGET_HEIGHT,
    STAND_VEL_MEAN_MAX,
    STAND_CONFIRM_DURATION,
)
from .curriculums import (
    P1_SETTLE_MAX,
    get_p1_settle_margin,
    P1_CLOSE_FOLLOWS_SETTLE,
    get_curriculum_time_scale,
)

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv



_GROUND_TH_S1 = 0.03            # 判定 S1 平躺高度阈值
_GROUND_TH_S2 = 0.04            # 判定 S2 趴地高度阈值
_P1_NOMINAL = P1_END            # 参考到达 S1 的名义时长
_P2_NOMINAL = T3                # 参考达到 S2 的名义时长
WINDOW_LATE_S = 0.50            # 时间段后的保留余量
STAND_UPRIGHT_COS = 0.9         # 判定正置的背腹轴余弦下限
STAND_MIN_HEIGHT = 0.05         # 判定站起的最低躯干高度
STAND_UPRIGHT_COS_STAY = 0.8    # 维持站立窗口的宽松正置阈值
STAND_MIN_HEIGHT_STAY = 0.045   # 维持站立窗口的宽松高度阈值



# 对含 NaN 的记录取有效值均值; 没有有效值时返回 0
def _mean_valid(values: torch.Tensor) -> float:
    valid = torch.isfinite(values)
    count = valid.sum().clamp_min(1).to(values.dtype)
    return (torch.where(valid, values, torch.zeros_like(values)).sum() / count).item()



class BackupCommand(CommandTerm):
    cfg: "BackupCommandCfg"
    def __init__(self, cfg: "BackupCommandCfg", env: "ManagerBasedRlEnv"):
        for name in ("window_late_s",):
            value = getattr(cfg, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须为有限的非负实际秒数")
        if not isfinite(float(cfg.window_late_s)) or float(cfg.window_late_s) < 0.0:
            raise ValueError("window_late_s 必须为有限的非负实际秒数")
        angle = float(cfg.pose_angle_tolerance_deg)
        if not isfinite(angle) or not 0.0 < angle < 90.0:
            raise ValueError("pose_angle_tolerance_deg 必须为有限的 0 到 90 度之间的角度")
        for name in ("pose_confirm_s", "inverted_confirm_s"):
            value = float(getattr(cfg, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} 必须为有限的正实际秒数")
        super().__init__(cfg, env)
        self.fixed_time_scale = cfg.fixed_time_scale
        self._pose_cos_threshold = cos(radians(angle))
        self.command_tensor = torch.zeros(self.num_envs, 9, device=self.device)
        self.vel_command = self.command_tensor[:, 0]
        self.height_f_command = self.command_tensor[:, 1]
        self.height_h_command = self.command_tensor[:, 2]
        self.gait_freq_command = self.command_tensor[:, 3]
        self.curvature_command = self.command_tensor[:, 4]
        self.time_scale_command = self.command_tensor[:, 5]
        self.phase_command = self.command_tensor[:, 6]
        self.wall_x_neg_command = self.command_tensor[:, 7]
        self.wall_x_pos_command = self.command_tensor[:, 8]
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.t_phase = torch.zeros(self.num_envs, device=self.device)
        self.retry = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_s1_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_s2_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_advance1 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_advance2 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_retry_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_s1_milestone = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_s2_milestone = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_back_to_p1 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_back_to_p2 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_both_inverted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_s1_confirmed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_s2_confirmed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_inverted_confirmed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._s1_confirm_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._s2_confirm_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._inverted_confirm_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._s1_onset = torch.full((self.num_envs,), float("nan"), device=self.device)
        self._s2_onset = torch.full((self.num_envs,), float("nan"), device=self.device)
        self._s1_criterion_first = torch.full((self.num_envs,), float("nan"), device=self.device)
        self._s2_criterion_first = torch.full((self.num_envs,), float("nan"), device=self.device)
        self._s1_cycle_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._s2_cycle_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cycles_this_episode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._pending_episode_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_s1_gated_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_s2_gated_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stand_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._stand_vel_integral = torch.zeros(self.num_envs, device=self.device)
        self._pending_cycle_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_cycle_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_cycle_end_pulse = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._s1_awarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._s2_awarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ep_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ep_had_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ep_cycle_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ep_seq = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_ep_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_ep_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ep_stood_pose = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ep_stood_onset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_ep_stood_pose = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_ep_stood_onset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._wall_d = torch.zeros(self.num_envs, device=self.device)
        self._last_ep_wall_d = torch.zeros(self.num_envs, device=self.device)
        self._prev_stand_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pose_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._pose_cos_cache: torch.Tensor | None = None
        self._pose_stage = torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device)
        self._early_s2_entered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._early_s2_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._early_s2_count = torch.zeros(self.num_envs, device=self.device)
        self._s1_hold_lost = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._s1_hold_lost_count = torch.zeros(self.num_envs, device=self.device)
        self._update_dt = 0.0
        self._asset = self._env.scene.entities[cfg.asset_name]
        resolve_model_indices(self._asset)

        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(env_ids)
        t_range = self.cfg.resampling_time_range
        self.time_left[env_ids] = torch.rand(len(env_ids), device=self.device) * (t_range[1] - t_range[0]) + t_range[0]


    @property
    def command(self) -> torch.Tensor:
        return self.command_tensor


    @property
    def stage_t(self) -> torch.Tensor:
        return self.t_phase


    @property
    def s1_transition_pulse(self) -> torch.Tensor:
        return self._last_advance1


    @property
    def s2_transition_pulse(self) -> torch.Tensor:
        return self._last_advance2


    @property
    def s1_milestone_pulse(self) -> torch.Tensor:
        return self._last_s1_milestone


    @property
    def s2_milestone_pulse(self) -> torch.Tensor:
        return self._last_s2_milestone


    @property
    def s1_dev_early(self) -> torch.Tensor:
        return self._s1_criterion_first - _P1_NOMINAL * self.time_scale_command


    @property
    def s2_dev_early(self) -> torch.Tensor:
        return self._s2_criterion_first - _P2_NOMINAL * self.time_scale_command


    @property
    def s1_dev_late(self) -> torch.Tensor:
        return self._s1_criterion_first - _P1_NOMINAL * self.time_scale_command

    @property
    def s2_dev_late(self) -> torch.Tensor:
        return self._s2_criterion_first - _P2_NOMINAL * self.time_scale_command


    # 建立 _resample_command / _clear_cycle_state 会写的全部缓冲
    def _ensure_buffers(self) -> None:
        if not hasattr(self, "_s1_onset"):
            self._s1_onset = torch.full_like(self.t_phase, float("nan"))
            self._s2_onset = torch.full_like(self.t_phase, float("nan"))
            self._s1_criterion_first = torch.full_like(self.t_phase, float("nan"))
            self._s2_criterion_first = torch.full_like(self.t_phase, float("nan"))
        if not hasattr(self, "_s1_cycle_latched"):
            self._s1_cycle_latched = torch.zeros_like(self.phase, dtype=torch.bool)
            self._s2_cycle_latched = torch.zeros_like(self.phase, dtype=torch.bool)
            self._cycles_this_episode = torch.zeros_like(self.phase)
            self._pending_episode_reset = torch.zeros_like(self.phase, dtype=torch.bool)
        if not hasattr(self, "_stand_elapsed"):
            self._stand_elapsed = torch.zeros_like(self.t_phase)
            self._stand_vel_integral = torch.zeros_like(self.t_phase)
        if not hasattr(self, "_pending_cycle_reset"):
            self._pending_cycle_reset = torch.zeros_like(self.phase, dtype=torch.bool)
            self._last_cycle_reset = torch.zeros_like(self.phase, dtype=torch.bool)
            self._last_cycle_end_pulse = torch.zeros_like(self.phase, dtype=torch.bool)
        if not hasattr(self, "_last_s1_ok"):
            self._last_s1_ok = torch.zeros_like(self.phase, dtype=torch.bool)
            self._last_s2_ok = torch.zeros_like(self.phase, dtype=torch.bool)
        if not hasattr(self, "_last_s1_confirmed"):
            self._last_s1_confirmed = torch.zeros_like(self.phase, dtype=torch.bool)
            self._last_s2_confirmed = torch.zeros_like(self.phase, dtype=torch.bool)
        if not hasattr(self, "_last_s1_gated_ok"):
            self._last_s1_gated_ok = torch.zeros_like(self.phase, dtype=torch.bool)
            self._last_s2_gated_ok = torch.zeros_like(self.phase, dtype=torch.bool)
        if not hasattr(self, "_pose_cache"):
            self._pose_cache = None
            self._pose_cos_cache = None
        if not hasattr(self, "_ep_index"):
            self._ep_index = torch.zeros_like(self.phase)
            self._ep_had_success = torch.zeros_like(self.phase, dtype=torch.bool)
            self._ep_cycle_count = torch.zeros_like(self.phase)
            self._ep_seq = torch.zeros_like(self.phase)
            self._last_ep_valid = torch.zeros_like(self.phase, dtype=torch.bool)
            self._last_ep_success = torch.zeros_like(self.phase, dtype=torch.bool)
        if not hasattr(self, "_ep_stood_pose"):
            self._ep_stood_pose = torch.zeros_like(self.phase, dtype=torch.bool)
            self._ep_stood_onset = torch.zeros_like(self.phase, dtype=torch.bool)
            self._last_ep_stood_pose = torch.zeros_like(self.phase, dtype=torch.bool)
            self._last_ep_stood_onset = torch.zeros_like(self.phase, dtype=torch.bool)
            self._prev_stand_active = torch.zeros_like(self.phase, dtype=torch.bool)
        if not hasattr(self, "_wall_d"):
            self._wall_d = torch.zeros_like(self.t_phase)
            self._last_ep_wall_d = torch.zeros_like(self.t_phase)


    # 获取受限空间实体
    def _wall_entity(self):
        scene = getattr(self._env, "scene", None)
        if scene is None:
            return None
        return getattr(scene, "entities", {}).get("restricted_space")


    # 右墙的固定中心: 显式配置优先, 否则沿用场景实体的编译值
    def _wall_x_pos_fixed(self) -> float:
        if self.cfg.wall_x_pos is not None:
            return float(self.cfg.wall_x_pos)
        entity = self._wall_entity()
        return float(entity.cfg.wall_x_pos) if entity is not None else 0.0


    # 左墙距离 d 的采样范围
    def _wall_bounds(self) -> tuple[float, float]:
        entity = self._wall_entity()
        compiled = -float(entity.cfg.wall_x_neg) if entity is not None else 0.20
        lo = compiled if self.cfg.wall_d_min is None else float(self.cfg.wall_d_min)
        hi = compiled if self.cfg.wall_d_max is None else float(self.cfg.wall_d_max)
        return lo, max(lo, hi)


    # 把本回合的逐环境左墙距离推到物理 (mocap) 与观测。
    def _push_wall(self, env_ids: torch.Tensor) -> None:
        entity = self._wall_entity()
        if entity is None:
            self.wall_x_neg_command[env_ids] = 0.0
            self.wall_x_pos_command[env_ids] = 0.0
            return
        x_pos = self._wall_x_pos_fixed()
        d = self._wall_d[env_ids]
        origins = getattr(self._env.scene, "env_origins", None)
        origin = origins[env_ids, 0] if origins is not None else torch.zeros_like(d)
        entity.write_wall_x(self._env, -d + origin, torch.full_like(d, x_pos) + origin, env_ids=env_ids)
        self.wall_x_neg_command[env_ids] = -d
        self.wall_x_pos_command[env_ids] = x_pos


    # 完整回合复位时逐环境采样本回合的墙位
    def _sample_wall(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        lo, hi = self._wall_bounds()
        if hi > lo:
            d = torch.empty(n, device=self.device).uniform_(lo, hi)
            frac = float(self.cfg.wall_d_min_frac)
            if frac >= 1.0:
                d = torch.full_like(d, lo)
            elif frac > 0.0:
                pick = torch.rand(n, device=self.device) < frac
                d = torch.where(pick, torch.full_like(d, lo), d)
        else:
            d = torch.full((n,), lo, device=self.device)
        self._wall_d[env_ids] = d
        self._push_wall(env_ids)


    # 更新墙间距下界并让后续回合按新范围采样。
    def set_wall_curriculum(self, d_min: float | None, d_max: float | None = None, frac: float | None = None) -> None:
        self.cfg.wall_d_min = None if d_min is None else float(d_min)
        self.cfg.wall_d_max = None if d_max is None else float(d_max)
        if frac is not None:
            self.cfg.wall_d_min_frac = float(frac)


    # 按课程采样 time_scale λ (episode 内固定)
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._ensure_buffers()
        n = len(env_ids)
        if n == 0:
            return
        self.vel_command[env_ids] = 0.0                # 跌倒爬起无速度跟踪 (占位)
        self.height_f_command[env_ids] = 0.055         # 期望前体高度 (命令占位)
        self.height_h_command[env_ids] = 0.055         # 期望后体高度
        self.gait_freq_command[env_ids] = 1.0          # 占位
        self.curvature_command[env_ids] = 0.0          # 无转向
        if self.fixed_time_scale is not None:
            lam = torch.full((n,), float(self.fixed_time_scale), device=self.device)
        else:
            lam = get_curriculum_time_scale(self._env.common_step_counter, n, self.device)
        self.time_scale_command[env_ids] = lam         # 参考时间缩放
        self._sample_wall(env_ids)                     # 每回合重采墙位 (物理 + 观测 + 统计同源)
        self.phase[env_ids] = 0
        self.t_phase[env_ids] = 0.0
        self.retry[env_ids] = 0
        self.phase_command[env_ids] = 0.0
        self._last_s1_ok[env_ids] = False
        self._last_s2_ok[env_ids] = False
        self._last_advance1[env_ids] = False
        self._last_advance2[env_ids] = False
        self._last_retry_mask[env_ids] = False
        self._last_s1_milestone[env_ids] = False
        self._last_s2_milestone[env_ids] = False
        self._last_back_to_p1[env_ids] = False
        self._last_back_to_p2[env_ids] = False
        self._last_both_inverted[env_ids] = False
        self._last_s1_confirmed[env_ids] = False
        self._last_s2_confirmed[env_ids] = False
        self._last_inverted_confirmed[env_ids] = False
        self._s1_confirm_elapsed[env_ids] = 0.0
        self._s2_confirm_elapsed[env_ids] = 0.0
        self._inverted_confirm_elapsed[env_ids] = 0.0
        self._s1_onset[env_ids] = float("nan")
        self._s2_onset[env_ids] = float("nan")
        self._s1_criterion_first[env_ids] = float("nan")
        self._s2_criterion_first[env_ids] = float("nan")
        self._last_s1_gated_ok[env_ids] = False
        self._last_s2_gated_ok[env_ids] = False
        self._s1_awarded[env_ids] = False
        self._s2_awarded[env_ids] = False
        self._early_s2_count[env_ids] = 0.0
        self._s1_hold_lost_count[env_ids] = 0.0
        self._clear_cycle_state(env_ids)


    # 清楚新循环内部状态
    def _clear_cycle_state(self, env_ids: torch.Tensor) -> None:
        self._stand_elapsed[env_ids] = 0.0
        self._stand_vel_integral[env_ids] = 0.0
        self._pending_cycle_reset[env_ids] = False
        self._last_cycle_end_pulse[env_ids] = False
        self._s1_cycle_latched[env_ids] = False
        self._s2_cycle_latched[env_ids] = False
        self._early_s2_entered[env_ids] = False
        self._early_s2_elapsed[env_ids] = 0.0
        self._s1_hold_lost[env_ids] = False


    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        self._ensure_buffers()
        if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
            max_len = int(self._env.max_episode_length)
            ended = self._env.episode_length_buf[env_ids] >= max_len
            ids = env_ids[ended]
            if len(ids) > 0:
                valid = self._ep_index[ids] >= 1
                self._last_ep_valid[ids] = valid
                self._last_ep_success[ids] = self._ep_had_success[ids] & valid
                self._last_ep_stood_pose[ids] = self._ep_stood_pose[ids] & valid
                self._last_ep_stood_onset[ids] = self._ep_stood_onset[ids] & valid
                self._last_ep_wall_d[ids] = self._wall_d[ids]     # 本回合墙位 (重采之前)
                self._ep_seq[ids] += 1
                self._ep_index[ids] += 1
            self._ep_had_success[env_ids] = False
            self._ep_cycle_count[env_ids] = 0
            self._ep_stood_pose[env_ids] = False
            self._ep_stood_onset[env_ids] = False
        return super().reset(env_ids)


    def compute(self, dt: float) -> None:
        self._update_dt = dt
        super().compute(dt)


    # 身体段方向判定
    def _segment_u(self, idx: int) -> torch.Tensor:
        pairs = _MODEL_INDICES.segment_belly_back_ids
        assert pairs is not None, "segment_belly_back_ids 未解析, 请先调用 resolve_model_indices"
        belly_id, back_id = pairs[idx]
        sp = self._asset.data.site_pos_w
        delta = sp[:, back_id, :] - sp[:, belly_id, :]
        norm = torch.linalg.vector_norm(delta, dim=-1)
        finite = torch.isfinite(delta).all(dim=-1) & torch.isfinite(norm)
        valid = finite & (norm > torch.finfo(sp.dtype).eps)
        u = delta[:, 2] / norm.clamp_min(torch.finfo(sp.dtype).tiny)
        return torch.where(valid, u, torch.full_like(u, float("nan")))


    def _pose_cos(self) -> torch.Tensor:
        return torch.stack((self._segment_u(0), self._segment_u(1)), dim=1)


    def _flags_from_cos(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        finite = torch.isfinite(u)
        threshold = self._pose_cos_threshold
        return finite & (u >= threshold), finite & (u <= -threshold)


    def _pose_flags(self) -> tuple[torch.Tensor, torch.Tensor]:
        u = self._pose_cos()
        self._pose_cos_cache = u
        return self._flags_from_cos(u)


    def _get_pose_flags(self) -> tuple[torch.Tensor, torch.Tensor]:
        cached = getattr(self, "_pose_cache", None)
        if cached is None:
            return self._pose_flags()
        return cached


    def _get_pose_cos(self) -> torch.Tensor:
        cached = getattr(self, "_pose_cos_cache", None)
        if cached is None:
            return self._pose_cos()
        return cached


    @staticmethod
    def _ramp(u: torch.Tensor, sign: float) -> torch.Tensor:
        return (sign * u).clamp(0.0, 1.0)


    @property
    def progress_s1(self) -> torch.Tensor:
        u = torch.nan_to_num(self._get_pose_cos(), nan=0.0)
        return self._ramp(u[:, 1], 1.0)


    @property
    def progress_s2(self) -> torch.Tensor:
        u = torch.nan_to_num(self._get_pose_cos(), nan=0.0)
        front = (u[:, 0].clamp(-1.0, 1.0) + 1.0) / 2.0
        return self._ramp(u[:, 1], 1.0) * front


    def standing_metrics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self._pose_cos()
        heights = torch.stack((self._body_height(_MODEL_INDICES.f_body_id), self._body_height(_MODEL_INDICES.h_body_id)), dim=1)
        u_floor = u.amin(dim=1)
        h_floor = heights.amin(dim=1)
        return u_floor, h_floor, self._joint_vel_rms()


    def standing_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        u = self._pose_cos()
        heights = torch.stack((self._body_height(_MODEL_INDICES.f_body_id), self._body_height(_MODEL_INDICES.h_body_id)), dim=1)
        valid = torch.isfinite(u).all(dim=1) & torch.isfinite(heights).all(dim=1)
        u_floor = u.amin(dim=1)
        h_floor = heights.amin(dim=1)
        standing = valid & (u_floor > STAND_UPRIGHT_COS) & (h_floor > STAND_MIN_HEIGHT)
        cone = float(self._pose_cos_threshold)
        orient = ((u_floor - cone) / (1.0 - cone)).clamp(0.0, 1.0)
        height_progress = ((h_floor - STAND_GROUND_HEIGHT) / (STAND_TARGET_HEIGHT - STAND_GROUND_HEIGHT)).clamp(0.0, 1.0)
        progress = orient * height_progress
        return standing, torch.where(valid, progress, torch.zeros_like(progress))


    def stand_gate(self) -> tuple[torch.Tensor, torch.Tensor]:
        u_floor, h_floor, vel_rms = self.standing_metrics()
        finite = torch.isfinite(u_floor) & torch.isfinite(h_floor) & torch.isfinite(vel_rms)
        hold = finite & (u_floor > STAND_UPRIGHT_COS_STAY) & (h_floor > STAND_MIN_HEIGHT_STAY)
        strict = finite & (u_floor > STAND_UPRIGHT_COS) & (h_floor > STAND_MIN_HEIGHT)
        return hold, strict


    def _joint_vel_rms(self) -> torch.Tensor:
        vel = self._asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
        return vel.square().mean(dim=1).sqrt()


    def _segment_upright(self, idx: int) -> torch.Tensor:
        upright, _ = self._get_pose_flags()
        return upright[:, idx]


    def _segment_inverted(self, idx: int) -> torch.Tensor:
        _, inverted = self._get_pose_flags()
        return inverted[:, idx]


    def _body_height(self, body_id: int) -> torch.Tensor:
        return self._asset.data.body_link_pos_w[:, body_id, 2]


    # 检测是否满足 S1 姿态
    def _check_S1(self) -> torch.Tensor:
        f_inv = self._segment_inverted(0)
        h_up = self._segment_upright(1)
        fz = self._body_height(_MODEL_INDICES.f_body_id)
        hz = self._body_height(_MODEL_INDICES.h_body_id)
        return f_inv & h_up & (fz < _GROUND_TH_S1) & (hz < _GROUND_TH_S1)


    # 检测是否满足 S2 姿态
    def _check_S2(self) -> torch.Tensor:
        f_up = self._segment_upright(0)
        h_up = self._segment_upright(1)
        fz = self._body_height(_MODEL_INDICES.f_body_id)
        hz = self._body_height(_MODEL_INDICES.h_body_id)
        grounded = f_up & h_up & torch.isfinite(fz) & torch.isfinite(hz) & (fz < _GROUND_TH_S2) & (hz < _GROUND_TH_S2)
        standing, _ = self.standing_state()
        return grounded | standing


    # 检测是否处于双倒状态
    def _check_both_inverted(self) -> torch.Tensor:
        _, inverted = self._get_pose_flags()
        return inverted[:, 0] & inverted[:, 1]

    
    @staticmethod
    def _update_confirmation(
        elapsed: torch.Tensor,
        candidate: torch.Tensor,
        running: torch.Tensor,
        dt: float,
        duration: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = torch.zeros_like(elapsed)
        elapsed = torch.where(candidate & running, elapsed, zero)
        if dt > 0.0 and isfinite(dt):
            elapsed = elapsed + (candidate & running).to(elapsed.dtype) * dt
        eps = torch.finfo(elapsed.dtype).eps * max(1.0, abs(duration), abs(dt)) * 8.0
        confirmed = elapsed >= (duration - eps)
        return elapsed, confirmed


    @staticmethod
    def _update_stand_window(
        elapsed: torch.Tensor,
        vel_integral: torch.Tensor,
        active: torch.Tensor,
        running: torch.Tensor,
        vel_rms: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        run = running & active
        step = run.to(elapsed.dtype) * (dt if (dt > 0.0 and isfinite(dt)) else 0.0)
        zero = torch.zeros_like(elapsed)
        speed = torch.nan_to_num(vel_rms, nan=0.0, posinf=0.0, neginf=0.0)
        nxt_t = torch.where(run, elapsed + step, zero)
        nxt_v = torch.where(run, vel_integral + step * speed, zero)
        return nxt_t, nxt_v


    # 更新命令
    def _update_command(self) -> None:
        dt = float(self._update_dt)
        if not hasattr(self, "_s1_confirm_elapsed"):
            self._s1_confirm_elapsed = torch.zeros_like(self.t_phase)
            self._s2_confirm_elapsed = torch.zeros_like(self.t_phase)
            self._inverted_confirm_elapsed = torch.zeros_like(self.t_phase)
        if not hasattr(self, "_s1_awarded"):
            self._s1_awarded = torch.zeros_like(self.phase, dtype=torch.bool)
            self._s2_awarded = torch.zeros_like(self.phase, dtype=torch.bool)
        if not hasattr(self, "_last_s1_confirmed"):
            self._last_s1_confirmed = torch.zeros_like(self.phase, dtype=torch.bool)
            self._last_s2_confirmed = torch.zeros_like(self.phase, dtype=torch.bool)
        self._ensure_buffers()
        lam = self.time_scale_command.clamp(min=0.1)
        running = self._env.episode_length_buf > 0
        real_dt = dt if dt > 0.0 and isfinite(dt) else 0.0
        self.t_phase = self.t_phase + running.to(self.t_phase.dtype) * real_dt
        pose_flags = getattr(self, "_pose_flags", None)
        self._pose_cache = pose_flags() if pose_flags is not None else None
        p1 = self.phase == 0
        settle1 = get_p1_settle_margin(int(getattr(self._env, "common_step_counter", 0)))
        expected1 = (P1_END + settle1) * lam
        s1_ok = self._check_S1()
        p2 = self.phase == 1
        expected2 = _P2_NOMINAL * lam
        s2_ok = self._check_S2()
        both_inverted = getattr(self, "_check_both_inverted", lambda: torch.zeros_like(s1_ok))()
        self._pose_cache = None
        self._pose_cos_cache = None

        self._s1_confirm_elapsed, s1_confirmed = BackupCommand._update_confirmation(self._s1_confirm_elapsed, s1_ok, running, real_dt, float(self.cfg.pose_confirm_s))
        self._s2_confirm_elapsed, s2_confirmed = BackupCommand._update_confirmation(self._s2_confirm_elapsed, s2_ok, running, real_dt, float(self.cfg.pose_confirm_s))
        self._inverted_confirm_elapsed, inverted_confirmed = BackupCommand._update_confirmation(
            self._inverted_confirm_elapsed,
            both_inverted,
            running,
            real_dt,
            float(self.cfg.inverted_confirm_s),
        )

        # 阶段推进: P1 -> P2 -> P3
        close_settle = settle1 if P1_CLOSE_FOLLOWS_SETTLE else P1_SETTLE_MAX
        close1 = (P1_END + close_settle) * lam + float(self.cfg.window_late_s)
        close2 = expected2 + float(self.cfg.window_late_s)
        gated1 = p1 & (self.t_phase <= close1)
        gated2 = p2 & (self.t_phase <= close2)
        self._s1_confirm_elapsed = torch.where(p1 & ~gated1, torch.zeros_like(self._s1_confirm_elapsed), self._s1_confirm_elapsed)
        self._s2_confirm_elapsed = torch.where(p2 & ~gated2, torch.zeros_like(self._s2_confirm_elapsed), self._s2_confirm_elapsed)
        s1_confirmed = s1_confirmed & gated1
        s2_confirmed = s2_confirmed & gated2
        advance1 = p1 & s1_confirmed & (self.t_phase >= expected1)
        advance2 = p2 & s2_confirmed & (self.t_phase >= expected2)
        retry1 = p1 & (self.t_phase > close1) & ~advance1
        retry2 = p2 & (self.t_phase > close2) & ~advance2
        gated_ok1 = gated1 & s1_ok
        gated_ok2 = gated2 & s2_ok
        onset1 = gated_ok1 & ~self._last_s1_gated_ok
        onset2 = gated_ok2 & ~self._last_s2_gated_ok
        first1 = p1 & s1_ok & ~self._s1_cycle_latched
        first2 = p2 & s2_ok & ~self._s2_cycle_latched
        phase_next = self.phase.clone()
        phase_next = torch.where(advance1, torch.ones_like(phase_next), phase_next)
        phase_next = torch.where(advance2, torch.full_like(phase_next, 2), phase_next)
        phase_changed = phase_next != self.phase
        self.phase = phase_next
        retry_mask = retry1 | retry2
        clear_timers = phase_changed | retry_mask
        self.t_phase = torch.where(clear_timers, torch.zeros_like(self.t_phase), self.t_phase)
        self.retry = torch.where(retry_mask, self.retry + 1, self.retry)
        self.phase_command[:] = self.phase.float()
        self._s1_confirm_elapsed = torch.where(clear_timers, torch.zeros_like(self._s1_confirm_elapsed), self._s1_confirm_elapsed)
        self._s2_confirm_elapsed = torch.where(clear_timers, torch.zeros_like(self._s2_confirm_elapsed), self._s2_confirm_elapsed)
        self._inverted_confirm_elapsed = torch.where(clear_timers, torch.zeros_like(self._inverted_confirm_elapsed), self._inverted_confirm_elapsed)
        s1_milestone = advance1 & ~self._s1_awarded
        s2_milestone = advance2 & ~self._s2_awarded
        self._s1_awarded |= advance1
        self._s2_awarded |= advance2
        self._s1_onset = torch.where(onset1, self.t_phase, self._s1_onset)
        self._s2_onset = torch.where(onset2, self.t_phase, self._s2_onset)
        self._s1_criterion_first = torch.where(first1, self.t_phase, self._s1_criterion_first)
        self._s2_criterion_first = torch.where(first2, self.t_phase, self._s2_criterion_first)
        self._s1_cycle_latched |= first1
        self._s2_cycle_latched |= first2
        nan = torch.full_like(self._s1_onset, float("nan"))
        for buf in (self._s1_onset, self._s2_onset, self._s1_criterion_first, self._s2_criterion_first):
            buf.copy_(torch.where(retry_mask, nan, buf))
        self._s1_cycle_latched &= ~retry_mask
        self._s2_cycle_latched &= ~retry_mask
        self._last_s1_gated_ok = gated_ok1 & ~retry_mask
        self._last_s2_gated_ok = gated_ok2 & ~retry_mask
        self._last_s1_ok = s1_ok
        self._last_s2_ok = s2_ok
        self._last_advance1 = advance1
        self._last_advance2 = advance2
        self._last_s1_milestone = s1_milestone
        self._last_s2_milestone = s2_milestone
        self._last_back_to_p1 = torch.zeros_like(advance1)
        self._last_back_to_p2 = torch.zeros_like(advance2)
        self._last_both_inverted = both_inverted
        self._last_s1_confirmed = s1_confirmed
        self._last_s2_confirmed = s2_confirmed
        self._last_inverted_confirmed = inverted_confirmed
        self._last_retry_mask = retry_mask
        self._apply_cycle_reset(running)
        pending = getattr(self, "_pending_episode_reset", None)
        if pending is not None and pending.any():
            self._cycles_this_episode[pending] = 0
            self._pending_episode_reset = torch.zeros_like(pending)


    # 完成单次循环时的部分复位
    def _apply_cycle_reset(self, running: torch.Tensor) -> None:
        ids = self._pending_cycle_reset.nonzero(as_tuple=False).squeeze(-1)
        self._last_cycle_reset = self._pending_cycle_reset.clone()
        self._pending_cycle_reset = torch.zeros_like(self._pending_cycle_reset)
        if len(ids) == 0:
            return
        env = self._env
        env.sim.reset(ids)
        scene_reset = getattr(env.scene, "reset", None)
        if scene_reset is not None:
            scene_reset(ids)
        apply_fallen_state(env, ids)
        env.observation_manager.reset(ids)
        env.action_manager.reset(ids)
        keep_d = self._wall_d[ids].clone()
        self._resample_command(ids)
        self._wall_d[ids] = keep_d
        self._push_wall(ids)
        self._clear_cycle_state(ids)
        env.sim.forward()


    # 站立窗口的累积与完成的判定
    def stand_reward_and_pulse(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hold, strict = self.stand_gate()
        in_p3 = self.phase == 2
        running = self._env.episode_length_buf > 0
        active = hold & in_p3
        vel_rms = self.standing_metrics()[2]
        obs_t, obs_v = self._stand_elapsed, self._stand_vel_integral
        self._stand_elapsed, self._stand_vel_integral = BackupCommand._update_stand_window(obs_t, obs_v, active, running, vel_rms, float(self._update_dt))
        steps = max(1.0, abs(STAND_CONFIRM_DURATION) / max(abs(float(self._update_dt)), 1e-9))
        eps = torch.finfo(self._stand_elapsed.dtype).eps * steps * 4.0
        mean_vel = self._stand_vel_integral / self._stand_elapsed.clamp_min(torch.finfo(self._stand_elapsed.dtype).tiny)
        confirmed = active & running & strict & (self._stand_elapsed >= (STAND_CONFIRM_DURATION - eps)) & (mean_vel <= STAND_VEL_MEAN_MAX)
        confirmed = confirmed & ~self._pending_cycle_reset
        held = active & running & (self._stand_elapsed >= (STAND_CONFIRM_DURATION - eps))
        self._ep_stood_pose |= held & strict
        stand_active = active & running
        self._ep_stood_onset |= stand_active & ~self._prev_stand_active
        self._prev_stand_active = stand_active
        self._ep_had_success |= confirmed
        self._ep_cycle_count += confirmed.long()
        self._last_cycle_end_pulse |= confirmed
        self._pending_cycle_reset |= confirmed
        log = getattr(self._env, "extras", {}).get("log") if hasattr(self._env, "extras") else None
        if log is not None:
            window = active & running & (self._stand_elapsed > 0.0)
            count = window.sum().clamp_min(1).to(self._stand_elapsed.dtype)
            log["Progress/standing"] = (strict & in_p3).float().mean().item()
            log["Progress/stand_hold"] = self._stand_elapsed.mean().item()
            log["Progress/stand_mean_vel"] = (torch.where(window, mean_vel, torch.zeros_like(mean_vel)).sum() / count).item()
        return obs_t, obs_v, mean_vel, confirmed


    def windowed_mean_vel(self) -> torch.Tensor:
        elapsed = getattr(self, "_stand_elapsed", None)
        integral = getattr(self, "_stand_vel_integral", None)
        if elapsed is None or integral is None:
            return torch.full_like(self.t_phase, float("nan"))
        return integral / elapsed.clamp_min(torch.finfo(elapsed.dtype).tiny)


    # 消费本步的循环完成脉冲 — 奖励项调用后立刻清零, 保证恰好结算一次。
    def consume_cycle_end_pulse(self) -> torch.Tensor:
        pulse = getattr(self, "_last_cycle_end_pulse", None)
        if pulse is None:
            pulse = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_cycle_end_pulse = torch.zeros_like(pulse)
        return pulse


    @property
    def cycle_completed_pulse(self) -> torch.Tensor:
        return getattr(self, "_last_cycle_reset", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))


    def _update_metrics(self) -> None:
        self._ensure_buffers()
        log = self._env.extras["log"]
        log["Progress/enter_p2"] = self._s1_awarded.float().mean().item()
        log["Progress/enter_p3"] = self._s2_awarded.float().mean().item()
        log["Progress/relapse"] = self._last_both_inverted.float().mean().item()
        log["Progress/s1_dev_s"] = _mean_valid(self.s1_dev_early)
        log["Progress/s2_dev_s"] = _mean_valid(self.s2_dev_early)
        pose_cos = self._pose_cos()
        thr = self._pose_cos_threshold
        stage = torch.where(pose_cos >= thr, torch.ones_like(pose_cos, dtype=torch.long), torch.where(pose_cos <= -thr, 
                            -torch.ones_like(pose_cos, dtype=torch.long), torch.zeros_like(pose_cos, dtype=torch.long)))
        stage = torch.where(torch.isfinite(pose_cos), stage, torch.zeros_like(stage))
        self._pose_stage = stage
        both_inv = (stage[:, 0] == -1) & (stage[:, 1] == -1)
        s1_pose = (stage[:, 0] == -1) & (stage[:, 1] == 1)
        s2_pose = (stage[:, 0] == 1) & (stage[:, 1] == 1)
        wrong = (stage[:, 0] == 1) & (stage[:, 1] == -1)
        cls = torch.full_like(stage[:, 0], 6)
        cls = torch.where(s2_pose, torch.full_like(cls, 4), cls)
        cls = torch.where(s1_pose, torch.full_like(cls, 2), cls)
        cls = torch.where(wrong, torch.full_like(cls, 5), cls)
        cls = torch.where(both_inv, torch.full_like(cls, 0), cls)
        mid_h = (stage[:, 0] == -1) & (stage[:, 1] == 0)
        mid_f = (stage[:, 0] == 0) & (stage[:, 1] == 1)
        cls = torch.where(mid_f, torch.full_like(cls, 3), cls)
        cls = torch.where(mid_h, torch.full_like(cls, 1), cls)
        running = self._env.episode_length_buf > 0
        early = s2_pose & (self.phase == 0) & ~self._s1_awarded & running
        rising = early & ~self._early_s2_entered
        self._early_s2_count += rising.float()
        self._early_s2_entered |= early
        step = self._update_dt if (self._update_dt > 0.0 and isfinite(self._update_dt)) else 0.0
        self._early_s2_elapsed += early.float() * step
        lost = (self.phase >= 1) & ~s1_pose & running
        rising_lost = lost & ~self._s1_hold_lost
        self._s1_hold_lost_count += rising_lost.float()
        self._s1_hold_lost |= lost
        completed = self._last_cycle_reset
        log["Cycle/completed"] = completed.float().mean().item()
        self._cycles_this_episode += completed.long()
        log["Cycle/per_episode"] = self._cycles_this_episode.float().mean().item()
        self._last_cycle_reset = torch.zeros_like(completed)
        reset_buf = getattr(self._env, "reset_buf", None)
        self._pending_episode_reset = (reset_buf.clone() if reset_buf is not None else torch.zeros_like(completed))
        log["Phase/p2"] = (self.phase == 1).float().mean().item()
        log["Gate/s1_pose"] = self._last_s1_ok.float().mean().item()
        log["Gate/s2_pose"] = self._last_s2_ok.float().mean().item()


    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        pass



@dataclass(kw_only=True)
class BackupCommandCfg(CommandTermCfg):
    asset_name: str = "robot"
    resampling_time_range: Tuple[float, float] = (1000.0, 1000.0)   # 不重采样 (episode 内固定)
    debug_vis: bool = False
    fixed_time_scale: float | None = None
    window_late_s: float = WINDOW_LATE_S
    pose_angle_tolerance_deg: float = 45.0
    pose_confirm_s: float = 0.10
    inverted_confirm_s: float = 0.15
    wall_d_min: float | None = None
    wall_d_max: float | None = None
    wall_d_min_frac: float = 0.3
    wall_x_pos: float | None = None

    @dataclass
    class VizCfg:
        z_offset: float = 0.1
        scale: float = 1.0

    viz: VizCfg = field(default_factory=VizCfg)
    class_type: type[CommandTerm] = BackupCommand

    def build(self, env: "ManagerBasedRlEnv") -> CommandTerm:
        return self.class_type(self, env)
