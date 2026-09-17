from __future__ import annotations
import torch
from math import cos, isfinite, radians
from typing import TYPE_CHECKING, Tuple
from dataclasses import dataclass, field
from mjlab.managers import CommandTermCfg
from mjlab.managers.command_manager import CommandTerm
from .curriculums import get_curriculum_time_scale
from .indices import _MODEL_INDICES, resolve_model_indices
from .timing import P1_BUFFER_DURATION, P1_END, P2_BUFFER_DURATION, P2_DURATION
from .timing import STAND_GROUND_HEIGHT, STAND_MIN_HEIGHT, STAND_MIN_HEIGHT_STAY, STAND_TARGET_HEIGHT
from .timing import STAND_UPRIGHT_COS, STAND_UPRIGHT_COS_STAY

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# 阶段状态检测阈值
_GROUND_TH = 0.03     # S1 平躺高度阈值
_GROUND_TH_S2 = 0.04  # S2 趴地高度阈值 (段3末 H 后肢略翘≈0.034)
# 阶段预期时长 (名义, ×λ)
_P1_EXPECT = P1_END
_P2_EXPECT = P2_DURATION
_MAX_RETRY = 5


class BackupCommand(CommandTerm):
    cfg: "BackupCommandCfg"
    def __init__(self, cfg: "BackupCommandCfg", env: "ManagerBasedRlEnv"):
        for name in ("p1_buffer_s", "p2_buffer_s"):
            value = getattr(cfg, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须为有限的非负实际秒数")
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
        self.command_tensor = torch.zeros(self.num_envs, 7, device=self.device)
        self.vel_command = self.command_tensor[:, 0]
        self.height_f_command = self.command_tensor[:, 1]
        self.height_h_command = self.command_tensor[:, 2]
        self.gait_freq_command = self.command_tensor[:, 3]
        self.curvature_command = self.command_tensor[:, 4]
        self.time_scale_command = self.command_tensor[:, 5]
        self.phase_command = self.command_tensor[:, 6]

        # 阶段状态机状态 (每 env 独立)
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.t_phase = torch.zeros(self.num_envs, device=self.device)
        self.retry = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 状态机指标缓存 (供 _update_metrics 记录上一步检测结果)
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
        # 确认时间使用实际环境秒数，不乘参考时间缩放 λ。
        self._s1_confirm_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._s2_confirm_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._inverted_confirm_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._s1_awarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._s2_awarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pose_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._pose_cos_cache: torch.Tensor | None = None
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

    # 当前阶段的阶段内时间 t_phase (reference 查询用)
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

    # 按课程采样 time_scale λ (episode 内固定); 其余字段与 Slalom/Tunnel 语义对齐
    def _resample_command(self, env_ids: torch.Tensor) -> None:
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
        # 状态机重置
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
        self._s1_awarded[env_ids] = False
        self._s2_awarded[env_ids] = False

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
            self._resample_command(env_ids)
        return extras

    def compute(self, dt: float) -> None:
        # CommandTerm._update_command() 不接收 dt，因此在调用父类前暂存本次真实步长。
        self._update_dt = dt
        super().compute(dt)

    # 身体段方向判定 — 用腹/背标记 site 的世界坐标, 不依赖四元数约定
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
        # 返回 [env, segment] 的 belly->back 世界 Z 方向余弦, 无效值置 NaN。
        return torch.stack((self._segment_u(0), self._segment_u(1)), dim=1)

    def _flags_from_cos(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 方向余弦 -> [env, segment] 的正置与倒置标记；中间姿态和未知均为 False。
        finite = torch.isfinite(u)
        threshold = self._pose_cos_threshold
        return finite & (u >= threshold), finite & (u <= -threshold)

    def _pose_flags(self) -> tuple[torch.Tensor, torch.Tensor]:
        u = self._pose_cos()
        # 状态机本步的方向余弦留存给区间奖励复用, 避免重复读取 site。
        self._pose_cos_cache = u
        return self._flags_from_cos(u)

    def _get_pose_flags(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 状态机同一步复用一次 site 方向计算；外部诊断调用则即时计算。
        cached = getattr(self, "_pose_cache", None)
        if cached is None:
            return self._pose_flags()
        return cached

    def _get_pose_cos(self) -> torch.Tensor:
        # 与 _get_pose_flags 相同的缓存语义。
        cached = getattr(self, "_pose_cos_cache", None)
        if cached is None:
            return self._pose_cos()
        return cached

    @staticmethod
    def _ramp(u: torch.Tensor, sign: float) -> torch.Tensor:
        # 方向余弦 -> [0, 1] 线性爬升, 原点取 u=0。
        return (sign * u).clamp(0.0, 1.0)

    @property
    def progress_s1(self) -> torch.Tensor:
        # 后段翻正进度: 仰卧(-1) -> 俯卧(+1) 单调爬升, 未知姿态按 0 处理。
        u = torch.nan_to_num(self._get_pose_cos(), nan=0.0)
        return self._ramp(u[:, 1], 1.0)

    @property
    def progress_s2(self) -> torch.Tensor:
        # 前段进度乘以后段进度; 前段系数取 (clamp(uF)+1)/2 而非 clamp(uF), 目的是让两项
        # 合成的激励地形全程单调 (取值依据见技术细节 §2)。
        u = torch.nan_to_num(self._get_pose_cos(), nan=0.0)
        front = (u[:, 0].clamp(-1.0, 1.0) + 1.0) / 2.0
        return self._ramp(u[:, 1], 1.0) * front

    # 站立几何与连续进度共用背腹轴与两段高度；阶段门控由调用方负责。
    def standing_metrics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 返回 [env] 的两段最差背腹余弦、最低躯干高度、14 个驱动关节速度 RMS。
        u = self._pose_cos()
        heights = torch.stack((self._body_height(_MODEL_INDICES.f_body_id),
                               self._body_height(_MODEL_INDICES.h_body_id)), dim=1)
        # 取两段的较低/较差者, 不允许只抬起或只翻正一端。
        u_floor = u.amin(dim=1)
        h_floor = heights.amin(dim=1)
        return u_floor, h_floor, self._joint_vel_rms()

    def standing_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        u = self._pose_cos()
        heights = torch.stack((self._body_height(_MODEL_INDICES.f_body_id),
                               self._body_height(_MODEL_INDICES.h_body_id)), dim=1)
        valid = torch.isfinite(u).all(dim=1) & torch.isfinite(heights).all(dim=1)
        u_floor = u.amin(dim=1)
        h_floor = heights.amin(dim=1)
        # 严格侧只含几何, 速度条件由 terminations 与静止奖励各自负责。
        standing = valid & (u_floor > STAND_UPRIGHT_COS) & (h_floor > STAND_MIN_HEIGHT)
        # 站立进度: 朝向从锥边界到完全正置、高度从贴地到站立目标各线性增长, 完全站立为 1。
        cone = float(self._pose_cos_threshold)
        orient = ((u_floor - cone) / (1.0 - cone)).clamp(0.0, 1.0)
        height_progress = ((h_floor - STAND_GROUND_HEIGHT) / (STAND_TARGET_HEIGHT - STAND_GROUND_HEIGHT)).clamp(0.0, 1.0)
        progress = orient * height_progress
        return standing, torch.where(valid, progress, torch.zeros_like(progress))

    def stand_gate(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 返回 (维持站立窗口的宽松门控, 结算与静止奖励要求的严格门控); 速度门限不在此处。
        u_floor, h_floor, vel_rms = self.standing_metrics()
        finite = torch.isfinite(u_floor) & torch.isfinite(h_floor) & torch.isfinite(vel_rms)
        hold = finite & (u_floor > STAND_UPRIGHT_COS_STAY) & (h_floor > STAND_MIN_HEIGHT_STAY)
        strict = finite & (u_floor > STAND_UPRIGHT_COS) & (h_floor > STAND_MIN_HEIGHT)
        return hold, strict

    def _joint_vel_rms(self) -> torch.Tensor:
        # 14 个驱动关节速度的瞬时 RMS (rad/s), 顺序与 _ACTUATED_JOINT_NAMES 一致。
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

    def _check_S1(self) -> torch.Tensor:
        # S1 瞬时候选：前段倒置、后段正置, 且两段躯干都平躺贴地。
        f_inv = self._segment_inverted(0)
        h_up = self._segment_upright(1)
        fz = self._body_height(_MODEL_INDICES.f_body_id)
        hz = self._body_height(_MODEL_INDICES.h_body_id)
        return f_inv & h_up & (fz < _GROUND_TH) & (hz < _GROUND_TH)

    def _check_S2(self) -> torch.Tensor:
        # S2 瞬时候选取两条路径的并集: 贴地翻正, 或已达成站立几何 (见技术细节 §1)。
        f_up = self._segment_upright(0)
        h_up = self._segment_upright(1)
        fz = self._body_height(_MODEL_INDICES.f_body_id)
        hz = self._body_height(_MODEL_INDICES.h_body_id)
        grounded = f_up & h_up & torch.isfinite(fz) & torch.isfinite(hz) & (fz < _GROUND_TH_S2) & (hz < _GROUND_TH_S2)
        standing, _ = self.standing_state()
        return grounded | standing

    def _check_both_inverted(self) -> torch.Tensor:
        # 双倒回退只看两个身体段的方向，不借用 S1/S2 的高度条件。
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
        # 只有完整候选且真实环境步才累加；候选中断或非运行环境立即清零。
        zero = torch.zeros_like(elapsed)
        elapsed = torch.where(candidate & running, elapsed, zero)
        if dt > 0.0 and isfinite(dt):
            elapsed = elapsed + (candidate & running).to(elapsed.dtype) * dt
        # 给 float32 的 0.01×10/15 步留出舍入余量，不改变实际确认秒数。
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
        # 站立窗口: 窗口成立时累加时长 T 与"速度×时间"积分 V, 中断则一起清零。
        # 结算量是 V/T 即窗口平均关节速度, 设计与理由见技术细节 §7.2.1。
        run = running & active
        step = run.to(elapsed.dtype) * (dt if (dt > 0.0 and isfinite(dt)) else 0.0)
        zero = torch.zeros_like(elapsed)
        speed = torch.nan_to_num(vel_rms, nan=0.0, posinf=0.0, neginf=0.0)
        nxt_t = torch.where(run, elapsed + step, zero)
        nxt_v = torch.where(run, vel_integral + step * speed, zero)
        return nxt_t, nxt_v

    def _update_command(self) -> None:
        dt = float(self._update_dt)
        # 兼容无物理仿真的单元测试/回放 mock；真实实例在构造时已完成初始化。
        if not hasattr(self, "_s1_confirm_elapsed"):
            self._s1_confirm_elapsed = torch.zeros_like(self.t_phase)
            self._s2_confirm_elapsed = torch.zeros_like(self.t_phase)
            self._inverted_confirm_elapsed = torch.zeros_like(self.t_phase)
        if not hasattr(self, "_s1_awarded"):
            self._s1_awarded = torch.zeros_like(self.phase, dtype=torch.bool)
            self._s2_awarded = torch.zeros_like(self.phase, dtype=torch.bool)
        if not hasattr(self, "_pose_cache"):
            self._pose_cache = None
        if not hasattr(self, "_pose_cos_cache"):
            self._pose_cos_cache = None
        lam = self.time_scale_command.clamp(min=0.1)
        # t_phase 是所有阶段共用的段内时钟; 只有真实环境步才推进, 避免初始参考提前一步。
        running = self._env.episode_length_buf > 0
        real_dt = dt if dt > 0.0 and isfinite(dt) else 0.0
        self.t_phase = self.t_phase + running.to(self.t_phase.dtype) * real_dt
        # 本步缓存姿态, 避免 S1/S2/双倒重复读取 site; 测试可注入替代的 _pose_flags。
        pose_flags = getattr(self, "_pose_flags", None)
        self._pose_cache = pose_flags() if pose_flags is not None else None
        # 瞬时候选只用于日志与确认计时, 不能直接当作阶段转移。
        p1 = self.phase == 0
        expected1 = _P1_EXPECT * lam
        s1_ok = self._check_S1()
        p2 = self.phase == 1
        expected2 = _P2_EXPECT * lam
        s2_ok = self._check_S2()
        both_inverted = getattr(self, "_check_both_inverted", lambda: torch.zeros_like(s1_ok))()
        self._pose_cache = None
        self._pose_cos_cache = None

        self._s1_confirm_elapsed, s1_confirmed = BackupCommand._update_confirmation(
            self._s1_confirm_elapsed, s1_ok, running, real_dt, float(self.cfg.pose_confirm_s)
        )
        self._s2_confirm_elapsed, s2_confirmed = BackupCommand._update_confirmation(
            self._s2_confirm_elapsed, s2_ok, running, real_dt, float(self.cfg.pose_confirm_s)
        )
        self._inverted_confirm_elapsed, inverted_confirmed = BackupCommand._update_confirmation(
            self._inverted_confirm_elapsed,
            both_inverted,
            running,
            real_dt,
            float(self.cfg.inverted_confirm_s),
        )

        # 阶段推进: P1 -> P2 -> P3 单向, 只在阶段内做超时重试。
        # 取消阶段回退的理由(策略会挑阶段套利、形成极限环)见技术细节 §2。
        advance1 = p1 & s1_confirmed
        advance2 = p2 & s2_confirmed
        retry1 = p1 & (self.t_phase >= expected1 + self.cfg.p1_buffer_s) & ~s1_confirmed
        p2_deadline = expected2 + self.cfg.p2_buffer_s
        # 截止前刚进入候选时允许完成这次连续确认, 参考保持 T3 末端; 中断即清零。
        s2_pending = s2_ok & (self._s2_confirm_elapsed > 0.0) & ~s2_confirmed
        s2_grace = s2_pending & (self.t_phase < p2_deadline + self.cfg.pose_confirm_s)
        retry2 = p2 & (self.t_phase >= p2_deadline) & ~s2_confirmed & ~s2_grace
        # P3 无超时机制; inverted_confirmed 与 both_inverted 仅作诊断。
        phase_next = self.phase.clone()
        phase_next = torch.where(advance1, torch.ones_like(phase_next), phase_next)
        phase_next = torch.where(advance2, torch.full_like(phase_next, 2), phase_next)
        phase_changed = phase_next != self.phase
        self.phase = phase_next
        # 阶段切换和阶段重试都清理本 env 的阶段/候选确认时钟，但不动物理状态。
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
        # 保存本步检测结果供 _update_metrics 记录 (metrics 在 command 前被调用, 记录上一步状态)
        self._last_s1_ok = s1_ok
        self._last_s2_ok = s2_ok
        self._last_advance1 = advance1
        self._last_advance2 = advance2
        self._last_s1_milestone = s1_milestone
        self._last_s2_milestone = s2_milestone
        # 阶段已单向, 这两个字段恒为 False; 保留供录像/日志字段兼容。
        self._last_back_to_p1 = torch.zeros_like(advance1)
        self._last_back_to_p2 = torch.zeros_like(advance2)
        self._last_both_inverted = both_inverted
        self._last_s1_confirmed = s1_confirmed
        self._last_s2_confirmed = s2_confirmed
        self._last_inverted_confirmed = inverted_confirmed
        self._last_retry_mask = retry_mask

    def _update_metrics(self) -> None:
        # 只记录三组共 10 条: Progress/* 回合级成就, Phase/* 当前阶段, Gate/* 姿态与推进速率。
        # 注意 _last_* 是上一步的检测结果(metrics 在 _update_command 之前被调用);
        # 这些属性全部保留, SQuRo_Backup_play.py 的录像列依赖它们。
        log = self._env.extras["log"]
        log["Progress/enter_p2"] = self._s1_awarded.float().mean().item()
        log["Progress/enter_p3"] = self._s2_awarded.float().mean().item()
        log["Progress/relapse"] = self._last_both_inverted.float().mean().item()
        log["Phase/p2"] = (self.phase == 1).float().mean().item()
        log["Phase/p3"] = (self.phase == 2).float().mean().item()
        log["Phase/retry"] = self.retry.float().mean().item()
        log["Gate/s1_pose"] = self._last_s1_ok.float().mean().item()
        log["Gate/s2_pose"] = self._last_s2_ok.float().mean().item()
        log["Gate/s1_advance"] = self._last_advance1.float().mean().item()
        log["Gate/s2_advance"] = self._last_advance2.float().mean().item()

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        pass


@dataclass(kw_only=True)
class BackupCommandCfg(CommandTermCfg):
    asset_name: str = "robot"
    resampling_time_range: Tuple[float, float] = (1000.0, 1000.0)   # 不重采样 (episode 内固定)
    debug_vis: bool = False
    fixed_time_scale: float | None = None
    # P1=T1+T2、P2=T3；末端等待均为实际秒，不随 λ 缩放。
    p1_buffer_s: float = P1_BUFFER_DURATION
    p2_buffer_s: float = P2_BUFFER_DURATION
    # site 方向夹角容差与连续确认时长（均为实际秒，不乘 λ）。
    pose_angle_tolerance_deg: float = 45.0
    pose_confirm_s: float = 0.10
    inverted_confirm_s: float = 0.15

    @dataclass
    class VizCfg:
        z_offset: float = 0.1
        scale: float = 1.0

    viz: VizCfg = field(default_factory=VizCfg)
    class_type: type[CommandTerm] = BackupCommand

    def build(self, env: "ManagerBasedRlEnv") -> CommandTerm:
        return self.class_type(self, env)
