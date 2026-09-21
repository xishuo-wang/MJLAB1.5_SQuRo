from __future__ import annotations
import torch
from math import cos, isfinite, radians
from typing import TYPE_CHECKING, Tuple
from dataclasses import dataclass, field
from mjlab.managers import CommandTermCfg
from mjlab.managers.command_manager import CommandTerm
from .curriculums import get_curriculum_time_scale
from .indices import _MODEL_INDICES, resolve_model_indices
from .timing import P1_END, P2_DURATION, PRE_DURATION, STAND_CONFIRM_DURATION
from .timing import STAND_VEL_MEAN_MAX, WINDOW_LATE_S
from .timing import STAND_GROUND_HEIGHT, STAND_MIN_HEIGHT, STAND_MIN_HEIGHT_STAY, STAND_TARGET_HEIGHT
from .timing import STAND_UPRIGHT_COS, STAND_UPRIGHT_COS_STAY

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# 阶段状态检测阈值
_GROUND_TH = 0.03     # S1 平躺高度阈值
_GROUND_TH_S2 = 0.04  # S2 趴地高度阈值 (段3末 H 后肢略翘≈0.034)
# 阶段预期时长 (名义, ×λ)
# P1 期望要含前置收腿段: 相位时钟从送参考那一刻起算, 而参考的前 PRE_DURATION 秒只是收腿。
_P1_EXPECT = PRE_DURATION + P1_END
_P2_EXPECT = P2_DURATION
_MAX_RETRY = 5


# 对含 NaN 的记录取"有效值均值"; 没有有效值时返回 0 (供条件型日志使用)。
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
        # 时序偏差记录: NaN 表示本回合还没有达成记录。
        # _s*_onset 是"导致结算的那次确认脉冲"的起点(实际秒, 阶段内计时), 入奖励质量核;
        # _s*_dev_early/_s*_dev_late 是它与名义段末之差, 分开存两份便于各自 clamp 与记录。
        self._s1_onset = torch.full((self.num_envs,), float("nan"), device=self.device)
        self._s2_onset = torch.full((self.num_envs,), float("nan"), device=self.device)
        # 未加掩码的真实首次达成时刻, 只作诊断, 不参与奖励。
        self._s1_criterion_first = torch.full((self.num_envs,), float("nan"), device=self.device)
        self._s2_criterion_first = torch.full((self.num_envs,), float("nan"), device=self.device)
        # "本次尝试已锁存首次达成"标志: 每个循环只锁一次, 重试时清除(否则记录会被后续
        # 上升沿覆盖, 提前达成可被"短暂跨出判据再进入"洗掉)。
        self._s1_cycle_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._s2_cycle_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 本回合已完成的循环数: 完成时 +1, 归零推迟到 _update_metrics 上报之后(见那里的注释)。
        self._cycles_this_episode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 本步"回合已结束"掩码的快照: metrics 在回合重置之前取, _update_command 之后用它归零计数。
        self._pending_episode_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 门控开启前的候选上一拍状态, 用于取脉冲起点(eligible 内的上升沿)。
        self._last_s1_gated_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_s2_gated_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 循环完成 (站立 1.5s 窗口) 的判定状态。判定必须发生在奖励项里、复位之前,
        # 因此窗口张量放在这里而不是 terminations; 见 _stand_reward_and_pulse 的时序说明。
        self._stand_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._stand_vel_integral = torch.zeros(self.num_envs, device=self.device)
        # 待复位标志: 本步冻结合成完成脉冲, 下一步在 _update_command 里执行部分复位。
        # _last_cycle_end_pulse 必须在这里就建好 —— 奖励项在 _update_command 之后才被调用,
        # 而它以 |= 的方式累积, 首次读到时不能是缺失属性。
        self._pending_cycle_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_cycle_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_cycle_end_pulse = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._s1_awarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._s2_awarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pose_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._pose_cos_cache: torch.Tensor | None = None
        # 姿态识别观测 (技术细节 §7.8): 与 phase 分开, 只作指标, 不参与任何推进判定。
        # early1 = "已进入过 S2 姿态格但 S1 尚未确认"; s1_hold = "S1 确认后仍在 (倒置,正置) 格"。
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

    # 里程碑时间质量核的输入: 真实首次到达时刻与名义段末之差(实际秒, 迟为正; NaN = 尚未达成)。
    # 必须用 _s*_criterion_first(未加任何掩码), 不能用 _s*_onset —— 后者被"参考播完门"夹住,
    # 恒等于段末, 会让核恒等于 1.0, 等于没有(实测确认过)。
    @property
    def s1_dev_early(self) -> torch.Tensor:
        return self._s1_criterion_first - _P1_EXPECT * self.time_scale_command

    @property
    def s2_dev_early(self) -> torch.Tensor:
        return self._s2_criterion_first - _P2_EXPECT * self.time_scale_command

    @property
    def s1_dev_late(self) -> torch.Tensor:
        return self._s1_criterion_first - _P1_EXPECT * self.time_scale_command

    @property
    def s2_dev_late(self) -> torch.Tensor:
        return self._s2_criterion_first - _P2_EXPECT * self.time_scale_command

    # 建立 _resample_command / _clear_cycle_state 会写的全部缓冲。
    # 抽出来是为了让"回合重置"与"循环复位"两条路径共用同一份生命周期定义 ——
    # 历史上就是因为站立窗口只清在 events.reset_model 里, 完整回合重置漏清 pending 标志。
    # 也可以被不跑 __init__ 的测试替身直接调用。
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

    # 按课程采样 time_scale λ (episode 内固定); 其余字段与 Slalom/Tunnel 语义对齐
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._ensure_buffers()      # 可能被不跑 __init__ 的测试替身直接调用
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
        self._s1_onset[env_ids] = float("nan")
        self._s2_onset[env_ids] = float("nan")
        self._s1_criterion_first[env_ids] = float("nan")
        self._s2_criterion_first[env_ids] = float("nan")
        self._last_s1_gated_ok[env_ids] = False
        self._last_s2_gated_ok[env_ids] = False
        self._s1_awarded[env_ids] = False
        self._s2_awarded[env_ids] = False
        # 姿态识别: 计数按回合清, "本轮是否已进入过" 按循环清 (见 _clear_cycle_state)。
        self._early_s2_count[env_ids] = 0.0
        self._s1_hold_lost_count[env_ids] = 0.0
        # 完整回合重置的入口: 由 reset() 调用。只清"每次(重)采样都该回到起点"的字段;
        # 循环级状态(站立窗口/latch/pending)由 _clear_cycle_state 统一清。
        self._clear_cycle_state(env_ids)

    # 清"新循环内部状态" — 站立窗口、待复位标志、锁存等。
    # **不碰 _last_cycle_reset**: 那是"本步完成事件", 必须活到消费者(metrics 与回放)读走
    # 之后才能清 —— 否则完成事件在复位流程里就被抹掉, Cycle/* 恒为 0、回放也判不出 DONE。
    # (这正是上一版的 bug: _apply_cycle_reset 写完后立刻被这里清成 False。)
    # "本回合循环数"的归零也不在这里, 而在 _update_metrics 消费完重置掩码之后 ——
    # 因为完整回合重置发生在 metrics **之前**, 在这里清会让超时回合的最终计数读成 0。
    def _clear_cycle_state(self, env_ids: torch.Tensor) -> None:
        self._stand_elapsed[env_ids] = 0.0
        self._stand_vel_integral[env_ids] = 0.0
        self._pending_cycle_reset[env_ids] = False
        self._last_cycle_end_pulse[env_ids] = False
        self._s1_cycle_latched[env_ids] = False
        self._s2_cycle_latched[env_ids] = False
        # 姿态识别诊断也按循环清 (每个循环重新观察一次提前到位)。
        self._early_s2_entered[env_ids] = False
        self._early_s2_elapsed[env_ids] = 0.0
        self._s1_hold_lost[env_ids] = False

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
        if not hasattr(self, "_last_s1_confirmed"):
            self._last_s1_confirmed = torch.zeros_like(self.phase, dtype=torch.bool)
            self._last_s2_confirmed = torch.zeros_like(self.phase, dtype=torch.bool)
        # 单测用 object.__new__ 造实例, 可能没有任何缓冲; 统一在这里补齐。
        self._ensure_buffers()
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
        # 必须等本段参考播完(expected)才能推进: 姿态判据可被抄近路提前满足, 一提前推进
        # 参考就会瞬移到段末(实测 P1->P2 一步跳 1.37 rad), 跟踪奖励随之失去意义。
        # 取消阶段回退的理由(策略会挑阶段套利、形成极限环)见技术细节 §2。
        # 可接受区间 = [段末, 段末 + 晚侧余量]: 早侧地板已删除 —— 它被"参考播完门"完全吞噬
        # (达成时刻恒为 max(真实到达时刻, 段末)), 0.60λ 的地板从未起过作用。
        # 真正拦"提前到达并保持"的是里程碑时间质量核, 按"真实首次到达时刻"打折。
        # 设计依据见技术细节开头的"2026-09-18 达成时刻约束 + 循环复位"一节。
        close1 = expected1 + float(self.cfg.window_late_s)
        close2 = expected2 + float(self.cfg.window_late_s)
        gated1 = p1 & (self.t_phase <= close1)
        gated2 = p2 & (self.t_phase <= close2)
        # 门控外的候选不累积确认时长, 避免门控前起算的确认把阶段直接放行。
        self._s1_confirm_elapsed = torch.where(p1 & ~gated1, torch.zeros_like(self._s1_confirm_elapsed), self._s1_confirm_elapsed)
        self._s2_confirm_elapsed = torch.where(p2 & ~gated2, torch.zeros_like(self._s2_confirm_elapsed), self._s2_confirm_elapsed)
        # 必须在清零之后重算 confirmed: _update_confirmation 返回的是清零前的判定,
        # 直接沿用会让"门控外已攒满"的环境跳过门控与窗界, 当步就推进。
        s1_confirmed = s1_confirmed & gated1
        s2_confirmed = s2_confirmed & gated2
        advance1 = p1 & s1_confirmed & (self.t_phase >= expected1)
        advance2 = p2 & s2_confirmed & (self.t_phase >= expected2)
        retry1 = p1 & (self.t_phase > close1) & ~advance1
        retry2 = p2 & (self.t_phase > close2) & ~advance2
        # 脉冲起点用于诊断与相位推进, 取区间内候选的上升沿。
        gated_ok1 = gated1 & s1_ok
        gated_ok2 = gated2 & s2_ok
        onset1 = gated_ok1 & ~self._last_s1_gated_ok
        onset2 = gated_ok2 & ~self._last_s2_gated_ok
        first1 = p1 & s1_ok & ~self._s1_cycle_latched
        first2 = p2 & s2_ok & ~self._s2_cycle_latched
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
        # 偏差锁存: 只在脉冲起点那一步记一次, 值取该步的 t_phase(不要读 metrics 时的 t_phase,
        # 那时可能已被阶段切换清零); 早/晚两份同源, 由奖励侧各自 clamp。
        # 重试意味着本次尝试作废, 记录一起清掉; 阶段切换保留(供 P2/P3 继续读)。
        self._s1_onset = torch.where(onset1, self.t_phase, self._s1_onset)
        self._s2_onset = torch.where(onset2, self.t_phase, self._s2_onset)
        # 未加任何掩码的真实首次达成: 每个"尝试"(整个循环, 含各阶段)只锁存一次。
        # 必须用独立的 latch 而不是 _last_s1_ok 的上升沿 —— 后者每次离开再进入候选都会
        # 重新触发, 于是"提前达成 -> 短暂跨出判据 -> 重新进入"就能把记录覆盖成临近段末
        # 的时刻, 从而拿到接近满额的时间奖励(实测可绕过)。
        self._s1_criterion_first = torch.where(first1, self.t_phase, self._s1_criterion_first)
        self._s2_criterion_first = torch.where(first2, self.t_phase, self._s2_criterion_first)
        self._s1_cycle_latched |= first1
        self._s2_cycle_latched |= first2
        nan = torch.full_like(self._s1_onset, float("nan"))
        # 重试 = 本次尝试作废: 记录与 latch 一起清, 使下一次尝试能重新锁存首次达成。
        for buf in (self._s1_onset, self._s2_onset,
                    self._s1_criterion_first, self._s2_criterion_first):
            buf.copy_(torch.where(retry_mask, nan, buf))
        self._s1_cycle_latched &= ~retry_mask
        self._s2_cycle_latched &= ~retry_mask
        # 重试会清候选锁存: 否则重播后候选仍成立时不会产生新的上升沿, 脉冲起点会丢失。
        self._last_s1_gated_ok = gated_ok1 & ~retry_mask
        self._last_s2_gated_ok = gated_ok2 & ~retry_mask
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
        # 循环复位放在最后: 它会把循环状态清零, 但本步的 _last_* 检测结果与冻结脉冲必须留下。
        self._apply_cycle_reset(running)
        # 回合已结束的环境: 归零"本回合循环数"。必须在这里做而不是 metrics 里 ——
        # 完整回合重置发生在 metrics **之前**, 在那里清会把超时回合的最终计数读成 0。
        pending = getattr(self, "_pending_episode_reset", None)
        if pending is not None and pending.any():
            self._cycles_this_episode[pending] = 0
            self._pending_episode_reset = torch.zeros_like(pending)

    # 完成一次循环的部分复位 — 只搬回机器人与短期记忆, 不结束回合。
    # 为什么不在本步做: 完成脉冲要在奖励项里被读走(只领一次)之后才允许清状态。
    def _apply_cycle_reset(self, running: torch.Tensor) -> None:
        ids = self._pending_cycle_reset.nonzero(as_tuple=False).squeeze(-1)
        self._last_cycle_reset = self._pending_cycle_reset.clone()
        self._pending_cycle_reset = torch.zeros_like(self._pending_cycle_reset)
        if len(ids) == 0:
            return
        env = self._env
        from .events import apply_fallen_state
        # 1) 清物理状态: sim.reset 会按 world 掩码清 qpos/qvel/qacc/qacc_warmstart/ctrl/act/
        #    qfrc_applied/M/contact/sensordata/time —— 关键是清掉求解器热启动, 否则复位后
        #    第一步会带着上一循环的接触冲量。
        env.sim.reset(ids)
        # 2) 实体与传感器: contact_sensor.reset 会零掉 air-time 历史并把 last_time 重设到新的
        #    仿真时间, 正好修掉 sim.reset 把仿真时间归零带来的时间基跳变。
        #    兼容测试替身: scene 可能是 dict (取不到 reset) 或带 reset 的对象。
        scene_reset = getattr(env.scene, "reset", None)
        if scene_reset is not None:
            scene_reset(ids)
        # 3) 写仰卧初态 (与回合重置同一份代码)
        apply_fallen_state(env, ids)
        # 4) 清策略侧短期记忆: last_action 的历史环形缓冲与 prev/prev_prev_action。
        env.observation_manager.reset(ids)
        env.action_manager.reset(ids)
        # 5) 清循环状态: 重采 λ、phase=0、t_phase=0、确认计时、里程碑锁存、脉冲起点、站立窗口。
        #    **不碰 _last_cycle_reset**(本步完成事件, 留给 metrics 与回放消费)。
        self._resample_command(ids)
        self._clear_cycle_state(ids)
        # 6) 刷新派生量 —— 写 qpos/qvel 之后必须 forward 才能让 site/body 位置与新状态一致
        #    (entity/data.py 明确要求"写后读前先 forward")。
        env.sim.forward()

    # 站立窗口的累积与"循环完成"判定 — 由奖励项每步调用, 必须在复位之前。
    # 返回本步刚凑满窗口的完成脉冲; 同时把完成脉冲冻结一步(cycle_end_pulse), 供下一步的
    # 奖励项读取并结算 —— 因为 _update_command 在 reward 之后, 直接在本步产生又清除的脉冲
    # 到不了奖励, 这一步的延后是必须的。
    def stand_reward_and_pulse(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hold, strict = self.stand_gate()
        in_p3 = self.phase == 2
        running = self._env.episode_length_buf > 0
        active = hold & in_p3
        vel_rms = self.standing_metrics()[2]
        obs_t, obs_v = self._stand_elapsed, self._stand_vel_integral
        self._stand_elapsed, self._stand_vel_integral = BackupCommand._update_stand_window(
            obs_t, obs_v, active, running, vel_rms, float(self._update_dt))
        # float32 逐步累加的舍入容差（约 7e-5 秒，不改变实际确认秒数）。
        steps = max(1.0, abs(STAND_CONFIRM_DURATION) / max(abs(float(self._update_dt)), 1e-9))
        eps = torch.finfo(self._stand_elapsed.dtype).eps * steps * 4.0
        mean_vel = self._stand_vel_integral / self._stand_elapsed.clamp_min(torch.finfo(self._stand_elapsed.dtype).tiny)
        confirmed = active & running & strict & (self._stand_elapsed >= (STAND_CONFIRM_DURATION - eps)) & (mean_vel <= STAND_VEL_MEAN_MAX)
        # 每个循环只能完成一次: 窗口在复位前一直保持"已攒满", 不加这道锁的话在"完成帧"
        # 与"复位帧"之间(以及同一帧内重复结算时)会反复结算同一个循环。
        confirmed = confirmed & ~self._pending_cycle_reset
        # 冻结: 下一步的奖励项读 _last_cycle_end_pulse, 读走即被置回 False(只领一次)。
        self._last_cycle_end_pulse |= confirmed
        self._pending_cycle_reset |= confirmed
        # 标定日志(不参与判据): 窗口时长、窗口平均速度(判据量本身)与严格几何占比。
        # §7.2.2 明确要求"先看 stand_mean_vel 离门限多远, 再决定调阈值还是改结构" ——
        # 这几个量在重写 terminations 时被漏掉, 导致无法判断卡在"速度"还是"几何不维持"。
        log = getattr(self._env, "extras", {}).get("log") if hasattr(self._env, "extras") else None
        if log is not None:
            window = active & running & (self._stand_elapsed > 0.0)
            count = window.sum().clamp_min(1).to(self._stand_elapsed.dtype)
            log["Progress/standing"] = (strict & in_p3).float().mean().item()
            log["Progress/stand_hold"] = self._stand_elapsed.mean().item()
            log["Progress/stand_mean_vel"] = (
                torch.where(window, mean_vel, torch.zeros_like(mean_vel)).sum() / count).item()
        return obs_t, obs_v, mean_vel, confirmed

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
        # metrics 由 MetricsManager 调用, 可能早于本类任何方法; 单测替身也没有缓冲。
        self._ensure_buffers()
        # 只记录三组共 10 条: Progress/* 回合级成就与节奏偏差, Phase/* 当前阶段, Gate/* 姿态到达速率。
        # 注意 _last_* 是上一步的检测结果(metrics 在 _update_command 之前被调用);
        # 这些属性全部保留, SQuRo_Backup_play.py 的录像列依赖它们。
        log = self._env.extras["log"]
        log["Progress/enter_p2"] = self._s1_awarded.float().mean().item()
        log["Progress/enter_p3"] = self._s2_awarded.float().mean().item()
        log["Progress/relapse"] = self._last_both_inverted.float().mean().item()
        # 主指标直接用未过窗的真实首次达成时刻减去名义段末: 迟为正、负值即"提前达成"。
        # 曾同时上报 Data/s*_onset_s(过窗后)与 Data/s*_first_s, 但现行门控把两者压到几乎
        # 同值(都≈段末), 纯冗余, 已删; 需要真实到达时刻时看这里的 dev 即可。
        log["Progress/s1_dev_s"] = _mean_valid(self.s1_dev_early)
        log["Progress/s2_dev_s"] = _mean_valid(self.s2_dev_early)
        # 姿态识别观测量 (技术细节 §7.8): 与相位推进完全解耦, **只写日志**。
        # 用与门控同一套 ±45° 锥把每段分成 -1=倒置 / 0=过渡 / +1=正置, 且记录"到达顺序"。
        # 之所以需要它: 相位是单向的(只有 0->1->2), phase_retry 只反映超时、relapse 只反映双倒,
        # 都抓不到"已到达 S2 姿态又退回 S1 姿态"这种过程信息。
        # 阈值直接复用 _pose_cos_threshold, 不引入第二套判定标准。
        pose_cos = self._pose_cos()
        thr = self._pose_cos_threshold
        stage = torch.where(pose_cos >= thr, torch.ones_like(pose_cos, dtype=torch.long),
                            torch.where(pose_cos <= -thr, -torch.ones_like(pose_cos, dtype=torch.long),
                                        torch.zeros_like(pose_cos, dtype=torch.long)))
        # NaN(退化向量) 归为 0=未判定, 不参与统计。
        stage = torch.where(torch.isfinite(pose_cos), stage, torch.zeros_like(stage))
        self._pose_stage = stage
        log["Pose/stage_F"] = stage[:, 0].float().mean().item()
        log["Pose/stage_H"] = stage[:, 1].float().mean().item()
        # 9 格压缩成一个可上报的类别: 0=两段倒置 1=后段翻正中 2=S1(前倒后正)
        # 3=前段翻正中 4=两段正置(S2姿态) 5=前正后倒(错误顺序) 6=其他未知
        both_inv = (stage[:, 0] == -1) & (stage[:, 1] == -1)
        s1_pose = (stage[:, 0] == -1) & (stage[:, 1] == 1)
        s2_pose = (stage[:, 0] == 1) & (stage[:, 1] == 1)
        wrong = (stage[:, 0] == 1) & (stage[:, 1] == -1)
        cls = torch.full_like(stage[:, 0], 6)
        cls = torch.where(s2_pose, torch.full_like(cls, 4), cls)
        cls = torch.where(s1_pose, torch.full_like(cls, 2), cls)
        cls = torch.where(wrong, torch.full_like(cls, 5), cls)
        cls = torch.where(both_inv, torch.full_like(cls, 0), cls)
        # 后段翻正中: 后段已离开倒置但前段仍倒置, 且不是 S1。
        mid_h = (stage[:, 0] == -1) & (stage[:, 1] == 0)
        mid_f = (stage[:, 0] == 0) & (stage[:, 1] == 1)
        cls = torch.where(mid_f, torch.full_like(cls, 3), cls)
        cls = torch.where(mid_h, torch.full_like(cls, 1), cls)
        log["Pose/class"] = cls.float().mean().item()
        # ① P1 内提前到达 S2 姿态: 进入"两段正置"格且 S1 尚未确认。
        running = self._env.episode_length_buf > 0
        early = s2_pose & (self.phase == 0) & ~self._s1_awarded & running
        rising = early & ~self._early_s2_entered
        self._early_s2_count += rising.float()
        self._early_s2_entered |= early
        step = self._update_dt if (self._update_dt > 0.0 and isfinite(self._update_dt)) else 0.0
        self._early_s2_elapsed += early.float() * step
        log["Pose/early_s2_count"] = self._early_s2_count.mean().item()
        log["Pose/early_s2_elapsed"] = self._early_s2_elapsed.mean().item()
        # ② S1 确认后丢失 (倒置, 正置) 格: 说明已到达 S1 姿态又离开。
        lost = (self.phase >= 1) & ~s1_pose & running
        rising_lost = lost & ~self._s1_hold_lost
        self._s1_hold_lost_count += rising_lost.float()
        self._s1_hold_lost |= lost
        log["Pose/s1_hold_lost_count"] = self._s1_hold_lost_count.mean().item()
        # 循环统计。计数在**完成时**累计(而不是在复位后), 并在完整回合重置之前上报 ——
        # 框架顺序是 奖励 → 完整回合重置 → command.compute → _update_metrics,
        # 所以超时回合若在 reset() 里清零, 这里读到的就是 0(上一版的错)。
        # 生命周期: 完成时 +1; 本步日志上报当前值; 读完立刻清"本步完成事件";
        # 回合重置的掩码在此快照, 归零推迟到 _update_command 里做。
        completed = self._last_cycle_reset
        log["Cycle/completed"] = completed.float().mean().item()
        self._cycles_this_episode += completed.long()
        log["Cycle/per_episode"] = self._cycles_this_episode.float().mean().item()
        # 本步完成事件已被消费者读走 -> 清掉, 保证"每个循环恰好计一次"。
        self._last_cycle_reset = torch.zeros_like(completed)
        # 快照本步的回合重置掩码 (单测 mock 可能没有 reset_buf)。
        reset_buf = getattr(self._env, "reset_buf", None)
        self._pending_episode_reset = (reset_buf.clone() if reset_buf is not None
                                       else torch.zeros_like(completed))
        # 相位占比: p3 与 Progress/enter_p3 同义, Phase/retry 在相位单向推进后长期为 0,
        # 两者已删。p2 保留 (它是唯一能看出"卡在 P2"的量)。
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
    # 晚侧余量（实际秒，加在名义段末之后），同时是重试截止。
    # 可接受区间 = [段末, 段末 + window_late_s]；早侧地板已删除(被"参考播完门"吞噬)。
    window_late_s: float = WINDOW_LATE_S
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
