from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Tuple
from dataclasses import dataclass, field
from mjlab.managers import CommandTermCfg
from mjlab.managers.command_manager import CommandTerm
from .curriculums import get_curriculum_time_scale
from .indices import _MODEL_INDICES, resolve_model_indices
from .timing import P1_END, P2_DURATION

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


# =========================================================================================
# 命令系统 + 阶段状态机 — 7D 命令 [vel_x, height_f, height_h, gait_freq, curvature, time_scale, phase]
# 前 5 维与 Slalom/Tunnel 对齐; 第 6 维 time_scale λ (参考时间缩放);
# 第 7 维 phase 编码: 0=P1(段1+2), 1=P2(段3), 2=P3(站立)
# 阶段状态机: 每步推进 t_phase, 检测 S1/S2, 达标进入下一阶段;
#   超时+缓冲未达标则重试 (t_phase 归零, 不重置机器人)
# 关节参考/期望高度按"阶段时间"查询 (reference.py 读 stage_t)
# =========================================================================================

# 阶段状态检测阈值
_GROUND_TH = 0.03     # S1 平躺高度阈值
_GROUND_TH_S2 = 0.04  # S2 趴地高度阈值 (段3末 H 后肢略翘≈0.034)
# 阶段预期时长 (名义, ×λ)
_P1_EXPECT = P1_END
_P2_EXPECT = P2_DURATION
_BUFFER = 0.3         # 缓冲时间 (s): 超过预期时长后缓冲期内持续检测, 未达标才重试
_MAX_RETRY = 5


class BackupCommand(CommandTerm):
    cfg: "BackupCommandCfg"

    def __init__(self, cfg: "BackupCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.fixed_time_scale = cfg.fixed_time_scale
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
        """仅在 P1→P2 后的一个奖励步内为 True。"""
        return self._last_advance1

    @property
    def s2_transition_pulse(self) -> torch.Tensor:
        """仅在 P2→P3 后的一个奖励步内为 True。"""
        return self._last_advance2

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

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
            self._resample_command(env_ids)
        return extras

    def compute(self, dt: float) -> None:
        # CommandTerm._update_command() 不接收 dt，因此在调用父类前暂存本次真实步长。
        self._update_dt = dt
        super().compute(dt)

    # 身体段"正置"判定 — 用腹/背标记 site 的世界坐标, 不依赖四元数约定
    # 踩坑: data.body_link_quat_w 名为 world, 但实测 reset 后 root_link_quat_w 为单位四元数
    #   (base_Link 的 XML 安装旋转约为绕 XY 对角线 180°), 说明该量并非世界系表达;
    #   其参考系至今未定论 (见 docs/SQuRo_Backup_技术细节.md §3)。
    #   旧写法 sign*2(yz+wx) 在翻正过程中会失效, 故改用标记 site:
    #     belly_z < back_z  ⇔  腹面朝下  ⇔  该段已翻正
    #   F/H 两段的局部坐标相反 (F 腹面在局部 +Y, H 腹面在局部 -Y), 但用世界坐标比较可自动消除该差异。
    def _segment_upright(self, idx: int) -> torch.Tensor:
        pairs = _MODEL_INDICES.segment_belly_back_ids
        assert pairs is not None, "segment_belly_back_ids 未解析, 请先调用 resolve_model_indices"
        belly_id, back_id = pairs[idx]
        sp = self._asset.data.site_pos_w
        return sp[:, belly_id, 2] < sp[:, back_id, 2]

    def _body_height(self, body_id: int) -> torch.Tensor:
        return self._asset.data.body_link_pos_w[:, body_id, 2]

    def _check_S1(self) -> torch.Tensor:
        # S1: 后段已翻正、前段未翻正, 且两段躯干都平躺贴地
        f_up = self._segment_upright(0)
        h_up = self._segment_upright(1)
        fz = self._body_height(_MODEL_INDICES.f_body_id)
        hz = self._body_height(_MODEL_INDICES.h_body_id)
        return (~f_up) & h_up & (fz < _GROUND_TH) & (hz < _GROUND_TH)

    def _check_S2(self) -> torch.Tensor:
        # S2: 两段躯干都已翻正并重新贴地 — 完整翻转完成
        f_up = self._segment_upright(0)
        h_up = self._segment_upright(1)
        fz = self._body_height(_MODEL_INDICES.f_body_id)
        hz = self._body_height(_MODEL_INDICES.h_body_id)
        return f_up & h_up & (fz < _GROUND_TH_S2) & (hz < _GROUND_TH_S2)

    def _update_command(self) -> None:
        dt = self._update_dt
        lam = self.time_scale_command.clamp(min=0.1)
        # 所有阶段推进 t_phase (P1/P2 用于段内参考, P3 用于 time5/站立)
        # reset() 末尾也会调用 command_manager.compute(dt=0)。重置环境的参考时钟应停在 0，
        # 只有真实环境步开始后才推进，避免初始参考提前一个控制步。
        running = self._env.episode_length_buf > 0
        self.t_phase = self.t_phase + running.to(self.t_phase.dtype) * dt
        # P1: 检测 S1
        p1 = self.phase == 0
        expected1 = _P1_EXPECT * lam
        s1_ok = self._check_S1()
        advance1 = p1 & (self.t_phase >= expected1) & s1_ok
        retry1 = p1 & (self.t_phase >= expected1 + _BUFFER) & ~s1_ok
        # P2: 检测 S2
        p2 = self.phase == 1
        expected2 = _P2_EXPECT * lam
        s2_ok = self._check_S2()
        advance2 = p2 & (self.t_phase >= expected2) & s2_ok
        retry2 = p2 & (self.t_phase >= expected2 + _BUFFER) & ~s2_ok
        # 达标推进
        advance = advance1 | advance2
        self.phase = torch.where(advance, self.phase + 1, self.phase)
        self.t_phase = torch.where(advance, torch.zeros_like(self.t_phase), self.t_phase)
        # 重试: t_phase 归零 (不重置机器人)
        retry_mask = retry1 | retry2
        self.t_phase = torch.where(retry_mask, torch.zeros_like(self.t_phase), self.t_phase)
        self.retry = torch.where(retry_mask, self.retry + 1, self.retry)
        self.phase_command[:] = self.phase.float()
        # 保存本步检测结果供 _update_metrics 记录 (metrics 在 command 前被调用, 记录上一步状态)
        self._last_s1_ok = s1_ok
        self._last_s2_ok = s2_ok
        self._last_advance1 = advance1
        self._last_advance2 = advance2
        self._last_retry_mask = retry_mask

    def _update_metrics(self) -> None:
        # 状态机阶段/重试指标 -> wandb 日志 (env.step 中已初始化 extras['log'])
        log = self._env.extras["log"]
        log["Data/backup_phase"] = self.phase.float().mean().item()
        log["Data/backup_t_phase"] = self.t_phase.mean().item()
        log["Data/backup_retry_cum"] = self.retry.float().mean().item()
        log["Data/backup_retry_rate"] = self._last_retry_mask.float().mean().item()
        log["Data/backup_s1_ok"] = self._last_s1_ok.float().mean().item()
        log["Data/backup_s2_ok"] = self._last_s2_ok.float().mean().item()
        log["Data/backup_s1_transition"] = self._last_advance1.float().mean().item()
        log["Data/backup_s2_transition"] = self._last_advance2.float().mean().item()

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        pass


@dataclass(kw_only=True)
class BackupCommandCfg(CommandTermCfg):
    asset_name: str = "robot"
    resampling_time_range: Tuple[float, float] = (1000.0, 1000.0)   # 不重采样 (episode 内固定)
    debug_vis: bool = False
    fixed_time_scale: float | None = None
    """固定参考时间缩放 (回放/demo 用, 如 1.4); None 时按课程采样"""

    @dataclass
    class VizCfg:
        z_offset: float = 0.1
        scale: float = 1.0

    viz: VizCfg = field(default_factory=VizCfg)
    class_type: type[CommandTerm] = BackupCommand

    def build(self, env: "ManagerBasedRlEnv") -> CommandTerm:
        return self.class_type(self, env)
