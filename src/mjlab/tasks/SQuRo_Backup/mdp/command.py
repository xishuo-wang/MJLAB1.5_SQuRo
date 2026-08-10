from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Tuple
from dataclasses import dataclass, field
from mjlab.managers import CommandTermCfg
from mjlab.managers.command_manager import CommandTerm
from .curriculums import get_curriculum_time_scale
from .indices import _MODEL_INDICES, resolve_model_indices

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
_UP_TH = 0.5          # 背腹轴朝上/朝下判定
_GROUND_TH = 0.03     # S1 平躺高度阈值
_GROUND_TH_S2 = 0.04  # S2 趴地高度阈值 (段3末 H 后肢略翘≈0.034)
# 阶段预期时长 (名义, ×λ)
_P1_EXPECT = 0.8
_P2_EXPECT = 0.15
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

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
            self._resample_command(env_ids)
        return extras

    # 身体背腹轴 (body+Y) 世界 Z 分量: F 直接取, H 取负 (局部坐标相反, 踩坑)
    def _body_up(self, body_id: int, sign: float) -> torch.Tensor:
        q = self._asset.data.body_link_quat_w[:, body_id]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return sign * 2.0 * (y * z + w * x)

    def _check_S1(self) -> torch.Tensor:
        fu = self._body_up(_MODEL_INDICES.f_body_id, +1.0)
        hu = self._body_up(_MODEL_INDICES.h_body_id, -1.0)
        fz = self._asset.data.body_link_pos_w[:, _MODEL_INDICES.f_body_id, 2]
        hz = self._asset.data.body_link_pos_w[:, _MODEL_INDICES.h_body_id, 2]
        return (fu > _UP_TH) & (hu < -_UP_TH) & (fz < _GROUND_TH) & (hz < _GROUND_TH)

    def _check_S2(self) -> torch.Tensor:
        fu = self._body_up(_MODEL_INDICES.f_body_id, +1.0)
        hu = self._body_up(_MODEL_INDICES.h_body_id, -1.0)
        fz = self._asset.data.body_link_pos_w[:, _MODEL_INDICES.f_body_id, 2]
        hz = self._asset.data.body_link_pos_w[:, _MODEL_INDICES.h_body_id, 2]
        return (fu < -_UP_TH) & (hu < -_UP_TH) & (fz < _GROUND_TH_S2) & (hz < _GROUND_TH_S2)

    def _update_command(self) -> None:
        dt = self._env.step_dt
        lam = self.time_scale_command.clamp(min=0.1)
        # 所有阶段推进 t_phase (P1/P2 用于段内参考, P3 用于 time5/站立)
        self.t_phase = self.t_phase + dt
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

    def _update_metrics(self) -> None:
        pass

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
