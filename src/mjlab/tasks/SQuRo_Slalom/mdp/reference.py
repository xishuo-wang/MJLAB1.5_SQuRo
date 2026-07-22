"""SQuRo Trot 参考轨迹生成器 — 正弦模式

为第一阶段直行训练提供模仿学习信号。
纯 Python 正弦生成，无 CSV/IK 依赖，课程早期权重高/后期衰减至零。

Trot 步态: 对角腿对同步 (FL+HR=0, FR+HL=0.5)，脊柱直行零位。

控制接口:
  get_reference_joint_state(env) -> (ref_pos, ref_vel)
  返回 [num_envs, 12] 参考关节位置/速度张量
"""

from __future__ import annotations
import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

# Trot 步态参数
TROT_FREQ = 1.5            # 步频 (Hz)
TROT_SHOULDER_AMP = 0.35   # 肩/髋关节摆动幅度 (rad)
TROT_ELBOW_AMP = 0.45      # 肘/膝关节摆动幅度 (rad)
TROT_SHOULDER_BIAS = 0.1   # 肩关节偏置（前腿）
TROT_HIP_BIAS = -0.1       # 髋关节偏置（后腿）
TROT_ELBOW_BIAS = -0.3     # 肘关节偏置（前腿）
TROT_KNEE_BIAS = 0.3       # 膝关节偏置（后腿）

# 12 个执行器的相位偏移 (Trot: 对角对同步)
# 顺序: FL_sh, FL_el, FR_sh, FR_el, HL_hip, HL_knee, HR_hip, HR_knee,
#        F_spine1, F_body, H_spine1, H_body
_PHASE_OFFSETS = torch.tensor([
    0.0, 0.0,   # FL — 相位 0
    0.5, 0.5,   # FR — 相位 0.5 (对角交替)
    0.5, 0.5,   # HL — 相位 0.5 (与 FL 对角同步)
    0.0, 0.0,   # HR — 相位 0   (与 FR 对角同步)
    0.0, 0.0, 0.0, 0.0,  # 脊柱 — 直行零位
])

# 正弦幅值
_AMPLITUDES = torch.tensor([
    TROT_SHOULDER_AMP, TROT_ELBOW_AMP,   # FL
    TROT_SHOULDER_AMP, TROT_ELBOW_AMP,   # FR
    TROT_SHOULDER_AMP, TROT_ELBOW_AMP,   # HL
    TROT_SHOULDER_AMP, TROT_ELBOW_AMP,   # HR
    0.0, 0.0, 0.0, 0.0,                  # 脊柱 — 直行零位
])

# 偏置
_BIASES = torch.tensor([
    TROT_SHOULDER_BIAS, TROT_ELBOW_BIAS,   # FL
    TROT_SHOULDER_BIAS, TROT_ELBOW_BIAS,   # FR
    TROT_HIP_BIAS, TROT_KNEE_BIAS,          # HL
    TROT_HIP_BIAS, TROT_KNEE_BIAS,          # HR
    0.0, 0.0, 0.0, 0.0,                     # 脊柱 — 直行零位
])

_tensors_moved = False


def _move_tensors(device: torch.device | str) -> None:
    """将静态张量迁移到目标设备（仅一次）"""
    global _PHASE_OFFSETS, _AMPLITUDES, _BIASES, _tensors_moved
    if _tensors_moved and _PHASE_OFFSETS.device == device:
        return
    _PHASE_OFFSETS = _PHASE_OFFSETS.to(device)
    _AMPLITUDES = _AMPLITUDES.to(device)
    _BIASES = _BIASES.to(device)
    _tensors_moved = True


def _ensure_ref_state(env: ManagerBasedRlEnv) -> None:
    """初始化每环境相位状态"""
    if getattr(env, "_ref_phase", None) is None:
        env._ref_phase = torch.zeros(env.num_envs, device=env.device)


def get_reference_joint_state(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """返回当前步的参考关节位置和速度（步级缓存：同一步多次调用复用）

    Returns:
        ref_pos: [num_envs, 12] 参考关节位置
        ref_vel: [num_envs, 12] 参考关节速度
    """
    # 步级缓存
    current_step = env.common_step_counter
    cached = getattr(env, "_ref_state_cache", None)
    if cached is not None and cached[0] == current_step:
        return cached[1], cached[2]

    _ensure_ref_state(env)
    _move_tensors(env.device)

    dt = float(env.step_dt)
    num_envs = env.num_envs
    device = env.device

    # 当前相位 [num_envs]
    phase = env._ref_phase

    # 每个关节的相位 (含偏移) [N, 12]
    joint_phase = 2.0 * math.pi * (
        phase.unsqueeze(-1) + _PHASE_OFFSETS.unsqueeze(0)
    )

    # 参考位置: bias + amp * sin(phase)
    ref_pos = _BIASES.unsqueeze(0) + _AMPLITUDES.unsqueeze(0) * torch.sin(joint_phase)

    # 参考速度: amp * omega * cos(phase)
    omega = 2.0 * math.pi * TROT_FREQ
    ref_vel = _AMPLITUDES.unsqueeze(0) * omega * torch.cos(joint_phase)

    # 推进相位
    env._ref_phase = (phase + TROT_FREQ * dt) % 1.0

    # 重置已终止环境
    reset_ids = getattr(env, "reset_terminated", None)
    if reset_ids is not None:
        ids = reset_ids.nonzero(as_tuple=False).flatten()
        if len(ids) > 0:
            env._ref_phase[ids] = 0.0

    env._ref_state_cache = (current_step, ref_pos, ref_vel)
    return ref_pos, ref_vel
