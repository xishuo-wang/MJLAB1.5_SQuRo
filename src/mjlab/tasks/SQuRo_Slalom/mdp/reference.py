"""SQuRo Trot 参考轨迹生成器 — CSV足端轨迹 + IK → 关节角度

只支持 trot 步态，Phase 由固定步频驱动（无命令依赖）。
首次调用 get_reference_joint_state() 时自动:
  1. 加载 Trot_F.csv / Trot_H.csv
  2. IK 求解 → 预计算相位查找表
  3. 解析 _MODEL_INDICES（body/site/joint ID）
"""

from __future__ import annotations
import math
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING

from .indices import resolve_model_indices

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

# Trot 步态参数
TROT_FREQ = 1.5             # 步频 (Hz)

# 腿节长度 (m)
L1_FRONT, L2_FRONT = 0.040, 0.040
L1_HIND, L2_HIND = 0.040, 0.036

# 相位偏移（Trot: 对角同步）
PHASE_LAG = {"FL": 0.0, "FR": 0.5, "HL": 0.5, "HR": 0.0}

# 预计算表分辨率
_TABLE_RESOLUTION = 50

# Bio_Data 目录
_BIO_DATA_DIR = Path(__file__).parent / "Bio_Data"

# 12 关节顺序: FL_sh, FL_el, FR_sh, FR_el, HL_hip, HL_knee, HR_hip, HR_knee, spn...
_STAND_ANGLES = [0.1, -0.3, 0.1, -0.3, -0.1, 0.3, -0.1, 0.3, 0, 0, 0, 0]

# 全局状态
_tables_initialized = False
_pos_table: torch.Tensor | None = None     # [50, 12]
_vel_table: torch.Tensor | None = None     # [50, 12]
_table_device: str | None = None


# =========================================================================================
# 逆运动学（来自 mouse_spg，参数已校准至 SQuRo 模型）
def _inverse_kinematics(x: torch.Tensor, y: torch.Tensor, is_front: bool = True):
    """足端坐标 (x, y) → (proximal_angle, distal_angle)"""
    L1, L2 = (L1_FRONT, L2_FRONT) if is_front else (L1_HIND, L2_HIND)

    R = torch.sqrt(x**2 + y**2)
    K = (L2**2 - x**2 - y**2 - L1**2) / (2 * L1)
    theta = torch.atan2(y, x)
    K_over_R = torch.clamp(K / R, -1.0, 1.0)
    phi = torch.acos(K_over_R)

    if is_front:
        a1 = theta + phi
        sin_a2 = (y + L1 * torch.sin(a1)) / L2
        cos_a2 = (-x - L1 * torch.cos(a1)) / L2
        a2 = torch.atan2(sin_a2, cos_a2)
        shoulder = a1 - 0.868
        elbow = -(a2 + 2.643)
        elbow = torch.where(a2 > 2, elbow - 2 * math.pi, elbow)
        return shoulder, elbow
    else:
        a1 = theta - phi
        sin_a2 = (x + L1 * torch.cos(a1)) / L2
        cos_a2 = (y + L1 * torch.sin(a1)) / L2
        a2 = torch.atan2(sin_a2, cos_a2)
        hip = a1 + 4.35
        knee = -(a2 + 1.75)
        knee = torch.where(a2 > 2, knee - 2 * math.pi, knee)
        return hip, knee


# =========================================================================================
# 加载 CSV + IK 预计算
def _init_tables(device: torch.device | str) -> None:
    global _tables_initialized, _pos_table, _vel_table, _table_device

    if _tables_initialized and _table_device == str(device):
        return

    # 加载 CSV
    df_f = pd.read_csv(_BIO_DATA_DIR / "Trot_F.csv")
    df_h = pd.read_csv(_BIO_DATA_DIR / "Trot_H.csv")
    phase_csv = df_f["Phase"].values.astype(np.float64)
    y_f = df_f["Y_mean"].values.astype(np.float64)
    z_f = df_f["Z_mean"].values.astype(np.float64)
    y_h = df_h["Y_mean"].values.astype(np.float64)
    z_h = df_h["Z_mean"].values.astype(np.float64)

    # 生成等距相位网格
    phases = np.linspace(0, 1, _TABLE_RESOLUTION)

    # 在相位网格上插值足端轨迹
    y_f_grid = np.interp(phases, phase_csv, y_f)
    z_f_grid = np.interp(phases, phase_csv, z_f)
    y_h_grid = np.interp(phases, phase_csv, y_h)
    z_h_grid = np.interp(phases, phase_csv, z_h)

    # 转为 torch
    dev = torch.device(device)
    y_f_t = torch.tensor(y_f_grid, device=dev, dtype=torch.float32)
    z_f_t = torch.tensor(z_f_grid, device=dev, dtype=torch.float32)
    y_h_t = torch.tensor(y_h_grid, device=dev, dtype=torch.float32)
    z_h_t = torch.tensor(z_h_grid, device=dev, dtype=torch.float32)

    # IK 求解：对每个相位 bin 计算 8 个腿关节角度
    pos = torch.zeros(_TABLE_RESOLUTION, 12, device=dev)
    for leg, (y_t, z_t, lag, is_front) in enumerate([
        (y_f_t, z_f_t, PHASE_LAG["FL"], True),   # FL
        (y_f_t, z_f_t, PHASE_LAG["FR"], True),   # FR (同CSV，相位差)
        (y_h_t, z_h_t, PHASE_LAG["HL"], False),  # HL
        (y_h_t, z_h_t, PHASE_LAG["HR"], False),  # HR
    ]):
        # 对网格应用相位偏移（循环移位）
        shift = int(lag * _TABLE_RESOLUTION)
        y_shifted = torch.roll(y_t, shifts=shift)
        z_shifted = torch.roll(z_t, shifts=shift)
        proximal, distal = _inverse_kinematics(y_shifted, z_shifted, is_front)
        pos[:, leg * 2] = proximal
        pos[:, leg * 2 + 1] = distal

    # 脊柱保持零位
    pos[:, 8:] = torch.tensor(_STAND_ANGLES[8:], device=dev)

    # 中心差分计算速度
    vel = torch.zeros_like(pos)
    two_dt = 2.0 / _TABLE_RESOLUTION
    vel[1:-1] = (pos[2:] - pos[:-2]) / two_dt
    vel[0] = (pos[1] - pos[-1]) / two_dt
    vel[-1] = (pos[0] - pos[-2]) / two_dt

    _pos_table = pos.contiguous()
    _vel_table = vel.contiguous()
    _table_device = str(device)
    _tables_initialized = True

    print(f"\n[SQuRo Trot] 参考轨迹表生成完成: {_TABLE_RESOLUTION} bins × 12 joints")


# =========================================================================================
# 公共接口
def get_reference_joint_state(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """返回当前步的参考关节位置和速度

    首次调用自动触发: CSV加载 → IK预计算 → 模型索引解析
    步级缓存: 同一步内多次调用复用结果

    Returns:
        ref_pos: [num_envs, 12] 参考关节位置
        ref_vel: [num_envs, 12] 参考关节速度
    """
    # 步级缓存
    current_step = env.common_step_counter
    cached = getattr(env, "_ref_state_cache", None)
    if cached is not None and cached[0] == current_step:
        return cached[1], cached[2]

    # 首次: 初始化相位状态 + 解析模型索引
    if getattr(env, "_ref_phase", None) is None:
        env._ref_phase = torch.zeros(env.num_envs, device=env.device)
        resolve_model_indices(env.scene["robot"])

    # 首次: 预计算查找表
    if not _tables_initialized:
        _init_tables(env.device)

    # 确保表在正确设备上
    if _table_device != str(env.device):
        _init_tables(env.device)

    dt = float(env.step_dt)
    num_envs = env.num_envs

    # 当前相位 → 查表索引
    phase = env._ref_phase  # [num_envs]
    phase_indices = (phase * (_TABLE_RESOLUTION - 1)).long().clamp_(0, _TABLE_RESOLUTION - 1)

    ref_pos = _pos_table[phase_indices]  # [N, 12]
    ref_vel = _vel_table[phase_indices] * TROT_FREQ  # [N, 12]

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
