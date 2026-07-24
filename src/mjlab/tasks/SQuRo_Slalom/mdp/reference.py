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


# =========================================================================================
# 步态配置
_BIO_DATA_DIR = Path(__file__).parent / "Bio_Data"
PHASE_LAG = {"FL": 0.0, "FR": 0.5, "HL": 0.5, "HR": 0.0}    # 步态相位差
TROT_FREQ = 1.0                                             # 步频 (Hz)


# 预计算表分辨率
_TABLE_RESOLUTION = 50                      # 预计算表分辨率
_tables_initialized = False
_pos_table: torch.Tensor | None = None     # [50, 12]
_vel_table: torch.Tensor | None = None     # [50, 12]
_table_device: str | None = None



# =========================================================================================
# 逆运动学
def _inverse_kinematics(x: torch.Tensor, y: torch.Tensor, is_front: bool = True):
    L1, L2 = (0.040, 0.040) if is_front else (0.040, 0.036)

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
# 加载 CSV + IK 预计算（仅腿部8关节；脊柱4关节运行时动态覆盖）
def _init_tables(device: torch.device | str) -> None:
    global _tables_initialized, _pos_table, _vel_table, _table_device

    if _tables_initialized and _table_device == str(device):
        return

    # 加载 CSV
    df_f = pd.read_csv(_BIO_DATA_DIR / "Trot_F.csv")  # type: ignore[arg-type]
    df_h = pd.read_csv(_BIO_DATA_DIR / "Trot_H.csv")  # type: ignore[arg-type]
    phase_csv: np.ndarray = df_f["Phase"].values.astype(np.float64)  # type: ignore[union-attr]
    y_f: np.ndarray = df_f["Y_mean"].values.astype(np.float64)  # type: ignore[union-attr]
    z_f: np.ndarray = df_f["Z_mean"].values.astype(np.float64)  # type: ignore[union-attr]
    y_h: np.ndarray = df_h["Y_mean"].values.astype(np.float64)  # type: ignore[union-attr]
    z_h: np.ndarray = df_h["Z_mean"].values.astype(np.float64)  # type: ignore[union-attr]

    # 生成等距相位网格
    phases = np.linspace(0, 1, _TABLE_RESOLUTION)

    # 在相位网格上插值足端轨迹
    y_f_grid = np.interp(phases, phase_csv, y_f)
    z_f_grid = np.interp(phases, phase_csv, z_f)
    y_h_grid = np.interp(phases, phase_csv, y_h)
    z_h_grid = np.interp(phases, phase_csv, z_h)

    # 转为 torch
    dev = torch.device(device)  # type: ignore[arg-type]
    y_f_t = torch.tensor(y_f_grid, device=dev, dtype=torch.float32)
    z_f_t = torch.tensor(z_f_grid, device=dev, dtype=torch.float32)
    y_h_t = torch.tensor(y_h_grid, device=dev, dtype=torch.float32)
    z_h_t = torch.tensor(z_h_grid, device=dev, dtype=torch.float32)

    # IK 求解：按 _ACTUATED_JOINT_NAMES 顺序填充
    # 列序: F_sp1(0) F_bd(1) FL_sh(2) FL_el(3) FR_sh(4) FR_el(5)
    #        H_sp1(6) H_bd(7) HL_hp(8) HL_kn(9) HR_hp(10) HR_kn(11)
    pos = torch.zeros(_TABLE_RESOLUTION, 12, device=dev)
    legs = [
        (y_f_t, z_f_t, PHASE_LAG["FL"], True, 2),    # FL → 列 2-3
        (y_f_t, z_f_t, PHASE_LAG["FR"], True, 4),    # FR → 列 4-5
        (y_h_t, z_h_t, PHASE_LAG["HL"], False, 8),   # HL → 列 8-9
        (y_h_t, z_h_t, PHASE_LAG["HR"], False, 10),  # HR → 列 10-11
    ]
    for y_t, z_t, lag, is_front, col_offset in legs:
        shift = int(lag * _TABLE_RESOLUTION)
        y_shifted = torch.roll(y_t, shifts=shift)
        z_shifted = torch.roll(z_t, shifts=shift)
        proximal, distal = _inverse_kinematics(y_shifted, z_shifted, is_front)
        pos[:, col_offset] = proximal  # type: ignore[call-overload]
        pos[:, col_offset + 1] = distal  # type: ignore[call-overload]

    # 脊柱保持零位（直行参考姿态）
    pos[:, 0] = 0.0   # F_spine1
    pos[:, 1] = 0.0   # F_body
    pos[:, 6] = 0.0   # H_spine1
    pos[:, 7] = 0.0   # H_body

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
# 获取当前步的参考关节位置和速度
def get_reference_joint_state(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    # 步级缓存
    current_step = env.common_step_counter
    cached = getattr(env, "_ref_state_cache", None)
    if cached is not None and cached[0] == current_step:
        return cached[1], cached[2]

    # 首次: 初始化相位状态 + 解析模型索引
    if getattr(env, "_ref_phase", None) is None:
        env._ref_phase = torch.zeros(env.num_envs, device=env.device)  # type: ignore[attr-defined]
        resolve_model_indices(env.scene["robot"])

    # 首次: 预计算查找表
    if not _tables_initialized:
        _init_tables(env.device)

    # 确保表在正确设备上
    if _table_device != str(env.device):
        _init_tables(env.device)

    assert _pos_table is not None and _vel_table is not None

    dt = float(env.step_dt)

    # 当前相位 → 查表索引
    phase: torch.Tensor = env._ref_phase  # type: ignore[attr-defined]  # [num_envs]
    phase_indices = (phase * (_TABLE_RESOLUTION - 1)).long().clamp_(0, _TABLE_RESOLUTION - 1)

    ref_pos = _pos_table[phase_indices].clone()  # [N, 12] — clone 避免修改预计算表
    ref_vel = _vel_table[phase_indices] * TROT_FREQ  # [N, 12]

    # 动态覆盖脊柱侧摆参考: F_body_heading = base_heading - F_spine1
    # 左转(κ>0, ω>0) → 需F_body左偏 → F_spine1<0 → 取负号
    curvature_cmd = env.command_manager._terms["slalom_cmd"].command[:, 4]  # type: ignore[union-attr]
    vel_cmd = env.command_manager._terms["slalom_cmd"].command[:, 0]  # type: ignore[union-attr]
    omega_cmd = curvature_cmd * vel_cmd
    # 各关节增益 (rad·s/rad) —— 使 ω=0.5 时达到目标角度
    K_sp1  =  0.6 / 0.5   # = 1.2
    K_fbd  = -0.8 / 0.5   # = -1.6
    K_hsp1 =  0.6 / 0.5   # = 1.2
    K_hbd  = -0.7 / 0.5   # = -1.4

    # 计算原始角度
    raw_sp1  = K_sp1  * omega_cmd
    raw_fbd  = K_fbd  * omega_cmd
    raw_hsp1 = K_hsp1 * omega_cmd
    raw_hbd  = K_hbd  * omega_cmd

    # 限幅到各自安全范围
    sp1  = torch.clamp(raw_sp1,  -0.6, 0.6)
    fbd  = torch.clamp(raw_fbd,  -0.8, 0.8)
    hsp1 = torch.clamp(raw_hsp1, -0.6, 0.6)
    hbd  = torch.clamp(raw_hbd,  -0.7, 0.7)

    # 覆盖预计算表中的脊柱列 (索引需与你的 _MODEL_INDICES 一致)
    ref_pos[:, 0] = sp1    # F_spine1
    ref_pos[:, 1] = fbd    # F_body
    ref_pos[:, 6] = hsp1   # H_spine1
    ref_pos[:, 7] = hbd    # H_body

    ref_vel[:, [0, 1, 6, 7]] = 0.0
    # 推进相位
    env._ref_phase = (phase + TROT_FREQ * dt) % 1.0  # type: ignore[attr-defined]

    # 重置已终止环境
    reset_ids = getattr(env, "reset_terminated", None)
    if reset_ids is not None:
        ids = reset_ids.nonzero(as_tuple=False).flatten()
        if len(ids) > 0:
            env._ref_phase[ids] = 0.0  # type: ignore[attr-defined]

    env._ref_state_cache = (current_step, ref_pos, ref_vel)  # type: ignore[attr-defined]
    return ref_pos, ref_vel
