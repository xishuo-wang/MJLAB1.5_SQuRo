from __future__ import annotations
import math
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING
from .command import CURVATURE_TARGET_MAX
from .indices import resolve_model_indices
from .path import get_path_curvature, _RMIN as _SLALOM_RMIN
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv



# =========================================================================================
# 步态配置
_BIO_DATA_DIR = Path(__file__).parent / "Bio_Data"
PHASE_LAG = {"FL": 0.0, "FR": 0.5, "HL": 0.5, "HR": 0.0}    # 步态相位差
STRIDE_MIN = 0.0                                            # 最小步幅


# 离散曲率绝对值表（运行时在此范围内线性插值）
# 由 CURVATURE_TARGET_MAX 动态生成：[0.0, 0.5, ..., 5.0] + (5, max] 步长 1.0
def _generate_curvature_bins(max_k: float) -> list[float]:
    bins = [i * 0.5 for i in range(11)]  # [0.0, 0.5, ..., 5.0]
    if max_k > 5.0:
        bins.extend(float(x) for x in range(6, int(max_k) + 1))
    return bins

_CURVATURE_BINS = _generate_curvature_bins(CURVATURE_TARGET_MAX)
_NUM_CURV = len(_CURVATURE_BINS)


# 预计算表分辨率
_TABLE_RESOLUTION = 50                      # 预计算表分辨率
_tables_initialized = False
_table_device: str | None = None
_k_bins: torch.Tensor | None = None        # [_NUM_CURV] 曲率查找表(缓存)
_pos_table: torch.Tensor | None = None     # [_NUM_CURV, 50, 12]
_vel_table: torch.Tensor | None = None     # [_NUM_CURV, 50, 12]



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
# 加载 CSV + IK 预计算（腿部随曲率差速，脊柱全零直行参考）
def _init_tables(device: torch.device | str) -> None:
    global _tables_initialized, _pos_table, _vel_table, _k_bins, _table_device

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

    # 预分配三维表 [曲率, 相位, 关节]
    pos = torch.zeros(_NUM_CURV, _TABLE_RESOLUTION, 14, device=dev)

    # 对每个离散曲率生成“左转参考表”（左腿为内侧，右腿为外侧）
    for i, abs_k in enumerate(_CURVATURE_BINS):
        scale_inner = 1.0 - (1.0 - STRIDE_MIN) * abs_k / CURVATURE_TARGET_MAX   # 内侧腿侧向缩放因子（κ=0 → scale=1 全步幅, |κ|=κ_max → scale=0）

        # 左腿（FL, HL）为内侧，缩放其 Y_mean
        y_fL = y_f_t * scale_inner
        z_fL = z_f_t                      # 高度不变
        y_hL = y_h_t * scale_inner
        z_hL = z_h_t

        # 右腿（FR, HR）为外侧，保持原轨迹
        y_fR = y_f_t
        z_fR = z_f_t
        y_hR = y_h_t
        z_hR = z_h_t

        # 分别计算每条腿的关节角（应用相位差）
        # FL
        shift = int(PHASE_LAG["FL"] * _TABLE_RESOLUTION)
        y_shifted = torch.roll(y_fL, shifts=shift)
        z_shifted = torch.roll(z_fL, shifts=shift)
        sh, el = _inverse_kinematics(y_shifted, z_shifted, True)
        pos[i, :, 4] = sh
        pos[i, :, 5] = el

        # FR
        shift = int(PHASE_LAG["FR"] * _TABLE_RESOLUTION)
        y_shifted = torch.roll(y_fR, shifts=shift)
        z_shifted = torch.roll(z_fR, shifts=shift)
        sh, el = _inverse_kinematics(y_shifted, z_shifted, True)
        pos[i, :, 6] = sh
        pos[i, :, 7] = el

        # HL
        shift = int(PHASE_LAG["HL"] * _TABLE_RESOLUTION)
        y_shifted = torch.roll(y_hL, shifts=shift)
        z_shifted = torch.roll(z_hL, shifts=shift)
        hp, kn = _inverse_kinematics(y_shifted, z_shifted, False)
        pos[i, :, 10] = hp
        pos[i, :, 11] = kn

        # HR
        shift = int(PHASE_LAG["HR"] * _TABLE_RESOLUTION)
        y_shifted = torch.roll(y_hR, shifts=shift)
        z_shifted = torch.roll(z_hR, shifts=shift)
        hp, kn = _inverse_kinematics(y_shifted, z_shifted, False)
        pos[i, :, 12] = hp
        pos[i, :, 13] = kn

        # 脊柱四列保持零位（直行参考姿态）
        pos[i, :, 0] = 0.0   # F_spine1
        pos[i, :, 1] = 0.0   # F_body
        pos[i, :, 8] = 0.0   # H_spine1
        pos[i, :, 9] = 0.0   # H_body

    # 中心差分计算速度表（沿相位维度）
    vel = torch.zeros_like(pos)
    two_dt = 2.0 / _TABLE_RESOLUTION
    vel[:, 1:-1] = (pos[:, 2:] - pos[:, :-2]) / two_dt
    vel[:, 0] = (pos[:, 1] - pos[:, -1]) / two_dt
    vel[:, -1] = (pos[:, 0] - pos[:, -2]) / two_dt

    _pos_table = pos.contiguous()
    _vel_table = vel.contiguous()
    _k_bins = torch.tensor(_CURVATURE_BINS, device=dev, dtype=torch.float32)
    _table_device = str(device)
    _tables_initialized = True

    print(f"\n[SQuRo Trot] 参考轨迹表生成完成: {_NUM_CURV} 曲率 × {_TABLE_RESOLUTION} bins × 12 joints")



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
    phase: torch.Tensor = env._ref_phase  # type: ignore[attr-defined]  # [N]
    phase_indices = (phase * (_TABLE_RESOLUTION - 1)).long().clamp_(0, _TABLE_RESOLUTION - 1)

    # 获取路径瞬时曲率 + 步频 (Phase 0: 静态命令值, Phase 1: LUT 插值)
    curvature_cmd = get_path_curvature(env)  # [N]
    cmd_tensor = env.command_manager._terms["slalom_cmd"].command  # type: ignore[union-attr]
    vel_cmd = cmd_tensor[:, 0]  # [N]
    gait_freq = cmd_tensor[:, 3]  # [N] — 动态步频
    slalom_mode = env.command_manager._terms["slalom_cmd"].slalom_mode_active  # type: ignore[union-attr]
    kappa_norm = 1.0 / _SLALOM_RMIN if slalom_mode else CURVATURE_TARGET_MAX  # 15 or 20

    # 曲率绝对值及插值因子（k_bins 在 _init_tables 时缓存）
    abs_k = curvature_cmd.abs()  # [N]
    assert _k_bins is not None

    # 找到每个环境的曲率区间索引（左侧）
    idx = torch.searchsorted(_k_bins, abs_k) - 1
    idx = idx.clamp(0, _NUM_CURV - 2)  # 防止边界外，使 idx+1 有效

    # 对应的曲率值
    k0 = _k_bins[idx]        # [N]
    k1 = _k_bins[idx + 1]
    t = (abs_k - k0) / (k1 - k0 + 1e-12)  # 插值因子，[0,1]

    # 从表中取出对应相位的关节参考，两个曲率层
    # pos_table: [_NUM_CURV, 50, 12], phase_indices: [N]
    # 使用高级索引: pos_table[idx, phase_indices] -> [N,12]
    pos0 = _pos_table[idx, phase_indices]      # [N,12]
    pos1 = _pos_table[idx + 1, phase_indices]  # [N,12]
    vel0 = _vel_table[idx, phase_indices]      # [N,12]
    vel1 = _vel_table[idx + 1, phase_indices]

    # 线性插值腿部参考（脊柱部分后续覆盖，但插值也参与）
    ref_pos = (1 - t.unsqueeze(1)) * pos0 + t.unsqueeze(1) * pos1
    ref_vel = ((1 - t.unsqueeze(1)) * vel0 + t.unsqueeze(1) * vel1) * gait_freq.unsqueeze(1)

    # -----------------------------------------------------------------
    # 根据曲率符号交换左右腿关节（右转时内侧为右腿）
    swap_mask = curvature_cmd < 0  # [N] bool
    if swap_mask.any():
        # 保存原值
        FL = ref_pos[:, [4, 5]].clone()
        FR = ref_pos[:, [6, 7]].clone()
        HL = ref_pos[:, [10, 11]].clone()
        HR = ref_pos[:, [12, 13]].clone()
        FL_v = ref_vel[:, [4, 5]].clone()
        FR_v = ref_vel[:, [6, 7]].clone()
        HL_v = ref_vel[:, [10, 11]].clone()
        HR_v = ref_vel[:, [12, 13]].clone()

        ref_pos[swap_mask, 4:6] = FR[swap_mask]
        ref_pos[swap_mask, 6:8] = FL[swap_mask]
        ref_pos[swap_mask, 10:12] = HR[swap_mask]
        ref_pos[swap_mask, 12:14] = HL[swap_mask]

        ref_vel[swap_mask, 4:6] = FR_v[swap_mask]
        ref_vel[swap_mask, 6:8] = FL_v[swap_mask]
        ref_vel[swap_mask, 10:12] = HR_v[swap_mask]
        ref_vel[swap_mask, 12:14] = HL_v[swap_mask]

    # 动态覆盖脊柱侧摆参考（四关节线性映射，依据 ω_cmd）
    k_norm = curvature_cmd / kappa_norm          # 归一化曲率，范围 [-1, 1]
    abs_k_norm = curvature_cmd.abs() / kappa_norm

    # 各脊柱关节目标角度（弧度），在 |κ| = max_k 时达到极值
    ref_pos[:, 0] = -0.6 * k_norm           # f_spine1 κ=-max → +0.6, κ=+max → -0.6
    ref_pos[:, 1] = -0.9 * k_norm           # f_body κ=-max → +0.9, κ=+max → -0.9
    ref_pos[:, 2] = 0.8 * k_norm            # neck_yaw
    ref_pos[:, 3] = -0.3 * abs_k_norm       # neck_pitch
    ref_pos[:, 8] = -0.6 * abs_k_norm       # h_spine1 始终 ≤0, |κ|=max → -0.6
    ref_pos[:, 9] = -0.7 * k_norm           # h_body κ=-max → +0.7, κ=+max → -0.7

    # 推进相位
    env._ref_phase = (phase + gait_freq * dt) % 1.0  # type: ignore[attr-defined]

    # 重置已终止环境
    reset_ids = getattr(env, "reset_terminated", None)
    if reset_ids is not None:
        ids = reset_ids.nonzero(as_tuple=False).flatten()
        if len(ids) > 0:
            env._ref_phase[ids] = 0.0  # type: ignore[attr-defined]

    env._ref_state_cache = (current_step, ref_pos, ref_vel)  # type: ignore[attr-defined]
    return ref_pos, ref_vel
