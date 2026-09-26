from __future__ import annotations
import math
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING
from .path import get_path_curvature
from .indices import resolve_model_indices
from .curriculums import CURVATURE_TARGET_MAX
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv



# =========================================================================================
# 步态配置
_BIO_DATA_DIR = Path(__file__).parent / "Bio_Data"
PHASE_LAG = {"FL": 0.0, "FR": 0.5, "HL": 0.5, "HR": 0.0}    # 步态相位差
STRIDE_MIN = 0.0                                            # 最小步幅

# 参考限位 (rad): 取 SQuRo.xml 中关节 range 与执行器 ctrlrange 的较小者
F_SPINE1_LIMIT = 0.6
H_SPINE1_LIMIT = 0.6
NECK_YAW_LIMIT = 0.8


# 离散曲率绝对值表
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
_pos_table: torch.Tensor | None = None     # [2, _NUM_CURV, 50, 14] (内侧, 曲率, 相位, 关节)
_vel_table: torch.Tensor | None = None     # 同上



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

    # 预分配四维表 [内侧, 曲率, 相位, 关节]: 0=左腿为内侧(左转), 1=右腿为内侧(右转)
    # 两张表各自只缩放"内侧腿"的 Y_mean, 每侧腿部相位不随曲率符号改变
    pos = torch.zeros(2, _NUM_CURV, _TABLE_RESOLUTION, 14, device=dev)

    # 对每个离散曲率生成左内/右内两张参考表
    for i, abs_k in enumerate(_CURVATURE_BINS):
        scale_inner = 1.0 - (1.0 - STRIDE_MIN) * abs_k / CURVATURE_TARGET_MAX

        for side in (0, 1):
            # 内侧腿缩放其 Y_mean (side=0: FL/HL, side=1: FR/HR), 外侧腿保持原轨迹
            scale_fL = scale_inner if side == 0 else 1.0
            scale_fR = 1.0 if side == 0 else scale_inner
            scale_hL = scale_inner if side == 0 else 1.0
            scale_hR = 1.0 if side == 0 else scale_inner

            # 分别计算每条腿的关节角（应用相位差）
            # FL
            shift = int(PHASE_LAG["FL"] * _TABLE_RESOLUTION)
            sh, el = _inverse_kinematics(torch.roll(y_f_t * scale_fL, shifts=shift),
                                         torch.roll(z_f_t, shifts=shift), True)
            pos[side, i, :, 4] = sh
            pos[side, i, :, 5] = el

            # FR
            shift = int(PHASE_LAG["FR"] * _TABLE_RESOLUTION)
            sh, el = _inverse_kinematics(torch.roll(y_f_t * scale_fR, shifts=shift),
                                         torch.roll(z_f_t, shifts=shift), True)
            pos[side, i, :, 6] = sh
            pos[side, i, :, 7] = el

            # HL
            shift = int(PHASE_LAG["HL"] * _TABLE_RESOLUTION)
            hp, kn = _inverse_kinematics(torch.roll(y_h_t * scale_hL, shifts=shift),
                                         torch.roll(z_h_t, shifts=shift), False)
            pos[side, i, :, 10] = hp
            pos[side, i, :, 11] = kn

            # HR
            shift = int(PHASE_LAG["HR"] * _TABLE_RESOLUTION)
            hp, kn = _inverse_kinematics(torch.roll(y_h_t * scale_hR, shifts=shift),
                                         torch.roll(z_h_t, shifts=shift), False)
            pos[side, i, :, 12] = hp
            pos[side, i, :, 13] = kn

            # 脊柱四列保持零位（由 κ 解析驱动, 见 get_reference_joint_state）
            pos[side, i, :, 0] = 0.0   # F_spine1
            pos[side, i, :, 1] = 0.0   # F_body
            pos[side, i, :, 8] = 0.0   # H_spine1
            pos[side, i, :, 9] = 0.0   # H_body

    # 中心差分计算速度表（沿相位维度）
    vel = torch.zeros_like(pos)
    two_dt = 2.0 / _TABLE_RESOLUTION
    vel[:, :, 1:-1] = (pos[:, :, 2:] - pos[:, :, :-2]) / two_dt
    vel[:, :, 0] = (pos[:, :, 1] - pos[:, :, -1]) / two_dt
    vel[:, :, -1] = (pos[:, :, 0] - pos[:, :, -2]) / two_dt

    _pos_table = pos.contiguous()
    _vel_table = vel.contiguous()
    _k_bins = torch.tensor(_CURVATURE_BINS, device=dev, dtype=torch.float32)
    _table_device = str(device)
    _tables_initialized = True

    print(f"\n[SQuRo Trot] 参考轨迹表生成完成: 2 内侧 × {_NUM_CURV} 曲率 × {_TABLE_RESOLUTION} bins × 14 joints")



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
    kappa_norm = CURVATURE_TARGET_MAX

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

    # pos_table: [2, _NUM_CURV, 50, 14]; side 选择内侧腿, 每条腿各自相位不变
    side = (curvature_cmd < 0).long()          # [N] 0=左内(κ≥0), 1=右内(κ<0)
    pos0 = _pos_table[side, idx, phase_indices]      # [N,14]
    pos1 = _pos_table[side, idx + 1, phase_indices]  # [N,14]
    vel0 = _vel_table[side, idx, phase_indices]
    vel1 = _vel_table[side, idx + 1, phase_indices]

    # 线性插值腿部参考（脊柱部分后续覆盖，但插值也参与）
    ref_pos = (1 - t.unsqueeze(1)) * pos0 + t.unsqueeze(1) * pos1
    ref_vel = ((1 - t.unsqueeze(1)) * vel0 + t.unsqueeze(1) * vel1) * gait_freq.unsqueeze(1)

    # 曲率变化率 κ̇: 把"位置参考随 κ 变化"反映到速度参考上 (回合首步置零, 避免重置瞬间的假尖峰)
    kappa_prev = getattr(env, "_ref_kappa_prev", None)
    if kappa_prev is None:
        kappa_dot = torch.zeros_like(curvature_cmd)
    else:
        kappa_dot = (curvature_cmd - kappa_prev) / dt
        kappa_dot = torch.where(env.episode_length_buf <= 1, torch.zeros_like(kappa_dot), kappa_dot)
    env._ref_kappa_prev = curvature_cmd.clone()  # type: ignore[attr-defined]

    # 腿部: 表沿曲率轴的导数 × |κ|̇ (相位轴分量已由表速度给出, 与步频解耦故在此之后加)
    d_pos_d_abs_k = (pos1 - pos0) / (k1 - k0).unsqueeze(1)  # [N,14]
    ref_vel = ref_vel + d_pos_d_abs_k * (kappa_dot * torch.sign(curvature_cmd)).unsqueeze(1)

    # 动态覆盖脊柱侧摆参考（四关节线性映射，依据 ω_cmd）
    k_norm = curvature_cmd / kappa_norm          # 归一化曲率，范围 [-1, 1]
    abs_k_norm = curvature_cmd.abs() / kappa_norm

    # 各脊柱关节目标角度（弧度），超出限位的部分被截住
    f_spine1_raw = -0.65 * k_norm
    h_spine1_raw = -0.65 * abs_k_norm
    neck_yaw_raw = 0.8 * k_norm
    ref_pos[:, 0] = torch.clamp(f_spine1_raw, -F_SPINE1_LIMIT, F_SPINE1_LIMIT)
    ref_pos[:, 1] = -0.9 * k_norm           # f_body
    ref_pos[:, 2] = torch.clamp(neck_yaw_raw, -NECK_YAW_LIMIT, NECK_YAW_LIMIT)
    ref_pos[:, 3] = -0.3                    # neck_pitch
    ref_pos[:, 8] = torch.clamp(h_spine1_raw, -H_SPINE1_LIMIT, H_SPINE1_LIMIT)
    ref_pos[:, 9] = -0.7 * k_norm           # h_body

    # 脊柱/颈速度参考 = 位置参考的解析时间导数 (被限位截住处导数为 0)
    zero_dot = torch.zeros_like(kappa_dot)
    unclamped_f = f_spine1_raw.abs() < F_SPINE1_LIMIT
    unclamped_h = h_spine1_raw.abs() < H_SPINE1_LIMIT
    unclamped_yaw = neck_yaw_raw.abs() < NECK_YAW_LIMIT
    ref_vel[:, 0] = torch.where(unclamped_f, -0.65 * kappa_dot / kappa_norm, zero_dot)
    ref_vel[:, 1] = -0.9 * kappa_dot / kappa_norm
    ref_vel[:, 2] = torch.where(unclamped_yaw, 0.8 * kappa_dot / kappa_norm, zero_dot)
    ref_vel[:, 3] = 0.0
    ref_vel[:, 8] = torch.where(unclamped_h,
                                -0.65 * torch.sign(curvature_cmd) * kappa_dot / kappa_norm,
                                zero_dot)
    ref_vel[:, 9] = -0.7 * kappa_dot / kappa_norm
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
