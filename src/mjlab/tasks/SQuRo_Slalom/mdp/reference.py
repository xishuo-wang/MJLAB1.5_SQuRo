from __future__ import annotations
import math
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.asset_zoo.robots.SQuRo.SQuRo_constants import SPINE_FRONT_FRAME, SPINE_HIND_FRAME
from .config import (
    APPROACH_LENGTH,
    BASE_VEL,
    CMD_CURVATURE_IDS,
    CMD_GAIT_FREQ_IDS,
    CMD_VEL_X_IDS,
    CURVATURE_TARGET_MAX,
    F_SPINE1_LIMIT,
    H_SPINE1_LIMIT,
    MAX_STRIDE_SCALE,
    NECK_YAW_LIMIT,
    PHASE0_EPISODE_SECONDS,
    PHASE_LAG,
    REFERENCE_CURVATURE_BLEND,
    SLALOM_CURVATURE_MAX,
    SLALOM_CURVATURE_MIN,
    SMOOTH_TIME,
    SMOOTH_VEL,
    SPINE_REFERENCE_TIME_CONSTANT,
    STRIDE_MIN,
    TABLE_RESOLUTION,
)
from .curriculums import (
    CURVATURE_TARGET,
    Rmin,
    SLALOM_CURVATURES,
    get_velocity_scale,
    phase0_kappa_at,
)
from .indices import (
    _ACT_SPN_IDS,
    _MODEL_INDICES,
    _ACT_F_BODY_ID,
    _ACT_H_BODY_ID,
    _ACT_HL_HIP_ID,
    _ACT_HR_HIP_ID,
    _ACT_HL_KNEE_ID,
    _ACT_HR_KNEE_ID,
    _ACT_JOINT_NAMES,
    _ACT_F_SPINE1_ID,
    _ACT_H_SPINE1_ID,
    _ACT_FL_ELBOW_ID,
    _ACT_FR_ELBOW_ID,
    _ACT_NECK_YAW_ID,
    _ACT_NECK_PITCH_ID,
    _ACT_FL_SHOULDER_ID,
    _ACT_FR_SHOULDER_ID,
    resolve_model_indices,
    resolve_body_envelope_bounds,
)
from mjlab.entity import Entity

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv



# 生物步态数据目录
_BIO_DATA_DIR = Path(__file__).parent / "Bio_Data"


# 腿部参考限位 (rad): 关节 range 与执行器 ctrlrange 的交集
_LEG_LIMITS: dict[int, tuple[float, float]] = {
    _ACT_FL_SHOULDER_ID: (-1.5, 1.9),
    _ACT_FL_ELBOW_ID: (-1.2, 2.5),
    _ACT_FR_SHOULDER_ID: (-1.5, 1.9),
    _ACT_FR_ELBOW_ID: (-1.2, 2.5),
    _ACT_HL_HIP_ID: (-1.5, 0.8),
    _ACT_HL_KNEE_ID: (-0.5, 1.9),
    _ACT_HR_HIP_ID: (-1.5, 0.8),
    _ACT_HR_KNEE_ID: (-0.5, 1.9),
}



# 离散曲率绝对值表
def _generate_curvature_bins(max_k: float) -> list[float]:
    bins = [i * 0.5 for i in range(11)]  # [0.0, 0.5, ..., 5.0]
    if max_k > 5.0:
        bins.extend(float(x) for x in range(6, int(max_k) + 1))
    return bins

_CURVATURE_BINS = _generate_curvature_bins(CURVATURE_TARGET_MAX)
_NUM_CURV = len(_CURVATURE_BINS)



_tables_initialized = False
_table_device: str | None = None
_table_base_vel: float = BASE_VEL
_k_bins: torch.Tensor | None = None        # [_NUM_CURV] 曲率查找表(缓存)
_pos_table: torch.Tensor | None = None     # [峰值档, 内侧, 瞬时档, 相位, 关节]
_vel_table: torch.Tensor | None = None     # 同上



# 平滑后脊柱曲率幅值及其导数，零点保留微小弯曲以避免速度翻转。
def smooth_curvature_magnitude(curvature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    magnitude = curvature.abs()
    ratio = (magnitude / REFERENCE_CURVATURE_BLEND).clamp(max=1.0)
    rounded = REFERENCE_CURVATURE_BLEND * (3.0 + 6.0 * ratio.square() - ratio.pow(4)) / 8.0
    slope = (1.5 * ratio - 0.5 * ratio.pow(3)) * curvature.sign()
    return torch.where(magnitude < REFERENCE_CURVATURE_BLEND, rounded, magnitude), slope



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



# 把 IK 输出折回关节范围附近并夹到机械限位, 防止不可达目标产生非法参考
def _clamp_leg_angle(angle: torch.Tensor, joint_id: int) -> torch.Tensor:
    low, high = _LEG_LIMITS[joint_id]
    wrapped = (angle - low) % (2.0 * math.pi) + low
    return torch.where(wrapped > high, wrapped - 2.0 * math.pi, wrapped).clamp(low, high)


# 加载 CSV + IK 预计算（步幅按本回合目标速度缩放，瞬时曲率只负责左右差速）
def _init_tables(device: torch.device | str, base_velocity: float = BASE_VEL) -> None:
    global _tables_initialized, _pos_table, _vel_table, _k_bins, _table_device, _table_base_vel

    base_velocity = float(base_velocity)
    if _tables_initialized and _table_device == str(device) and _table_base_vel == base_velocity:
        return

    # 加载 CSV
    df_f = pd.read_csv(_BIO_DATA_DIR / "Trot_F.csv")
    df_h = pd.read_csv(_BIO_DATA_DIR / "Trot_H.csv")
    phase_csv: np.ndarray = df_f["Phase"].values.astype(np.float64)
    y_f: np.ndarray = df_f["Y_mean"].values.astype(np.float64)
    z_f: np.ndarray = df_f["Z_mean"].values.astype(np.float64)
    y_h: np.ndarray = df_h["Y_mean"].values.astype(np.float64)
    z_h: np.ndarray = df_h["Z_mean"].values.astype(np.float64)

    # 生成等距相位网格
    phases = np.linspace(0, 1, TABLE_RESOLUTION)

    # 在相位网格上插值足端轨迹
    y_f_grid = np.interp(phases, phase_csv, y_f)
    z_f_grid = np.interp(phases, phase_csv, z_f)
    y_h_grid = np.interp(phases, phase_csv, y_h)
    z_h_grid = np.interp(phases, phase_csv, z_h)

    # 双腿平均整周行程, 用于把步幅换算成每周前进距离
    stride_mean = 0.5 * (float(y_f_grid.max() - y_f_grid.min()) + float(y_h_grid.max() - y_h_grid.min()))

    # 转为 torch
    dev = torch.device(device)
    y_f_t = torch.tensor(y_f_grid, device=dev, dtype=torch.float32)
    z_f_t = torch.tensor(z_f_grid, device=dev, dtype=torch.float32)
    y_h_t = torch.tensor(y_h_grid, device=dev, dtype=torch.float32)
    z_h_t = torch.tensor(z_h_grid, device=dev, dtype=torch.float32)

    # 五维表 [峰值档, 内侧, 瞬时档, 相位, 关节]: 峰值决定本回合速度, 瞬时决定左右差速
    # 每侧只缩放"内侧腿"的 Y_mean, 腿部相位不随曲率符号改变
    pos = torch.zeros(_NUM_CURV, 2, _NUM_CURV, TABLE_RESOLUTION, len(_ACT_JOINT_NAMES), device=dev)

    # 对每个峰值档与瞬时曲率生成参考：平均步幅 = 本回合每周前进距离
    for p, peak_k in enumerate(_CURVATURE_BINS):
        advance = base_velocity * float(get_velocity_scale(peak_k))
        for i, abs_k in enumerate(_CURVATURE_BINS):
            scale_inner = 1.0 - (1.0 - STRIDE_MIN) * abs_k / CURVATURE_TARGET_MAX
            mean_factor = 0.5 * (1.0 + scale_inner)
            row_scale = min(advance / max(stride_mean * mean_factor, 1e-6), MAX_STRIDE_SCALE)

            for side in (0, 1):
                # 内侧腿缩放其 Y_mean (side=0: FL/HL, side=1: FR/HR), 外侧腿保持 1.0
                scale_fL = scale_inner if side == 0 else 1.0
                scale_fR = 1.0 if side == 0 else scale_inner
                scale_hL = scale_inner if side == 0 else 1.0
                scale_hR = 1.0 if side == 0 else scale_inner

                # 分别计算每条腿的关节角（应用相位差并夹到关节限位）
                shift = int(PHASE_LAG["FL"] * TABLE_RESOLUTION)
                sh, el = _inverse_kinematics(torch.roll(y_f_t * row_scale * scale_fL, shifts=shift), torch.roll(z_f_t, shifts=shift), True)
                pos[p, side, i, :, _ACT_FL_SHOULDER_ID] = _clamp_leg_angle(sh, _ACT_FL_SHOULDER_ID)
                pos[p, side, i, :, _ACT_FL_ELBOW_ID] = _clamp_leg_angle(el, _ACT_FL_ELBOW_ID)

                shift = int(PHASE_LAG["FR"] * TABLE_RESOLUTION)
                sh, el = _inverse_kinematics(torch.roll(y_f_t * row_scale * scale_fR, shifts=shift), torch.roll(z_f_t, shifts=shift), True)
                pos[p, side, i, :, _ACT_FR_SHOULDER_ID] = _clamp_leg_angle(sh, _ACT_FR_SHOULDER_ID)
                pos[p, side, i, :, _ACT_FR_ELBOW_ID] = _clamp_leg_angle(el, _ACT_FR_ELBOW_ID)

                shift = int(PHASE_LAG["HL"] * TABLE_RESOLUTION)
                hp, kn = _inverse_kinematics(torch.roll(y_h_t * row_scale * scale_hL, shifts=shift), torch.roll(z_h_t, shifts=shift), False)
                pos[p, side, i, :, _ACT_HL_HIP_ID] = _clamp_leg_angle(hp, _ACT_HL_HIP_ID)
                pos[p, side, i, :, _ACT_HL_KNEE_ID] = _clamp_leg_angle(kn, _ACT_HL_KNEE_ID)

                shift = int(PHASE_LAG["HR"] * TABLE_RESOLUTION)
                hp, kn = _inverse_kinematics(torch.roll(y_h_t * row_scale * scale_hR, shifts=shift), torch.roll(z_h_t, shifts=shift), False)
                pos[p, side, i, :, _ACT_HR_HIP_ID] = _clamp_leg_angle(hp, _ACT_HR_HIP_ID)
                pos[p, side, i, :, _ACT_HR_KNEE_ID] = _clamp_leg_angle(kn, _ACT_HR_KNEE_ID)

                # 脊柱四列保持零位（由 κ 解析驱动, 见 get_reference_joint_state）
                pos[p, side, i, :, _ACT_F_SPINE1_ID] = 0.0   # F_spine1
                pos[p, side, i, :, _ACT_F_BODY_ID] = 0.0   # F_body
                pos[p, side, i, :, _ACT_H_SPINE1_ID] = 0.0   # H_spine1
                pos[p, side, i, :, _ACT_H_BODY_ID] = 0.0   # H_body

    # 中心差分计算速度表（沿相位维度）
    vel = torch.zeros_like(pos)
    two_dt = 2.0 / TABLE_RESOLUTION
    vel[:, :, :, 1:-1] = (pos[:, :, :, 2:] - pos[:, :, :, :-2]) / two_dt
    vel[:, :, :, 0] = (pos[:, :, :, 1] - pos[:, :, :, -1]) / two_dt
    vel[:, :, :, -1] = (pos[:, :, :, 0] - pos[:, :, :, -2]) / two_dt

    _pos_table = pos.contiguous()
    _vel_table = vel.contiguous()
    _k_bins = torch.tensor(_CURVATURE_BINS, device=dev, dtype=torch.float32)
    _table_device = str(device)
    _table_base_vel = base_velocity
    _tables_initialized = True

    print(f"\n[SQuRo Trot] 参考轨迹表生成完成: {_NUM_CURV} 峰值 × 2 内侧 × {_NUM_CURV} 瞬时 × "
          f"{TABLE_RESOLUTION} bins × {len(_ACT_JOINT_NAMES)} joints, 基础速度 {base_velocity:.3f} m/s")



# 重置代次 +1: 使同一控制步内已缓存的参考失效 (框架在同一步内先算奖励、再重置、再算观测)
def invalidate_reference_cache(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None) -> None:
    env._ref_epoch = getattr(env, "_ref_epoch", 0) + 1  # type: ignore[attr-defined]
    state = getattr(env, "_ref_spine_filter", None)
    if state is not None and env_ids is not None:
        state["step"][env_ids] = -1



# 二阶段平滑四脊柱位置并由同一轨迹差分速度，重置与步级缓存语义见碰撞课程文档
def _smooth_spine_reference(env, ref_pos, ref_vel, phase1) -> None:
    raw = ref_pos[:, _ACT_SPN_IDS]
    state = getattr(env, "_ref_spine_filter", None)
    if state is None:
        state = {
            "position": raw.clone(), "velocity": torch.zeros_like(raw),
            "step": torch.full_like(env.episode_length_buf, -1),
            "episode": env.episode_length_buf.clone(), "active": phase1.clone(),
        }
        env._ref_spine_filter = state
    initialize = ((state["step"] < 0) | (env.episode_length_buf < state["episode"]) | (phase1 != state["active"]))
    advance = state["step"] != env.common_step_counter
    alpha = env.step_dt / (SPINE_REFERENCE_TIME_CONSTANT + env.step_dt)
    next_pos = state["position"] + alpha * (raw - state["position"])
    next_vel = (next_pos - state["position"]) / env.step_dt
    position = torch.where(initialize[:, None], raw, torch.where(advance[:, None], next_pos, state["position"]))
    velocity = torch.where(initialize[:, None], torch.zeros_like(raw), torch.where(advance[:, None], next_vel, state["velocity"]))
    ref_pos[:, _ACT_SPN_IDS] = torch.where(phase1[:, None], position, raw)
    ref_vel[:, _ACT_SPN_IDS] = torch.where(phase1[:, None], velocity, ref_vel[:, _ACT_SPN_IDS])
    state.update(position=ref_pos[:, _ACT_SPN_IDS].clone(), velocity=ref_vel[:, _ACT_SPN_IDS].clone(),
                 step=torch.full_like(state["step"], env.common_step_counter),
                 episode=env.episode_length_buf.clone(), active=phase1.clone())



# 获取当前步的参考关节位置和速度
def get_reference_joint_state(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    # 步级缓存: 键含重置代次, 重置后同一控制步内必须重算 (观测要用新回合的参考)
    current_step = env.common_step_counter
    epoch = getattr(env, "_ref_epoch", 0)
    cached = getattr(env, "_ref_state_cache", None)
    if cached is not None and cached[0] == current_step and cached[3] == epoch:
        return cached[1], cached[2]

    # 首次: 初始化相位状态 + 解析模型索引
    if getattr(env, "_ref_phase", None) is None:
        env._ref_phase = torch.zeros(env.num_envs, device=env.device)  # type: ignore[attr-defined]
        resolve_model_indices(env.scene["robot"])

    # 预计算查找表: 设备或实际基础速度变化时重建 (回放可指定固定速度)
    base_vel = float(getattr(env, "_slalom_base_vel_scalar", BASE_VEL))
    if not _tables_initialized or _table_device != str(env.device) or _table_base_vel != base_vel:
        _init_tables(env.device, base_vel)

    assert _pos_table is not None and _vel_table is not None

    dt = float(env.step_dt)

    # 当前相位 → 查表索引
    phase: torch.Tensor = env._ref_phase  # type: ignore[attr-defined]  # [N]
    phase_indices = (phase * (TABLE_RESOLUTION - 1)).long().clamp_(0, TABLE_RESOLUTION - 1)

    # 瞬时曲率与参考位置同源: 由本步的路径参考算出 (重置后 t=0, 故重置回合拿到接近段起点曲率)
    compute_path_ref(env)
    curvature_cmd = get_path_curvature(env)  # [N]
    cmd_tensor = env.command_manager._terms["slalom_cmd"].command
    vel_cmd = cmd_tensor[:, 0]  # [N]
    gait_freq = cmd_tensor[:, 3]  # [N] — 动态步频
    kappa_norm = CURVATURE_TARGET_MAX

    # 曲率绝对值及插值因子（k_bins 在 _init_tables 时缓存）
    abs_k = curvature_cmd.abs()  # [N]
    assert _k_bins is not None

    # 本回合目标峰值: 二阶段取采样档位, 一阶段取日程峰值 (动态回合同样按速度缩放)
    command_term = env.command_manager._terms["slalom_cmd"]
    phase1 = command_term.phase1_mask  # type: ignore
    peak_attr = getattr(command_term, "curvature_peak_command", None)
    peak_k = peak_attr if peak_attr is not None else torch.where(phase1, command_term.slalom_peak_curvature, abs_k)  # type: ignore

    # 峰值轴插值: 表按平均步幅对应本回合每周前进距离, 一阶段曲率连续需要插值
    pidx = torch.searchsorted(_k_bins, peak_k) - 1
    pidx = pidx.clamp(0, _NUM_CURV - 2)
    pk0 = _k_bins[pidx]
    pk1 = _k_bins[pidx + 1]
    tp = ((peak_k - pk0) / (pk1 - pk0 + 1e-12)).clamp(0.0, 1.0).unsqueeze(1)

    # 找到每个环境的曲率区间索引（左侧）
    idx = torch.searchsorted(_k_bins, abs_k) - 1
    idx = idx.clamp(0, _NUM_CURV - 2)  # 防止边界外，使 idx+1 有效

    # 对应的曲率值
    k0 = _k_bins[idx]        # [N]
    k1 = _k_bins[idx + 1]
    t = (abs_k - k0) / (k1 - k0 + 1e-12)  # 插值因子，[0,1]

    # pos_table: [峰值档, 2, 瞬时档, 50, 14]; side 选择内侧腿, 每条腿各自相位不变
    side = (curvature_cmd < 0).long()          # [N] 0=左内(κ≥0), 1=右内(κ<0)
    pos00 = _pos_table[pidx, side, idx, phase_indices]
    pos01 = _pos_table[pidx, side, idx + 1, phase_indices]
    pos10 = _pos_table[pidx + 1, side, idx, phase_indices]
    pos11 = _pos_table[pidx + 1, side, idx + 1, phase_indices]
    pos_lo = (1 - tp) * pos00 + tp * pos10
    pos_hi = (1 - tp) * pos01 + tp * pos11
    vel00 = _vel_table[pidx, side, idx, phase_indices]
    vel01 = _vel_table[pidx, side, idx + 1, phase_indices]
    vel10 = _vel_table[pidx + 1, side, idx, phase_indices]
    vel11 = _vel_table[pidx + 1, side, idx + 1, phase_indices]
    vel_lo = (1 - tp) * vel00 + tp * vel10
    vel_hi = (1 - tp) * vel01 + tp * vel11

    # 线性插值腿部参考（脊柱部分后续覆盖，但插值也参与）
    ref_pos = (1 - t.unsqueeze(1)) * pos_lo + t.unsqueeze(1) * pos_hi
    ref_vel = ((1 - t.unsqueeze(1)) * vel_lo + t.unsqueeze(1) * vel_hi) * gait_freq.unsqueeze(1)

    # 曲率变化率 κ̇: 把"位置参考随 κ 变化"反映到速度参考上 (回合首步置零, 避免重置瞬间的假尖峰)
    kappa_prev = getattr(env, "_ref_kappa_prev", None)
    if kappa_prev is None:
        kappa_dot = torch.zeros_like(curvature_cmd)
    else:
        kappa_dot = (curvature_cmd - kappa_prev) / dt
        kappa_dot = torch.where(env.episode_length_buf <= 1, torch.zeros_like(kappa_dot), kappa_dot)

    # 腿部: 表沿曲率轴的导数 × |κ|̇ (相位轴分量已由表速度给出, 与步频解耦故在此之后加)
    d_pos_d_abs_k = (pos_hi - pos_lo) / (k1 - k0).unsqueeze(1)  # [N,14]
    ref_vel = ref_vel + d_pos_d_abs_k * (kappa_dot * torch.sign(curvature_cmd)).unsqueeze(1)

    # 峰值轴: 回合峰值在回合内固定, 用历史差分; 旧桩环境无峰值字段时退回瞬时曲率语义
    d_pos_d_peak = ((1 - t.unsqueeze(1)) * (pos10 - pos00) + t.unsqueeze(1) * (pos11 - pos01)) \
        / (pk1 - pk0).unsqueeze(1)
    if peak_attr is not None:
        peak_prev = getattr(env, "_ref_peak_prev", None)
        if peak_prev is None:
            peak_dot = torch.zeros_like(kappa_dot)
        else:
            peak_dot = (peak_k - peak_prev) / dt
            peak_dot = torch.where(env.episode_length_buf <= 1, torch.zeros_like(peak_dot), peak_dot)
    else:
        peak_dot = torch.where(phase1, torch.zeros_like(kappa_dot), kappa_dot * torch.sign(curvature_cmd))
    ref_vel = ref_vel + d_pos_d_peak * peak_dot.unsqueeze(1)

    # 动态覆盖脊柱侧摆参考（四关节线性映射，依据 ω_cmd）
    k_norm = curvature_cmd / kappa_norm          # 归一化曲率，范围 [-1, 1]
    smooth_magnitude, smooth_slope = smooth_curvature_magnitude(curvature_cmd)
    # 二阶段与动态一阶段都经过零点: 幅值、斜率与四脊柱滤波统一用同源平滑, 恒定一阶段保持原状
    schedule = getattr(command_term, "phase0_schedule", None)
    dynamic_phase0 = (schedule["dynamic"] & ~phase1) if schedule is not None else torch.zeros_like(phase1)
    smooth_mask = phase1 | dynamic_phase0
    hind_magnitude = torch.where(smooth_mask, smooth_magnitude, abs_k)
    hind_slope = torch.where(smooth_mask, smooth_slope, curvature_cmd.sign())
    abs_k_norm = hind_magnitude / kappa_norm

    # 各脊柱关节目标角度（弧度），超出限位的部分被截住
    f_spine1_raw = -0.65 * k_norm
    h_spine1_raw = -0.65 * abs_k_norm
    neck_yaw_raw = 0.8 * k_norm
    ref_pos[:, _ACT_F_SPINE1_ID] = torch.clamp(f_spine1_raw, -F_SPINE1_LIMIT, F_SPINE1_LIMIT)
    ref_pos[:, _ACT_F_BODY_ID] = -0.9 * k_norm           # f_body
    ref_pos[:, _ACT_NECK_YAW_ID] = torch.clamp(neck_yaw_raw, -NECK_YAW_LIMIT, NECK_YAW_LIMIT)
    ref_pos[:, _ACT_NECK_PITCH_ID] = -0.3                    # neck_pitch
    ref_pos[:, _ACT_H_SPINE1_ID] = torch.clamp(h_spine1_raw, -H_SPINE1_LIMIT, H_SPINE1_LIMIT)
    ref_pos[:, _ACT_H_BODY_ID] = -0.7 * k_norm           # h_body

    # 脊柱/颈速度参考 = 位置参考的解析时间导数 (被限位截住处导数为 0)
    zero_dot = torch.zeros_like(kappa_dot)
    unclamped_f = f_spine1_raw.abs() < F_SPINE1_LIMIT
    unclamped_h = h_spine1_raw.abs() < H_SPINE1_LIMIT
    unclamped_yaw = neck_yaw_raw.abs() < NECK_YAW_LIMIT
    ref_vel[:, _ACT_F_SPINE1_ID] = torch.where(unclamped_f, -0.65 * kappa_dot / kappa_norm, zero_dot)
    ref_vel[:, _ACT_F_BODY_ID] = -0.9 * kappa_dot / kappa_norm
    ref_vel[:, _ACT_NECK_YAW_ID] = torch.where(unclamped_yaw, 0.8 * kappa_dot / kappa_norm, zero_dot)
    ref_vel[:, _ACT_NECK_PITCH_ID] = 0.0
    ref_vel[:, _ACT_H_SPINE1_ID] = torch.where(unclamped_h, -0.65 * hind_slope * kappa_dot / kappa_norm, zero_dot)
    ref_vel[:, _ACT_H_BODY_ID] = -0.7 * kappa_dot / kappa_norm
    _smooth_spine_reference(env, ref_pos, ref_vel, smooth_mask)
    # 相位推进 / κ 历史 / 重置清零每步只做一次: 重置后的重算不得二次推进或二次记录
    if getattr(env, "_ref_phase_step", None) != current_step:
        env._ref_kappa_prev = curvature_cmd.clone()  # type: ignore[attr-defined]
        env._ref_peak_prev = peak_k.clone()  # type: ignore[attr-defined]
        env._ref_phase = (phase + gait_freq * dt) % 1.0  # type: ignore[attr-defined]
        # 重置环境 (含超时, reset_buf = 终止 | 超时) 的步态相位清零, 保证每个回合从同一相位起步
        reset_buf = getattr(env, "reset_buf", None)
        if reset_buf is not None:
            ids = reset_buf.nonzero(as_tuple=False).flatten()
            if len(ids) > 0:
                env._ref_phase[ids] = 0.0  # type: ignore[attr-defined]
        env._ref_phase_step = current_step  # type: ignore[attr-defined]

    env._ref_state_cache = (current_step, ref_pos, ref_vel, epoch)  # type: ignore[attr-defined]
    return ref_pos, ref_vel



# 获取指定 body_link 的偏航角
def get_body_heading(env: "ManagerBasedRlEnv", body_id: int | None = None) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    if body_id is None:
        quat = asset.data.root_link_quat_w
    else:
        quat = asset.data.body_link_quat_w[:, body_id]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sin_h = 2.0 * (w * z + x * y)
    cos_h = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(sin_h, cos_h)



# 躯干物理偏航与滚转，与 Hole 的 _get_body_orientation 使用同一套解剖坐标标定
def get_body_orientation(env: "ManagerBasedRlEnv") -> tuple[torch.Tensor, torch.Tensor]:
    asset: Entity = env.scene["robot"]
    body_ids = (_MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id)
    quat = asset.data.body_link_quat_w[:, body_ids]
    rotation = matrix_from_quat(quat) @ quat.new_tensor((SPINE_FRONT_FRAME, SPINE_HIND_FRAME))
    yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    roll = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    return yaw, roll



# F_body_Link 物理前向 heading（与 Hole 的 yaw 同源）
def get_f_body_physical_heading(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return get_body_orientation(env)[0][..., 0]



# H_body_Link 物理前向 heading
def get_h_body_physical_heading(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return get_body_orientation(env)[0][..., 1]



# 按整数曲率返回接近段起点，指定杆距时选择最近的可实现条件。
def get_approach_start(spacing: float | None = None, curvature: int | None = None) -> tuple[float, float, float]:
    k = curvature if curvature is not None else (int(CURVATURE_TARGET) if spacing is None else curvature_from_spacing(spacing))
    approach = get_slalom_geometry(k)["approach"]
    x = float(approach["x"][0])
    y = float(approach["y"][0])
    heading = float(approach["heading"][0])
    return x, y, heading



# Phase 0 圆弧接近段
def get_phase0_approach(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s0 = _INIT_DIST
    mask = torch.abs(k) > 1e-6
    k_s = torch.where(mask, k, torch.ones_like(k))
    x = torch.where(mask, -torch.sin(k * s0) / k_s, torch.full_like(k, -s0))
    y = torch.where(mask, (1.0 - torch.cos(k * s0)) / k_s, torch.zeros_like(k))
    h = torch.where(mask, -k * s0, torch.zeros_like(k))
    return x, y, h



# 机器人初始位置/朝向 (名义 vel), 与 get_approach_start 同义
def get_phase1_approach(spacing: float | None = None) -> tuple[float, float, float]:
    return get_approach_start(spacing)



# 路径参考调度 — 根据训练阶段自动选择圆弧或绕杆路径
def compute_path_ref(env: "ManagerBasedRlEnv"):
    cmd_term = env.command_manager._terms["slalom_cmd"]
    # 逐环境阶段: 两条路径都算再按阶段选取, 阶段切换后的一个回合内会同时存在两种环境
    phase1 = cmd_term.phase1_mask  # type: ignore[union-attr]
    sx, sy, svx, svy, sh = compute_slalom_path_ref(env)
    kappa_slalom = env._path_kappa  # type: ignore[attr-defined]
    ax, ay, avx, avy, ah = compute_arc_path_ref(env)
    env._path_kappa = torch.where(phase1, kappa_slalom, getattr(env, "_phase0_path_kappa", cmd_term.command[:, CMD_CURVATURE_IDS])) # type: ignore
    return (
        torch.where(phase1, sx, ax),
        torch.where(phase1, sy, ay),
        torch.where(phase1, svx, avx),
        torch.where(phase1, svy, avy),
        torch.where(phase1, sh, ah),
    )



# 预计算一阶段回合路径: 按曲率日程积分成逐控制步位置/朝向数组, 长度按一阶段有效回合
def build_phase0_episode_path(env: "ManagerBasedRlEnv", env_ids, start_x, start_y, start_heading, schedule, velocity) -> None:
    n = len(env_ids)
    dt = float(env.step_dt)
    steps = min(int(env.max_episode_length), int(round(PHASE0_EPISODE_SECONDS / dt)))
    segments = schedule["segments"][env_ids]
    bounds = schedule["bounds"][env_ids]
    path_xy = getattr(env, "_phase0_path_xy", None)
    if path_xy is None or path_xy.shape[1] != steps + 1:
        env._phase0_path_xy = torch.zeros(env.num_envs, steps + 1, 2, device=env.device)  # type: ignore
        env._phase0_path_heading = torch.zeros(env.num_envs, steps + 1, device=env.device)  # type: ignore
    env._phase0_schedule = schedule  # type: ignore
    # 逐控制步中点曲率, 用圆弧弦增量累积: 常数曲率下与解析圆弧一致
    mid_t = (torch.arange(steps, device=env.device).float() + 0.5) * dt
    kappa = phase0_kappa_at(mid_t[None, :].expand(n, -1), {"segments": segments, "bounds": bounds})
    delta_theta = kappa * velocity[:, None] * dt
    heading = torch.cat((start_heading[:, None], start_heading[:, None] + torch.cumsum(delta_theta, dim=1)), dim=1)
    previous, following = heading[:, :-1], heading[:, 1:]
    turning = kappa.abs() > 1e-6
    safe = torch.where(turning, kappa, torch.ones_like(kappa))
    chord_x = torch.where(turning, (torch.sin(following) - torch.sin(previous)) / safe, velocity[:, None] * dt * torch.cos(previous))
    chord_y = torch.where(turning, -(torch.cos(following) - torch.cos(previous)) / safe, velocity[:, None] * dt * torch.sin(previous))
    env._phase0_path_xy[env_ids, :, 0] = torch.cat(  # type: ignore
        (start_x[:, None], start_x[:, None] + torch.cumsum(chord_x, dim=1)), dim=1)
    env._phase0_path_xy[env_ids, :, 1] = torch.cat(  # type: ignore
        (start_y[:, None], start_y[:, None] + torch.cumsum(chord_y, dim=1)), dim=1)
    env._phase0_path_heading[env_ids, :] = heading  # type: ignore



# 路径参考计算 — 基元阶段: 查本回合预计算路径, 动态切换时路径保持连续
def compute_arc_path_ref(env: "ManagerBasedRlEnv"):
    cmd_term = env.command_manager._terms["slalom_cmd"]
    device = env.device
    path_xy = getattr(env, "_phase0_path_xy", None)
    if path_xy is None:
        env._phase0_path_xy = torch.zeros(env.num_envs, 1, 2, device=device)  # type: ignore
        env._phase0_path_heading = torch.zeros(env.num_envs, 1, device=device)  # type: ignore
        path_xy = env._phase0_path_xy  # type: ignore
    steps = path_xy.shape[1] - 1
    idx = env.episode_length_buf.long().clamp(0, steps)
    rows = torch.arange(env.num_envs, device=device)
    xy = path_xy[rows, idx]
    path_heading = env._phase0_path_heading[rows, idx]  # type: ignore
    schedule = getattr(env, "_phase0_schedule", None)
    if schedule is None:
        schedule = getattr(cmd_term, "phase0_schedule", None)
    if schedule is None:
        schedule = {"segments": cmd_term.command[:, CMD_CURVATURE_IDS:CMD_CURVATURE_IDS + 1],
                    "bounds": torch.full_like(cmd_term.command[:, CMD_VEL_X_IDS:CMD_VEL_X_IDS + 1], 1e9),
                    "peak": cmd_term.command[:, CMD_CURVATURE_IDS].abs()}
    env._phase0_path_kappa = phase0_kappa_at(idx.float() * env.step_dt, schedule)  # type: ignore[attr-defined]
    vel_cmd = cmd_term.command[:, CMD_VEL_X_IDS]
    vx_des = vel_cmd * torch.cos(path_heading)
    vy_des = vel_cmd * torch.sin(path_heading)
    origins = getattr(getattr(env, "scene", None), "env_origins", None)
    x_ref, y_ref = xy[:, 0], xy[:, 1]
    if origins is not None:
        x_ref, y_ref = x_ref + origins[:, 0], y_ref + origins[:, 1]
    return x_ref, y_ref, vx_des, vy_des, path_heading



# 历史纯圆弧绘图对照；实际训练使用下方无直行平滑路径。
def _generate_slalom_lut_one_period(X: float, n_arc_pts: int = 15):
    def _arc_np(start, end, r, clockwise, steps=n_arc_pts):
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        mid = (start + end) / 2
        chord_vec = end - start
        chord_len = np.linalg.norm(chord_vec)
        d = np.sqrt(max(0, r**2 - (chord_len / 2)**2))
        perp = np.array([-chord_vec[1], chord_vec[0]]) / chord_len
        sign = -1 if clockwise else 1
        center = mid + sign * d * perp
        v_s = start - center; v_e = end - center
        a_s = np.arctan2(v_s[1], v_s[0]); a_e = np.arctan2(v_e[1], v_e[0])
        if clockwise:
            if a_e > a_s: a_e -= 2 * np.pi
        else:
            if a_e < a_s: a_e += 2 * np.pi
        theta = np.linspace(a_s, a_e, steps)
        return center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)

    r = Rmin
    x0, y0 = 0.0, 0.0
    pts_x, pts_y = [x0], [y0]

    # 直行段采样点数 — 与弧段密度一致
    arc_len = r * np.pi / 2            # 单个 1/4 弧的弧长
    straight_len = X - 2 * r           # 单个直行段长度
    n_straight = max(2, int(straight_len / arc_len * n_arc_pts)) if straight_len > 1e-9 else 0

    # S1: CW ¼ arc (0,0) → (r, -r)  [跳过首点，因与 waypoint 重复]
    ax, ay = _arc_np((x0, y0), (x0 + r, y0 - r), r, clockwise=True)
    pts_x.extend(ax[1:]); pts_y.extend(ay[1:])
    # S2: CCW ¼ arc (r, -r) → (2r, -2r)
    ax, ay = _arc_np((x0 + r, y0 - r), (x0 + 2*r, y0 - 2*r), r, clockwise=False)
    pts_x.extend(ax[1:]); pts_y.extend(ay[1:])
    # S3: straight → (X, -2r)  [多点插值]
    if n_straight > 0:
        sx = np.linspace(x0 + 2*r, x0 + X, n_straight)[1:]  # 跳过首点
        sy = np.full(n_straight - 1, y0 - 2*r)
        pts_x.extend(sx); pts_y.extend(sy)
    # S4: CCW ¼ arc (X, -2r) → (X+r, -r)
    ax, ay = _arc_np((x0 + X, y0 - 2*r), (x0 + X + r, y0 - r), r, clockwise=False)
    pts_x.extend(ax[1:]); pts_y.extend(ay[1:])
    # S5: CW ¼ arc (X+r, -r) → (X+2r, 0)
    ax, ay = _arc_np((x0 + X + r, y0 - r), (x0 + X + 2*r, y0), r, clockwise=True)
    pts_x.extend(ax[1:]); pts_y.extend(ay[1:])
    # S6: straight → (2X, 0)  [多点插值]
    if n_straight > 0:
        sx = np.linspace(x0 + X + 2*r, x0 + 2*X, n_straight)[1:]  # 跳过首点
        sy = np.zeros(n_straight - 1)
        pts_x.extend(sx); pts_y.extend(sy)

    # 累积弧长 + heading + curvature (前向差分)
    pts = np.column_stack([pts_x, pts_y])
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lens)])

    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    headings = np.concatenate([headings, headings[-1:]])  # 最后一点复用

    # 曲率: κ = Δheading / Δarc (弧度差需 wrap 到 [-π, π])
    dtheta = np.diff(headings)
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    kappa_vals = dtheta / (seg_lens + 1e-12)
    kappa_vals = np.concatenate([kappa_vals, kappa_vals[-1:]])

    return arc, pts_x, pts_y, headings, kappa_vals



# 将输入杆距映射到整数峰值曲率对应的真实杆距。
def get_effective_pole_spacing(spacing: float) -> float:
    return get_slalom_geometry(curvature_from_spacing(spacing))["spacing"]



# 保留诊断脚本入口，当前绕杆路径全部采用无直行结构。
def is_no_straight_spacing(spacing: float) -> bool:
    del spacing
    return True



# 返回无直行周期的几何表，曲率与杆距由同一份几何确定
def _generate_slalom_lut_smooth_period(X: float, smooth_time=None, vel=None, curvature: int | None = None):
    if smooth_time is not None or vel is not None:
        raise ValueError("无直行路径使用集中配置的固定平滑弧长")
    k = curvature_from_spacing(X) if curvature is None else curvature
    period = get_slalom_geometry(k)["period"]
    return tuple(period[name] for name in ("arc", "x", "y", "heading", "kappa"))



# 把全部整数曲率的名义时间表缓存到当前设备。
def get_slalom_tables(env, base_velocity: float) -> dict[str, torch.Tensor]:
    cached = getattr(env, "_slalom_tables", None)
    if cached is None or cached[0] != base_velocity:
        cached = (base_velocity, {name: torch.as_tensor(value, device=env.device, dtype=torch.float32) for name, value in get_timed_slalom_tables(base_velocity).items()})
        env._slalom_tables = cached
    return cached[1]



# 按逐环境曲率行和归一化时间插值，避免每步复制完整路径表
def _sample_slalom_table(table: torch.Tensor, rows: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
    coordinate = progress.clamp(0, 1) * (table.shape[1] - 1)
    left = coordinate.long().clamp(max=table.shape[1] - 2)
    fraction = (coordinate - left).unsqueeze(-1)
    return torch.lerp(table[rows, left], table[rows, left + 1], fraction)



# 绕杆参考按本回合的整数曲率和自身步频推进，杆距不会随其他环境重置改变。
def compute_slalom_path_ref(env: "ManagerBasedRlEnv"):
    cmd = env.command_manager._terms["slalom_cmd"]
    base_velocity = float(getattr(env, "_slalom_base_vel_scalar", BASE_VEL))
    tables = get_slalom_tables(env, base_velocity)
    rows = cmd.slalom_curvature_index  # type: ignore
    tau = env.episode_length_buf.float() * env.step_dt * cmd.command[:, CMD_GAIT_FREQ_IDS]
    approach_time = tables["approach_time"][rows]
    period_time = tables["period_time"][rows]
    in_approach = tau < approach_time
    approach = _sample_slalom_table(tables["approach"], rows, tau / approach_time)
    elapsed = (tau - approach_time).clamp_min(0)
    cycles = torch.floor(elapsed / period_time)
    period = _sample_slalom_table(tables["period"], rows, elapsed / period_time - cycles)
    x_period = period[:, 0] + cmd.pole_spacing * (1 + 2 * cycles)  # type: ignore
    x = torch.where(in_approach, approach[:, 0], x_period)
    y = torch.where(in_approach, approach[:, 1], period[:, 1])
    heading = torch.where(in_approach, approach[:, 2], period[:, 2])
    kappa = torch.where(in_approach, approach[:, 3], period[:, 3])
    origins = getattr(getattr(env, "scene", None), "env_origins", None)
    if origins is not None:
        x, y = x + origins[:, 0], y + origins[:, 1]
    velocity = base_velocity * cmd.command[:, CMD_GAIT_FREQ_IDS] * get_velocity_scale(cmd.slalom_peak_curvature)  # type: ignore
    env._path_kappa = kappa  # type: ignore
    return x, y, velocity * heading.cos(), velocity * heading.sin(), heading



# 获取当前路径瞬时曲率 κ(t) — 由 compute_path_ref 逐环境写入 (Phase 0 为命令值, Phase 1 为 LUT 插值)
def get_path_curvature(env: "ManagerBasedRlEnv") -> torch.Tensor:
    cmd_term = env.command_manager._terms["slalom_cmd"]
    return getattr(env, "_path_kappa", cmd_term.command[:, CMD_CURVATURE_IDS])



# 无直行路径几何 LUT
_SEGMENT_SAMPLES = 256
_APPROACH_SAMPLES = 256
_PROJECTION_PERIOD_SAMPLES = 512
_PROJECTION_APPROACH_SAMPLES = 32



# 按每段真实步长积分曲率，保留位置、切线和曲率的连续端点。
def integrate_segments(segments: list[tuple[float, float, float]]) -> dict[str, np.ndarray]:
    lengths = np.concatenate([np.full(_SEGMENT_SAMPLES, length / _SEGMENT_SAMPLES) for length, _, _ in segments])
    pieces = [np.linspace(start, end, _SEGMENT_SAMPLES + 1) for _, start, end in segments]
    kappa = np.concatenate([pieces[0]] + [piece[1:] for piece in pieces[1:]])
    delta_heading = 0.5 * (kappa[:-1] + kappa[1:]) * lengths
    heading = np.concatenate(([0.0], np.cumsum(delta_heading)))
    mid_heading = heading[:-1] + delta_heading / 2
    return {
        "arc": np.concatenate(([0.0], np.cumsum(lengths))),
        "x": np.concatenate(([0.0], np.cumsum(np.cos(mid_heading) * lengths))),
        "y": np.concatenate(([0.0], np.cumsum(np.sin(mid_heading) * lengths))),
        "heading": heading,
        "kappa": kappa,
    }



# 构建整数峰值对应的无直行路径；杆心距取半周期的真实纵向位移
@lru_cache(maxsize=len(SLALOM_CURVATURES))
def get_slalom_geometry(curvature: int) -> dict[str, Any]:
    if curvature not in SLALOM_CURVATURES:
        raise ValueError(f"绕杆峰值曲率必须是 {SLALOM_CURVATURES[0]}～{SLALOM_CURVATURES[-1]} 的整数")
    k = float(curvature)
    transition = SMOOTH_TIME * SMOOTH_VEL
    platform = np.pi / (2 * k) - transition / 2
    period = integrate_segments([
        (platform, -k, -k), (transition, -k, 0.0),
        (transition, 0.0, k), (platform, k, k),
        (platform, k, k), (transition, k, 0.0),
        (transition, 0.0, -k), (platform, -k, -k),
    ])
    approach = integrate_segments([(transition, 0.0, -k), (APPROACH_LENGTH - transition, -k, -k)])
    approach["heading"] -= approach["heading"][-1]
    ds = np.diff(approach["arc"])
    mid_heading = (approach["heading"][:-1] + approach["heading"][1:]) / 2
    approach["x"] = np.concatenate(([0.0], np.cumsum(np.cos(mid_heading) * ds)))
    approach["y"] = np.concatenate(([0.0], np.cumsum(np.sin(mid_heading) * ds)))
    spacing = float(period["x"][-1] / 2)
    approach["x"] += spacing - approach["x"][-1]
    approach["y"] -= approach["y"][-1]
    return {"curvature": curvature, "spacing": spacing,
            "pole_y": float((period["y"].min() + period["y"].max()) / 2),
            "period": period, "approach": approach}



# 将指定杆距映射到最近的整数曲率条件，实际杆距必须向调用者回显
def curvature_from_spacing(spacing: float) -> int:
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("杆心距必须是有限正数")
    return min(SLALOM_CURVATURES, key=lambda k: abs(get_slalom_geometry(k)["spacing"] - spacing))



# 按弧长生成接近段和两个周期，供身体包络查询实际空间轨迹
@lru_cache(maxsize=1)
def get_spatial_slalom_tables() -> np.ndarray:
    tables = []
    for curvature in SLALOM_CURVATURES:
        geometry = get_slalom_geometry(curvature)
        pieces = []
        for path, count, offset in ((geometry["approach"], _PROJECTION_APPROACH_SAMPLES, 0.0),
                                    (geometry["period"], _PROJECTION_PERIOD_SAMPLES, geometry["spacing"]),
                                    (geometry["period"], _PROJECTION_PERIOD_SAMPLES, 3 * geometry["spacing"])):
            arc = np.linspace(0, path["arc"][-1], count + 1)
            piece = np.stack([np.interp(arc, path["arc"], path[name]) for name in ("x", "y", "heading", "kappa")], axis=-1)
            piece[:, 0] += offset
            pieces.append(piece if not pieces else piece[1:])
        tables.append(np.concatenate(pieces))
    return np.stack(tables)



# 将空间路径重采样到等名义时间网格，运行时只需按行插值
def timed_samples(path: dict[str, np.ndarray], base_velocity: float, count: int) -> tuple[np.ndarray, float]:
    velocity = base_velocity * get_velocity_scale(float(np.abs(path["kappa"]).max()))
    time = path["arc"] / velocity
    samples = np.stack([np.interp(np.linspace(0, time[-1], count), time, path[name]) for name in ("x", "y", "heading", "kappa")], axis=-1)
    return samples, float(time[-1])



# 预计算全部整数曲率，步频仅在运行时缩放时间
@lru_cache(maxsize=8)
def get_timed_slalom_tables(base_velocity: float) -> dict[str, np.ndarray]:
    if not np.isfinite(base_velocity) or base_velocity <= 0:
        raise ValueError("基础速度必须是有限正数")
    geometries = [get_slalom_geometry(k) for k in SLALOM_CURVATURES]
    periods = [timed_samples(g["period"], base_velocity, 8 * _SEGMENT_SAMPLES + 1) for g in geometries]
    approaches = [timed_samples(g["approach"], base_velocity, _APPROACH_SAMPLES + 1) for g in geometries]
    return {
        "period": np.stack([p[0] for p in periods]),
        "period_time": np.array([p[1] for p in periods]),
        "approach": np.stack([a[0] for a in approaches]),
        "approach_time": np.array([a[1] for a in approaches]),
        "spacing": np.array([g["spacing"] for g in geometries]),
        "pole_y": np.array([g["pole_y"] for g in geometries]),
        "start": np.array([[g["approach"][name][0] for name in ("x", "y", "heading")] for g in geometries]),
    }



# 初始接近段弧长 (m): 机器人从圆弧起点走 _INIT_DIST 到路径起点 (0,0)
_INIT_DIST = APPROACH_LENGTH



# 无直行路径族的最小和最大杆心距
def get_pole_spacing_bounds() -> tuple[float, float]:
    return (get_slalom_geometry(SLALOM_CURVATURE_MAX)["spacing"], get_slalom_geometry(SLALOM_CURVATURE_MIN)["spacing"])



# 回放未指定条件时采用峰值十五；训练按回合采样整数峰值
def get_curriculum_pole_spacing(step_counter: int) -> float:
    del step_counter
    return get_pole_spacing_bounds()[0]



# 躯干包络投影与边采样
_BOX_SIGNS = tuple(product((-1.0, 1.0), repeat=3))
_BOX_EDGES = tuple((i, j) for i in range(8) for j in range(i + 1, 8) if sum(a != b for a, b in zip(_BOX_SIGNS[i], _BOX_SIGNS[j], strict=True)) == 1)
_ENVELOPE_SAMPLE_STEP = 0.002



# 将完整三维包围盒随躯干实际姿态投影到水平面
def project_body_envelopes(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    local_vertices = getattr(env, "_slalom_body_envelope_vertices", None)
    if local_vertices is None:
        bounds = torch.as_tensor(resolve_body_envelope_bounds(robot, env.sim.mj_model), device=env.device, dtype=torch.float32)
        signs = torch.tensor(_BOX_SIGNS, device=env.device, dtype=bounds.dtype)
        center = bounds.mean(dim=1)
        half_size = (bounds[:, 1] - bounds[:, 0]) / 2
        local_vertices = center[:, None] + signs[None] * half_size[:, None]
        env._slalom_body_envelope_vertices = local_vertices  # type: ignore
    body_ids = (_MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id)
    positions = robot.data.body_link_pos_w[:, body_ids]
    rotations = matrix_from_quat(robot.data.body_link_quat_w[:, body_ids])
    vertices = torch.einsum("nbij,bvj->nbvi", rotations, local_vertices) + positions[:, :, None]
    return vertices[..., :2]



# 沿投影盒十二条边采样，边界相邻点间距不超过给定步长 (按步长分别缓存)
def sample_body_envelopes(env: ManagerBasedRlEnv, vertices: torch.Tensor, step: float = _ENVELOPE_SAMPLE_STEP) -> torch.Tensor:
    cache = getattr(env, "_slalom_envelope_samples", None)
    if cache is None:
        cache = {}
        env._slalom_envelope_samples = cache  # type: ignore
    cached = cache.get(float(step))
    if cached is None:
        local = env._slalom_body_envelope_vertices  # type: ignore
        starts, ends, fractions = list(range(8)), list(range(8)), [0.0] * 8
        for start, end in _BOX_EDGES:
            count = max(1, int(torch.ceil(torch.linalg.vector_norm(local[:, end] - local[:, start], dim=-1).max() / step).item()))
            starts.extend([start] * (count - 1))
            ends.extend([end] * (count - 1))
            fractions.extend(i / count for i in range(1, count))
        cached = (torch.tensor(starts, device=env.device), torch.tensor(ends, device=env.device), torch.tensor(fractions, device=env.device, dtype=vertices.dtype))
        cache[float(step)] = cached
    starts, ends, fractions = cached
    return torch.lerp(vertices[:, :, starts], vertices[:, :, ends], fractions[None, None, :, None])
