from __future__ import annotations
import math
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING
from .path import compute_path_ref, get_path_curvature
from .indices import (
    resolve_model_indices,
    _ACT_JOINT_NAMES,
    _ACT_F_SPINE1_ID,
    _ACT_F_BODY_ID,
    _ACT_NECK_YAW_ID,
    _ACT_NECK_PITCH_ID,
    _ACT_FL_SHOULDER_ID,
    _ACT_FL_ELBOW_ID,
    _ACT_FR_SHOULDER_ID,
    _ACT_FR_ELBOW_ID,
    _ACT_H_SPINE1_ID,
    _ACT_H_BODY_ID,
    _ACT_HL_HIP_ID,
    _ACT_HL_KNEE_ID,
    _ACT_HR_HIP_ID,
    _ACT_HR_KNEE_ID,
)
from .curriculums import BASE_VEL, CURVATURE_TARGET_MAX, REFERENCE_CURVATURE_BLEND, get_velocity_scale
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv



# 步态配置
_BIO_DATA_DIR = Path(__file__).parent / "Bio_Data"
PHASE_LAG = {"FL": 0.0, "FR": 0.5, "HL": 0.5, "HR": 0.0}    # 步态相位差
STRIDE_MIN = 0.0                                            # 最小步幅

# 参考限位 (rad): 取 SQuRo.xml 中关节 range 与执行器 ctrlrange 的较小者
F_SPINE1_LIMIT = 0.6
H_SPINE1_LIMIT = 0.6
NECK_YAW_LIMIT = 0.8
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
# 步幅放大上限: 参考不超过生物数据原幅值, 更快的命令由策略吸收
MAX_STRIDE_SCALE = 1.0


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

    # 双腿平均整周行程, 用于把步幅换算成每周前进距离
    stride_mean = 0.5 * (float(y_f_grid.max() - y_f_grid.min()) + float(y_h_grid.max() - y_h_grid.min()))

    # 转为 torch
    dev = torch.device(device)  # type: ignore[arg-type]
    y_f_t = torch.tensor(y_f_grid, device=dev, dtype=torch.float32)
    z_f_t = torch.tensor(z_f_grid, device=dev, dtype=torch.float32)
    y_h_t = torch.tensor(y_h_grid, device=dev, dtype=torch.float32)
    z_h_t = torch.tensor(z_h_grid, device=dev, dtype=torch.float32)

    # 五维表 [峰值档, 内侧, 瞬时档, 相位, 关节]: 峰值决定本回合速度, 瞬时决定左右差速
    # 每侧只缩放"内侧腿"的 Y_mean, 腿部相位不随曲率符号改变
    pos = torch.zeros(_NUM_CURV, 2, _NUM_CURV, _TABLE_RESOLUTION, len(_ACT_JOINT_NAMES), device=dev)

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
                shift = int(PHASE_LAG["FL"] * _TABLE_RESOLUTION)
                sh, el = _inverse_kinematics(torch.roll(y_f_t * row_scale * scale_fL, shifts=shift),
                                             torch.roll(z_f_t, shifts=shift), True)
                pos[p, side, i, :, _ACT_FL_SHOULDER_ID] = _clamp_leg_angle(sh, _ACT_FL_SHOULDER_ID)
                pos[p, side, i, :, _ACT_FL_ELBOW_ID] = _clamp_leg_angle(el, _ACT_FL_ELBOW_ID)

                shift = int(PHASE_LAG["FR"] * _TABLE_RESOLUTION)
                sh, el = _inverse_kinematics(torch.roll(y_f_t * row_scale * scale_fR, shifts=shift),
                                             torch.roll(z_f_t, shifts=shift), True)
                pos[p, side, i, :, _ACT_FR_SHOULDER_ID] = _clamp_leg_angle(sh, _ACT_FR_SHOULDER_ID)
                pos[p, side, i, :, _ACT_FR_ELBOW_ID] = _clamp_leg_angle(el, _ACT_FR_ELBOW_ID)

                shift = int(PHASE_LAG["HL"] * _TABLE_RESOLUTION)
                hp, kn = _inverse_kinematics(torch.roll(y_h_t * row_scale * scale_hL, shifts=shift),
                                             torch.roll(z_h_t, shifts=shift), False)
                pos[p, side, i, :, _ACT_HL_HIP_ID] = _clamp_leg_angle(hp, _ACT_HL_HIP_ID)
                pos[p, side, i, :, _ACT_HL_KNEE_ID] = _clamp_leg_angle(kn, _ACT_HL_KNEE_ID)

                shift = int(PHASE_LAG["HR"] * _TABLE_RESOLUTION)
                hp, kn = _inverse_kinematics(torch.roll(y_h_t * row_scale * scale_hR, shifts=shift),
                                             torch.roll(z_h_t, shifts=shift), False)
                pos[p, side, i, :, _ACT_HR_HIP_ID] = _clamp_leg_angle(hp, _ACT_HR_HIP_ID)
                pos[p, side, i, :, _ACT_HR_KNEE_ID] = _clamp_leg_angle(kn, _ACT_HR_KNEE_ID)

                # 脊柱四列保持零位（由 κ 解析驱动, 见 get_reference_joint_state）
                pos[p, side, i, :, _ACT_F_SPINE1_ID] = 0.0   # F_spine1
                pos[p, side, i, :, _ACT_F_BODY_ID] = 0.0   # F_body
                pos[p, side, i, :, _ACT_H_SPINE1_ID] = 0.0   # H_spine1
                pos[p, side, i, :, _ACT_H_BODY_ID] = 0.0   # H_body

    # 中心差分计算速度表（沿相位维度）
    vel = torch.zeros_like(pos)
    two_dt = 2.0 / _TABLE_RESOLUTION
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
          f"{_TABLE_RESOLUTION} bins × {len(_ACT_JOINT_NAMES)} joints, 基础速度 {base_velocity:.3f} m/s")



# 重置代次 +1: 使同一控制步内已缓存的参考失效 (框架在同一步内先算奖励、再重置、再算观测)
def invalidate_reference_cache(env: ManagerBasedRlEnv) -> None:
    env._ref_epoch = getattr(env, "_ref_epoch", 0) + 1  # type: ignore[attr-defined]


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
    phase_indices = (phase * (_TABLE_RESOLUTION - 1)).long().clamp_(0, _TABLE_RESOLUTION - 1)

    # 瞬时曲率与参考位置同源: 由本步的路径参考算出 (重置后 t=0, 故重置回合拿到接近段起点曲率)
    compute_path_ref(env)
    curvature_cmd = get_path_curvature(env)  # [N]
    cmd_tensor = env.command_manager._terms["slalom_cmd"].command  # type: ignore[union-attr]
    vel_cmd = cmd_tensor[:, 0]  # [N]
    gait_freq = cmd_tensor[:, 3]  # [N] — 动态步频
    kappa_norm = CURVATURE_TARGET_MAX

    # 曲率绝对值及插值因子（k_bins 在 _init_tables 时缓存）
    abs_k = curvature_cmd.abs()  # [N]
    assert _k_bins is not None

    # 本回合目标峰值: 二阶段取采样档位, 一阶段取恒定转弯命令
    command_term = env.command_manager._terms["slalom_cmd"]  # type: ignore[union-attr]
    phase1 = command_term.phase1_mask
    peak_k = torch.where(phase1, command_term.slalom_peak_curvature, abs_k)

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

    # 峰值轴: 一阶段的峰值随转弯命令变化, 该方向的变化同样计入速度
    peak_dot = torch.where(phase1, torch.zeros_like(kappa_dot), kappa_dot * torch.sign(curvature_cmd))
    d_pos_d_peak = ((1 - t.unsqueeze(1)) * (pos10 - pos00) + t.unsqueeze(1) * (pos11 - pos01)) \
        / (pk1 - pk0).unsqueeze(1)
    ref_vel = ref_vel + d_pos_d_peak * peak_dot.unsqueeze(1)

    # 动态覆盖脊柱侧摆参考（四关节线性映射，依据 ω_cmd）
    k_norm = curvature_cmd / kappa_norm          # 归一化曲率，范围 [-1, 1]
    smooth_magnitude, smooth_slope = smooth_curvature_magnitude(curvature_cmd)
    hind_magnitude = torch.where(phase1, smooth_magnitude, abs_k)
    hind_slope = torch.where(phase1, smooth_slope, curvature_cmd.sign())
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
    ref_vel[:, _ACT_H_SPINE1_ID] = torch.where(unclamped_h,
                                -0.65 * hind_slope * kappa_dot / kappa_norm,
                                zero_dot)
    ref_vel[:, _ACT_H_BODY_ID] = -0.7 * kappa_dot / kappa_norm
    # 相位推进 / κ 历史 / 重置清零每步只做一次: 重置后的重算不得二次推进或二次记录
    if getattr(env, "_ref_phase_step", None) != current_step:
        env._ref_kappa_prev = curvature_cmd.clone()  # type: ignore[attr-defined]
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
