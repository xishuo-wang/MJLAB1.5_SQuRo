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


# 步态配置: 一个周期内四足的相位差 (trot)
_BIO_DATA_DIR = Path(__file__).parent / "Bio_Data"
PHASE_LAG = {"FL": 0.0, "FR": 0.5, "HL": 0.5, "HR": 0.0}

# 预计算表维度: [受限模式, 高度档, 相位, 关节]
NUM_MODES = 3                            # 0=都高 1=前低后高 2=前高后低
MODE_BOTH_HIGH = 0
MODE_FRONT_LOW = 1
MODE_REAR_LOW = 2
HEIGHT_LIST = [0.02, 0.04, 0.05, 0.06]  # 离散高度档 (m)
BASE_HEIGHT = 0.06                       # 高度缩放基准 (m)
HEIGHT_LOW_THRESHOLD = 0.04              # 低于此值该肢冻结不动
NUM_HEIGHTS = len(HEIGHT_LIST)

# 冻结姿态: 低高度肢体的足端相对偏移 (m), 不随高度缩放
FROZEN_FOOT_X_FRONT = 0.005
FROZEN_FOOT_X_REAR = 0.002
FROZEN_FOOT_Z = -0.02

# 足端轨迹水平偏移 (m)
X_OFFSET_FRONT = 0.0
X_OFFSET_REAR = -0.01

# 颈部参考 (非脊柱, 保持 Tunnel 原值)
NECK_PITCH_REF = -0.3

_TABLE_RESOLUTION = 200                  # 相位分辨率 (对齐 CSV 200 点)
_tables_initialized = False
_table_device: str | None = None
_height_bins: torch.Tensor | None = None      # [NUM_HEIGHTS]
_pos_table: torch.Tensor | None = None        # [NUM_MODES, NUM_HEIGHTS, 200, 14]
_vel_table: torch.Tensor | None = None        # 同上


# 高度对应的足端轨迹缩放系数 (基准高度处为 1)
def height_scale_for(target_height: float) -> float:
    return target_height / BASE_HEIGHT


# 该高度档下, 某条腿是否处于低高度 (冻结)
def is_low_height(target_height: float) -> bool:
    return target_height < HEIGHT_LOW_THRESHOLD


# 判断受限模式: 由前后肢高度命令决定
def mode_from_heights(height_f: torch.Tensor, height_h: torch.Tensor) -> torch.Tensor:
    low_f = height_f < HEIGHT_LOW_THRESHOLD
    low_h = height_h < HEIGHT_LOW_THRESHOLD
    mode = torch.full_like(height_f, MODE_BOTH_HIGH, dtype=torch.long)
    mode = torch.where(low_f & ~low_h, torch.full_like(mode, MODE_FRONT_LOW), mode)
    mode = torch.where(low_h & ~low_f, torch.full_like(mode, MODE_REAR_LOW), mode)
    return mode


# 单条腿的足端 IK 目标: 低高度冻结, 否则 CSV 轨迹的偏移量按高度等比缩放
def _leg_target(x_raw: np.ndarray, z_raw: np.ndarray, is_front: bool,
                target_height: float, frozen: bool) -> tuple[np.ndarray, np.ndarray]:
    if frozen:
        x_val = FROZEN_FOOT_X_FRONT if is_front else FROZEN_FOOT_X_REAR
        return np.full_like(x_raw, x_val), np.full_like(z_raw, FROZEN_FOOT_Z)
    scale = height_scale_for(target_height)
    x_off = X_OFFSET_FRONT if is_front else X_OFFSET_REAR
    return (x_raw + x_off) * scale, z_raw * scale


# 逆运动学: 足端相对肩/髋的 (x, z) -> 关节角
def _inverse_kinematics_leg(x: torch.Tensor, y: torch.Tensor, is_front: bool = True):
    L1, L2 = (0.040, 0.040) if is_front else (0.040, 0.036)
    R = torch.sqrt(x**2 + y**2)
    K = (L2**2 - x**2 - y**2 - L1**2) / (2 * L1)
    theta = torch.atan2(y, x)
    phi = torch.acos(torch.clamp(K / R, -1.0, 1.0))
    if is_front:
        a1 = theta + phi
        sin_a2 = (y + L1 * torch.sin(a1)) / L2
        cos_a2 = (-x - L1 * torch.cos(a1)) / L2
        a2 = torch.atan2(sin_a2, cos_a2)
        shoulder = a1 - 0.868
        elbow = -(a2 + 2.643)
        elbow = torch.where(a2 > 2, elbow + 2 * math.pi, elbow)
        return shoulder, elbow
    a1 = theta - phi
    sin_a2 = (x + L1 * torch.cos(a1)) / L2
    cos_a2 = (y + L1 * torch.sin(a1)) / L2
    a2 = torch.atan2(sin_a2, cos_a2)
    hip = a1 + 4.35
    knee = -(a2 + 1.75)
    return hip, knee


# 生成预计算表: 每个 (模式, 高度档) 一套 14 关节参考
def _init_tables(device: torch.device | str) -> None:
    global _tables_initialized, _pos_table, _vel_table, _height_bins, _table_device

    if _tables_initialized and _table_device == str(device):
        return

    df_f = pd.read_csv(_BIO_DATA_DIR / "Trot_F.csv")  # type: ignore[arg-type]
    df_h = pd.read_csv(_BIO_DATA_DIR / "Trot_H.csv")  # type: ignore[arg-type]
    phase_csv: np.ndarray = df_f["Phase"].values.astype(np.float64)  # type: ignore[union-attr]
    x_f_csv: np.ndarray = df_f["Y_mean"].values.astype(np.float64)  # type: ignore[union-attr]
    z_f_csv: np.ndarray = df_f["Z_mean"].values.astype(np.float64)  # type: ignore[union-attr]
    x_h_csv: np.ndarray = df_h["Y_mean"].values.astype(np.float64)  # type: ignore[union-attr]
    z_h_csv: np.ndarray = df_h["Z_mean"].values.astype(np.float64)  # type: ignore[union-attr]

    phases = np.linspace(0.0, 1.0, _TABLE_RESOLUTION, endpoint=False)
    x_f_grid = np.interp(phases, phase_csv, x_f_csv)
    z_f_grid = np.interp(phases, phase_csv, z_f_csv)
    x_h_grid = np.interp(phases, phase_csv, x_h_csv)
    z_h_grid = np.interp(phases, phase_csv, z_h_csv)

    dev = torch.device(device)  # type: ignore[arg-type]
    pos = torch.zeros(NUM_MODES, NUM_HEIGHTS, _TABLE_RESOLUTION, 14, device=dev)

    for mode in range(NUM_MODES):
        for h_idx, target_height in enumerate(HEIGHT_LIST):
            low = is_low_height(target_height)
            front_frozen = low or mode == MODE_FRONT_LOW
            rear_frozen = low or mode == MODE_REAR_LOW
            for name, is_front, frozen, x_raw, z_raw, ids in (
                ("FL", True, front_frozen, x_f_grid, z_f_grid, (4, 5)),
                ("FR", True, front_frozen, x_f_grid, z_f_grid, (6, 7)),
                ("HL", False, rear_frozen, x_h_grid, z_h_grid, (10, 11)),
                ("HR", False, rear_frozen, x_h_grid, z_h_grid, (12, 13)),
            ):
                x_leg, z_leg = _leg_target(x_raw, z_raw, is_front, target_height, frozen)
                shift = int(PHASE_LAG[name] * _TABLE_RESOLUTION)
                x_t = torch.tensor(np.roll(x_leg, shift), device=dev, dtype=torch.float32)
                z_t = torch.tensor(np.roll(z_leg, shift), device=dev, dtype=torch.float32)
                a1, a2 = _inverse_kinematics_leg(x_t, z_t, is_front)
                pos[mode, h_idx, :, ids[0]] = a1
                pos[mode, h_idx, :, ids[1]] = a2
            # 脊柱四关节静止 (高度靠腿与脊柱姿态区分), 颈部仅 pitch 固定
            pos[mode, h_idx, :, 3] = NECK_PITCH_REF

    # 中心差分得到速度表 (相位周期为 1)
    vel = torch.zeros_like(pos)
    dphi = 1.0 / _TABLE_RESOLUTION
    vel[..., 1:-1, :] = (pos[..., 2:, :] - pos[..., :-2, :]) / (2 * dphi)
    vel[..., 0, :] = (pos[..., 1, :] - pos[..., -1, :]) / (2 * dphi)
    vel[..., -1, :] = (pos[..., 0, :] - pos[..., -2, :]) / (2 * dphi)

    _pos_table = pos.contiguous()
    _vel_table = vel.contiguous()
    _height_bins = torch.tensor(HEIGHT_LIST, device=dev, dtype=torch.float32)
    _table_device = str(device)
    _tables_initialized = True

    print(f"\n[SQuRo Tunnel] 参考表生成完成: {NUM_MODES} 模式 × {NUM_HEIGHTS} 高度档 × "
          f"{_TABLE_RESOLUTION} 相位 × 14 关节")


# 取当前命令对应的 (模式, 前肢档, 后肢档)
def _resolve_mode_and_height(height_f: torch.Tensor, height_h: torch.Tensor,
                             height_bins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mode = mode_from_heights(height_f, height_h)
    # 低高度那一侧取最低档 (冻结姿态), 另一侧按自身高度就近取档
    eff_f = torch.where(mode == MODE_FRONT_LOW, torch.full_like(height_f, HEIGHT_LIST[0]), height_f)
    eff_h = torch.where(mode == MODE_REAR_LOW, torch.full_like(height_h, HEIGHT_LIST[0]), height_h)
    idx_f = (eff_f.unsqueeze(1) - height_bins.unsqueeze(0)).abs().argmin(dim=1)
    idx_h = (eff_h.unsqueeze(1) - height_bins.unsqueeze(0)).abs().argmin(dim=1)
    return mode, idx_f, idx_h


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

    if not _tables_initialized or _table_device != str(env.device):
        _init_tables(env.device)

    assert _pos_table is not None and _vel_table is not None and _height_bins is not None

    dt = float(env.step_dt)
    phase: torch.Tensor = env._ref_phase  # type: ignore[attr-defined]
    phase_indices = (phase * _TABLE_RESOLUTION).long().clamp_(0, _TABLE_RESOLUTION - 1)

    cmd_tensor = env.command_manager._terms["tunnel_cmd"].command  # type: ignore[union-attr]
    height_f = cmd_tensor[:, 1]
    height_h = cmd_tensor[:, 2]
    gait_freq = cmd_tensor[:, 3]

    mode, idx_f, idx_h = _resolve_mode_and_height(height_f, height_h, _height_bins)

    # 前肢关节用前肢档, 后肢关节用后肢档, 脊柱/颈取前肢档 (两者表内相同)
    pos_f = _pos_table[mode, idx_f, phase_indices]
    pos_h = _pos_table[mode, idx_h, phase_indices]
    vel_f = _vel_table[mode, idx_f, phase_indices]
    vel_h = _vel_table[mode, idx_h, phase_indices]
    rear_ids = [10, 11, 12, 13]
    ref_pos = pos_f.clone()
    ref_vel = vel_f.clone()
    ref_pos[:, rear_ids] = pos_h[:, rear_ids]
    ref_vel[:, rear_ids] = vel_h[:, rear_ids]
    # 参考速度按步频缩放 (相位推进用同一频率)
    ref_vel = ref_vel * gait_freq.unsqueeze(1)

    env._ref_phase = (phase + gait_freq * dt) % 1.0  # type: ignore[attr-defined]

    # 重置已终止环境
    reset_ids = getattr(env, "reset_terminated", None)
    if reset_ids is not None:
        ids = reset_ids.nonzero(as_tuple=False).flatten()
        if len(ids) > 0:
            env._ref_phase[ids] = 0.0  # type: ignore[attr-defined]

    env._ref_state_cache = (current_step, ref_pos, ref_vel)  # type: ignore[attr-defined]
    return ref_pos, ref_vel
