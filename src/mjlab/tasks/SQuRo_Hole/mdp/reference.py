from __future__ import annotations
import math
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# 步态与高度配置
USE_SPINE_CSV = False                  # 脊柱 CSV 不在仓库内, 默认关闭
HEIGHT_LIST = [0.02, 0.04, 0.05, 0.06]
BASE_HEIGHT = 0.06

_BIO_DATA_DIR = Path(__file__).parent / "Bio_Data"

# 足端/脊柱轨迹参数 (相对肩/髋的足端坐标与旋转)
CSV_PARAMS = {
    "fps": 60,
    "foot_scale": 1.0,
    "x_offset_F": 0.00,
    "x_offset_H": -0.01,
    "z_offset_F": 0.00,
    "z_offset_H": 0.00,
    "rotate_angle_F": -0.1,
    "rotate_angle_H": -0.0,
    "xoy_angle_scale": 1.0,
    "yoz_angle_scale": -1.0,
    "phase_lag": {"FL": 0.0, "FR": 0.5, "HL": 0.5, "HR": 0.0},
}

# 单段低高度时的摆线参数 (低高度那一段的腿仍要在动)
CYCLOID_PARAMS = {
    "normal": {
        "freq": 2.0, "stride_F": 0.05, "stride_H": 0.05, "height_F": 0.01, "height_H": 0.015,
        "body_height_F": 0.06, "body_height_H": 0.057, "swing_ratio": 0.4,
        "rotate_angle_F": 0.0, "rotate_angle_H": 0.0, "x_offset_F": -0.01, "x_offset_H": -0.03,
        "phase_lag": {"FL": 0.0, "FR": 0.5, "HL": 0.5, "HR": 0.0},
    },
    "front_low": {
        "freq": 2.0, "stride_H": 0.04, "height_H": 0.005, "body_height_H": 0.05,
        "swing_ratio": 0.4, "rotate_angle_H": 0.3, "x_offset_H": -0.03,
        "phase_lag": {"FL": 0.0, "FR": 0.5, "HL": 0.5, "HR": 0.0},
    },
    "hind_low": {
        "freq": 2.0, "stride_F": 0.04, "height_F": 0.005, "body_height_F": 0.05,
        "swing_ratio": 0.4, "rotate_angle_F": -0.4, "x_offset_F": -0.02,
        "phase_lag": {"FL": 0.0, "FR": 0.5, "HL": 0.5, "HR": 0.0},
    },
}

# 关节列口径: 与 indices.py 的执行器顺序一致 (14 维: 参考表只有 12 个被控关节)
REF_FRONT_IDS = (4, 5, 6, 7)      # FL/FR shoulder, elbow
REF_HIND_IDS = (10, 11, 12, 13)   # HL/HR hip, knee
REF_SPINE_IDS = (0, 1, 8, 9)      # F_spine1, F_body, H_spine1, H_body
REF_NECK_IDS = (2, 3)             # Neck_yaw, Neck_pitch
ACTUATOR_NUM = len(REF_FRONT_IDS) + len(REF_HIND_IDS) + len(REF_SPINE_IDS)

# body 索引 (旧版口径硬编码; 经模型核对: F_body_Link=4, H_body_Link=24)
F_BODY_ID = 4
H_BODY_ID = 24

# 受限模式: 0 = 前后肢都高, 1 = 前肢低, 2 = 后肢低
NUM_MODES = 3

_TABLE_RESOLUTION = 500

_LEG_CSV_CACHE: Dict[str, Optional[Dict[str, Any]]] = {"front": None, "hind": None}
_SPINE_CSV_CACHE: Dict[str, Optional[Dict[str, Any]]] = {"xoy_spine": None, "yoz_spine": None}
_PRECOMPUTED_TABLES: Dict[str, Any] = {}
_IS_TABLE_INITIALIZED = False


# 高度缩放系数 (与 command.py 口径一致, 低高度时另行处理)
def get_height_scale_factor(target_height: float, base_height: float = BASE_HEIGHT) -> float:
    return target_height / base_height


# 读取腿部 CSV (仓库自带列名: Phase / Y_mean / Z_mean)
def Load_CSV_Leg(csv_path: Path, is_front: bool = True) -> Optional[Dict[str, Any]]:
    cache_key = "front" if is_front else "hind"
    if _LEG_CSV_CACHE[cache_key] is not None:
        return _LEG_CSV_CACHE[cache_key]

    if not csv_path.exists():
        print(f"[SQuRo Hole] 腿部 CSV 不存在: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    fps = CSV_PARAMS["fps"]
    time_data = np.arange(len(df)) / fps
    data = {
        "x_data": df["Y_mean"].values * CSV_PARAMS["foot_scale"],
        "z_data": df["Z_mean"].values * CSV_PARAMS["foot_scale"],
        "time_data": time_data,
        "T": time_data[-1] if len(time_data) > 0 else 0.5,
    }
    _LEG_CSV_CACHE[cache_key] = data
    return data


# 读取脊柱 CSV (可选)
def Load_CSV_Spine(csv_path: Path, spine_type: str) -> Optional[Dict[str, Any]]:
    cache_key = spine_type
    if _SPINE_CSV_CACHE[cache_key] is not None:
        return _SPINE_CSV_CACHE[cache_key]
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    time_data = np.arange(len(df)) / CSV_PARAMS["fps"]
    data = {
        "angle_data": np.radians(df["Angle_Deg"].values),
        "time_data": time_data,
        "T": time_data[-1] if len(time_data) > 0 else 0.5,
    }
    _SPINE_CSV_CACHE[cache_key] = data
    return data


# 腿部 CSV 轨迹: 低高度时冻结, 否则插值 + 高度缩放 + 旋转
def CSV_Leg_Trajectory(phases: np.ndarray, csv_data: Dict[str, Any], phase_lag: float,
                       rotate_angle: float, height_scale: float = 1.0,
                       x_offset: float = 0.0, z_offset: float = 0.0,
                       current_height: float = BASE_HEIGHT,
                       is_front: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if current_height < 0.04:
        x_val = 0.005 if is_front else 0.002
        return np.full_like(phases, x_val), np.full_like(phases, -0.02)

    t_time = ((phases + phase_lag) % 1.0) * csv_data["T"]
    x_raw = np.interp(t_time, csv_data["time_data"], csv_data["x_data"])
    z_raw = np.interp(t_time, csv_data["time_data"], csv_data["z_data"])

    x = (x_raw + x_offset) * height_scale
    z = (z_raw + z_offset) * height_scale

    cos_a, sin_a = math.cos(rotate_angle), math.sin(rotate_angle)
    x_rotated = x * cos_a - z * sin_a
    z_rotated = x * sin_a + z * cos_a
    return x_rotated, z_rotated


# 脊柱 CSV 轨迹插值
def CSV_Spine_Trajectory(phases: np.ndarray, spine_data: Dict[str, Any]) -> np.ndarray:
    t_time = (phases % 1.0) * spine_data["T"]
    return np.interp(t_time, spine_data["time_data"], spine_data["angle_data"])


# 摆线轨迹 (单段低高度时另一段腿的运动)
def Cycloid_Trajectory(t_mods: np.ndarray, T_sw: float, T_st: float, stride: float,
                       height: float, body_height: float, rotate_angle: float,
                       x_offset: float, height_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros_like(t_mods)
    z = np.zeros_like(t_mods)

    sw_mask = t_mods <= T_sw
    st_mask = ~sw_mask

    t_norm_sw = t_mods[sw_mask] / T_sw
    x[sw_mask] = stride * (t_norm_sw - (1 / (2 * np.pi)) * np.sin(2 * np.pi * t_norm_sw))
    z[sw_mask] = 0.5 * height * (1 - np.cos(2 * np.pi * t_norm_sw))

    t_st_norm = (t_mods[st_mask] - T_sw) / T_st
    x[st_mask] = stride * (1 - t_st_norm)
    z[st_mask] = 0.0

    cos_a, sin_a = np.cos(rotate_angle), np.sin(rotate_angle)
    x_rotated = x * cos_a - z * sin_a
    z_rotated = x * sin_a + z * cos_a

    x_leg = (x_rotated + x_offset) * height_scale
    z_leg = (z_rotated - body_height) * height_scale
    return x_leg, z_leg


# 逆运动学: 足端相对肩/髋的 (x, z) -> 关节角
def Inverse_Kinematics(x: torch.Tensor, y: torch.Tensor,
                       is_front: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    L1, L2 = (0.040, 0.040) if is_front else (0.040, 0.036)
    R = torch.sqrt(x ** 2 + y ** 2)
    K = (L2 ** 2 - x ** 2 - y ** 2 - L1 ** 2) / (2 * L1)
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
    knee = torch.where(a2 > 2, knee + 2 * math.pi, knee)
    return hip, knee


# 生成预计算表: [模式, 高度档, 相位, 4 关节] x3 (前腿/后腿/脊柱)
def Initialize_Tables_Hole(device: torch.device) -> Dict[str, Any]:
    front_trajectory = Load_CSV_Leg(_BIO_DATA_DIR / "Trot_F.csv", is_front=True)
    hind_trajectory = Load_CSV_Leg(_BIO_DATA_DIR / "Trot_H.csv", is_front=False)
    if front_trajectory is None or hind_trajectory is None:
        raise FileNotFoundError(
            f"腿部 CSV 缺失, 需要 {_BIO_DATA_DIR}/Trot_F.csv 与 Trot_H.csv")

    xoy_spine_data, yoz_spine_data = None, None
    if USE_SPINE_CSV:
        xoy_spine_data = Load_CSV_Spine(_BIO_DATA_DIR / "XoY_Spine_Smooth.csv", "xoy_spine")
        yoz_spine_data = Load_CSV_Spine(_BIO_DATA_DIR / "YoZ_Spine_Smooth.csv", "yoz_spine")

    front_pos_table = torch.zeros(NUM_MODES, len(HEIGHT_LIST), _TABLE_RESOLUTION, 4, device=device)
    hind_pos_table = torch.zeros(NUM_MODES, len(HEIGHT_LIST), _TABLE_RESOLUTION, 4, device=device)
    spine_pos_table = torch.zeros(NUM_MODES, len(HEIGHT_LIST), _TABLE_RESOLUTION, 4, device=device)

    phases_np = np.linspace(0, 1, _TABLE_RESOLUTION, endpoint=False)
    normal = CYCLOID_PARAMS["normal"]
    front_low = CYCLOID_PARAMS["front_low"]
    hind_low = CYCLOID_PARAMS["hind_low"]

    for height_idx, target_height in enumerate(HEIGHT_LIST):
        height_scale = get_height_scale_factor(target_height, BASE_HEIGHT)

        for mode in range(NUM_MODES):
            # 前腿: 模式1 冻结, 否则走 CSV 轨迹
            if mode == 1:
                x_fl, z_fl = CSV_Leg_Trajectory(phases_np, front_trajectory,
                                                CSV_PARAMS["phase_lag"]["FL"], 0.0,
                                                1.0, 0.0, 0.0, 0.02, True)
                x_fr, z_fr = CSV_Leg_Trajectory(phases_np, front_trajectory,
                                                CSV_PARAMS["phase_lag"]["FR"], 0.0,
                                                1.0, 0.0, 0.0, 0.02, True)
            else:
                x_fl, z_fl = CSV_Leg_Trajectory(
                    phases_np, front_trajectory, CSV_PARAMS["phase_lag"]["FL"],
                    CSV_PARAMS["rotate_angle_F"], height_scale,
                    CSV_PARAMS["x_offset_F"], CSV_PARAMS["z_offset_F"], target_height, True)
                x_fr, z_fr = CSV_Leg_Trajectory(
                    phases_np, front_trajectory, CSV_PARAMS["phase_lag"]["FR"],
                    CSV_PARAMS["rotate_angle_F"], height_scale,
                    CSV_PARAMS["x_offset_F"], CSV_PARAMS["z_offset_F"], target_height, True)

            sh_fl, el_fl = Inverse_Kinematics(torch.tensor(x_fl, dtype=torch.float32),
                                              torch.tensor(z_fl, dtype=torch.float32), True)
            sh_fr, el_fr = Inverse_Kinematics(torch.tensor(x_fr, dtype=torch.float32),
                                              torch.tensor(z_fr, dtype=torch.float32), True)
            front_pos_table[mode, height_idx, :, 0:2] = torch.stack([sh_fl, el_fl], dim=-1)
            front_pos_table[mode, height_idx, :, 2:4] = torch.stack([sh_fr, el_fr], dim=-1)

            # 后腿: 模式2 冻结, 模式1 用摆线, 否则走 CSV 轨迹
            if mode == 2:
                x_hl, z_hl = CSV_Leg_Trajectory(phases_np, hind_trajectory,
                                                CSV_PARAMS["phase_lag"]["HL"], 0.0,
                                                1.0, 0.0, 0.0, 0.02, False)
                x_hr, z_hr = CSV_Leg_Trajectory(phases_np, hind_trajectory,
                                                CSV_PARAMS["phase_lag"]["HR"], 0.0,
                                                1.0, 0.0, 0.0, 0.02, False)
            elif mode == 1:
                freq = front_low["freq"]
                period, t_sw = 1.0 / freq, (1.0 / freq) * front_low["swing_ratio"]
                t_mods = (phases_np % 1.0) * period
                x_hl, z_hl = Cycloid_Trajectory(t_mods, t_sw, period - t_sw,
                                                front_low["stride_H"], front_low["height_H"],
                                                front_low["body_height_H"],
                                                front_low["rotate_angle_H"],
                                                front_low["x_offset_H"], 1.0)
                x_hr, z_hr = Cycloid_Trajectory(t_mods, t_sw, period - t_sw,
                                                front_low["stride_H"], front_low["height_H"],
                                                front_low["body_height_H"],
                                                front_low["rotate_angle_H"],
                                                front_low["x_offset_H"], 1.0)
            else:
                x_hl, z_hl = CSV_Leg_Trajectory(
                    phases_np, hind_trajectory, CSV_PARAMS["phase_lag"]["HL"],
                    CSV_PARAMS["rotate_angle_H"], height_scale,
                    CSV_PARAMS["x_offset_H"], CSV_PARAMS["z_offset_H"], target_height, False)
                x_hr, z_hr = CSV_Leg_Trajectory(
                    phases_np, hind_trajectory, CSV_PARAMS["phase_lag"]["HR"],
                    CSV_PARAMS["rotate_angle_H"], height_scale,
                    CSV_PARAMS["x_offset_H"], CSV_PARAMS["z_offset_H"], target_height, False)

            hp_hl, kn_hl = Inverse_Kinematics(torch.tensor(x_hl, dtype=torch.float32),
                                              torch.tensor(z_hl, dtype=torch.float32), False)
            hp_hr, kn_hr = Inverse_Kinematics(torch.tensor(x_hr, dtype=torch.float32),
                                              torch.tensor(z_hr, dtype=torch.float32), False)
            hind_pos_table[mode, height_idx, :, 0:2] = torch.stack([hp_hl, kn_hl], dim=-1)
            hind_pos_table[mode, height_idx, :, 2:4] = torch.stack([hp_hr, kn_hr], dim=-1)

            # 脊柱: 低高度或单段低时固定弯曲; 否则用脊柱 CSV 或直立
            spine_angles = np.zeros((_TABLE_RESOLUTION, 4))
            if target_height < 0.04 or mode in (1, 2):
                spine_angles[:, 2] = -0.65
            elif USE_SPINE_CSV and xoy_spine_data is not None and yoz_spine_data is not None:
                spine_angles[:, 0] = CSV_Spine_Trajectory(phases_np, xoy_spine_data)
                spine_angles[:, 1] = CSV_Spine_Trajectory(phases_np, yoz_spine_data)
            spine_pos_table[mode, height_idx] = torch.tensor(spine_angles, dtype=torch.float32,
                                                             device=device)

    # 速度表: 沿相位中心差分 (相位周期 1, 步频在运行时乘)
    def _diff(table: torch.Tensor) -> torch.Tensor:
        dphi = 1.0 / _TABLE_RESOLUTION
        vel = torch.zeros_like(table)
        vel[..., 1:-1, :] = (table[..., 2:, :] - table[..., :-2, :]) / (2 * dphi)
        vel[..., 0, :] = (table[..., 1, :] - table[..., -1, :]) / (2 * dphi)
        vel[..., -1, :] = (table[..., 0, :] - table[..., -2, :]) / (2 * dphi)
        return vel

    tables = {
        "front_pos": front_pos_table,
        "hind_pos": hind_pos_table,
        "spine_pos": spine_pos_table,
        "front_vel": _diff(front_pos_table),
        "hind_vel": _diff(hind_pos_table),
        "spine_vel": _diff(spine_pos_table),
        "device": device,
    }
    print(f"[SQuRo Hole] 参考表生成完成: {NUM_MODES} 模式 × {len(HEIGHT_LIST)} 高度档 × "
          f"{_TABLE_RESOLUTION} 相位")
    return tables


# 惰性初始化入口
def Initialize_Tables(device: torch.device | str | None = None,
                      force_reload: bool = False) -> Dict[str, Any]:
    global _PRECOMPUTED_TABLES, _IS_TABLE_INITIALIZED

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    if _IS_TABLE_INITIALIZED and not force_reload:
        if _PRECOMPUTED_TABLES.get("device") == device:
            return _PRECOMPUTED_TABLES

    _PRECOMPUTED_TABLES = Initialize_Tables_Hole(device)
    _IS_TABLE_INITIALIZED = True
    return _PRECOMPUTED_TABLES


# 取当前 (模式, 相位) 的参考关节位置 (12 维, 顺序: 前腿/后腿/脊柱)
def get_reference_joint_pos(env: "ManagerBasedRlEnv", goal_height: float | None = None) -> torch.Tensor:
    return _get_reference_state(env, goal_height)[0]


# 取当前参考关节速度 (12 维)
def get_reference_joint_vel(env: "ManagerBasedRlEnv", goal_height: float | None = None) -> torch.Tensor:
    return _get_reference_state(env, goal_height)[1]


# 内部: 按 (模式, 高度档, 相位) 查表并拼成 12 维参考
def _get_reference_state(env: "ManagerBasedRlEnv",
                         goal_height: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if getattr(env, "_ref_phase", None) is None:
        env._ref_phase = torch.zeros(env.num_envs, device=env.device)  # type: ignore[attr-defined]

    tables = Initialize_Tables(env.device)
    device = env.device
    num_envs = env.num_envs

    cmd_term = env.command_manager._terms["hole_cmd"]  # type: ignore[union-attr]
    height_F = cmd_term.command[:, 3]
    height_H = cmd_term.command[:, 4]

    # 模式: 前低→1, 后低→2, 都高→0
    low_F = height_F < 0.04
    low_H = height_H < 0.04
    mode = torch.zeros(num_envs, dtype=torch.long, device=device)
    mode = torch.where(low_F & ~low_H, torch.ones_like(mode), mode)
    mode = torch.where(low_H & ~low_F, torch.full_like(mode, 2), mode)

    # 高度档: 低高度那一侧强制取最低档, 另一侧就近取档
    height_bins = torch.tensor(HEIGHT_LIST, device=device)
    eff_F = torch.where(low_F, torch.full_like(height_F, HEIGHT_LIST[0]), height_F)
    eff_H = torch.where(low_H, torch.full_like(height_H, HEIGHT_LIST[0]), height_H)
    idx_F = (eff_F.unsqueeze(1) - height_bins.unsqueeze(0)).abs().argmin(dim=1)
    idx_H = (eff_H.unsqueeze(1) - height_bins.unsqueeze(0)).abs().argmin(dim=1)
    # 脊柱档: 任一侧为低则取最低档
    idx_spine = torch.where(low_F | low_H, torch.zeros_like(idx_F), idx_F)

    phase = env._ref_phase  # type: ignore[attr-defined]
    phase_idx = (phase * _TABLE_RESOLUTION).long().clamp_(0, _TABLE_RESOLUTION - 1)

    front_pos = tables["front_pos"][mode, idx_F, phase_idx]     # [N, 4]
    hind_pos = tables["hind_pos"][mode, idx_H, phase_idx]
    spine_pos = tables["spine_pos"][mode, idx_spine, phase_idx]
    front_vel = tables["front_vel"][mode, idx_F, phase_idx]
    hind_vel = tables["hind_vel"][mode, idx_H, phase_idx]
    spine_vel = tables["spine_vel"][mode, idx_spine, phase_idx]

    ref_pos = torch.cat([front_pos, hind_pos, spine_pos], dim=1)
    ref_vel = torch.cat([front_vel, hind_vel, spine_vel], dim=1)

    # 相位推进: 2 Hz 名义步频 (与表内轨迹一致)
    gait_freq = 2.0
    env._ref_phase = (phase + gait_freq * float(env.step_dt)) % 1.0  # type: ignore[attr-defined]

    reset_ids = getattr(env, "reset_terminated", None)
    if reset_ids is not None:
        ids = reset_ids.nonzero(as_tuple=False).flatten()
        if len(ids) > 0:
            env._ref_phase[ids] = 0.0  # type: ignore[attr-defined]

    return ref_pos, ref_vel
