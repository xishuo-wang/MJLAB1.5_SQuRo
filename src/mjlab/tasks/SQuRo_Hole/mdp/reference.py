from __future__ import annotations
import math
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from .config import (
    BASE_FREQ,
    BASE_HEIGHT,
    HEIGHT_LIST,
    NUM_MODES,
    NECK_REF_POS,
    HIND_HOLD_POS,
    MODE_HIND_LOW,
    MODE_BOTH_LOW,
    USE_SPINE_CSV,
    MODE_BOTH_HIGH,
    MODE_FRONT_LOW,
    LOW_TARGET_HEIGHT,
    CMD_HEIGHT_F_IDS,
    CMD_HEIGHT_H_IDS,
    THRESHOLD_HEIGHT,
    TABLE_RESOLUTION,
    CMD_GAIT_FREQ_IDS,
)
from .command import get_height_mode
from .indices import (
    _ACT_NUM,
    _ACT_SPN_IDS,
    _ACT_NECK_IDS,
    _ACT_LEG_F_IDS,
    _ACT_LEG_H_IDS,
    _ACT_FL_LEG_IDS,
    _ACT_FR_LEG_IDS,
    _ACT_HL_LEG_IDS,
    _ACT_HR_LEG_IDS,
    _ACT_SPN_F_YAW_ID,
    _ACT_SPN_F_ROLL_ID,
    _ACT_SPN_H_PITCH_ID,
    _ACT_SPN_H_ROLL_ID,
)



# 生物数据路径配置
_BIO_DATA_DIR = Path(__file__).resolve().parent / "Bio_Data"
CSV_PATHS = {
    "front_leg": str(_BIO_DATA_DIR / "FL_Smooth.csv"),
    "hind_leg": str(_BIO_DATA_DIR / "HR_Smooth.csv"),
    "xoy_spine": str(_BIO_DATA_DIR / "XoY_Spine_Smooth.csv"),
    "yoz_spine": str(_BIO_DATA_DIR / "YoZ_Spine_Smooth.csv"),
}


# CSV 轨迹参数配置
CSV_PARAMS = {
    "fps": 60,  
    "foot_scale": 1.0,
    "x_offset_F": 0.00,
    "x_offset_H": -0.01,
    "z_offset_F": 0.00,
    "z_offset_H": 0.00,
    "rotate_angle_F": -0.1,
    "rotate_angle_H": -0.0,               
    "phase_lag": {"FL": 0.0, "FR": 0.5, "HL": 0.5, "HR": 0.0},
}


# 摆线轨迹参数配置
CYCLOID_PARAMS = {
    "front_low": {
        "freq": BASE_FREQ, "stride_H": 0.04, "height_H": 0.005, "body_height_H": 0.05,
        "swing_ratio": 0.4, "rotate_angle_H": 0.3, "x_offset_H": -0.03,
    },
    "hind_low": {
        "freq": BASE_FREQ, "stride_F": 0.04, "height_F": 0.005, "body_height_F": 0.05,
        "swing_ratio": 0.4, "rotate_angle_F": -0.4, "x_offset_F": -0.02,
    }
}


# 预计算表初始化相关
_IS_TABLE_INITIALIZED = False
_PRECOMPUTED_TABLES: Dict[str, Any] = {}
_LEG_CSV_CACHE: Dict[str, Optional[Dict[str, Any]]] = {"front": None, "hind": None}
_SPINE_CSV_CACHE: Dict[str, Optional[Dict[str, Any]]] = {"xoy_spine": None, "yoz_spine": None}



# 计算目标高度相对基准高度的缩放因子
def get_height_scale_factor(target_height: float, base_height: float = BASE_HEIGHT) -> float:
    return target_height / base_height



# 加载腿部 CSV 数据
def load_csv_leg(csv_path: str, is_front: bool = True) -> Dict[str, Any]:
    cache_key = "front" if is_front else "hind"
    if _LEG_CSV_CACHE[cache_key] is not None:
        return _LEG_CSV_CACHE[cache_key]    
    
    df = pd.read_csv(csv_path)
    fps = CSV_PARAMS["fps"]
    time_data = np.arange(len(df)) / fps
    
    data = {
        "x_data": df['X'].values * CSV_PARAMS["foot_scale"],
        "z_data": df['Z'].values * CSV_PARAMS["foot_scale"],
        "time_data": time_data,
        "T": time_data[-1] if len(time_data) > 0 else 0.5
    }
    _LEG_CSV_CACHE[cache_key] = data
    return data



# 加载脊柱 CSV 数据
def load_csv_spine(csv_path: str, spine_type: str) -> Dict[str, Any]:
    cache_key = spine_type
    if _SPINE_CSV_CACHE[cache_key] is not None:
        return _SPINE_CSV_CACHE[cache_key]
    
    df = pd.read_csv(csv_path)
    time_data = np.arange(len(df)) / CSV_PARAMS["fps"]
    data = {
        "angle_data": np.radians(df['Angle_Deg'].values),
        "time_data": time_data,
        "T": time_data[-1] if len(time_data) > 0 else 0.5
    }
    _SPINE_CSV_CACHE[cache_key] = data
    return data



# 计算向量化 CSV 腿部轨迹
def csv_leg_trajectory(phases: np.ndarray, csv_data: Dict[str, Any], phase_lag: float, 
                                  rotate_angle: float, height_scale: float = 1.0, 
                                  x_offset: float = 0.0, z_offset: float = 0.0,
                                  current_height: float = BASE_HEIGHT,
                                  is_front: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if current_height < THRESHOLD_HEIGHT:
        x_val = 0.005 if is_front else 0.002
        return np.full_like(phases, x_val), np.full_like(phases, -0.02)
    
    t_time = ((phases + phase_lag) % 1.0) * csv_data["T"]
    
    # 使用 np.interp 进行批量一维插值
    x_raw = np.interp(t_time, csv_data["time_data"], csv_data["x_data"])
    z_raw = np.interp(t_time, csv_data["time_data"], csv_data["z_data"])
    
    x = (x_raw + x_offset) * height_scale
    z = (z_raw + z_offset) * height_scale
    
    cos_a, sin_a = math.cos(rotate_angle), math.sin(rotate_angle)
    x_rotated = x * cos_a - z * sin_a
    z_rotated = x * sin_a + z * cos_a
    
    return x_rotated, z_rotated



# 计算向量化 CSV 脊柱轨迹
def csv_spine_trajectory(phases: np.ndarray, spine_data: Dict[str, Any]) -> np.ndarray:
    t_time = (phases % 1.0) * spine_data["T"]
    return np.interp(t_time, spine_data["time_data"], spine_data["angle_data"])



# 计算向量化摆线轨迹
def cycloid_trajectory(t_mods: np.ndarray, T_sw: float, T_st: float, stride: float, 
                                  height: float, body_height: float, rotate_angle: float, 
                                  x_offset: float, height_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros_like(t_mods)
    z = np.zeros_like(t_mods)
    
    # 摆动相掩码
    sw_mask = t_mods <= T_sw
    st_mask = ~sw_mask
    
    # 计算摆动相
    t_norm_sw = t_mods[sw_mask] / T_sw
    x[sw_mask] = stride * (t_norm_sw - (1/(2*np.pi)) * np.sin(2*np.pi*t_norm_sw))
    z[sw_mask] = 0.5 * height * (1 - np.cos(2*np.pi*t_norm_sw))
    
    # 计算支撑相
    t_st_norm = (t_mods[st_mask] - T_sw) / T_st
    x[st_mask] = stride * (1 - t_st_norm)
    z[st_mask] = 0.0 
    
    cos_a, sin_a = np.cos(rotate_angle), np.sin(rotate_angle)
    x_rotated = x * cos_a - z * sin_a
    z_rotated = x * sin_a + z * cos_a

    x_leg = (x_rotated + x_offset) * height_scale
    z_leg = (z_rotated - body_height) * height_scale
    
    return x_leg, z_leg



# 计算腿部逆运动学关节角
def inverse_kinematics(x: torch.Tensor, z: torch.Tensor, is_front: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    L1, L2 = (0.040, 0.040) if is_front else (0.040, 0.036)
    
    R = torch.sqrt(x**2 + z**2)
    K = (L2**2 - x**2 - z**2 - L1**2) / (2 * L1)
    theta = torch.atan2(z, x)
    K_over_R = torch.clamp(K / R, -1.0, 1.0)
    phi = torch.acos(K_over_R)
    
    if is_front:
        a1 = theta + phi
        sin_a1 = torch.sin(a1)
        cos_a1 = torch.cos(a1)
        sin_a2 = (z + L1 * sin_a1) / L2
        cos_a2 = (-x - L1 * cos_a1) / L2
        a2 = torch.atan2(sin_a2, cos_a2)
        shoulder_angle = a1 - 0.868
        elbow_angle = -(a2 + 2.643)
        elbow_angle = torch.where(a2 > 2, elbow_angle + 2 * torch.pi, elbow_angle)
        return shoulder_angle, elbow_angle
    else:
        a1 = theta - phi
        sin_a1 = torch.sin(a1)
        cos_a1 = torch.cos(a1)
        sin_a2 = (x + L1 * cos_a1) / L2
        cos_a2 = (z + L1 * sin_a1) / L2
        a2 = torch.atan2(sin_a2, cos_a2)
        hip_angle = a1 + 4.35
        knee_angle = -(a2 + 1.75)
        knee_angle = torch.where(a2 > 2, knee_angle + 2 * torch.pi, knee_angle)
        return hip_angle, knee_angle



# 初始化预计算表
def initialize_tables(device=None, force_reload=False) -> Dict[str, Any]:
    global _PRECOMPUTED_TABLES, _IS_TABLE_INITIALIZED
    target = torch.device(device) if device is not None else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    if _IS_TABLE_INITIALIZED and not force_reload:
        if _PRECOMPUTED_TABLES.get("device") == target:
            return _PRECOMPUTED_TABLES
            
    tables = build_reference_tables(target)
    _PRECOMPUTED_TABLES = tables
    _IS_TABLE_INITIALIZED = True
    return tables



# 初始化钻洞用预计算表
def build_reference_tables(device: torch.device) -> Dict[str, Any]:
    front_trajectory = load_csv_leg(CSV_PATHS["front_leg"], is_front=True)
    hind_trajectory = load_csv_leg(CSV_PATHS["hind_leg"], is_front=False)
    
    xoy_spine_data, yoz_spine_data = None, None
    if USE_SPINE_CSV:
        xoy_spine_data = load_csv_spine(CSV_PATHS["xoy_spine"], "xoy_spine")
        yoz_spine_data = load_csv_spine(CSV_PATHS["yoz_spine"], "yoz_spine")
    
    front_pos_table = torch.zeros(NUM_MODES, len(HEIGHT_LIST), TABLE_RESOLUTION, _ACT_NUM, device=device)
    hind_pos_table = torch.zeros(NUM_MODES, len(HEIGHT_LIST), TABLE_RESOLUTION, _ACT_NUM, device=device)
    spine_pos_table = torch.zeros(NUM_MODES, len(HEIGHT_LIST), TABLE_RESOLUTION, _ACT_NUM, device=device)
    
    cycloid_front_low = CYCLOID_PARAMS["front_low"]
    cycloid_hind_low = CYCLOID_PARAMS["hind_low"]

    # 预生成相位数组，避免最内层循环重复创建
    phases_np = np.linspace(0, 1, TABLE_RESOLUTION)
    
    for height_idx, target_height in enumerate(HEIGHT_LIST):
        height_scale = get_height_scale_factor(target_height, BASE_HEIGHT)
        
        for mode in range(NUM_MODES):
            current_height = LOW_TARGET_HEIGHT if mode == MODE_BOTH_LOW else target_height
            # 前肢参考轨迹
            if mode == MODE_FRONT_LOW:
                x_leg_fl, z_leg_fl = np.full(TABLE_RESOLUTION, 0.005), np.full(TABLE_RESOLUTION, -0.02)
                x_leg_fr, z_leg_fr = np.full(TABLE_RESOLUTION, 0.005), np.full(TABLE_RESOLUTION, -0.02)
            elif mode == MODE_HIND_LOW:
                freq = cycloid_hind_low["freq"]
                T, T_sw = 1.0 / freq, (1.0 / freq) * cycloid_hind_low["swing_ratio"]
                t_mods = (phases_np * T + CSV_PARAMS["phase_lag"]["FL"] * T) % T
                x_leg_fl, z_leg_fl = cycloid_trajectory(
                    t_mods, T_sw, T - T_sw, cycloid_hind_low["stride_F"], cycloid_hind_low["height_F"],
                    cycloid_hind_low["body_height_F"], cycloid_hind_low["rotate_angle_F"], cycloid_hind_low["x_offset_F"], 1
                )
                t_mods_fr = (phases_np * T + CSV_PARAMS["phase_lag"]["FR"] * T) % T
                x_leg_fr, z_leg_fr = cycloid_trajectory(
                    t_mods_fr, T_sw, T - T_sw, cycloid_hind_low["stride_F"], cycloid_hind_low["height_F"],
                    cycloid_hind_low["body_height_F"], cycloid_hind_low["rotate_angle_F"], cycloid_hind_low["x_offset_F"], 1
                )
            else:  
                x_leg_fl, z_leg_fl = csv_leg_trajectory(
                    phases_np, front_trajectory, CSV_PARAMS["phase_lag"]["FL"], CSV_PARAMS["rotate_angle_F"], 
                    height_scale, CSV_PARAMS["x_offset_F"], CSV_PARAMS["z_offset_F"], current_height, True
                )
                x_leg_fr, z_leg_fr = csv_leg_trajectory(
                    phases_np, front_trajectory, CSV_PARAMS["phase_lag"]["FR"], CSV_PARAMS["rotate_angle_F"], 
                    height_scale, CSV_PARAMS["x_offset_F"], CSV_PARAMS["z_offset_F"], current_height, True
                )
            
            x_tensor_F = torch.tensor(np.stack([x_leg_fl, x_leg_fr], axis=-1), device=device, dtype=torch.float32)
            z_tensor_F = torch.tensor(np.stack([z_leg_fl, z_leg_fr], axis=-1), device=device, dtype=torch.float32)
            shoulder_angles, elbow_angles = inverse_kinematics(x_tensor_F, z_tensor_F, is_front=True)
            front_pos_table[mode, height_idx, :, _ACT_FL_LEG_IDS] = torch.stack([shoulder_angles[:, 0], elbow_angles[:, 0]], dim=-1)
            front_pos_table[mode, height_idx, :, _ACT_FR_LEG_IDS] = torch.stack([shoulder_angles[:, 1], elbow_angles[:, 1]], dim=-1)

            # 后肢参考轨迹
            if mode == MODE_FRONT_LOW:
                freq = cycloid_front_low["freq"]
                T, T_sw = 1.0 / freq, (1.0 / freq) * cycloid_front_low["swing_ratio"]
                t_mods = (phases_np * T + CSV_PARAMS["phase_lag"]["HL"] * T) % T
                x_leg_hl, z_leg_hl = cycloid_trajectory(
                    t_mods, T_sw, T - T_sw, cycloid_front_low["stride_H"], cycloid_front_low["height_H"],
                    cycloid_front_low["body_height_H"], cycloid_front_low["rotate_angle_H"], cycloid_front_low["x_offset_H"], 1
                )
                t_mods_hr = (phases_np * T + CSV_PARAMS["phase_lag"]["HR"] * T) % T
                x_leg_hr, z_leg_hr = cycloid_trajectory(
                    t_mods_hr, T_sw, T - T_sw, cycloid_front_low["stride_H"], cycloid_front_low["height_H"],
                    cycloid_front_low["body_height_H"], cycloid_front_low["rotate_angle_H"], cycloid_front_low["x_offset_H"], 1
                )
                x_tensor_H = torch.tensor(np.stack([x_leg_hl, x_leg_hr], axis=-1), device=device, dtype=torch.float32)
                z_tensor_H = torch.tensor(np.stack([z_leg_hl, z_leg_hr], axis=-1), device=device, dtype=torch.float32)
                hip_angles, knee_angles = inverse_kinematics(x_tensor_H, z_tensor_H, is_front=False)
                hind_pos_table[mode, height_idx, :, _ACT_HL_LEG_IDS] = torch.stack([hip_angles[:, 0], knee_angles[:, 0]], dim=-1)
                hind_pos_table[mode, height_idx, :, _ACT_HR_LEG_IDS] = torch.stack([hip_angles[:, 1], knee_angles[:, 1]], dim=-1)
            elif mode == MODE_HIND_LOW:
                hold = torch.tensor(HIND_HOLD_POS, device=device, dtype=hind_pos_table.dtype)
                hind_pos_table[mode, height_idx, :, _ACT_HL_LEG_IDS] = hold
                hind_pos_table[mode, height_idx, :, _ACT_HR_LEG_IDS] = hold
            else:
                x_leg_hl, z_leg_hl = csv_leg_trajectory(
                    phases_np, hind_trajectory, CSV_PARAMS["phase_lag"]["HL"], CSV_PARAMS["rotate_angle_H"],
                    height_scale, CSV_PARAMS["x_offset_H"], CSV_PARAMS["z_offset_H"], current_height, False
                )
                x_leg_hr, z_leg_hr = csv_leg_trajectory(
                    phases_np, hind_trajectory, CSV_PARAMS["phase_lag"]["HR"], CSV_PARAMS["rotate_angle_H"],
                    height_scale, CSV_PARAMS["x_offset_H"], CSV_PARAMS["z_offset_H"], current_height, False
                )
                x_tensor_H = torch.tensor(np.stack([x_leg_hl, x_leg_hr], axis=-1), device=device, dtype=torch.float32)
                z_tensor_H = torch.tensor(np.stack([z_leg_hl, z_leg_hr], axis=-1), device=device, dtype=torch.float32)
                hip_angles, knee_angles = inverse_kinematics(x_tensor_H, z_tensor_H, is_front=False)
                hind_pos_table[mode, height_idx, :, _ACT_HL_LEG_IDS] = torch.stack([hip_angles[:, 0], knee_angles[:, 0]], dim=-1)
                hind_pos_table[mode, height_idx, :, _ACT_HR_LEG_IDS] = torch.stack([hip_angles[:, 1], knee_angles[:, 1]], dim=-1)

            # 脊柱参考轨迹: 整行按执行器列写入, 与表列序一致, 无需再散射
            if mode != MODE_BOTH_HIGH:
                # 低高度模式: 固定后脊柱俯仰, 其余列保持 0
                spine_angles = np.zeros((TABLE_RESOLUTION, _ACT_NUM))
                spine_angles[:, _ACT_SPN_H_PITCH_ID] = -0.65
                spine_tensor = torch.tensor(spine_angles, device=device, dtype=torch.float32)
            elif USE_SPINE_CSV and xoy_spine_data is not None and yoz_spine_data is not None:
                # 正常模式: CSV 脊柱数据按执行器列写入 (前侧摆/前扭转/后俯仰/后扭转)
                spine_angles = np.zeros((TABLE_RESOLUTION, _ACT_NUM))
                spine_angles[:, _ACT_SPN_F_YAW_ID] = csv_spine_trajectory(phases_np, xoy_spine_data)
                spine_angles[:, _ACT_SPN_F_ROLL_ID] = csv_spine_trajectory(phases_np, yoz_spine_data)
                spine_angles[:, _ACT_SPN_H_PITCH_ID] = csv_spine_trajectory(phases_np, xoy_spine_data)
                spine_angles[:, _ACT_SPN_H_ROLL_ID] = csv_spine_trajectory(phases_np, yoz_spine_data)
                spine_tensor = torch.tensor(spine_angles, device=device, dtype=torch.float32)
            else:
                spine_tensor = torch.zeros((TABLE_RESOLUTION, _ACT_NUM), device=device, dtype=torch.float32)

            spine_pos_table[mode, height_idx] = spine_tensor
        
    # 根据位置表差分计算速度表
    mode_periods = torch.full((NUM_MODES,), 1.0 / BASE_FREQ, device=device)
    dt = (mode_periods / TABLE_RESOLUTION).view(NUM_MODES, 1, 1, 1)

    front_vel_table = torch.zeros_like(front_pos_table)
    front_vel_table[:, :, :-1, :] = (front_pos_table[:, :, 1:, :] - front_pos_table[:, :, :-1, :]) / dt
    front_vel_table[:, :, -1:, :] = (front_pos_table[:, :, 0:1, :] - front_pos_table[:, :, -1:, :]) / dt

    hind_vel_table = torch.zeros_like(hind_pos_table)
    hind_vel_table[:, :, :-1, :] = (hind_pos_table[:, :, 1:, :] - hind_pos_table[:, :, :-1, :]) / dt
    hind_vel_table[:, :, -1:, :] = (hind_pos_table[:, :, 0:1, :] - hind_pos_table[:, :, -1:, :]) / dt  
    hind_vel_table[MODE_HIND_LOW, :, :, _ACT_LEG_H_IDS] = 0.0

    spine_vel_table = torch.zeros_like(spine_pos_table)

    tables = {
        "front_pos": front_pos_table, "hind_pos": hind_pos_table, "spine_pos": spine_pos_table,
        "neck_pos": torch.tensor(NECK_REF_POS, device=device, dtype=front_pos_table.dtype),
        "front_vel": front_vel_table, "hind_vel": hind_vel_table, "spine_vel": spine_vel_table,
        "phase_range": torch.linspace(0, 1, TABLE_RESOLUTION, device=device),
        "height_list": HEIGHT_LIST, "device": device,
        "front_T": front_trajectory["T"], "hind_T": hind_trajectory["T"],
        "mode_periods": mode_periods,
    }
    return tables



# 获取参考状态
def get_reference_joint_state(env) -> tuple[torch.Tensor, torch.Tensor]:
    if not _IS_TABLE_INITIALIZED:
        initialize_tables(env.device)

    tables = _PRECOMPUTED_TABLES
    device = env.device

    cmd_term = env.command_manager._terms["hole_cmd"]
    desired_height_F = cmd_term.command[:, CMD_HEIGHT_F_IDS]
    desired_height_H = cmd_term.command[:, CMD_HEIGHT_H_IDS]
    current_time = env.episode_length_buf.float() * env.step_dt

    # 判定高度模式
    mode = get_height_mode(desired_height_F, desired_height_H)

    # 计算相位
    frequency = cmd_term.command[:, CMD_GAIT_FREQ_IDS]
    period = frequency.reciprocal()
    phase = (current_time % period) / period
    phase_scaled = phase * (TABLE_RESOLUTION - 1)
    phase_indices = phase_scaled.long().clamp(0, TABLE_RESOLUTION - 1)

    # 计算高度索引
    height_list = torch.tensor(HEIGHT_LIST, device=device)
    height_diffs_F = torch.abs(desired_height_F.unsqueeze(1) - height_list.unsqueeze(0))
    height_indices_F = torch.argmin(height_diffs_F, dim=1)
    height_diffs_H = torch.abs(desired_height_H.unsqueeze(1) - height_list.unsqueeze(0))
    height_indices_H = torch.argmin(height_diffs_H, dim=1)
    min_height = torch.minimum(desired_height_F, desired_height_H)
    height_diffs_min = torch.abs(min_height.unsqueeze(1) - height_list.unsqueeze(0))
    height_indices_min = torch.argmin(height_diffs_min, dim=1)

    # 获取参考位置
    joint_pos = torch.zeros((mode.shape[0], _ACT_NUM), device=device, dtype=tables["front_pos"].dtype)
    joint_vel = torch.zeros_like(joint_pos)
    joint_pos[:, _ACT_LEG_F_IDS] = tables["front_pos"][mode, height_indices_F, phase_indices][:, _ACT_LEG_F_IDS]
    joint_pos[:, _ACT_LEG_H_IDS] = tables["hind_pos"][mode, height_indices_H, phase_indices][:, _ACT_LEG_H_IDS]
    joint_pos[:, _ACT_SPN_IDS] = tables["spine_pos"][mode, height_indices_min, phase_indices][:, _ACT_SPN_IDS]
    joint_pos[:, _ACT_NECK_IDS] = tables["neck_pos"]

    # 获取参考速度
    joint_vel[:, _ACT_LEG_F_IDS] = tables["front_vel"][mode, height_indices_F, phase_indices][:, _ACT_LEG_F_IDS]
    joint_vel[:, _ACT_LEG_H_IDS] = tables["hind_vel"][mode, height_indices_H, phase_indices][:, _ACT_LEG_H_IDS]
    joint_vel[:, _ACT_SPN_IDS] = tables["spine_vel"][mode, height_indices_min, phase_indices][:, _ACT_SPN_IDS]
    joint_vel = joint_vel * (frequency / BASE_FREQ).unsqueeze(1)

    return joint_pos, joint_vel
