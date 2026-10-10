from __future__ import annotations
import math
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from .config import (
    BASE_HEIGHT,
    HEIGHT_LIST,
    THRESHOLD_HEIGHT,
    NECK_REF_POS,
    NECK_REF_VEL,
    TABLE_RESOLUTION as _TABLE_RESOLUTION,
    USE_SPINE_CSV,
)
from .indices import (
    _ACT_NECK_IDS,
    _MODEL_INDICES,
    REF_SLOT_OF_ACT,
    REF_TABLE_ORDER,
    resolve_model_indices,
)

# 生物数据随任务打包 (原版为开发机绝对路径, 迁移到仓库内相对路径)
_BIO_DATA_DIR = Path(__file__).resolve().parent / "Bio_Data"
CSV_PATHS = {
    "front_leg": str(_BIO_DATA_DIR / "FL_Smooth.csv"),
    "hind_leg": str(_BIO_DATA_DIR / "HR_Smooth.csv"),
    "xoy_spine": str(_BIO_DATA_DIR / "XoY_Spine_Smooth.csv"),
    "yoz_spine": str(_BIO_DATA_DIR / "YoZ_Spine_Smooth.csv"),
}

# 参数配置
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
    "height_scale_method": "z_offset"
}

# ==================== 摆线轨迹参数配置 ====================
CYCLOID_PARAMS = {
    "normal": {
        "freq": 2.0, "stride_F": 0.05, "stride_H": 0.05, "height_F": 0.01, "height_H": 0.015,
        "body_height_F": 0.06, "body_height_H": 0.057, "swing_ratio": 0.4, 
        "rotate_angle_F": 0.0, "rotate_angle_H": 0.0, "x_offset_F": -0.01, "x_offset_H": -0.03,
        "phase_lag": {"FL": 0.0, "FR": 0.5, "HL": 0.5, "HR": 0.0}
    },
    "front_low": {
        "freq": 2.0, "stride_H": 0.04, "height_H": 0.005, "body_height_H": 0.05,
        "swing_ratio": 0.4, "rotate_angle_H": 0.3, "x_offset_H": -0.03,
        "phase_lag": {"FL": 0.0, "FR": 0.5, "HL": 0.5, "HR": 0.0}
    },
    "hind_low": {
        "freq": 2.0, "stride_F": 0.04, "height_F": 0.005, "body_height_F": 0.05,
        "swing_ratio": 0.4, "rotate_angle_F": -0.4, "x_offset_F": -0.02,
        "phase_lag": {"FL": 0.0, "FR": 0.5, "HL": 0.5, "HR": 0.0}
    }
}

# 参考表列序: REF_TABLE_ORDER 里的数字是**执行器序**的下标, 不是模型关节索引,
# 必须先用 indices.py 解析出模型关节索引再按该顺序排列 (顺序: 前腿4 + 后腿4 + 脊柱4 + 头颈2)
MODEL_JOINT_IDS: list = []
NECK_IDS = [4, 5]                                           # 模型关节索引: Neck_yaw, Neck_pitch
ACTUATOR_NUM = len(REF_TABLE_ORDER)                         # 被控关节数 (14)


# 解析参考表列序对应的模型关节索引 (幂等)
def resolve_joint_ids(entity) -> list:
    global MODEL_JOINT_IDS, NECK_IDS
    if MODEL_JOINT_IDS:
        return MODEL_JOINT_IDS
    resolve_model_indices(entity)
    act = _MODEL_INDICES.joint_ids
    MODEL_JOINT_IDS = [act[i] for i in REF_TABLE_ORDER]
    NECK_IDS = [act[i] for i in _ACT_NECK_IDS]
    return MODEL_JOINT_IDS


_LEG_CSV_CACHE: Dict[str, Optional[Dict[str, Any]]] = {"front": None, "hind": None}
_SPINE_CSV_CACHE: Dict[str, Optional[Dict[str, Any]]] = {"xoy_spine": None, "yoz_spine": None}
_PRECOMPUTED_TABLES: Dict[str, Any] = {}
_IS_TABLE_INITIALIZED = False


# 根据目标高度计算缩放因子
def get_height_scale_factor(target_height: float, base_height: float = BASE_HEIGHT) -> float:
    return target_height / base_height

# 加载CSV数据
def Load_CSV_Leg(csv_path: str, is_front: bool = True) -> Optional[Dict[str, Any]]:
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

def Load_CSV_Spine(csv_path: str, spine_type: str) -> Optional[Dict[str, Any]]:
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


# 向量化CSV腿部轨迹
def CSV_Leg_Trajectory(phases: np.ndarray, csv_data: Dict[str, Any], phase_lag: float, 
                                  rotate_angle: float, height_scale: float = 1.0, 
                                  x_offset: float = 0.0, z_offset: float = 0.0,
                                  current_height: float = BASE_HEIGHT,
                                  is_front: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if current_height < THRESHOLD_HEIGHT:
        x_val = 0.005 if is_front else 0.002
        return np.full_like(phases, x_val), np.full_like(phases, -0.02)
    
    t_time = ((phases + phase_lag) % 1.0) * csv_data["T"]
    
    # 利用 np.interp 进行高速批量一维插值
    x_raw = np.interp(t_time, csv_data["time_data"], csv_data["x_data"])
    z_raw = np.interp(t_time, csv_data["time_data"], csv_data["z_data"])
    
    x = (x_raw + x_offset) * height_scale
    z = (z_raw + z_offset) * height_scale
    
    cos_a, sin_a = math.cos(rotate_angle), math.sin(rotate_angle)
    x_rotated = x * cos_a - z * sin_a
    z_rotated = x * sin_a + z * cos_a
    
    return x_rotated, z_rotated


# 向量化CSV脊柱轨迹
def CSV_Spine_Trajectory(phases: np.ndarray, spine_data: Dict[str, Any]) -> np.ndarray:
    t_time = (phases % 1.0) * spine_data["T"]
    return np.interp(t_time, spine_data["time_data"], spine_data["angle_data"])


# 向量化摆线轨迹
def Cycloid_Trajectory(t_mods: np.ndarray, T_sw: float, T_st: float, stride: float, 
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


# 逆运动学求解器
def Inverse_Kinematics(x: torch.Tensor, y: torch.Tensor, is_front: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    L1, L2 = (0.040, 0.040) if is_front else (0.040, 0.036)
    
    R = torch.sqrt(x**2 + y**2)
    K = (L2**2 - x**2 - y**2 - L1**2) / (2 * L1)
    theta = torch.atan2(y, x)
    K_over_R = torch.clamp(K / R, -1.0, 1.0)
    phi = torch.acos(K_over_R)
    
    if is_front:
        a1 = theta + phi
        sin_a1 = torch.sin(a1)
        cos_a1 = torch.cos(a1)
        sin_a2 = (y + L1 * sin_a1) / L2
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
        cos_a2 = (y + L1 * sin_a1) / L2
        a2 = torch.atan2(sin_a2, cos_a2)
        hip_angle = a1 + 4.35
        knee_angle = -(a2 + 1.75)
        knee_angle = torch.where(a2 > 2, knee_angle + 2 * torch.pi, knee_angle)
        return hip_angle, knee_angle


# 初始化预计算表
def Initialize_Tables(device=None, force_reload=False) -> Dict[str, Any]:
    global _PRECOMPUTED_TABLES, _IS_TABLE_INITIALIZED
    
    if _IS_TABLE_INITIALIZED and not force_reload:
        if "device" in _PRECOMPUTED_TABLES and _PRECOMPUTED_TABLES["device"] == device:
            return _PRECOMPUTED_TABLES
            
    # 原版忽略传入的 device 而硬编码 cuda:0, 只能在 GPU 上跑; 这里改为尊重调用方设备
    target = torch.device(device) if device is not None else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    tables = Initialize_Tables_Hole(target)
    _PRECOMPUTED_TABLES = tables
    _IS_TABLE_INITIALIZED = True
    return tables


# 初始化钻洞用预计算表
def Initialize_Tables_Hole(device: torch.device) -> Dict[str, Any]:
    front_trajectory = Load_CSV_Leg(CSV_PATHS["front_leg"], is_front=True)
    hind_trajectory = Load_CSV_Leg(CSV_PATHS["hind_leg"], is_front=False)
    
    if front_trajectory is None or hind_trajectory is None:
        raise FileNotFoundError("错误: 无法加载腿部轨迹数据，请检查CSV文件路径")
    
    xoy_spine_data, yoz_spine_data = None, None
    if USE_SPINE_CSV:
        xoy_spine_data = Load_CSV_Spine(CSV_PATHS["xoy_spine"], "xoy_spine")
        yoz_spine_data = Load_CSV_Spine(CSV_PATHS["yoz_spine"], "yoz_spine")
    
    NUM_MODES = 3
    front_pos_table = torch.zeros(NUM_MODES, len(HEIGHT_LIST), _TABLE_RESOLUTION, ACTUATOR_NUM, device=device)
    hind_pos_table = torch.zeros(NUM_MODES, len(HEIGHT_LIST), _TABLE_RESOLUTION, ACTUATOR_NUM, device=device)
    spine_pos_table = torch.zeros(NUM_MODES, len(HEIGHT_LIST), _TABLE_RESOLUTION, ACTUATOR_NUM, device=device)
    
    cycloid_front_low = CYCLOID_PARAMS["front_low"]
    cycloid_hind_low = CYCLOID_PARAMS["hind_low"]
    
    # 将相位数组预先准备好，消除最内层循环
    phases_np = np.linspace(0, 1, _TABLE_RESOLUTION)
    
    for height_idx, target_height in enumerate(HEIGHT_LIST):
        height_scale = get_height_scale_factor(target_height, BASE_HEIGHT)
        
        for mode in range(NUM_MODES):
            # ===== 1. 前肢计算 =====
            if mode == 1:  
                x_leg_fl, z_leg_fl = np.full(_TABLE_RESOLUTION, 0.005), np.full(_TABLE_RESOLUTION, -0.02)
                x_leg_fr, z_leg_fr = np.full(_TABLE_RESOLUTION, 0.005), np.full(_TABLE_RESOLUTION, -0.02)
            elif mode == 2:  
                freq = cycloid_hind_low["freq"]
                T, T_sw = 1.0 / freq, (1.0 / freq) * cycloid_hind_low["swing_ratio"]
                t_mods = (phases_np * T + CSV_PARAMS["phase_lag"]["FL"] * T) % T
                x_leg_fl, z_leg_fl = Cycloid_Trajectory(
                    t_mods, T_sw, T - T_sw, cycloid_hind_low["stride_F"], cycloid_hind_low["height_F"],
                    cycloid_hind_low["body_height_F"], cycloid_hind_low["rotate_angle_F"], cycloid_hind_low["x_offset_F"], 1
                )
                t_mods_fr = (phases_np * T + CSV_PARAMS["phase_lag"]["FR"] * T) % T
                x_leg_fr, z_leg_fr = Cycloid_Trajectory(
                    t_mods_fr, T_sw, T - T_sw, cycloid_hind_low["stride_F"], cycloid_hind_low["height_F"],
                    cycloid_hind_low["body_height_F"], cycloid_hind_low["rotate_angle_F"], cycloid_hind_low["x_offset_F"], 1
                )
            else:  
                x_leg_fl, z_leg_fl = CSV_Leg_Trajectory(
                    phases_np, front_trajectory, CSV_PARAMS["phase_lag"]["FL"], CSV_PARAMS["rotate_angle_F"], 
                    height_scale, CSV_PARAMS["x_offset_F"], CSV_PARAMS["z_offset_F"], target_height, True
                )
                x_leg_fr, z_leg_fr = CSV_Leg_Trajectory(
                    phases_np, front_trajectory, CSV_PARAMS["phase_lag"]["FR"], CSV_PARAMS["rotate_angle_F"], 
                    height_scale, CSV_PARAMS["x_offset_F"], CSV_PARAMS["z_offset_F"], target_height, True
                )
            
            # 张量堆叠，直接交给逆运动学广播计算 [200, 2]
            x_tensor_F = torch.tensor(np.stack([x_leg_fl, x_leg_fr], axis=-1), device=device, dtype=torch.float32)
            z_tensor_F = torch.tensor(np.stack([z_leg_fl, z_leg_fr], axis=-1), device=device, dtype=torch.float32)
            shoulder_angles, elbow_angles = Inverse_Kinematics(x_tensor_F, z_tensor_F, is_front=True)
            
            front_pos_table[mode, height_idx, :, 0:2] = torch.stack([shoulder_angles[:, 0], elbow_angles[:, 0]], dim=-1)
            front_pos_table[mode, height_idx, :, 2:4] = torch.stack([shoulder_angles[:, 1], elbow_angles[:, 1]], dim=-1)

            # ===== 2. 后肢计算 =====
            if mode == 1:  
                freq = cycloid_front_low["freq"]
                T, T_sw = 1.0 / freq, (1.0 / freq) * cycloid_front_low["swing_ratio"]
                t_mods = (phases_np * T + CSV_PARAMS["phase_lag"]["HL"] * T) % T
                x_leg_hl, z_leg_hl = Cycloid_Trajectory(
                    t_mods, T_sw, T - T_sw, cycloid_front_low["stride_H"], cycloid_front_low["height_H"],
                    cycloid_front_low["body_height_H"], cycloid_front_low["rotate_angle_H"], cycloid_front_low["x_offset_H"], 1
                )
                t_mods_hr = (phases_np * T + CSV_PARAMS["phase_lag"]["HR"] * T) % T
                x_leg_hr, z_leg_hr = Cycloid_Trajectory(
                    t_mods_hr, T_sw, T - T_sw, cycloid_front_low["stride_H"], cycloid_front_low["height_H"],
                    cycloid_front_low["body_height_H"], cycloid_front_low["rotate_angle_H"], cycloid_front_low["x_offset_H"], 1
                )
            elif mode == 2:  
                x_leg_hl, z_leg_hl = np.full(_TABLE_RESOLUTION, 0.002), np.full(_TABLE_RESOLUTION, -0.02)
                x_leg_hr, z_leg_hr = np.full(_TABLE_RESOLUTION, 0.002), np.full(_TABLE_RESOLUTION, -0.02)
            else:  
                x_leg_hl, z_leg_hl = CSV_Leg_Trajectory(
                    phases_np, hind_trajectory, CSV_PARAMS["phase_lag"]["HL"], CSV_PARAMS["rotate_angle_H"], 
                    height_scale, CSV_PARAMS["x_offset_H"], CSV_PARAMS["z_offset_H"], target_height, False
                )
                x_leg_hr, z_leg_hr = CSV_Leg_Trajectory(
                    phases_np, hind_trajectory, CSV_PARAMS["phase_lag"]["HR"], CSV_PARAMS["rotate_angle_H"], 
                    height_scale, CSV_PARAMS["x_offset_H"], CSV_PARAMS["z_offset_H"], target_height, False
                )
            
            x_tensor_H = torch.tensor(np.stack([x_leg_hl, x_leg_hr], axis=-1), device=device, dtype=torch.float32)
            z_tensor_H = torch.tensor(np.stack([z_leg_hl, z_leg_hr], axis=-1), device=device, dtype=torch.float32)
            hip_angles, knee_angles = Inverse_Kinematics(x_tensor_H, z_tensor_H, is_front=False)
            
            hind_pos_table[mode, height_idx, :, 4:6] = torch.stack([hip_angles[:, 0], knee_angles[:, 0]], dim=-1)
            hind_pos_table[mode, height_idx, :, 6:8] = torch.stack([hip_angles[:, 1], knee_angles[:, 1]], dim=-1)

            is_low_height = target_height < THRESHOLD_HEIGHT

            if is_low_height or mode in [1, 2]:
                # 低高度模式：固定脊柱姿态
                spine_angles = np.zeros((_TABLE_RESOLUTION, 4))
                spine_angles[:, 2] = -0.65  # 后脊柱关节固定弯曲
                spine_tensor = torch.tensor(spine_angles, device=device, dtype=torch.float32)
            elif USE_SPINE_CSV and xoy_spine_data is not None and yoz_spine_data is not None:
                # 正常模式：使用CSV脊柱数据
                xoy_angles = CSV_Spine_Trajectory(phases_np, xoy_spine_data)
                yoz_angles = CSV_Spine_Trajectory(phases_np, yoz_spine_data)
                spine_tensor = torch.tensor(
                    np.stack([xoy_angles, yoz_angles, xoy_angles, yoz_angles], axis=-1),
                    device=device, dtype=torch.float32
                )
            else:
                spine_angles = np.zeros((_TABLE_RESOLUTION, 4))
                spine_tensor = torch.tensor(spine_angles, device=device, dtype=torch.float32)
                
            # 脊柱占 8:12, 头颈占 12:14 (头颈列保持 0, 期望位置与速度都是 0)
            spine_pos_table[mode, height_idx, :, 8:12] = spine_tensor
        
                
    # ===== 4. 计算速度张量 (基于差分，避免空表问题) =====
    mode_periods = torch.tensor([0.5, 0.5, 0.5], device=device)
    dt = (mode_periods / _TABLE_RESOLUTION).view(NUM_MODES, 1, 1, 1)

    front_vel_table = torch.zeros_like(front_pos_table)
    front_vel_table[:, :, :-1, :] = (front_pos_table[:, :, 1:, :] - front_pos_table[:, :, :-1, :]) / dt
    front_vel_table[:, :, -1:, :] = (front_pos_table[:, :, 0:1, :] - front_pos_table[:, :, -1:, :]) / dt

    hind_vel_table = torch.zeros_like(hind_pos_table)
    hind_vel_table[:, :, :-1, :] = (hind_pos_table[:, :, 1:, :] - hind_pos_table[:, :, :-1, :]) / dt
    hind_vel_table[:, :, -1:, :] = (hind_pos_table[:, :, 0:1, :] - hind_pos_table[:, :, -1:, :]) / dt  

    spine_vel_table = torch.zeros_like(spine_pos_table)

    tables = {
        "front_pos": front_pos_table, "hind_pos": hind_pos_table, "spine_pos": spine_pos_table,
        "front_vel": front_vel_table, "hind_vel": hind_vel_table, "spine_vel": spine_vel_table,
        "phase_range": torch.linspace(0, 1, _TABLE_RESOLUTION, device=device),
        "height_list": HEIGHT_LIST, "device": device,
        "front_T": front_trajectory["T"], "hind_T": hind_trajectory["T"],
        "mode_periods": mode_periods,
    }
    return tables


# 头颈参考位置: 与模式/高度无关, 逐关节取 NECK_REF_POS (顺序 Neck_yaw, Neck_pitch)
def _neck_ref_pos(mode: torch.Tensor) -> torch.Tensor:
    values = torch.tensor(NECK_REF_POS, device=mode.device, dtype=torch.float32)
    return values.unsqueeze(0).expand(mode.shape[0], len(NECK_IDS))


# 头颈参考速度: 同位置, 逐关节取 NECK_REF_VEL
def _neck_ref_vel(mode: torch.Tensor) -> torch.Tensor:
    values = torch.tensor(NECK_REF_VEL, device=mode.device, dtype=torch.float32)
    return values.unsqueeze(0).expand(mode.shape[0], len(NECK_IDS))


# 获取参考状态
def get_reference_joint_state(env) -> tuple[torch.Tensor, torch.Tensor]:
    if not _IS_TABLE_INITIALIZED:
        Initialize_Tables(env.device)

    tables = _PRECOMPUTED_TABLES
    device = env.device

    cmd_term = env.command_manager._terms["hole_cmd"]
    desired_heightF = cmd_term.command[:, 3]
    desired_heightH = cmd_term.command[:, 4]

    current_time = env.episode_length_buf.float() * env.step_dt  # [num_envs]

    # 1. 计算 mode
    front_low = desired_heightF < THRESHOLD_HEIGHT
    hind_low = desired_heightH < THRESHOLD_HEIGHT

    mode = torch.zeros_like(desired_heightF, dtype=torch.long)
    mode[(front_low) & (~hind_low)] = 1
    mode[(~front_low) & (hind_low)] = 2

    # 2. 根据 mode 获取周期并计算相位
    mode_periods = tables["mode_periods"]  # [3]
    period = mode_periods[mode]

    phase = (current_time % period) / period
    phase_scaled = phase * (_TABLE_RESOLUTION - 1)
    phase_indices = phase_scaled.long().clamp(0, _TABLE_RESOLUTION - 1)

    # 3. 高度索引
    height_list = torch.tensor(HEIGHT_LIST, device=device)

    height_diffs_F = torch.abs(desired_heightF.unsqueeze(1) - height_list.unsqueeze(0))
    height_indices_F = torch.argmin(height_diffs_F, dim=1)

    height_diffs_H = torch.abs(desired_heightH.unsqueeze(1) - height_list.unsqueeze(0))
    height_indices_H = torch.argmin(height_diffs_H, dim=1)

    min_height = torch.minimum(desired_heightF, desired_heightH)
    height_diffs_min = torch.abs(min_height.unsqueeze(1) - height_list.unsqueeze(0))
    height_indices_min = torch.argmin(height_diffs_min, dim=1)

    # 4. 取参考位置
    front_pos = tables["front_pos"][mode, height_indices_F, phase_indices, 0:4]
    hind_pos = tables["hind_pos"][mode, height_indices_H, phase_indices, 4:8]
    spine_pos = tables["spine_pos"][mode, height_indices_min, phase_indices, 8:12]

    # 5. 取参考速度
    front_vel = tables["front_vel"][mode, height_indices_F, phase_indices, 0:4]
    hind_vel = tables["hind_vel"][mode, height_indices_H, phase_indices, 4:8]
    spine_vel = tables["spine_vel"][mode, height_indices_min, phase_indices, 8:12]

    # 6. 拼接头颈，头颈位置/速度恒定
    joint_pos = torch.cat(
        [front_pos, hind_pos, spine_pos, _neck_ref_pos(mode)],
        dim=1
    )
    joint_vel = torch.cat(
        [front_vel, hind_vel, spine_vel, _neck_ref_vel(mode)],
        dim=1
    )

    # 7. 表列序同步为执行器列序, 与 _MODEL_INDICES.joint_ids / 动作空间对齐 (见 indices.REF_SLOT_OF_ACT)
    return joint_pos[:, REF_SLOT_OF_ACT], joint_vel[:, REF_SLOT_OF_ACT]