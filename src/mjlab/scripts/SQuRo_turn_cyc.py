"""
SQuRo 原地转弯 — 力矩版参数迁移至 MJLAB（位置执行器，默认查看器）
完全复刻力矩版步态、脊柱与逆运动学
"""

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from dataclasses import dataclass
import tyro

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots.SQuRo.SQuRo_constants import get_spec, INIT_STATE

# ==================== 力矩版同款数学函数 ====================
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def cycloid_trajectory(t_mod, T_sw, T_st, S, H, body_height, rotation_angle, x_offset, is_front=True):
    if t_mod <= T_sw:
        t_norm = t_mod / T_sw
        x = S * (t_norm - (1.0 / (2.0 * np.pi)) * np.sin(2.0 * np.pi * t_norm))
        z = 0.5 * H * (1.0 - np.cos(2.0 * np.pi * t_norm))
    else:
        t_st = (t_mod - T_sw) / T_st
        x = S * (1.0 - t_st)
        z = 0.0
    x_leg = x + x_offset
    z_leg = z - body_height
    cos_r = np.cos(rotation_angle)
    sin_r = np.sin(rotation_angle)
    x_rot = x_leg * cos_r - z_leg * sin_r
    z_rot = x_leg * sin_r + z_leg * cos_r
    return x_rot, z_rot, x_rot, z_rot

def inverse_kinematics_old(target_pos, is_front=True):
    x, y = target_pos
    if is_front:
        L1, L2 = 0.040, 0.040
        R_val = np.sqrt(x**2 + y**2)
        if R_val < 1e-8:
            return None
        K = (L2**2 - x**2 - y**2 - L1**2) / (2 * L1)
        theta = np.arctan2(y, x)
        K_over_R = np.clip(K / R_val, -1.0, 1.0)
        phi = np.arccos(K_over_R)
        a1 = theta + phi
        sin_a2 = (y + L1 * np.sin(a1)) / L2
        cos_a2 = (-x - L1 * np.cos(a1)) / L2
        a2 = np.arctan2(sin_a2, cos_a2)
        if a2 > 2:
            a2 -= 2 * np.pi
        return (a1 - 0.888, -(a2 + 2.648))
    else:
        L1, L2 = 0.040, 0.036
        R_val = np.sqrt(x**2 + y**2)
        if R_val < 1e-8:
            return None
        K = (L2**2 - x**2 - y**2 - L1**2) / (2 * L1)
        theta = np.arctan2(y, x)
        K_over_R = np.clip(K / R_val, -1.0, 1.0)
        phi = np.arccos(K_over_R)
        a1 = theta - phi
        sin_a2 = (x + L1 * np.cos(a1)) / L2
        cos_a2 = (y + L1 * np.sin(a1)) / L2
        a2 = np.arctan2(sin_a2, cos_a2)
        if a2 > 2:
            a2 -= 2 * np.pi
        return (a1 + 4.325, -(a2 + 1.794))

# ==================== 力矩版转弯参数（完全复刻）====================
@dataclass
class TurnConfig:
    total_time: float = 20.0
    start_time: float = 0.5
    freq: float = 1.0               # 对应力矩版 FREQ=1.0
    output: str | None = None
    high_level_dt: float = 0.005

    @property
    def t_cycle(self) -> float:
        return 1.0 / self.freq

# 步态参数
FREQ = 1.0
T_CYCLE = 1.0 / FREQ
SWING_RATIO_L = 0.4
SWING_RATIO_R = 0.4

STRIDE_FL = 0.05
STRIDE_FR = -0.0
STRIDE_HL = 0.05
STRIDE_HR = -0.0

HEIGHT_FL = 0.01
HEIGHT_FR = 0.01
HEIGHT_HL = 0.015
HEIGHT_HR = 0.015

BODY_HEIGHT_FL = 0.060
BODY_HEIGHT_FR = 0.053
BODY_HEIGHT_HL = 0.058
BODY_HEIGHT_HR = 0.05

X_OFFSET_FL = -0.015
X_OFFSET_FR = -0.0
X_OFFSET_HL = -0.025
X_OFFSET_HR = -0.01

PHASE_LAG = {"FL": 0.0, "FR": 0.5, "HL": 0.5, "HR": 0.0}

# 脊柱控制（力矩版参数）
SPINE_YAW  = 0.65
SPINE_F    = 0.9
SPINE_PITCH = -0.65
SPINE_H    = 0.7

def get_leg_params():
    return {
        "FL": {"stride": STRIDE_FL, "height": HEIGHT_FL, "body_height": BODY_HEIGHT_FL,
               "x_offset": X_OFFSET_FL, "is_front": True, "swing_ratio": SWING_RATIO_L},
        "FR": {"stride": STRIDE_FR, "height": HEIGHT_FR, "body_height": BODY_HEIGHT_FR,
               "x_offset": X_OFFSET_FR, "is_front": True, "swing_ratio": SWING_RATIO_R},
        "HL": {"stride": STRIDE_HL, "height": HEIGHT_HL, "body_height": BODY_HEIGHT_HL,
               "x_offset": X_OFFSET_HL, "is_front": False, "swing_ratio": SWING_RATIO_L},
        "HR": {"stride": STRIDE_HR, "height": HEIGHT_HR, "body_height": BODY_HEIGHT_HR,
               "x_offset": X_OFFSET_HR, "is_front": False, "swing_ratio": SWING_RATIO_R},
    }

def compute_joint_angles(leg_name, phase, leg_params, t_cycle):
    params = leg_params[leg_name]
    T_sw = t_cycle * params["swing_ratio"]
    T_st = t_cycle - T_sw
    t_mod = phase * t_cycle
    x_foot, z_foot, _, _ = cycloid_trajectory(
        t_mod, T_sw, T_st,
        params["stride"], params["height"], params["body_height"],
        0.0, params["x_offset"], params["is_front"]
    )
    angles = inverse_kinematics_old((x_foot, z_foot), params["is_front"])
    return angles, x_foot, z_foot

# ==================== MJLAB 模型接口 ====================
ACTUATOR_JOINT_NAMES = [
    "F_spine1_joint", "F_body_joint",
    "FL_shoulder_joint", "FL_elbow_joint",
    "FR_shoulder_joint", "FR_elbow_joint",
    "H_spine1_joint", "H_body_joint",
    "HL_hip_joint", "HL_knee_joint",
    "HR_hip_joint", "HR_knee_joint",
]

LEG_JOINT_MAP = {
    "FL": {"hip": 2, "knee": 3, "is_front": True},
    "FR": {"hip": 4, "knee": 5, "is_front": True},
    "HL": {"hip": 8, "knee": 9, "is_front": False},
    "HR": {"hip": 10, "knee": 11, "is_front": False},
}

# ==================== 数据记录器（极简版）====================
class SimpleRecorder:
    def __init__(self, model):
        self.model = model
        self.records = []
    def record(self, data, target):
        self.records.append({
            "time": data.time,
            "base_pos": data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")].copy(),
            "heading": np.arctan2(data.xquat[1], data.xquat[0]) * 2,  # 简化
        })
    def save(self, path):
        import pandas as pd
        df = pd.DataFrame(self.records)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"数据已保存至 {path}")

# ==================== 初始姿态设置 ====================
def set_initial_pose(model, data):
    data.qpos[:] = model.qpos0.copy()
    init_joint_pos = INIT_STATE.joint_pos
    if init_joint_pos is None:
        print("[WARN] INIT_STATE.joint_pos 为空，使用模型默认零位")
        mujoco.mj_forward(model, data)
        return
    for jnt_name, target_val in init_joint_pos.items():
        if jnt_name == ".*":
            continue
        try:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            qpos_adr = model.jnt_qposadr[jnt_id]
            data.qpos[qpos_adr] = target_val
        except Exception:
            pass
    mujoco.mj_forward(model, data)

# ==================== 主程序 ====================
def main():
    cfg = tyro.cli(TurnConfig)
    output_path = Path(cfg.output) if cfg.output else (MJLAB_SRC_PATH / "scripts" / "Data" / "Turn_Cyc_Min.csv")

    # 加载模型
    spec = get_spec()
    model = spec.compile()
    data = mujoco.MjData(model)

    # 初始姿态
    set_initial_pose(model, data)

    # 记录器
    recorder = SimpleRecorder(model)

    # 默认控制目标（初始姿态）
    default_ctrl = np.zeros(12)
    for i, jnt_name in enumerate(ACTUATOR_JOINT_NAMES):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        default_ctrl[i] = data.qpos[model.jnt_qposadr[jnt_id]]

    # 运行参数
    dt = model.opt.timestep
    leg_params = get_leg_params()
    high_level_target = default_ctrl.copy()
    last_high_level_time = -cfg.high_level_dt

    print(f"开始仿真，总时长 {cfg.total_time}s，步态频率 {cfg.freq}Hz，0.5s 后启动运动")
    print(f"脊柱角度: F_body={SPINE_F}, F_spine1={SPINE_YAW}, H_spine1={SPINE_PITCH}, H_body={SPINE_H}")

    # 启动查看器（默认一直使用）
    with mujoco.viewer.launch_passive(model, data,
                                       show_left_ui=False,
                                       show_right_ui=False) as viewer:
        viewer.cam.lookat = np.array([0.0, 0.0, 0.04])
        viewer.cam.distance = 1.2
        viewer.cam.elevation = -90
        viewer.cam.azimuth = 0

        while data.time < cfg.total_time:
            current_time = data.time

            # 上层控制
            if current_time - last_high_level_time >= cfg.high_level_dt or current_time == 0.0:
                last_high_level_time = current_time
                target = default_ctrl.copy()

                if current_time > cfg.start_time:
                    motion_time = current_time - cfg.start_time
                    base_phase = (motion_time % cfg.t_cycle) / cfg.t_cycle

                    # 腿部
                    for leg_name, leg_info in LEG_JOINT_MAP.items():
                        leg_phase = (base_phase + PHASE_LAG[leg_name]) % 1.0
                        angles, _, _ = compute_joint_angles(leg_name, leg_phase, leg_params, cfg.t_cycle)
                        if angles is not None:
                            target[leg_info["hip"]] = angles[0]
                            target[leg_info["knee"]] = angles[1]

                    # 脊柱（固定角度）
                    target[0] = SPINE_YAW    # F_spine1
                    target[1] = SPINE_F      # F_body
                    target[6] = SPINE_PITCH  # H_spine1
                    target[7] = SPINE_H      # H_body

                high_level_target = target.copy()

            data.ctrl[:] = high_level_target
            recorder.record(data, high_level_target)
            mujoco.mj_step(model, data)
            viewer.sync()

    recorder.save(output_path)
    print("仿真结束。")

if __name__ == "__main__":
    main()