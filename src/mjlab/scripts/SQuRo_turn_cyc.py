import tyro
import mujoco
import numpy as np
import pandas as pd
import mujoco.viewer
from pathlib import Path
from mjlab import MJLAB_SRC_PATH
from dataclasses import dataclass
from mjlab.asset_zoo.robots.SQuRo.SQuRo_constants import get_spec, INIT_STATE



@dataclass
class TurnConfig:
    headless: bool = False  # 无头模式（不显示查看器）
    total_time: float = 5.0
    start_time: float = 0.5
    freq: float = 2.0
    spine_yaw: float = 0.6  # 脊柱侧摆角（正值=右转，见 CLAUDE.md）
    output: str | None = None  # 自定义输出 CSV 路径
    high_level_dt: float = 0.005  # 上层控制时间步长

    @property
    def t_cycle(self) -> float:
        return 1.0 / self.freq


# 角度归一化
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# 摆线轨迹生成
def cycloid_trajectory(t_mod, T_sw, T_st, S, H, body_height, rotation_angle, x_offset, is_front=True):
    if t_mod <= T_sw:
        # 摆动相：摆线轨迹
        t_norm = t_mod / T_sw
        x = S * (t_norm - (1.0 / (2.0 * np.pi)) * np.sin(2.0 * np.pi * t_norm))
        z = 0.5 * H * (1.0 - np.cos(2.0 * np.pi * t_norm))
    else:
        # 支撑相：线性回退
        t_st = (t_mod - T_sw) / T_st
        x = S * (1.0 - t_st)
        z = 0.0

    # 应用偏移和身体高度
    x_leg = x + x_offset
    z_leg = z - body_height

    # 2D 旋转
    cos_r = np.cos(rotation_angle)
    sin_r = np.sin(rotation_angle)
    x_rot = x_leg * cos_r - z_leg * sin_r
    z_rot = x_leg * sin_r + z_leg * cos_r

    return x_rot, z_rot, x_rot, z_rot


# 逆运动学
def inverse_kinematics(target_pos, is_front=True):
    x, y = target_pos

    if is_front:
        L1 = 0.040  # 肩连杆长度
        L2 = 0.040  # 肘连杆长度
    else:
        L1 = 0.040  # 髋连杆长度
        L2 = 0.036  # 膝连杆长度

    R = np.sqrt(x**2 + y**2)
    if R < 1e-8:
        return None

    K = (L2**2 - x**2 - y**2 - L1**2) / (2.0 * L1)
    K = np.clip(K / max(R, 1e-8), -1.0, 1.0)

    theta = np.arctan2(y, x)
    phi = np.arccos(K)

    if is_front:
        a1 = theta + phi
        sin_a2 = (x + L1 * np.cos(a1)) / L2
        cos_a2 = (y + L1 * np.sin(a1)) / L2
        a2 = np.arctan2(sin_a2, cos_a2)
        return (a1 - 0.888, -(a2 + 2.648))
    else:
        a1 = theta - phi
        sin_a2 = (x + L1 * np.cos(a1)) / L2
        cos_a2 = (y + L1 * np.sin(a1)) / L2
        a2 = np.arctan2(sin_a2, cos_a2)
        return (a1 + 4.325, -(a2 + 1.794))


# ==================== 配置参数 ====================

OUTPUT_DIR = MJLAB_SRC_PATH / "scripts" / "Data"
OUTPUT_CSV_PATH = OUTPUT_DIR / "Turn_Cyc_Min.csv"
TOTAL_TIME = 10.0
START_TIME = 0.5

# 步态参数
FREQ = 2.0
T_CYCLE = 1.0 / FREQ
SWING_RATIO = 0.5

# 各腿步态参数（等步长，靠脊柱侧摆实现转弯）
STRIDE_FL = 0.04
STRIDE_FR = 0.04
STRIDE_HL = 0.04
STRIDE_HR = 0.04
HEIGHT_FL = 0.01
HEIGHT_FR = 0.01
HEIGHT_HL = 0.015
HEIGHT_HR = 0.015
BODY_HEIGHT_FL = 0.060
BODY_HEIGHT_FR = 0.060
BODY_HEIGHT_HL = 0.058
BODY_HEIGHT_HR = 0.058
X_OFFSET_FL = -0.02
X_OFFSET_FR = -0.02
X_OFFSET_HL = -0.03
X_OFFSET_HR = -0.03
PHASE_LAG = {"FL": 0.0, "FR": 0.5, "HL": 0.5, "HR": 0.0}



# ==================== 辅助函数 ====================
# 获取各腿步态参数
def get_leg_params(freq: float = FREQ):
    _ = freq  # 保留接口兼容
    return {
        "FL": {"stride": STRIDE_FL, "height": HEIGHT_FL, "body_height": BODY_HEIGHT_FL,
               "x_offset": X_OFFSET_FL, "is_front": True},
        "FR": {"stride": STRIDE_FR, "height": HEIGHT_FR, "body_height": BODY_HEIGHT_FR,
               "x_offset": X_OFFSET_FR, "is_front": True},
        "HL": {"stride": STRIDE_HL, "height": HEIGHT_HL, "body_height": BODY_HEIGHT_HL,
               "x_offset": X_OFFSET_HL, "is_front": False},
        "HR": {"stride": STRIDE_HR, "height": HEIGHT_HR, "body_height": BODY_HEIGHT_HR,
               "x_offset": X_OFFSET_HR, "is_front": False},
    }


# 计算给定腿的目标关节角度
def compute_joint_angles(leg_name, phase, leg_params, cfg: TurnConfig):
    params = leg_params[leg_name]
    t_cycle = cfg.t_cycle
    T_sw = t_cycle * SWING_RATIO
    T_st = t_cycle - T_sw
    t_mod = phase * t_cycle

    x_foot, z_foot, _, _ = cycloid_trajectory(
        t_mod=t_mod, T_sw=T_sw, T_st=T_st,
        S=params["stride"], H=params["height"], body_height=params["body_height"],
        rotation_angle=0.0, x_offset=params["x_offset"], is_front=params["is_front"]
    )

    angles = inverse_kinematics(target_pos=(x_foot, z_foot), is_front=params["is_front"])
    return angles, x_foot, z_foot


# 执行器对应的关节名称（按执行器顺序 0-11）
ACTUATOR_JOINT_NAMES = [
    "F_spine1_joint", "F_body_joint",
    "FL_shoulder_joint", "FL_elbow_joint",
    "FR_shoulder_joint", "FR_elbow_joint",
    "H_spine1_joint", "H_body_joint",
    "HL_hip_joint", "HL_knee_joint",
    "HR_hip_joint", "HR_knee_joint",
]

# CSV 中的关节列名（与 CSV_Anaylsis.py ACTION_SCALES 一致）
CSV_JOINT_NAMES = [
    "F_spine1", "F_body",
    "FL_shoulder", "FL_elbow",
    "FR_shoulder", "FR_elbow",
    "H_spine1", "H_body",
    "HL_hip", "HL_knee",
    "HR_hip", "HR_knee",
]

# 足端名称与 site 名称
FOOT_NAMES = ["FL", "FR", "HL", "HR"]
FOOT_SITE_NAMES = ["FL_elbow_site", "FR_elbow_site", "HL_knee_site", "HR_knee_site"]

# 腿关节在 12 个执行器中的索引
LEG_JOINT_MAP = {
    "FL": {"hip": 2, "knee": 3, "is_front": True},
    "FR": {"hip": 4, "knee": 5, "is_front": True},
    "HL": {"hip": 8, "knee": 9, "is_front": False},
    "HR": {"hip": 10, "knee": 11, "is_front": False},
}


class TurnDataRecorder:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.records = []
        self.step_count = 0

        # 预计算执行器对应的关节 qpos/qvel 地址
        self._act_joint_qpos_adr = []
        self._act_joint_dof_adr = []
        for jnt_name in ACTUATOR_JOINT_NAMES:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            self._act_joint_qpos_adr.append(model.jnt_qposadr[jnt_id])
            self._act_joint_dof_adr.append(model.jnt_dofadr[jnt_id])

        # 足端 site ID
        self._foot_site_ids = []
        for site_name in FOOT_SITE_NAMES:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            self._foot_site_ids.append(site_id)

        # F_body_Link body ID（用于计算 f_body_heading）
        self._f_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "F_body_Link")

        # 基座 body ID
        self._base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")

    # 记录一步仿真数据
    def record_step(self, data: mujoco.MjData, target_pos: np.ndarray):
        record = {"step": self.step_count}

        # 关节位置（相对于默认值的增量）
        for i, name in enumerate(CSV_JOINT_NAMES):
            qpos_adr = self._act_joint_qpos_adr[i]
            dof_adr = self._act_joint_dof_adr[i]
            # 位置（相对于默认关节位置）
            record[f"{name}_pos"] = float(data.qpos[qpos_adr] - self.model.qpos0[qpos_adr]) # type: ignore
            # 速度
            record[f"{name}_vel"] = float(data.qvel[dof_adr]) # type: ignore
            # 加速度
            record[f"{name}_acc"] = float(data.qacc[dof_adr]) # type: ignore
            # 力矩（执行器力）
            record[f"{name}_torque"] = float(data.qfrc_actuator[dof_adr]) # type: ignore
            # 动作（目标位置）
            record[f"{name}_action"] = float(target_pos[i]) # type: ignore
            # 参考位置/速度（占位）
            record[f"{name}_ref_pos"] = float(target_pos[i]) # type: ignore
            record[f"{name}_ref_vel"] = 0.0 # type: ignore

        # 基座位置
        base_pos = data.xpos[self._base_body_id]
        record["base_pos_x"] = float(base_pos[0]) # type: ignore
        record["base_pos_y"] = float(base_pos[1]) # type: ignore
        record["base_pos_z"] = float(base_pos[2]) # type: ignore

        # 基座线速度（世界系）
        base_lin_vel = data.cvel[self._base_body_id][3:6]  # 线速度部分
        record["base_lin_vel_x"] = float(base_lin_vel[0]) # type: ignore
        record["base_lin_vel_y"] = float(base_lin_vel[1]) # type: ignore
        record["base_lin_vel_z"] = float(base_lin_vel[2]) # type: ignore

        # 基座角速度（世界系）
        base_ang_vel = data.cvel[self._base_body_id][:3]  # 角速度部分
        record["base_ang_vel_x"] = float(base_ang_vel[0]) # type: ignore
        record["base_ang_vel_y"] = float(base_ang_vel[1]) # type: ignore
        record["base_ang_vel_z"] = float(base_ang_vel[2]) # type: ignore

        # heading（基座偏航角）
        base_quat = data.xquat[self._base_body_id]  # [w, x, y, z]
        w, x, y, z = base_quat[0], base_quat[1], base_quat[2], base_quat[3]
        sin_h = 2.0 * (w * z + x * y)
        cos_h = 1.0 - 2.0 * (y * y + z * z)
        record["heading"] = float(np.arctan2(sin_h, cos_h)) # type: ignore

        # f_body_heading（F_body_Link 偏航角）
        f_body_quat = data.xquat[self._f_body_id]
        w, x, y, z = f_body_quat[0], f_body_quat[1], f_body_quat[2], f_body_quat[3]
        sin_h_fb = 2.0 * (w * z + x * y)
        cos_h_fb = 1.0 - 2.0 * (y * y + z * z)
        record["f_body_heading"] = float(np.arctan2(sin_h_fb, cos_h_fb)) # type: ignore

        # 足端位置（世界系）
        for i, name in enumerate(FOOT_NAMES):
            site_id = self._foot_site_ids[i]
            site_pos = data.site_xpos[site_id]
            record[f"foot_{name}_x"] = float(site_pos[0]) # type: ignore
            record[f"foot_{name}_y"] = float(site_pos[1]) # type: ignore
            record[f"foot_{name}_z"] = float(site_pos[2]) # type: ignore

        # 接触力（无传感器，置 0）
        for name in FOOT_NAMES:
            for axis in ["x", "y", "z"]:
                record[f"contact_{name}_{axis}"] = 0.0 # type: ignore
            record[f"contact_{name}_mag"] = 0.0 # type: ignore

        # 命令参数（固定值，适配转弯场景）
        record["vel_command_x"] = 0.0        # 原地转弯，前进速度为 0 # type: ignore
        record["height_f_command"] = 0.06 # type: ignore
        record["height_h_command"] = 0.06 # type: ignore
        record["gait_freq_command"] = FREQ # type: ignore
        record["curvature_command"] = 2.0    # 高曲率（近似原地转弯） # type: ignore

        record["reward"] = 0.0 # type: ignore
        record["done"] = 0.0 # type: ignore

        self.records.append(record)
        self.step_count += 1

    # 保存CSV
    def save(self, path: Path):
        if not self.records:
            print("[WARN] 无数据可保存")
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.records)
        df["step"] = df["step"].astype(int)
        df.to_csv(path, index=False)
        print(f"[INFO] 数据已保存: {path}")
        print(f"[INFO] {len(self.records)} 步, {len(df.columns)} 列")


# 设置初始姿态
def set_initial_pose(model: mujoco.MjModel, data: mujoco.MjData):
    # 以 qpos0 为基础（保留正确的自由关节姿态和所有关节初始值）
    data.qpos[:] = model.qpos0.copy()

    # 覆盖 INIT_STATE 中定义的关节位置
    init_joint_pos = INIT_STATE.joint_pos
    for jnt_name, target_val in init_joint_pos.items(): # type: ignore
        if jnt_name == ".*":
            continue  # 通配符，跳过
        try:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            qpos_adr = model.jnt_qposadr[jnt_id]
            data.qpos[qpos_adr] = target_val
        except Exception:
            pass

    mujoco.mj_forward(model, data)


# 运行仿真
def run_simulation(model: mujoco.MjModel, data: mujoco.MjData, recorder: "TurnDataRecorder", default_ctrl: np.ndarray, cfg: TurnConfig):
    dt = model.opt.timestep
    leg_params_store = get_leg_params(cfg.freq)
    last_high_level_time = -cfg.high_level_dt
    high_level_target = default_ctrl.copy()
    initial_base_x = None
    initial_base_y = None
    initial_heading = None
    step_count = 0

    print(f"[INFO] 仿真时间步长: {dt:.4f}s")
    print(f"[INFO] 总仿真时间: {cfg.total_time}s")
    print(f"[INFO] 运动开始时间: {cfg.start_time}s")
    print(f"[INFO] 步态频率: {cfg.freq}Hz, 周期: {cfg.t_cycle:.3f}s")
    print(f"[INFO] 脊柱侧摆目标: {cfg.spine_yaw} rad")

    # 仿真步进函数
    def simulation_step():
        nonlocal last_high_level_time, high_level_target, initial_base_x
        nonlocal initial_base_y, initial_heading, step_count

        current_time = data.time

        # 记录初始基座位置和朝向
        if initial_base_x is None:
            initial_base_x = data.qpos[0]
            initial_base_y = data.qpos[1]
            w, x, y, z = (data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6])
            sin_h = 2.0 * (w * z + x * y)
            cos_h = 1.0 - 2.0 * (y * y + z * z)
            initial_heading = np.arctan2(sin_h, cos_h)
            print(f"[INFO] 初始基座 X={initial_base_x:.4f}, Y={initial_base_y:.4f}, "
                  f"Heading={initial_heading:.4f}")

        # 上层控制（高频更新目标位置）
        if current_time - last_high_level_time >= cfg.high_level_dt or current_time == 0.0:
            last_high_level_time = current_time
            target = default_ctrl.copy()

            if current_time > cfg.start_time:
                motion_time = current_time - cfg.start_time
                base_phase = (motion_time % cfg.t_cycle) / cfg.t_cycle

                # 腿部控制
                for leg_name, leg_info in LEG_JOINT_MAP.items():
                    leg_phase = (base_phase + PHASE_LAG[leg_name]) % 1.0
                    angles, _x_foot, _z_foot = compute_joint_angles(
                        leg_name, leg_phase, leg_params_store, cfg)

                    if angles is not None:
                        target[leg_info["hip"]] = angles[0]
                        target[leg_info["knee"]] = angles[1]

                # 脊柱控制：侧摆实现转弯
                target[0] = 0.6   # F_spine1 → 侧摆
                target[1] = 1.2              # F_body → 不扭转
                target[6] = -0.6             # H_spine1 → 保持中立
                target[7] = 0.9             # H_body → 不扭转

            high_level_target = target.copy()

        # 设置执行器目标位置
        data.ctrl[:] = high_level_target

        # 记录数据
        recorder.record_step(data, high_level_target)

        # 物理步进
        mujoco.mj_step(model, data)
        step_count += 1

    if cfg.headless:
        # 无头模式：直接循环
        print("[INFO] 无头模式运行中...")
        while data.time < cfg.total_time:
            simulation_step()
    else:
        # 带查看器模式
        print("[INFO] 启动 MuJoCo 查看器...")
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            # 设置俯视视角
            viewer.cam.lookat = np.array([0.0, 0.0, 0.04])
            viewer.cam.distance = 1.2
            viewer.cam.elevation = -90
            viewer.cam.azimuth = 0

            while data.time < cfg.total_time:
                simulation_step()
                viewer.sync()

    return initial_base_x, initial_base_y, initial_heading


def main():
    cfg = tyro.cli(TurnConfig, description="SQuRo 原地转弯 — 摆线轨迹 + 逆运动学")

    # 输出路径
    output_path = Path(cfg.output) if cfg.output else OUTPUT_CSV_PATH

    # 加载模型
    spec = get_spec()
    model = spec.compile()
    data = mujoco.MjData(model)

    # 设置初始姿态
    set_initial_pose(model, data)

    # 初始化数据记录器
    recorder = TurnDataRecorder(model)

    # 获取初始关节位置（作为默认控制目标）
    default_ctrl = np.zeros(12)
    for i, jnt_name in enumerate(ACTUATOR_JOINT_NAMES):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        qpos_adr = model.jnt_qposadr[jnt_id]
        default_ctrl[i] = data.qpos[qpos_adr]

    # 运行仿真
    try:
        initial_x, initial_y, initial_heading = run_simulation(model, data, recorder, default_ctrl, cfg)
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断仿真")

    # 保存数据
    recorder.save(output_path)

    # 打印运动分析
    dt = model.opt.timestep
    print("\n" + "=" * 60)
    print("运动分析")
    print("=" * 60)
    if recorder.records:
        df = pd.DataFrame(recorder.records)
        motion_mask = df["step"] * dt >= cfg.start_time
        motion_df = df[motion_mask]

        if len(motion_df) > 0:
            dx = (motion_df["base_pos_x"].iloc[-1] -
                  motion_df["base_pos_x"].iloc[0])
            dy = (motion_df["base_pos_y"].iloc[-1] -
                  motion_df["base_pos_y"].iloc[0])
            distance = np.sqrt(dx**2 + dy**2)
            h0 = motion_df["heading"].iloc[0]
            h1 = motion_df["heading"].iloc[-1]
            heading_change = normalize_angle(h1 - h0)

            print(f"初始基座: ({initial_x:.4f}, {initial_y:.4f})") # type: ignore
            print(f"最终基座: ({motion_df['base_pos_x'].iloc[-1]:.4f}, "
                  f"{motion_df['base_pos_y'].iloc[-1]:.4f})")
            print(f"位移: ({dx:.4f}, {dy:.4f})")
            print(f"移动距离: {distance:.4f} m")
            print(f"偏航角变化: {np.degrees(heading_change):.1f}°")
            print(f"运动时间: {motion_df['step'].iloc[-1] * dt - cfg.start_time:.2f}s")

    print(f"\n数据已保存至: {output_path}")
    print(f"可使用 CSV_Anaylsis.py 分析（修改 CSV_PATH 指向该文件）")


if __name__ == "__main__":
    main()
