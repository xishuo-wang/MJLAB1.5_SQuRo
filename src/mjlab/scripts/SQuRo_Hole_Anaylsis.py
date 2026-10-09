import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from scipy.integrate import trapezoid
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Optional, Union
from scipy.signal import welch, savgol_filter, find_peaks


# ==================================================================================================
# 文件路径配置
XML_PATH = r"D:\MuJoCoLab_1.5\src\mjlab\asset_zoo\robots\SQuRo\xmls\SQuRo.xml"
CSV_PATH = r"D:\MuJoCoLab_1.5\logs\rsl_rl\SQuRo_Hole\2026-10-09_15-29-55\videos\SQuRo_Hole_1900-fixed-hF40-hH40-col0-noobs.csv"

# 控制时间配置
TIMESTEP = 0.002
DECIMATION = 5
DT = TIMESTEP * DECIMATION

# 分析时间配置
START_TIME = 0.5
END_TIME = 5.0
TIME_RANGE = (START_TIME, END_TIME)

# 周期配置
T = 1.0
POINTS_PER_CYCLE = int(T / DT)
CONTACT_THRESHOLD = 0.00

# 关节配置 — 顺序与 entity actuator（关节树深度优先）一致
ACTION_SCALES = {
    "F_spine1": 0.3, "F_body": 0.3,
    "FL_shoulder": 0.3, "FL_elbow": 0.3,
    "FR_shoulder": 0.3, "FR_elbow": 0.3,
    "H_spine1": 0.3, "H_body": 0.3,
    "HL_hip": 0.3, "HL_knee": 0.3,
    "HR_hip": 0.3, "HR_knee": 0.3,
}
ACTUATED_JOINTS = list(ACTION_SCALES.keys())

# 头颈关节
NECK_JOINTS = ['Neck_yaw', 'Neck_pitch']

# 足端配置
FOOT_NAMES = ['FR', 'FL', 'HR', 'HL']
FOOT_LABELS = ['FR (右前)', 'FL (左前)', 'HR (右后)', 'HL (左后)']
FOOT_COLORS = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']

# 统一文本样式配置
STATS_FONTSIZE = 10
STATS_BGCOLOR = '#FFF8DC'
STATS_ALPHA = 1.0

# 脊柱频谱分析配置（保留，当前未使用）
_WELCH_NPERSEG = 128
_WELCH_NFFT = 512



def get_scaled_action(df, joint_name):
    action_col = f"{joint_name}_action"
    if action_col not in df.columns:
        return None
    scale = ACTION_SCALES.get(joint_name, 1.0)
    return df[action_col] * scale



def define_cycles_by_period(start_time: float, end_time: float, period: float) -> list[tuple[float, float]]:
    cycles = []
    t = start_time
    while t + period <= end_time + 1e-9:
        cycles.append((t, t + period))
        t += period
    return cycles


# 计算每个足端在固定周期内的支撑相时间占比
def compute_stance_duty_factors(motion_df: pd.DataFrame, forces: dict, cycles: list[tuple[float, float]]) -> dict:
    if forces is None:
        return None

    time = motion_df['time'].values.astype(float)
    results = {}

    for foot in FOOT_NAMES:
        fz = forces[foot]['z']
        duty_factors = []
        stance_durations = []
        swing_durations = []
        cycle_durations = []

        for t_start, t_end in cycles:
            mask = (time >= float(t_start)) & (time < float(t_end)) # type: ignore
            cycle_fz = fz[mask]

            if len(cycle_fz) == 0:
                continue

            stance_mask = cycle_fz > CONTACT_THRESHOLD
            stance_time = np.sum(stance_mask) * DT
            swing_time = np.sum(~stance_mask) * DT
            total_time = len(cycle_fz) * DT

            duty_factor = stance_time / total_time if total_time > 0 else 0.0

            duty_factors.append(duty_factor)
            stance_durations.append(stance_time)
            swing_durations.append(swing_time)
            cycle_durations.append(total_time)

        if len(duty_factors) > 0:
            results[foot] = {
                'duty_factor_mean': np.mean(duty_factors),
                'duty_factor_std': np.std(duty_factors),
                'stance_duration_mean': np.mean(stance_durations),
                'stance_duration_std': np.std(stance_durations),
                'swing_duration_mean': np.mean(swing_durations),
                'swing_duration_std': np.std(swing_durations),
                'n_cycles': len(duty_factors),
                'all_duty_factors': np.array(duty_factors),
                'all_stance_durations': np.array(stance_durations),
                'all_swing_durations': np.array(swing_durations),
            }
        else:
            results[foot] = None

    all_duty_factors = []
    for foot in FOOT_NAMES:
        if results[foot] is not None:
            all_duty_factors.extend(results[foot]['all_duty_factors'])

    if all_duty_factors:
        results['overall'] = {
            'mean_duty_factor': float(np.mean(all_duty_factors)),
            'std_duty_factor': float(np.std(all_duty_factors)),
            'min_duty_factor': float(np.min(all_duty_factors)),
            'max_duty_factor': float(np.max(all_duty_factors)),
        }
    else:
        results['overall'] = {
            'mean_duty_factor': 0.0,
            'std_duty_factor': 0.0,
            'min_duty_factor': 0.0,
            'max_duty_factor': 0.0,
        }

    return results



def compute_cycle_averaged_manual(df: pd.DataFrame, foot_name: str, cycles: list[tuple[float, float]]) -> Optional[dict]:
    rx_col = f'foot_{foot_name}_rx'
    rz_col = f'foot_{foot_name}_z'
    rx_matrix, rz_matrix = [], []
    for t_start, t_end in cycles:
        mask = (df['time'] >= t_start) & (df['time'] < t_end)
        cyc = df.loc[mask]
        if len(cyc) != POINTS_PER_CYCLE:
            continue
        rx_matrix.append(cyc[rx_col].values)
        rz_matrix.append(cyc[rz_col].values)
    if len(rx_matrix) < 2:
        return None
    rx_arr = np.array(rx_matrix)
    rz_arr = np.array(rz_matrix)
    rx_mean = np.mean(rx_arr, axis=0)
    rx_std = np.std(rx_arr, axis=0)
    rz_mean = np.mean(rz_arr, axis=0)
    rz_std = np.std(rz_arr, axis=0)
    window = min(11, POINTS_PER_CYCLE - 2)
    if window % 2 == 0:
        window -= 1
    if window >= 5:
        rx_smooth = savgol_filter(rx_mean, window, 2, mode='wrap')
        rz_smooth = savgol_filter(rz_mean, window, 2, mode='wrap')
    else:
        rx_smooth, rz_smooth = rx_mean, rz_mean
    return {
        'rx_mean': rx_mean, 'rx_std': rx_std, 'rx_smooth': rx_smooth,
        'rz_mean': rz_mean, 'rz_std': rz_std, 'rz_smooth': rz_smooth,
        'n_cycles': len(rx_matrix),
    }



# 解析MuJoCo配置
def parse_mujoco_config(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    option = root.find('option')
    gravity_str = option.get('gravity', '0 0 -9.81') if option is not None else '0 0 -9.81'
    gravity_z = abs(float(gravity_str.split()[2]))

    total_mass = 0.0
    for geom in root.iter('geom'):
        mass_str = geom.get('mass')
        if mass_str:
            total_mass += float(mass_str)
    total_mass = round(total_mass, 5)
    gravity_z = round(gravity_z, 5)
    return total_mass, gravity_z

ROBOT_MASS, GRAVITY = parse_mujoco_config(XML_PATH)



# 地反力解析
def extract_contact_forces_all(motion_df: pd.DataFrame) -> Optional[dict]:
    feet = ['FL', 'FR', 'HL', 'HR']
    required_cols = [f'contact_{f}_{d}' for f in feet for d in ['x', 'z']]
    if not all(col in motion_df.columns for col in required_cols):
        return None
    forces = {}
    for foot in feet:
        fx = motion_df[f'contact_{foot}_x'].values.astype(float)
        fz_raw = motion_df[f'contact_{foot}_z'].values.astype(float)
        fz = -fz_raw # type: ignore
        forces[foot] = {'x': fx, 'z': fz, 'xz': np.sqrt(fx**2 + fz**2)} # type: ignore
    return forces



# 计算总指标（适配转弯任务，基于路径长度、F-body前进速度、曲率）
def calculate_metrics(motion_df: pd.DataFrame) -> dict:
    motion_df = motion_df.copy()

    motion_df['power'] = 0.0
    for joint in ACTUATED_JOINTS:
        vel_col = f"{joint}_vel"
        torque_col = f"{joint}_torque"
        if vel_col in motion_df.columns and torque_col in motion_df.columns:
            motion_df['power'] += np.abs(motion_df[vel_col] * motion_df[torque_col])

    total_work = motion_df['power'].sum() * DT
    total_time = motion_df['time'].iloc[-1] - motion_df['time'].iloc[0]

    dx = np.diff(motion_df['base_pos_x'])
    dy = np.diff(motion_df['base_pos_y'])
    path_length = np.sum(np.sqrt(dx**2 + dy**2))

    forward_speed = motion_df['forward_speed']
    vel_avg = forward_speed.mean()
    max_forward_speed = forward_speed.abs().max()

    heading_col = 'f_body_heading' if 'f_body_heading' in motion_df.columns else 'heading'
    heading_vals = np.asarray(motion_df[heading_col].values, dtype=np.float64)
    heading_unwrapped = np.unwrap(heading_vals)
    delta_heading = heading_unwrapped[-1] - heading_unwrapped[0]
    overall_curvature = abs(delta_heading) / path_length if path_length > 0 else 0.0

    ang_vel_z = motion_df['base_ang_vel_z'].abs()
    eps = 1e-6
    curvature_instant = ang_vel_z / (forward_speed.abs() + eps)
    mean_abs_curvature_instant = curvature_instant.mean()
    max_curvature_instant = curvature_instant.max()

    if path_length > 0:
        cot_avg = total_work / (ROBOT_MASS * GRAVITY * path_length)
    else:
        cot_avg = float('inf')

    print(f"f_body_heading 首尾: {heading_vals[0]:.6f} -> {heading_vals[-1]:.6f}")
    print("=" * 60)
    print(f"总机械功: {total_work:.6f} J")
    print(f"平均功率: {total_work / total_time:.6f} W")
    print(f"路径总长: {path_length:.6f} m")
    print(f"运动时间: {total_time:.3f} s")
    print(f"平均前进速度: {vel_avg:.6f} m/s")
    print(f"最大前进速度: {max_forward_speed:.6f} m/s")
    print(f"平均 COT: {cot_avg:.6f}")
    print(f"整体曲率 (Δheading / path): {overall_curvature:.4f} rad/m")
    print(f"瞬时平均曲率 (含低速噪声): {mean_abs_curvature_instant:.4f} rad/m")
    print(f"最大瞬时曲率: {max_curvature_instant:.4f} rad/m")

    return {
        'cot_avg': cot_avg,
        'vel_avg': vel_avg,
        'max_forward_speed': max_forward_speed,
        'total_work': total_work,
        'work_avg': total_work / total_time,
        'path_length': path_length,
        'total_time': total_time,
        'overall_curvature': overall_curvature,
        'mean_abs_curvature_instant': mean_abs_curvature_instant,
        'max_curvature_instant': max_curvature_instant,
    }



# 计算平均冲量
def calculate_impulse(motion_df: pd.DataFrame, forces: dict) -> Optional[dict]:
    if len(motion_df) == 0:
        print("[冲量] 无有效数据")
        return None

    time = motion_df['time'].values
    cycles = define_cycles_by_period(TIME_RANGE[0], TIME_RANGE[1], T)
    n_cycles = len(cycles)

    impulse = {}
    for foot in FOOT_NAMES:
        fx = forces[foot]['x']
        fz = forces[foot]['z']
        Ix = float(trapezoid(fx, time))
        Iz = float(trapezoid(fz, time))
        avg_Ix = Ix / n_cycles if n_cycles > 0 else float('nan')
        avg_Iz = Iz / n_cycles if n_cycles > 0 else float('nan')
        impulse[foot] = {
            'I_x': Ix, 'I_z': Iz,
            'avg_I_x': avg_Ix, 'avg_I_z': avg_Iz
        }

    fore_avg_x = impulse['FL']['avg_I_x'] + impulse['FR']['avg_I_x']
    fore_avg_z = impulse['FL']['avg_I_z'] + impulse['FR']['avg_I_z']
    hind_avg_x = impulse['HL']['avg_I_x'] + impulse['HR']['avg_I_x']
    hind_avg_z = impulse['HL']['avg_I_z'] + impulse['HR']['avg_I_z']

    print(f"前腿周期冲量 X: {fore_avg_x:.4f} Ns, Z: {fore_avg_z:.4f} Ns")
    print(f"后腿周期冲量 X: {hind_avg_x:.4f} Ns, Z: {hind_avg_z:.4f} Ns")

    return impulse



# 计算足端周期平均缓存
def precompute_foot_cycle_average(motion_df: pd.DataFrame) -> dict:
    cycles = define_cycles_by_period(TIME_RANGE[0], TIME_RANGE[1], T)
    results = {}
    for name in FOOT_NAMES:
        results[name] = compute_cycle_averaged_manual(motion_df, name, cycles)
    return results



# 页面0: 前后肢中心高度随 X 位置变化（XZ 平面）
def plot_base_analysis(motion_df, ax, metrics: dict):
    ax.clear()

    # ---------- 全段数据（画图用） ----------
    x = motion_df['base_pos_x'].values
    time = motion_df['time'].values
    f_h = motion_df['F_body_height'].values * 1000.0
    h_h = motion_df['H_body_height'].values * 1000.0
    f_cmd = motion_df['height_F_command'].values * 1000.0 \
        if 'height_F_command' in motion_df.columns else None
    h_cmd = motion_df['height_H_command'].values * 1000.0 \
        if 'height_H_command' in motion_df.columns else None
    base_z = motion_df['base_pos_z'].values * 1000.0 \
        if 'base_pos_z' in motion_df.columns else None

    # ---------- 稳定段掩码（统计用，不含初始化） ----------
    stable_mask = time >= START_TIME
    if not np.any(stable_mask):
        stable_mask = np.ones_like(time, dtype=bool)

    f_h_s   = f_h[stable_mask]
    h_h_s   = h_h[stable_mask]
    f_cmd_s = f_cmd[stable_mask] if f_cmd is not None else None
    h_cmd_s = h_cmd[stable_mask] if h_cmd is not None else None

    # ---------- 绘制曲线（全段） ----------
    if base_z is not None:
        ax.plot(x, base_z, '-', color='gray', linewidth=1.0, alpha=0.5,
                label='基座 (root link)')
    ax.plot(x, f_h, '-', color='#E41A1C', linewidth=1.8, alpha=0.9,
            label='F_body 高度 (实测)')
    ax.plot(x, h_h, '-', color='#377EB8', linewidth=1.8, alpha=0.9,
            label='H_body 高度 (实测)')
    if f_cmd is not None:
        ax.plot(x, f_cmd, '--', color='#E41A1C', linewidth=1.0, alpha=0.55,
                label='F_body 命令')
    if h_cmd is not None:
        ax.plot(x, h_cmd, '--', color='#377EB8', linewidth=1.0, alpha=0.55,
                label='H_body 命令')

    if np.any(stable_mask) and not np.all(stable_mask):
        first_idx = int(np.argmax(stable_mask))
        ax.axvline(x[first_idx], color='green', linestyle=':', linewidth=1.0,
                   alpha=0.5, label=f'START_TIME={START_TIME}s')

    ax.set_xlabel('X 位置 (m)', fontsize=12)
    ax.set_ylabel('高度 (mm)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # ---------- 统计（仅稳定段） ----------
    f_err_mean = (f_h_s - f_cmd_s).mean() if f_cmd_s is not None else float('nan')
    f_err_std  = (f_h_s - f_cmd_s).std()  if f_cmd_s is not None else float('nan')
    h_err_mean = (h_h_s - h_cmd_s).mean() if h_cmd_s is not None else float('nan')
    h_err_std  = (h_h_s - h_cmd_s).std()  if h_cmd_s is not None else float('nan')
    n_stable = int(stable_mask.sum())

    stats_text = textwrap.dedent(f"""\
        总仿真步数: {len(motion_df)} 步 (稳定段 {n_stable} 步)
        分析时间 ({START_TIME}s-{END_TIME}s):
        ─────────────
        F_body: mean={f_h_s.mean():.1f} mm, range=[{f_h_s.min():.1f}, {f_h_s.max():.1f}] mm
        H_body: mean={h_h_s.mean():.1f} mm, range=[{h_h_s.min():.1f}, {h_h_s.max():.1f}] mm
        F 跟踪误差: mean={f_err_mean:+.2f} mm, std={f_err_std:.2f} mm
        H 跟踪误差: mean={h_err_mean:+.2f} mm, std={h_err_std:.2f} mm
        ─────────────
        路径总长: {metrics['path_length']:.3f} m
        平均前进速度: {metrics['vel_avg']:.3f} m/s
        最大前进速度: {metrics['max_forward_speed']:.3f} m/s
        ─────────────
        质量: {ROBOT_MASS} kg
        重力加速度: {GRAVITY} m/s^2""")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=STATS_FONTSIZE + 1, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=1', facecolor=STATS_BGCOLOR, alpha=STATS_ALPHA))

    ax.set_title(f'前后肢中心高度随 X 位置变化 ({START_TIME}-{END_TIME}s)', fontsize=13, fontweight='bold')



# 页面1-3: 关节数据分析（保持不变）
def plot_joint_analysis(motion_df, axs, joint_indices):
    for i, joint_idx in enumerate(joint_indices):
        if i >= len(axs):
            break
        ax = axs[i]
        ax.clear()
        joint_name = ACTUATED_JOINTS[joint_idx]
        scale = ACTION_SCALES.get(joint_name, 1.0)
        pos_col = f"{joint_name}_pos"
        torque_col = f"{joint_name}_torque"
        action_col = f"{joint_name}_action"
        if pos_col not in motion_df.columns:
            ax.text(0.5, 0.5, f'数据列不存在: {pos_col}', ha='center', va='center')
            continue
        time_data = motion_df['time']
        lines = []
        labels = []
        if pos_col in motion_df.columns:
            line1, = ax.plot(time_data, motion_df[pos_col], 'r-', linewidth=1.5, alpha=0.8, label='位置')
            lines.append(line1); labels.append('位置')
        if action_col in motion_df.columns and scale > 0.0:
            scaled_action = get_scaled_action(motion_df, joint_name)
            if scaled_action is not None:
                line3, = ax.plot(time_data, scaled_action, 'm:', linewidth=1.0, alpha=0.6, label=f'Action (scale={scale})')
                lines.append(line3); labels.append(f'Action (×{1/scale:.1f})')
        elif action_col in motion_df.columns and scale == 0.0:
            line3, = ax.plot(time_data, motion_df[action_col], 'm:', linewidth=1.0, alpha=0.3, label='Action (不控制)')
            lines.append(line3); labels.append('Action (不控制)')
        scale_info = f" [scale={scale}]" if scale > 0 else " [不控制]"
        ax.set_title(f'关节 {joint_idx}: {joint_name}{scale_info}', fontsize=11, pad=12)
        ax.set_xlabel('时间 (秒)', fontsize=9, labelpad=5)
        ax.set_ylabel('值', fontsize=9, labelpad=5)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)
        stats_text = f"{joint_name}"
        if scale > 0:
            stats_text += f" (scale={scale})"
        else:
            stats_text += " (不控制)"
        stats_text += "\n"
        stats_values = []
        if pos_col in motion_df.columns:
            stats_values.append(f"位置范围: [{motion_df[pos_col].min():.3f}, {motion_df[pos_col].max():.3f}]")
        if torque_col in motion_df.columns:
            stats_values.append(f"力矩: {motion_df[torque_col].mean():.3f} (max:{motion_df[torque_col].abs().max():.3f})")
        if action_col in motion_df.columns:
            if scale > 0.0:
                scaled_action = get_scaled_action(motion_df, joint_name)
                if scaled_action is not None:
                    stats_values.append(f"Action范围: [{scaled_action.min():.3f}, {scaled_action.max():.3f}]")
            else:
                stats_values.append("Action: 不参与控制")
        stats_text += "\n".join(stats_values)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=STATS_FONTSIZE,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor=STATS_BGCOLOR, alpha=STATS_ALPHA))
        if lines:
            ax.legend(handles=lines, labels=labels, loc='upper right', fontsize=8, framealpha=0.9, ncol=len(lines))
        ax.set_xlim(TIME_RANGE[0], TIME_RANGE[1])
    for i in range(len(joint_indices), len(axs)):
        axs[i].set_visible(False)



# 页面4: 头颈关节角度随时间变化（替换原脊柱频谱分析）
def plot_neck_analysis(motion_df, ax):
    ax.clear()
    if len(motion_df) == 0:
        ax.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax.transAxes)
        return

    time_data = motion_df['time']

    colors = {'Neck_yaw': '#E69F00', 'Neck_pitch': '#CC79A7'}
    labels = {'Neck_yaw': 'Neck_yaw (偏航)', 'Neck_pitch': 'Neck_pitch (俯仰)'}

    lines = []
    legend_labels = []
    stats_lines = []

    for name in NECK_JOINTS:
        pos_col = f'{name}_pos'
        if pos_col not in motion_df.columns:
            stats_lines.append(f"{name}: 数据缺失 ({pos_col})")
            continue

        vals = motion_df[pos_col].values
        line, = ax.plot(time_data, vals, '-',
                        color=colors[name], linewidth=1.8, alpha=0.9,
                        label=labels[name])
        lines.append(line)
        legend_labels.append(labels[name])

        stats_lines.append(
            f"{name}: mean={vals.mean():+.4f}, "
            f"range=[{vals.min():+.4f}, {vals.max():+.4f}] rad, "
            f"std={vals.std():.4f}"
        )

    ax.axhline(y=0.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)

    ax.set_xlabel('时间 (秒)', fontsize=11)
    ax.set_ylabel('关节角 (弧度)', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(TIME_RANGE[0], TIME_RANGE[1])

    if lines:
        ax.legend(handles=lines, labels=legend_labels,
                  loc='upper right', fontsize=10, framealpha=0.9)

    if stats_lines:
        stats_text = (
            f"分析时间 ({START_TIME}-{END_TIME}s)\n"
            f"──────────────────\n"
            + "\n".join(stats_lines)
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=STATS_FONTSIZE, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=STATS_BGCOLOR, alpha=STATS_ALPHA))

    ax.set_title(f'头颈关节角度随时间变化 ({START_TIME}-{END_TIME}s)',
                 fontsize=13, fontweight='bold')



# 页面5: 地反力分析（保持不变）
def plot_ground_reaction_forces(forces: dict, impulse_data: dict, ax):
    ax.clear()
    time = np.linspace(TIME_RANGE[0], TIME_RANGE[1], len(next(iter(forces.values()))['x']))

    fig = ax.get_figure()
    bbox = ax.get_position()
    ax.set_frame_on(False)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    gs = GridSpec(2, 1, figure=fig, top=bbox.y0 + bbox.height, bottom=bbox.y0,
                  left=bbox.x0, right=bbox.x0 + bbox.width, hspace=0.3)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0])
    quarter_weight = ROBOT_MASS * GRAVITY / 4

    # FL
    fl = forces['FL']
    fl_i = impulse_data.get('FL', {})
    fl_stats = (f"FL (左前腿):\n"
                f"  x: [{np.min(fl['x']):.2f}, {np.max(fl['x']):.2f}] N\n"
                f"  z: [{np.min(fl['z']):.2f}, {np.max(fl['z']):.2f}] N\n"
                f"  xz合力: [{np.min(fl['xz']):.2f}, {np.max(fl['xz']):.2f}] N (平均 {np.mean(fl['xz']):.2f} N)\n"
                f"  总冲量 X: {fl_i.get('I_x', float('nan')):.3f} Ns, Z: {fl_i.get('I_z', float('nan')):.3f} Ns\n"
                f"  每周期冲量 X: {fl_i.get('avg_I_x', float('nan')):.3f} Ns, Z: {fl_i.get('avg_I_z', float('nan')):.3f} Ns")
    ax_top.plot(time, fl['x'], 'b-', linewidth=1.5, label='FL_x')
    ax_top.plot(time, fl['z'], 'g-', linewidth=1.5, label='FL_z (向上为正)')
    ax_top.plot(time, fl['xz'], 'r-', linewidth=1.5, label='FL_xz (合力)')
    ax_top.axhline(y=quarter_weight, color='gray', linestyle='--', linewidth=1.0, alpha=0.5, label=f'理论支撑 {quarter_weight:.2f}N')
    ax_top.text(0.02, 0.98, fl_stats, transform=ax_top.transAxes, fontsize=STATS_FONTSIZE,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor=STATS_BGCOLOR, alpha=STATS_ALPHA))
    ax_top.set_xlim(TIME_RANGE[0], TIME_RANGE[1])
    ax_top.set_ylabel('力 (N)', fontsize=10)
    ax_top.set_title('FL (左前腿) 地反力', fontsize=13, fontweight='bold')
    ax_top.legend(loc='upper right', fontsize=8)
    ax_top.grid(True, alpha=0.3)

    # HR
    hr = forces['HR']
    hr_i = impulse_data.get('HR', {})
    hr_stats = (f"HR (右后腿):\n"
                f"  x: [{np.min(hr['x']):.2f}, {np.max(hr['x']):.2f}] N\n"
                f"  z: [{np.min(hr['z']):.2f}, {np.max(hr['z']):.2f}] N\n"
                f"  xz合力: [{np.min(hr['xz']):.2f}, {np.max(hr['xz']):.2f}] N (平均 {np.mean(hr['xz']):.2f} N)\n"
                f"  总冲量 X: {hr_i.get('I_x', float('nan')):.3f} Ns, Z: {hr_i.get('I_z', float('nan')):.3f} Ns\n"
                f"  每周期冲量 X: {hr_i.get('avg_I_x', float('nan')):.3f} Ns, Z: {hr_i.get('avg_I_z', float('nan')):.3f} Ns")
    ax_bottom.plot(time, hr['x'], 'b-', linewidth=1.5, label='HR_x')
    ax_bottom.plot(time, hr['z'], 'g-', linewidth=1.5, label='HR_z (向上为正)')
    ax_bottom.plot(time, hr['xz'], 'r-', linewidth=1.5, label='HR_xz (合力)')
    ax_bottom.axhline(y=quarter_weight, color='gray', linestyle='--', linewidth=1.0, alpha=0.5, label=f'理论支撑 {quarter_weight:.2f}N')
    ax_bottom.text(0.02, 0.98, hr_stats, transform=ax_bottom.transAxes, fontsize=STATS_FONTSIZE,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor=STATS_BGCOLOR, alpha=STATS_ALPHA))
    ax_bottom.set_xlim(TIME_RANGE[0], TIME_RANGE[1])
    ax_bottom.set_xlabel('时间 (秒)', fontsize=10)
    ax_bottom.set_ylabel('力 (N)', fontsize=10)
    ax_bottom.set_title('HR (右后腿) 地反力', fontsize=13, fontweight='bold')
    ax_bottom.legend(loc='upper right', fontsize=8)
    ax_bottom.grid(True, alpha=0.3)



# 页面6: 足端轨迹分析（保持不变）
def plot_foot_trajectory(motion_df, ax, cycle_avg_cache: dict, stance_duty_data: Optional[dict] = None):
    ax.clear()
    if len(motion_df) == 0:
        ax.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax.transAxes)
        return
    foot_names = ['FR', 'FL', 'HR', 'HL']
    foot_labels = ['FR (右前)', 'FL (左前)', 'HR (右后)', 'HL (左后)']
    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']
    if not all(f'foot_{name}_x' in motion_df.columns for name in foot_names):
        ax.text(0.5, 0.5, '未找到足端位置数据', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('足端轨迹数据不可用')
        return

    fig = ax.get_figure()
    bbox = ax.get_position()
    ax.set_frame_on(False); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    gs = GridSpec(2, 2, figure=fig, top=bbox.y0 + bbox.height, bottom=bbox.y0,
                  left=bbox.x0, right=bbox.x0 + bbox.width, hspace=0.4, wspace=0.35)

    for i, (name, label, color) in enumerate(zip(foot_names, foot_labels, colors)):
        row, col = i // 2, i % 2
        sub_ax = fig.add_subplot(gs[row, col])
        rx = motion_df[f'foot_{name}_rx'].values
        rz = motion_df[f'foot_{name}_z'].values
        sub_ax.plot(rx, rz, '-', color=color, linewidth=0.8, alpha=0.4)
        res = cycle_avg_cache[name]
        if res is not None and res['n_cycles'] >= 2:
            rx_smooth = res['rx_smooth']; rz_smooth = res['rz_smooth']
            sub_ax.plot(rx_smooth, rz_smooth, '-', color=color, linewidth=2.2, alpha=0.95, label='均值(平滑)')
            ground_mask = rz_smooth < 0.002
            g_str = f"触地X: {rx_smooth[ground_mask].max() - rx_smooth[ground_mask].min():.3f}m" if np.any(ground_mask) else "触地X: 无"
        else:
            g_str = "周期不足"

        rx_range = (float(np.min(rx)), float(np.max(rx)))
        rz_range = (float(np.min(rz)), float(np.max(rz)))

        stats_text = (
            f"X范围: [{rx_range[0]:.3f}, {rx_range[1]:.3f}] m\n"
            f"Z范围: [{rz_range[0]:.3f}, {rz_range[1]:.3f}] m\n"
            f"跨步X: {abs(rx_range[1]-rx_range[0]):.3f} m | 抬脚Z: {abs(rz_range[1]-rz_range[0]):.3f} m\n"
        )

        if stance_duty_data is not None and name in stance_duty_data and stance_duty_data[name] is not None:
            duty_data = stance_duty_data[name]
            stats_text += (
                f"支撑相占比: {duty_data['duty_factor_mean']:.1%} ± {duty_data['duty_factor_std']:.1%}\n"
                f"支撑时长: {duty_data['stance_duration_mean']:.3f}s ± {duty_data['stance_duration_std']:.3f}s\n"
                f"摆动时长: {duty_data['swing_duration_mean']:.3f}s ± {duty_data['swing_duration_std']:.3f}s"
            )
        else:
            stats_text += "支撑相: 数据缺失"

        stats_text += f"\n{g_str}"

        sub_ax.text(0.02, 0.98, stats_text, transform=sub_ax.transAxes, fontsize=STATS_FONTSIZE,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor=STATS_BGCOLOR, alpha=STATS_ALPHA))
        sub_ax.set_title(f'{label}', fontsize=10)
        sub_ax.set_xlabel('相对X (m)', fontsize=8); sub_ax.set_ylabel('Z (m)', fontsize=8)
        sub_ax.legend(fontsize=7, loc='lower right')
        sub_ax.grid(True, alpha=0.3)
        sub_ax.set_aspect('equal', adjustable='datalim')
    ax.text(0.5, 1.06, f'足端轨迹 (原始+固定周期平均) [{TIME_RANGE[0]}-{TIME_RANGE[1]}s]',
            ha='center', fontsize=13, fontweight='bold', transform=ax.transAxes)



# 页面7：支撑相分析（保持不变）
def plot_stance_phase(forces: dict, ax):
    ax.clear()
    if forces is None:
        ax.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax.transAxes)
        return

    time = np.linspace(TIME_RANGE[0], TIME_RANGE[1], len(next(iter(forces.values()))['z']))
    colors = {'FL': '#377EB8', 'FR': '#E41A1C', 'HR': '#4DAF4A', 'HL': '#984EA3'}
    height = 0.8

    stance_masks = {}
    for foot in ['FL', 'FR', 'HR', 'HL']:
        stance_masks[foot] = forces[foot]['z'] > CONTACT_THRESHOLD

    y_centers_feet = {'FL': 5.0, 'FR': 4.0, 'HL': 3.0, 'HR': 2.0}
    y_air = 1.0

    low_air = y_air - height/2
    high_air = y_air + height/2
    all_air = np.ones(len(time), dtype=bool)
    for foot in ['FL', 'FR', 'HR', 'HL']:
        all_air = all_air & ~stance_masks[foot]
    air_starts, air_ends = [], []
    in_air = False
    start_idx = 0
    for i in range(len(all_air)):
        if all_air[i] and not in_air:
            start_idx = i
            in_air = True
        elif not all_air[i] and in_air:
            air_starts.append(time[start_idx])
            air_ends.append(time[i - 1])
            in_air = False
    if in_air:
        air_starts.append(time[start_idx])
        air_ends.append(time[-1])
    for i in range(len(air_starts)):
        air_starts[i] = max(TIME_RANGE[0], air_starts[i] - DT)
        air_ends[i]   = min(TIME_RANGE[1], air_ends[i]   + DT)
    for t_start, t_end in zip(air_starts, air_ends):
        mask = (time >= t_start) & (time <= t_end)
        ax.fill_between(time, low_air, high_air, where=mask, color='red', alpha=0.25)

    for foot in ['FL', 'FR', 'HR', 'HL']:
        y_center = y_centers_feet[foot]
        low = y_center - height/2
        high = y_center + height/2
        ax.fill_between(time, low, high, where=stance_masks[foot], color=colors[foot], alpha=0.6)

    ax.set_xlim(TIME_RANGE[0], TIME_RANGE[1])
    y_max = 5.5
    ax.set_ylim(0.5, y_max)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['腾空', 'HR', 'HL', 'FR', 'FL'])
    ax.set_xlabel('时间 (秒)')
    ax.set_title('支撑相分析 (最下方红色 = 四足均离地)', fontsize=13, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)



# ============ 主可视化类 ============
class CSVDataAnalyzer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.motion_df = None
        self.metrics = None
        self.forces = None
        self.impulse_data = None
        self.cycle_avg_cache = None
        self.stance_duty_data = None
        self.current_page = 0
        self.PAGES = [
            "基座位置分析",
            "关节0-3分析",
            "关节4-7分析",
            "关节8-11分析",
            "头颈关节分析",
            "地反力分析",
            "足端轨迹分析",
            "支撑相分析",
        ]
        self.load_data()
        self.create_ui()
        self.update_plot()


    def load_data(self):
        df = pd.read_csv(self.csv_path)
        print(f"CSV文件加载成功: {self.csv_path}")
        print(f"数据形状: {df.shape}")
        print(f"数据步数: {len(df)}")
        if df is None:
            return

        new_columns = {'time': df['step'] * DT,}
        for name in FOOT_NAMES:
            new_columns[f'foot_{name}_rx'] = df[f'foot_{name}_x'] - df['base_pos_x']

        df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
        self.motion_df_full = df.copy()
        self.motion_df = df[(df['time'] >= START_TIME) & (df['time'] <= END_TIME)].copy()
        if len(self.motion_df) == 0:
            print("分析区间内无数据！")
            return

        # F_body heading = body+X 方向(≈90°) → 物理前向需要 -π/2
        if 'f_body_heading' in self.motion_df.columns:
            forward_heading = self.motion_df['f_body_heading'] - np.pi / 2
        else:
            print("[警告] 未找到 f_body_heading 列，回退使用基座 heading")
            forward_heading = self.motion_df['heading'] - np.pi / 2

        self.motion_df['forward_speed'] = (
            self.motion_df['base_lin_vel_x'] * np.cos(forward_heading) +
            self.motion_df['base_lin_vel_y'] * np.sin(forward_heading)
        )

        # ---------- 打印稳定段高度与跟踪误差统计 ----------
        stable = self.motion_df[self.motion_df['time'] >= START_TIME]
        if len(stable) > 0:
            f_h_arr = stable['F_body_height'].values * 1000.0
            h_h_arr = stable['H_body_height'].values * 1000.0
            f_cmd_arr = stable['height_F_command'].values * 1000.0 \
                if 'height_F_command' in stable.columns else None
            h_cmd_arr = stable['height_H_command'].values * 1000.0 \
                if 'height_H_command' in stable.columns else None

            print("=" * 60)
            print(f"[稳定段统计] 时间 >= {START_TIME}s, 共 {len(stable)} 步")
            print(f"F_body: mean={f_h_arr.mean():.1f} mm, range=[{f_h_arr.min():.1f}, {f_h_arr.max():.1f}] mm")
            print(f"H_body: mean={h_h_arr.mean():.1f} mm, range=[{h_h_arr.min():.1f}, {h_h_arr.max():.1f}] mm")
            if f_cmd_arr is not None:
                f_err = f_h_arr - f_cmd_arr
                print(f"F 跟踪误差: mean={f_err.mean():+.2f} mm, std={f_err.std():.2f} mm")
            if h_cmd_arr is not None:
                h_err = h_h_arr - h_cmd_arr
                print(f"H 跟踪误差: mean={h_err.mean():+.2f} mm, std={h_err.std():.2f} mm")
            print("=" * 60)

        self.metrics = calculate_metrics(self.motion_df)
        self.forces = extract_contact_forces_all(self.motion_df)
        if self.forces is None:
            print("[力数据] 缺失接触力列！")
        if self.forces is not None:
            self.impulse_data = calculate_impulse(self.motion_df, self.forces)
            cycles = define_cycles_by_period(TIME_RANGE[0], TIME_RANGE[1], T)
            self.stance_duty_data = compute_stance_duty_factors(self.motion_df, self.forces, cycles)
            if self.stance_duty_data:
                fore_duty = []
                for foot in ['FL', 'FR']:
                    if foot in self.stance_duty_data and self.stance_duty_data[foot] is not None:
                        fore_duty.append(self.stance_duty_data[foot]['duty_factor_mean'])
                fore_avg = np.mean(fore_duty) if fore_duty else 0.0

                hind_duty = []
                for foot in ['HL', 'HR']:
                    if foot in self.stance_duty_data and self.stance_duty_data[foot] is not None:
                        hind_duty.append(self.stance_duty_data[foot]['duty_factor_mean'])
                hind_avg = np.mean(hind_duty) if hind_duty else 0.0

                print(f"平均前腿支撑相占比: {fore_avg:.1%}")
                print(f"平均后腿支撑相占比: {hind_avg:.1%}")
        else:
            self.stance_duty_data = None
        self.cycle_avg_cache = precompute_foot_cycle_average(self.motion_df)
        self.print_foot_ground_statistics()


    def print_foot_ground_statistics(self):
        if self.cycle_avg_cache is None:
            return
        ground_spans = {}
        for name in FOOT_NAMES:
            res = self.cycle_avg_cache[name]
            if res is None:
                ground_spans[name] = None; continue
            rz_smooth = res['rz_smooth']; rx_smooth = res['rx_smooth']
            ground_mask = rz_smooth < 0.002
            if np.any(ground_mask):
                ground_spans[name] = rx_smooth[ground_mask].max() - rx_smooth[ground_mask].min()
            else:
                ground_spans[name] = None
        fore_spans = [ground_spans[n] for n in ['FL', 'FR'] if ground_spans[n] is not None]
        hind_spans = [ground_spans[n] for n in ['HL', 'HR'] if ground_spans[n] is not None]
        avg_fore = np.mean(fore_spans) if fore_spans else 0.0
        avg_hind = np.mean(hind_spans) if hind_spans else 0.0
        total_span = avg_fore + avg_hind
        print(f"平均前腿触地跨度 = {avg_fore:.3f} m")
        print(f"平均后腿触地跨度 = {avg_hind:.3f} m")
        print(f"平均总触地跨度 = {total_span:.3f} m")


    def create_ui(self):
        if self.motion_df is None:
            print("无法创建UI: 数据加载失败")
            return
        self.fig = plt.figure(figsize=(14, 10))
        plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12, hspace=0.5)
        ax_prev = plt.axes((0.30, 0.03, 0.15, 0.04))
        ax_next = plt.axes((0.55, 0.03, 0.15, 0.04))
        self.btn_prev = Button(ax_prev, '上一页')
        self.btn_next = Button(ax_next, '下一页')
        self.btn_prev.on_clicked(self.prev_page)
        self.btn_next.on_clicked(self.next_page)
        self.page_text = self.fig.text(0.5, 0.04, f'页面 {self.current_page+1}/{len(self.PAGES)}: {self.PAGES[self.current_page]}',
                                       ha='center', fontsize=12, fontweight='bold')
        self.pages = []
        ax = self.fig.add_subplot(111); self.pages.append(ax)
        for _ in range(3):
            page = [self.fig.add_subplot(4, 1, i+1) for i in range(4)]
            self.pages.append(page)
        for _ in range(4):
            ax = self.fig.add_subplot(111); self.pages.append(ax)


    def update_plot(self):
        if self.motion_df is None: return
        fig = self.pages[0].get_figure()
        pages_axes = []
        for page in self.pages:
            if isinstance(page, list):
                pages_axes.extend(page)
            else:
                pages_axes.append(page)
        for ax in fig.axes[:]:
            if ax not in pages_axes and ax not in [self.btn_prev.ax, self.btn_next.ax]:
                fig.delaxes(ax)
        for page in self.pages:
            if isinstance(page, list):
                for ax in page: ax.set_visible(False); ax.set_frame_on(False)
            else:
                page.set_visible(False); page.set_frame_on(False)
        self.page_text.set_text(f'页面 {self.current_page+1}/{len(self.PAGES)}: {self.PAGES[self.current_page]}')
        target = self.pages[self.current_page]
        if isinstance(target, list):
            for ax in target: ax.set_visible(True); ax.set_frame_on(True)
        else:
            target.set_visible(True); target.set_frame_on(True)

        if self.current_page == 0:
            plot_base_analysis(self.motion_df_full, target, self.metrics) # type: ignore
        elif self.current_page == 1:
            plot_joint_analysis(self.motion_df, target, [0,1,2,3])
        elif self.current_page == 2:
            plot_joint_analysis(self.motion_df, target, [4,5,6,7])
        elif self.current_page == 3:
            plot_joint_analysis(self.motion_df, target, [8,9,10,11])
        elif self.current_page == 4:
            plot_neck_analysis(self.motion_df, target)
        elif self.current_page == 5:
            if self.forces is not None and self.impulse_data is not None:
                plot_ground_reaction_forces(self.forces, self.impulse_data, target)
            else:
                target.clear()
                target.text(0.5, 0.5, '地反力数据缺失', ha='center', va='center', transform=target.transAxes) # type: ignore
        elif self.current_page == 6:
            plot_foot_trajectory(self.motion_df, target, self.cycle_avg_cache, self.stance_duty_data) # type: ignore
        elif self.current_page == 7:
            if self.forces is not None:
                plot_stance_phase(self.forces, target)
            else:
                target.clear()
                target.text(0.5, 0.5, '地反力数据缺失', ha='center', va='center', transform=target.transAxes) # type: ignore

        self.fig.canvas.draw_idle()


    def next_page(self, event):
        self.current_page = (self.current_page + 1) % len(self.PAGES)
        self.update_plot()


    def prev_page(self, event):
        self.current_page = (self.current_page - 1) % len(self.PAGES)
        self.update_plot()



def main():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    CSVDataAnalyzer(CSV_PATH)
    plt.show()



if __name__ == "__main__":
    main()