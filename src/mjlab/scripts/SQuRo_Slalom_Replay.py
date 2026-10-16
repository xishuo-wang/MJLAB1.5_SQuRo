# 手调摆线转弯步态在 Mjlab-SQuRo-Slalom 环境中的开环回放与参考对照
# 迁移来源: D:\Code\SQuRo-MuJoCo\Loco_Turn\Trun_Cyc_Min.py (力矩 PD 版, Mouse_Tor.xml)
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
import tyro

import mjlab.tasks  # noqa: F401  触发任务注册
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.scripts._diagnostics import SLALOM_HAND_TUNED_REPLAY_DIR
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Slalom.mdp.command import SlalomCommandCfg
from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import CURVATURE_TARGET_MAX
from mjlab.tasks.SQuRo_Slalom.mdp.indices import (
    _ACT_JOINT_NAMES,
    _MODEL_INDICES,
    FOOT_NAMES,
    FOOT_SITE_NAMES,
    resolve_foot_force_ids,
    resolve_model_indices,
)
from mjlab.tasks.SQuRo_Slalom.mdp.path import compute_path_ref
from mjlab.tasks.SQuRo_Slalom.mdp.reference import (
    _BIO_DATA_DIR,
    get_reference_joint_state,
)
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer

# 手调步态参数, 逐项对应 Trun_Cyc_Min.py 的同名常量
HAND_GAIT_FREQ = 1.0
HAND_SWING_RATIO = 0.4
HAND_START_TIME = 0.5
HAND_PHASE_LAG = {"FL": 0.0, "FR": 0.5, "HL": 0.5, "HR": 0.0}
HAND_LEGS = {
    "FL": {"stride": 0.05, "height": 0.01, "body_height": 0.060, "x_offset": -0.015, "is_front": True},
    "FR": {"stride": -0.0, "height": 0.01, "body_height": 0.053, "x_offset": -0.0, "is_front": True},
    "HL": {"stride": 0.05, "height": 0.015, "body_height": 0.058, "x_offset": -0.025, "is_front": False},
    "HR": {"stride": -0.0, "height": 0.015, "body_height": 0.05, "x_offset": -0.01, "is_front": False},
}
HAND_LEG_JOINTS = {
    "FL": ("FL_shoulder_joint", "FL_elbow_joint"),
    "FR": ("FR_shoulder_joint", "FR_elbow_joint"),
    "HL": ("HL_hip_joint", "HL_knee_joint"),
    "HR": ("HR_hip_joint", "HR_knee_joint"),
}
# 手调脚本的脊柱目标: F_spine1=0.65, F_body=0.9, H_spine1=-0.65, H_body=0.7
HAND_SPINE = {
    "F_spine1_joint": 0.65,
    "F_body_joint": 0.9,
    "H_spine1_joint": -0.65,
    "H_body_joint": 0.7,
}
# 逆运动学关节偏置: 手调脚本调用的是 inverse_kinematics_old, 当前任务用另一组标定
HAND_IK_BIAS = {
    "legacy": {"front": (0.888, 2.648), "rear": (4.325, 1.794)},
    "current": {"front": (0.868, 2.643), "rear": (4.35, 1.75)},
}
FOOT_ORDER = tuple(FOOT_NAMES)
COL_OF_JOINT = {name: col for col, name in enumerate(_ACT_JOINT_NAMES)}
SPINE_COLS = [_ACT_JOINT_NAMES.index(name) for name in HAND_SPINE]
NECK_COLS = [_ACT_JOINT_NAMES.index(name) for name in ("Neck_yaw_joint", "Neck_pitch_joint")]
LEG_COLS = [COL_OF_JOINT[name] for leg in FOOT_ORDER for name in HAND_LEG_JOINTS[leg]]
ANALYSIS_WARMUP_S = 1.5


# 摆线足端轨迹, 与手调脚本逐字一致 (x 纵向, z 竖直, 相对髋关节)
def cycloid_foot(
    t_mod: float, t_swing: float, t_stance: float,
    stride: float, height: float, body_height: float, x_offset: float,
) -> tuple[float, float]:
    if t_mod <= t_swing:
        t_norm = t_mod / t_swing
        x = stride * (t_norm - math.sin(2.0 * math.pi * t_norm) / (2.0 * math.pi))
        z = 0.5 * height * (1.0 - math.cos(2.0 * math.pi * t_norm))
    else:
        t_st = (t_mod - t_swing) / t_stance
        x = stride * (1.0 - t_st)
        z = 0.0
    return x + x_offset, z - body_height


# 两连杆逆运动学, 几何与手调脚本一致, 关节偏置由 ik_set 选择
def leg_inverse_kinematics(
    x: float, y: float, is_front: bool, bias: tuple[float, float],
) -> tuple[float, float]:
    l1, l2 = (0.040, 0.040) if is_front else (0.040, 0.036)
    radius = math.hypot(x, y)
    k = (l2**2 - x**2 - y**2 - l1**2) / (2.0 * l1)
    theta = math.atan2(y, x)
    phi = math.acos(max(-1.0, min(1.0, k / radius)))
    if is_front:
        a1 = theta + phi
        a2 = math.atan2((y + l1 * math.sin(a1)) / l2, (-x - l1 * math.cos(a1)) / l2)
        if a2 > 2.0:
            a2 -= 2.0 * math.pi
        return a1 - bias[0], -(a2 + bias[1])
    a1 = theta - phi
    a2 = math.atan2((x + l1 * math.cos(a1)) / l2, (y + l1 * math.sin(a1)) / l2)
    if a2 > 2.0:
        a2 -= 2.0 * math.pi
    return a1 + bias[0], -(a2 + bias[1])


# 按相位算四腿关节角, 返回执行器序目标角
def hand_leg_targets(motion_phase: float, gait_freq: float, ik_set: str) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    target = np.zeros(len(_ACT_JOINT_NAMES))
    feet: dict[str, tuple[float, float]] = {}
    t_cycle = 1.0 / gait_freq
    bias = HAND_IK_BIAS[ik_set]
    for leg in FOOT_ORDER:
        params = HAND_LEGS[leg]
        t_swing = t_cycle * HAND_SWING_RATIO
        t_stance = t_cycle - t_swing
        leg_phase = (motion_phase + HAND_PHASE_LAG[leg]) % 1.0
        x_foot, z_foot = cycloid_foot(
            leg_phase * t_cycle, t_swing, t_stance,
            params["stride"], params["height"], params["body_height"], params["x_offset"],
        )
        feet[leg] = (x_foot, z_foot)
        is_front = bool(params["is_front"])
        hip, knee = leg_inverse_kinematics(x_foot, z_foot, is_front, bias["front" if is_front else "rear"])
        target[COL_OF_JOINT[HAND_LEG_JOINTS[leg][0]]] = hip
        target[COL_OF_JOINT[HAND_LEG_JOINTS[leg][1]]] = knee
    return target, feet


# 读取参考足端轨迹表, 返回相位网格上的 (纵向, 竖直)
def load_reference_foot(is_front: bool, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(_BIO_DATA_DIR / ("Trot_F.csv" if is_front else "Trot_H.csv"))
    phase_grid = np.linspace(0.0, 1.0, resolution)
    longitudinal = np.interp(phase_grid, table["Phase"].to_numpy(), table["Y_mean"].to_numpy())
    vertical = np.interp(phase_grid, table["Phase"].to_numpy(), table["Z_mean"].to_numpy())
    return longitudinal, vertical


# 当前参考在给定曲率下对某个足端的纵向缩放 (内侧腿按 |κ| 归一化线性缩小)
def reference_inner_scale(curvature: float, leg: str) -> float:
    inner_legs = ("FL", "HL") if curvature >= 0.0 else ("FR", "HR")
    return 1.0 - abs(curvature) / CURVATURE_TARGET_MAX if leg in inner_legs else 1.0


# 平面点集拟合圆 (Kasa 代数法), 返回圆心, 半径与径向残差 RMS
def fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    design = np.column_stack([x, y, np.ones_like(x)])
    solution, *_ = np.linalg.lstsq(design, x**2 + y**2, rcond=None)
    cx, cy = solution[0] / 2.0, solution[1] / 2.0
    radius = math.sqrt(max(solution[2] + cx * cx + cy * cy, 0.0))
    residual = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - radius) ** 2)))
    return float(cx), float(cy), float(radius), residual


# 按相位网格重采样某个通道, 用于手调目标与当前参考的对齐比较
def resample_by_phase(phases: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(phases)
    wrapped = np.concatenate([phases[order], phases[order][:1] + 1.0])
    stacked = np.concatenate([values[order], values[order][:1]], axis=0)
    out = np.empty((len(grid), values.shape[1]))
    for col in range(values.shape[1]):
        out[:, col] = np.interp(grid, wrapped, stacked[:, col])
    return out


class HandTunedReplayPolicy:
    # 开环手调策略: 目标角来自摆线轨迹与逆运动学, 动作按执行器序换算
    def __init__(self, env: ManagerBasedRlEnv, ik_set: str, gait_freq: float, start_time: float) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        resolve_model_indices(self.robot)
        self.ik_set = ik_set
        self.gait_freq = gait_freq
        self.start_time = start_time

        self.action_term = cast(JointPositionAction, env.action_manager.get_term("joint_pos"))
        self.joint_ids = list(_MODEL_INDICES.joint_ids)
        joint_ids = self.joint_ids
        column_of_joint = {joint_id: col for col, joint_id in enumerate(joint_ids)}
        self.action_columns = [column_of_joint[i] for i in self.action_term.target_ids.tolist()]
        if len(self.action_columns) != len(joint_ids):
            raise ValueError("动作关节数与执行器序关节数不一致")
        offset = torch.as_tensor(self.action_term.offset).reshape(-1)
        raw_scale = self.action_term.scale
        scale_tensor = torch.as_tensor(raw_scale).reshape(-1)
        if scale_tensor.numel() != 1:
            if torch.any(scale_tensor == 0):
                raise ValueError("动作缩放不能为零")
            self.scale: float | torch.Tensor = scale_tensor.to(offset.device)
        else:
            self.scale = float(scale_tensor[0])
            if self.scale == 0.0:
                raise ValueError("动作缩放不能为零")
        self.offset = offset
        # 站姿目标 = 动作偏置 (即模型 INIT_STATE), 按执行器序还原
        self.default_target = np.zeros(len(_ACT_JOINT_NAMES), dtype=np.float64)
        for act_col, ref_col in enumerate(self.action_columns):
            self.default_target[ref_col] = float(offset[act_col])
        self.default_target[SPINE_COLS] = 0.0

        self.spine_target = np.array([HAND_SPINE[name] for name in HAND_SPINE])
        self.foot_site_names = list(FOOT_SITE_NAMES)
        self._foot_site_ids: torch.Tensor | None = None
        self.records: list[dict[str, Any]] = []
        self._last_step: tuple[int, int] | None = None
        self.command = np.zeros(5)

    def _ensure_foot_sites(self) -> None:
        if self._foot_site_ids is None:
            self._foot_site_ids, _ = self.robot.find_sites(self.foot_site_names, preserve_order=True)

    # 手调目标角: 起步前保持站姿, 起步后叠加摆线腿与固定脊柱角
    def target(self, elapsed: float) -> tuple[np.ndarray, float, dict[str, tuple[float, float]]]:
        if elapsed < self.start_time:
            return self.default_target.copy(), -1.0, {}
        motion_time = elapsed - self.start_time
        motion_phase = (motion_time * self.gait_freq) % 1.0
        target = self.default_target.copy()
        leg_target, feet = hand_leg_targets(motion_phase, self.gait_freq, self.ik_set)
        target[LEG_COLS] = leg_target[LEG_COLS]
        target[SPINE_COLS] = self.spine_target
        return target, motion_phase, feet

    # 记录一步的手调目标, 实际状态, 当前参考与基座运动
    def record(self, target: np.ndarray, motion_phase: float) -> None:
        env = self.env
        step = (int(env.common_step_counter), int(env.episode_length_buf[0]))
        if step == self._last_step:
            return
        self._last_step = step
        self._ensure_foot_sites()
        idx = _MODEL_INDICES

        reference, reference_vel = get_reference_joint_state(env)
        ref = reference[0].detach().cpu().numpy()
        ref_vel = reference_vel[0].detach().cpu().numpy()
        pos = self.robot.data.joint_pos[0, self.joint_ids].detach().cpu().numpy()
        vel = self.robot.data.joint_vel[0, self.joint_ids].detach().cpu().numpy()

        base_pos = self.robot.data.root_link_pos_w[0].detach().cpu().numpy()
        base_lin_vel = self.robot.data.root_link_lin_vel_w[0].detach().cpu().numpy()
        base_ang_vel = self.robot.data.root_link_ang_vel_w[0].detach().cpu().numpy()
        foot_pos = self.robot.data.site_pos_w[0, self._foot_site_ids].detach().cpu().numpy()
        x_ref, y_ref, vx_des, vy_des, path_heading = compute_path_ref(env)
        cmd = env.command_manager.get_command("slalom_cmd")[0].detach().cpu().numpy()
        self.command = cmd

        quat = self.robot.data.root_link_quat_w[0]
        w, qx, qy, qz = (float(v) for v in quat)
        yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        f_quat = self.robot.data.body_link_quat_w[0, idx.f_body_id]
        fw, fx, fy, fz = (float(v) for v in f_quat)

        row: dict[str, Any] = {
            "step": float(step[0]),
            "time": float(step[0]) * env.step_dt,
            "episode_step": float(step[1]),
            "motion_phase": motion_phase,
            "ref_phase": float(env._ref_phase[0]) if hasattr(env, "_ref_phase") else float("nan"),
            "base_pos_x": float(base_pos[0]),
            "base_pos_y": float(base_pos[1]),
            "base_pos_z": float(base_pos[2]),
            "base_lin_vel_x": float(base_lin_vel[0]),
            "base_lin_vel_y": float(base_lin_vel[1]),
            "base_lin_vel_z": float(base_lin_vel[2]),
            "base_ang_vel_z": float(base_ang_vel[2]),
            "base_yaw": yaw,
            "f_body_heading": math.atan2(2.0 * (fw * fz + fx * fy), 1.0 - 2.0 * (fy * fy + fz * fz)),
            "ref_x": float(x_ref[0]),
            "ref_y": float(y_ref[0]),
            "ref_vx": float(vx_des[0]),
            "ref_vy": float(vy_des[0]),
            "ref_heading": float(path_heading[0]),
            "vel_command_x": float(cmd[0]),
            "height_f_command": float(cmd[1]),
            "height_h_command": float(cmd[2]),
            "gait_freq_command": float(cmd[3]),
            "curvature_command": float(cmd[4]),
        }
        contact_sensor = env.scene["feet_ground_contact"]
        force_ids = resolve_foot_force_ids(contact_sensor)
        contact = contact_sensor.data.force.reshape(
            env.num_envs, len(contact_sensor.primary_names), -1, 3
        ).sum(dim=2)[0].detach().cpu().numpy()
        for i, leg in enumerate(FOOT_ORDER):
            row[f"foot_{leg}_x"] = float(foot_pos[i, 0])
            row[f"foot_{leg}_y"] = float(foot_pos[i, 1])
            row[f"foot_{leg}_z"] = float(foot_pos[i, 2])
            row[f"contact_{leg}_mag"] = float(np.linalg.norm(contact[force_ids[i]]))
        for col, name in enumerate(_ACT_JOINT_NAMES):
            clean = name[:-6] if name.endswith("_joint") else name
            row[f"{clean}_pos"] = float(pos[col])
            row[f"{clean}_vel"] = float(vel[col])
            row[f"{clean}_ref_pos"] = float(ref[col])
            row[f"{clean}_ref_vel"] = float(ref_vel[col])
            row[f"{clean}_hand_target"] = float(target[col])
        self.records.append(row)

    def __call__(self, obs: Any) -> torch.Tensor:
        del obs
        elapsed = float(self.env.common_step_counter) * self.env.step_dt
        target, motion_phase, _ = self.target(elapsed)
        self.record(target, motion_phase)
        action_target = torch.tensor(target, dtype=torch.float32)[self.action_columns].to(self.offset.device)
        return ((action_target - self.offset) / self.scale).unsqueeze(0)


def summarize(records: list[dict[str, Any]], start_time: float, gait_freq: float) -> dict[str, Any]:
    data = pd.DataFrame(records)
    if data.empty:
        raise ValueError("没有记录到任何控制步")
    window = data[data["time"] >= start_time + ANALYSIS_WARMUP_S].reset_index(drop=True)
    if len(window) < 10:
        raise ValueError("有效分析窗口过短")
    dt = float(np.median(np.diff(window["time"].to_numpy())))
    duration = float(window["time"].iloc[-1] - window["time"].iloc[0])

    x = window["base_pos_x"].to_numpy()
    y = window["base_pos_y"].to_numpy()
    steps = np.hypot(np.diff(x), np.diff(y))
    path_length = float(steps.sum())
    displacement = float(math.hypot(x[-1] - x[0], y[-1] - y[0]))
    speed_path = path_length / duration
    speed_xy = float(np.mean(np.hypot(window["base_lin_vel_x"], window["base_lin_vel_y"])))
    heading = window["f_body_heading"].to_numpy()
    forward = window["base_lin_vel_x"].to_numpy() * np.cos(heading) + window["base_lin_vel_y"].to_numpy() * np.sin(heading)
    speed_forward = float(np.mean(forward))

    cx, cy, radius, residual = fit_circle(x, y)
    yaw_unwrapped = np.unwrap(window["base_yaw"].to_numpy())
    yaw_rate = (yaw_unwrapped[-1] - yaw_unwrapped[0]) / duration
    yaw_rate_z = float(np.mean(window["base_ang_vel_z"].to_numpy()))
    sign = 1.0 if yaw_rate >= 0.0 else -1.0
    curvature_fit = sign / radius if radius > 1e-9 else float("inf")
    curvature_kinematic = yaw_rate / speed_path if speed_path > 1e-9 else float("inf")

    rx, ry, rradius, rresidual = fit_circle(window["ref_x"].to_numpy(), window["ref_y"].to_numpy())
    reference_speed = float(np.mean(np.hypot(window["ref_vx"], window["ref_vy"])))

    grid = np.linspace(0.0, 1.0, 100, endpoint=False)
    hand = resample_by_phase(window["motion_phase"].to_numpy(), window[[f"{n[:-6]}_hand_target" for n in _ACT_JOINT_NAMES]].to_numpy(), grid)
    ref = resample_by_phase(window["ref_phase"].to_numpy() % 1.0, window[[f"{n[:-6]}_ref_pos" for n in _ACT_JOINT_NAMES]].to_numpy(), grid)
    residual_curves = hand - ref
    # 相位差扫描: 手调轨迹是否是当前参考的某个相位平移
    shifts = np.arange(100) / 100.0
    shift_error = [float(np.sqrt(np.mean((hand - np.roll(ref, int(round(s * 100)), axis=0)) ** 2))) for s in shifts]
    best_shift = float(shifts[int(np.argmin(shift_error))])

    def group_rms(columns: list[int]) -> float:
        return float(np.sqrt(np.mean(residual_curves[:, columns] ** 2)))

    foot_compare: dict[str, Any] = {}
    for leg in FOOT_ORDER:
        params = HAND_LEGS[leg]
        is_front = bool(params["is_front"])
        longitudinal, vertical = load_reference_foot(is_front, 100)
        scale = reference_inner_scale(float(window["curvature_command"].iloc[-1]), leg)
        t_cycle = 1.0 / gait_freq
        hand_x, hand_z = [], []
        for phase in grid:
            fx, fz = cycloid_foot(
                phase * t_cycle, t_cycle * HAND_SWING_RATIO, t_cycle * (1.0 - HAND_SWING_RATIO),
                params["stride"], params["height"], params["body_height"], params["x_offset"],
            )
            hand_x.append(fx)
            hand_z.append(fz)
        hand_x_arr = np.asarray(hand_x)
        hand_z_arr = np.asarray(hand_z)
        foot_compare[leg] = {
            "手调纵向行程_mm": float((hand_x_arr.max() - hand_x_arr.min()) * 1000.0),
            "参考纵向行程_mm": float((longitudinal * scale).max() * 1000.0 - (longitudinal * scale).min() * 1000.0),
            "手调竖直行程_mm": float((hand_z_arr.max() - hand_z_arr.min()) * 1000.0),
            "参考竖直行程_mm": float((vertical.max() - vertical.min()) * 1000.0),
            "手调平均离髋高度_mm": float(-hand_z_arr.mean() * 1000.0),
            "参考平均离髋高度_mm": float(-vertical.mean() * 1000.0),
            "参考内侧缩放": float(scale),
        }

    summary: dict[str, Any] = {
        "分析窗口_s": [float(window["time"].iloc[0]), float(window["time"].iloc[-1])],
        "控制周期_s": dt,
        "基座路径长度_m": path_length,
        "基座首末位移_m": displacement,
        "路径平均速度_mps": speed_path,
        "水平速度均值_mps": speed_xy,
        "机体前向速度均值_mps": speed_forward,
        "每步态周期路径长度_m": path_length / max(duration * gait_freq, 1e-9),
        "每步态周期净前进_m": abs(yaw_rate) * radius / max(gait_freq, 1e-9),
        "净前进速度_mps": abs(yaw_rate) * radius,
        "偏航角总变化_rad": float(yaw_unwrapped[-1] - yaw_unwrapped[0]),
        "平均偏航角速度_radps": yaw_rate,
        "平均Z角速度_radps": yaw_rate_z,
        "拟合圆": {
            "圆心_x_m": cx, "圆心_y_m": cy, "半径_m": radius,
            "半径_mm": radius * 1000.0, "径向残差_mm": residual * 1000.0,
        },
        "等效曲率_拟合_1pm": curvature_fit,
        "等效曲率_运动学_1pm": curvature_kinematic,
        "等效半径_拟合_mm": radius * 1000.0,
        "参考轨迹": {
            "圆心_x_m": rx, "圆心_y_m": ry, "半径_m": rradius, "半径_mm": rradius * 1000.0,
            "径向残差_mm": rresidual * 1000.0, "参考速度_mps": reference_speed,
        },
        # 参考速度律 (Phase 0): v = base × gait × (1 − 0.85·|κ|/20)
        "参考在实测曲率档速度_mps": 0.1 * max(1.0 - 0.85 * abs(curvature_fit) / CURVATURE_TARGET_MAX, 0.15),
        "关节对照_rad": {
            "腿RMS": group_rms(LEG_COLS),
            "脊柱RMS": group_rms(SPINE_COLS),
            "颈部RMS": group_rms(NECK_COLS),
            "整体RMS": group_rms(list(range(len(_ACT_JOINT_NAMES)))),
            "最佳相位平移": best_shift,
            "平移后整体RMS": float(min(shift_error)),
        },
        "足端对照": foot_compare,
    }
    return summary


# 保存 CSV 与四张诊断图, 并打印关键结论
def save_outputs(records: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path, gait_freq: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(records)
    data.to_csv(output_dir / "hand_tuned_replay.csv", index=False, encoding="utf-8-sig")
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    window = data[data["time"] >= summary["分析窗口_s"][0]]

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.plot(data.base_pos_x * 1000, data.base_pos_y * 1000, "-", linewidth=1.2, label="手调基座轨迹")
    ax.plot(window.ref_x * 1000, window.ref_y * 1000, "--", linewidth=1.0, label="当前参考路径")
    circle_x = summary["拟合圆"]["圆心_x_m"] * 1000
    circle_y = summary["拟合圆"]["圆心_y_m"] * 1000
    circle_r = summary["拟合圆"]["半径_mm"]
    theta = np.linspace(0.0, 2.0 * math.pi, 200)
    ax.plot(circle_x + circle_r * np.cos(theta), circle_y + circle_r * np.sin(theta),
            ":", color="tab:red", label=f"拟合圆 R={circle_r:.1f} mm")
    ax.plot(data.base_pos_x.iloc[0] * 1000, data.base_pos_y.iloc[0] * 1000, "ko", markersize=4, label="起点")
    ax.set_aspect("equal")
    ax.set_xlabel("世界 X（毫米）")
    ax.set_ylabel("世界 Y（毫米）")
    ax.set_title("手调转弯步态基座轨迹与当前参考对照")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(output_dir / "trajectory.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    axes[0].plot(data.time, np.hypot(data.base_lin_vel_x, data.base_lin_vel_y) * 1000, label="基座水平速度")
    axes[0].plot(window.time, np.hypot(window.ref_vx, window.ref_vy) * 1000, "--", label="参考期望速度")
    axes[0].set_ylabel("速度（毫米／秒）")
    axes[0].legend(fontsize=8)
    axes[1].plot(data.time, data.base_ang_vel_z, label="基座 Z 角速度")
    axes[1].set_ylabel("角速度（弧度／秒）")
    axes[1].legend(fontsize=8)
    axes[2].plot(data.time, np.unwrap(data.base_yaw), label="基座偏航角（展开）")
    axes[2].set_ylabel("偏航角（弧度）")
    axes[2].set_xlabel("仿真时间（秒）")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.suptitle("手调转弯步态速度与偏航")
    fig.savefig(output_dir / "motion.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(16, 11), sharex=True, constrained_layout=True)
    for col, name in enumerate(_ACT_JOINT_NAMES):
        clean = name[:-6] if name.endswith("_joint") else name
        axis = axes.flat[col]
        axis.plot(data.time, data[f"{clean}_hand_target"], label="手调目标", linestyle="--")
        axis.plot(data.time, data[f"{clean}_ref_pos"], label="当前参考")
        axis.plot(data.time, data[f"{clean}_pos"], label="实际角", linewidth=1.0)
        axis.set_title(clean)
        axis.set_ylabel("角度（弧度）")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    for axis in list(axes.flat)[len(_ACT_JOINT_NAMES):]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("仿真时间（秒）")
    fig.suptitle("手调目标／当前参考／实际关节角")
    fig.savefig(output_dir / "joint_compare.png", dpi=130)
    plt.close(fig)

    grid = np.linspace(0.0, 1.0, 100, endpoint=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for axis, leg in zip(axes.flat, FOOT_ORDER, strict=True):
        params = HAND_LEGS[leg]
        t_cycle = 1.0 / gait_freq
        hand_x, hand_z = [], []
        for phase in grid:
            fx, fz = cycloid_foot(
                phase * t_cycle, t_cycle * HAND_SWING_RATIO, t_cycle * (1.0 - HAND_SWING_RATIO),
                params["stride"], params["height"], params["body_height"], params["x_offset"],
            )
            hand_x.append(fx)
            hand_z.append(fz)
        longitudinal, vertical = load_reference_foot(bool(params["is_front"]), 100)
        scale = reference_inner_scale(float(data.curvature_command.iloc[-1]), leg)
        axis.plot(np.asarray(hand_x) * 1000, np.asarray(hand_z) * 1000, "-", label="手调摆线")
        axis.plot(longitudinal * scale * 1000, vertical * 1000, "--", label=f"当前参考（缩放 {scale:.2f}）")
        axis.set_title(f"{leg} 足端轨迹（相对髋）")
        axis.set_xlabel("纵向（毫米）")
        axis.set_ylabel("竖直（毫米）")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("手调足端轨迹与当前参考足端轨迹")
    fig.savefig(output_dir / "foot_compare.png", dpi=140)
    plt.close(fig)

    print(f"[输出] CSV、metrics.json 与四张图：{output_dir.resolve()}")
    print(f"[实测] 拟合半径 {summary['等效半径_拟合_mm']:.2f} mm，等效曲率 "
          f"{summary['等效曲率_拟合_1pm']:.2f} 1/m，路径速度 {summary['路径平均速度_mps']:.4f} m/s，"
          f"偏航角速度 {summary['平均偏航角速度_radps']:.3f} rad/s")
    print(f"[对照] 参考半径 {summary['参考轨迹']['半径_mm']:.2f} mm，参考速度 "
          f"{summary['参考轨迹']['参考速度_mps']:.4f} m/s；关节 RMS "
          f"腿 {summary['关节对照_rad']['腿RMS']:.3f} / 脊柱 {summary['关节对照_rad']['脊柱RMS']:.3f} / "
          f"颈 {summary['关节对照_rad']['颈部RMS']:.3f} rad，最佳相位平移 "
          f"{summary['关节对照_rad']['最佳相位平移']:.2f} 周期")


@dataclass(frozen=True)
class ReplayConfig:
    visualize: Literal["viewer", "video", "none"] = "none"
    duration: float = 20.0
    gait_freq: float = HAND_GAIT_FREQ
    start_time: float = HAND_START_TIME
    curvature: float = -CURVATURE_TARGET_MAX
    ik_set: Literal["legacy", "current"] = "legacy"
    device: str | None = None
    output_dir: Path = SLALOM_HAND_TUNED_REPLAY_DIR


# 固定命令: 只为复现当前参考, 手调策略不读取命令
def configure_env(args: ReplayConfig) -> ManagerBasedRlEnvCfg:
    cfg = load_env_cfg("Mjlab-SQuRo-Slalom")
    cfg.scene.num_envs = 1
    cfg.episode_length_s = args.duration + 5.0
    cfg.rewards = {}
    cfg.terminations = {}
    command = cast(SlalomCommandCfg, cfg.commands["slalom_cmd"])
    command.phase0_only = True
    command.fixed_curvature = args.curvature
    command.fixed_gait_freq = args.gait_freq
    command.fixed_velocity = 0.1
    command.debug_vis = False
    command.resampling_time_range = (args.duration + 10.0, args.duration + 20.0)
    return cfg


def main() -> None:
    args = tyro.cli(ReplayConfig)
    if args.duration <= args.start_time:
        raise ValueError("回放时长必须大于起步时间")
    if args.gait_freq <= 0.0:
        raise ValueError("步频必须为正")
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_dir / (
        f"{stamp}_f{args.gait_freq:g}-k{args.curvature:g}-ik{args.ik_set}-{args.visualize}"
    )

    cfg = configure_env(args)
    env = ManagerBasedRlEnv(
        cfg=cfg,
        device=device,
        render_mode="rgb_array" if args.visualize == "video" else None,
    )
    policy: HandTunedReplayPolicy | None = None
    try:
        steps = int(math.ceil(args.duration / env.step_dt))
        if args.visualize == "video":
            env = VideoRecorder(
                env,
                video_folder=output_dir,
                step_trigger=lambda step: step == 0,
                video_length=steps,
                name_prefix="hand_tuned",
                disable_logger=True,
            )
        with torch.no_grad():
            env.reset()
            policy = HandTunedReplayPolicy(env.unwrapped, args.ik_set, args.gait_freq, args.start_time)
            print(f"[手调回放] 步频 {args.gait_freq:g} Hz，起步 {args.start_time:g} s，"
                  f"逆运动学 {args.ik_set}，参考曲率 {args.curvature:g} 1/m")
            if args.visualize == "viewer":
                viewer = NativeMujocoViewer(env, policy, frame_rate=60)  # type: ignore[arg-type]
                viewer.run(num_steps=steps)
            else:
                for _ in range(steps):
                    env.step(policy(env.get_observations()))
            policy(env.get_observations())
        summary = summarize(policy.records, args.start_time, args.gait_freq)
        save_outputs(policy.records, summary, output_dir, args.gait_freq)
    finally:
        env.close()
    if args.visualize == "video":
        print(f"[视频] {(output_dir / 'hand_tuned.mp4').resolve()}")


if __name__ == "__main__":
    main()
