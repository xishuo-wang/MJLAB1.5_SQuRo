from __future__ import annotations

import math
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Any, Literal, cast
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Hole.mdp.command import HoleCommandCfg
from mjlab.tasks.SQuRo_Hole.mdp.config import (
    HEIGHT_LIST,
    TABLE_RESOLUTION,
    THRESHOLD_HEIGHT,
)
from mjlab.tasks.SQuRo_Hole.mdp.hole import build_hole_entities
from mjlab.tasks.SQuRo_Hole.mdp.indices import _MODEL_INDICES, resolve_model_indices
from mjlab.tasks.SQuRo_Hole.mdp.reference import (
    Initialize_Tables,
    get_reference_joint_state,
)
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer


@dataclass(frozen=True)
class ReplayConfig:
    height_f: float
    height_h: float
    visualize: Literal["viewer", "video", "none"] = "viewer"
    duration: float = 10.0
    device: str | None = None
    enable_collision: bool = False
    output_dir: Path = Path("logs/rsl_rl/SQuRo_Hole/reference_replay")


# ==================== 直接给定的回放参数 ====================
HEIGHT_F = 0.055
HEIGHT_H = 0.055
VISUALIZE = "viewer"
DURATION = 10.0
DEVICE = None
ENABLE_COLLISION = False
OUTPUT_DIR = Path("logs/rsl_rl/SQuRo_Hole/reference_replay")
# ==========================================================


# ---------- 工具函数 ----------
def _strip_joint_suffix(name: str) -> str:
    """'F_spine1_joint' → 'F_spine1'，让列名与 CSV_Anaylsis.py 的约定一致。"""
    return name[:-6] if name.endswith("_joint") else name


def _quat_to_yaw(q: torch.Tensor) -> float:
    """四元数 (w, x, y, z) → 偏航角 (rad)。"""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def configure_env(args: ReplayConfig) -> ManagerBasedRlEnvCfg:
    cfg = load_env_cfg("Mjlab-SQuRo-Hole")
    cfg.scene.num_envs = 1
    cfg.scene.entities.update(build_hole_entities(args.enable_collision))
    command = cast(HoleCommandCfg, cfg.commands["hole_cmd"])
    command.fixed_height_F = args.height_f
    command.fixed_height_H = args.height_h
    command.fixed_angle = 0.0
    command.use_position_schedule = False
    command.position_schedule = []
    command.use_height_schedule = False
    command.height_schedule = []
    command.stage_schedule_fallback = False
    command.debug_vis = False
    cfg.rewards = {}
    cfg.terminations = {}
    return cfg


class ReferenceTablePolicy:
    # 按名称绑定参考列、动作列和模型控制范围。
    def __init__(self, env: ManagerBasedRlEnv) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        # 参考列序已与执行器序同步, 关节顺序直接取按名解析结果 (执行器序)
        resolve_model_indices(self.robot)
        self.joint_ids = list(_MODEL_INDICES.joint_ids)
        self.names = [self.robot.joint_names[i] for i in self.joint_ids]
        # 去掉 "_joint" 后缀，与 CSV_Anaylsis.py 的列名一致
        self.clean_names = [_strip_joint_suffix(n) for n in self.names]

        self.action_term = cast(JointPositionAction, env.action_manager.get_term("joint_pos"))
        column_by_joint = {joint_id: col for col, joint_id in enumerate(self.joint_ids)}
        self.action_columns = [column_by_joint[i] for i in self.action_term.target_ids.tolist()]
        if len(self.action_columns) != len(self.joint_ids):
            raise ValueError("动作关节与参考表关节数量不一致")
        if self.action_term.cfg.clip is not None:
            raise ValueError("参考回放要求动作项没有额外裁剪")
        if torch.any(torch.as_tensor(self.action_term.scale) == 0):
            raise ValueError("动作缩放不能为零")

        # 动作顺序 → 参考顺序 的反向映射（用于记录 {joint}_action）
        self.ref_to_action_col: list[int | None] = [None] * len(self.joint_ids)
        for act_col, ref_col in enumerate(self.action_columns):
            self.ref_to_action_col[ref_col] = act_col

        model = env.sim.mj_model
        ctrl_ids = self.robot.indexing.ctrl_ids.tolist()
        global_joint_ids = self.robot.indexing.joint_ids[self.joint_ids].tolist()
        ctrl_by_joint = {int(model.actuator_trnid[i, 0]): i for i in ctrl_ids}
        self.ctrl_ids = [ctrl_by_joint[i] for i in global_joint_ids]
        local_ctrl = {global_id: local_id for local_id, global_id in enumerate(ctrl_ids)}
        self.force_columns = [local_ctrl[i] for i in self.ctrl_ids]
        self.joint_limits = self.robot.data.joint_pos_limits[0, self.joint_ids].cpu().numpy()
        self.ctrl_limits = model.actuator_ctrlrange[self.ctrl_ids].copy()
        self.ctrl_limits[~model.actuator_ctrllimited[self.ctrl_ids].astype(bool)] = [-np.inf, np.inf]

        # 足端 site —— 与 indices.py 的 FOOT_SITE_NAMES 对应
        self.foot_names = ["FL", "FR", "HL", "HR"]
        self.foot_site_names = ["FL_elbow_site", "FR_elbow_site", "HL_knee_site", "HR_knee_site"]
        self._foot_site_ids: torch.Tensor | None = None

        # 步态基频：参考表固定为 2.0 Hz（CYCLOID_PARAMS / BASE_FREQ）
        self.gait_freq = 2.0

        self.records: list[dict[str, Any]] = []
        self._last_step: tuple[int, int] | None = None

    def _ensure_foot_sites(self) -> None:
        if self._foot_site_ids is None:
            self._foot_site_ids, _ = self.robot.find_sites(self.foot_site_names, preserve_order=True)

    # ---------- 与原来相同：扫描一个完整周期，检查限位冲突 ----------
    def report_reference_limits(self) -> None:
        command = self.env.command_manager.get_command("hole_cmd")
        tables = Initialize_Tables(self.env.device)
        period = float(tables["mode_periods"].max())
        probe = SimpleNamespace(
            device=self.env.device,
            step_dt=period / TABLE_RESOLUTION,
            episode_length_buf=torch.arange(TABLE_RESOLUTION, device=self.env.device),
            command_manager=SimpleNamespace(_terms={
                "hole_cmd": SimpleNamespace(command=command.expand(TABLE_RESOLUTION, -1))  # type: ignore
            }),
        )
        reference, _ = get_reference_joint_state(probe)
        reference = reference.cpu().numpy()
        lower = np.maximum(self.joint_limits[:, 0], self.ctrl_limits[:, 0])
        upper = np.minimum(self.joint_limits[:, 1], self.ctrl_limits[:, 1])
        conflicts = (reference < lower - 1e-6) | (reference > upper + 1e-6)
        for col, name in enumerate(self.names):
            if conflicts[:, col].any():
                print(f"[限位冲突] {name}：参考 {reference[:, col].min():.4f}"
                      f"～{reference[:, col].max():.4f} rad，关节范围 "
                      f"{self.joint_limits[col].tolist()}，控制范围 {self.ctrl_limits[col].tolist()}")
        if not conflicts.any():
            print("[限位检查] 当前高度对应的参考未超出关节和执行器控制范围")

    # ---------- 写入 CSV_Anaylsis.py 所需的全部列 ----------
    def record(self, reference: torch.Tensor, velocity: torch.Tensor,
               action: torch.Tensor | None = None) -> None:
        env = self.env
        step = (int(env.common_step_counter), int(env.episode_length_buf[0]))
        if step == self._last_step:
            return
        self._last_step = step

        robot = self.robot
        idx = _MODEL_INDICES
        self._ensure_foot_sites()

        # ---- 参考 ----
        ref = reference[0].cpu().numpy()
        ref_vel = velocity[0].cpu().numpy()

        # ---- 实际关节状态 ----
        pos = robot.data.joint_pos[0, self.joint_ids].cpu().numpy()
        vel = robot.data.joint_vel[0, self.joint_ids].cpu().numpy()
        force = robot.data.actuator_force[0, self.force_columns].cpu().numpy()

        # joint_acc 某些版本没有暴露，读不到就填 0（RL 脚本有这一列，保证列一致）
        try:
            acc = robot.data.joint_acc[0, self.joint_ids].cpu().numpy()
        except (AttributeError, IndexError):
            acc = np.zeros_like(pos)

        # ---- 动作按参考顺序重排 ----
        if action is not None:
            action_np = action[0].detach().cpu().numpy()
            action_in_ref = np.zeros(len(self.joint_ids), dtype=np.float32)
            for ref_col, act_col in enumerate(self.ref_to_action_col):
                if act_col is not None:
                    action_in_ref[ref_col] = action_np[act_col]
        else:
            action_in_ref = None

        target = np.clip(ref, self.ctrl_limits[:, 0], self.ctrl_limits[:, 1])

        # ---- 基座位姿 / 速度 / 角速度 ----
        base_pos = robot.data.root_link_pos_w[0].cpu().numpy()
        base_lin_vel = robot.data.root_link_lin_vel_w[0].cpu().numpy()
        base_ang_vel = robot.data.root_link_ang_vel_w[0].cpu().numpy()

        # ---- F_body / H_body 偏航角 ----
        f_quat = robot.data.body_link_quat_w[0, idx.f_body_id]
        h_quat = robot.data.body_link_quat_w[0, idx.h_body_id]
        f_body_yaw = _quat_to_yaw(f_quat)
        h_body_yaw = _quat_to_yaw(h_quat)

        # ---- 足端世界坐标 ----
        foot_pos = robot.data.site_pos_w[0, self._foot_site_ids].cpu().numpy()  # (4, 3)

        # ---- 足端接触力 ----
        contact_sensor = env.scene["feet_ground_contact"]
        feet_contact = contact_sensor.data.force.flatten(start_dim=1)[0].cpu().numpy()  # (12,)

        # ---- 命令 ----
        cmd = env.command_manager.get_command("hole_cmd")[0].cpu().numpy()  # type: ignore
        hF_cmd = float(cmd[3])
        hH_cmd = float(cmd[4])
        hF_act = float(robot.data.body_link_pos_w[0, idx.f_body_id, 2])
        hH_act = float(robot.data.body_link_pos_w[0, idx.h_body_id, 2])

        t_global = step[0] * env.step_dt
        t_episode = step[1] * env.step_dt

        row: dict[str, Any] = {
            # ---- 时间与步 ----
            "step": float(step[0]),
            "time": t_global,
            "reference_time": t_episode,

            # ---- 基座 ----
            "base_pos_x": float(base_pos[0]),
            "base_pos_y": float(base_pos[1]),
            "base_pos_z": float(base_pos[2]),
            "base_vel_x": float(base_lin_vel[0]),
            "base_vel_y": float(base_lin_vel[1]),
            "base_vel_z": float(base_lin_vel[2]),
            # 分析脚本使用的命名
            "base_lin_vel_x": float(base_lin_vel[0]),
            "base_lin_vel_y": float(base_lin_vel[1]),
            "base_lin_vel_z": float(base_lin_vel[2]),
            "base_ang_vel_x": float(base_ang_vel[0]),
            "base_ang_vel_y": float(base_ang_vel[1]),
            "base_ang_vel_z": float(base_ang_vel[2]),

            # ---- 朝向（三级回退由分析脚本负责） ----
            "f_body_heading": float(f_body_yaw),
            "h_body_heading": float(h_body_yaw),
            "heading": float(f_body_yaw),

            # ---- 旧字段保留（save() 里的曲线还引用 x / vx） ----
            "x": float(base_pos[0]),
            "vx": float(base_lin_vel[0]),

            # ---- 命令 ----
            "vel_command_x": float(cmd[0]),
            "vel_command_y": float(cmd[1]),
            "vel_command_z": float(cmd[2]),
            "height_F_command": hF_cmd,
            "height_H_command": hH_cmd,
            "angle_command": float(cmd[5]),
            "gait_freq_command": float(self.gait_freq),

            # ---- 高度（命令 vs 实测，两种命名都写） ----
            "height_F_actual": hF_act,
            "height_H_actual": hH_act,
            "F_body_height": hF_act,
            "H_body_height": hH_act,
            "front_surface_max_z": float(robot.data.site_pos_w[0, list(idx.f_body_site_ids), 2].max()),
            "rear_surface_max_z": float(robot.data.site_pos_w[0, list(idx.h_body_site_ids), 2].max()),
        }

        # ---- 高度模式与误差 ----
        low_F = hF_cmd < THRESHOLD_HEIGHT
        low_H = hH_cmd < THRESHOLD_HEIGHT
        if low_F and not low_H:
            row["mode"] = "前低后高"
        elif low_H and not low_F:
            row["mode"] = "前高后低"
        elif low_F and low_H:
            row["mode"] = "双低"
        else:
            row["mode"] = "都高"
        row["height_F_error"] = hF_act - hF_cmd
        row["height_H_error"] = hH_act - hH_cmd

        # ---- 足端位置 + 接触力 ----
        for i, fname in enumerate(self.foot_names):
            row[f"foot_{fname}_x"] = float(foot_pos[i, 0])
            row[f"foot_{fname}_y"] = float(foot_pos[i, 1])
            row[f"foot_{fname}_z"] = float(foot_pos[i, 2])
            fx = float(feet_contact[i * 3])
            fy = float(feet_contact[i * 3 + 1])
            fz = float(feet_contact[i * 3 + 2])
            row[f"contact_{fname}_x"] = fx
            row[f"contact_{fname}_y"] = fy
            row[f"contact_{fname}_z"] = fz
            row[f"contact_{fname}_mag"] = float(np.sqrt(fx * fx + fy * fy + fz * fz))

        # ---- 每个关节 ----
        for col, cname in enumerate(self.clean_names):
            row[f"{cname}_pos"] = float(pos[col])
            row[f"{cname}_vel"] = float(vel[col])
            row[f"{cname}_acc"] = float(acc[col])           # 与 RL 脚本同名
            row[f"{cname}_force"] = float(force[col])
            # 分析脚本使用的别名
            row[f"{cname}_torque"] = float(force[col])
            row[f"{cname}_ref_pos"] = float(ref[col])
            row[f"{cname}_ref_vel"] = float(ref_vel[col])
            row[f"{cname}_control_target"] = float(target[col])
            if action_in_ref is not None:
                row[f"{cname}_action"] = float(action_in_ref[col])

        # ---- RL 专用字段：参考回放里没有，填 NaN / 0 保持列一致 ----
        row["reward"] = float("nan")
        row["done"] = 0.0

        self.records.append(row)

    # ---- 与原版相同，但把 action 传给 record ----
    def __call__(self, obs: Any) -> torch.Tensor:
        reference, velocity = get_reference_joint_state(self.env)
        target = reference[:, self.action_columns]
        action = (target - self.action_term.offset) / self.action_term.scale
        self.record(reference, velocity, action=action)
        return action

    # ---------- save：把 self.names 换成 clean_names ----------
    def save(self, output_dir: Path) -> None:
        if not self.records:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        output_dir.mkdir(parents=True, exist_ok=True)
        data = pd.DataFrame(self.records)
        data.to_csv(output_dir / "reference.csv", index=False, encoding="utf-8-sig")

        fig, axes = plt.subplots(4, 4, figsize=(16, 11), sharex=True, constrained_layout=True)
        for col, cname in enumerate(self.clean_names):
            ax = axes.flat[col]
            ax.plot(data.time, data[f"{cname}_ref_pos"], label="原始参考", linestyle="--")
            ax.plot(data.time, data[f"{cname}_control_target"], label="限幅控制目标", linestyle=":")
            ax.plot(data.time, data[f"{cname}_pos"], label="实际角度")
            for limit in self.joint_limits[col]:
                if np.isfinite(limit):
                    ax.axhline(limit, color="red", linewidth=.6, alpha=.5)
            ax.set_title(f"关节：{cname}")
            ax.set_ylabel("角度（弧度）")
            ax.set_xlabel("仿真时间（秒）")
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(self.clean_names):]:
            ax.set_visible(False)
        fig.suptitle("Hole 固定高度参考与实际关节对照")
        fig.savefig(output_dir / "joint_tracking.png", dpi=130)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        for ax, segment in zip(axes[:2], ["F", "H"], strict=True):
            ax.plot(data.time, data[f"height_{segment}_actual"] * 1000, label="实际高度")
            ax.plot(data.time, data[f"height_{segment}_command"] * 1000,
                    label="命令高度", linestyle="--")
            ax.set_ylabel("高度（毫米）")
            ax.set_title("前躯干" if segment == "F" else "后躯干")
            ax.legend()
            ax.grid(alpha=.2)
        axes[2].plot(data.time, data.vx, label="实际前向速度")
        axes[2].set_ylabel("速度（米／秒）")
        axes[2].set_xlabel("仿真时间（秒）")
        axes[2].legend()
        axes[2].grid(alpha=.2)
        fig.savefig(output_dir / "height_tracking.png", dpi=130)
        plt.close(fig)
        print(f"[输出] 参考／实际关节与高度曲线、CSV：{output_dir.resolve()}")


# ---------- main 保持原样 ----------
def main() -> None:
    args = ReplayConfig(
        height_f=HEIGHT_F,
        height_h=HEIGHT_H,
        visualize=VISUALIZE,
        duration=DURATION,
        device=DEVICE,
        enable_collision=ENABLE_COLLISION,
        output_dir=OUTPUT_DIR,
    )

    if not all(math.isfinite(v) and v > 0 for v in [args.height_f, args.height_h, args.duration]):
        raise ValueError("前后高度和回放时长必须为有限正数")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output_dir = args.output_dir / (
        f"{stamp}_fixed-hF{args.height_f * 1000:g}-hH{args.height_h * 1000:g}"
        f"-col{int(args.enable_collision)}"
    )

    cfg = configure_env(args)
    env = ManagerBasedRlEnv(
        cfg=cfg,
        device=device,
        render_mode="rgb_array" if args.visualize == "video" else None,
    )
    policy = None
    try:
        if args.duration < env.step_dt:
            raise ValueError(f"回放时长不能小于控制周期 {env.step_dt:g} 秒")
        steps = math.ceil(args.duration / env.step_dt)
        if args.visualize == "video":
            env = VideoRecorder(
                env,
                video_folder=output_dir,
                step_trigger=lambda step: step == 0,
                video_length=steps,
                name_prefix="reference",
                disable_logger=True,
            )
        with torch.no_grad():
            env.reset()
            policy = ReferenceTablePolicy(env.unwrapped)
            cmd = env.command_manager.get_command("hole_cmd")[0]  # type: ignore
            print(
                f"[固定参考] 前高 {float(cmd[3]) * 1000:g} mm，"
                f"后高 {float(cmd[4]) * 1000:g} mm，"
                f"限高板碰撞 {'开启' if args.enable_collision else '关闭'}"
            )
            for label, col in [("前", 3), ("后", 4)]:
                height = float(cmd[col])
                nearest = min(HEIGHT_LIST, key=lambda value: abs(value - height))
                if not math.isclose(height, nearest, abs_tol=1e-6):
                    print(
                        f"[参考查表] {label}高度 {height * 1000:g} mm → "
                        f"最近档 {nearest * 1000:g} mm，沿用训练查询规则"
                    )
            policy.report_reference_limits()
            if args.visualize == "viewer":
                viewer = NativeMujocoViewer(env, policy, frame_rate=60)  # type: ignore
                viewer.run(num_steps=steps)
            else:
                for _ in range(steps):
                    env.step(policy(env.get_observations()))
            policy(env.get_observations())
    finally:
        try:
            if policy is not None:
                policy.save(output_dir)
        finally:
            env.close()
    if args.visualize == "video":
        print(f"[视频] {(output_dir / 'reference.mp4').resolve()}")


if __name__ == "__main__":
    main()