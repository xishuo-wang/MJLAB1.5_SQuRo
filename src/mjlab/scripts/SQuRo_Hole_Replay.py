from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
import tyro

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Hole.mdp.command import HoleCommandCfg
from mjlab.tasks.SQuRo_Hole.mdp.config import HEIGHT_LIST, TABLE_RESOLUTION
from mjlab.tasks.SQuRo_Hole.mdp.hole import build_hole_entities
from mjlab.tasks.SQuRo_Hole.mdp.indices import _MODEL_INDICES
from mjlab.tasks.SQuRo_Hole.mdp.reference import (
    Initialize_Tables,
    get_reference_joint_pos,
    get_reference_joint_vel,
    resolve_joint_ids,
)
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer


@dataclass(frozen=True)
class ReplayConfig:
    # 前躯干目标高度，单位为米。
    height_f: float
    # 后躯干目标高度，单位为米。
    height_h: float
    # 显示查看器、录视频或无窗运行。
    visualize: Literal["viewer", "video", "none"] = "viewer"
    # 最大回放仿真时长，单位为秒。
    duration: float = 10.0
    # 仿真设备，默认自动选择。
    device: str | None = None
    # 是否启用限高板真实碰撞。
    enable_collision: bool = False
    # CSV、曲线和视频的输出根目录。
    output_dir: Path = Path("logs/rsl_rl/SQuRo_Hole/reference_replay")


# 锁定高度命令并关闭自动课程，详见 docs/SQuRo_Hole_参考回放.md。
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


# 按参考列序解析动作和执行器映射，保留模型限位。
class ReferenceTablePolicy:
    # 按名称绑定参考列、动作列和模型控制范围。
    def __init__(self, env: ManagerBasedRlEnv) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        self.joint_ids = resolve_joint_ids(self.robot)
        self.names = [self.robot.joint_names[i] for i in self.joint_ids]
        self.action_term = cast(JointPositionAction, env.action_manager.get_term("joint_pos"))
        column_by_joint = {joint_id: col for col, joint_id in enumerate(self.joint_ids)}
        self.action_columns = [column_by_joint[i] for i in self.action_term.target_ids.tolist()]
        if len(self.action_columns) != len(self.joint_ids):
            raise ValueError("动作关节与参考表关节数量不一致")
        if self.action_term.cfg.clip is not None:
            raise ValueError("参考回放要求动作项没有额外裁剪")
        if torch.any(torch.as_tensor(self.action_term.scale) == 0):
            raise ValueError("动作缩放不能为零")

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
        self.records: list[dict[str, float]] = []
        self._last_step: tuple[int, int] | None = None

    # 使用训练侧查询函数扫描完整周期并报告不可达目标。
    def report_reference_limits(self) -> None:
        command = self.env.command_manager.get_command("hole_cmd")
        tables = Initialize_Tables(self.env.device)
        period = float(tables["mode_periods"].max())
        probe = SimpleNamespace(
            device=self.env.device,
            step_dt=period / TABLE_RESOLUTION,
            episode_length_buf=torch.arange(TABLE_RESOLUTION, device=self.env.device),
            command_manager=SimpleNamespace(_terms={
                "hole_cmd": SimpleNamespace(command=command.expand(TABLE_RESOLUTION, -1))
            }),
        )
        reference = get_reference_joint_pos(probe).cpu().numpy()
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

    # 同一仿真时刻记录原始参考、实际状态与执行器控制范围。
    def record(self, reference: torch.Tensor, velocity: torch.Tensor) -> None:
        env = self.env
        step = (int(env.common_step_counter), int(env.episode_length_buf[0]))
        if step == self._last_step:
            return
        self._last_step = step
        robot = self.robot
        idx = _MODEL_INDICES
        cmd = env.command_manager.get_command("hole_cmd")[0].cpu().numpy()
        ref = reference[0].cpu().numpy()
        ref_vel = velocity[0].cpu().numpy()
        pos = robot.data.joint_pos[0, self.joint_ids].cpu().numpy()
        vel = robot.data.joint_vel[0, self.joint_ids].cpu().numpy()
        force = robot.data.actuator_force[0, self.force_columns].cpu().numpy()
        target = np.clip(ref, self.ctrl_limits[:, 0], self.ctrl_limits[:, 1])
        row = {
            "time": step[0] * env.step_dt,
            "reference_time": step[1] * env.step_dt,
            "x": float(robot.data.root_link_pos_w[0, 0]),
            "vx": float(robot.data.root_link_lin_vel_w[0, 0]),
            "height_F_command": float(cmd[3]),
            "height_H_command": float(cmd[4]),
            "height_F_actual": float(robot.data.body_link_pos_w[0, idx.f_body_id, 2]),
            "height_H_actual": float(robot.data.body_link_pos_w[0, idx.h_body_id, 2]),
            "front_surface_max_z": float(robot.data.site_pos_w[0, list(idx.front_seg_site_ids), 2].max()),
            "rear_surface_max_z": float(robot.data.site_pos_w[0, list(idx.rear_seg_site_ids), 2].max()),
        }
        for col, name in enumerate(self.names):
            for suffix, values in [("ref_pos", ref), ("ref_vel", ref_vel), ("pos", pos),
                                   ("vel", vel), ("control_target", target), ("force", force)]:
                row[f"{name}_{suffix}"] = float(values[col])
        self.records.append(row)

    # 将原始参考按动作项实际关节顺序反解为位置控制动作。
    def __call__(self, obs: Any) -> torch.Tensor:
        reference = get_reference_joint_pos(self.env)
        self.record(reference, get_reference_joint_vel(self.env))
        target = reference[:, self.action_columns]
        return (target - self.action_term.offset) / self.action_term.scale

    # 保存对照数据与曲线，查看器和视频共用同一记录。
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
        for col, name in enumerate(self.names):
            ax = axes.flat[col]
            ax.plot(data.time, data[f"{name}_ref_pos"], label="原始参考", linestyle="--")
            ax.plot(data.time, data[f"{name}_control_target"], label="限幅控制目标", linestyle=":")
            ax.plot(data.time, data[f"{name}_pos"], label="实际角度")
            for limit in self.joint_limits[col]:
                if np.isfinite(limit):
                    ax.axhline(limit, color="red", linewidth=.6, alpha=.5)
            ax.set_title(f"关节：{name}")
            ax.set_ylabel("角度（弧度）")
            ax.set_xlabel("仿真时间（秒）")
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(self.names):]:
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


# 回放固定高度的生产参考，不加载策略检查点。
def main() -> None:
    args = tyro.cli(ReplayConfig)
    if not all(math.isfinite(v) and v > 0 for v in [args.height_f, args.height_h, args.duration]):
        raise ValueError("前后高度和回放时长必须为有限正数")
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output_dir = args.output_dir / (
        f"{stamp}_fixed-hF{args.height_f * 1000:g}-hH{args.height_h * 1000:g}"
        f"-col{int(args.enable_collision)}"
    )
    cfg = configure_env(args)
    env = ManagerBasedRlEnv(cfg=cfg, device=device,
                            render_mode="rgb_array" if args.visualize == "video" else None)
    policy = None
    try:
        if args.duration < env.step_dt:
            raise ValueError(f"回放时长不能小于控制周期 {env.step_dt:g} 秒")
        steps = math.ceil(args.duration / env.step_dt)
        if args.visualize == "video":
            env = VideoRecorder(env, video_folder=output_dir, step_trigger=lambda step: step == 0,
                                video_length=steps, name_prefix="reference", disable_logger=True)
        with torch.no_grad():
            env.reset()
            policy = ReferenceTablePolicy(env.unwrapped)
            cmd = env.command_manager.get_command("hole_cmd")[0]
            print(f"[固定参考] 前高 {float(cmd[3]) * 1000:g} mm，后高 {float(cmd[4]) * 1000:g} mm，"
                  f"限高板碰撞 {'开启' if args.enable_collision else '关闭'}")
            for label, col in [("前", 3), ("后", 4)]:
                height = float(cmd[col])
                nearest = min(HEIGHT_LIST, key=lambda value: abs(value - height))
                if not math.isclose(height, nearest, abs_tol=1e-6):
                    print(f"[参考查表] {label}高度 {height * 1000:g} mm → 最近档 {nearest * 1000:g} mm，沿用训练查询规则")
            policy.report_reference_limits()
            if args.visualize == "viewer":
                viewer = NativeMujocoViewer(env, policy, frame_rate=60)
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
