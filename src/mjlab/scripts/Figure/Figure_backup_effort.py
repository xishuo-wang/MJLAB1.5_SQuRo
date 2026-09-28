# 论文图: 受限空间翻正的"费劲"代价 —— 单文件 (采集 + 三种画法), 全部用物理量
# 四个指标 (都不涉及任何奖励项):
#   1) 机械功率   P = Σ_j |τ_j · q̇_j|                    (W)
#   2) 足端接触力 Σ_feet |F_i| 的峰值与均值 (地反力)       (N, 另给体重倍数)
#   3) 力矩饱和率 fraction(|τ_j| >= 0.98 · τ_max,j)       (逐执行器 forcerange, 取自编译模型)
#   4) 关节速度 RMS (附: 关节加速度 RMS)                   (rad/s, rad/s²)
# 三种画法:
#   (a) 相位对齐时间序列 (功率 / 地反力 / 饱和率), 叠加三种条件
#   (b) 相对无约束条件的倍数柱状 (四指标 × 三条件, 误差棒取重复)
#   (c) 相位分解 (P1/P2/P3) 的功率与地反力冲量
# 配置都写在下面的配置段, 不需要命令行参数。
from __future__ import annotations

import csv
import importlib
import re
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.SQuRo_Backup.mdp import entity as mdp_entity
from mjlab.tasks.SQuRo_Backup.mdp.curriculums import WALL_X_POS, _STEPS_PER_ITER
from mjlab.tasks.SQuRo_Backup.mdp.entity import WALL_HALF_THICKNESS
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES


# 配置段
TASK_NAME = "Mjlab-SQuRo-Backup"
CHECKPOINT = "logs/rsl_rl/SQuRo_Backup/2026-09-27_14-33-27_v1/model_8999.pt"
OUT_DIR = Path("logs/backup_effort")
STEPS = 320                       # 一个循环 (λ=1 约 2.7 s) 并留余量
REPEATS = 3                       # 每条件重复次数 (同一构建内, 给误差棒)
WARMUP_STEPS = 30                 # 丢弃首次构建态
TIME_SCALE = 1.0                  # 固定 λ, 与回放口径一致
DEVICE = "cuda:0"
SEED = 0
ROBOT_MASS = 0.2750               # kg; 自重 2.698 N
# 条件: (标签, 净宽 m, 是否开碰撞)。0.23 m = 训练课程起点 (无约束参考), 0.09 m = 课程末档
CONDITIONS = (
    ("wide_ON", 0.23, True),
    ("narrow_ON", 0.09, True),
    ("narrow_OFF", 0.09, False),
)
# 墙接触参数: None = 实体默认 (solref 0.005/1.0); 下面这套是"力可物理解释"的软接触
WALL_SOLREF: tuple[float, ...] | None = (0.03, 0.8)
WALL_SOLIMP: tuple[float, ...] | None = (0.85, 0.95, 0.0008, 0.5, 2.0)
# 每步记录: [0] 功率, [1] 地反力, [2] 饱和率, [3] 速度 RMS, [4] 加速度 RMS, [5] 相位
ROW_DIM = 6
SAT_EPS = 0.98                    # 饱和判据系数
PHASES = (0, 1, 2)
PHASE_NAME = {0: "P1 (twist)", 1: "P2 (roll)", 2: "P3 (stand)"}
COND_COLOR = {"wide_ON": "#7f8c8d", "narrow_ON": "#c0392b", "narrow_OFF": "#2471a3"}
DT = 0.01                         # = env_cfg 的 timestep 0.002 × decimation 5


# 构造固定净宽的环境
def _build(clear_width: float, collision: bool, iter_num: int):
    importlib.import_module("mjlab.tasks.SQuRo_Backup.config")
    env_cfg = load_env_cfg(TASK_NAME)
    env_cfg.scene.num_envs = 1
    env_cfg.events.pop("init_restricted_space", None)
    env_cfg.commands["backup_cmd"].fixed_time_scale = TIME_SCALE
    half = 0.5 * clear_width + WALL_HALF_THICKNESS
    mdp_entity.configure_restricted_space(env_cfg, wall_x_neg=-half, wall_x_pos=half,
                                         enable_collision=collision, fixed_width=True,
                                         solref=WALL_SOLREF, solimp=WALL_SOLIMP)
    raw = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE)
    raw.common_step_counter = iter_num * _STEPS_PER_ITER
    return RslRlVecEnvWrapper(raw), raw


# 载入检查点 actor 的确定性推理策略
def _policy(env):
    runner = load_runner_cls(TASK_NAME)(env, asdict(load_rl_cfg(TASK_NAME)), None, device=DEVICE)
    runner.load(CHECKPOINT, load_cfg={"actor": True}, strict=True, map_location=DEVICE)
    return runner.get_inference_policy(device=DEVICE)


# 逐执行器力矩上限 (取自编译模型, 不硬编码)
def _torque_limit(raw) -> torch.Tensor:
    fr = raw.sim.model.actuator_forcerange[0]
    return torch.maximum(fr[:, 0].abs(), fr[:, 1].abs())


# 跑一个条件的 REPEATS 次 rollout; 每次记录**恰好一个循环** (phase 2->0 即停), 返回逐循环列表
def _rollouts(clear_width: float, collision: bool, iter_num: int) -> list[np.ndarray]:
    env, raw = _build(clear_width, collision, iter_num)
    robot = raw.scene.entities["robot"]
    cmd = raw.unwrapped.command_manager.get_term("backup_cmd")
    joint_ids = _MODEL_INDICES.joint_ids
    tau_max = _torque_limit(raw)
    cycles: list[np.ndarray] = []
    with torch.inference_mode():
        for _ in range(REPEATS):
            policy = _policy(env)
            env.reset()
            obs = env.get_observations()
            rows = []
            prev_phase = 0
            for k in range(STEPS + WARMUP_STEPS):
                obs, _, _, _ = env.step(policy(obs).to(env.device))
                tau = robot.data.actuator_force[0]
                vel = robot.data.joint_vel[0, joint_ids]
                acc = robot.data.joint_acc[0, joint_ids]
                grf = raw.scene["feet_ground_contact"].data.force[0].norm(dim=1).sum()
                phase = int(cmd.phase[0].item())
                if k >= WARMUP_STEPS:
                    rows.append(torch.stack((
                        (tau * vel).abs().sum(),
                        grf,
                        (tau.abs() >= SAT_EPS * tau_max).float().mean(),
                        torch.sqrt((vel ** 2).mean()),
                        torch.sqrt((acc ** 2).mean()),
                        cmd.phase[0].float(),
                    )))
                if k > WARMUP_STEPS and phase == 0 and prev_phase == 2:
                    break                      # 循环完成, 本段到此为止
                prev_phase = phase
            cycles.append(torch.stack(rows).cpu().numpy())
    env.close()
    return cycles


# 把逐循环列表补齐成 [REPEATS, t_max, ROW_DIM] 的 NaN 填充数组 (便于按时间对齐/取均值)
def _pad(cycles: list[np.ndarray], t_max: int) -> np.ndarray:
    out = np.full((len(cycles), t_max, ROW_DIM), np.nan)
    for i, c in enumerate(cycles):
        out[i, :len(c)] = c[:t_max]
    return out


# 指标名与取值口径 (相对柱状图用这里)
METRICS = (
    ("power_mean_W", "mechanical power (W)", 0, "mean"),
    ("grf_p95_N", "foot contact force p95 (N)", 1, "p95"),
    ("saturation", "torque saturation fraction", 2, "mean"),
    ("vel_rms_rad_s", "joint velocity RMS (rad/s)", 3, "mean"),
    ("acc_rms_rad_s2", "joint acceleration RMS (rad/s²)", 4, "mean"),
)


# 在给定窗口上按重复取指标: a [REPEATS, STEPS, ROW_DIM], mask [REPEATS, STEPS] -> [REPEATS]
def _window_metrics(a: np.ndarray, col: int, how: str, mask: np.ndarray) -> np.ndarray:
    out = []
    for r in range(a.shape[0]):
        v = a[r, mask[r], col]
        if v.size == 0:
            out.append(np.nan)
        elif how == "max":
            out.append(float(v.max()))
        elif how == "p95":
            out.append(float(np.percentile(v, 95)))
        else:
            out.append(float(v.mean()))
    return np.asarray(out)


# 主流程: 采集 -> 落盘 -> 三张图
def main() -> None:
    torch.manual_seed(SEED)
    m = re.search(r"model_(\d+)", Path(CHECKPOINT).name)
    iter_num = int(m.group(1)) if m else 0
    out_dir = OUT_DIR / f"iter{iter_num}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"检查点 {CHECKPOINT}  λ={TIME_SCALE}  步数={STEPS}  重复={REPEATS}  "
          f"接触参数 solref={WALL_SOLREF} solimp={WALL_SOLIMP}", flush=True)

    raw_cycles = {tag: _rollouts(clear, collision, iter_num)
                  for tag, clear, collision in CONDITIONS}
    t_max = max(len(c) for cs in raw_cycles.values() for c in cs)
    data: dict[str, np.ndarray] = {tag: _pad(cs, t_max) for tag, cs in raw_cycles.items()}
    for tag, clear, collision in CONDITIONS:
        a = data[tag]
        n = int(np.isfinite(a[:, :, 0]).sum(axis=1).mean())
        print(f"  [{tag}] 净宽 {clear * 1000:.0f} mm 碰撞={'开' if collision else '关'}  "
              f"循环长 {n * DT:.2f} s  功率均值 {np.nanmean(a[:, :, 0]):6.2f} W  "
              f"地反力 p95 {np.nanpercentile(a[:, :, 1], 95):6.2f} N  "
              f"饱和率 {np.nanmean(a[:, :, 2]):5.3f}  "
              f"速度RMS {np.nanmean(a[:, :, 3]):5.2f} rad/s", flush=True)

    # 逐条件 / 逐相位的汇总 (NaN 感知)
    summary: list[dict] = []
    for tag, _, _ in CONDITIONS:
        a = data[tag]
        for p in PHASES:
            mask = np.isfinite(a[:, :, 5]) & (np.nan_to_num(a[:, :, 5], nan=-1) == p)
            if not mask.any():
                continue
            rec = {"condition": tag, "phase": PHASE_NAME[p], "steps": int(mask.sum())}
            power = a[:, :, 0][mask]
            grf = a[:, :, 1][mask]
            rec["power_mean_W"] = float(np.nanmean(power))
            rec["power_peak_W"] = float(np.nanmax(power))
            rec["grf_mean_N"] = float(np.nanmean(grf))
            rec["grf_p95_N"] = float(np.nanpercentile(grf, 95))
            rec["grf_peak_N"] = float(np.nanmax(grf))
            rec["saturation"] = float(np.nanmean(a[:, :, 2][mask]))
            rec["vel_rms_rad_s"] = float(np.nanmean(a[:, :, 3][mask]))
            rec["acc_rms_rad_s2"] = float(np.nanmean(a[:, :, 4][mask]))
            rec["energy_J"] = float(np.nansum(power) * DT)
            rec["grf_impulse_Ns"] = float(np.nansum(grf) * DT)
            summary.append(rec)
    with (out_dir / "effort_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        wr.writeheader()
        for r in summary:
            wr.writerow(r)
    with (out_dir / "effort_steps.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["condition", "repeat", "step", "power_W", "grf_N", "saturation",
                     "vel_rms", "acc_rms", "phase"])
        for tag, _, _ in CONDITIONS:
            a = data[tag]
            for r in range(REPEATS):
                for k in range(a.shape[1]):
                    if not np.isfinite(a[r, k, 0]):
                        continue
                    row = a[r, k]
                    wr.writerow([tag, r, k, *[f"{v:.6f}" for v in row]])

    weight = ROBOT_MASS * 9.81
    t_max = max(a.shape[1] for a in data.values())
    t = np.arange(t_max) * DT
    phase = data["wide_ON"][0, :, 5]

    # 画法 (a): 相位对齐时间序列
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.0), sharex=True)
    for tag, _, _ in CONDITIONS:
        a = data[tag]
        for ax, col, ylab in ((axes[0], 0, "mechanical power $\\sum_j|\\tau_j\\dot q_j|$ (W)"),
                              (axes[1], 1, "foot contact force $\\sum|F_i|$ (N)"),
                              (axes[2], 2, "torque saturation fraction (-)")):
            ax.plot(t, np.nanmean(a[:, :, col], axis=0), color=COND_COLOR[tag], lw=1.8,
                    label=tag.replace("_", " "))
    edges = t[np.flatnonzero(np.nan_to_num(np.diff(phase), nan=0.0) != 0)]
    for ax in axes:
        ax.grid(alpha=0.25)
        for edge in edges:
            ax.axvline(edge, color="k", ls=":", lw=0.9)
    axes[1].axhline(weight, color="k", ls="--", lw=1.0,
                    label=f"body weight {weight:.2f} N")
    axes[1].legend(fontsize=7.5, loc="upper left")
    axes[0].set_title("(a) phase-aligned effort time series "
                      f"(mean of {REPEATS} rollouts)", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[2].set_xlabel("time from cycle start (s)  |  dotted lines = "
                       + " / ".join(PHASE_NAME[p] for p in PHASES))
    fig.tight_layout()
    fig.savefig(out_dir / "fig_a_timeseries.png", dpi=250)

    # 画法 (b): 相对无约束条件的倍数柱状 (窗口 = 翻正动作段 P1+P2, 与 §7.16 口径一致)
    base = "wide_ON"
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    keys = [k for k in METRICS if k[0] != "acc_rms_rad_s2"]
    width_bar = 0.8 / len(keys)
    for i, (name, lab, col, how) in enumerate(keys):
        xs, ys, es = [], [], []
        for j, (tag, _, _) in enumerate(CONDITIONS):
            a = data[tag]
            mask = a[:, :, 5] <= 1                          # P1+P2 = 翻正动作段
            vals = _window_metrics(a, col, how, mask)
            xs.append(j + (i - (len(keys) - 1) / 2) * width_bar)
            ys.append(float(np.nanmean(vals)))
            es.append(float(np.nanstd(vals)))
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        base_val = ys[0]
        ax.bar(xs, ys / base_val, width=width_bar * 0.9, yerr=es / base_val,
               capsize=3, label=lab)
    ax.set_xticks(np.arange(len(CONDITIONS)))
    ax.set_xticklabels([c[0].replace("_", "\n") for c in CONDITIONS], fontsize=9)
    ax.axhline(1.0, color="k", ls=":", lw=1.0)
    ax.set_ylabel(f"normalized to {base} (= 1)")
    ax.set_title("(b) effort multiples relative to the unconstrained condition "
                 "(righting action, P1+P2)", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_b_relative_bars.png", dpi=250)

    # 画法 (c): 相位分解 (功率与地反力冲量)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, key, ylab in ((axes[0], "energy_J", "mechanical work per phase (J)"),
                          (axes[1], "grf_impulse_Ns", "contact-force impulse per phase (N·s)")):
        width_bar = 0.8 / len(PHASES)
        for i, p in enumerate(PHASES):
            xs, ys = [], []
            for j, (tag, _, _) in enumerate(CONDITIONS):
                rec = next((r for r in summary
                            if r["condition"] == tag and r["phase"] == PHASE_NAME[p]), None)
                xs.append(j + (i - (len(PHASES) - 1) / 2) * width_bar)
                ys.append(0.0 if rec is None else rec[key])
            ax.bar(np.asarray(xs), ys, width=width_bar * 0.9,
                   color=["#5dade2", "#f5b041", "#58d68d"][i], label=PHASE_NAME[p])
        ax.set_xticks(np.arange(len(CONDITIONS)))
        ax.set_xticklabels([c[0].replace("_", "\n") for c in CONDITIONS], fontsize=9)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=8)
    axes[0].set_title("(c) where the effort is spent: phase decomposition", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_c_phase_split.png", dpi=250)

    print(f"\n[INFO] 输出目录: {out_dir}")
    print(f"  逐条件/逐相位汇总: effort_summary.csv")
    print(f"  逐绘图: fig_a_timeseries.png / fig_b_relative_bars.png / fig_c_phase_split.png")
    for rec in summary:
        print(f"  {rec['condition']:11s} {rec['phase']:14s} 功率 {rec['power_mean_W']:6.2f} W  "
              f"地反力峰 {rec['grf_peak_N']:7.2f} N 饱和 {rec['saturation']:.3f}  "
              f"速度RMS {rec['vel_rms_rad_s']:5.2f} 功 {rec['energy_J']:6.2f} J")


if __name__ == "__main__":
    main()
