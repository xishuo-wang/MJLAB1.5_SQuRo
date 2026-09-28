# 诊断: 受限空间翻正任务下脊柱基元四定义随约束强度的变化 (只读, 落盘逐循环 CSV)
# 四定义 = 论文关键定义 9~12:
#   A_i   激活度   = S_i / (theta_max_i * T)
#   C_i   贡献系数 = S_i / sum_j S_j
#   tau_i 时序中心 = (∫ t·|θ_i| dt / ∫|θ_i| dt) / T
#   O_ij  时序重叠 = ∫ min(ŝ_i, ŝ_j) dt / T,  ŝ = |θ| / max|θ|
# 约束强度轴 = 训练课程本来的那一维: 右墙固定 +0.05, 左墙距离 d 从 0.20 收到 0.06,
# 净宽 = d + 0.05 - 0.02。**逐环境墙位**与训练同机制 (mocap 逐环境写), 所以一档一个环境,
# 一次构建即可并行扫完所有档位。
# 通道口径 = **名义关节角** (轴向 = F_body - H_body, 侧摆 = F_spine1, 俯仰 = H_spine1)。
# 实际等效角 (θ_L/θ_S/θ_A) 定义确定后, 只需替换 _finish 里的通道组装。
# 用法:
#   uv run python -B -m mjlab.scripts.Backup.diag_righting_primitives \
#       --checkpoint logs/rsl_rl/SQuRo_Backup/<run>/model_8999.pt
from __future__ import annotations

import csv
import importlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.SQuRo_Backup.mdp import entity as mdp_entity
from mjlab.tasks.SQuRo_Backup.mdp.curriculums import WALL_X_POS, _STEPS_PER_ITER
from mjlab.tasks.SQuRo_Backup.mdp.entity import WALL_HALF_THICKNESS
from mjlab.tasks.SQuRo_Backup.mdp.indices import _ACTUATED_JOINT_NAMES, _MODEL_INDICES


TASK_NAME = "Mjlab-SQuRo-Backup"
BODY_NOMINAL_WIDTH = 0.070        # 站立构型左右宽 (F/H body 半宽 0.035 × 2)
RESET_FOOT_HALF_WIDTH = 0.0382    # reset 姿态足端 X 半宽 -> 走廊几何下界来源
PRIMITIVES = ("axial", "lateral", "sagittal")
# theta_max 暂取机构行程 (轴向 = 前后两段反向拧满的相对扭转 2×1.57, 弯曲 = 关节行程)。
# 论文「机构允许的最大实际形变幅值」确定后替换这里即可, S_i 同时落盘以便重算。
THETA_MAX = {"axial": 3.14, "lateral": 0.60, "sagittal": 0.60}
CONTROL_DT = 0.01                 # = env_cfg 的 timestep 0.002 × decimation 5
# 墙接触传感器的 primary: 机器人全部 body (几何名有空串, 只能按 body 取; 前缀互不重叠)
CONTACT_BODY_PATTERNS = (
    "base_Link", "FU_.*", "FD_.*", "F_spine.*", "F_body_Link",
    "H_spine.*", "H_body_Link", "Neck_.*", "FL_.*", "FR_.*", "HL_.*", "HR_.*",
)


@dataclass
class Cfg:
    checkpoint: str
    # 一档一个环境: 左墙距离 d, 训练课程从 0.20 收到 0.06 (步长 0.01)
    wall_neg_distances: tuple[float, ...] = (0.20, 0.18, 0.16, 0.14, 0.12, 0.11,
                                             0.10, 0.09, 0.08, 0.07, 0.06)
    collisions: tuple[bool, ...] = (True, False)   # 同标称墙位开/关碰撞 = 隔离"力学"与"感知"
    episodes: int = 2                              # 每个档位跑几个 12 s 回合
    time_scale: float = 1.0                        # 固定 λ, 与回放口径一致
    # 墙接触力: 12 body × 2 场 × 2 墙 = 96 个接触传感器, 实测把步频从 ~200 压到 26 步/秒
    wall_contact_sensor: bool = False
    device: str = "cuda:0"
    seed: int = 0
    out_dir: str = "logs/backup_primitives"


# 名义通道角的列序: [F_spine1, F_body, H_spine1, H_body] (参考表顺序)
def _spine_columns() -> list[int]:
    idx = {n: _ACTUATED_JOINT_NAMES.index(n) for n in
           ("F_spine1_joint", "F_body_joint", "H_spine1_joint", "H_body_joint")}
    return [idx["F_spine1_joint"], idx["F_body_joint"],
            idx["H_spine1_joint"], idx["H_body_joint"]]


# 归一化激活包络 (按自身峰值)
def _norm(env: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(env))) if env.size else 0.0
    return np.abs(env) / m if m > 1e-9 else np.zeros_like(env)


# 一个循环上的四个定义 + 任务量
def _cycle_metrics(t: np.ndarray, ch: dict[str, np.ndarray], phase: np.ndarray,
                   base_x: np.ndarray, foot_x: np.ndarray, duty: np.ndarray,
                   wall_f: np.ndarray, effort: dict[str, np.ndarray]) -> dict:
    T = float(t[-1] - t[0]) + CONTROL_DT
    out: dict = {"T": T}
    S = {}
    for k in PRIMITIVES:
        s = np.abs(ch[k])
        S[k] = float(s.sum() * CONTROL_DT)
        out[f"S_{k}"] = S[k]
        out[f"A_{k}"] = S[k] / (THETA_MAX[k] * T)
        out[f"rms_{k}"] = float(np.sqrt((ch[k] ** 2).mean()))
        out[f"peak_{k}"] = float(np.abs(ch[k]).max())
        out[f"tau_{k}"] = float((t * s).sum() / max(s.sum(), 1e-12) / T)
    tot = sum(S.values())
    for k in PRIMITIVES:
        out[f"C_{k}"] = S[k] / tot if tot > 0 else float("nan")
    ne = {k: _norm(ch[k]) for k in PRIMITIVES}
    for a in range(len(PRIMITIVES)):
        for b in range(a + 1, len(PRIMITIVES)):
            ka, kb = PRIMITIVES[a], PRIMITIVES[b]
            out[f"O_{ka}_{kb}"] = float(np.minimum(ne[ka], ne[kb]).sum() * CONTROL_DT / T)
    for p, tag in ((0, "P1"), (1, "P2"), (2, "P3")):
        out[f"dur_{tag}"] = float((phase == p).sum() * CONTROL_DT)
    out["base_x_absmax"] = float(np.abs(base_x).max())
    out["foot_x_absmax"] = float(np.abs(foot_x).max())
    out["foot_duty"] = float(duty.mean())
    out["wall_hit"] = float((wall_f > 0).any())
    hit = np.flatnonzero(wall_f > 0)
    out["t_wall_contact"] = float(hit[0] * CONTROL_DT) if len(hit) else float("nan")
    out["wall_force_max"] = float(wall_f.max())
    out["wall_force_impulse"] = float(wall_f.sum() * CONTROL_DT)
    for k, v in effort.items():
        out[k] = float(v.mean())
    return out


# 构造一个"一档一个环境"的环境 (右墙固定 +0.05, 左墙取本批最宽档, 之后逐环境改写)
def _build_env(cfg: Cfg, dists: tuple[float, ...], collision: bool, iter_num: int):
    importlib.import_module("mjlab.tasks.SQuRo_Backup.config")
    env_cfg = load_env_cfg(TASK_NAME)
    env_cfg.scene.num_envs = len(dists)
    env_cfg.events.pop("init_restricted_space", None)
    env_cfg.commands["backup_cmd"].fixed_time_scale = cfg.time_scale
    mdp_entity.configure_restricted_space(env_cfg, wall_x_neg=-max(dists), wall_x_pos=WALL_X_POS,
                                         enable_collision=collision, fixed_width=True)
    if collision and cfg.wall_contact_sensor:
        sensors = tuple(env_cfg.scene.sensors or ())
        for tag, geom in (("neg", "restricted_space_wall_n_geom"),
                          ("pos", "restricted_space_wall_p_geom")):
            sensors += (ContactSensorCfg(
                name=f"wall_contact_{tag}",
                primary=ContactMatch(mode="body", pattern=CONTACT_BODY_PATTERNS, entity="robot"),
                secondary=ContactMatch(mode="geom", pattern=geom, entity="restricted_space"),
                fields=("found", "force"), reduce="netforce", num_slots=1),)
        env_cfg.scene.sensors = sensors
    raw = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
    raw.common_step_counter = iter_num * _STEPS_PER_ITER
    env = RslRlVecEnvWrapper(raw)
    return env, raw


# 跑一个 (碰撞开关) 配置, 一次并行扫完全部档位; 返回逐循环记录
def _run_config(cfg: Cfg, dists: tuple[float, ...], collision: bool, iter_num: int,
                policy_factory) -> list[dict]:
    env, _ = _build_env(cfg, dists, collision, iter_num)
    scene = env.unwrapped.scene
    cmd = env.unwrapped.command_manager.get_term("backup_cmd")
    robot = scene.entities["robot"]
    n_env = len(dists)
    feet_ids = _MODEL_INDICES.foot_site_ids
    joint_ids = _MODEL_INDICES.joint_ids
    spn_cols = _spine_columns()
    all_ids = torch.arange(n_env, device=env.device)
    d_t = torch.tensor(dists, device=env.device, dtype=torch.float32)
    # 逐环境墙位: 覆盖命令项每次重置采样的结果, 保证所有回合都是指定档位
    cmd._wall_d[all_ids] = d_t
    cmd._push_wall(all_ids)
    got_neg, got_pos = scene.entities["restricted_space"].read_wall_x(env.unwrapped)
    print(f"  自检 逐环境墙位 x_neg={[-round(float(v), 3) for v in got_neg]} "
          f"x_pos={round(float(got_pos[0]), 3)} 碰撞={'开' if collision else '关'}", flush=True)
    policy = policy_factory(env)
    obs = env.get_observations()
    ep_len = int(env.unwrapped.max_episode_length)
    buf: list[torch.Tensor] = []
    for _ in range(ep_len * cfg.episodes):
        with torch.inference_mode():
            act = policy(obs)
            obs, _, _, _ = env.step(act.to(env.device))
            cmd._wall_d[all_ids] = d_t
            cmd._push_wall(all_ids)
            q = robot.data.joint_pos[:, joint_ids]
            vel = robot.data.joint_vel[:, joint_ids]
            tau = robot.data.actuator_force
            if collision and cfg.wall_contact_sensor:
                wall_f = sum(scene[f"wall_contact_{t}"].data.force.abs().sum(dim=(1, 2))
                             for t in ("neg", "pos"))
            else:
                wall_f = torch.zeros(n_env, device=env.device)
            buf.append(torch.stack((
                cmd.phase.float(),
                q[:, spn_cols[0]], q[:, spn_cols[1]], q[:, spn_cols[2]], q[:, spn_cols[3]],
                robot.data.root_link_pos_w[:, 0],
                torch.abs(robot.data.site_pos_w[:, feet_ids, 0]).amax(dim=1),
                (scene["feet_ground_contact"].data.found > 0).any(dim=1).float(),
                torch.sqrt((vel ** 2).mean(dim=1)),
                (tau * vel).abs().sum(dim=1),
                (tau.abs() > 0.118).float().mean(dim=1),
                wall_f,
            ), dim=1))
    arr = torch.stack(buf).cpu().numpy()          # [T, N, 13]
    rows: list[dict] = []
    for n, d in enumerate(dists):
        clear = d + WALL_X_POS - 2 * WALL_HALF_THICKNESS
        for start, end, ep in _segment(arr[:, n, 0], ep_len):
            seg = arr[start:end + 1, n]
            # 两个"最小任务周期"口径: full = 整循环; action = 到进入 P3 为止的翻正动作段
            # (站立保持段三个基元都接近 0, 按整循环积分会把差异稀释掉)
            phase = seg[:, 0].astype(int)
            entry = np.flatnonzero(phase == 2)
            cut = int(entry[0]) if len(entry) else len(seg)
            for window, sl in (("full", seg),
                               ("action", seg[:cut] if cut >= 100 else seg[:0])):
                row = _finish(sl, cfg, clear, collision, ep, len(rows))
                if row:
                    row["wall_neg_distance"] = d
                    row["window"] = window
                    rows.append(row)
    env.close()
    return rows


# 按 phase 下降沿切循环: 只保留 >=1.0 s 且在回合内闭合的完整循环
def _segment(phase: np.ndarray, ep_len: int) -> list[tuple[int, int, int]]:
    boundaries = [i for i in range(1, len(phase)) if phase[i] == 0 and phase[i - 1] != 0]
    starts = [0] + boundaries
    out = []
    for k, s in enumerate(starts):
        e = (starts[k + 1] - 1) if k + 1 < len(starts) else (len(phase) - 1)
        if e - s + 1 < 100:
            continue
        out.append((s, e, s // ep_len))
    return out


# 把累积的步记录折算成一个循环的指标行
def _finish(arr: np.ndarray, cfg: Cfg, clear_width: float, collision: bool,
            ep: int, k: int) -> dict:
    n = len(arr)
    if n < 100:
        return {}
    t = np.arange(n) * CONTROL_DT
    ch = {"axial": arr[:, 2] - arr[:, 4], "lateral": arr[:, 1], "sagittal": arr[:, 3]}
    row = _cycle_metrics(
        t, ch, arr[:, 0].astype(int), arr[:, 5], arr[:, 6], arr[:, 7], arr[:, 11],
        {"joint_vel_rms": arr[:, 8], "power": arr[:, 9], "torque_sat": arr[:, 10]})
    row.update({"clear_width": clear_width, "wall_over_body": clear_width / BODY_NOMINAL_WIDTH,
                "collision": int(collision), "episode": ep, "cycle": k})
    return row


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _rl_cfg_dict() -> dict:
    from dataclasses import asdict
    return asdict(load_rl_cfg(TASK_NAME))


def main() -> None:
    cfg = tyro.cli(Cfg)
    torch.manual_seed(cfg.seed)
    m = re.search(r"model_(\d+)", Path(cfg.checkpoint).name)
    iter_num = int(m.group(1)) if m else 0

    def policy_factory(env):
        runner = load_runner_cls(TASK_NAME)(env, _rl_cfg_dict(), None, device=cfg.device)
        runner.load(cfg.checkpoint, load_cfg={"actor": True}, strict=True,
                    map_location=cfg.device)
        return runner.get_inference_policy(device=cfg.device)

    rows: list[dict] = []
    lines: list[str] = []
    keys = ("A_axial", "A_lateral", "A_sagittal", "C_axial", "C_lateral", "C_sagittal",
            "tau_axial", "tau_lateral", "tau_sagittal", "O_axial_lateral",
            "O_axial_sagittal", "O_lateral_sagittal", "rms_axial", "rms_lateral",
            "rms_sagittal", "T", "t_wall_contact", "wall_force_max", "base_x_absmax",
            "foot_x_absmax")
    for collision in cfg.collisions:
        print(f"\n[配置] 碰撞={'开' if collision else '关'}  档位 d="
              f"{[round(d, 2) for d in cfg.wall_neg_distances]}", flush=True)
        got = _run_config(cfg, cfg.wall_neg_distances, collision, iter_num, policy_factory)
        rows.extend(got)
        by_width: dict[tuple[float, str], list[dict]] = {}
        for r in got:
            by_width.setdefault((r["clear_width"], r["window"]), []).append(r)
        for clear, window in sorted(by_width, key=lambda k: (-k[0], k[1])):
            sub = by_width[(clear, window)]
            agg = {k: _nanmean([r[k] for r in sub]) for k in keys}
            lines.append(
                f"净宽 {clear * 1000:5.0f} mm (W/W_body {clear / BODY_NOMINAL_WIDTH:4.2f}) "
                f"碰撞={'开' if collision else '关'} 窗口={window:6s} n={len(sub):2d}  "
                f"A=({agg['A_axial']:.2f},{agg['A_lateral']:.2f},{agg['A_sagittal']:.2f}) "
                f"C=({agg['C_axial']:.2f},{agg['C_lateral']:.2f},{agg['C_sagittal']:.2f}) "
                f"tau=({agg['tau_axial']:.2f},{agg['tau_lateral']:.2f},{agg['tau_sagittal']:.2f}) "
                f"O=({agg['O_axial_lateral']:.3f},{agg['O_axial_sagittal']:.3f},"
                f"{agg['O_lateral_sagittal']:.3f})  rms=({agg['rms_axial']:.2f},"
                f"{agg['rms_lateral']:.2f},{agg['rms_sagittal']:.2f})  T={agg['T']:.2f}s  "
                f"t_contact={agg['t_wall_contact']:.2f}s  Fw_max={agg['wall_force_max']:.1f}  "
                f"|x|max={agg['base_x_absmax'] * 1000:.1f}mm  "
                f"foot|x|max={agg['foot_x_absmax'] * 1000:.1f}mm")
            print(lines[-1], flush=True)
    if not rows:
        raise SystemExit("没有采到任何完整循环")
    out_dir = Path(cfg.out_dir) / Path(cfg.checkpoint).parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"primitives_iter{iter_num}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    (out_dir / f"summary_iter{iter_num}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[INFO] 逐循环明细: {out_path}  ({len(rows)} 个循环)")
    print(f"[INFO] 汇总: {out_dir / f'summary_iter{iter_num}.txt'}")


if __name__ == "__main__":
    main()
