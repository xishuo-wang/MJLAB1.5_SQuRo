# 探针: 量"机器人在真实 rollout 里压进墙里多少" (只读, 不落盘)
# 墙是刚性静态几何, 但接触约束的软硬由 solref/solimp 决定; 不显式给就是 MuJoCo 默认软接触。
# 本探针直接读 mjwarp 的接触数据 (Contact 带 worldid, 可精确定位每个世界的墙-机器人接触对),
# 统计最小有符号距离 (负 = 压入) 和压入分布, 同时给出三个回合级比率以便对比行为变化。
# 用法:
#   uv run python -B -m mjlab.scripts.Backup.probe_wall_penetration \
#       --checkpoint <path> --wall-x-neg -0.05 --wall-x-pos 0.05 --solref 0.005,1.0
#   --solref "" 表示沿用代码默认 (硬接触); 给 "0.02,1.0" 即复现旧的软接触做对照。
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.SQuRo_Backup.mdp import entity as mdp_entity
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES
from mjlab.tasks.SQuRo_Backup.rl.runner import _STEPS_PER_ITER


@dataclass
class Cfg:
    checkpoint: str
    wall_x_neg: float = -0.05
    wall_x_pos: float = 0.05
    solref: str = "0.005,1.0"      # "t,d"; 空串 = 用代码默认
    solimp: str = ""               # 空串 = 用代码默认
    mode: str = "sample"           # sample | deterministic
    num_envs: int = 128
    steps: int = 2600
    device: str = "cuda:0"
    seed: int = 0


def parse_pair(s: str) -> tuple[float, ...] | None:
    s = s.strip()
    if not s:
        return None
    return tuple(float(v) for v in s.split(",") if v.strip())


def main() -> None:
    cfg = tyro.cli(Cfg)
    torch.manual_seed(cfg.seed)
    m = re.search(r"model_(\d+)", Path(cfg.checkpoint).name)
    it = int(m.group(1)) if m else 0

    env_cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.events.pop("init_restricted_space", None)
    mdp_entity.configure_restricted_space(env_cfg, wall_x_neg=cfg.wall_x_neg,
                                          wall_x_pos=cfg.wall_x_pos,
                                          enable_collision=True,
                                          solref=parse_pair(cfg.solref),
                                          solimp=parse_pair(cfg.solimp))
    raw = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
    raw.common_step_counter = it * _STEPS_PER_ITER
    env = RslRlVecEnvWrapper(raw)
    cmd = env.unwrapped.command_manager.get_term("backup_cmd")

    runner = load_runner_cls("Mjlab-SQuRo-Backup")(env, asdict(load_rl_cfg("Mjlab-SQuRo-Backup")),
                                                  None, device=cfg.device)
    runner.load(cfg.checkpoint, load_cfg={"actor": True}, strict=True, map_location=cfg.device)
    policy = runner.get_inference_policy(device=cfg.device)

    ent = env.unwrapped.scene.entities["restricted_space"]
    wall_ids = set(int(g) for g in ent.wall_geom_ids)
    robot_ids = set(int(g) for g in
                    env.unwrapped.scene.entities["robot"].indexing.geom_ids.detach().cpu().tolist())
    nw = cfg.num_envs

    print("=" * 92)
    print(f"检查点 {Path(cfg.checkpoint).name} (iter {it})  墙位 ({cfg.wall_x_neg:+.4f}, "
          f"{cfg.wall_x_pos:+.4f})  净宽 {cfg.wall_x_pos - cfg.wall_x_neg - 0.02:.4f} m")
    print(f"solref={ent.cfg.solref}  solimp={ent.cfg.solimp}  "
          f"({'代码默认' if not cfg.solref else '显式'})")
    print("=" * 92)

    obs = env.get_observations()
    seq_seen = cmd._ep_seq.detach().to("cpu").clone()
    stat = {"episodes": 0, "valid": 0, "onset": 0, "stood": 0, "success": 0}
    vt_sum, vt_n = 0.0, 0
    dists: list[np.ndarray] = []
    n_wall_contacts = 0
    layout_checked = False
    step_ms = 0.0

    for k in range(cfg.steps):
        t0 = time.perf_counter()
        with torch.inference_mode():
            act = policy(obs, stochastic_output=True) if cfg.mode == "sample" else policy(obs)
            obs, _, _, _ = env.step(act.to(env.device))
        torch.cuda.synchronize()
        step_ms += (time.perf_counter() - t0) * 1000.0
        if k == 0:
            step_ms = 0.0        # 丢掉首步 (含 CUDA graph 首次启动)

        # 读接触: mjwarp 3.10 的 nacon 是**全世界总接触数** (array(1,), 不是逐世界),
        # 而 contact.* 是 (naconmax,), **前 n 行** 才是有效接触, 每行自带 worldid。
        # 不要按世界分块索引 —— 那个假设已被 types.py 的定义否掉。
        wd = env.unwrapped.sim.wp_data
        n = int(wd.nacon.numpy()[0])
        if n > 0:
            g = wd.contact.geom.numpy()[:n]
            cd = wd.contact.dist.numpy()[:n]
            hit = np.isin(g[:, 0], list(wall_ids)) | np.isin(g[:, 1], list(wall_ids))
            if hit.any():
                n_wall_contacts += int(hit.sum())
                dists.append(cd[hit].astype(np.float64))
            if not layout_checked:
                print(f"接触: 总 {n} 对, 其中墙相关 {int(hit.sum())} 对; "
                      f"worldid 范围 [{int(wd.contact.worldid.numpy()[:n].min())}, "
                      f"{int(wd.contact.worldid.numpy()[:n].max())}]")
                layout_checked = True

        seq = cmd._ep_seq.detach().to("cpu")
        new = (seq > seq_seen).nonzero(as_tuple=False).squeeze(-1)
        if len(new) > 0:
            valid = cmd._last_ep_valid.detach().to("cpu")[new]
            onset = cmd._last_ep_stood_onset.detach().to("cpu")[new]
            stood = cmd._last_ep_stood_pose.detach().to("cpu")[new]
            succ = cmd._last_ep_success.detach().to("cpu")[new]
            for i in range(len(new)):
                stat["episodes"] += 1
                if bool(valid[i]):
                    stat["valid"] += 1
                    stat["onset"] += int(onset[i])
                    stat["stood"] += int(stood[i])
                    stat["success"] += int(succ[i])
            seq_seen[new] = seq[new]
        active = cmd._stand_elapsed > 0
        if bool(active.any()):
            vt_sum += float(cmd.windowed_mean_vel()[active].mean())
            vt_n += 1

    v = max(1, stat["valid"])
    print(f"\n单步耗时 {step_ms / max(1, cfg.steps - 1):.2f} ms (含策略推理与 env.step)")
    print(f"回合: 收到 {stat['episodes']} (有效 {stat['valid']})")
    print(f"  p_onset = {stat['onset'] / v:.3f}   p_stood = {stat['stood'] / v:.3f}   "
          f"p_done = {stat['success'] / v:.3f}   窗口 V/T = "
          f"{(vt_sum / vt_n) if vt_n else float('nan'):.2f}")
    print("  (p_onset 只说明出现过站立窗口; 它 >0 而 p_stood =0 表示站得起来但维持不住)")

    print(f"\n墙-机器人接触: 累计 {n_wall_contacts} 对")
    if dists:
        d = np.concatenate(dists)
        pen = -d                       # 正值 = 压入深度
        print(f"  有符号距离 dist (负=压入): min={d.min():+.5f}  p1={np.percentile(d, 1):+.5f}  "
              f"p50={np.percentile(d, 50):+.5f}")
        print(f"  压入深度 (m): 最大 {pen.max():.5f}   p99 {np.percentile(pen, 99):.5f}   "
              f"p50 {np.percentile(pen, 50):.5f}")
        for thr in (0.0005, 0.001, 0.002, 0.005):
            print(f"    压入 > {thr * 1000:.1f} mm 的接触占 {100.0 * (pen > thr).mean():.2f}%")
    else:
        print("  全程没有墙-机器人接触 (墙没被碰到)")


if __name__ == "__main__":
    main()
