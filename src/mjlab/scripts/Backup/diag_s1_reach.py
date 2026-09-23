# 诊断: 同一批初始状态 + 确定性动作下, 对比检查点的 S1/S2 姿态达成率与关键门控量。
# 用来区分"策略真的更差"与"训练日志口径不同"。只读, 不落盘。
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.SQuRo_Backup.mdp import entity as mdp_entity
from mjlab.tasks.SQuRo_Backup.mdp.config import STAND_CONFIRM_DURATION, STAND_VEL_MEAN_MAX
from mjlab.utils.torch import configure_torch_backends

TASK = "Mjlab-SQuRo-Backup"


# 加载检查点并返回 (env, 包装后的 env, 推理策略)
def load_policy(ckpt: Path, device: str, num_envs: int, seed: int, align_iter: int):
    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    mdp_entity.configure_restricted_space(cfg, 0.40, enable_collision=False)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.common_step_counter = align_iter * 96
    wrapped = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK).clip_actions)
    runner_cls = load_runner_cls(TASK)
    runner = runner_cls(wrapped, asdict(load_rl_cfg(TASK)), log_dir=None, device=device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    return env, wrapped, runner.get_inference_policy(device=device)


# 跑一次确定性 rollout, 统计姿态达成与门控阻断原因
def probe(ckpt: Path, args) -> dict:
    env, wrapped, policy = load_policy(ckpt, args.device, args.num_envs, args.seed,
                                       args.align_iter)
    cmd = env.command_manager.get_term("backup_cmd")

    n, dev = args.num_envs, args.device
    ever_s1 = torch.zeros(n, dtype=torch.bool, device=dev)   # S1 瞬时候选 (前倒后正+贴地)
    ever_s2 = torch.zeros(n, dtype=torch.bool, device=dev)   # S2 瞬时候选 (两段正置)
    ever_strict = torch.zeros(n, dtype=torch.bool, device=dev)
    s1_steps = torch.zeros(n, device=dev)
    max_prog_s1 = torch.zeros(n, device=dev)
    best_u_floor = torch.full((n,), -2.0, device=dev)
    max_h_floor = torch.zeros(n, device=dev)
    min_vel_rms = torch.full((n,), float("inf"), device=dev)
    fail_speed = 0
    fail_geom = 0

    obs = wrapped.get_observations().to(dev)
    for _ in range(args.steps):
        with torch.inference_mode():
            obs, _, _, _ = wrapped.step(policy(obs).to(wrapped.device))
        # 直接复用生产判据, 避免诊断与训练口径不一致
        cand_s1 = cmd._check_S1()
        cand_s2 = cmd._check_S2()
        u_floor, h_floor, vel_rms = cmd.standing_metrics()
        _, strict = cmd.stand_gate()
        in_p3 = cmd.phase == 2
        ever_s1 |= cand_s1
        ever_s2 |= cand_s2
        ever_strict |= strict & in_p3
        s1_steps += cand_s1.float()
        max_prog_s1 = torch.maximum(max_prog_s1, cmd.progress_s1)
        best_u_floor = torch.maximum(best_u_floor, u_floor)
        max_h_floor = torch.maximum(max_h_floor, h_floor)
        min_vel_rms = torch.minimum(min_vel_rms, vel_rms)
        elapsed = cmd._stand_elapsed
        mean_vel = cmd._stand_vel_integral / elapsed.clamp_min(1e-9)
        full = elapsed >= (STAND_CONFIRM_DURATION - 1e-3)
        fail_speed += int((full & in_p3 & strict & (mean_vel > STAND_VEL_MEAN_MAX)).sum())
        fail_geom += int((in_p3 & ~strict & (elapsed > 0)).sum())

    env.close()
    return {
        "ckpt": ckpt.name,
        "s1_rate": float(ever_s1.float().mean()),
        "s2_rate": float(ever_s2.float().mean()),
        "strict_rate": float(ever_strict.float().mean()),
        "s1_step_ratio": float((s1_steps / args.steps).mean()),
        "max_prog_s1": float(max_prog_s1.mean()),
        "best_u_floor": float(best_u_floor.mean()),
        "max_h_floor": float(max_h_floor.mean()),
        "min_vel_rms": float(min_vel_rms.mean()),
        "fail_speed": fail_speed,
        "fail_geom": fail_geom,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--align-iter", type=int, default=900)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    configure_torch_backends()
    rows = [probe(Path(c), args) for c in args.ckpt]

    keys = ["s1_rate", "s2_rate", "strict_rate", "s1_step_ratio", "max_prog_s1",
            "best_u_floor", "max_h_floor", "min_vel_rms", "fail_speed", "fail_geom"]
    print()
    print("=== 确定性 rollout 姿态达成对比 (同 seed / 同初态 / %d 环境 x %d 步) ==="
          % (args.num_envs, args.steps))
    print("%-20s %s" % ("checkpoint", "  ".join(keys)))
    for r in rows:
        cells = []
        for k in keys:
            v = r[k]
            cells.append("%.4f" % v if isinstance(v, float) else str(v))
        print("%-20s %s" % (r["ckpt"], "  ".join(cells)))


if __name__ == "__main__":
    main()
