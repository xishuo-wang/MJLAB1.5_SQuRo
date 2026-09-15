from __future__ import annotations
import sys
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES, resolve_model_indices
from mjlab.tasks.SQuRo_Backup.mdp.reference import get_reference_joint_state


# 经济性测算: 同一环境、同一回合长度下, 比较几种策略能拿到的 Episode_Reward 构成
#   1) 零动作 (仰面躺)
#   2) 纯开环跟踪参考 (完美跟踪的理想上界)
#   3) 跟踪参考 + 在站立姿态静止 (模拟"翻正成功后不动") —— 仅作对比参考
#
# 关键读数: 里程碑三项在"完美跟踪"下能拿多少, 与密集项相比占比如何

KEYS = [
    "Episode_Reward/mimic_pos",
    "Episode_Reward/mimic_vel",
    "Episode_Reward/spine_target",
    "Episode_Reward/height",
    "Episode_Reward/milestone_s1",
    "Episode_Reward/milestone_s2",
    "Episode_Reward/milestone_success",
    "Episode_Reward/progress_s1",
    "Episode_Reward/progress_s2",
    "Episode_Reward/action_L1",
    "Episode_Reward/action_L2",
    "Episode_Reward/energy",
]


def run_case(env, fn, n_steps):
    env.reset()
    acc: dict[str, float] = {}
    term_acc: dict[str, float] = {}
    obs = env.unwrapped.get_observations()
    for _ in range(n_steps):
        obs, _rew, _d, _to, extras = env.step(fn(obs))
        log = extras.get("log", {})
        for k in KEYS:
            if k in log:
                acc[k] = acc.get(k, 0.0) + float(log[k])
        for k in ("Episode_Termination/timeout", "Episode_Termination/stand"):
            if k in log:
                term_acc[k] = term_acc.get(k, 0.0) + float(log[k])
    return acc, term_acc


def main() -> None:
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    cfg.scene.num_envs = 64
    cfg.commands["backup_cmd"].fixed_time_scale = lam  # type: ignore[attr-defined]
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
    asset = env.unwrapped.scene.entities["robot"]
    resolve_model_indices(asset)

    default = asset.data.default_joint_pos[:, _MODEL_INDICES.joint_ids]
    scale = float(env.unwrapped.cfg.actions["joint_pos"].scale)  # type: ignore[union-attr]
    n = int(env.unwrapped.max_episode_length)
    dev = env.unwrapped.device
    print(f"λ={lam}  {cfg.scene.num_envs} envs  episode {n} 步 ({n*float(env.step_dt):.1f}s)")

    def zero(o):
        return torch.zeros(cfg.scene.num_envs, 14, device=dev)

    def openloop(o):
        del o
        ref, _ = get_reference_joint_state(env.unwrapped)
        return (ref - default) / scale

    for tag, fn in [("零动作 (仰面躺)", zero), ("纯开环跟踪参考", openloop)]:
        acc, term = run_case(env, fn, n)
        dense = sum(v for k, v in acc.items() if "milestone" not in k and "progress" not in k)
        sparse = sum(v for k, v in acc.items() if "milestone" in k)
        prog = sum(v for k, v in acc.items() if "progress" in k)
        total = dense + sparse + prog
        print(f"\n=== {tag} ===")
        for k in KEYS:
            print(f"    {k.split('/')[-1]:18s} {acc.get(k, 0.0):+9.4f}")
        print(f"    {'-'*30}")
        print(f"    {'密集项合计':18s} {dense:+9.4f}")
        print(f"    {'区间项合计':18s} {prog:+9.4f}")
        print(f"    {'里程碑合计':18s} {sparse:+9.4f}")
        print(f"    {'总计':18s} {total:+9.4f}")
        print(f"    {'成就相关占比':18s} {(sparse+prog)/total if total else 0.0:.4f}")
        print(f"    {'stand 终止占比':18s} {term.get('Episode_Termination/stand', 0.0)/n:.4f}")

    env.close()


if __name__ == "__main__":
    main()
