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


# 基线对照: 用零动作 / 随机动作 / 开环跟踪参考 跑完整 episode,
# 量化"什么都不做(仰面躺)"这个平凡解能拿多少奖励。
#   若零动作的累计回报接近训练日志里的 Mean reward → 策略尚未超出平凡解
#   逐项 Episode_Reward 构成可定位是哪一项在"养懒汉"

# 两个已踩过的坑:
#   env.action_space.shape 是 (num_envs, action_dim), 不能取 shape[0]
#   env.step() 返回 5 元组 (obs, rew, dones, timeouts, extras)

ACTION_DIM = 14


def rollout(env, fn, n_steps: int):
    env.reset()
    # 环境在 episode 结束时会自动 reset, 因此必须用 (1 - dones) 屏蔽跨 episode 累加,
    # 否则回报会被多个 episode 叠加而虚高。extras["log"] 里的 Episode_Reward/* 是
    # 每步的即时值, 直接逐步累加即可 (不含跨 episode 问题)。
    total = torch.zeros(env.num_envs, device=env.unwrapped.device)
    comp: dict[str, float] = {}
    obs = env.unwrapped.get_observations()
    for _ in range(n_steps):
        obs, rew, dones, _timeouts, extras = env.step(fn(obs))
        total = total * (1.0 - dones.float()) + rew
        for k, v in extras.get("log", {}).items():
            if k.startswith("Episode_Reward/"):
                comp[k] = comp.get(k, 0.0) + float(v)
    n = max(1, n_steps)
    return total.mean().item(), {k: v / n for k, v in comp.items()}


def main() -> None:
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    n_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    cfg.scene.num_envs = n_envs
    cfg.commands["backup_cmd"].fixed_time_scale = lam  # type: ignore[attr-defined]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    asset = env.unwrapped.scene.entities["robot"]
    resolve_model_indices(asset)

    n = int(env.unwrapped.max_episode_length)
    scale = float(env.unwrapped.cfg.actions["joint_pos"].scale)
    default = asset.data.default_joint_pos[:, _MODEL_INDICES.joint_ids]
    print(f"λ={lam}  {n_envs} envs  episode {n} 步 ({n * float(env.step_dt):.2f}s)  "
          f"action_dim={ACTION_DIM}  scale={scale}")

    def openloop(obs):
        # 开环跟踪参考表: action = (ref - default) / scale, 等价于手调脚本的下发方式
        del obs
        pos, _ = get_reference_joint_state(env.unwrapped)
        return (pos - default) / scale

    cases = {
        "零动作 (纯仰面躺)": lambda o: torch.zeros(n_envs, ACTION_DIM, device=device),
        "随机动作 U(-1,1)": lambda o: 2 * torch.rand(n_envs, ACTION_DIM, device=device) - 1,
        "开环跟踪参考": openloop,
    }

    for tag, fn in cases.items():
        ret, comp = rollout(env, fn, n)
        print(f"\n=== {tag} ===")
        print(f"  累计回报(均值): {ret:.2f}")
        for k in sorted(comp):
            if abs(comp[k]) > 1e-6:
                print(f"    {k:36s} {comp[k]:+9.4f}")

    print("\n[训练日志对照] iter 22: Mean reward 48.28 | iter 95: 102.84")
    print("[训练日志对照] milestone_s1/s2/success 全程 0; height_actual 0.0311~0.0320")
    env.close()


if __name__ == "__main__":
    main()
