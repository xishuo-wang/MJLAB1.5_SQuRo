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


# 区间奖励 (progress) 权重标定测算 — 只测量, 不注册奖励项
# 拟定公式:
#   ramp(u, s) = clamp(s * u, 0, 1)        u = 背腹标记 site 的 belly->back 世界 Z 余弦
#   progress_s1 = min(ramp(uF, -1), ramp(uH, +1))   前段保持仰面 + 后段翻到俯卧
#   progress_s2 = min(ramp(uF, +1), ramp(uH, +1))   两段都翻到俯卧
# 输出的是"权重=1 时该奖励项在 Episode_Reward/* 上的读数"(每秒速率),
# 乘以候选权重即可直接与其他奖励项比较。

KEYS = [
    "Episode_Reward/mimic_pos",
    "Episode_Reward/mimic_vel",
    "Episode_Reward/spine_target",
    "Episode_Reward/height",
]


def ramp(u: torch.Tensor, sign: float) -> torch.Tensor:
    return (sign * u).clamp(0.0, 1.0)


def segment_u(asset, idx: int) -> torch.Tensor:
    pairs = _MODEL_INDICES.segment_belly_back_ids
    assert pairs is not None, "segment_belly_back_ids 未解析"
    belly_id, back_id = pairs[idx]
    sp = asset.data.site_pos_w
    delta = sp[:, back_id, :] - sp[:, belly_id, :]
    norm = torch.linalg.vector_norm(delta, dim=-1)
    valid = torch.isfinite(delta).all(dim=-1) & (norm > torch.finfo(sp.dtype).eps)
    u = delta[:, 2] / norm.clamp_min(torch.finfo(sp.dtype).tiny)
    return torch.where(valid, u, torch.full_like(u, float("nan")))


def run_case(env, fn, n_steps):
    env.reset()
    asset = env.unwrapped.scene.entities["robot"]
    dt = float(env.step_dt)
    dev = env.unwrapped.device
    obs = env.unwrapped.get_observations()
    # 逐帧累加: 积分 (秒) 与"权重=1 的 Episode_Reward 速率"
    itg1 = torch.zeros(env.unwrapped.num_envs, device=dev)
    itg2 = torch.zeros(env.unwrapped.num_envs, device=dev)
    s1_ts = torch.full((env.unwrapped.num_envs,), float("nan"), device=dev)
    s2_ts = torch.full((env.unwrapped.num_envs,), float("nan"), device=dev)
    s1_done = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=dev)
    s2_done = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=dev)
    t = torch.zeros(env.unwrapped.num_envs, device=dev)
    rew = {k: 0.0 for k in KEYS}
    for _ in range(n_steps):
        obs, _r, _d, _to, extras = env.step(fn(obs))
        uf = segment_u(asset, 0)
        uh = segment_u(asset, 1)
        p1 = torch.minimum(ramp(uf, -1.0), ramp(uh, +1.0))
        p2 = torch.minimum(ramp(uf, +1.0), ramp(uh, +1.0))
        itg1 += torch.nan_to_num(p1) * dt
        itg2 += torch.nan_to_num(p2) * dt
        t += dt
        newly1 = (p1 >= 0.999) & ~s1_done
        newly2 = (p2 >= 0.999) & ~s2_done
        s1_ts = torch.where(newly1, t, s1_ts)
        s2_ts = torch.where(newly2, t, s2_ts)
        s1_done |= p1 >= 0.999
        s2_done |= p2 >= 0.999
        log = extras.get("log", {})
        for k in KEYS:
            if k in log:
                rew[k] += float(log[k])
    ep_s = n_steps * dt
    out = {
        "progress_s1_itg": float(itg1.mean()),
        "progress_s2_itg": float(itg2.mean()),
        "progress_s1_rate": float(itg1.mean()) / ep_s,
        "progress_s2_rate": float(itg2.mean()) / ep_s,
        "s1_saturate_t": float(torch.nan_to_num(s1_ts, nan=-1.0).mean()),
        "s2_saturate_t": float(torch.nan_to_num(s2_ts, nan=-1.0).mean()),
        "rew": rew,
    }
    return out


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
    ep_s = n * float(env.step_dt)
    print(f"λ={lam}  {cfg.scene.num_envs} envs  episode {n} 步 ({ep_s:.1f}s)")

    def zero(o):
        return torch.zeros(cfg.scene.num_envs, 14, device=dev)

    def openloop(o):
        del o
        ref, _ = get_reference_joint_state(env.unwrapped)
        return (ref - default) / scale

    for tag, fn in [("零动作 (仰面躺)", zero), ("纯开环跟踪参考", openloop)]:
        r = run_case(env, fn, n)
        print(f"\n=== {tag} ===")
        print(f"    progress_s1  ∫dt={r['progress_s1_itg']:7.3f}s  Episode_Reward读数(权重1)={r['progress_s1_rate']:+7.4f}"
              f"  首次饱和 t={r['s1_saturate_t']:.2f}s")
        print(f"    progress_s2  ∫dt={r['progress_s2_itg']:7.3f}s  Episode_Reward读数(权重1)={r['progress_s2_rate']:+7.4f}"
              f"  首次饱和 t={r['s2_saturate_t']:.2f}s")
        for k in KEYS:
            print(f"    [对照] {k.split('/')[-1]:16s} {r['rew'].get(k, 0.0):+9.4f}")

    env.close()


if __name__ == "__main__":
    main()
