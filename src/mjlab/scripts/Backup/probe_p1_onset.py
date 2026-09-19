# P1 起步瞬态解耦探针 — 分离"脊柱跟踪参考"与"腿部收缩"两个来源
# 三种模式:
#   full   : 脊柱 + 腿 都跟踪参考 (对照, 与 SQuRo_Backup_Replay 等价)
#   legs   : 腿跟踪参考, 脊柱四关节冻结在初态 (0)
#   spine  : 脊柱跟踪参考, 腿冻结在初态
# 判据: 若"腿收缩导致脊柱尖峰"成立, 则 legs 模式下脊柱不应出现尖峰。
# 用法: uv run python -B -m mjlab.scripts.Backup.probe_p1_onset --steps 60 --time-scale 2.0 --mode legs
from __future__ import annotations

from dataclasses import dataclass

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES, resolve_model_indices
from mjlab.tasks.SQuRo_Backup.mdp.reference import get_reference_joint_state
from mjlab.tasks.registry import load_env_cfg
from mjlab.scripts.SQuRo_Backup_Replay import T_OFFSET, slow1_target

SPINE_IDX = (0, 1, 8, 9)


# 取参考关节角 (按参考表顺序), 并把名义时间对齐到手调脚本的 time1 padding
# 注意: slow1_target 内部已按 scale 缩放过段长, 传入的必须是**未缩放**的 t_phase。
def ref_at(env: ManagerBasedRlEnv, t_phase: float, lam: float) -> tuple[torch.Tensor, list[float]]:
    cmd = env.command_manager.get_term("backup_cmd")
    saved = cmd.t_phase.clone()
    try:
        # get_reference_joint_state 读 command.t_phase, 其约定已是缩放后的相位时间
        cmd.t_phase[:] = t_phase
        ref_pos, ref_vel = get_reference_joint_state(env)
    finally:
        cmd.t_phase[:] = saved
    return ref_pos, slow1_target(T_OFFSET + t_phase, lam)


@dataclass
class ProbeCfg:
    steps: int = 40
    time_scale: float = 2.0
    device: str = "cuda:0"
    mode: str = "full"
    # 每多少步打印一行
    every: int = 1


def main() -> None:
    cfg = tyro.cli(ProbeCfg)
    env_cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    env_cfg.scene.num_envs = 1
    env_cfg.commands["backup_cmd"].fixed_time_scale = cfg.time_scale
    env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
    env.reset()

    asset = env.scene.entities["robot"]
    resolve_model_indices(asset)
    jids = _MODEL_INDICES.joint_ids
    names = [n.replace("_joint", "") for n in _MODEL_INDICES.actuated_joint_names] if hasattr(_MODEL_INDICES, "actuated_joint_names") else None
    if names is None:
        from mjlab.tasks.SQuRo_Backup.mdp.indices import _ACTUATED_JOINT_NAMES
        names = [n.replace("_joint", "") for n in _ACTUATED_JOINT_NAMES]
    idx = {n: k for k, n in enumerate(names)}
    watch = ["F_spine1", "F_body", "H_spine1", "H_body"]
    print(f"模式={cfg.mode}  λ={cfg.time_scale:g}  监视={' '.join(watch)}")

    scale = float(env.cfg.actions["joint_pos"].scale)
    offset = asset.data.default_joint_pos[:, jids]
    lam = cfg.time_scale
    dt = env.step_dt
    cmd = env.command_manager.get_term("backup_cmd")
    peak = {w: 0.0 for w in watch}

    print()
    print("%4s %6s | %s" % ("step", "t", " | ".join("%-24s" % w for w in watch)))
    print("%4s %6s | %s" % ("", "", " | ".join("%7s%8s%8s" % ("ref", "act", "err") for _ in watch)))
    for i in range(cfg.steps):
        t_phase = float(cmd.t_phase[0])
        ref_pos, hand = ref_at(env, t_phase, lam)
        # 手调脚本目标 (r[0],r[1],r[8],r[9] 对应四个脊柱; 腿在 r[4..7],r[10..13])
        target = torch.tensor(hand, device=env.device, dtype=torch.float32).unsqueeze(0)
        # 按模式冻结: 被冻结的组直接令目标 = 当前关节角, 使其不动
        cur = asset.data.joint_pos[:, jids].clone()
        freeze = []
        if cfg.mode == "legs":
            freeze = [k for k in range(14) if k not in SPINE_IDX]
        elif cfg.mode == "spine":
            freeze = list(SPINE_IDX)
        if freeze:
            target[0, freeze] = cur[0, freeze]
        action = (target - offset) / scale
        if i % cfg.every == 0:
            cells = []
            for w in watch:
                k = idx[w]
                r = float(ref_pos[0, k])
                a = float(cur[0, k])
                cells.append("%+7.3f %+8.3f %+8.3f" % (r, a, a - r))
                peak[w] = max(peak[w], abs(a - r))
            print("%4d %6.2f | %s" % (i, i * dt, " | ".join(cells)))
        env.step(action)

    print()
    print("本模式脊柱跟踪峰值误差: " + "  ".join(f"{w}={peak[w]:.3f}" for w in watch))
    env.close()


if __name__ == "__main__":
    main()
