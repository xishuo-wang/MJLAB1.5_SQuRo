# 诊断: 用已有检查点评估墙位课程的门控量是否达标 (只读, 不落盘)
#   p_stood : 回合"站起来"率 = 站姿维持满确认窗口且当步严格几何, **不含关节速度** <- 课程推进量
#   p_onset : 回合内出现过站立窗口 (单帧, 备用/诊断)
#   p_done  : 回合稳定站立成功率 (现行判据, 含 V/T <= 门限)
# 训练时的门控量是在**采样动作**下统计的, 所以默认两种动作模式都跑, 以 sample 为准。
# 用法:
#   uv run python -B -m mjlab.scripts.Backup.diag_curriculum_gate <checkpoint> \
#       --wall_x_neg -0.20 --wall_x_pos 0.08
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.SQuRo_Backup.mdp import entity as mdp_entity
from mjlab.tasks.SQuRo_Backup.mdp.curriculums import (
    CURRICULUM_GATE_P_STOOD,
    _STEPS_PER_ITER,
)

TASK_NAME = "Mjlab-SQuRo-Backup"


@dataclass
class Cfg:
    checkpoint: str
    wall_x_neg: float = -0.20          # 墙位必须显式给 (自动课程的墙位无法按轮次反推)
    wall_x_pos: float = 0.08
    enable_collision: bool = True
    mode: str = "both"                 # sample | deterministic | both
    num_envs: int = 128
    steps: int = 3900                  # 约 3 个 12s 回合 (首个回合无效)
    device: str = "cuda:0"
    seed: int = 0


# 跑一种动作模式, 统计回合级的三个率 (与训练侧 _ingest_episode_results 同一套发布机制)
def run_mode(name: str, env, policy, cmd, steps: int, num_envs: int) -> dict:
    obs = env.get_observations()
    # 只统计本模式开始之后发布的回合: 起点取当前序号快照, **不能**对推理张量做就地清零
    seq_seen = cmd._ep_seq.detach().to("cpu").clone()
    stat = {"episodes": 0, "valid": 0, "onset": 0, "stood": 0, "success": 0}
    vt_sum, vt_n = 0.0, 0
    for _ in range(steps):
        with torch.inference_mode():
            act = policy(obs, stochastic_output=True) if name == "sample" else policy(obs)
            obs, _, _, _ = env.step(act.to(env.device))
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
        # 判据量本身: 站立窗口活跃环境上的 V/T
        active = cmd._stand_elapsed > 0
        if bool(active.any()):
            vt_sum += float(cmd.windowed_mean_vel()[active].mean())
            vt_n += 1
    stat["mean_vt"] = (vt_sum / vt_n) if vt_n else float("nan")
    return stat


def main() -> None:
    cfg = tyro.cli(Cfg)
    torch.manual_seed(cfg.seed)
    m = re.search(r"model_(\d+)", Path(cfg.checkpoint).name)
    it = int(m.group(1)) if m else 0

    env_cfg = load_env_cfg(TASK_NAME)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.events.pop("init_restricted_space", None)
    mdp_entity.configure_restricted_space(env_cfg, wall_x_neg=cfg.wall_x_neg,
                                         wall_x_pos=cfg.wall_x_pos,
                                         enable_collision=cfg.enable_collision)
    raw = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
    raw.common_step_counter = it * _STEPS_PER_ITER      # 让 λ 课程落在训练同段
    env = RslRlVecEnvWrapper(raw)
    cmd = env.unwrapped.command_manager.get_term("backup_cmd")

    runner = load_runner_cls(TASK_NAME)(env, asdict(load_rl_cfg(TASK_NAME)), None,
                                        device=cfg.device)
    runner.load(cfg.checkpoint, load_cfg={"actor": True}, strict=True,
                map_location=cfg.device)
    policy = runner.get_inference_policy(device=cfg.device)

    net = cfg.wall_x_pos - cfg.wall_x_neg - 0.02
    print("=" * 92)
    print(f"检查点 {Path(cfg.checkpoint).name} (iter {it})  墙位 x_neg={cfg.wall_x_neg:+.4f} "
          f"x_pos={cfg.wall_x_pos:+.4f} (净宽 {net:.4f} m)  碰撞={cfg.enable_collision}  "
          f"环境数={cfg.num_envs}  步数={cfg.steps}")
    print(f"课程门控: p_stood >= {CURRICULUM_GATE_P_STOOD}")
    print("=" * 92)
    modes = ["deterministic", "sample"] if cfg.mode == "both" else [cfg.mode]
    for name in modes:
        s = run_mode(name, env, policy, cmd, cfg.steps, cfg.num_envs)
        v = max(1, s["valid"])
        print(f"\n[{name} 动作]")
        print(f"  收到回合 {s['episodes']} 个 (有效 {s['valid']} 个)")
        print(f"  p_onset (站立窗口出现)   = {s['onset'] / v:.3f}")
        print(f"  p_stood (站起来并维持)   = {s['stood'] / v:.3f}"
              f"   {'>= 门控, 会推进' if s['stood'] / v >= CURRICULUM_GATE_P_STOOD else '< 门控, 不推进'}")
        print(f"  p_done  (稳定站立成功)   = {s['success'] / v:.3f}")
        print(f"  站立窗口内平均 V/T       = {s['mean_vt']:.2f} rad/s")
        # 每次换模式前重置环境, 避免上一次的回合尾巴混进统计 (序号快照在 run_mode 里取)。
        # 必须在 inference_mode 内: 上一步 step 已把管理器里的缓冲张量标成推理张量,
        # 在外面做就地写会直接抛错。
        with torch.inference_mode():
            env.reset()


if __name__ == "__main__":
    main()
