# 单步耗时分解探测 — 定位"与环境数无关的固定开销"花在哪一段 (只读, 不改训练代码)
# 用法: uv run python -B -m mjlab.scripts.Backup.probe_step_budget --num-envs 1024
# 消融: --ablate metrics|legpose|both, 用监控补丁关掉两处 .item() 密集的上报, 量同步的真实代价
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.SQuRo_Backup.mdp import rewards as backup_rewards
from mjlab.tasks.registry import load_env_cfg


@dataclass
class ProbeCfg:
    device: str = "cuda:0"
    num_envs: int = 1024
    warmup: int = 20
    steps: int = 60
    ablate: str = "none"          # none | metrics | legpose | both


# 用 cuda synchronize 卡住两端逐段计时; 段耗时在 ms 量级, 同步本身的开销可忽略
class Timer:
    def __init__(self) -> None:
        self.acc: dict[str, float] = {}
        self.n = 0

    def add(self, name: str, dt: float) -> None:
        self.acc[name] = self.acc.get(name, 0.0) + dt

    def mark(self, name: str) -> None:
        torch.cuda.synchronize()
        self._t = time.perf_counter()
        self._name = name

    def done(self) -> None:
        torch.cuda.synchronize()
        self.add(self._name, time.perf_counter() - self._t)


def main() -> None:
    cfg = tyro.cli(ProbeCfg)
    env_cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    env_cfg.scene.num_envs = cfg.num_envs
    env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
    cmd = env.command_manager.get_term("backup_cmd")

    if cfg.ablate in ("metrics", "both"):
        cmd._update_metrics = lambda: None
    if cfg.ablate in ("legpose", "both"):
        backup_rewards._log_leg_pose = lambda *a, **k: None

    obs = env.get_observations()
    act = torch.zeros(cfg.num_envs, sum(env.action_manager.action_term_dim), device=cfg.device)
    for _ in range(cfg.warmup):
        env.step(act)

    tm = Timer()
    for _ in range(cfg.steps):
        # 以下顺序照抄 ManagerBasedRlEnv.step 的主体, 只加计时点
        tm.mark("action.process")
        env.action_manager.process_action(act.to(env.device))
        tm.done()

        tm.mark("physics(5 substep)")
        for _ in range(env.cfg.decimation):
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
            env.sim.step()
            env.metrics_manager.compute_substep()
        tm.done()

        tm.mark("termination")
        env.reset_buf = env.termination_manager.compute()
        tm.done()

        tm.mark("reward_manager")
        env.reward_buf = env.reward_manager.compute(dt=env.step_dt)
        tm.done()

        tm.mark("metrics_manager")
        env.metrics_manager.compute()
        tm.done()

        tm.mark("sim.forward")
        env.sim.forward()
        tm.done()

        tm.mark("command_manager")
        env.command_manager.compute(dt=env.step_dt)
        tm.done()

        tm.mark("event(step)")
        if "step" in env.event_manager.available_modes:
            env.event_manager.apply(mode="step", dt=env.step_dt)
        if "interval" in env.event_manager.available_modes:
            env.event_manager.apply(mode="interval", dt=env.step_dt)
        tm.done()

        tm.mark("sim.sense")
        env.sim.sense()
        tm.done()

        tm.mark("observation_manager")
        env.obs_buf = env.observation_manager.compute(update_history=True)
        tm.done()

        tm.mark("recorder")
        env.recorder_manager.record_post_step()
        tm.done()

        tm.n += 1

    total = sum(tm.acc.values()) / tm.n
    print("=" * 84)
    print(f"环境数 {cfg.num_envs}  消融 {cfg.ablate}  每步合计 {total:.2f} ms  "
          f"(折算 {1000.0 / total * cfg.num_envs:.0f} 控制步/s)")
    print("=" * 84)
    print(f"  {'段':24s} {'ms/步':>9s} {'占比':>8s}")
    for k, v in sorted(tm.acc.items(), key=lambda kv: -kv[1]):
        ms = v / tm.n
        print(f"  {k:24s} {ms:9.3f} {100.0 * ms / total:7.1f}%")
    # 单步物理 (含 5 次 substep 之外的其余部分) 用于判断是不是"物理算不动"
    print(f"\n  物理段每个 substep = {tm.acc['physics(5 substep)'] / tm.n / env.cfg.decimation:.3f} ms")


if __name__ == "__main__":
    main()
