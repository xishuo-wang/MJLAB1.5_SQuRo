from __future__ import annotations
import sys
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.SQuRo_backup_Replay import StateMachinePolicy
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Backup.mdp.indices import resolve_model_indices


# 诊断: 手调脚本的相位推进 与 训练 BackupCommand 的相位推进 为何不同步
# 逐帧打印两侧的 phase / t_phase, 定位差异


def main() -> None:
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    cfg.scene.num_envs = 1
    cfg.commands["backup_cmd"].fixed_time_scale = lam  # type: ignore[attr-defined]
    for tc in cfg.terminations.values():
        tc.func = lambda e: torch.zeros(e.num_envs, dtype=torch.bool, device=e.device)
    cfg.episode_length_s = 8.0 * lam + 2.0

    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
    env.reset()
    resolve_model_indices(env.unwrapped.scene.entities["robot"])
    cmd = env.unwrapped.command_manager.get_term("backup_cmd")
    pol = StateMachinePolicy(env, lam, 0, log_events=False)

    print(f"λ={lam}  step_dt={env.step_dt}")
    print(f"命令期望时长: P1_END*λ={0.8*lam:.2f}s  T3*λ={0.15*lam:.2f}s")
    print()
    print(f"{'t':>6} {'t/λ':>6} | {'脚本ph':>7} {'脚本tp':>8} | {'cmd ph':>7} {'cmd tp':>9} | "
          f"{'cmd expected1':>13} | 说明")
    prev = None
    with torch.no_grad():
        for i in range(int((6.0 * lam + 1.0) / env.step_dt)):
            env.step(pol(env.unwrapped.get_observations()))
            c_ph = int(cmd.phase[0])
            c_tp = float(cmd.t_phase[0])
            s_ph = pol.phase
            changed = (c_ph, s_ph) != prev
            if i % max(1, int(0.3 * lam / env.step_dt)) == 0 or changed:
                exp1 = 0.8 * lam
                note = ""
                if s_ph == "P2" and c_ph == 0:
                    note = "脚本已进 P2, cmd 仍在 P1"
                elif s_ph == "P3" and c_ph != 2:
                    note = f"脚本已进 P3, cmd 在 {c_ph}"
                print(f"{pol._elapsed:6.2f} {pol._elapsed/lam:6.2f} | {s_ph:>7} {pol.t_phase:8.2f} | "
                      f"{c_ph:>7} {c_tp:9.2f} | {exp1:13.2f} | {note}")
                prev = (c_ph, s_ph)
            if pol.phase == "DONE":
                break

    print()
    print(f"最终: 脚本 phase={pol.phase}  cmd phase={int(cmd.phase[0])} cmd t_phase={float(cmd.t_phase[0]):.2f}")
    print(f"cmd retry={int(cmd.retry[0])}")
    env.close()


if __name__ == "__main__":
    main()
