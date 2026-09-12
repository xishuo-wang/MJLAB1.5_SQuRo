from __future__ import annotations
import sys
import torch
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.SQuRo_backup_Replay import StateMachinePolicy
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES, resolve_model_indices


# 验收: mdp/command.py 新的 site 判据 (_check_S1 / _check_S2) 与独立地面真值是否一致
#   地面真值: belly_z - back_z < 0 ⇔ 该段正置 (与 command.py 同源但独立重算)
# 期望: 两个门控都在名义边界处成立, 且 S1 在 P1_END 之前不成立

SITES = (("F_body_belly_site", "F_body_back_site"),
         ("H_body_belly_site", "H_body_back_site"))


def main() -> None:
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    cfg.scene.num_envs = 1
    cfg.commands["backup_cmd"].fixed_time_scale = lam  # type: ignore[attr-defined]
    for tc in cfg.terminations.values():
        tc.func = lambda e: torch.zeros(e.num_envs, dtype=torch.bool, device=e.device)
    cfg.episode_length_s = 3.0 * lam + 1.0

    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
    env.reset()
    a = env.unwrapped.scene.entities["robot"]
    resolve_model_indices(a)
    term = env.unwrapped.command_manager.get_term("backup_cmd")

    ids = [tuple(a.find_sites(list(p), preserve_order=True)[0]) for p in SITES]
    pol = StateMachinePolicy(env, lam, 5, 0.3, log_events=False)

    def truth(i: int, d) -> bool:
        b, k = ids[i]
        return bool(d.site_pos_w[0, b, 2] < d.site_pos_w[0, k, 2])

    n_f1 = n_f2 = 0          # 两法不一致计数
    first_s1 = first_s2 = None
    print(f"λ={lam}  期望门控时刻: P1→P2 @ {0.80 * lam:.2f}s, P2→P3 @ {0.95 * lam:.2f}s")
    print(f"{'t':>6} {'t_nom':>6} {'ph':>3} | {'S1':>3} {'S2':>3} | "
          f"{'真F':>4} {'真H':>4} | {'一致':>4}")
    with torch.no_grad():
        for i in range(int((3.0 * lam) / env.step_dt)):
            before = pol.phase
            env.step(pol(env.unwrapped.get_observations()))
            d = a.data
            s1 = bool(term._check_S1()[0])
            s2 = bool(term._check_S2()[0])
            tF, tH = truth(0, d), truth(1, d)
            # 独立重算 S1/S2 的朝向部分 (高度条件同源)
            fz = float(d.body_link_pos_w[0, _MODEL_INDICES.f_body_id, 2])
            hz = float(d.body_link_pos_w[0, _MODEL_INDICES.h_body_id, 2])
            e1 = (not tF) and tH and fz < 0.03 and hz < 0.03
            e2 = tF and tH and fz < 0.04 and hz < 0.04
            if s1 != e1:
                n_f1 += 1
            if s2 != e2:
                n_f2 += 1
            if s1 and first_s1 is None:
                first_s1 = pol._elapsed
            if s2 and first_s2 is None:
                first_s2 = pol._elapsed
            if (pol.phase != before) or i % 10 == 0:
                mark = "  <== 切换" if pol.phase != before else ""
                print(f"{pol._elapsed:6.2f} {pol._elapsed / lam:6.3f} {pol.phase:>3} | "
                      f"{int(s1):>3} {int(s2):>3} | {int(tF):>4} {int(tH):>4} | "
                      f"{'OK' if (s1 == e1 and s2 == e2) else '**不符**':>4}{mark}")
            if pol.phase == "DONE":
                break
    print(f"\n[S1 首次成立] {first_s1 if first_s1 is not None else '未成立'} s")
    print(f"[S2 首次成立] {first_s2 if first_s2 is not None else '未成立'} s")
    print(f"[两法不一致帧数] S1={n_f1}  S2={n_f2}")
    print(f"[最终] phase={pol.phase} 重试 P1={pol.retry['P1']} P2={pol.retry['P2']}")
    env.close()


if __name__ == "__main__":
    main()
