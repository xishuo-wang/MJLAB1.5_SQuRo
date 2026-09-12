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


# 地面真值验收: 用 XML 中新增的 F/H_body_belly_site 与 _back_site 的世界坐标差
#   背腹轴 d = p(belly) - p(back)         (世界系, 由 MuJoCo 直接给出, 不依赖四元数约定)
#   正置度   = -(d_z / |d|)               +1 = 正置(腹面朝下, 已翻正), -1 = 倒置(腹面朝上)
# 取负号是因为 d_z 在仰面躺时为 +1 (腹面朝上), 而本任务语境下"正置"应记 +1。
# 验收锚点: 初始(-1,-1) | T2末(-1,+1) | T3末(+1,+1)

SITES = {
    "F": ("F_body_belly_site", "F_body_back_site"),
    "H": ("H_body_belly_site", "H_body_back_site"),
}


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

    idx = {}
    for k, (sb, sk) in SITES.items():
        ids, names = a.find_sites([sb, sk], preserve_order=True)
        idx[k] = (ids[0], ids[1])
        print(f"  {k}: {names[0]}={ids[0]}  {names[1]}={ids[1]}")

    pol = StateMachinePolicy(env, lam, 5, 0.3, log_events=False)
    fb, hb = _MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id

    def seg_up(d, key):
        ia, ib = idx[key]
        v = (d.site_pos_w[0, ia] - d.site_pos_w[0, ib]).detach().cpu().numpy().astype(float)
        return -float(v[2] / (np.linalg.norm(v) + 1e-12))

    print(f"\nλ={lam}  正置度: +1=正置(腹面朝下)  -1=倒置(腹面朝上)")
    print(f"验收锚点: 初始(-1,-1) | T2末(-1,+1) | T3末(+1,+1)")
    print(f"{'t':>6} {'t_nom':>6} {'ph':>3} | {'base_up':>8} | {'F':>8} {'H':>8} | "
          f"{'zF':>7} {'zH':>7} | 判定")
    with torch.no_grad():
        for i in range(int((3.0 * lam) / env.step_dt)):
            before = pol.phase
            env.step(pol(env.unwrapped.get_observations()))
            d = a.data
            uF, uH = seg_up(d, "F"), seg_up(d, "H")
            t = pol._elapsed
            if (pol.phase != before) or i % 8 == 0:
                tg = lambda x: "倒置" if x < -0.5 else ("正置" if x > 0.5 else "侧立")
                mark = "  <== 切换" if pol.phase != before else ""
                print(f"{t:6.2f} {t/lam:6.3f} {pol.phase:>3} | "
                      f"{float(d.projected_gravity_b[0, 2]):+8.3f} | {uF:+8.3f} {uH:+8.3f} | "
                      f"{float(d.body_link_pos_w[0, fb, 2]):7.4f} "
                      f"{float(d.body_link_pos_w[0, hb, 2]):7.4f} | "
                      f"{tg(uF)},{tg(uH)}{mark}")
            if pol.phase == "DONE":
                break
    env.close()


if __name__ == "__main__":
    main()
