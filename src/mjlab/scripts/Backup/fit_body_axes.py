from __future__ import annotations
import sys
import torch
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.SQuRo_Backup_Replay import StateMachinePolicy
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES, resolve_model_indices


# 用 body 的 site 世界坐标做最小二乘姿态拟合 (不依赖任何四元数约定)
#   P_world = R @ P_local + t
# 对每个 body 用它全部 site 的 (local, world) 配对标定 R, 再报告:
#   各局部轴在世界系的指向 (R 的各列), 以及世界 Z 分量

FIT = {
    "F": ("F_body_Link", ["F_body_%d_site" % i for i in range(1, 10)]),
    "H": ("H_body_Link", ["H_body_%d_site" % i for i in range(1, 10)]),
}


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 求 R, t 使 R@P + t ≈ Q
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, cQ - R @ cP


def main() -> None:
    lam = 1.0
    cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    cfg.scene.num_envs = 1
    cfg.commands["backup_cmd"].fixed_time_scale = lam  # type: ignore[attr-defined]
    for tc in cfg.terminations.values():
        tc.func = lambda e: torch.zeros(e.num_envs, dtype=torch.bool, device=e.device)
    cfg.episode_length_s = 3.0

    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
    env.reset()
    a = env.unwrapped.scene.entities["robot"]
    resolve_model_indices(a)

    # 取 site 局部坐标 (从模型 XML 读)
    import mujoco
    from pathlib import Path
    import mjlab.asset_zoo.robots.SQuRo as sq
    xml = Path(sq.__file__).resolve().parent / "xmls" / "SQuRo.xml"
    m = mujoco.MjModel.from_xml_path(str(xml))
    local = {}
    for key, (bname, snames) in FIT.items():
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
        P = []
        for sn in snames:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, sn)
            assert sid >= 0, sn
            P.append(m.site_pos[sid])
        local[key] = np.array(P, dtype=float)

    order = {}
    for key, (bname, snames) in FIT.items():
        ids, names = a.find_sites(snames, preserve_order=True)
        order[key] = list(ids)

    pol = StateMachinePolicy(env, lam, 5, 0.3, log_events=False)
    print("用 site 世界坐标最小二乘拟合 body 姿态; 报告三个局部轴在世界系的 Z 分量")
    print("初始姿态: 机器人仰面躺, 两段躯干水平, 腹面朝上")
    print()
    print(f"{'t':>6} {'t_nom':>6} {'ph':>3} | {'F:ax0z':>8} {'F:ax1z':>8} {'F:ax2z':>8} | "
          f"{'H:ax0z':>8} {'H:ax1z':>8} {'H:ax2z':>8} | 拟合残差")
    with torch.no_grad():
        for i in range(200):
            before = pol.phase
            env.step(pol(env.unwrapped.get_observations()))
            d = a.data
            sp = d.site_pos_w[0].detach().cpu().numpy().astype(float)
            line = []
            res = []
            for key in ("F", "H"):
                Q = sp[order[key]]
                R, _ = kabsch(local[key], Q)
                line.append(R[:, 0][2])
                line.append(R[:, 1][2])
                line.append(R[:, 2][2])
                res.append(np.linalg.norm((R @ local[key].T).T - Q) / len(local[key]))
            t = pol._elapsed
            if (pol.phase != before) or i % 10 == 0:
                mark = "  <== 切换" if pol.phase != before else ""
                print(f"{t:6.2f} {t/lam:6.3f} {pol.phase:>3} | "
                      f"{line[0]:+8.3f} {line[1]:+8.3f} {line[2]:+8.3f} | "
                      f"{line[3]:+8.3f} {line[4]:+8.3f} {line[5]:+8.3f} | "
                      f"F={res[0]:.5f} H={res[1]:.5f}{mark}")
            if pol.phase == "DONE":
                break
    env.close()


if __name__ == "__main__":
    main()
