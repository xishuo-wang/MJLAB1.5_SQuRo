from __future__ import annotations
import sys
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES, resolve_model_indices
from mjlab.tasks.SQuRo_Backup.mdp.reference import (
    get_reference_joint_state, _get_body_traj, _stage_t_nom)
from mjlab.tasks.SQuRo_Backup.mdp.config import P1_END, P2_END

# 身体高度参考表的"速度一致性"检查
# backup_body_traj.npy 声明是 λ=1 开环重放录制的, 但训练用 λ∈[2,4]。
# 这里在纯开环跟踪下逐 λ 记录实际 F/H body 世界高度, 按**真实名义时间**与参考表对齐,
# 判断"完美跟踪者"能拿到多少 height 奖励、以及参考是否与 S1/S2 闸门的高度条件冲突。
#
# 注意: 名义时间必须从 command 的 _stage_t_nom 读, 不能按 step*dt/λ 推算 ——
# 阶段一旦推进, 参考时钟会从新阶段起点重新计时。

_TRAJ = _get_body_traj("cpu")


def ref_at(t_nom: float) -> tuple[float, float]:
    t = _TRAJ["t"].numpy()
    i = int(np.argmin(np.abs(t - t_nom)))
    return float(_TRAJ["zF"][i]), float(_TRAJ["zH"][i])


def main() -> None:
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    cfg.scene.num_envs = 1
    cfg.commands["backup_cmd"].fixed_time_scale = lam  # type: ignore[attr-defined]
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
    asset = env.unwrapped.scene.entities["robot"]
    resolve_model_indices(asset)
    default = asset.data.default_joint_pos[:, _MODEL_INDICES.joint_ids]
    scale = float(env.unwrapped.cfg.actions["joint_pos"].scale)  # type: ignore[union-attr]
    dt = float(env.step_dt)
    n = int(3.0 / dt * lam)          # 覆盖到名义 3.0 s

    env.reset()
    obs = env.unwrapped.get_observations()
    rows: list[tuple[float, float, float, float]] = []
    for _ in range(n):
        ref, _ = get_reference_joint_state(env.unwrapped)
        obs, _r, _d, _t, _e = env.step((ref - default) / scale)
        t_nom = float(_stage_t_nom(env.unwrapped)[0])
        zf = float(asset.data.body_link_pos_w[0, _MODEL_INDICES.f_body_id, 2])
        zh = float(asset.data.body_link_pos_w[0, _MODEL_INDICES.h_body_id, 2])
        rows.append((t_nom, zf, zh, _stage_t_nom(env.unwrapped)[0].item()))
    env.close()

    tn = np.array([r[0] for r in rows])
    zf = np.array([r[1] for r in rows])
    zh = np.array([r[2] for r in rows])

    print(f"λ={lam}  纯开环跟踪 {n} 步; 名义时间覆盖 {tn.min():.3f}~{tn.max():.3f}")
    print(f"{'名义时刻':>10} {'实际 zF':>9} {'实际 zH':>9} | {'参考 zF':>9} {'参考 zH':>9} | {'ΔzF':>8} {'ΔzH':>8} | r_height")
    for name, tgt in (("P1_END", P1_END), ("P2_END", P2_END)):
        i = int(np.argmin(np.abs(tn - tgt)))
        rf, rh = ref_at(tgt)
        af, ah = zf[i], zh[i]
        r_h = 0.5 * np.exp(-500 * (rf - af) ** 2) + 0.5 * np.exp(-500 * (rh - ah) ** 2)
        print(f"{name:>10} {af:9.4f} {ah:9.4f} | {rf:9.4f} {rh:9.4f} | {rf-af:+8.4f} {rh-ah:+8.4f} | {r_h:.3f}")
    # 全程: 完美跟踪者能拿到的 height 奖励均值, 以及它与参考的最大偏差
    d = np.abs(np.array([ref_at(t) for t in tn]) - np.stack([zf, zh], axis=1))
    print(f"全程 |Δ| 均值 zF={d[:,0].mean():.4f} zH={d[:,1].mean():.4f}  最大 zF={d[:,0].max():.4f} zH={d[:,1].max():.4f}")
    print(f"实际高度范围 zF={zf.min():.4f}~{zf.max():.4f}  zH={zh.min():.4f}~{zh.max():.4f}")


if __name__ == "__main__":
    main()
