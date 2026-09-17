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


# 用"独立于四元数"的几何真值判定 body_link_quat_w 的口径:
#   body_Link 相对 base_Link 的位置矢量在 base 系下是固定的(脊柱安装偏移);
#   若把 mjlab 的 q_root 作用在该固定矢量上, 能连续复现实测的世界位置差,
#   则 q_root 就是真正的"世界朝向", 从而 q_body(= q_root^-1 ⊗ q_body_world) 是相对量。
#
# 同时给出各段"背腹轴在 base 系 Z 上的投影"的两种候选, 与用户物理锚点比对:
#   锚点: 初始 (F倒, H倒) | T2末 (F倒, H正) | T3末 (F正, H正)

def quat2mat_np(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


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
    pol = StateMachinePolicy(env, lam, 5, 0.3, log_events=True)
    fb, hb = _MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id

    # 初始时刻记录 base->F / base->H 的 body 系固定偏移
    d = a.data
    qb0 = d.root_link_quat_w[0].detach().cpu().numpy().astype(float)
    pF0 = d.body_link_pos_w[0, fb].detach().cpu().numpy().astype(float)
    pH0 = d.body_link_pos_w[0, hb].detach().cpu().numpy().astype(float)
    pB0 = d.root_link_pos_w[0].detach().cpu().numpy().astype(float)
    Rb0 = quat2mat_np(qb0)
    off_F = Rb0.T @ (pF0 - pB0)      # base 系下 F 相对 base 的偏移
    off_H = Rb0.T @ (pH0 - pB0)
    print(f"base 系下固定偏移: F={np.round(off_F, 4).tolist()}  H={np.round(off_H, 4).tolist()}")
    print("（若上面的偏移在整段运动里能由 q_root 连续复现实测位置差, 则 q_root 为真世界朝向）")
    print()
    print(f"{'t':>6} {'t_nom':>6} {'ph':>3} | {'残差F':>8} {'残差H':>8} | "
          f"{'rawF':>7} {'rawH':>7} | {'世界F':>7} {'世界H':>7} | 物理判读(基于高度)")
    with torch.no_grad():
        for i in range(int((3.0 * lam) / env.step_dt)):
            before = pol.phase
            env.step(pol(env.unwrapped.get_observations()))
            d = a.data
            qb = d.root_link_quat_w[0].detach().cpu().numpy().astype(float)
            Rb = quat2mat_np(qb)
            pB = d.root_link_pos_w[0].detach().cpu().numpy().astype(float)
            pF = d.body_link_pos_w[0, fb].detach().cpu().numpy().astype(float)
            pH = d.body_link_pos_w[0, hb].detach().cpu().numpy().astype(float)
            rF = np.linalg.norm(pB + Rb @ off_F - pF)
            rH = np.linalg.norm(pB + Rb @ off_H - pH)
            qF = d.body_link_quat_w[0, fb].detach().cpu().numpy().astype(float)
            qH = d.body_link_quat_w[0, hb].detach().cpu().numpy().astype(float)
            rawF = 2.0 * (qF[2] * qF[3] + qF[0] * qF[1])
            rawH = 2.0 * (qH[2] * qH[3] + qH[0] * qH[1])
            RF = Rb @ quat2mat_np(qF)
            RH = Rb @ quat2mat_np(qH)
            wldF = RF[1, 2]
            wldH = RH[1, 2]
            # 物理判读: body 背腹轴(局部 +Y)在世界 Z 的指向; 用 base->body 高度差交叉验证
            dzF = pB[2] - pF[2]
            dzH = pB[2] - pH[2]
            t = pol._elapsed
            if (i % 10 == 0) or pol.phase != before:
                print(f"{t:6.2f} {t/lam:6.3f} {pol.phase:>3} | {rF:8.5f} {rH:8.5f} | "
                      f"{rawF:+7.3f} {rawH:+7.3f} | {wldF:+7.3f} {wldH:+7.3f} | "
                      f"dzF={dzF:+.4f}({'腹上' if dzF > 0 else '腹下'}) "
                      f"dzH={dzH:+.4f}({'腹上' if dzH > 0 else '腹下'})")
            if pol.phase == "DONE":
                break
    env.close()


if __name__ == "__main__":
    main()
