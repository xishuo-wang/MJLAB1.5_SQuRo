from __future__ import annotations
import sys
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

sys.path.insert(0, "src")
from mjlab.tasks.SQuRo_Backup.mdp.indices import _ACTUATOR_CTRL_RANGE

# 检查点动作质量对照: 站立窗口内的关节速度 / 力矩饱和 / 目标角相邻步变化
# 站立窗口必须用**与代码版本无关**的原始列定义, 不能用 stand_confirm_elapsed:
# 判据从 "0.5 s @ u>0.9" 改成 "1.5 s @ u>0.8" 之后, 同一个阈值选中的帧集完全不同
# (实测同一检查点 600: 旧口径 39 步 vs 新口径 139 步), 跨版本读数会被判据本身污染。
# 这里直接复现判据的"严格几何": P3 且 min(u_F,u_H) > 0.9 且 min(z_F,z_H) > 0.05 m。
SPINE = ("F_spine1", "F_body", "H_spine1", "H_body")
LEG = ("FL_shoulder", "FL_elbow", "FR_shoulder", "FR_elbow",
       "HL_hip", "HL_knee", "HR_hip", "HR_knee")
NECK = ("Neck_yaw", "Neck_pitch")
DEFAULT = {"F_spine1": 0.0, "F_body": 0.0, "Neck_yaw": 0.0, "Neck_pitch": 0.0,
           "FL_shoulder": 0.1, "FL_elbow": -0.3, "FR_shoulder": 0.1, "FR_elbow": -0.3,
           "H_spine1": 0.0, "H_body": 0.0,
           "HL_hip": -0.1, "HL_knee": 0.3, "HR_hip": -0.1, "HR_knee": 0.3}
FORCE = {"F_spine1": 0.15, "H_spine1": 0.15, "F_body": 0.2, "H_body": 0.2,
         "Neck_yaw": 0.1, "Neck_pitch": 0.1}
SCALE = 0.3


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")


def target_clamped(d, name):
    lo, hi = _ACTUATOR_CTRL_RANGE[name + "_joint"]
    return np.clip(d[name + "_action"].to_numpy(float) * SCALE + DEFAULT[name], lo, hi)


def report(tag: str, path: str) -> dict:
    d = pd.read_csv(path)
    n = len(d)
    st = d["step"].to_numpy()
    cont = np.diff(st) == 1                       # 同一 rollout 内相邻步
    u_floor = np.minimum(d["f_body_up_cos"].to_numpy(float), d["h_body_up_cos"].to_numpy(float))
    z_floor = np.minimum(d["f_body_height"].to_numpy(float), d["h_body_height"].to_numpy(float))
    stand = (d["backup_phase"].to_numpy() == 2) & (u_floor > 0.9) & (z_floor > 0.05)
    print(f"\n=== {tag} ===   步数 {n}   站立窗口占比 {stand.mean()*100:.1f}% ({stand.sum()} 步)")
    print(f"  阶段分布: P1 {(d['backup_phase']==0).mean()*100:5.1f}%  P2 {(d['backup_phase']==1).mean()*100:5.1f}%  P3 {(d['backup_phase']==2).mean()*100:5.1f}%"
          f"   重试帧 {int((d['phase_retry']>0).sum())}   done {int(d['done'].sum())}")
    print(f"  高度: mean {d['height_actual'].mean():.4f}  站立窗口内 mean {d['height_actual'][stand].mean():.4f}  max {d['height_actual'].max():.4f}")

    out = {}
    for label, mask in (("全段", np.ones(n, bool)), ("站立窗口", stand)):
        if mask.sum() < 5:
            continue
        sp = np.concatenate([d[f"{j}_vel"].to_numpy(float)[mask] for j in SPINE])
        lg = np.concatenate([d[f"{j}_vel"].to_numpy(float)[mask] for j in LEG])
        nk = np.concatenate([d[f"{j}_vel"].to_numpy(float)[mask] for j in NECK])
        out[label] = dict(spine=rms(sp), leg=rms(lg), neck=rms(nk))
        print(f"  [{label}] 关节速度 RMS: 脊柱 {rms(sp):5.2f}  腿 {rms(lg):5.2f}  颈 {rms(nk):5.2f} rad/s")

    m = stand
    sat = []
    for j in LEG:
        fr = FORCE.get(j, 0.12)
        sat.append(np.abs(d[f"{j}_torque"].to_numpy(float)[m]) >= 0.99 * fr)
    sat = np.concatenate(sat)
    dcl = []
    dact = []
    for j in LEG + SPINE + NECK:
        t = target_clamped(d, j)
        a = d[f"{j}_action"].to_numpy(float)
        c = cont & m[:-0] if False else cont
        keep = cont & m[1:] & m[:-1]
        dcl.append(np.diff(t)[keep])
        dact.append(np.diff(a)[keep])
    dcl = np.concatenate(dcl)
    dact = np.concatenate(dact)
    print(f"  [站立窗口] 腿部力矩饱和比例 {sat.mean()*100:.1f}%   "
          f"目标角相邻步变化 RMS {rms(dcl):.3f} rad   |Δaction| RMS {rms(dact):.2f}")
    print(f"  站立窗口内步奖励 mean {d['reward'].to_numpy(float)[m].mean():+.4f} / step")
    return out


def main() -> None:
    for tag, p in [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in sys.argv[1:]]:
        report(tag, p)


if __name__ == "__main__":
    main()
