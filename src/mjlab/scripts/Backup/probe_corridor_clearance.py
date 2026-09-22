# 受限空间几何复核 — 确认机器人在 backup 初态下的体轴朝向与左右横向净宽 (只读)
# 用法: uv run python -B -m mjlab.scripts.Backup.probe_corridor_clearance
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.SQuRo_Backup.mdp.indices import resolve_model_indices
from mjlab.tasks.registry import load_env_cfg


@dataclass
class ProbeCfg:
    device: str = "cuda:0"
    stand_steps: int = 400


# 从 GPU 侧读 geom 世界位置算包围盒 (步进后 CPU mj_data 不会自动同步)
def _gpu_extent(env, ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    xpos = env.sim.wp_data.geom_xpos.numpy()[0]
    pts = np.array([xpos[g] for g in ids])
    return pts.min(axis=0), pts.max(axis=0)


# 指定 geom 集合的世界包围盒
def extent(model, data, ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    pts = []
    for gid in ids:
        pos = data.geom_xpos[gid]
        mat = data.geom_xmat[gid].reshape(3, 3)
        size = model.geom_size[gid]
        gtype = model.geom_type[gid]
        if gtype == 6:
            half = np.array(size[:3])
        elif gtype == 5:
            half = np.array([size[0], size[0], size[1]])
        else:
            half = np.array([size[0]] * 3)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    pts.append(pos + mat @ np.array([sx * half[0], sy * half[1], sz * half[2]]))
    pts = np.array(pts)
    return pts.min(axis=0), pts.max(axis=0)


def show(tag: str, model, data, ids: list[int]) -> None:
    if not ids:
        print(f"  {tag:26s} (空集)")
        return
    lo, hi = extent(model, data, ids)
    print(f"  {tag:26s} X[{lo[0]:+.4f},{hi[0]:+.4f}]  Y[{lo[1]:+.4f},{hi[1]:+.4f}]  "
          f"Z[{lo[2]:+.4f},{hi[2]:+.4f}]")


def main() -> None:
    cfg = tyro.cli(ProbeCfg)
    env_cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    env_cfg.scene.num_envs = 1
    env_cfg.commands["backup_cmd"].fixed_time_scale = 1.0
    env_cfg.episode_length_s = 1.0e6
    env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
    env.reset()

    asset = env.scene.entities["robot"]
    resolve_model_indices(asset)
    mjm, mjd = env.sim.mj_model, env.sim.mj_data

    print("=" * 100)
    print("[1] 基座姿态 — 判断体轴在世界系的朝向")
    quat = np.array(asset.data.root_link_quat_w[0].detach().cpu(), dtype=np.float64)
    print(f"  root_link_quat_w = {np.round(quat, 6).tolist()}")
    w, x, y, z = quat
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    for axis, name in ((R[:, 0], "base body+X"), (R[:, 1], "base body+Y"), (R[:, 2], "base body+Z")):
        print(f"    {name} -> world ({axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f})")

    # 只统计机器人自身的 geom: 排除地面与受限空间墙体, 否则"最小通行宽度"会被墙污染
    scene = [g for g in range(mjm.ngeom)
             if "floor" in mjm.geom(g).name or "restricted_space" in mjm.geom(g).name]
    allr = [g for g in range(mjm.ngeom) if g not in scene]
    coll = [g for g in allr if mjm.geom_contype[g] > 0]
    feet = [g for g in coll if "foot" in mjm.geom(g).name]
    print("\n[2] geom 分组与世界包围盒 (已排除地面与受限空间墙体)")
    show("机器人全部", mjm, mjd, allr)
    show("机器人碰撞体", mjm, mjd, coll)
    show("足端碰撞", mjm, mjd, feet)
    show("身体/腿碰撞", mjm, mjd, [g for g in coll if g not in feet])

    print("\n[3] 躯干与头部的世界位置")
    for b in range(mjm.nbody):
        nm = mjm.body(b).name or ""
        if "body_Link" in nm or "Neck" in nm or "Head" in nm or "base_Link" in nm:
            p = mjd.xpos[b]
            print(f"    {nm:20s} ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")

    print("\n[4] 横向 (世界 X) 净宽核算 — 墙在 x=±a/2")
    lo, hi = extent(mjm, mjd, coll)
    half_x = max(abs(lo[0]), abs(hi[0]))
    half_y = max(abs(lo[1]), abs(hi[1]))
    print(f"  碰撞体 X 半宽上限 = {half_x:.4f} m   Y 半宽上限 = {half_y:.4f} m")
    print(f"  => a 的理论下界 = 2 x {half_x:.4f} = {2*half_x:.4f} m")
    for a in (0.4, 0.3, 0.25, 0.2):
        m = a / 2 - half_x
        print(f"    a={a:.2f}: 允许 |x|<={a/2:.3f}, 余量 {m:+.4f} m ({'可行' if m > 0 else '干涉'})")

    print("\n[5] 零动作稳定后的实际包围盒 (读 GPU geom_xpos, 不用未同步的 CPU mj_data)")
    zmax0 = _gpu_extent(env, allr)
    print(f"  仰卧初态 z_max = {zmax0[1][2]:.4f}")
    with torch.no_grad():
        zero = torch.zeros(1, 14, device=env.device)
        zmax = 0.0
        for k in range(cfg.stand_steps):
            env.step(zero)
            if k % 25 == 0:
                zmax = max(zmax, _gpu_extent(env, allr)[1][2])
    lo, hi = _gpu_extent(env, allr)
    print(f"  零动作 {cfg.stand_steps} 步后: X[{lo[0]:+.4f},{hi[0]:+.4f}] "
          f"Y[{lo[1]:+.4f},{hi[1]:+.4f}] Z[{lo[2]:+.4f},{hi[2]:+.4f}]")
    print(f"  过程 z_max = {zmax:.4f}   base z = {float(asset.data.root_link_pos_w[0, 2]):.4f}")


if __name__ == "__main__":
    main()
