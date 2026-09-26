# 走廊间距的几何下界探测 — 用网格顶点算真实世界包围盒, 并按墙顶高度分带 (只读)
# 为什么必须分带: 墙只有 0.10 m 高, 高于墙顶的部分可以伸到墙外, 只有低于墙顶的部分才决定间距下界。
# 用法: uv run python -B -m mjlab.scripts.Backup.probe_corridor_floor
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.SQuRo_Backup.mdp.entity import DEFAULT_WALL_HEIGHT, WALL_HALF_THICKNESS
from mjlab.tasks.registry import load_env_cfg

_MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
_CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
_CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
_ELLIPSOID = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)


@dataclass
class ProbeCfg:
    device: str = "cuda:0"
    wall_height: float = DEFAULT_WALL_HEIGHT
    top_n: int = 12


# 单个 geom 的世界顶点: 网格读顶点, 图元按类型展开 (mesh 的 geom_size 恒为 0, 不能当盒子用)
def geom_points(mjm, g: int, xpos, xmat) -> np.ndarray:
    gtype = int(mjm.geom_type[g])
    if gtype == _MESH:
        mid = int(mjm.geom_dataid[g])
        adr = int(mjm.mesh_vertadr[mid])
        num = int(mjm.mesh_vertnum[mid])
        local = np.asarray(mjm.mesh_vert[adr:adr + num], dtype=np.float64)
    else:
        size = np.asarray(mjm.geom_size[g], dtype=np.float64)
        if gtype == _BOX:
            half = size[:3].copy()
        elif gtype in (_CAPSULE, _CYLINDER):
            half = np.array([size[0], size[0], size[1]])
        elif gtype == _ELLIPSOID:
            half = size[:3].copy()
        else:
            half = np.array([size[0]] * 3)
        local = np.array([[sx * half[0], sy * half[1], sz * half[2]]
                          for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    mat = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    return local @ mat.T + np.asarray(xpos, dtype=np.float64)


def main() -> None:
    cfg = tyro.cli(ProbeCfg)
    env_cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
    env.reset()

    mjm = env.sim.mj_model
    xpos = env.sim.wp_data.geom_xpos.numpy()[0]
    xmat = env.sim.wp_data.geom_xmat.numpy()[0]
    walls = set(env.scene.entities["restricted_space"].wall_geom_ids)
    wall_contype = int(mjm.geom_contype[next(iter(walls))])
    wall_conaff = int(mjm.geom_conaffinity[next(iter(walls))])
    wh = cfg.wall_height

    print("=" * 100)
    print(f"墙顶高度 = {wh:.3f} m   墙半厚 = {WALL_HALF_THICKNESS:.3f} m")
    print(f"墙体碰撞位 = contype {wall_contype} / conaffinity {wall_conaff}")
    print("=" * 100)

    rows = []
    for g in range(mjm.ngeom):
        name = mjm.geom(g).name or ""
        if g in walls or "floor" in name:
            continue
        # 能否与墙碰撞: MuJoCo 规则 (contype1 & conaffinity2) | (contype2 & conaffinity1)
        c_g, a_g = int(mjm.geom_contype[g]), int(mjm.geom_conaffinity[g])
        hits_wall = bool((c_g & wall_conaff) or (wall_contype & a_g))
        pts = geom_points(mjm, g, xpos[g], xmat[g])
        below = pts[pts[:, 2] < wh]
        rows.append({
            "name": name, "hits": hits_wall,
            "body": mjm.body(int(mjm.geom_bodyid[g])).name or "?",
            "all": (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max(),
                    pts[:, 2].min(), pts[:, 2].max()),
            "bx": (float(np.abs(below[:, 0]).max()) if below.size else 0.0),
            "n_below": int(below.shape[0]),
        })

    for tag, sel in (("全部机器人 geom", rows),
                     ("可与墙碰撞的 geom", [r for r in rows if r["hits"]]),
                     ("含可视 geom 的整体", [r for r in rows if not r["hits"]])):
        if not sel:
            print(f"\n[{tag}] 空集")
            continue
        lo = [min(r["all"][0] for r in sel), min(r["all"][2] for r in sel),
              min(r["all"][4] for r in sel)]
        hi = [max(r["all"][1] for r in sel), max(r["all"][3] for r in sel),
              max(r["all"][5] for r in sel)]
        print(f"\n[{tag}] {len(sel)} 个")
        print(f"  世界包围盒  X[{lo[0]:+.4f}, {hi[0]:+.4f}]  Y[{lo[1]:+.4f}, {hi[1]:+.4f}]"
              f"  Z[{lo[2]:+.4f}, {hi[2]:+.4f}]")

    coll = [r for r in rows if r["hits"]]
    allb = [r for r in rows]
    for tag, sel in (("可与墙碰撞", coll), ("全部(含可视)", allb)):
        if not sel:
            continue
        widest = max(r["bx"] for r in sel)
        print(f"\n[{tag}] 低于墙顶 (z<{wh:.3f}) 的 X 半宽上限 = {widest:.4f} m")
        print(f"  => 墙心 |x| 的理论下界 = {widest + WALL_HALF_THICKNESS:.4f} m "
              f"(= 半宽 + 半厚), 净宽下界 = {2 * widest:.4f} m")
        for neg in (-0.08, -0.07, -0.06, -0.05, -0.045, -0.04):
            inner = abs(neg) - WALL_HALF_THICKNESS
            margin = inner - widest
            print(f"    wall_x_neg={neg:+.3f}: 内壁 {inner:.4f}, 净宽 {inner + 0.07:.4f}, "
                  f"余量 {margin:+.4f} m ({'可行' if margin > 0 else '干涉'})")

    print(f"\n[低于墙顶部分最宽的 {cfg.top_n} 个 geom]")
    ranked = sorted(rows, key=lambda r: -r["bx"])[: cfg.top_n]
    print(f"  {'geom':26s} {'body':22s} {'|x|max':>8s}  X 范围                 Z 范围")
    for r in ranked:
        a = r["all"]
        print(f"  {r['name'][:26]:26s} {r['body'][:22]:22s} {r['bx']:8.4f}  "
              f"[{a[0]:+.4f}, {a[1]:+.4f}]  [{a[4]:+.4f}, {a[5]:+.4f}]")

    print("\n[按 body 归并: 低于墙顶的 X 半宽上限]")
    by_body: dict[str, float] = {}
    for r in rows:
        by_body[r["body"]] = max(by_body.get(r["body"], 0.0), r["bx"])
    for b, v in sorted(by_body.items(), key=lambda kv: -kv[1]):
        if v > 1e-6:
            print(f"  {b[:30]:30s} {v:.4f}")


if __name__ == "__main__":
    main()
