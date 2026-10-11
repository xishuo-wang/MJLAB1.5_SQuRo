# uv run python -B -m mjlab.scripts.Hole.verify_hole_obstacle_z
# 验收: 限高板运行期改板底高度 (mocap) 这条链
#   [1] 板体是独立 mocap body, 不贡献自由度, 复位后板底 == HOLE_LAYOUT 默认
#   [2] 逐环境写板底 + sim.forward 后, geom 世界高度跟随 (物理链路真的动了)
#   [3] 碰撞跟着板走: 两个板底高度下机器人落点差 ≈ 板底差
#   [4] mode="reset" 采样事件: 复位后板底 == 采样值, 不被 sim.reset 打回默认
#   [5] cfg 口径与运行期口径在默认配置下一致 (Phase B 改奖励前的基线)
# 任一失败以非零码退出
from __future__ import annotations

import sys

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import EventTermCfg
from mjlab.tasks.SQuRo_Hole.mdp import (
    HOLE_LAYOUT,
    OBSTACLE_Z_HIGH,
    OBSTACLE_Z_LOW_START,
    reset_obstacle_height,
)
from mjlab.tasks.SQuRo_Hole.mdp.entity import configure_hole_entities, hole_geometry
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-SQuRo-Hole"
BOARD_KEYS = ("hole1", "hole2", "hole3")
N = 6
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"    {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        fails.append(msg)


def np_(t) -> np.ndarray:
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


# 建环境; collision 控制实体碰撞, disable_terminations 供"落下不被自动复位"的测试用
def build(collision: bool, num_envs: int = N, disable_terminations: bool = False):
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = num_envs
    configure_hole_entities(cfg, enable_collision=collision)
    if disable_terminations:
        cfg.terminations = {}
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    return env


# [1] 结构与复位默认值
def check_structure() -> None:
    print("[1] 板体 mocap 结构 (默认配置, 不注册采样事件)")
    env = build(False)
    mjm = env.sim.mj_model
    ents = [env.scene.entities[k] for k in BOARD_KEYS]
    body_ids = [e.board_body_id for e in ents]
    mocap_ids = [e.board_mocap_id for e in ents]
    print(f"     nmocap={mjm.nmocap} nq={mjm.nq} nv={mjm.nv}")
    print(f"     板 body 下标={body_ids}  mocap 下标={mocap_ids}")
    check(all(m >= 0 for m in mocap_ids), "三块板都有有效 mocap 下标")
    check(len(set(mocap_ids)) == 3, "三块板是三个独立 mocap body")
    check(all(int(mjm.body_dofnum[i]) == 0 and int(mjm.body_jntnum[i]) == 0 for i in body_ids),
          "板体不贡献自由度 (nq/nv 不变)")
    got = [float(e.read_bottom_z(env)[0]) for e in ents]
    want = [bz for _, _, _, bz, _, _ in HOLE_LAYOUT]
    check(bool(np.allclose(got, want, atol=1e-6)),
          f"复位后板底 = HOLE_LAYOUT 默认 (want {want} got {[round(v, 4) for v in got]})")
    env.close()


# [2] 逐环境写入 → geom 世界高度跟随 (这是 geom_pos 域随机化做不到的一步)
def check_write_follows() -> None:
    print("\n[2] 逐环境写板底 + forward, geom 世界高度跟随")
    env = build(False)
    ent = env.scene.entities["hole1"]
    half_t = float(ent.cfg.size[2])
    want = torch.linspace(0.05, 0.10, N, device=env.device)
    ent.write_bottom_z(env, want)
    env.sim.forward()
    got_mocap = np_(ent.read_bottom_z(env))
    check(bool(np.allclose(got_mocap, np_(want), atol=1e-6)), "mocap_pos.z 与写入一致")
    geom_bottom = np_(ent.data.geom_pos_w[:, 0, 2]) - half_t
    err = float(np.abs(geom_bottom - np_(want)).max())
    check(err < 1e-5, f"geom 世界板底跟随写入 (最大误差 {err:.2e})")
    geom_x = np_(ent.data.geom_pos_w[:, 0, 0])
    check(bool(np.allclose(geom_x, ent.cfg.position[0], atol=1e-6)), "只改 z, x 不受影响")
    env.close()


# [3] 碰撞跟着板走: 机器人在两个板底高度下落点差 ≈ 板底差
def check_physics() -> None:
    print("\n[3] 碰撞跟着板走 (从板顶上方落下, 关闭终止条件)")

    # 起点必须高于两块板的板顶 (0.0575 / 0.08), 否则机器人一开始就嵌在板里
    def drop(bottom_z: float, z0: float = 0.13) -> float:
        env = build(True, num_envs=1, disable_terminations=True)
        ent = env.scene.entities["hole1"]
        robot = env.scene["robot"]
        ent.write_bottom_z(env, torch.tensor([bottom_z], device=env.device))
        st = torch.zeros(1, 13, device=env.device)
        st[:, 0] = 0.2
        st[:, 2] = z0
        st[:, 3] = 1.0
        robot.write_root_state_to_sim(st)
        env.sim.forward()
        for _ in range(200):
            env.step(torch.zeros(1, env.action_space.shape[-1], device=env.device))
        out = float(robot.data.root_link_pos_w[0, 2])
        env.close()
        return out

    z_low = drop(0.0475)
    z_high = drop(0.07)
    print(f"     板底 0.0475 → 末 z={z_low:.4f};  板底 0.0700 → 末 z={z_high:.4f}")
    check(z_high > z_low + 1e-3, f"板抬高后落点更高 (Δz={z_high - z_low:+.4f})")
    check(abs((z_high - z_low) - 0.0225) < 0.006,
          f"落点高度差 ≈ 板底差 0.0225 (实测 {z_high - z_low:+.4f})")


# [4] mode="reset" 采样事件: 复位后是采样值, 不是默认值
def check_reset_event() -> None:
    print("\n[4] reset 采样事件 (复位不被 sim.reset 打回默认值)")
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = N
    configure_hole_entities(cfg, enable_collision=False)
    cfg.events["reset_obstacle_height"] = EventTermCfg(func=reset_obstacle_height, mode="reset")
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    ents = [env.scene.entities[k] for k in BOARD_KEYS]
    z = np_(ents[0].read_bottom_z(env))
    print(f"     复位后板底 = {np.round(z, 4).tolist()}")
    lo, hi = OBSTACLE_Z_LOW_START, OBSTACLE_Z_HIGH
    check(bool(((z >= lo - 1e-6) & (z <= hi + 1e-6)).all()), f"板底落在 [{lo}, {hi}]")
    check(not bool(np.allclose(z, 0.0475, atol=1e-6)), "板底不是默认 0.0475 (复位事件确实生效)")
    check(all(bool(np.allclose(np_(e.read_bottom_z(env)), z, atol=1e-6)) for e in ents[1:]),
          "三块板共用同一采样值")
    for i in range(3):
        env.reset()
        z2 = np_(ents[0].read_bottom_z(env))
        check(bool(((z2 >= lo - 1e-6) & (z2 <= hi + 1e-6)).all()),
              f"第 {i + 2} 次复位仍在区间内 (min={z2.min():.4f} max={z2.max():.4f})")
    z3 = np_(ents[0].read_bottom_z(env))
    check(not bool(np.allclose(z3, 0.0475, atol=1e-6)), "循环复位后仍不是默认值")
    env.close()


# [5] 默认配置下 cfg 口径与运行期口径一致
def check_cfg_vs_live() -> None:
    print("\n[5] 默认配置下 cfg 口径与运行期口径一致 (Phase B 基线)")
    env = build(False)
    ents = [env.scene.entities[k] for k in BOARD_KEYS]
    geo = hole_geometry(env)
    live = [float(e.read_bottom_z(env)[0]) for e in ents]
    cfg_z = [g["bottom_z"] for g in geo]
    check(bool(np.allclose(live, cfg_z, atol=1e-6)),
          f"live 板底 == hole_geometry(env) (live {[round(v, 4) for v in live]})")
    env.close()


def main() -> None:
    check_structure()
    check_write_follows()
    check_physics()
    check_reset_event()
    check_cfg_vs_live()
    print("\n" + "=" * 80)
    if fails:
        print(f"结论: 未通过 {len(fails)} 项")
        for f in fails:
            print("  - " + f)
        sys.exit(1)
    print("结论: 全部通过")


if __name__ == "__main__":
    main()
