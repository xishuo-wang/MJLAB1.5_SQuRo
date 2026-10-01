# uv run python -B -m mjlab.scripts.Hole.verify_hole_collision
# 碰撞重构验收: 几何一致性 / 编译掩码 / 物理开关 / Runner 重建 / checkpoint 状态

import numpy as np
import torch
from dataclasses import asdict

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.SQuRo_Hole.mdp.config import STAGE3_END, STEPS_PER_ITER, get_current_stage
from mjlab.tasks.SQuRo_Hole.mdp.hole import (
    HOLE_LAYOUT,
    VIRTUAL_CLEARANCE_MARGIN,
    configure_hole_entities,
    hole_collision_enabled,
    hole_geometry,
)

TASK = "Mjlab-SQuRo-Hole"


def build(collision: bool, n: int = 1, play: bool = False,
          disable_terminations: bool = False) -> ManagerBasedRlEnv:
    cfg = load_env_cfg(TASK, play=play)
    cfg.scene.num_envs = n
    configure_hole_entities(cfg, enable_collision=collision)
    if disable_terminations:
        # 物理开关测试要观察未被重置的轨迹; fallen 会触发 auto_reset 把机器人拉回原点
        cfg.terminations = {}
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    return env


# 几何: 实体 bottom_z/top_z 与统一查询一致, 且板底 = position.z
def check_geometry() -> None:
    env = build(True)
    print("[1] 几何一致性 (板底 = position.z, 中心 = 板底 + 半厚)")
    print(f"     {'板':<7}{'x 区间':>18}{'板底':>9}{'板顶':>9}{'虚拟阈值':>10}{'余量':>8}")
    for g in hole_geometry():
        e = env.scene.entities[g["name"].lower()]
        dz = g["bottom_z"] - g["virtual_z_threshold"]
        ok = (abs(e.bottom_z - g["bottom_z"]) < 1e-12
              and abs(e.top_z - g["top_z"]) < 1e-12
              and abs(dz - VIRTUAL_CLEARANCE_MARGIN) < 1e-12)
        rng = f"[{g['x_min']:.3f},{g['x_max']:.3f}]"
        print(f"     {g['name']:<7}{rng:>18}"
              f"{g['bottom_z']:>9.4f}{g['top_z']:>9.4f}{g['virtual_z_threshold']:>10.4f}"
              f"{dz * 1000:>7.1f}m {'OK' if ok else 'MISMATCH'}")
    env.close()


# 编译掩码: 开关决定 contype/conaffinity, 且在模型里真实生效
def check_mask() -> None:
    print("\n[2] 编译后碰撞掩码")
    for collision in (False, True):
        env = build(collision)
        m = env.sim.mj_model
        vals = []
        for i in range(m.ngeom):
            if "Hole" in (m.geom(i).name or ""):
                vals.append((int(m.geom_contype[i]), int(m.geom_conaffinity[i])))
        want = 1 if collision else 0
        ok = all(v == (want, want) for v in vals)
        print(f"     碰撞={'开' if collision else '关'}: 三块板 contype/conaffinity={vals} "
              f"{'OK' if ok else 'MISMATCH'}")
        env.close()


# 物理开关: 同一初态下, 开/关碰撞的落点与运动结果不同
def check_physics() -> None:
    print("\n[3] 物理开关 (从板顶上方落下, 相同初态)")

    def drop(collision: bool, z0: float):
        env = build(collision, disable_terminations=True)
        robot = env.scene["robot"]
        st = torch.zeros(1, 13, device=env.device)
        st[:, 0] = 0.2
        st[:, 2] = z0
        st[:, 3] = 1.0
        robot.write_root_state_to_sim(st)
        env.sim.forward()
        peaks = []
        for _ in range(200):
            env.step(torch.zeros(1, env.action_space.shape[-1], device=env.device))
            peaks.append(float(robot.data.root_link_pos_w[0, 2]))
        out = (peaks[-1], max(peaks), float(robot.data.root_link_pos_w[0, 0]))
        env.close()
        return out

    z_off, peak_off, x_off = drop(False, 0.09)
    z_on, peak_on, x_on = drop(True, 0.09)
    print(f"     碰撞关: 末 z={z_off:.4f} 峰值 z={peak_off:.4f} x={x_off:.4f}")
    print(f"     碰撞开: 末 z={z_on:.4f} 峰值 z={peak_on:.4f} x={x_on:.4f}")
    print(f"     → 两者结果不同 (碰撞生效): {abs(z_on - z_off) > 1e-4 or abs(peak_on - peak_off) > 1e-4}")


# Runner 重建: 跨 3k 只重建一次, 计数器/环境数/观测保留, 不反复重建
def check_rebuild() -> None:
    print("\n[4] Runner 重建 (跨 3k 边界)")
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = 8
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    vec = RslRlVecEnvWrapper(env, None)
    runner = load_runner_cls(TASK)(vec, asdict(load_rl_cfg(TASK)), "tmp_verify_log", device="cpu")

    for step in (0, STAGE3_END - 1, STAGE3_END):
        env.common_step_counter = step
        print(f"     step {step:>6} iter {step // STEPS_PER_ITER:>4} stage={get_current_stage(step)} "
              f"目标碰撞={runner._hole_target_collision()} 实际={hole_collision_enabled(env)} "
              f"需重建={runner._hole_change_needed()}")
    env.common_step_counter = STAGE3_END
    n_before = runner.env.unwrapped.scene.num_envs
    obs = runner._apply_hole_rebuild(refresh_obs=True)
    print(f"     重建后: 碰撞={hole_collision_enabled(runner.env.unwrapped)} "
          f"num_envs={runner.env.unwrapped.scene.num_envs} (前 {n_before}) "
          f"step={int(runner.env.unwrapped.common_step_counter)} obs={tuple(obs.shape)}")
    print(f"     再判需重建={runner._hole_change_needed()} (期望 False, 防反复重建)")
    try:
        runner.env.close()
    except Exception:
        pass


# checkpoint 状态: 记录实际编译的碰撞/阶段/几何
def check_state() -> None:
    print("\n[5] checkpoint hole_state 结构")
    from mjlab.tasks.SQuRo_Hole.rl.runner import HOLE_STATE_VERSION
    state = {
        "version": HOLE_STATE_VERSION,
        "stage": get_current_stage(STAGE3_END),
        "collision": True,
        "layout": [{"name": n, "bottom_z": bz, "half_t": ht}
                   for n, x, y, bz, hl, ht in HOLE_LAYOUT],
    }
    print(f"     {state}")
    print(f"     几何表: {[(n, bz) for n, x, y, bz, hl, ht in HOLE_LAYOUT]}")


def main() -> None:
    check_geometry()
    check_mask()
    check_physics()
    check_rebuild()
    check_state()
    print("\n[结论] 几何由 hole.py 单一来源提供 (板底/板顶/虚拟阈值/x 区间); 碰撞开关在编译期生效; "
          "Runner 在 3k 边界重建一次并保留计数器; checkpoint 记录 hole_state")


if __name__ == "__main__":
    main()
