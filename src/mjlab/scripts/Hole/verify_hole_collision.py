# uv run python -B -m mjlab.scripts.Hole.verify_hole_collision
# 碰撞重构验收: 几何一致性 / 编译掩码 / 物理开关 / Runner 重建 / checkpoint 状态
# 任一项不通过即以非零码退出 (此前只打印结果, 失败也返回 0)

import sys
import numpy as np
import torch
from dataclasses import asdict

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.SQuRo_Hole.mdp.config import STAGE3_END, STEPS_PER_ITER, get_current_stage
from mjlab.tasks.SQuRo_Hole.mdp.hole import (
    VIRTUAL_CLEARANCE_MARGIN,
    configure_hole_entities,
    hole_collision_enabled,
    hole_geometry,
)

TASK = "Mjlab-SQuRo-Hole"
_FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"     [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(name)


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


# 几何: 实体 bottom_z/top_z 与统一查询一致, 虚拟阈值 = 板底 - 余量
def check_geometry() -> None:
    print("[1] 几何一致性 (板底 = position.z, 中心 = 板底 + 半厚)")
    env = build(True)
    print(f"     {'板':<7}{'x 区间':>18}{'板底':>9}{'板顶':>9}{'虚拟阈值':>10}{'余量':>8}")
    for g in hole_geometry():
        e = env.scene.entities[g["name"].lower()]
        dz = g["bottom_z"] - g["virtual_z_threshold"]
        rng = f"[{g['x_min']:.3f},{g['x_max']:.3f}]"
        print(f"     {g['name']:<7}{rng:>18}{g['bottom_z']:>9.4f}{g['top_z']:>9.4f}"
              f"{g['virtual_z_threshold']:>10.4f}{dz * 1000:>7.1f}m")
        check(f"{g['name']} 实体与查询接口一致", abs(e.bottom_z - g["bottom_z"]) < 1e-12
              and abs(e.top_z - g["top_z"]) < 1e-12)
        check(f"{g['name']} 虚拟余量 = {VIRTUAL_CLEARANCE_MARGIN}", abs(dz - VIRTUAL_CLEARANCE_MARGIN) < 1e-12)
    env.close()


# 编译掩码: 开关决定 contype/conaffinity, 且在编译后的模型里真实生效
def check_mask() -> None:
    print("\n[2] 编译后碰撞掩码")
    for collision in (False, True):
        env = build(collision)
        m = env.sim.mj_model
        vals = [(int(m.geom_contype[i]), int(m.geom_conaffinity[i]))
                for i in range(m.ngeom) if "Hole" in (m.geom(i).name or "")]
        want = 1 if collision else 0
        check(f"碰撞={'开' if collision else '关'} 三块板掩码 = {vals}",
              len(vals) == 3 and all(v == (want, want) for v in vals))
        env.close()


# 物理开关: 同一初态下, 开碰撞时机器人被板挡住
def check_physics() -> None:
    print("\n[3] 物理开关 (从板顶上方落下, 相同初态, 关闭终止条件)")

    def drop(collision: bool, z0: float):
        env = build(collision, disable_terminations=True)
        robot = env.scene["robot"]
        st = torch.zeros(1, 13, device=env.device)
        st[:, 0] = 0.2
        st[:, 2] = z0
        st[:, 3] = 1.0
        robot.write_root_state_to_sim(st)
        env.sim.forward()
        for _ in range(200):
            env.step(torch.zeros(1, env.action_space.shape[-1], device=env.device))
        out = (float(robot.data.root_link_pos_w[0, 2]), float(robot.data.root_link_pos_w[0, 0]))
        env.close()
        return out

    z_off, x_off = drop(False, 0.09)
    z_on, x_on = drop(True, 0.09)
    print(f"     碰撞关: 末 z={z_off:.4f} x={x_off:.4f}")
    print(f"     碰撞开: 末 z={z_on:.4f} x={x_on:.4f}")
    check("开碰撞后落点明显更高 (被板挡住)", z_on > z_off + 1e-3, f"Δz={z_on - z_off:+.4f}")
    check("两者确实不同 (碰撞生效)", abs(z_on - z_off) > 1e-4 or abs(x_on - x_off) > 1e-4)


# Runner 重建: 跨 3k 只重建一次, 计数器/环境数/观测保留, 不反复重建
def check_rebuild() -> None:
    print("\n[4] Runner 重建 (跨 3k 边界)")
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = 8
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    vec = RslRlVecEnvWrapper(env, None)
    runner = load_runner_cls(TASK)(vec, asdict(load_rl_cfg(TASK)), "tmp_verify_log", device="cpu")

    env.common_step_counter = STAGE3_END - 1
    check("iter 2999 不需重建", not runner._hole_change_needed())
    env.common_step_counter = STAGE3_END
    check("iter 3000 需重建", runner._hole_change_needed())
    n_before = runner.env.unwrapped.scene.num_envs
    obs = runner._apply_hole_rebuild(refresh_obs=True)
    check("重建后碰撞 = 开", hole_collision_enabled(runner.env.unwrapped))
    check("num_envs 保持", runner.env.unwrapped.scene.num_envs == n_before,
          f"{runner.env.unwrapped.scene.num_envs}")
    check("common_step_counter 保持",
          int(runner.env.unwrapped.common_step_counter) == STAGE3_END)
    check("观测形状保持", tuple(obs.shape) == (n_before,), f"{tuple(obs.shape)}")
    check("不反复重建", not runner._hole_change_needed())
    try:
        runner.env.close()
    except Exception:
        pass


# 重建保留自定义几何与接触参数 (不能被默认表覆盖)
def check_rebuild_preserves_custom() -> None:
    print("\n[5] 重建保留自定义板位与接触参数")
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = 1
    ent = cfg.scene.entities["hole1"]
    ent.position = (0.35, 0.0, 0.0675)
    ent.solref = (0.03, 1.0)
    configure_hole_entities(cfg, enable_collision=True)
    ent2 = cfg.scene.entities["hole1"]
    check("位置保留", tuple(ent2.position) == (0.35, 0.0, 0.0675), f"{tuple(ent2.position)}")
    check("solref 保留", ent2.solref is not None and tuple(ent2.solref) == (0.03, 1.0),
          f"{ent2.solref}")
    check("碰撞掩码被设为开", int(ent2.contype) == 1 and int(ent2.conaffinity) == 1)


# checkpoint 状态: 真实保存 / 读取 (不用示例字典)
def check_state_roundtrip() -> None:
    print("\n[6] checkpoint hole_state 真实保存与读取")
    import os
    from mjlab.tasks.SQuRo_Hole.rl.runner import HOLE_STATE_VERSION, read_hole_state, read_env_step
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    vec = RslRlVecEnvWrapper(env, None)
    runner = load_runner_cls(TASK)(vec, asdict(load_rl_cfg(TASK)), "tmp_verify_log", device="cpu")
    env.common_step_counter = STAGE3_END
    runner._apply_hole_rebuild(refresh_obs=False)
    path = "tmp_verify_log/hole_state_probe.pt"
    os.makedirs("tmp_verify_log", exist_ok=True)
    runner.save(path)
    saved = read_hole_state(path)
    check("hole_state 已写入", bool(saved), f"version={saved.get('version')}")
    check("版本号正确", saved.get("version") == HOLE_STATE_VERSION)
    check("stage 正确", saved.get("stage") == get_current_stage(STAGE3_END))
    check("collision 记录为实际编译值 (开)", saved.get("collision") is True)
    layout = saved.get("layout") or []
    check("layout 三块板", len(layout) == 3, f"{len(layout)}")
    if layout:
        check("layout 记录实际板底 (0.0475)",
              abs(float(layout[0]["position"][2]) - 0.0475) < 1e-9,
              f"{layout[0]['position']}")
    check("env_state 步数可读", read_env_step(path) == STAGE3_END, f"{read_env_step(path)}")
    try:
        runner.env.close()
    except Exception:
        pass
    # 清理探针目录
    import shutil
    shutil.rmtree("tmp_verify_log", ignore_errors=True)


# 虚拟约束必须读实际场景几何 (自定义板位下不能仍按默认表算)
def check_reward_geometry() -> None:
    print("\n[7] 虚拟约束读实际场景几何 (自定义板位)")
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = 1
    ent = cfg.scene.entities["hole1"]
    ent.position = (0.35, 0.0, 0.0675)
    ent.solref = (0.03, 1.0)
    configure_hole_entities(cfg, enable_collision=False)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    g0 = hole_geometry(env)[0]
    check("x 区间跟随自定义位置", abs(g0["x_min"] - 0.335) < 1e-9 and abs(g0["x_max"] - 0.365) < 1e-9,
          f"[{g0['x_min']:.3f},{g0['x_max']:.3f}]")
    check("板底跟随自定义位置", abs(g0["bottom_z"] - 0.0675) < 1e-9, f"{g0['bottom_z']:.4f}")
    check("虚拟阈值 = 板底 - 2.5mm", abs(g0["virtual_z_threshold"] - 0.0650) < 1e-9,
          f"{g0['virtual_z_threshold']:.4f}")
    env.close()


# checkpoint 保存的几何/接触参数能在回放侧复现
def check_layout_roundtrip() -> None:
    print("\n[8] checkpoint 几何/接触参数 保存 → 回放复现")
    import os
    import shutil
    from mjlab.tasks.SQuRo_Hole.mdp.hole import apply_saved_layout
    probe_dir = "tmp_hole_geom_probe"
    os.makedirs(probe_dir, exist_ok=True)
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = 2
    ent = cfg.scene.entities["hole1"]
    ent.position = (0.35, 0.0, 0.0675)
    ent.solref = (0.03, 1.0)
    configure_hole_entities(cfg, enable_collision=True)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    vec = RslRlVecEnvWrapper(env, None)
    runner = load_runner_cls(TASK)(vec, asdict(load_rl_cfg(TASK)), probe_dir, device="cpu")
    path = f"{probe_dir}/model_0.pt"
    runner.save(path)
    from mjlab.tasks.SQuRo_Hole.rl.runner import read_hole_state
    saved = read_hole_state(path)
    d0 = (saved.get("layout") or [{}])[0]
    check("保存自定义位置", d0.get("position") == [0.35, 0.0, 0.0675], f"{d0.get('position')}")
    check("保存 solref", d0.get("solref") == [0.03, 1.0], f"{d0.get('solref')}")
    # 干净配置 + apply_saved_layout → 复现
    cfg2 = load_env_cfg(TASK, play=True)
    n = apply_saved_layout(cfg2, saved.get("layout") or [])
    e2 = cfg2.scene.entities["hole1"]
    check("回放复现位置", tuple(e2.position) == (0.35, 0.0, 0.0675), f"{tuple(e2.position)}")
    check("回放复现 solref", e2.solref is not None and tuple(e2.solref) == (0.03, 1.0),
          f"{e2.solref}")
    check("三块板都被应用", n == 3, f"{n}")
    try:
        runner.env.close()
    except Exception:
        pass
    shutil.rmtree(probe_dir, ignore_errors=True)


def main() -> None:
    check_geometry()
    check_mask()
    check_physics()
    check_rebuild()
    check_rebuild_preserves_custom()
    check_state_roundtrip()
    check_reward_geometry()
    check_layout_roundtrip()
    print()
    if _FAILURES:
        print(f"[结论] 失败 {len(_FAILURES)} 项: {_FAILURES}")
        sys.exit(1)
    print("[结论] 全部通过: 几何单一来源且与实际一致; 碰撞掩码按开关编译生效; "
          "开碰撞时机器人被板挡住; Runner 在 3k 边界重建一次且保留计数器; "
          "自定义板位/接触参数不被重建覆盖且被虚拟约束采用; hole_state 可真实保存与回放复现")


if __name__ == "__main__":
    main()
