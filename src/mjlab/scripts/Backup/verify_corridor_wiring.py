# 受限空间接线自检 — 课程取值 / 墙体几何与净宽 / 碰撞是否真的生效 (只读, 带失败断言)
# 用法: uv run python -B -m mjlab.scripts.Backup.verify_corridor_wiring
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.SQuRo_Backup.mdp import entity as mdp_entity
from mjlab.tasks.SQuRo_Backup.mdp.curriculums import (
    CORRIDOR_WIDTH_MIN,
    CORRIDOR_WIDTH_START,
    STAGE1_3_ITER,
    STAGE2_1_ITER,
    STAGE2_2_ITER,
    _STEPS_PER_ITER,
    get_curriculum_corridor_width,
    get_training_phase,
)
from mjlab.tasks.SQuRo_Backup.mdp.entity import WALL_HALF_THICKNESS


@dataclass
class ProbeCfg:
    device: str = "cuda:0"
    # 物理验证用的间距: 要小到墙能碰到机器人 (机器人 |x|max≈0.14)
    test_width: float = 0.24
    steps: int = 40


# 绕开 mjlab.tasks 的包扫描 (其他任务重构期间扫描会失败), 直接注册本任务
def load_backup_env_cfg():
    importlib.import_module("mjlab.tasks.SQuRo_Backup.config")
    from mjlab.tasks.registry import load_env_cfg
    return load_env_cfg("Mjlab-SQuRo-Backup")


# 建一个带指定 a 与编译期碰撞开关的环境 (不装 startup 事件, 避免干扰)
def make_env(cfg: ProbeCfg, width: float, collision: bool):
    env_cfg = load_backup_env_cfg()
    env_cfg.scene.num_envs = 1
    env_cfg.events.pop("init_restricted_space", None)
    mdp_entity.configure_restricted_space(env_cfg, width, enable_collision=collision)
    return ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)


# 把机器人放到指定 root x 并跑若干步, 返回 (墙接触对总数, x 末值)
def run_probe(env, ent, start_x: float, steps: int) -> tuple[int, float]:
    ids = ent.wall_geom_ids
    asset = env.scene.entities["robot"]
    dtype = asset.data.root_link_pos_w.dtype
    root = torch.zeros(1, 13, device=env.device, dtype=dtype)
    root[0, 0] = start_x
    root[0, 2] = 0.06
    root[0, 3] = 1.0
    asset.write_root_state_to_sim(root)
    env.sim.forward()
    hits = 0
    with torch.no_grad():
        for _ in range(steps):
            env.step(torch.zeros(1, 14, device=env.device))
            n = int(env.sim.wp_data.nacon.numpy()[0])
            pairs = env.sim.wp_data.contact.geom.numpy()[:n].tolist()
            hits += len([p for p in pairs if p[0] in ids or p[1] in ids])
    return hits, float(asset.data.root_link_pos_w[0, 0])


def main() -> None:
    cfg = tyro.cli(ProbeCfg)
    print("=" * 92)
    print(f"[1] 课程纯函数核对 (STAGE1 固定 {CORRIDOR_WIDTH_START:.2f} / "
          f"STAGE2_1 {STAGE1_3_ITER}~{STAGE2_1_ITER} 收缩 / STAGE2_2 保持 {CORRIDOR_WIDTH_MIN:.2f})")
    for it in (0, 2999, 3000, 4000, 5000, 6000, 7000):
        a = get_curriculum_corridor_width(it * _STEPS_PER_ITER)
        print(f"  iter {it:>5}  phase {get_training_phase(it * _STEPS_PER_ITER)}  a={a:.4f}")
    assert get_curriculum_corridor_width(0) == CORRIDOR_WIDTH_START
    assert get_curriculum_corridor_width((STAGE1_3_ITER - 1) * _STEPS_PER_ITER) == CORRIDOR_WIDTH_START
    assert abs(get_curriculum_corridor_width(STAGE2_1_ITER * _STEPS_PER_ITER)
               - CORRIDOR_WIDTH_MIN) < 1e-9
    assert get_curriculum_corridor_width(9999 * _STEPS_PER_ITER) == CORRIDOR_WIDTH_MIN
    assert get_training_phase((STAGE1_3_ITER - 1) * _STEPS_PER_ITER) == 0
    assert get_training_phase(STAGE1_3_ITER * _STEPS_PER_ITER) == 1
    # 收缩段必须单调不增
    prev = None
    for it in range(STAGE1_3_ITER, STAGE2_1_ITER + 1, 100):
        a = get_curriculum_corridor_width(it * _STEPS_PER_ITER)
        if prev is not None:
            assert a <= prev + 1e-12, "收缩段必须单调不增"
        prev = a
    assert STAGE2_2_ITER == 6000
    print("  纯函数断言通过")

    print("\n[2] 墙体几何与净宽口径")
    for width in (0.40, 0.20):
        env = make_env(cfg, width, collision=False)
        ent = env.scene.entities["restricted_space"]
        xs = [float(env.sim.mj_data.geom_xpos[g][0]) for g in ent.wall_geom_ids]
        print(f"  a={width:.2f}: 墙世界 x={[round(x, 4) for x in xs]}  "
              f"实际内侧净宽 {ent.clear_width:.4f} m (= a - 2×{WALL_HALF_THICKNESS})")
        assert abs(abs(xs[1]) - width / 2) < 1e-4, "墙中心未落在 ±a/2"
        assert abs(ent.clear_width - (width - 2 * WALL_HALF_THICKNESS)) < 1e-9

    print(f"\n[3] 碰撞是否真的生效 (机器人搬到墙中心 x=±{cfg.test_width/2:.3f}, 跑 {cfg.steps} 步)")
    # 编译期开碰撞 + 几何重叠 => 必须有墙接触
    env_on = make_env(cfg, cfg.test_width, collision=True)
    ent_on = env_on.scene.entities["restricted_space"]
    hits_on, x_on = run_probe(env_on, ent_on, cfg.test_width / 2, cfg.steps)
    print(f"  编译期碰撞=开: 墙接触 {hits_on} 个, x 末值 {x_on:+.4f}")
    assert hits_on > 0, "编译期开碰撞时墙必须产生接触, 实测为 0"

    # 编译期关碰撞 => 同一位置必须无墙接触, 机器人不被挡
    env_off = make_env(cfg, cfg.test_width, collision=False)
    ent_off = env_off.scene.entities["restricted_space"]
    hits_off, x_off = run_probe(env_off, ent_off, cfg.test_width / 2, cfg.steps)
    print(f"  编译期碰撞=关: 墙接触 {hits_off} 个, x 末值 {x_off:+.4f}")
    assert hits_off == 0, "编译期关碰撞时不应有墙接触"

    print("\n  注: 运行期改 contype/geom_pos 均无效 (mjwarp 在 put_model 时固化),")
    print("      所以碰撞开关与 a 都只能靠重建环境切换, 见 mdp/entity.py 的说明。")
    print("\n全部断言通过")


if __name__ == "__main__":
    main()
