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
    CURRICULUM_LEVELS,
    STAGE1_3_ITER,
    STAGE2_2_ITER,
    WALL_X_NEG_END,
    WALL_X_NEG_LEVELS,
    WALL_X_NEG_START,
    WALL_X_NEG_STEP,
    WALL_X_POS,
    _STEPS_PER_ITER,
    get_level_for_wall_x_neg,
    get_training_phase,
    get_wall_positions_for_level,
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


# 建一个带指定墙位与编译期碰撞开关的环境 (不装 startup 事件, 避免干扰)
def make_env(cfg: ProbeCfg, walls: tuple[float, float], collision: bool):
    env_cfg = load_backup_env_cfg()
    env_cfg.scene.num_envs = 1
    env_cfg.events.pop("init_restricted_space", None)
    mdp_entity.configure_restricted_space(env_cfg, wall_x_neg=walls[0], wall_x_pos=walls[1],
                                          enable_collision=collision)
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
    print("=" * 94)
    print(f"[1] 墙位课程档位表 (+X 固定 {WALL_X_POS:.2f}, −X {WALL_X_NEG_START:.2f} → "
          f"{WALL_X_NEG_END:.2f}, 步长 {WALL_X_NEG_STEP:.2f}, 共 {CURRICULUM_LEVELS} 档)")
    print(f"  −X 档位表: {list(WALL_X_NEG_LEVELS)}")
    for lv in range(CURRICULUM_LEVELS):
        neg, pos = get_wall_positions_for_level(lv)
        print(f"  档位 {lv}: x_neg={neg:+.4f} x_pos={pos:+.4f} "
              f"净宽 {pos - neg - 2 * WALL_HALF_THICKNESS:.4f} m")
    # 末档必须精确等于下界, 否则课程永远到不了目标墙位
    assert abs(WALL_X_NEG_LEVELS[-1] - WALL_X_NEG_END) < 1e-9, "档位表末档必须等于下界"
    assert abs(WALL_X_NEG_LEVELS[0] - WALL_X_NEG_START) < 1e-9
    # 必须严格单调收紧 (朝原点移动), 且每一档都比上一档窄
    for a, b in zip(WALL_X_NEG_LEVELS, WALL_X_NEG_LEVELS[1:]):
        assert b > a + 1e-12, "−X 墙位必须逐档朝原点收紧"
    for lv in range(CURRICULUM_LEVELS - 1):
        w0 = get_wall_positions_for_level(lv)
        w1 = get_wall_positions_for_level(lv + 1)
        assert (w1[1] - w1[0]) < (w0[1] - w0[0]), "净宽必须逐档变小"
        assert w1[1] == w0[1] == WALL_X_POS, "+X 墙位必须全程固定"
    # 越界要 clamp 到端点 (课程停在末档时仍要能算出墙位)
    assert get_wall_positions_for_level(-3) == get_wall_positions_for_level(0)
    assert get_wall_positions_for_level(999) == get_wall_positions_for_level(CURRICULUM_LEVELS - 1)
    # 由墙位反查档位必须自洽
    for lv in range(CURRICULUM_LEVELS):
        assert get_level_for_wall_x_neg(WALL_X_NEG_LEVELS[lv]) == lv
    assert STAGE2_2_ITER == 6000
    assert get_training_phase((STAGE1_3_ITER - 1) * _STEPS_PER_ITER) == 0
    assert get_training_phase(STAGE1_3_ITER * _STEPS_PER_ITER) == 1
    print("  档位/阶段断言通过 (末档可达下界, 逐档收紧, +X 固定, 越界 clamp)")

    print("\n[2] 阶段一的初始碰撞状态必须与阶段语义一致 (STAGE1 = 关碰撞, 第一帧起)")
    env_cfg = load_backup_env_cfg()
    ent_cfg = dict(env_cfg.scene.entities)["restricted_space"]
    print(f"  默认 env_cfg: x_neg={ent_cfg.wall_x_neg:+.4f} x_pos={ent_cfg.wall_x_pos:+.4f} "
          f"contype={ent_cfg.contype} conaffinity={ent_cfg.conaffinity} "
          f"fixed_width={ent_cfg.fixed_width}")
    assert ent_cfg.contype == 0 and ent_cfg.conaffinity == 0, \
        "STAGE1 的默认配置必须编译成不开碰撞 (否则首轮采样会带着碰撞跑)"
    neg0, pos0 = get_wall_positions_for_level(0)
    assert abs(ent_cfg.wall_x_neg - neg0) < 1e-12 and abs(ent_cfg.wall_x_pos - pos0) < 1e-12, \
        "默认配置必须编译成课程第 0 档墙位"
    print("  默认配置断言通过")

    print("\n[3] 墙体几何与净宽口径 (第 0 档 / 末档)")
    for lv in (0, CURRICULUM_LEVELS - 1):
        walls = get_wall_positions_for_level(lv)
        env = make_env(cfg, walls, collision=False)
        ent = env.scene.entities["restricted_space"]
        xs = [float(env.sim.mj_data.geom_xpos[g][0]) for g in ent.wall_geom_ids]
        print(f"  档位 {lv}: 墙世界 x={[round(x, 4) for x in xs]}  "
              f"实际内侧净宽 {ent.clear_width:.4f} m "
              f"(= 间距 {walls[1] - walls[0]:.4f} - 2×{WALL_HALF_THICKNESS})")
        assert abs(xs[0] - walls[0]) < 1e-4, "−X 墙中心未落在 wall_x_neg"
        assert abs(xs[1] - walls[1]) < 1e-4, "+X 墙中心未落在 wall_x_pos"
        assert abs(ent.clear_width - (walls[1] - walls[0] - 2 * WALL_HALF_THICKNESS)) < 1e-9
        # 墙位观测必须等于**实际编译值** (策略靠它区分各档; 各档的重置姿态完全相同)
        cmd_term = env.command_manager.get_term("backup_cmd")
        cmd0 = cmd_term.command[0]
        assert cmd_term.command.shape[-1] == 9, "命令张量必须是 9 维 (末两维为墙位观测)"
        # 命令张量是 float32, 容差按 float32 给
        assert abs(float(cmd0[7]) - walls[0]) < 1e-6, "观测里的 −X 墙位与实际编译值不一致"
        assert abs(float(cmd0[8]) - walls[1]) < 1e-6, "观测里的 +X 墙位与实际编译值不一致"
        print(f"    命令张量 {cmd_term.command.shape[-1]} 维, "
              f"墙位观测 ({float(cmd0[7]):+.4f}, {float(cmd0[8]):+.4f}) 与实际编译值一致")

    print(f"\n[4] 碰撞是否真的生效 (机器人搬到墙中心 x=±{cfg.test_width/2:.3f}, 跑 {cfg.steps} 步)")
    # 编译期开碰撞 + 几何重叠 => 必须有墙接触
    test_walls = (-0.5 * cfg.test_width, 0.5 * cfg.test_width)
    env_on = make_env(cfg, test_walls, collision=True)
    ent_on = env_on.scene.entities["restricted_space"]
    hits_on, x_on = run_probe(env_on, ent_on, cfg.test_width / 2, cfg.steps)
    print(f"  编译期碰撞=开: 墙接触 {hits_on} 个, x 末值 {x_on:+.4f}")
    assert hits_on > 0, "编译期开碰撞时墙必须产生接触, 实测为 0"

    # 编译期关碰撞 => 同一位置必须无墙接触, 机器人不被挡
    env_off = make_env(cfg, test_walls, collision=False)
    ent_off = env_off.scene.entities["restricted_space"]
    hits_off, x_off = run_probe(env_off, ent_off, cfg.test_width / 2, cfg.steps)
    print(f"  编译期碰撞=关: 墙接触 {hits_off} 个, x 末值 {x_off:+.4f}")
    assert hits_off == 0, "编译期关碰撞时不应有墙接触"

    print("\n  注: 运行期改 contype/geom_pos 均无效 (mjwarp 在 put_model 时固化),")
    print("      所以碰撞开关与 a 都只能靠重建环境切换, 见 mdp/entity.py 的说明。")
    print("\n全部断言通过")


if __name__ == "__main__":
    main()
