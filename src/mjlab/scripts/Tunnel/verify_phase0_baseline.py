# uv run python -B -m mjlab.scripts.Tunnel.verify_phase0_baseline
# Phase0 基线验证: 参考表结构 / 命令采样配对约束 / 速度公式 / 各模式实测躯干高度

import numpy as np
import torch
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.SQuRo_Tunnel.mdp.indices import _MODEL_INDICES, resolve_model_indices
from mjlab.tasks.SQuRo_Tunnel.mdp.curriculums import get_training_phase, _STEPS_PER_ITER
from mjlab.tasks.SQuRo_Tunnel.mdp.reference import (
    BASE_HEIGHT,
    HEIGHT_LIST,
    HEIGHT_LOW_THRESHOLD,
    NUM_HEIGHTS,
    NUM_MODES,
    _TABLE_RESOLUTION,
)
from mjlab.tasks.SQuRo_Tunnel.mdp.command import PHASE0_V_BASE
import mjlab.tasks.SQuRo_Tunnel.mdp.reference as ref_mod


MODE_TAGS = {0: "都高", 1: "前低", 2: "后低"}


# 参考表结构与关键行: 验证模式/高度档确实改变了参考姿态
def check_table() -> None:
    ref_mod._init_tables("cpu")
    table = ref_mod._pos_table
    assert table is not None
    print(f"[1] 参考表形状 {tuple(table.shape)} "
          f"(模式 {NUM_MODES} × 高度档 {NUM_HEIGHTS} × 相位 {_TABLE_RESOLUTION} × 14 关节)")
    print(f"{'模式':>6}{'高度(mm)':>10}{'FL肩幅度':>12}{'HL髋幅度':>12}{'脊柱最大幅':>12}")
    for mode in range(NUM_MODES):
        for h_idx, h in enumerate(HEIGHT_LIST):
            row = table[mode, h_idx]
            fl = float(row[:, 4].max() - row[:, 4].min())
            hl = float(row[:, 10].max() - row[:, 10].min())
            spine = float(row[:, [0, 1, 8, 9]].abs().max())
            print(f"{MODE_TAGS[mode]:>6}{h * 1000:>10.0f}{fl:>12.4f}{hl:>12.4f}{spine:>12.4f}")


# 命令采样: 配对约束 (无双低) 与速度公式逐项核对
def check_commands(num_envs: int = 512) -> None:
    cfg = load_env_cfg("Mjlab-SQuRo-Tunnel", play=True)
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    cmd = env.command_manager.get_command("tunnel_cmd")
    h_f, h_h, vel = cmd[:, 1], cmd[:, 2], cmd[:, 0]
    pairs = sorted({(round(float(a), 4), round(float(b), 4)) for a, b in zip(h_f, h_h)})
    both_low = int(((h_f < HEIGHT_LOW_THRESHOLD) & (h_h < HEIGHT_LOW_THRESHOLD)).sum())
    print(f"\n[2] 命令采样 {num_envs} 环境, 高度组合 {len(pairs)} 种, 双低组合 {both_low} 个 (应为 0)")
    all_ok = True
    print(f"{'(h_f,h_h)':>14}{'n低':>5}{'运动侧高度':>12}{'实测速度':>10}{'公式':>10}{'一致':>6}")
    for a, b in pairs:
        m = (torch.round(h_f, decimals=4) == a) & (torch.round(h_h, decimals=4) == b)
        n_low = float(a < HEIGHT_LOW_THRESHOLD) + float(b < HEIGHT_LOW_THRESHOLD)
        moving = max(a, b)
        expect = PHASE0_V_BASE * moving / BASE_HEIGHT * (2 - n_low)
        got = float(vel[m][0])
        ok = abs(got - expect) < 1e-6
        all_ok = all_ok and ok
        print(f"{f'({a*1000:.0f},{b*1000:.0f})':>14}{n_low:>5.0f}{moving * 1000:>12.0f}"
              f"{got:>10.4f}{expect:>10.4f}{'✓' if ok else '✗':>6}")
    print(f"    速度公式全部一致: {'是' if all_ok else '否'}")
    env.close()


# 实测: 按参考表逐相位驱动, 量各模式/高度档的躯干高度范围
def check_heights() -> None:
    cfg = load_env_cfg("Mjlab-SQuRo-Tunnel", play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    robot = env.scene["robot"]
    resolve_model_indices(robot)
    default = robot.data.default_joint_pos[0, list(_MODEL_INDICES.joint_ids)].clone()
    ref_mod._init_tables("cpu")
    table = ref_mod._pos_table
    assert table is not None

    print(f"\n[3] 按参考表驱动完整周期, 实测躯干重心高度 (顶部 = 重心 + 24mm)")
    print(f"{'模式':>6}{'高度档':>8}{'前均值':>9}{'后均值':>9}{'前最低':>9}{'后最低':>9}{'顶部(前)':>10}")
    for mode in range(NUM_MODES):
        for h_idx, h in enumerate(HEIGHT_LIST):
            env.reset()
            f_hist, h_hist = [], []
            for ph in range(_TABLE_RESOLUTION):
                row = table[mode, h_idx, ph]
                target = default.clone()
                for jid in (4, 5, 6, 7, 10, 11, 12, 13):
                    target[jid] = row[jid]
                act = ((target - default) / 0.3).unsqueeze(0)
                for _ in range(2):
                    env.step(act)
                pos = robot.data.body_link_pos_w
                f_hist.append(float(pos[0, _MODEL_INDICES.f_body_id, 2]))
                h_hist.append(float(pos[0, _MODEL_INDICES.h_body_id, 2]))
            print(f"{MODE_TAGS[mode]:>6}{h * 1000:>8.0f}{np.mean(f_hist) * 1000:>9.1f}"
                  f"{np.mean(h_hist) * 1000:>9.1f}{np.min(f_hist) * 1000:>9.1f}"
                  f"{np.min(h_hist) * 1000:>9.1f}{(np.mean(f_hist) + 0.024) * 1000:>10.1f}")
    env.close()


# 阶段边界: 一阶段 0~3k, 二阶段 3k~6k
def check_phase_boundary() -> None:
    lo = get_training_phase((3000 - 1) * _STEPS_PER_ITER)
    hi = get_training_phase(3000 * _STEPS_PER_ITER)
    print(f"\n[4] 阶段边界: iter 2999 → phase {lo}, iter 3000 → phase {hi} "
          f"({'正确' if lo == 0 and hi == 1 else '错误'})")


def main() -> None:
    check_table()
    check_commands()
    check_heights()
    check_phase_boundary()
    print("\n[结论] 若 [2] 速度全一致、双低为 0, [4] 边界正确, 则 Phase0 迁移完成")


if __name__ == "__main__":
    main()
