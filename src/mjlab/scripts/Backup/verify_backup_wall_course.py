# 验收: 逐环境墙位课程的成绩归属与生命周期 + P1 沉降余量课程 (只读, 不落盘)
# 覆盖此前审查要求但一直没有自动化证据的六项:
#   [1] 回合成绩绑定到**本回合**墙位 (发布早于父类重采样; 结构性守卫)
#   [2] 重建/物理条件变化时清空 d_min 批次统计
#   [3] 批次门控: 只统计 d==d_min 的回合, 且要连续两批达标
#   [4] 新建/重建后把课程范围推给命令项 (首回合不跑在范围之外)
#   [5] 回放: 随机课程检查点默认取课程下界, 不用编译模板墙位
#   [6] 墙位采样 frac 语义 (1.0 = 全部最窄档) 与 P1 沉降余量课程取值
# 失败时非零退出码。用法: uv run python -B -m mjlab.scripts.Backup.verify_backup_wall_course
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from mjlab.tasks.SQuRo_Backup.mdp import curriculums as C
from mjlab.tasks.SQuRo_Backup.mdp.command import BackupCommand, BackupCommandCfg
from mjlab.tasks.SQuRo_Backup.rl.runner import SQuRoBackupOnPolicyRunner

fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"    {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        fails.append(msg)


# 造一个只带采样所需属性的命令替身: _sample_wall 只碰 _wall_d / cfg / device / _push_wall
def make_cmd(n: int, frac: float):
    cmd = object.__new__(BackupCommand)
    cmd.cfg = BackupCommandCfg(wall_d_min_frac=frac)
    cmd._env = SimpleNamespace(device=torch.device("cpu"))   # device 是只读属性, 取自 _env
    cmd._wall_d = torch.zeros(n)
    cmd._push_wall = lambda ids: None
    return cmd


# 造一个只带课程状态与 env 替身的 runner。
# 起始轮次必须 >= CURRICULUM_START_ITER: 否则会被"课程未启动"提前拦掉, 门控根本没被执行,
# 断言会变成假阳性 (实测踩过)。
def make_runner(num_envs: int = 2, d_min: float = C.WALL_D_MAX,
                start_iter: int = C.CURRICULUM_START_ITER):
    r = object.__new__(SQuRoBackupOnPolicyRunner)
    r._corridor_num_envs = num_envs
    r._wall_random = True
    r._wall_d_min = d_min
    r._cur_level_iter = start_iter
    r._cur_level_history = []
    r._corridor_fixed = False
    r.env = SimpleNamespace(unwrapped=SimpleNamespace(
        common_step_counter=(start_iter + C.CURRICULUM_MIN_DWELL_ITER + 1) * C._STEPS_PER_ITER))
    r._corridor_change_needed = lambda: False
    pushed: list = []
    r._push_wall_curriculum = lambda enabled=None: pushed.append(enabled)
    r._pushed = pushed
    r._reset_curriculum_window()
    return r


def main() -> None:
    print("=" * 88)
    print("[1] 回合成绩绑定到本回合墙位")
    src = Path(inspect.getsourcefile(BackupCommand)).read_text(encoding="utf-8")
    reset_src = inspect.getsource(BackupCommand.reset)
    i_pub = reset_src.find("_last_ep_wall_d[ids] = self._wall_d[ids]")
    i_super = reset_src.find("return super().reset(env_ids)")
    check(i_pub >= 0, "reset() 里存在回合墙位的发布语句")
    check(i_super >= 0, "reset() 末尾调用父类 reset")
    # 父类 reset 内部会 _resample_command 重采墙位; 发布若排在它之后就会贴上下一回合的墙位
    check(0 <= i_pub < i_super, "发布排在父类重采样之前 (否则宽墙位的成功会被记成最窄档)")
    check("self._resample_command(env_ids)" not in reset_src,
          "reset() 不再额外重采一次墙位 (父类已完成)")

    print("\n[2] 重建/条件变化时清空 d_min 批次统计")
    r = make_runner()
    r._dmin_batch_ep, r._dmin_batch_stood, r._dmin_batch_pass = 256, 256, 1
    r._reset_curriculum_window()
    check((r._dmin_batch_ep, r._dmin_batch_stood, r._dmin_batch_pass) == (0, 0, 0),
          "批次计数与连续达标数被清零 (否则开碰撞后的第一批会来自无墙成绩)")

    print("\n[3] 批次门控: 只算 d==d_min, 且要连续两批")
    r = make_runner(d_min=0.10)
    raw = C.CURRICULUM_BATCH_EPISODES
    r._dmin_batch_ep, r._dmin_batch_stood = raw - 1, raw - 1
    check(r._maybe_lower_dmin(r._cur_level_iter + C.CURRICULUM_MIN_DWELL_ITER + 1) is False,
          f"不足 {raw} 个回合时不结算")
    r._dmin_batch_ep, r._dmin_batch_stood = raw, raw
    it = r._cur_level_iter + C.CURRICULUM_MIN_DWELL_ITER + 1
    check(r._maybe_lower_dmin(it) is False, "第一批达标只记 1/2, 不降档")
    check(r._dmin_batch_pass == 1 and r._dmin_batch_ep == 0, "第一批结算后计数清零")
    r._dmin_batch_ep, r._dmin_batch_stood = raw, raw
    check(r._maybe_lower_dmin(it) is True, "连续两批达标才降档")
    check(abs(r._wall_d_min - 0.09) < 1e-9, f"d_min 降一档 (0.10 -> {r._wall_d_min:.3f})")
    check(r._pushed and r._pushed[-1] is None, "降档后把新范围推给命令项")
    # 未达标必须把连续计数打断
    r2 = make_runner(d_min=0.10)
    r2._dmin_batch_ep, r2._dmin_batch_stood = raw, raw
    r2._maybe_lower_dmin(r2._cur_level_iter + C.CURRICULUM_MIN_DWELL_ITER + 1)
    r2._dmin_batch_ep, r2._dmin_batch_stood = raw, 0
    r2._maybe_lower_dmin(r2._cur_level_iter + C.CURRICULUM_MIN_DWELL_ITER + 1)
    check(r2._dmin_batch_pass == 0, "一批不达标就把连续计数打断")
    # 到下限不再降
    r3 = make_runner(d_min=C.WALL_D_MIN_END)
    r3._dmin_batch_ep, r3._dmin_batch_stood = raw, raw
    check(r3._maybe_lower_dmin(r3._cur_level_iter + C.CURRICULUM_MIN_DWELL_ITER + 1) is False,
          f"到达下限 {C.WALL_D_MIN_END} 后不再降档")

    print("\n[4] 首回合必须遵守课程范围 (推送 + 强制重采)")
    seen: list = []
    r4 = object.__new__(SQuRoBackupOnPolicyRunner)
    r4._corridor_num_envs = 4
    r4._wall_random = True
    r4._wall_d_min = 0.08
    cmd_stub = SimpleNamespace(device=torch.device("cpu"),
                               set_wall_curriculum=lambda a, b, c: seen.append(("set", a, b, c)),
                               _resample_command=lambda ids: seen.append(("resample", int(ids.numel()))))
    r4._command_term = lambda: cmd_stub
    r4._push_wall_curriculum()
    r4._force_wall_resample()
    check(("set", 0.08, C.WALL_D_MAX, C.WALL_D_MIN_FRAC) in seen, "推送了 d_min/上界/下界比例")
    check(("resample", 4) in seen, "推送后强制重采当前回合 (否则首回合跑在范围之外)")

    print("\n[5] 回放: 随机课程检查点默认取课程下界")
    from mjlab.scripts.SQuRo_Backup_play import resolve_corridor
    saved = {"wall_x_neg": -0.08, "wall_x_pos": 0.05, "wall_d_min": 0.06}
    def mk(**kw):
        base = dict(wall_x_neg=None, wall_x_pos=None, corridor_width=None, enable_collision=None)
        base.update(kw)
        return SimpleNamespace(**base)
    cfg = mk()
    neg, pos, _coll, src, _csrc = resolve_corridor(cfg, saved, 1)
    check(abs(neg - (-0.06)) < 1e-9 and abs(pos - 0.05) < 1e-9,
          f"默认取 d_min 对应的墙位 ({neg:+.3f}, {pos:+.3f}), 不是编译模板 -0.08")
    check("d_min" in src, "来源说明里点明用的是课程下界")
    cfg2 = mk(wall_x_neg=-0.10, wall_x_pos=0.05)
    neg2, _p2, _c2, _s2, _cs2 = resolve_corridor(cfg2, saved, 1)
    check(abs(neg2 - (-0.10)) < 1e-9, "命令行显式给墙位时优先")
    legacy = {"wall_x_neg": -0.08, "wall_x_pos": 0.05}
    neg3, _p3, _c3, _s3, _cs3 = resolve_corridor(mk(), legacy, 1)
    check(abs(neg3 - (-0.08)) < 1e-9, "旧检查点 (无 wall_d_min) 仍按记录墙位")

    print("\n[6] 墙位采样 frac 语义 与 P1 沉降余量课程")
    lo, hi, n = 0.06, 0.20, 4000
    for frac, want_all_min in ((1.0, True), (0.0, False)):
        cmd = make_cmd(n, frac)
        cmd._wall_bounds = lambda: (lo, hi)
        cmd._sample_wall(torch.arange(n))
        d = cmd._wall_d
        at_min = (d - lo).abs() < 1e-9
        if want_all_min:
            check(bool(at_min.all()), f"frac=1.0 时全部落在下界 {lo} (实测 {int(at_min.sum())}/{n})")
        else:
            check(int(at_min.sum()) == 0, f"frac=0.0 时不做下界采样 (实测 {int(at_min.sum())}/{n})")
            check(bool((d >= lo).all() and (d <= hi).all()), "frac=0.0 仍落在 [lo, hi] 内")
    cmd = make_cmd(n, C.WALL_D_MIN_FRAC)
    cmd._wall_bounds = lambda: (lo, hi)
    cmd._sample_wall(torch.arange(n))
    share = float(((cmd._wall_d - lo).abs() < 1e-9).float().mean())
    check(abs(share - C.WALL_D_MIN_FRAC) < 0.03,
          f"frac={C.WALL_D_MIN_FRAC} 时下界占比 {share:.3f} (期望 ≈{C.WALL_D_MIN_FRAC})")

    S = C._STEPS_PER_ITER
    check(abs(C.get_p1_settle_margin(0) - 0.50) < 1e-9, "iter 0 余量 0.50")
    check(abs(C.get_p1_settle_margin(C.P1_SETTLE_START_ITER * S) - 0.50) < 1e-9,
          f"iter {C.P1_SETTLE_START_ITER} 余量仍为 0.50 (课程起点)")
    mid = (C.P1_SETTLE_START_ITER + C.P1_SETTLE_END_ITER) // 2
    midv = C.get_p1_settle_margin(mid * S)
    check(abs(midv - 0.25) < 1e-9, f"iter {mid} 线性中点余量 {midv:.3f} (期望 0.25)")
    check(abs(C.get_p1_settle_margin(C.P1_SETTLE_END_ITER * S) - C.P1_SETTLE_MIN) < 1e-9,
          f"iter {C.P1_SETTLE_END_ITER} 余量 {C.P1_SETTLE_MIN}")
    check(abs(C.get_p1_settle_margin(99999 * S) - C.P1_SETTLE_MIN) < 1e-9, "之后保持下限")
    check(C.P1_SETTLE_MIN >= 0.0, "下限非负 (为负会截断真实扭转动作)")
    check(abs(C.P1_SETTLE_MAX - 0.50) < 1e-9, "起始余量 0.50 与旧推进门口径一致")

    print("\n" + "=" * 88)
    if fails:
        print(f"结论: 未通过 {len(fails)} 项")
        for f in fails:
            print("  - " + f)
        sys.exit(1)
    print("结论: 全部通过")


if __name__ == "__main__":
    main()
