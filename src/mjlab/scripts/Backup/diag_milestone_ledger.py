# 里程碑时间质量账本 — 量化"提前完成"在 S1/S2/成功三个里程碑上损失多少钱
# 目的: 判断"跳过 P1 斜坡"是不是理性选择。核函数与 mdp/rewards.py::_milestone_time_quality 同源。
# 用法: uv run python -B -m mjlab.scripts.Backup.diag_milestone_ledger
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tyro

from mjlab.tasks.SQuRo_Backup.mdp.timing import (
    P1_BUILD_DURATION,
    P1_END,
    P2_END,
    QUALITY_SIGMA_EARLY_FRAC,
    QUALITY_SIGMA_LATE_S,
    REFERENCE_TOTAL_TIME,
    STAND_CONFIRM_DURATION,
)
from mjlab.tasks.SQuRo_Backup.mdp.curriculums import _CURVES


@dataclass
class LedgerCfg:
    # 时间缩放 λ 列表
    lambdas: tuple[float, ...] = (1.0, 2.0, 3.0)
    # 提前量扫描范围 (s, 实际时间), 从 0 到该值
    max_early_s: float = 0.6
    # 每个里程碑采样点数
    samples: int = 13


# 与 rewards.py::_milestone_time_quality 逐字一致: 早侧 σ 按 λ 缩放, 晚侧 σ 为绝对秒
def quality(dev: np.ndarray, lam: float) -> np.ndarray:
    sigma = np.where(dev < 0.0, QUALITY_SIGMA_EARLY_FRAC * lam, QUALITY_SIGMA_LATE_S)
    return np.exp(-0.5 * (dev / sigma) ** 2)


def main() -> None:
    cfg = tyro.cli(LedgerCfg)
    w1 = _CURVES["weight_milestone_s1"][0]
    w2 = _CURVES["weight_milestone_s2"][0]
    ws = _CURVES["weight_milestone_success"][0]
    print("里程碑权重: S1=%.1f  S2=%.1f  成功=%.1f" % (w1, w2, ws))
    print("段末 (名义秒): P1_END=%.2f  P2_END=%.2f  参考总长=%.2f" % (P1_END, P2_END, REFERENCE_TOTAL_TIME))
    print("核参数: 早侧 σ=%.2f·λ, 晚侧 σ=%.2f s" % (QUALITY_SIGMA_EARLY_FRAC, QUALITY_SIGMA_LATE_S))
    print()

    early = np.linspace(0.0, cfg.max_early_s, cfg.samples)
    for lam in cfg.lambdas:
        print("=" * 96)
        print("λ=%.2f   名义段末 (实际秒): S1=%.2f  S2=%.2f  成功=%.2f" % (
            lam, P1_END * lam, P2_END * lam, REFERENCE_TOTAL_TIME * lam))
        print("-" * 96)
        hdr = "%10s | %12s %12s | %12s %12s | %12s %12s | %10s" % (
            "提前(s)", "S1按时", "S1提前", "S2按时", "S2提前", "成功按时", "成功提前", "总损失")
        print(hdr)
        for e in early:
            # 提前量按实际秒计, dev < 0 表示早到
            q = quality(np.array([-e]), lam)[0]
            s1_late, s1_early = w1 * 1.0, w1 * q
            s2_late, s2_early = w2 * 1.0, w2 * q
            ss_late, ss_early = ws * 1.0, ws * q
            # 三个里程碑同时提前 e 秒的总损失
            total = (s1_late - s1_early) + (s2_late - s2_early) + (ss_late - ss_early)
            print("%10.3f | %12.2f %12.2f | %12.2f %12.2f | %12.2f %12.2f | %10.2f" % (
                e, s1_late, s1_early, s2_late, s2_early, ss_late, ss_early, total))
        print()

    # 反向求: 要让"提前 e 秒"的总损失达到 X 分, 需要把权重提到多少
    print("=" * 96)
    print("标定参考: 若希望「提前 0.3 s」的总损失达到给定目标, 三个权重需同步放大的倍数")
    print("-" * 96)
    base = w1 + w2 + ws
    for lam in cfg.lambdas:
        q = quality(np.array([-0.3]), lam)[0]
        cur = base * (1.0 - q)
        print("λ=%.2f  当前提前0.3s损失=%.2f 分   放大1倍→%.2f   放大2倍→%.2f   放大3倍→%.2f" % (
            lam, cur, cur, cur * 2, cur * 3))
    print()
    print("占位说明: 单回合总奖励量级请配合 TensorBoard 的 Episode_Reward/* 一起读, "
          "本表只给出时间质量这一项的绝对金额。")


if __name__ == "__main__":
    main()
