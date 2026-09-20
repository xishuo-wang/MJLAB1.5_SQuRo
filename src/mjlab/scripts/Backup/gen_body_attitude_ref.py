# 从策略成功回放的 CSV 提取躯干姿态参考 (两段背腹轴的世界 Z 余弦随时间)
#
# 为什么取策略回放而不是手调脚本: 手调 slow1_target 的脊柱参考在 P1 是 0 -> ∓1.57 的斜坡,
# 它对应"整机顺时针转 90° + 前后段反向折 75°"的构型, 与参考表 t=0 的姿态余弦(u=+1)矛盾;
# 实测脚本随动器的姿态余弦全程停在 -1 从未翻正(见技术细节 §7.7), 故取不到可用参考。
# 策略在 P3 前完成了一次完整成功循环, 是现存的唯一自洽成功轨迹。
#
# 输出: [N,3] = (t_nom, u_F_ref, u_H_ref), t_nom 以 P1 起点为 0, 段内时间按 λ 可缩放。
# 用法: uv run python -B -m mjlab.scripts.Backup.gen_body_attitude_ref
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro

from mjlab.tasks.SQuRo_Backup.mdp.timing import P1_END, P2_END, REFERENCE_TOTAL_TIME

# 取哪一份回放: 该 run 的策略在 P3 前完成了一次完整成功循环
DEFAULT_CSV = "logs/rsl_rl/SQuRo_Backup/2026-09-20_01-40-20/videos/SQuRo_Backup_2999-ts2.00.csv"
OUT_PATH = Path(__file__).resolve().parents[2] / "tasks" / "SQuRo_Backup" / "mdp" / "Bio_Data" / "backup_body_attitude.npy"


@dataclass
class GenCfg:
    csv: str = DEFAULT_CSV
    out: str = str(OUT_PATH)
    dt: float = 0.005
    # 只取第一个完整循环: 从下一处双倒(t≈-1/1)起截断; None 时自动找
    cut_frame: int | None = None


# 段内时钟 -> 统一名义时间 (与 mdp/reference.py::_stage_t_nom 同一约定)
def stage_t_nom(phase: np.ndarray, t_phase: np.ndarray, dt: float) -> np.ndarray:
    # 回放 CSV 的 backup_t_phase 是段内时钟; λ 由文件名给出, 表按名义时间存
    lam = 1.0
    local = t_phase
    p1 = local
    p2 = P1_END + np.minimum(local, P2_END - P1_END)
    p3 = P2_END + local
    return np.where(phase == 0, p1, np.where(phase == 1, p2, p3))


def main() -> None:
    cfg = tyro.cli(GenCfg)
    df = pd.read_csv(cfg.csv)
    stem = Path(cfg.csv).stem
    lam = float(stem.split("-ts")[-1]) if "-ts" in stem else 1.0
    print(f"来源: {cfg.csv}")
    print(f"λ={lam:g}  帧数={len(df)}")

    phase = df["backup_phase"].values.astype(int)
    t_phase = df["backup_t_phase"].values.astype(float)
    # 段内时钟是"实际秒", 除以 λ 得名义秒 (与技术细节的段内时钟约定一致)
    t_nom = stage_t_nom(phase, t_phase / lam, cfg.dt)
    uf = df["f_body_up_cos"].values.astype(float)
    uh = df["h_body_up_cos"].values.astype(float)

    # 只保留第一个完整循环: 循环复位后两段会一起跳回仰卧, 会把参考拉回 -1。
    if cfg.cut_frame is not None:
        cut = int(cfg.cut_frame)
    else:
        cut = len(df)
        for i in range(1, len(df)):
            # 已离开仰卧(曾同时 > -0.9)之后, 又同时回到接近倒置 => 循环复位
            if uf[i] < -0.9 and uh[i] < -0.9 and np.nanmax(uf[:i]) > -0.9:
                cut = i
                break
    print(f"取前 {cut} 帧 (t_nom 至 {t_nom[cut - 1]:.3f} s), 之后为循环复位")

    t_nom, uf, uh = t_nom[:cut], uf[:cut], uh[:cut]
    # 重采样到均匀网格 (t_nom 在阶段切换处会跳变, 用索引插值)
    order = np.argsort(t_nom)
    t_nom, uf, uh = t_nom[order], uf[order], uh[order]
    grid = np.arange(0.0, REFERENCE_TOTAL_TIME + cfg.dt, cfg.dt)
    uf_g = np.interp(grid, t_nom, uf)
    uh_g = np.interp(grid, t_nom, uh)
    # 起点必须与参考表 t=0 的物理姿态一致 (仰卧 -> -1)
    uf_g[0], uh_g[0] = -1.0, -1.0
    rows = np.stack([grid, uf_g, uh_g], axis=1)

    print()
    print("提取结果 (名义时间):")
    print("%8s %10s %10s" % ("t_nom", "u_F_ref", "u_H_ref"))
    for tt in (0.0, 0.20, 0.40, 0.80, 0.95, 1.20, 1.45, 2.00, 2.50):
        i = int(np.argmin(np.abs(grid - tt)))
        print("%8.2f %+10.3f %+10.3f" % (grid[i], uf_g[i], uh_g[i]))
    print()
    print("合理性: 起点 u=%.2f/%.2f (应≈-1 仰卧)  终点 u=%.2f/%.2f (应≈+1 正置)" % (
        uf_g[0], uh_g[0], uf_g[-1], uh_g[-1]))
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, rows)
    print(f"\n已保存 {rows.shape} -> {out}")


if __name__ == "__main__":
    main()
