# 图 2-2C(e): 受限空间翻正的序贯姿态重构
# 上: 前/后躯正置程度 u_F, u_H 与"双倒-一正一倒-双正"三段背景
# 下: 实际轴向旋转角 theta_A^act 与三点折线总弯曲角 (后者用于证明上下面板非冗余)
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 输入回放 CSV: 一个文件是一段回放, 跨文件的所有完整循环一起做均值与带
CSV_PATHS: tuple[str, ...] = (
    r"C:\Users\MECHREVO\AppData\Local\Temp\backup_replay_90mm\videos\Temp_8999-ts1.00-xn-0.06-xp0.05.csv",
)
OUT_DIR = Path(r"D:\MuJoCoLab_1.5\Figure\Backup\output")
OUT_STEM = "fig_righting_sequence"

# 横轴口径: True = 按各循环自身时长归一到 [0,1]; False = 绝对秒
USE_NORMALIZED_TIME = True
# 单循环最少步数, 以及是否要求该循环内出现过 P3 (相位 2)
MIN_CYCLE_STEPS = 80
REQUIRE_PHASE3 = True

DT = 0.01                 # 控制步长 (s), 与 _STEPS_PER_ITER/step_dt 一致
COS45 = 0.7071            # 判据锥: 与 BackupCommandCfg.pose_angle_tolerance_deg=45 同源
DPI = 300
LABEL_LANG = "en"         # 论文图用 en, 需要中文标签时改 "zh"

TEXT = {
    "en": {
        "u": "Trunk uprightness  $u_F$ / $u_H$",
        "ax": "Spinal deformation (rad)",
        "t": "Normalized righting time",
        "t_abs": "Time (s)",
        "f": "$u_F$ (forebody)",
        "h": "$u_H$ (hindbody)",
        "axial": r"$\theta_A^{act}$ (axial rotation)",
        "bend": r"$\theta_{bend}$ (total bending)",
        "stage1": "Double inverted",
        "stage2": "Asymmetric",
        "stage3": "Double upright",
        "cone": r"$\pm\cos 45^\circ$",
    },
    "zh": {
        "u": "前/后躯正置程度 $u_F$ / $u_H$",
        "ax": "脊柱形变角 (rad)",
        "t": "归一化翻正时间",
        "t_abs": "时间 (s)",
        "f": "$u_F$ (前躯)",
        "h": "$u_H$ (后躯)",
        "axial": r"$\theta_A^{act}$ (实际轴向旋转)",
        "bend": r"$\theta_{bend}$ (总弯曲角)",
        "stage1": "双倒",
        "stage2": "一正一倒",
        "stage3": "双正",
        "cone": r"$\pm\cos 45^\circ$",
    },
}


COLOR_F = "#C0504D"
COLOR_H = "#4F81BD"
COLOR_AX = "#7F6000"
COLOR_BEND = "#9BBB59"
STAGE_COLORS = ("#DCE6F1", "#FDE9D9", "#E2EFDA")


# 按相位回落切循环, 丢掉未走完的末段
def split_cycles(df: pd.DataFrame) -> list[tuple[int, int]]:
    phase = df["backup_phase"].to_numpy(int)
    starts = [0] + [i for i in range(1, len(phase)) if phase[i] < phase[i - 1] and phase[i] == 0]
    cycles = []
    for k, s in enumerate(starts):
        if k + 1 >= len(starts):
            break                      # 末段没有下一次复位, 完成与否不可知
        e = starts[k + 1]
        if e - s < MIN_CYCLE_STEPS:
            continue
        if REQUIRE_PHASE3 and 2 not in phase[s:e]:
            continue
        cycles.append((s, e))
    return cycles


# 首次跨过 +cos45 的下标 (即该段被判定为正置的时刻)
def first_upright(u: np.ndarray) -> int | None:
    hit = np.nonzero(u >= COS45)[0]
    return int(hit[0]) if hit.size else None


# 取单循环的曲线, 时间轴按需归一
def cycle_curve(df: pd.DataFrame, s: int, e: int) -> dict:
    t = np.arange(e - s) * DT
    out = {
        "t": t / t[-1] if USE_NORMALIZED_TIME else t,
        "duration": t[-1],
        "u_f": df["f_body_up_cos"].to_numpy(float)[s:e],
        "u_h": df["h_body_up_cos"].to_numpy(float)[s:e],
    }
    if "spine_axial" in df.columns:
        out["axial"] = df["spine_axial"].to_numpy(float)[s:e]
    if "spine_bend_total" in df.columns:
        out["bend"] = df["spine_bend_total"].to_numpy(float)[s:e]
    return out


# 把多个循环重采样到公共横轴, 返回均值与标准差
def stack(cycles: list[dict], key: str, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = [np.interp(grid, c["t"], c[key]) for c in cycles if key in c]
    return np.mean(rows, axis=0), np.std(rows, axis=0)


# 汇总各循环的时序读数, 供正文取数与自检
def report(cycles: list[dict]) -> None:
    print(f"\n完整循环 {len(cycles)} 个")
    print(f"{'#':>3}{'时长s':>9}{'t(后躯正)':>11}{'t(前躯正)':>11}{'Δt':>8}{'先翻':>7}"
          f"{'|Δu|峰':>9}{'轴向峰':>9}{'总弯曲峰':>10}")
    for i, c in enumerate(cycles):
        i_f, i_h = first_upright(c["u_f"]), first_upright(c["u_h"])
        tf = np.nan if i_f is None else c["t"][i_f] * (c["duration"] if USE_NORMALIZED_TIME else 1.0)
        th = np.nan if i_h is None else c["t"][i_h] * (c["duration"] if USE_NORMALIZED_TIME else 1.0)
        lead = "-" if (i_f is None or i_h is None) else ("后躯" if i_h < i_f else "前躯")
        ax = np.nanmax(np.abs(c.get("axial", np.array([np.nan]))))
        bd = np.nanmax(c.get("bend", np.array([np.nan])))
        print(f"{i:>3}{c['duration']:>9.2f}{th:>11.2f}{tf:>11.2f}{tf - th:>8.2f}{lead:>7}"
              f"{np.abs(c['u_f'] - c['u_h']).max():>9.3f}{ax:>9.3f}{bd:>10.3f}")


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    txt = TEXT[LABEL_LANG]

    frames = [pd.read_csv(p) for p in CSV_PATHS]
    for path, df in zip(CSV_PATHS, frames, strict=True):
        print(f"{Path(path).name}: {len(df)} 步, {len(df.columns)} 列")
    cycles = [c for df in frames for c in (cycle_curve(df, s, e) for s, e in split_cycles(df))]
    if not cycles:
        raise SystemExit("没有可用的完整循环, 检查 CSV 与 MIN_CYCLE_STEPS / REQUIRE_PHASE3")
    report(cycles)

    has_deform = "axial" in cycles[0]
    if not has_deform:
        print("[WARN] CSV 缺少 spine_axial 列, 下方面板将跳过; 需要重录回放 (见 SQuRo_Backup_play.py)")

    grid = (np.linspace(0.0, 1.0, 201) if USE_NORMALIZED_TIME
            else np.linspace(0.0, min(c["duration"] for c in cycles), 201))
    mean_f, std_f = stack(cycles, "u_f", grid)
    mean_h, std_h = stack(cycles, "u_h", grid)

    # 三段边界取各循环跨越时刻的均值; 必须与 grid 同单位 (归一化时 c["t"] 已在 [0,1])
    onsets = []
    for c in cycles:
        i_f, i_h = first_upright(c["u_f"]), first_upright(c["u_h"])
        if i_f is not None and i_h is not None:
            onsets.append((float(c["t"][i_h]), float(c["t"][i_f])))
    if onsets:
        t_h = float(np.mean([o[0] for o in onsets]))
        t_f = float(np.mean([o[1] for o in onsets]))
    else:
        t_h, t_f = 0.33 * grid[-1], 0.66 * grid[-1]
    unit = "(归一化)" if USE_NORMALIZED_TIME else "s"
    print(f"\n三段边界: t(后躯正置)={t_h:.3f}  t(前躯正置)={t_f:.3f}  Δt={t_f - t_h:.3f} {unit}")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.12},
    )

    # 三段背景只在 ax1 铺满两个面板的竖直范围, 视觉上贯穿
    edges = [grid[0], t_h, t_f, grid[-1]]
    labels = [txt["stage1"], txt["stage2"], txt["stage3"]]
    span = grid[-1] - grid[0]
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        ax1.axvspan(lo, hi, color=STAGE_COLORS[k], zorder=0)
        ax2.axvspan(lo, hi, color=STAGE_COLORS[k], zorder=0)
        if hi - lo > 0.12 * span:            # 太窄的区间不写字, 避免标签互相压住
            ax1.text(0.5 * (lo + hi), 1.02, labels[k], ha="center", va="bottom",
                     transform=ax1.get_xaxis_transform(), fontsize=8, color="#404040")

    ax1.axhline(COS45, color="gray", lw=0.8, ls=":", zorder=1)
    ax1.axhline(-COS45, color="gray", lw=0.8, ls=":", zorder=1)
    ax1.text(grid[-1], COS45, " " + txt["cone"], va="bottom", ha="right", fontsize=7, color="gray")

    ax1.plot(grid, mean_f, color=COLOR_F, lw=2.0, label=txt["f"], zorder=3)
    ax1.plot(grid, mean_h, color=COLOR_H, lw=2.0, label=txt["h"], zorder=3)
    ax1.fill_between(grid, mean_f - std_f, mean_f + std_f, color=COLOR_F, alpha=0.18, lw=0, zorder=2)
    ax1.fill_between(grid, mean_h - std_h, mean_h + std_h, color=COLOR_H, alpha=0.18, lw=0, zorder=2)

    for t_c, col in ((t_h, COLOR_H), (t_f, COLOR_F)):
        ax1.axvline(t_c, color=col, lw=1.0, ls="--", alpha=0.8, zorder=4)
    ax1.annotate("", xy=(t_h, -0.62), xytext=(t_f, -0.62),
                 arrowprops=dict(arrowstyle="<->", color="#606060", lw=1.0))
    ax1.text(0.5 * (t_h + t_f), -0.58, rf"$\Delta t$ = {t_f - t_h:.2f}", ha="center",
             va="bottom", fontsize=9, color="#606060")

    ax1.set_ylabel(txt["u"])
    ax1.set_ylim(-1.12, 1.12)
    ax1.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax1.legend(loc="lower right", frameon=False, fontsize=8, ncol=2)
    ax1.grid(axis="y", ls=":", alpha=0.35)

    if has_deform:
        mean_a, std_a = stack(cycles, "axial", grid)
        ax2.plot(grid, mean_a, color=COLOR_AX, lw=2.0, label=txt["axial"], zorder=3)
        ax2.fill_between(grid, mean_a - std_a, mean_a + std_a, color=COLOR_AX, alpha=0.18, lw=0, zorder=2)
        if "bend" in cycles[0]:
            mean_b, std_b = stack(cycles, "bend", grid)
            ax2.plot(grid, mean_b, color=COLOR_BEND, lw=1.8, ls="-", label=txt["bend"], zorder=3)
            ax2.fill_between(grid, mean_b - std_b, mean_b + std_b, color=COLOR_BEND, alpha=0.18, lw=0, zorder=2)
        ax2.axhline(0.0, color="gray", lw=0.8, ls=":", zorder=1)
        ax2.set_ylabel(txt["ax"])
        ax2.legend(loc="upper right", frameon=False, fontsize=8)

    ax2.set_xlabel(txt["t"] if USE_NORMALIZED_TIME else txt["t_abs"])
    ax2.set_xlim(edges[0], edges[-1])
    ax2.grid(axis="y", ls=":", alpha=0.35)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"{OUT_STEM}.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"已保存 {path}")


if __name__ == "__main__":
    main()
