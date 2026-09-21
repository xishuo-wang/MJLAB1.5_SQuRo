# 动作尺度诊断 — 扫描策略回放 CSV, 回答"策略是否长期工作在动作空间标称范围 ±1 之外"
# 背景与结论写入 docs/SQuRo_Backup_技术细节.md; 动机是 P1 扭转关节被推到限位。
# 用法:
#   uv run python -B -m mjlab.scripts.Backup.diag_action_vs_ref --pattern "logs/rsl_rl/SQuRo_Backup/**/videos/*.csv"
#   uv run python -B -m mjlab.scripts.Backup.diag_action_vs_ref --plot <单个 csv>
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro

# 动作空间标称边界; JointPositionActionCfg.scale
ACTION_NOMINAL = 1.0
ACTION_SCALE = 0.3
# T1 段长 (s, 名义时间), 与 mdp/config.py 的 T1 对应
P1_BUILD_NOMINAL = 0.65
# 判定"已到达 T1 终点"的容差 (rad), 参考终点为 ∓1.57
ENDPOINT_TOL = 0.15
SPINE = ["F_spine1", "F_body", "H_spine1", "H_body"]
LEG = ["FL_shoulder", "FL_elbow", "FR_shoulder", "FR_elbow",
       "HL_hip", "HL_knee", "HR_hip", "HR_knee"]


@dataclass
class DiagCfg:
    pattern: str = "logs/rsl_rl/SQuRo_Backup/**/videos/*.csv"
    plot: str | None = None
    # 输出目录; 训练日志目录在受限沙箱下不可写, 故默认落在仓库内
    out_dir: str = "diag/action_scale"
    # 表格里最多列出多少个文件 (按修改时间倒序); 0 表示全部
    limit: int = 0


# 单个回放文件的核心指标
def analyse_one(path: Path) -> tuple[dict, dict]:
    df = pd.read_csv(path)
    n = len(df)
    rec: dict = {"file": path.parent.parent.name + "/" + path.name, "frames": n}
    # 时间缩放从文件名 ts<λ> 解析
    lam = np.nan
    if "-ts" in path.stem:
        try:
            lam = float(path.stem.split("-ts")[-1])
        except ValueError:
            lam = np.nan
    rec["lambda"] = lam

    # 动作幅度: 脊柱四关节与腿部八关节分开统计
    for group, joints in (("spn", SPINE), ("leg", LEG)):
        acts = [df[j + "_action"].abs().values for j in joints if j + "_action" in df.columns]
        if not acts:
            continue
        a = np.concatenate(acts)
        rec[f"{group}_act_absmax"] = float(a.max())
        rec[f"{group}_act_absmean"] = float(a.mean())
        rec[f"{group}_act_over1_frac"] = float((a > ACTION_NOMINAL).mean())

    # 跟踪误差: 参考角存在时才算
    spn_errs = []
    for j in SPINE + LEG:
        pc, rc = j + "_pos", j + "_ref_pos"
        if pc in df.columns and rc in df.columns:
            err = (df[pc] - df[rc]).abs()
            rec[f"{j}_err_mean"] = float(err.mean())
            rec[f"{j}_err_max"] = float(err.max())
            if j in SPINE:
                spn_errs.append(float(err.mean()))
    rec["spn_err_mean"] = float(np.mean(spn_errs)) if spn_errs else np.nan

    # P1 提前到位: 扭转关节首次进入参考终点附近的时刻 (相对名义段长)
    ph = df["backup_phase"].values if "backup_phase" in df.columns else None
    for j, sign in (("F_body", -1.57), ("H_body", 1.57)):
        pc = j + "_pos"
        if pc not in df.columns:
            continue
        v = df[pc].values
        hit = np.nonzero(np.abs(v - sign) <= ENDPOINT_TOL)[0]
        if ph is not None:
            hit = hit[ph[hit] == 0]
        rec[f"{j}_p1_endpoint_step"] = int(hit[0]) if hit.size else -1
        # 名义上要到 P1_BUILD_NOMINAL*λ 才到位
        rec[f"{j}_p1_endpoint_ratio"] = (
            float(hit[0] * 0.01 / (P1_BUILD_NOMINAL * lam)) if hit.size and lam == lam and lam > 0 else np.nan
        )
    # 有效增益: pos 变化量 / action 变化量 (仅 P1 段)
    series = {}
    if ph is not None:
        m = ph == 0
        t = np.arange(n) * 0.01
        series = {"t": t[m], "F_body": df["F_body_pos"].values[m], "H_body": df["H_body_pos"].values[m]}
        if "F_body_ref_pos" in df.columns:
            series["F_body_ref"] = df["F_body_ref_pos"].values[m]
            series["H_body_ref"] = df["H_body_ref_pos"].values[m]
        acts = df["F_body_action"].values[m]
        posv = df["F_body_pos"].values[m]
        dpos, dact = np.diff(posv), np.diff(acts)
        msk = np.abs(dact) > 0.05
        rec["F_body_gain"] = float(np.median(dpos[msk] / dact[msk])) if msk.sum() > 3 else np.nan
    return rec, series


# 扫描全部回放文件, 输出汇总表
def scan(cfg: DiagCfg) -> pd.DataFrame:
    files = sorted(Path(".").glob(cfg.pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print(f"[DIAG] 未匹配到文件: {cfg.pattern}")
        return pd.DataFrame()
    if cfg.limit:
        files = files[: cfg.limit]
    print(f"[DIAG] 匹配 {len(files)} 个回放 CSV")
    rows = []
    for f in files:
        try:
            rec, _ = analyse_one(f)
            rec["mtime"] = pd.Timestamp(f.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
            rows.append(rec)
        except Exception as exc:
            print(f"[DIAG] 跳过 {f.name}: {exc}")
    out = pd.DataFrame(rows)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    dest = Path(cfg.out_dir) / "action_scale_scan.csv"
    out.to_csv(dest, index=False)
    print(f"[DIAG] 汇总已保存: {dest}")
    return out


# 单文件详细曲线: 动作 / 参考角 / 实际角
def plot_one(path: Path, out_dir: str) -> None:
    df = pd.read_csv(path)
    n = len(df)
    t = np.arange(n) * 0.01
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    # 左列: F 段, 右列: H 段; 上=动作, 中=目标角(动作*scale), 下=参考 vs 实际
    for col, (pre, sign) in enumerate((("F", -1.0), ("H", 1.0))):
        pairs = [(f"{pre}_body", f"{pre}_body"), (f"{pre}_spine1", f"{pre}_spine1")]
        ax = axes[0][col]
        for j, _ in pairs:
            if j + "_action" in df.columns:
                ax.plot(t, df[j + "_action"].values, linewidth=1.2, label=j)
        ax.axhline(ACTION_NOMINAL, color="r", linestyle=":", linewidth=1.0)
        ax.axhline(-ACTION_NOMINAL, color="r", linestyle=":", linewidth=1.0)
        ax.set_ylabel("原始动作")
        ax.set_title(f"{pre} 段: 策略原始动作 (红虚线=标称 ±1)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[1][col]
        for j, _ in pairs:
            if j + "_action" in df.columns:
                ax.plot(t, df[j + "_action"].values * ACTION_SCALE, linewidth=1.2, label=f"{j} 目标角")
                if j + "_ref_pos" in df.columns:
                    ax.plot(t, df[j + "_ref_pos"].values, "--", linewidth=1.0, label=f"{j} 参考角")
        ax.set_ylabel("关节角 (rad)")
        ax.set_title(f"{pre} 段: 目标角 vs 参考角")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[2][col]
        for j, _ in pairs:
            if j + "_pos" in df.columns:
                ax.plot(t, df[j + "_pos"].values, linewidth=1.2, label=f"{j} 实际角")
            if j + "_ref_pos" in df.columns:
                ax.plot(t, df[j + "_ref_pos"].values, "--", linewidth=1.0, label=f"{j} 参考角")
        ax.set_ylabel("关节角 (rad)")
        ax.set_xlabel("time (s)")
        ax.set_title(f"{pre} 段: 实际角 vs 参考角")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"动作尺度诊断 — {path.name}")
    fig.tight_layout()
    dest = Path(out_dir) / (path.stem + "_action_diag.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"[DIAG] 曲线已保存: {dest}")


def main() -> None:
    cfg = tyro.cli(DiagCfg)
    if cfg.plot:
        plot_one(Path(cfg.plot), cfg.out_dir)
        return
    out = scan(cfg)
    if out.empty:
        return
    cols = ["file", "lambda", "frames", "spn_act_absmax", "spn_act_over1_frac",
            "leg_act_absmax", "leg_act_over1_frac", "F_body_gain",
            "F_body_p1_endpoint_ratio", "H_body_p1_endpoint_ratio",
            "F_body_err_mean", "H_body_err_mean", "mtime"]
    cols = [c for c in cols if c in out.columns]
    with pd.option_context("display.width", 220, "display.max_columns", 50):
        print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
