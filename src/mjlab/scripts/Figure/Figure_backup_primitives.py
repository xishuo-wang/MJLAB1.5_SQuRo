# 论文图: 脊柱基元四定义随受限强度的变化 (读 diag_righting_primitives 的逐循环 CSV)
# 四个面板 = 论文关键定义 9~12: A_i 激活度 / C_i 贡献系数 / tau_i 时序中心 / O_ij 时序重叠
# 图内标签用英文 (投稿图), 控制台与注释用中文。
# C_i 有两种口径, 都在图里画出来:
#   normalized = A_i / sum_j A_j   (无量纲, 三基元可比 -> 判"主导基元"用这个)
#   raw        = S_i / sum_j S_j   (按未归一化激活面积, 会被行程最大的轴向压制)
# 用法:
#   uv run python -B -m mjlab.scripts.Figure.Figure_backup_primitives \
#       logs/backup_primitives/<run>/primitives_iter8999.csv --window action
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro


BODY_NOMINAL_WIDTH = 0.070        # 站立构型左右宽 (m), 归一化约束强度的参考
GEOMETRIC_FLOOR = 0.0765          # reset 姿态下的走廊净宽硬下界 (m)
PRIMITIVES = ("axial", "lateral", "sagittal")
PRETTY = {"axial": "Axial rotation ($P_A$)",
          "lateral": "Lateral bending ($P_L$)",
          "sagittal": "Sagittal bending ($P_S$)"}
COLOR = {"axial": "#c0392b", "lateral": "#2471a3", "sagittal": "#1e8449"}
PAIRS = (("axial", "lateral"), ("axial", "sagittal"), ("lateral", "sagittal"))


@dataclass
class Cfg:
    csv_path: str
    window: str = "action"            # action = 到进入 P3 的翻正动作段; full = 整循环
    out_suffix: str | None = None


# 按 (净宽, 碰撞) 聚合; C 同时给出两种口径
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cw, coll), sub in df.groupby(["clear_width", "collision"]):
        rec = {"clear_width": float(cw), "collision": int(coll), "n": len(sub),
               "w_over_body": float(cw) / BODY_NOMINAL_WIDTH}
        a_vals = {k: float(sub[f"A_{k}"].mean()) for k in PRIMITIVES}
        s_vals = {k: float(sub[f"S_{k}"].mean()) for k in PRIMITIVES}
        a_sum, s_sum = sum(a_vals.values()), sum(s_vals.values())
        for k in PRIMITIVES:
            for metric in ("A", "tau", "rms", "peak"):
                key = f"{metric}_{k}"
                rec[key] = float(sub[key].mean())
                rec[key + "_std"] = float(sub[key].std())
            rec[f"Cn_{k}"] = a_vals[k] / a_sum if a_sum else np.nan
            rec[f"Cr_{k}"] = s_vals[k] / s_sum if s_sum else np.nan
        for a, b in PAIRS:
            key = f"O_{a}_{b}"
            rec[key] = float(sub[key].mean())
            rec[key + "_std"] = float(sub[key].std())
        for extra in ("T", "base_x_absmax", "foot_x_absmax", "joint_vel_rms", "power"):
            if extra in sub:
                rec[extra] = float(sub[extra].mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["collision", "clear_width"])


def main() -> None:
    cfg = tyro.cli(Cfg)
    df = pd.read_csv(cfg.csv_path, encoding="utf-8")
    if "window" in df.columns:
        df = df[df["window"] == cfg.window]
    table = aggregate(df)
    suffix = cfg.out_suffix or f"primitives_{cfg.window}"
    agg_path = Path(cfg.csv_path).with_name(f"{suffix}_aggregate.csv")
    table.to_csv(agg_path, index=False)
    x_max = float(np.ceil(table["w_over_body"].max() * 10) / 10)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    panels = [
        ("A", None, r"Activation level $A_i$", r"activation level ($A_i$)"),
        (None, "C", r"Contribution coefficient $C_i$",
         r"contribution ($C_i$; solid = $A_i/\sum A_j$, dashed = $S_i/\sum S_j$)"),
        ("tau", None, r"Temporal center $\tau_i$", r"temporal center ($\tau_i$, normalized)"),
        (None, "O", r"Temporal overlap $O_{ij}$", r"overlap ($O_{ij}$)"),
    ]
    for ax, (a_prefix, o_or_c, title, ylab) in zip(axes.ravel(), panels):
        handles: list = []
        labels: list[str] = []
        for collision, ls, mk, alpha in ((1, "-", "o", 1.0), (0, "--", "s", 0.5)):
            style = "collision ON" if collision else "collision OFF"
            sub_all = table[table["collision"] == collision].sort_values("w_over_body")
            if not len(sub_all):
                continue
            if o_or_c == "O":
                combos = [(f"O_{a}_{b}", f"{PRETTY[a]} + {PRETTY[b]}", COLOR[a])
                          for a, b in PAIRS]
            elif o_or_c == "C":
                combos = [(f"Cn_{k}", f"{PRETTY[k]} (normalized)", COLOR[k])
                          for k in PRIMITIVES] + \
                         [(f"Cr_{k}", f"{PRETTY[k]} (raw area)", COLOR[k])
                          for k in PRIMITIVES]
            else:
                combos = [(f"{a_prefix}_{k}", PRETTY[k], COLOR[k]) for k in PRIMITIVES]
            for key, lab, col in combos:
                if key not in sub_all.columns:
                    continue
                dash = ls if o_or_c != "C" or key.startswith("Cn") else ":"
                h = ax.errorbar(sub_all["w_over_body"], sub_all[key],
                                yerr=sub_all.get(key + "_std"),
                                color=col, linestyle=dash, marker=mk, markersize=5,
                                lw=1.7, capsize=3, alpha=alpha)
                handles.append(h)
                labels.append(f"{lab} — {style}")
        ax.axvline(GEOMETRIC_FLOOR / BODY_NOMINAL_WIDTH, color="k", ls="-.", lw=1.0)
        ax.set_xlim(x_max, 1.15)                   # 反向: 右侧 = 约束更强
        ax.set_xlabel(r"Normalized clearance $W / W_{\mathrm{body}}$")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(handles, labels, fontsize=5.8, loc="best", framealpha=0.85)
    fig.suptitle(f"Spinal primitive recruitment versus confinement — checkpoint 8999, "
                 f"$\\lambda$=1 (window: {cfg.window})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png = Path(cfg.csv_path).with_name(f"{suffix}_figure.png")
    fig.savefig(out_png, dpi=300)
    print(f"[INFO] 图: {out_png}\n[INFO] 聚合表: {agg_path}", flush=True)
    cols = ["clear_width", "w_over_body", "collision", "n"] + \
           [f"{m}_{k}" for m in ("A", "tau", "rms") for k in PRIMITIVES] + \
           [f"Cn_{k}" for k in PRIMITIVES] + [f"Cr_{k}" for k in PRIMITIVES] + \
           [f"O_{a}_{b}" for a, b in PAIRS] + ["T", "base_x_absmax", "foot_x_absmax"]
    print(table[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
