# 论文图: "行为改变来自闭环力学接触" 的证据链 (读 diag_contact_causality 的 npz)
# 四个面板:
#   (a) base_x 时程: 闭环(开/关) + 同一段宽走廊动作开环复放进两种环境
#   (b) |Δbase_x|(t): 同观测配对(开-关) 与 同条件重复(噪声底), 竖直标注首次接触 t_c
#   (c) 横向占位柱状 (max|base_x|, 误差棒取重复)
#   (d) 墙接触力时程: 接触事件本身
# 图内标签用英文 (投稿图), 注释与打印用中文。
# 用法:
#   uv run python -B -m mjlab.scripts.Figure.Figure_backup_contact_causality \
#       logs/backup_contact_causality/<run>/traces_iter8999.npz
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro


COL_BASE_X, COL_FOOT, COL_WNEG, COL_WPOS = 14, 15, 20, 21
DT = 0.01
CLEAR_WIDTH = 0.09                 # 窄走廊净宽 (m), 与实验脚本默认一致
ROBOT_MASS = 0.2750                # 机器人本体质量 (kg), 自重 2.698 N


@dataclass
class Cfg:
    npz_path: str
    t_contact: float = 0.56            # 首次墙接触时刻 (s), 由实验脚本打印
    out_suffix: str = "causality"


# 接触后 (t>=t_c) 的平均 |base_x|, 单位 mm
def _post_mean(tr: np.ndarray, t_c: float) -> float:
    cut = int(t_c / DT)
    return float(np.abs(tr[cut:, COL_BASE_X]).mean() * 1000)


def main() -> None:
    cfg = tyro.cli(Cfg)
    data = np.load(cfg.npz_path)
    t = np.arange(len(data["narrow_on"])) * DT
    on, off = data["narrow_on"], data["narrow_off"]
    on_rep, wide = data["narrow_on_rep1"], data["wide_on"]
    ol_on, ol_off = data["ol_wide_act_in_narrow_on"], data["ol_wide_act_in_narrow_off"]
    face = 0.5 * CLEAR_WIDTH * 1000

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))

    ax = axes[0, 0]
    ax.plot(t, ol_off[:, COL_BASE_X] * 1000, color="#8e44ad", ls=":", lw=1.8,
            label="open-loop (wide-corridor action) in narrow, collision OFF")
    ax.plot(t, off[:, COL_BASE_X] * 1000, color="#2471a3", ls="--", lw=1.8,
            label="closed-loop, narrow, collision OFF")
    ax.plot(t, ol_on[:, COL_BASE_X] * 1000, color="#e67e22", ls="-.", lw=1.8,
            label="open-loop (wide-corridor action) in narrow, collision ON")
    ax.plot(t, on[:, COL_BASE_X] * 1000, color="#c0392b", lw=2.0,
            label="closed-loop, narrow, collision ON")
    ax.plot(t, wide[:, COL_BASE_X] * 1000, color="#7f8c8d", lw=1.2, alpha=0.8,
            label="closed-loop, wide corridor")
    for s in (1, -1):
        ax.axhline(s * face, color="k", ls=":", lw=1.0)
    ax.axvline(cfg.t_contact, color="k", ls="--", lw=1.0)
    ax.annotate(f"first wall contact $t_c$ = {cfg.t_contact:.2f} s",
                xy=(cfg.t_contact, ax.get_ylim()[0] * 0.75), fontsize=7.5,
                xytext=(cfg.t_contact + 0.25, ax.get_ylim()[0] * 0.75),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"lateral position of trunk centre $x$ (mm)")
    ax.set_title("(a) lateral trajectory: same observation, different physics", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.8, loc="lower left", framealpha=0.85)

    ax = axes[0, 1]
    dx_pair = np.abs(on[:, COL_BASE_X] - off[:, COL_BASE_X]) * 1000
    dx_noise = np.abs(on[:, COL_BASE_X] - on_rep[:, COL_BASE_X]) * 1000
    dx_wide = np.abs(on[:, COL_BASE_X] - wide[:, COL_BASE_X]) * 1000
    ax.plot(t, dx_pair, color="#c0392b", lw=1.8,
            label="collision ON vs OFF (same nominal wall, same observation)")
    ax.plot(t, dx_noise, color="#7f8c8d", lw=1.4, ls="--",
            label="same condition, repeated rollout (noise floor)")
    ax.plot(t, dx_wide, color="#2471a3", lw=1.2, ls=":", alpha=0.9,
            label="narrow vs wide corridor (observation also differs)")
    ax.axvline(cfg.t_contact, color="k", ls="--", lw=1.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$|\Delta x|$ (mm)")
    ax.set_title("(b) divergence starts only after contact", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.8, loc="upper left", framealpha=0.85)

    ax = axes[1, 0]
    bars = [
        ("closed-loop\nnarrow ON", [np.abs(on[:, COL_BASE_X]).max() * 1000],
         "#c0392b"),
        ("open-loop wide-act\nnarrow ON", [np.abs(ol_on[:, COL_BASE_X]).max() * 1000],
         "#e67e22"),
        ("closed-loop\nnarrow OFF", [np.abs(off[:, COL_BASE_X]).max() * 1000,
                                     np.abs(data["narrow_off_rep1"][:, COL_BASE_X]).max() * 1000],
         "#2471a3"),
        ("open-loop wide-act\nnarrow OFF", [np.abs(ol_off[:, COL_BASE_X]).max() * 1000],
         "#8e44ad"),
        ("closed-loop\nwide corridor", [np.abs(wide[:, COL_BASE_X]).max() * 1000,
                                        np.abs(data["wide_on_rep1"][:, COL_BASE_X]).max() * 1000],
         "#7f8c8d"),
    ]
    xs = np.arange(len(bars))
    for i, (lab, vals, col) in enumerate(bars):
        m, s = float(np.mean(vals)), float(np.std(vals))
        ax.bar(i, m, color=col, alpha=0.85, width=0.62)
        if len(vals) > 1:
            ax.errorbar(i, m, yerr=s, color="k", capsize=4, lw=1.0)
        ax.text(i, m + 2, f"{m:.0f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=7)
    ax.axhline(face, color="k", ls=":", lw=1.0)
    ax.set_ylabel(r"max $|x|$ of trunk centre (mm)")
    ax.set_title("(c) lateral occupancy is set by contact, not by the command", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1, 1]
    # 单位 N 的口径: 每面墙先把 37 个 body 的净力矢量和起来, 再取两墙范数之和 (见 docs §7.17)
    force = (on[:, COL_WNEG] + on[:, COL_WPOS])
    weight = ROBOT_MASS * 9.81
    ax.plot(t, force, color="#c0392b", lw=1.5,
            label=r"wall squeeze force $|F_L|+|F_R|$")
    ax.axhline(weight, color="#2471a3", ls=":", lw=1.2,
               label=f"robot weight = {weight:.2f} N (0.275 kg)")
    ax.axvline(cfg.t_contact, color="k", ls="--", lw=1.0)
    contact = force > 1e-6
    if contact.any():
        ax.axhline(float(force[contact].mean()), color="#7f8c8d", ls="-.", lw=1.0,
                   label=f"mean while in contact = {force[contact].mean():.1f} N "
                         f"({force[contact].mean() / weight:.1f}× weight)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("contact force (N)")
    ax.set_title("(d) the mechanical event itself", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.0, loc="upper left", framealpha=0.85)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top)
    ax2 = ax.twinx()
    ax2.set_ylim(0, top / weight)
    ax2.set_ylabel("normalized by body weight (-)", fontsize=8.5)

    fig.suptitle("Closed-loop mechanical contact, not corridor awareness, drives the "
                 "behaviour change (checkpoint 8999, $\\lambda$=1)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(cfg.npz_path).with_name(f"{cfg.out_suffix}_figure.png")
    fig.savefig(out, dpi=300)
    print(f"[INFO] 图: {out}")
    print(f"  同一段宽走廊动作: 窄走廊+碰撞开 max|x|={np.abs(ol_on[:, COL_BASE_X]).max() * 1000:.1f} mm, "
          f"接触后 mean|x|={_post_mean(ol_on, cfg.t_contact):.1f} mm")
    print(f"                    窄走廊+碰撞关 max|x|={np.abs(ol_off[:, COL_BASE_X]).max() * 1000:.1f} mm, "
          f"接触后 mean|x|={_post_mean(ol_off, cfg.t_contact):.1f} mm")
    print(f"  闭环窄走廊(碰撞开) max|x|={np.abs(on[:, COL_BASE_X]).max() * 1000:.1f} mm, "
          f"foot|x|max={np.abs(on[:, COL_FOOT]).max() * 1000:.1f} mm")
    print(f"  闭环窄走廊(碰撞关) max|x|={np.abs(off[:, COL_BASE_X]).max() * 1000:.1f} mm, "
          f"foot|x|max={np.abs(off[:, COL_FOOT]).max() * 1000:.1f} mm")


if __name__ == "__main__":
    main()
