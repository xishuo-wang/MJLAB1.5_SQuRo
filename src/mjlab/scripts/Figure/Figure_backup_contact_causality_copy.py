# 论文图: "行为改变来自闭环力学接触" 的证据链 (读取 diag_contact_causality 的 npz)
# 只保留两个面板:
#   (a) base_x 时程: 闭环(开/关) + 同一段宽走廊动作开环复放进两种环境
#   (d) 墙接触力时程: 接触事件本身
#       - 挤压力曲线 (红)
#       - 机器人自重参考线 (蓝点线)
#       - 接触期间平均力 (灰点划线, 并给出 ×自重 倍数)
#       - 右轴: 以自重为单位归一化
# 图例: 每个面板各自独立图例 (常规做法)
# 图内文字、注释与打印全部用中文。
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 中文字体配置
# ============================================================
plt.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "Noto Sans CJK SC",
    "PingFang SC", "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 配置区
# ============================================================
NPZ_PATH = r"D:\MuJoCoLab_1.5\logs\backup_contact_causality\2026-09-27_14-33-27_v1_soft\traces_iter8999.npz"
T_CONTACT = 0.56
OUT_SUFFIX = "causality_ad"

# ---------- 字号配置 (单位: pt) ----------
FS_SUPTITLE = 25
FS_TITLE    = 21
FS_AXLABEL  = 19
FS_TICK     = 16
FS_LEGEND   = 14
FS_ANNOT    = 20

# ---------- 画布与分辨率 ----------
FIGSIZE = (15.5, 6.8)
DPI     = 300

# ---------- 布局配置 (手动 subplots_adjust) ----------
LAY_LEFT   = 0.06
LAY_RIGHT  = 0.955   # 给 (d) 右轴标签留空间
LAY_TOP    = 0.90
LAY_BOTTOM = 0.11    # 图例回到各面板内, 底部不用留空间
LAY_WSPACE = 0.20

# ---------- 线宽配置 ----------
LW_MAIN = 3.0
LW_MED  = 2.6
LW_THIN = 2.2
LW_REF  = 1.4

# ---------- 配色配置 ----------
# (a) 5 种实验条件
C_CL_ON  = "#c0392b"  # 闭环, 窄走廊, 碰撞开  (红, 同时用于 (d) 接触力)
C_OL_ON  = "#e67e22"  # 开环动作复放, 碰撞开
C_CL_OFF = "#2471a3"  # 闭环, 窄走廊, 碰撞关  (蓝, 同时用于 (d) 自重参考线)
C_OL_OFF = "#8e44ad"  # 开环动作复放, 碰撞关
C_WIDE   = "#7f8c8d"  # 闭环, 宽走廊  (灰, 同时用于 (d) 接触期间平均线)

# ---------- 绘图物理参数 ----------
COL_BASE_X, COL_FOOT, COL_WNEG, COL_WPOS = 14, 15, 20, 21
DT = 0.01
CLEAR_WIDTH = 0.09
ROBOT_MASS = 0.2750   # 机器人本体质量 (kg), 自重 2.698 N


def _post_mean(tr: np.ndarray, t_c: float) -> float:
    cut = int(t_c / DT)
    return float(np.abs(tr[cut:, COL_BASE_X]).mean() * 1000)


def main() -> None:
    npz_path = NPZ_PATH
    t_contact = T_CONTACT
    out_suffix = OUT_SUFFIX

    data = np.load(npz_path)
    t = np.arange(len(data["narrow_on"])) * DT
    on, off = data["narrow_on"], data["narrow_off"]
    wide = data["wide_on"]
    ol_on, ol_off = data["ol_wide_act_in_narrow_on"], data["ol_wide_act_in_narrow_off"]
    face = 0.5 * CLEAR_WIDTH * 1000

    # 1x2 横排布局
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    # ---------- (a) 横向轨迹 ----------
    ax = axes[0]
    ax.plot(t, ol_off[:, COL_BASE_X] * 1000, color=C_OL_OFF, ls=":",  lw=LW_MED,
            label="开环动作复放, 碰撞关")
    ax.plot(t, off[:, COL_BASE_X] * 1000,    color=C_CL_OFF, ls="--", lw=LW_MED,
            label="闭环窄走廊, 碰撞关")
    ax.plot(t, ol_on[:, COL_BASE_X] * 1000,  color=C_OL_ON,  ls="-.", lw=LW_MED,
            label="开环动作复放, 碰撞开")
    ax.plot(t, on[:, COL_BASE_X] * 1000,     color=C_CL_ON,  lw=LW_MAIN,
            label="闭环窄走廊, 碰撞开")
    ax.plot(t, wide[:, COL_BASE_X] * 1000,   color=C_WIDE,   lw=LW_THIN, alpha=0.9,
            label="闭环, 宽走廊")
    for s in (1, -1):
        ax.axhline(s * face, color="k", ls=":", lw=LW_REF)
    ax.axvline(t_contact, color="k", ls="--", lw=LW_REF)
    ax.annotate(f"首次墙接触 $t_c$ = {t_contact:.2f} s",
                xy=(t_contact, 25), xytext=(0.75, 30),
                fontsize=FS_ANNOT, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=1.6, color="black"))
    ax.set_xlabel("时间 (s)", fontsize=FS_AXLABEL)
    ax.set_ylabel("躯干中心横向位置 $x$ (mm)", fontsize=FS_AXLABEL)
#     ax.set_title("(a) 横向轨迹: 同一观测, 不同物理", fontsize=FS_TITLE)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=FS_LEGEND - 2, loc="lower left", framealpha=0.85)

    # ---------- (d) 机械接触事件 ----------
    # 单位 N 的口径: 每面墙先把 37 个 body 的净力矢量和起来, 再取两墙范数之和 (见 docs §7.17)
    ax = axes[1]
    force = (on[:, COL_WNEG] + on[:, COL_WPOS])
    weight = ROBOT_MASS * 9.81

    # 挤压力曲线 (与 (a) 红线同色, 语义延续)
    ax.plot(t, force, color=C_CL_ON, lw=LW_THIN,
            label="墙挤压力 $|F_L|+|F_R|$")

    # 机器人自重参考线
    ax.axhline(weight, color=C_CL_OFF, ls=":", lw=LW_REF,
               label=f"机器人自重 = {weight:.2f} N")

    # 首次墙接触时刻
    ax.axvline(t_contact, color="k", ls="--", lw=LW_REF)

    # 接触期间的平均力 (力大于阈值视为处于接触状态)
    contact = force > 1e-6
    if contact.any():
        ax.axhline(float(force[contact].mean()), color=C_WIDE, ls="-.", lw=LW_REF,
                   label=f"接触期间平均 = {force[contact].mean():.1f} N "
                         f"({force[contact].mean() / weight:.1f}× 自重)")

    ax.set_xlabel("时间 (s)", fontsize=FS_AXLABEL)
    ax.set_ylabel("接触力 (N)", fontsize=FS_AXLABEL)
#     ax.set_title("(d) 机械事件本身", fontsize=FS_TITLE)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=FS_LEGEND - 2, loc="upper left", framealpha=0.85)

    # 右轴: 以自重为单位归一化
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top)
    ax2 = ax.twinx()
    ax2.set_ylim(0, top / weight)
    ax2.set_ylabel("归一化 (以自重为单位)", fontsize=FS_AXLABEL - 3)
    ax2.tick_params(axis="y", labelsize=FS_TICK - 2)

    # ============================================================
    # 布局
    # ============================================================
    fig.subplots_adjust(left=LAY_LEFT, right=LAY_RIGHT, top=LAY_TOP,
                        bottom=LAY_BOTTOM, wspace=LAY_WSPACE)

    # ---------- 总标题 ----------
#     fig.suptitle("行为改变来自闭环力学接触, 而非走廊感知 "
#                  "(checkpoint 8999, $\\lambda$=1)", fontsize=FS_SUPTITLE)

    # ---------- 保存 ----------
    out = Path(npz_path).with_name(f"{out_suffix}_figure.png")
    fig.savefig(out, dpi=DPI)
    print(f"[INFO] 图: {out}")

    # 终端打印统计量
    print(f"  同一段宽走廊动作: 窄走廊+碰撞开 max|x|={np.abs(ol_on[:, COL_BASE_X]).max() * 1000:.1f} mm, "
          f"接触后 mean|x|={_post_mean(ol_on, t_contact):.1f} mm")
    print(f"                    窄走廊+碰撞关 max|x|={np.abs(ol_off[:, COL_BASE_X]).max() * 1000:.1f} mm, "
          f"接触后 mean|x|={_post_mean(ol_off, t_contact):.1f} mm")
    print(f"  闭环窄走廊(碰撞开) max|x|={np.abs(on[:, COL_BASE_X]).max() * 1000:.1f} mm, "
          f"foot|x|max={np.abs(on[:, COL_FOOT]).max() * 1000:.1f} mm")
    print(f"  闭环窄走廊(碰撞关) max|x|={np.abs(off[:, COL_BASE_X]).max() * 1000:.1f} mm, "
          f"foot|x|max={np.abs(off[:, COL_FOOT]).max() * 1000:.1f} mm")
    if contact.any():
        print(f"  (d) 机器人自重 = {weight:.3f} N; "
              f"接触期间平均力 = {force[contact].mean():.2f} N "
              f"({force[contact].mean() / weight:.2f}× 自重)")


if __name__ == "__main__":
    main()