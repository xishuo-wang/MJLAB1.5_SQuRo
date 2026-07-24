"""
SQuRo 脊柱逆运动学分析 v2（MuJoCo 验证修正版）
==============================================
基于 MuJoCo 实测修正后的运动学模型。

MuJoCo 实测结论（spine_mujoco_verify.py）：
    - F_body_Link 和 H_body_Link 在零位时 forward 方向相反（180° 差）
    - F_body_joint 世界轴 = -Y，对 heading 增益 = -1.0
    - F_spine1_joint 世界轴 = -Z，对 heading 增益 = -1.0
    - H_spine1_joint 世界轴 = +X，对 heading 增益 = 0（扭转轴）
    - H_body_joint 世界轴 = -Y，对 heading 增益 = 0

修正后线性模型：Δψ ≈ -(θ_fb + θ_fs)
    - θ_fb = F_body 角（范围 ±90°）
    - θ_fs = F_spine1 角（范围 ±34.4°）
    - H_spine1 和 H_body 不直接贡献 heading，但存在非线性耦合

IK 策略：
    1. 优先使用 F_spine1（±34.4°满量程）
    2. 不足部分由 F_body 补偿（±90°满量程）
    3. H_spine1 保持为 0（最大 heading 效率）
    4. H_body 保持为 0（对 heading 无影响）
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 尝试中文字体
try:
    matplotlib.font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
except Exception:
    pass
plt.rcParams['axes.unicode_minus'] = False

# 关节限位
JOINT_LIMITS = {
    "F_body":    (-90.0, 90.0),     # 度
    "F_spine1":  (-34.4, 34.4),
    "H_spine1":  (-34.4, 34.4),
    "H_body":    (-90.0, 90.0),
}

# MuJoCo 实测线性增益 (deg heading / deg joint)
SLOPES = {
    "F_body":    -1.0,
    "F_spine1":  -1.0,
    "H_spine1":   0.0,
    "H_body":     0.0,
}

# 非线性耦合参数（从 MuJoCo 数据估计）
# 当 H_spine1 非零时，F_body/F_spine1 的 heading 效率降低
# coupling_factor: 有效 heading = -(θ_fb + θ_fs) * (1 - coupling * |sin(θ_hs)|)
COUPLING_FACTOR = 0.12  # 经验值，从 MuJoCo 对比中估计


def spine_heading_linear(fb_deg, fs_deg, hs_deg=0.0, hb_deg=0.0):
    """线性模型：Δψ ≈ -(θ_fb + θ_fs)"""
    return -(fb_deg + fs_deg)


def spine_heading_coupled(fb_deg, fs_deg, hs_deg=0.0, hb_deg=0.0):
    """含非线性耦合的 heading 预测"""
    linear = -(fb_deg + fs_deg)
    coupling = COUPLING_FACTOR * abs(np.sin(np.deg2rad(hs_deg)))
    return linear * (1.0 - coupling)


def solve_ik(target_heading_deg):
    """
    逆运动学求解：给定目标 heading Δψ（度，正=左转），求关节角度。

    策略：
        1. 先用 F_spine1（±34.4°），符号与目标一致
        2. 不足部分由 F_body 补偿（±90°）
        3. H_spine1 = H_body = 0
    """
    target = target_heading_deg

    # F_spine1: 取与目标同号，最大 ±34.4°
    fs_sign = np.sign(target) if target != 0 else 1.0
    fs = fs_sign * min(abs(target), JOINT_LIMITS["F_spine1"][1])

    # 剩余由 F_body 补偿
    # target = -(fb + fs) → fb = -target - fs
    remaining = -target - fs

    # 限幅
    fb = np.clip(remaining, *JOINT_LIMITS["F_body"])

    # 如果 F_body 不够，F_spine1 已经最大，剩下无法达成
    if abs(fb) >= JOINT_LIMITS["F_body"][1] * 0.99:
        # F_body 打满，重新计算 F_spine1
        fb = np.sign(remaining) * JOINT_LIMITS["F_body"][1]
        fs = -target - fb
        fs = np.clip(fs, *JOINT_LIMITS["F_spine1"])

    hs = 0.0  # H_spine1 = 0（最大 heading 效率）
    hb = 0.0  # H_body = 0

    achieved = spine_heading_linear(fb, fs, hs, hb)

    return {
        "F_body": fb,
        "F_spine1": fs,
        "H_spine1": hs,
        "H_body": hb,
        "target": target,
        "achieved": achieved,
        "error": achieved - target,
    }


def analyze_reachable_range():
    """分析纯水平 heading 可达范围"""
    print("=" * 70)
    print("SQuRo 脊柱水平转弯能力分析（MuJoCo 修正版）")
    print("=" * 70)

    # F_body + F_spine1 线性叠加
    max_heading = JOINT_LIMITS["F_body"][1] + JOINT_LIMITS["F_spine1"][1]
    print(f"\n线性模型可达 heading 范围：")
    print(f"  F_body (±{JOINT_LIMITS['F_body'][1]:.0f}°) + F_spine1 (±{JOINT_LIMITS['F_spine1'][1]:.1f}°)")
    print(f"  最大右转: -{max_heading:.1f}°")
    print(f"  最大左转: +{max_heading:.1f}°")
    print(f"  总范围:   {2*max_heading:.1f}°")

    print(f"\n非线性耦合修正后：")
    print(f"  H_spine1=0 时保持最大效率")
    print(f"  H_spine1≠0 会使 heading 衰减约 {COUPLING_FACTOR*100:.0f}% × |sin(θ_hs)|")

    print(f"\n分阶段策略：")
    print(f"  |Δψ| ≤ {JOINT_LIMITS['F_spine1'][1]:.0f}°: 仅用 F_spine1")
    print(f"  {JOINT_LIMITS['F_spine1'][1]:.0f}° < |Δψ| ≤ {max_heading:.0f}°: F_spine1 (满) + F_body")


def print_ik_table():
    """打印 IK 对照表"""
    print("\n" + "=" * 70)
    print("逆运动学对照表（修正版）")
    print("=" * 70)
    print(f"{'目标Δψ':>8}  {'F_body':>8}  {'F_spine1':>10}  {'H_spine1':>10}  {'H_body':>8}  {'实际Δψ':>8}")
    print("-" * 60)

    for target in range(-120, 130, 10):
        sol = solve_ik(target)
        marker = " *" if abs(sol["error"]) > 0.5 else ""
        print(f"{sol['target']:>+7.1f}°  {sol['F_body']:>+7.1f}°  {sol['F_spine1']:>+9.1f}°  "
              f"{sol['H_spine1']:>+9.1f}°  {sol['H_body']:>+7.1f}°  {sol['achieved']:>+7.1f}°{marker}")

    print("-" * 60)
    print("* 标记表示超出可达范围，实际 heading 小于目标值")
    print("  H_spine1 和 H_body 始终为 0（最大 heading 效率）")


def plot_heading_map():
    """绘制 F_body × F_spine1 heading 热力图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SQuRo Spine Heading: F_body vs F_spine1 (H_spine1=H_body=0)", fontsize=14)

    fb_range = np.linspace(*JOINT_LIMITS["F_body"], 100)
    fs_range = np.linspace(*JOINT_LIMITS["F_spine1"], 100)
    FB, FS = np.meshgrid(fb_range, fs_range)

    # 线性 heading
    H_linear = spine_heading_linear(FB, FS, 0, 0)

    ax = axes[0]
    levels = np.linspace(-130, 130, 27)
    cs = ax.contourf(FB, FS, H_linear, levels=levels, cmap='RdBu_r', extend='both')
    ax.contour(FB, FS, H_linear, levels=[-34.4, 0, 34.4], colors=['blue', 'green', 'red'],
               linewidths=2, linestyles=['--', '-', '--'])
    ax.set_xlabel("F_body [deg]")
    ax.set_ylabel("F_spine1 [deg]")
    ax.set_title("Heading Δψ = -(F_body + F_spine1)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.colorbar(cs, ax=ax, label='Delta Heading [deg]')

    # 验证：标注 IK 解
    ax = axes[1]
    targets = np.arange(-120, 130, 20)
    for t in targets:
        sol = solve_ik(t)
        ax.plot(sol["F_body"], sol["F_spine1"], 'ko', markersize=8)
        ax.annotate(f'{t}°', (sol["F_body"], sol["F_spine1"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)

    ax.plot([-90, 90], [34.4, 34.4], 'r-', linewidth=1.5, alpha=0.5, label='F_spine1 max')
    ax.plot([-90, 90], [-34.4, -34.4], 'r-', linewidth=1.5, alpha=0.5)
    ax.plot([90, 90], [-34.4, 34.4], 'b-', linewidth=1.5, alpha=0.5, label='F_body max')
    ax.plot([-90, -90], [-34.4, 34.4], 'b-', linewidth=1.5, alpha=0.5)
    ax.set_xlabel("F_body [deg]")
    ax.set_ylabel("F_spine1 [deg]")
    ax.set_title("IK Solutions (F_spine1 first, then F_body)")
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("spine_ik_v2_heading_map.png", dpi=150, bbox_inches='tight')
    print("\nHeading 热力图已保存至 spine_ik_v2_heading_map.png")


def plot_ik_curves():
    """绘制 IK 曲线：目标 heading → 关节角度"""
    targets = np.arange(-130, 135, 5)
    solutions = [solve_ik(t) for t in targets]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Spine IK: Target Heading -> Joint Angles (Corrected Model)", fontsize=14)

    t = [s["target"] for s in solutions]
    fb = [s["F_body"] for s in solutions]
    fs = [s["F_spine1"] for s in solutions]
    achieved = [s["achieved"] for s in solutions]

    ax = axes[0]
    ax.plot(t, fs, 's-', label='F_spine1', linewidth=2, markersize=3)
    ax.plot(t, fb, 'o-', label='F_body', linewidth=2, markersize=3)
    ax.fill_between(t, -34.4, 34.4, alpha=0.1, color='orange',
                    label='F_spine1 limits ±34.4°')
    ax.fill_between(t, -90, 90, alpha=0.05, color='blue',
                    label='F_body limits ±90°')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Target Heading [deg]")
    ax.set_ylabel("Joint Angle [deg]")
    ax.set_title("Joint Angles vs Target Heading")
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, [a - tgt for a, tgt in zip(achieved, t)], 'r-o', linewidth=2, markersize=3)
    ax.fill_between(t, -0.5, 0.5, alpha=0.1, color='green', label='±0.5° tolerance')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Target Heading [deg]")
    ax.set_ylabel("Tracking Error [deg]")
    ax.set_title("IK Tracking Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("spine_ik_v2_curves.png", dpi=150, bbox_inches='tight')
    print("IK 曲线已保存至 spine_ik_v2_curves.png")


def print_model_summary():
    """打印模型总结"""
    print("\n" + "=" * 70)
    print("运动学模型总结")
    print("=" * 70)
    print("""
MuJoCo 验证后的正确运动学模型：

    线性模型:  Δψ ≈ -(θ_F_body + θ_F_spine1)

    关节角色:
        F_spine1 (world -Z, 偏航):  主要 heading 控制，±34.4°
        F_body   (world -Y):       辅助 heading 控制，±90°（等效偏航）
        H_spine1 (world +X, 扭转): 不直接影响 heading（扭转自由度）
        H_body   (world -Y):       不直接影响 heading

    非线性耦合:
        当 H_spine1 ≠ 0 时，F_body 和 F_spine1 的 heading 效率降低
        Δψ ≈ -(θ_fb + θ_fs) × (1 - 0.12 × |sin(θ_hs)|)

    H_spine1 和 H_body 的作用:
        - 调节身体左右倾斜（roll）
        - 在复杂地形中辅助维持平衡
        - 通过耦合间接微调 heading（通常是副作用）
        - 在最大转弯场景下应保持为 0

    关键差异 vs 初始简化模型:
        初始猜测: R = Rx(α)·Rz(β)·Ry(γ)·Rx(δ), 纯水平条件 tan(α)sin(β)=tan(γ)
        MuJoCo实测: 脊柱闭链机构使得 F_body 和 F_spine1 都等效于偏航，
                   H_spine1 是扭转轴而非俯仰轴。
""")


if __name__ == "__main__":
    print_model_summary()
    analyze_reachable_range()
    print_ik_table()
    plot_heading_map()
    plot_ik_curves()
    print("\n分析完成！")
