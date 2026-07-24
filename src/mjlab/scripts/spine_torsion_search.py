"""
SQuRo 脊柱扭转角搜索
====================
对于给定的偏航/俯仰关节角度（F_spine1 = H_spine1），
搜索 F_body 和 H_body 扭转角，使 F_body_Link 和 H_body_Link 的滚转角为 0。

物理条件：保持重力，free base，开启碰撞，腿部置零。

坐标系修正：
    F_body: canonical (fwd,left,up) = (body_X, -body_Z, body_Y)
    H_body: canonical (fwd,left,up) = (-body_X, -body_Z, -body_Y)
    滚转角 roll = atan2(R_canonical[2,1], R_canonical[2,2])
"""

import sys
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize, Bounds

# 尝试中文字体
try:
    matplotlib.font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
except Exception:
    pass
plt.rcParams['axes.unicode_minus'] = False

XML_PATH = Path("src/mjlab/asset_zoo/robots/SQuRo/xmls/SQuRo.xml")

# 脊柱关节名称
SPINE_JOINTS = ["F_body_joint", "F_spine1_joint", "H_spine1_joint", "H_body_joint"]

# F_body→canonical 修正矩阵
R_CORR_F = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
# H_body→canonical 修正矩阵
R_CORR_H = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float64)

# 仿真参数
SETTLE_STEPS = 2000       # 稳定步数（2秒@0.001步长）
CHECK_EVERY = 50
STABLE_CHECKS = 4
POS_TOL = 1e-4


def quat_to_rotmat(quat):
    """四元数 [w,x,y,z] → 3x3 旋转矩阵"""
    w, x, y, z = quat
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)],
    ])


def get_body_roll(mj_data, body_name, debug=False):
    """提取指定 body 的滚转角（canonical frame, 弧度）"""
    body_id = mujoco.mj_name2id(mj_data.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    R_body = quat_to_rotmat(mj_data.xquat[body_id])

    # 获取 body 在世界系中的各轴朝向
    body_fwd = R_body[:, 0]   # 局部X在世界系
    body_y   = R_body[:, 1]   # 局部Y在世界系
    body_z   = R_body[:, 2]   # 局部Z在世界系

    # 判断哪个 body 轴最接近世界 +X（物理前方）
    dots_x = [abs(np.dot(R_body[:, i], [1, 0, 0])) for i in range(3)]
    idx_fwd = np.argmax(dots_x)
    sign_fwd = 1.0 if np.dot(R_body[:, idx_fwd], [1, 0, 0]) > 0 else -1.0

    # 判断哪个 body 轴最接近世界 +Z（物理上方）
    dots_z = [abs(np.dot(R_body[:, i], [0, 0, 1])) for i in range(3)]
    idx_up = np.argmax(dots_z)
    sign_up = 1.0 if np.dot(R_body[:, idx_up], [0, 0, 1]) > 0 else -1.0

    # 剩余的轴就是 left 方向
    remaining = list({0, 1, 2} - {idx_fwd, idx_up})[0]
    # left 方向应接近世界 +Y
    sign_left = 1.0 if np.dot(R_body[:, remaining], [0, 1, 0]) > 0 else -1.0

    # 构建 物理(fwd,left,up) → body 的映射
    phys_to_body = np.zeros((3, 3))
    phys_to_body[idx_fwd, 0] = sign_fwd
    phys_to_body[remaining, 1] = sign_left
    phys_to_body[idx_up, 2] = sign_up

    # R_physical = R_body @ phys_to_body（物理坐标系在世界系中的旋转矩阵）
    R_physical = R_body @ phys_to_body

    if debug:
        print(f"\n  [{body_name}] 诊断:")
        print(f"    R_body columns in world:")
        print(f"      X (fwd): [{R_body[0,0]:+.3f}, {R_body[1,0]:+.3f}, {R_body[2,0]:+.3f}]")
        print(f"      Y:       [{R_body[0,1]:+.3f}, {R_body[1,1]:+.3f}, {R_body[2,1]:+.3f}]")
        print(f"      Z:       [{R_body[0,2]:+.3f}, {R_body[1,2]:+.3f}, {R_body[2,2]:+.3f}]")
        print(f"    物理轴映射: fwd=body_axis{idx_fwd}(sign={sign_fwd:+.0f}), "
              f"left=body_axis{remaining}(sign={sign_left:+.0f}), "
              f"up=body_axis{idx_up}(sign={sign_up:+.0f})")
        print(f"    phys_to_body =")
        for row in phys_to_body:
            print(f"      [{row[0]:+.0f}, {row[1]:+.0f}, {row[2]:+.0f}]")
        print(f"    R_physical =")
        for row in R_physical:
            print(f"      [{row[0]:+.3f}, {row[1]:+.3f}, {row[2]:+.3f}]")

    # ZYX Euler: roll = atan2(R[2,1], R[2,2])
    roll = np.arctan2(R_physical[2, 1], R_physical[2, 2])
    return roll


def build_model():
    """加载 MuJoCo 模型，保持原始配置（含地板、重力、freejoint）"""
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    return model, data


def reset_and_set_spine(model, data, fb, fs, hs, hb):
    """重置仿真，设置脊柱角度，所有腿部关节置零"""
    mujoco.mj_resetData(model, data)

    # 所有关节置零（qpos 全部归零，包括 freejoint 和所有 hinge）
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    # freejoint 初始位置：放在原点上方，无旋转
    # freejoint qpos: [x, y, z, qw, qx, qy, qz]
    data.qpos[0:7] = [0.0, 0.0, 0.085, 1.0, 0.0, 0.0, 0.0]

    # 设置脊柱关节角度
    for name, angle in zip(SPINE_JOINTS, [fb, fs, hs, hb]):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            dof_addr = model.jnt_dofadr[jid]
            if dof_addr >= 0:
                data.qpos[dof_addr] = angle

    mujoco.mj_forward(model, data)


def run_settle(model, data, verbose=False):
    """运行仿真至稳定，返回最终是否收敛"""
    prev_base_pos = data.xpos[0].copy()

    for i in range(1, SETTLE_STEPS + 1):
        mujoco.mj_step(model, data)

        if i % CHECK_EVERY == 0:
            cur_base_pos = data.xpos[0]
            disp = float(np.linalg.norm(cur_base_pos - prev_base_pos))
            prev_base_pos = cur_base_pos.copy()

            if disp < POS_TOL and i > 500:
                if verbose:
                    print(f"  稳定于 step {i}, disp={disp:.6f}")
                return True

    return False


def evaluate_rolls(fb, hb, fs_hs, model, data):
    """
    评估函数：给定脊柱角度，返回两个 body 的滚转角绝对值之和。

    Args:
        fb, hb: F_body, H_body 扭转角 [rad]
        fs_hs:  F_spine1 = H_spine1 的值 [rad]

    Returns:
        total_abs_roll: |roll_F| + |roll_H|
    """
    fs = fs_hs
    hs = fs_hs
    reset_and_set_spine(model, data, fb, fs, hs, hb)
    run_settle(model, data)

    roll_f = get_body_roll(data, "F_body_Link")
    roll_h = get_body_roll(data, "H_body_Link")

    return abs(roll_f) + abs(roll_h)


def search_torsion(fs_hs, model, data, fb_init, hb_init, verbose=True):
    """
    对于给定的 F_spine1=H_spine1=fs_hs，搜索最优 (F_body, H_body)。

    Args:
        fs_hs:    偏航/俯仰角度 [rad]
        fb_init:  F_body 初始猜测 [rad]
        hb_init:  H_body 初始猜测 [rad]

    Returns:
        dict with optimal angles and roll values
    """
    bounds = Bounds([-1.57, -1.57], [1.57, 1.57])

    def objective(x):
        fb, hb = x
        return evaluate_rolls(fb, hb, fs_hs, model, data)

    if verbose:
        init_val = objective([fb_init, hb_init])
        print(f"  fs=hs={np.rad2deg(fs_hs):.0f}°: 初始 (fb={np.rad2deg(fb_init):.1f}°, "
              f"hb={np.rad2deg(hb_init):.1f}°) → cost={np.rad2deg(init_val):.2f}°")

    # Nelder-Mead 优化（不需要梯度，适合含噪声的目标函数）
    res = minimize(
        objective,
        x0=[fb_init, hb_init],
        method='Nelder-Mead',
        bounds=bounds,
        options={
            'xatol': 0.005,       # 角度容差 ~0.3°
            'fatol': 0.001,       # cost 容差 ~0.06°
            'maxiter': 100,
            'maxfev': 200,
        }
    )

    fb_opt, hb_opt = res.x
    final_cost = objective([fb_opt, hb_opt])

    # 单独获取最终滚转角
    reset_and_set_spine(model, data, fb_opt, fs_hs, fs_hs, hb_opt)
    run_settle(model, data)
    roll_f = get_body_roll(data, "F_body_Link")
    roll_h = get_body_roll(data, "H_body_Link")

    # 获取 body 位置和朝向用于曲率计算
    f_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "F_body_Link")
    h_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "H_body_Link")
    f_pos = data.xpos[f_id].copy()
    h_pos = data.xpos[h_id].copy()
    R_f = quat_to_rotmat(data.xquat[f_id])
    R_h = quat_to_rotmat(data.xquat[h_id])

    result = {
        "fs_hs": fs_hs,
        "fb_opt": fb_opt,
        "hb_opt": hb_opt,
        "roll_f": roll_f,
        "roll_h": roll_h,
        "final_cost": final_cost,
        "f_pos": f_pos,
        "h_pos": h_pos,
        "R_f": R_f,
        "R_h": R_h,
        "success": res.success,
        "nfev": res.nfev,
    }

    if verbose:
        print(f"           最优 (fb={np.rad2deg(fb_opt):.1f}°, hb={np.rad2deg(hb_opt):.1f}°) "
              f"→ roll_F={np.rad2deg(roll_f):.2f}°, roll_H={np.rad2deg(roll_h):.2f}° "
              f"(cost={np.rad2deg(final_cost):.2f}°, nfev={res.nfev})")

    return result


def compute_curvature(result):
    """
    从 F_body 和 H_body 的位姿计算水平面转弯曲率。

    方法1（Y轴交点）：当两个 body 滚转角为0时，canonical Y轴在水平面内，
    交点即为转弯中心。但对噪声敏感。

    方法2（heading差）：κ = Δψ / d，其中 Δψ 为两 body heading 差，
    d 为两 body 在水平面的距离。更稳健。

    Returns:
        curvature: 曲率 [1/m]，正=左转
        delta_heading: F→H heading差 [rad]
        center: 转弯中心在水平面的坐标 [x, y]
    """
    R_f_canon = result["R_f"] @ R_CORR_F
    R_h_canon = result["R_h"] @ R_CORR_H

    f_pos = result["f_pos"].copy()
    h_pos = result["h_pos"].copy()

    # ---- 方法2：heading 差 ----
    heading_f = np.arctan2(R_f_canon[1, 0], R_f_canon[0, 0])
    heading_h = np.arctan2(R_h_canon[1, 0], R_h_canon[0, 0])
    delta_heading = heading_h - heading_f

    # 两 body 水平距离
    d = np.linalg.norm(h_pos[:2] - f_pos[:2])
    curv_heading = delta_heading / d if d > 0.001 else 0.0

    # ---- 方法1：Y轴交点（作为参考）----
    f_left = R_f_canon[:, 1].copy()
    h_left = R_h_canon[:, 1].copy()
    f_pos_2d = f_pos[:2]
    h_pos_2d = h_pos[:2]
    f_left_2d = f_left[:2]
    h_left_2d = h_left[:2]

    A = np.column_stack([f_left_2d, -h_left_2d])
    b = h_pos_2d - f_pos_2d

    try:
        ts = np.linalg.solve(A, b)
        t = ts[0]
        center_2d = f_pos_2d + t * f_left_2d

        R_turn = np.linalg.norm(f_pos_2d - center_2d)
        if R_turn < 1e-6:
            curv_intersect = 0.0
        else:
            to_center = center_2d - f_pos_2d
            sign = np.sign(np.dot(to_center, f_left_2d))
            curv_intersect = sign / R_turn
    except np.linalg.LinAlgError:
        curv_intersect = 0.0
        center_2d = np.array([np.inf, np.inf])

    return curv_heading, curv_intersect, delta_heading, d, center_2d


def main():
    print("=" * 70)
    print("SQuRo 脊柱扭转角搜索（保持重力 + 自由基座 + 碰撞）")
    print("=" * 70)

    model, data = build_model()
    print(f"模型加载成功: DOF={model.nv}, 关节数={model.njnt}")

    # ================================================================
    # 诊断：检查零位时 body 的实际朝向
    # ================================================================
    print("\n--- 诊断：零位配置下的 body 朝向 ---")
    reset_and_set_spine(model, data, 0.0, 0.0, 0.0, 0.0)
    print("初始 qpos 设置后 (mj_forward):")
    get_body_roll(data, "F_body_Link", debug=True)
    get_body_roll(data, "H_body_Link", debug=True)

    print("\n重力沉降后:")
    run_settle(model, data, verbose=True)
    roll_f = get_body_roll(data, "F_body_Link", debug=True)
    roll_h = get_body_roll(data, "H_body_Link", debug=True)
    print(f"  roll_F = {np.rad2deg(roll_f):.2f}°, roll_H = {np.rad2deg(roll_h):.2f}°")

    f_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "F_body_Link")
    h_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "H_body_Link")
    print(f"  F_body pos: {data.xpos[f_id]}")
    print(f"  H_body pos: {data.xpos[h_id]}")
    print(f"  base pos:   {data.xpos[0]}")
    print("-" * 70)

    # F_spine1 = H_spine1 扫描范围
    fs_hs_values = np.arange(0.0, 0.65, 0.1)  # [0.0, 0.1, ..., 0.6]
    print(f"\nF_spine1 = H_spine1 扫描: {[f'{np.rad2deg(v):.0f}°' for v in fs_hs_values]}")

    # 初始猜测：基于诊断结果修正
    # fs=hs=0.0: fb=0.0, hb=0.0（零位时不需要扭转，诊断确认 roll≈0）
    # fs=hs=0.6: fb=-0.8, hb=-0.7（用户提供的最大弯曲参考值）
    fb_init_start, fb_init_end = 0.0, -0.8
    hb_init_start, hb_init_end = 0.0, -0.7

    results = []
    print("\n开始搜索...")
    print("-" * 70)

    for fs_hs in fs_hs_values:
        # 线性插值初始猜测
        t = fs_hs / 0.6 if fs_hs_values[-1] > 0 else 0.0
        fb_init = fb_init_start + t * (fb_init_end - fb_init_start)
        hb_init = hb_init_start + t * (hb_init_end - hb_init_start)

        result = search_torsion(fs_hs, model, data, fb_init, hb_init, verbose=True)
        results.append(result)

    # ================================================================
    # 结果汇总
    # ================================================================
    print("\n" + "=" * 70)
    print("搜索结果汇总")
    print("=" * 70)
    print(f"{'fs=hs':>8}  {'F_body':>8}  {'H_body':>8}  {'roll_F':>8}  {'roll_H':>8}  {'cost':>8}  {'Δψ':>8}  {'κ_head':>8}")
    print("-" * 75)

    for r in results:
        curv_h, curv_i, delta_h, dist, center = compute_curvature(r)
        r["curvature_heading"] = curv_h
        r["curvature_intersect"] = curv_i
        r["delta_heading"] = delta_h
        r["body_distance"] = dist
        r["center"] = center
        print(f"{np.rad2deg(r['fs_hs']):>+7.1f}°  {np.rad2deg(r['fb_opt']):>+7.1f}°  "
              f"{np.rad2deg(r['hb_opt']):>+7.1f}°  {np.rad2deg(r['roll_f']):>+7.2f}°  "
              f"{np.rad2deg(r['roll_h']):>+7.2f}°  {np.rad2deg(r['final_cost']):>+7.2f}°  "
              f"{np.rad2deg(delta_h):>+7.2f}°  {curv_h:>+7.2f}")

    # ================================================================
    # 可视化
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Spine Torsion Search Results (Gravity ON, Free Base)", fontsize=14)

    fs_deg = [np.rad2deg(r["fs_hs"]) for r in results]
    fb_opt = [np.rad2deg(r["fb_opt"]) for r in results]
    hb_opt = [np.rad2deg(r["hb_opt"]) for r in results]
    roll_f = [np.rad2deg(r["roll_f"]) for r in results]
    roll_h = [np.rad2deg(r["roll_h"]) for r in results]
    curvatures = [r.get("curvature_heading", 0) for r in results]
    delta_headings = [np.rad2deg(r.get("delta_heading", 0)) for r in results]

    # 1. 扭转角 vs 偏航/俯仰角
    ax = axes[0, 0]
    ax.plot(fs_deg, fb_opt, 'bo-', label='F_body (torsion)', linewidth=2, markersize=8)
    ax.plot(fs_deg, hb_opt, 'rs-', label='H_body (torsion)', linewidth=2, markersize=8)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("F_spine1 = H_spine1 [deg]")
    ax.set_ylabel("Optimal Torsion Angle [deg]")
    ax.set_title("Torsion Angles vs Spine Bend")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 残余滚转角
    ax = axes[0, 1]
    ax.plot(fs_deg, roll_f, 'bo-', label='F_body roll', linewidth=2, markersize=8)
    ax.plot(fs_deg, roll_h, 'rs-', label='H_body roll', linewidth=2, markersize=8)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("F_spine1 = H_spine1 [deg]")
    ax.set_ylabel("Residual Roll [deg]")
    ax.set_title("Residual Roll after Optimization")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 曲率 vs 偏航/俯仰角
    ax = axes[1, 0]
    ax.plot(fs_deg, delta_headings, 'mo-', label='Delta Heading [deg]', linewidth=2, markersize=8)
    ax.plot(fs_deg, curvatures, 'go-', label='Curvature (heading/d) [1/m]', linewidth=2, markersize=8)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("F_spine1 = H_spine1 [deg]")
    ax.set_ylabel("Value")
    ax.set_title("Delta Heading & Curvature")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. 转弯中心位置
    ax = axes[1, 1]
    for i, r in enumerate(results):
        center = r.get("center", np.array([np.nan, np.nan]))
        if np.isfinite(center[0]):
            ax.plot(center[0], center[1], 'o', markersize=10,
                    label=f'{fs_deg[i]:.0f}°' if i % 2 == 0 else '')
    # 标注 F_body 和 H_body 的平均位置
    f_positions = np.array([r["f_pos"][:2] for r in results])
    ax.plot(f_positions[:, 0], f_positions[:, 1], 'k+', markersize=12, label='F_body pos')
    ax.set_xlabel("World X [m]")
    ax.set_ylabel("World Y [m]")
    ax.set_title("Turn Centers (Y-axis intersection)")
    ax.set_aspect('equal')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("spine_torsion_search_results.png", dpi=150, bbox_inches='tight')
    print("\n结果图已保存至 spine_torsion_search_results.png")

    # ================================================================
    # 曲率-关节角映射总结
    # ================================================================
    print("\n" + "=" * 70)
    print("曲率 → 脊柱关节角映射")
    print("=" * 70)
    print(f"{'κ_head[1/m]':>12}  {'F_body':>8}  {'F_spine1':>10}  {'H_spine1':>10}  {'H_body':>8}  {'Δψ[deg]':>9}")
    print("-" * 70)
    for r in results:
        print(f"{r['curvature_heading']:>+11.2f}  {np.rad2deg(r['fb_opt']):>+7.1f}°  "
              f"{np.rad2deg(r['fs_hs']):>+9.1f}°  {np.rad2deg(r['fs_hs']):>+9.1f}°  "
              f"{np.rad2deg(r['hb_opt']):>+7.1f}°  {np.rad2deg(r['delta_heading']):>+8.2f}°")

    print("\n完成！")
    return results


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
