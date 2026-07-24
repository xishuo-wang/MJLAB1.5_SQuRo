"""
SQuRo 脊柱运动学 MuJoCo 验证 v2
===============================
修正版：先测量零位 F_body↔H_body 相对姿态作为基准，
再计算脊柱关节角度引起的"相对变化"，与简化模型对比。

关键修正：
    1. 零位偏移补偿：ΔR_effect = R_rel(θ) · R_rel(0)⁻¹
    2. 从 MuJoCo 数据反推正确的关节轴和旋转顺序
    3. 逐关节测试，隔离每个关节的效果
"""

import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

XML_PATH = "src/mjlab/asset_zoo/robots/SQuRo/xmls/SQuRo.xml"

SPINE_JOINTS = ["F_body_joint", "F_spine1_joint", "H_spine1_joint", "H_body_joint"]


def quat_to_rotmat(quat):
    """四元数 [w,x,y,z] → 3x3 旋转矩阵"""
    w, x, y, z = quat
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)],
    ])


def heading_from_rotmat(R):
    """从旋转矩阵提取水平面朝向角（第1列=forward在3D中的投影）"""
    return np.arctan2(R[1, 0], R[0, 0])


def pitch_from_rotmat(R):
    """从旋转矩阵提取俯仰角"""
    return np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))


def setup_mujoco():
    """加载模型，获取句柄"""
    mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    mj_data = mujoco.MjData(mj_model)

    # 获取关节 DOF 地址
    joint_dof = {}
    for name in SPINE_JOINTS:
        jid = mj_model.joint(name).id
        joint_dof[name] = mj_model.jnt_dofadr[jid]

    body_ids = {
        "base": mj_model.body("base_Link").id,
        "F_body": mj_model.body("F_body_Link").id,
        "H_body": mj_model.body("H_body_Link").id,
    }

    return mj_model, mj_data, joint_dof, body_ids


def set_spine_angles(mj_model, mj_data, joint_dof, angles_deg):
    """设置脊柱关节角度并运行 FK。angles_deg: [F_body, F_spine1, H_spine1, H_body]"""
    mj_data.qpos[:] = 0.0
    mj_data.qvel[:] = 0.0
    # 固定 base_Link（freejoint: [x,y,z, qw,qx,qy,qz]）
    mj_data.qpos[0:7] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    for i, name in enumerate(SPINE_JOINTS):
        dof = joint_dof[name]
        if dof >= 0:
            mj_data.qpos[dof] = np.deg2rad(angles_deg[i])

    mujoco.mj_forward(mj_model, mj_data)


def get_relative_pose(mj_data, body_ids):
    """获取 F_body → H_body 的相对旋转"""
    R_f = quat_to_rotmat(mj_data.xquat[body_ids["F_body"]])
    R_h = quat_to_rotmat(mj_data.xquat[body_ids["H_body"]])
    R_rel = R_h @ R_f.T  # H_body 在 F_body 系中的姿态
    return R_f, R_h, R_rel


def analyze_zero_config(mj_model, mj_data, joint_dof, body_ids):
    """分析零位配置：各 body 在世界系中的朝向"""
    print("=" * 70)
    print("零位配置分析（所有关节=0，base_Link 固定）")
    print("=" * 70)

    set_spine_angles(mj_model, mj_data, joint_dof, [0, 0, 0, 0])
    R_f, R_h, R_rel_0 = get_relative_pose(mj_data, body_ids)

    print(f"\nF_body_Link 在世界系中的朝向：")
    print(f"  forward (X): [{R_f[0,0]:+.4f}, {R_f[1,0]:+.4f}, {R_f[2,0]:+.4f}]")
    print(f"  left    (Y): [{R_f[0,1]:+.4f}, {R_f[1,1]:+.4f}, {R_f[2,1]:+.4f}]")
    print(f"  up      (Z): [{R_f[0,2]:+.4f}, {R_f[1,2]:+.4f}, {R_f[2,2]:+.4f}]")
    print(f"  world heading: {np.rad2deg(heading_from_rotmat(R_f)):+.2f}°")

    print(f"\nH_body_Link 在世界系中的朝向：")
    print(f"  forward (X): [{R_h[0,0]:+.4f}, {R_h[1,0]:+.4f}, {R_h[2,0]:+.4f}]")
    print(f"  left    (Y): [{R_h[0,1]:+.4f}, {R_h[1,1]:+.4f}, {R_h[2,1]:+.4f}]")
    print(f"  up      (Z): [{R_h[0,2]:+.4f}, {R_h[1,2]:+.4f}, {R_h[2,2]:+.4f}]")
    print(f"  world heading: {np.rad2deg(heading_from_rotmat(R_h)):+.2f}°")

    print(f"\nF_body → H_body 相对旋转（零位基准 R_rel_0）：")
    h0 = heading_from_rotmat(R_rel_0)
    p0 = pitch_from_rotmat(R_rel_0)
    print(f"  relative heading: {np.rad2deg(h0):+.2f}°")
    print(f"  relative pitch:   {np.rad2deg(p0):+.2f}°")
    print(f"  R_rel_0 =")
    print(f"    [{R_rel_0[0,0]:+.4f} {R_rel_0[0,1]:+.4f} {R_rel_0[0,2]:+.4f}]")
    print(f"    [{R_rel_0[1,0]:+.4f} {R_rel_0[1,1]:+.4f} {R_rel_0[1,2]:+.4f}]")
    print(f"    [{R_rel_0[2,0]:+.4f} {R_rel_0[2,1]:+.4f} {R_rel_0[2,2]:+.4f}]")

    # 保存零位基准用于后续对比
    return R_rel_0


def test_single_joint(mj_model, mj_data, joint_dof, body_ids, R_rel_0, joint_idx, angles_test):
    """
    单独测试一个关节：固定其他关节为0，扫描该关节在不同角度下的效果。

    返回 Δheading vs joint_angle 的数据。
    """
    results = []
    base_angles = [0.0, 0.0, 0.0, 0.0]

    for angle in angles_test:
        base_angles[joint_idx] = angle
        set_spine_angles(mj_model, mj_data, joint_dof, base_angles)
        _, _, R_rel = get_relative_pose(mj_data, body_ids)

        # 计算相对于零位的变化
        R_effect = R_rel @ R_rel_0.T  # ΔR = R_rel(θ) · R_rel(0)⁻¹
        h = heading_from_rotmat(R_effect)
        p = pitch_from_rotmat(R_effect)

        results.append({
            "angle": angle,
            "delta_heading": np.rad2deg(h),
            "delta_pitch": np.rad2deg(p),
        })

    return results


def analyze_joint_effects(mj_model, mj_data, joint_dof, body_ids, R_rel_0):
    """逐关节分析：每个关节独立运动时对 heading 的影响"""
    print("\n" + "=" * 70)
    print("逐关节独立效果分析（相对零位的变化）")
    print("=" * 70)

    angles_test = np.linspace(-60, 60, 13)  # 测试角度范围

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Individual Joint Effects on F_body -> H_body Relative Heading", fontsize=14)

    joint_summary = {}

    for i, name in enumerate(SPINE_JOINTS):
        results = test_single_joint(mj_model, mj_data, joint_dof, body_ids, R_rel_0, i, angles_test)
        joint_summary[name] = results

        ang = [r["angle"] for r in results]
        dh = [r["delta_heading"] for r in results]
        dp = [r["delta_pitch"] for r in results]

        # 线性拟合：heading vs angle
        slope, intercept = np.polyfit(ang, dh, 1)

        ax = axes[i // 2, i % 2]
        ax.plot(ang, dh, 'bo-', label=f'delta_heading (slope={slope:.3f})', linewidth=2)
        ax.plot(ang, dp, 'rs--', label='delta_pitch', linewidth=1.5, alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel("Joint Angle [deg]")
        ax.set_ylabel("Effect [deg]")
        ax.set_title(f"{name} (slope={slope:.3f} deg/deg)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        print(f"\n{name}:")
        print(f"  heading 斜率: {slope:.4f} deg/deg (即该关节对 heading 的增益)")
        print(f"  pitch 影响范围: [{min(dp):.1f}°, {max(dp):.1f}°]")

    plt.tight_layout()
    plt.savefig("spine_joint_effects.png", dpi=150, bbox_inches='tight')
    print("\n关节效果图已保存至 spine_joint_effects.png")

    return joint_summary


def compute_joint_world_axes(mj_model, mj_data, joint_dof):
    """计算每个脊柱关节在零位时的世界旋转轴"""
    print("\n" + "=" * 70)
    print("关节世界轴详细分析（零位）")
    print("=" * 70)

    axes_info = {}
    set_spine_angles(mj_model, mj_data, joint_dof, [0, 0, 0, 0])

    for name in SPINE_JOINTS:
        jid = mj_model.joint(name).id
        body_id = mj_model.jnt_bodyid[jid]
        body_name = mj_model.body(body_id).name
        parent_id = mj_model.body_parentid[body_id]

        # body 在世界系中的旋转矩阵
        body_quat = mj_data.xquat[body_id]
        R_body = quat_to_rotmat(body_quat)

        # 关节局部轴（在 body 的局部坐标系中）
        local_axis = mj_model.jnt_axis[jid].copy()

        # 世界轴 = R_body @ local_axis
        world_axis = R_body @ local_axis

        # 同时也计算父 body 的朝向（关节连接的两端）
        parent_quat = mj_data.xquat[parent_id] if parent_id >= 0 else np.array([1.0, 0.0, 0.0, 0.0])
        R_parent = quat_to_rotmat(parent_quat) if parent_id >= 0 else np.eye(3)

        print(f"\n{name}:")
        print(f"  所属 body:      {body_name}")
        print(f"  父 body:        {mj_model.body(parent_id).name if parent_id >= 0 else 'world'}")
        print(f"  局部轴:         [{local_axis[0]:+.4f}, {local_axis[1]:+.4f}, {local_axis[2]:+.4f}]")
        print(f"  世界轴:         [{world_axis[0]:+.4f}, {world_axis[1]:+.4f}, {world_axis[2]:+.4f}]")
        print(f"  世界轴幅值:     {np.linalg.norm(world_axis):.4f}")
        print(f"  Body forward:   [{R_body[0,0]:+.4f}, {R_body[1,0]:+.4f}, {R_body[2,0]:+.4f}]")
        print(f"  Body up:        [{R_body[0,2]:+.4f}, {R_body[1,2]:+.4f}, {R_body[2,2]:+.4f}]")

        axes_info[name] = {
            "world_axis": world_axis,
            "body_name": body_name,
            "R_body": R_body,
        }

    return axes_info


def build_corrected_model(axes_info, joint_summary):
    """
    从 MuJoCo 数据反推正确的简化模型。

    根据关节独立测试的斜率，确定每个关节对 heading 的贡献。
    """
    print("\n" + "=" * 70)
    print("反推正确的简化运动学模型")
    print("=" * 70)

    slopes = {}
    for name in SPINE_JOINTS:
        results = joint_summary[name]
        ang = [r["angle"] for r in results]
        dh = [r["delta_heading"] for r in results]
        slope, _ = np.polyfit(ang, dh, 1)
        slopes[name] = slope

    print("\n各关节对 heading 的线性增益（deg heading / deg joint）：")
    for name, s in slopes.items():
        print(f"  {name}: {s:+.4f}")

    # 解释增益的物理意义
    print("\n增益解释：")
    for name, s in slopes.items():
        if abs(s) > 0.9:
            print(f"  {name}: 增益≈{s:+.1f} → 该关节直接贡献水平面偏航")
        elif abs(s) < 0.1:
            print(f"  {name}: 增益≈{s:+.1f} → 该关节对水平面偏航几乎无贡献（纯扭转轴）")
        else:
            print(f"  {name}: 增益≈{s:+.1f} → 该关节部分贡献偏航（受 body 姿态影响）")

    return slopes


def compare_corrected_models(mj_model, mj_data, joint_dof, body_ids, R_rel_0, slopes):
    """
    在实际任务配置下测试：给定简化模型 IK 产生的角度，
    在 MuJoCo 中测量实际 heading 变化。
    """
    print("\n" + "=" * 70)
    print("多关节组合效果验证")
    print("=" * 70)

    # 线性叠加模型预测: Δψ = Σ(slope_i × θ_i)
    test_configs = [
        ([0, 30, 0, 0], "纯偏航30°"),
        ([0, 0, 30, 0], "纯俯仰30°"),
        ([0, 30, 30, 0], "偏航+俯仰各30°"),
        ([30, 0, 0, 0], "纯扭转30°"),
        ([30, 30, 0, 0], "扭转+偏航"),
        ([30, 0, 30, 0], "扭转+俯仰"),
        ([0, 34.4, 34.4, 0], "最大偏航+俯仰"),
        ([50, 34.4, 0, 0], "大扭转+最大偏航"),
        ([0, 34.4, -34.4, 0], "最大偏航-俯仰"),
        ([-50, -34.4, -34.4, 0], "全最大负向"),
    ]

    print(f"{'配置描述':<22} {'线性模型Δψ':>10} {'MuJoCo Δψ':>10} {'误差':>8} {'MuJoCo pitch':>10}")
    print("-" * 65)

    for angles_deg, desc in test_configs:
        # 线性叠加预测
        linear_pred = sum(slopes[name] * angles_deg[i] for i, name in enumerate(SPINE_JOINTS))

        # MuJoCo 实测
        set_spine_angles(mj_model, mj_data, joint_dof, angles_deg)
        _, _, R_rel = get_relative_pose(mj_data, body_ids)
        R_effect = R_rel @ R_rel_0.T
        h_mj = heading_from_rotmat(R_effect)
        p_mj = pitch_from_rotmat(R_effect)

        err = np.rad2deg(h_mj) - linear_pred

        print(f"{desc:<22} {linear_pred:>+9.2f}° {np.rad2deg(h_mj):>+9.2f}° {err:>+7.2f}° {np.rad2deg(p_mj):>+9.2f}°")

    return test_configs


def plot_corrected_comparison(mj_model, mj_data, joint_dof, body_ids, R_rel_0, slopes):
    """可视化校正后的模型对比"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Corrected Spine Kinematics Model vs MuJoCo", fontsize=14)

    # 1. 随机采样验证
    ax = axes[0]
    np.random.seed(42)
    n_samples = 500

    linear_preds, mujoco_vals = [], []
    for _ in range(n_samples):
        angles_deg = [
            np.random.uniform(-80, 80),       # F_body
            np.random.uniform(-34.4, 34.4),   # F_spine1
            np.random.uniform(-34.4, 34.4),   # H_spine1
            np.random.uniform(-80, 80),       # H_body
        ]

        linear_pred = sum(slopes[name] * angles_deg[i] for i, name in enumerate(SPINE_JOINTS))

        set_spine_angles(mj_model, mj_data, joint_dof, angles_deg)
        _, _, R_rel = get_relative_pose(mj_data, body_ids)
        R_effect = R_rel @ R_rel_0.T
        h_mj = np.rad2deg(heading_from_rotmat(R_effect))

        linear_preds.append(linear_pred)
        mujoco_vals.append(h_mj)

    ax.scatter(linear_preds, mujoco_vals, s=5, alpha=0.5)
    lims = max(abs(np.array(linear_preds + mujoco_vals)).max() * 1.1, 90)
    ax.plot([-lims, lims], [-lims, lims], 'r--', linewidth=1, label='y=x')
    ax.set_xlabel("Linear Model Prediction [deg]")
    ax.set_ylabel("MuJoCo Measurement [deg]")
    ax.set_title(f"Random Sampling (N={n_samples})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 2. 误差分布
    ax = axes[1]
    errs = np.array(mujoco_vals) - np.array(linear_preds)
    ax.hist(errs, bins=50, edgecolor='k', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--')
    ax.axvline(x=np.mean(errs), color='orange', linestyle='-',
               label=f'Mean: {np.mean(errs):.3f}°\nStd: {np.std(errs):.3f}°')
    ax.set_xlabel("Error (MuJoCo - Linear Model) [deg]")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. F_spine1 vs H_spine1 协同效果
    ax = axes[2]
    betas = np.linspace(-34.4, 34.4, 15)
    gammas = np.linspace(-34.4, 34.4, 15)
    BB, GG = np.meshgrid(betas, gammas)
    heading_map = np.zeros_like(BB)

    for i in range(len(betas)):
        for j in range(len(gammas)):
            angles_deg = [0.0, betas[i], gammas[j], 0.0]
            set_spine_angles(mj_model, mj_data, joint_dof, angles_deg)
            _, _, R_rel = get_relative_pose(mj_data, body_ids)
            R_effect = R_rel @ R_rel_0.T
            heading_map[j, i] = np.rad2deg(heading_from_rotmat(R_effect))

    levels = np.linspace(-60, 60, 25)
    cs = ax.contourf(BB, GG, heading_map, levels=levels, cmap='RdBu_r', extend='both')
    ax.set_xlabel("F_spine1 [deg]")
    ax.set_ylabel("H_spine1 [deg]")
    ax.set_title("MuJoCo: Heading from F_spine1 + H_spine1\n(F_body=H_body=0)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.colorbar(cs, ax=ax, label='Delta Heading [deg]')

    plt.tight_layout()
    plt.savefig("spine_corrected_model.png", dpi=150, bbox_inches='tight')
    print("\n校正模型对比图已保存至 spine_corrected_model.png")


def main():
    print("加载 MuJoCo 模型...")
    mj_model, mj_data, joint_dof, body_ids = setup_mujoco()

    # 1. 零位分析
    R_rel_0 = analyze_zero_config(mj_model, mj_data, joint_dof, body_ids)

    # 2. 关节世界轴分析
    axes_info = compute_joint_world_axes(mj_model, mj_data, joint_dof)

    # 3. 逐关节独立效果测试
    joint_summary = analyze_joint_effects(mj_model, mj_data, joint_dof, body_ids, R_rel_0)

    # 4. 反推正确模型
    slopes = build_corrected_model(axes_info, joint_summary)

    # 5. 多关节组合验证
    compare_corrected_models(mj_model, mj_data, joint_dof, body_ids, R_rel_0, slopes)

    # 6. 可视化
    plot_corrected_comparison(mj_model, mj_data, joint_dof, body_ids, R_rel_0, slopes)

    print("\n" + "=" * 70)
    print("验证完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
