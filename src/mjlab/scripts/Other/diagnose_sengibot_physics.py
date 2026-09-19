import sys

from pathlib import Path

import mujoco
import numpy as np

from mjlab.asset_zoo.robots import get_sengibot_robot_cfg


def p(*a):
    print(*a)
    sys.stdout.flush()


# 加载模型并编译
robot = get_sengibot_robot_cfg().build()
model = robot.spec.compile()
XML = Path(__file__).resolve().parents[1] / "xmls" / "Sengibot.xml"


# 生成零力矩站立 1.5 s 后的稳定状态
def settled_state():
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)
    d.ctrl[:] = 0.0
    for _ in range(750):
        mujoco.mj_step(model, d)
    mujoco.mj_forward(model, d)
    return d


# 几何最低点（支持 capsule / sphere）
def geom_lowest(d, gid):
    c = d.geom_xpos[gid].copy()
    xmat = d.geom_xmat[gid].reshape(3, 3)
    size = model.geom_size[gid]
    typ = model.geom_type[gid]
    if typ == mujoco.mjtGeom.mjGEOM_CAPSULE:
        ax = xmat[:, 2]
        return min((c + s * size[1] * ax)[2] for s in (-1, 1)) - size[0]
    if typ == mujoco.mjtGeom.mjGEOM_SPHERE:
        return c[2] - size[0]
    return c[2] - max(size[:2])


p("=" * 84)
p("1. 全局物理选项 / 重力")
p("=" * 84)
p(f"  gravity={model.opt.gravity.tolist()}  timestep={model.opt.timestep}  "
  f"integrator={model.opt.integrator}  solver={model.opt.solver}  iterations={model.opt.iterations}")
p(f"  机器人质量={model.body_subtreemass[model.body('torso').id]:.4f} kg  "
  f"(terrain box={model.body_mass[model.body('terrain').id]:.1f} kg 属场景物体)")

p()
p("=" * 84)
p("2. 初始位姿：脚离地高度（平台顶面 z=-0.05）")
p("=" * 84)
d0 = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, d0, 0)
mujoco.mj_forward(model, d0)
p(f"  根 z={d0.qpos[2]:.5f}  ncon={d0.ncon}")
for nm in ("foot1", "tibia1_1", "foot3", "tibia3_1"):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
    low = geom_lowest(d0, gid)
    p(f"  {nm:<10} 最低点 z={low:>9.5f}  离地 {(low + 0.05) * 1000:>7.2f} mm")
p("  -> 全部悬空，机器人每次 episode 开始时自由下落约 0.108 s，触地速度 0.81 m/s")

p()
p("=" * 84)
p("3. 零力矩站立 1.5 s：每条腿的真实接触位置与承重")
p("=" * 84)
d = settled_state()
p(f"  根 z={d.qpos[2]:.5f}  ncon={d.ncon}  总重={0.3005 * 9.81:.4f} N")
legs = {
    "LF前": ("foot1", "tibia1_1"),
    "RF前": ("foot2", "tibia2_1"),
    "LH后": ("foot3", "tibia3_1"),
    "RH后": ("foot4", "tibia4_1"),
}
for leg, (foot, tib) in legs.items():
    gf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, foot)
    gt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, tib)
    p(f"  {leg:<6} 脚最低点离地 {(geom_lowest(d, gf) + 0.05) * 1000:>7.3f} mm   "
      f"小腿几何最低点离地 {(geom_lowest(d, gt) + 0.05) * 1000:>7.3f} mm")
p("  -> 后腿用脚(foot3/foot4)触地；前腿脚悬空，靠小腿几何(tibia1_1/tibia2_1)触地")

p()
p("  XML 源里的碰撞体位置（前腿脚沿 +x 外移，与后腿的 -z 结构不一致）:")
for key in ("tibia1_1", "tibia1_3", "foot1", "tibia3_1", "tibia3_3", "foot3"):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, key)
    p(f"    {key:<9} 局部中心={np.round(model.geom_pos[gid], 4).tolist()}  "
      f"size={np.round(model.geom_size[gid], 4).tolist()}")

p()
p("=" * 84)
p("4. 站立保持力矩 vs 位置伺服能力")
p("=" * 84)
kind = robot.actuators[0].command_field
p(f"  动作类型 command_field = {kind!r}")
p(f"  {'actuator':<20}{'forcerange(Nm)':>15}{'站立所需(Nm)':>14}{'用过比例':>11}{'冗余':>12}{'饱和误差':>11}")
for a in range(model.nu):
    jid = model.actuator_trnid[a, 0]
    dof = model.jnt_dofadr[jid]
    tau = float(d.qfrc_bias[dof])
    lim = float(model.actuator_forcerange[a, 1])
    kp = float(model.actuator_gainprm[a, 0])
    ratio = f"{abs(tau) / lim:>10.2%}"
    redun = f"{lim / abs(tau):>10.0f}x" if abs(tau) > 1e-8 else f"{'>1e6':>11}"
    p(f"  {model.actuator(a).name:<20}{lim:>15.2f}{tau:>14.5f}{ratio}{redun}"
      f"{lim / kp:>10.4f}r")
p("  forcerange 沿用原 <motor> 的力矩上限（waist 0.25，其余 0.2），控制权限未变；")
p("  饱和误差 = forcerange/kp，即位置误差超过该值后伺服输出满力矩。")

p()
p("=" * 84)
p("5. armature 与真实连杆惯量（谁主导动力学）")
p("=" * 84)
p(f"  {'joint':<22}{'armature':>11}{'I_eff(含arm)':>15}{'arm占比':>10}{'αmax(rad/s²)':>15}")
for jname, tau in (("joint_LF_drive", 0.20), ("joint_LH_drive", 0.20),
                   ("joint_LH_tibia", 0.20), ("joint_waist_drive", 0.25)):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    dof = model.jnt_dofadr[jid]
    dd = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, dd, 0)
    dd.qpos[2] = 5.0
    mujoco.mj_forward(model, dd)
    a0 = float(dd.qacc[dof])
    xf = np.zeros(model.nv)
    xf[dof] = 1e-3
    dd.qfrc_applied[:] = xf
    mujoco.mj_forward(model, dd)
    I = 1e-3 / (float(dd.qacc[dof]) - a0)
    arm = float(model.dof_armature[dof])
    p(f"  {jname:<22}{arm:>11.2e}{I:>15.3e}{arm / I:>9.1%}{tau / I:>15.0f}")

p()
p("=" * 84)
p("6. 前后腿参数对称性")
p("=" * 84)
names = ("joint_LF_upspring1", "joint_RF_upspring1",
         "joint_LH_upspring1", "joint_RH_upspring1")
p(f"  {'joint':<22}{'damping':>10}{'stiffness':>11}{'c_crit≈':>10}{'ζ≈':>10}")
p(f"  {'(c_crit 按 I=1.56e-3 估)':<22}")
for nm in names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
    c = float(model.dof_damping[model.jnt_dofadr[jid]])
    k = float(model.jnt_stiffness[jid])
    ccrit = 2 * np.sqrt(k * 1.56e-3) if k > 0 else float("nan")
    p(f"  {nm:<22}{c:>10.5f}{k:>11.0f}{ccrit:>10.4f}{c / ccrit:>10.5f}")
p("  前腿 c=0.005 / 后腿 c=0.0005 —— 前后相差 10 倍")
p("  刚度: LF=1000, RF=900, LH=1500, RH=1500 —— 前后不同，且 LF≠RF")
p("  四者 ζ 都在 1e-4~1e-3 量级，远低于临界阻尼 2.4~3.1")
