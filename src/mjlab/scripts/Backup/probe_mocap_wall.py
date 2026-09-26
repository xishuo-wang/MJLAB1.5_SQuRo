# 探针: 验证 mjwarp 是否支持"逐环境可移动的 mocap 墙体" (只读, 不接触任务代码)
# 要回答四件事:
#   1) mocap body 是否增加自由度 (墙不应带来 DOF)
#   2) 每个 world 能否写不同的 mocap_pos, 且 geom_xpos 随之改变
#   3) mocap geom 是否真的参与碰撞 (球被挡 / 不被挡)
#   4) 改写 mocap_pos 是否需要重建模型
# 用法: uv run python -B -m mjlab.scripts.Backup.probe_mocap_wall
import mujoco
import mujoco_warp as mjwarp
import numpy as np

# 无重力、无地面: 球在空中以恒定速度直飞, 只有墙可能与它接触, 避免地面接触干扰判读。
# 墙做成上下贯通 (z 半长 0.5), 所以球的高度不影响结论。
XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="wall" mocap="true" pos="0 0 0">
      <geom name="wall_geom" type="box" size="0.01 0.3 0.5"
            contype="1" conaffinity="1" rgba="0.6 0.6 0.6 1"/>
    </body>
    <body name="ball" pos="0 0 0.3">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.02" density="500"
            contype="1" conaffinity="1" rgba="1 0.4 0.4 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main() -> None:
    mjm = mujoco.MjModel.from_xml_string(XML)
    mjd = mujoco.MjData(mjm)
    wall_gid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "wall_geom")

    print("=" * 88)
    print(f"[1] 模型: nbody={mjm.nbody} ngeom={mjm.ngeom} nmocap={mjm.nmocap} "
          f"nq={mjm.nq} nv={mjm.nv}")
    print(f"    墙 body 的 mocapid = {mjm.body_mocapid[mjm.geom_bodyid[wall_gid]]} "
          f"(>=0 表示是 mocap body)")
    print(f"    nq/nv 只来自球的 freejoint (7/6) ⇒ mocap 墙不贡献自由度: "
          f"{mjm.nq == 7 and mjm.nv == 6}")

    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd, nworld=2)
    print(f"\n[2] warp data: mocap_pos.shape={d.mocap_pos.shape}  qpos.shape={d.qpos.shape}")

    # world 0: 墙放在 +0.05 (应该挡住球); world 1: 墙挪到 -5 (球应畅通)
    wall = np.zeros((2, mjm.nmocap, 3), dtype=np.float32)
    wall[0, 0] = (0.05, 0.0, 0.0)
    wall[1, 0] = (-5.0, 0.0, 0.0)
    d.mocap_pos.assign(wall)

    # 球初态: x=-0.2, z=0.3, vx=+1.0 m/s
    qpos = np.zeros((2, mjm.nq), dtype=np.float32)
    qpos[:, 0] = -0.2
    qpos[:, 2] = 0.3
    qpos[:, 3] = 1.0
    qvel = np.zeros((2, mjm.nv), dtype=np.float32)
    qvel[:, 0] = 1.0
    d.qpos.assign(qpos)
    d.qvel.assign(qvel)

    mjwarp.forward(m, d)
    xpos0 = d.geom_xpos.numpy()[:, wall_gid]
    print("\n[3] 写入 mocap_pos 后, 墙体 geom 的世界位置 (每 world 独立):")
    for w in range(2):
        print(f"    world {w}: mocap_pos={wall[w, 0]}  ->  geom_xpos={xpos0[w]}")
    ok_pose = all(np.allclose(xpos0[w], wall[w, 0], atol=1e-6) for w in range(2))
    print(f"    逐环境墙位生效: {ok_pose}")

    # 用"到达过的最大 x"判读: 无阻挡时应到 +0.8; 有墙时不得越过墙内壁 0.04。
    # 注意判据不能定成"停在 0.02": 默认接触刚度 (solref 未显式给) 下球会压进去几毫米。
    xmax = np.full(2, -np.inf, dtype=np.float64)
    for _ in range(500):                       # 1.0 s
        mjwarp.step(m, d)
        x = d.qpos.numpy()[:, 0]
        xmax = np.maximum(xmax, x)
    print(f"\n[4] 走 1.0 s: 球心最大 x  world0(墙在+0.05)={xmax[0]:+.4f}  "
          f"world1(墙在-5)={xmax[1]:+.4f}")
    blocked = bool(xmax[0] < 0.04)      # 未越过墙内壁 (0.05 - 0.01)
    passed = bool(xmax[1] > 0.50)       # 无阻挡, 一路走到底
    print(f"    world0 被墙挡住 (未越过内壁 0.04): {blocked}   world1 畅通 (超过 0.5): {passed}")
    print(f"    接触压入深度 = (最大 x + 球半径) - 内壁 0.0400 = "
          f"{(xmax[0] + 0.02) - 0.04:+.4f} m  (默认 solref 的软接触; 若墙需要更硬, "
          f"给墙 geom 显式 solref/solimp)")

    # 运行期再改一次墙位, 检查不需要重建模型
    wall[1, 0] = (0.05, 0.0, 0.0)
    d.mocap_pos.assign(wall)
    mjwarp.forward(m, d)
    xpos1 = d.geom_xpos.numpy()[:, wall_gid]
    ok_move = bool(np.allclose(xpos1[1], wall[1, 0], atol=1e-6))
    print(f"\n[5] 运行期改写墙位 (未重建模型): world1 geom_xpos={xpos1[1]}  生效={ok_move}")

    allok = ok_pose and blocked and passed and ok_move
    print("\n结论: " + ("四项全部通过 ⇒ mjwarp 支持逐环境可移动 mocap 墙体"
                        if allok else "存在未通过项"))


if __name__ == "__main__":
    main()
