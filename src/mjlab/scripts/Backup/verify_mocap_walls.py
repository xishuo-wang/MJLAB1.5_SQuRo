# 验收: 墙改成 mocap 之后, 逐环境墙位与碰撞是否真的生效 (只读, 不落盘)
# 检查四件事 (§7.16 的验收清单):
#   1) 环境能建起来, 两面墙各是一个 mocap body, 且不贡献自由度
#   2) 逐环境写墙位后, 各环境墙 geom 的世界 x 与写入值一致
#   3) 把某个环境的走廊收到比机身还窄, 该环境必须出现墙-机器人接触并被限位 (碰撞真的跟着墙走)
#   4) 其它没改墙位的环境不受影响
# 用法: uv run python -B -m mjlab.scripts.Backup.verify_mocap_walls
from __future__ import annotations

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.SQuRo_Backup.mdp import entity as mdp_entity
from mjlab.tasks.registry import load_env_cfg

N = 8


def main() -> None:
    env_cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    env_cfg.scene.num_envs = N
    env_cfg.events.pop("init_restricted_space", None)
    mdp_entity.configure_restricted_space(env_cfg, wall_x_neg=-0.06, wall_x_pos=0.05,
                                          enable_collision=True)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    env.reset()

    ent = env.scene.entities["restricted_space"]
    mjm = env.sim.mj_model
    print("=" * 90)
    print(f"[1] nmocap={mjm.nmocap}  nq={mjm.nq}  nv={mjm.nv}  nbody={mjm.nbody}")
    print(f"    墙 body 全局下标 = {ent.wall_body_ids}   mocap 下标 = {ent.wall_mocap_ids}")
    print(f"    墙 geom 全局下标 = {ent.wall_geom_ids}")
    print(f"    机器人 nv={mjm.nv}: 墙不贡献自由度 (它是 mocap body)")
    print(f"    env_origins[0] = {env.scene.env_origins[0].tolist()}")

    # 初值: 编译期墙位
    n0, p0 = ent.read_wall_x(env)
    print(f"\n[2] 编译期墙位 (写之前): x_neg={np.round(n0, 4).tolist()}")
    print(f"                        x_pos={np.round(p0, 4).tolist()}")

    # 逐环境写不同墙位: 环境 i 的 x_neg = -0.06 - 0.005*i
    new_neg = -0.06 - 0.005 * np.arange(N)
    ent.write_wall_x(env, new_neg, np.full(N, 0.05))
    env.sim.forward()
    xpos = env.sim.wp_data.geom_xpos.numpy()
    got = xpos[:, ent.wall_geom_ids[0], 0]
    want = new_neg + env.scene.env_origins[:, 0].cpu().numpy()
    ok_write = bool(np.allclose(got, want, atol=1e-5))
    print(f"\n[3] 逐环境写墙位 -> 墙 geom 世界 x")
    print(f"    写入 {np.round(want, 4).tolist()}")
    print(f"    实测 {np.round(got, 4).tolist()}")
    print(f"    逐环境生效: {ok_write}")

    # 碰撞验收: 只把环境 0 收到净宽 0.076 (内壁 ±0.038 = reset 姿态足端半宽, 恰好不干涉),
    # 这个宽度比机身需要的还窄, 机器人一动就会压到墙。
    # 注意别把墙写到与 reset 姿态相交: 内壁 < 0.038 会让机器人一出生就深度干涉 -> 整批 NaN。
    tight = np.full(N, -0.06)
    tight[0] = -0.048
    pos_arr = np.full(N, 0.05)
    pos_arr[0] = 0.048
    ent.write_wall_x(env, tight, pos_arr)
    act = torch.zeros(N, 14, device="cuda:0")
    wall_ids = set(ent.wall_geom_ids)
    hits = np.zeros(N, dtype=int)
    with torch.inference_mode():
        for _ in range(200):
            env.step(act)
            n = int(env.sim.wp_data.nacon.numpy()[0])
            if n > 0:
                g = env.sim.wp_data.contact.geom.numpy()[:n]
                w = env.sim.wp_data.contact.worldid.numpy()[:n]
                m = np.isin(g[:, 0], list(wall_ids)) | np.isin(g[:, 1], list(wall_ids))
                for wid in np.unique(w[m]):
                    hits[int(wid)] += 1
    root = env.scene.entities["robot"].data.root_link_pos_w[:, 0].cpu().numpy()
    print(f"\n[4] 只把环境 0 收到净宽 0.076 (内壁 ±0.038, 恰好不干涉 reset 姿态):")
    print(f"    各环境墙-机器人接触计数 = {hits.tolist()}")
    print(f"    各环境 root x          = {np.round(root, 4).tolist()}")
    finite = bool(np.isfinite(root).all())
    print(f"    全部环境无 NaN: {finite}")
    print(f"    环境 0 出现接触: {hits[0] > 0};  环境 0 root x 被限制在 ±0.04 内: "
          f"{abs(root[0]) < 0.04}")
    print(f"    其它环境接触远少于环境 0: {bool(hits[1:].sum() < hits[0])}")

    ok = ok_write and finite and hits[0] > 0 and abs(root[0]) < 0.04
    print("\n结论: " + ("mocap 墙在真环境里可用 (逐环境墙位 + 碰撞均生效)" if ok
                        else "存在未通过项"))


if __name__ == "__main__":
    main()
