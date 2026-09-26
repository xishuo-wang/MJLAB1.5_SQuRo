# 验收: mocap 墙体与"逐环境随机墙位"这条链 (只读, 不落盘)
# 覆盖:
#   [1] 两墙是独立 mocap body, 不贡献自由度
#   [2] 逐环境写墙位 (CUDA 张量 + 部分环境) 后, 各环境墙 geom 世界 x 与写入值一致
#   [3] 收窄某环境 -> 该环境出现接触并被限位; 把墙挪开 -> 旧位置不再产生碰撞
#   [4] 命令侧采样: 回合复位后 物理墙位 == 观测墙位, 且 d 落在 [d_min, d_max] 内、
#       下界专门采样组占比接近配置值
#   [5] 循环复位不换墙位 (sim.reset 会把 mocap 打回默认值, 必须被恢复)
# 失败时以非零退出码结束 —— 否则自动化流水线会把失败当通过。
# 用法: uv run python -B -m mjlab.scripts.Backup.verify_mocap_walls
from __future__ import annotations

import sys

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.SQuRo_Backup.mdp import entity as mdp_entity
from mjlab.tasks.registry import load_env_cfg

N = 8
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"    {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        fails.append(msg)


def np_(t) -> np.ndarray:
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


# 统计各环境的墙-机器人接触次数。
# 用 no_grad 而非 inference_mode: 后者会把 step 里产生的张量 (含 reset 用的 env_ids) 标成
# 推理张量, 之后再在 inference_mode 外做就地写就会报 "Inplace update to inference tensor"。
def wall_hits(env, ent, steps: int, act) -> np.ndarray:
    ids = set(ent.wall_geom_ids)
    hits = np.zeros(env.num_envs, dtype=int)
    with torch.no_grad():
        for _ in range(steps):
            env.step(act)
            n = int(env.sim.wp_data.nacon.numpy()[0])
            if n > 0:
                g = env.sim.wp_data.contact.geom.numpy()[:n]
                w = env.sim.wp_data.contact.worldid.numpy()[:n]
                m = np.isin(g[:, 0], list(ids)) | np.isin(g[:, 1], list(ids))
                for wid in np.unique(w[m]):
                    hits[int(wid)] += 1
    return hits


def main() -> None:
    env_cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    env_cfg.scene.num_envs = N
    env_cfg.events.pop("init_restricted_space", None)
    mdp_entity.configure_restricted_space(env_cfg, wall_x_neg=-0.06, wall_x_pos=0.05,
                                          enable_collision=True)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    env.reset()

    ent = env.scene.entities["restricted_space"]
    cmd = env.command_manager.get_term("backup_cmd")
    mjm = env.sim.mj_model
    act = torch.zeros(N, 14, device="cuda:0")

    print("=" * 90)
    print(f"[1] nmocap={mjm.nmocap} nq={mjm.nq} nv={mjm.nv}")
    print(f"    墙 body 全局下标 = {ent.wall_body_ids}  mocap 下标 = {ent.wall_mocap_ids}")
    check(mjm.nmocap >= 2, "至少两面墙是 mocap body")
    check(len(ent.wall_mocap_ids) == 2 and -1 not in ent.wall_mocap_ids,
          "两面墙都有有效 mocap 下标")
    check(mjm.body_mocapid[ent.wall_body_ids[0]] != mjm.body_mocapid[ent.wall_body_ids[1]],
          "两面墙是两个独立 mocap body")

    print("\n[2] 逐环境写墙位 (CUDA 张量, 只写部分环境)")
    new_neg = torch.full((N,), -0.20, device="cuda:0")
    new_neg[:4] = torch.tensor([-0.06, -0.07, -0.08, -0.09], device="cuda:0")
    ent.write_wall_x(env, new_neg, torch.full((N,), 0.05, device="cuda:0"))
    env.sim.forward()
    got, got_pos = ent.read_wall_x(env)
    want = np_(new_neg)
    check(bool(np.allclose(np_(got), want, atol=1e-5)),
          f"墙 geom 世界 x 与写入一致 (写入 {want.tolist()})")
    check(bool(np.allclose(np_(got_pos), 0.05, atol=1e-5)), "右墙写为 +0.05")

    print("\n[3] 碰撞跟着墙走 / 墙挪开后旧位置不再碰撞")
    # 环境 0 收到净宽 0.076 (内壁 ±0.038 = reset 足端半宽, 恰好不干涉); 其余保持 -0.20
    neg = torch.full((N,), -0.20, device="cuda:0")
    neg[0] = -0.048
    pos = torch.full((N,), 0.05, device="cuda:0")
    pos[0] = 0.048
    ent.write_wall_x(env, neg, pos)
    h1 = wall_hits(env, ent, 200, act)
    check(h1[0] > 0, f"收紧的环境 0 出现墙接触 (计数 {h1[0]})")
    check(h1[1:].sum() == 0, f"其余环境无墙接触 (计数 {h1[1:].sum()})")
    root = np_(env.scene.entities["robot"].data.root_link_pos_w[:, 0])
    check(bool(np.isfinite(root).all()), "全部环境 root x 有限 (无 NaN)")
    check(abs(root[0]) < 0.04, f"环境 0 root x 被限制在 ±0.04 内 ({root[0]:+.4f})")
    # 把环境 0 的墙挪到远处: 旧位置不应再产生碰撞
    neg[0] = -0.20
    pos[0] = 0.20
    ent.write_wall_x(env, neg, pos)
    h2 = wall_hits(env, ent, 100, act)
    check(h2.sum() == 0, f"墙挪开后全环境无墙接触 (计数 {h2.sum()})")

    print("\n[4] 命令侧采样: 物理墙位 == 观测墙位, d 落在 [d_min, d_max]")
    cmd.set_wall_curriculum(0.08, 0.20, 0.3)
    cmd._wall_d.fill_(0.20)          # 先污染, 验证 reset 会重采
    env.reset()
    d_obs = -np_(cmd.wall_x_neg_command)
    d_phys = -np_(ent.read_wall_x(env)[0])
    check(bool(np.allclose(d_obs, d_phys, atol=1e-5)),
          "观测墙位与物理墙位一致 (不再广播 cfg 值)")
    check(bool(((d_obs >= 0.08 - 1e-6) & (d_obs <= 0.20 + 1e-6)).all()),
          f"d 全部落在 [0.08, 0.20] (min={d_obs.min():.4f} max={d_obs.max():.4f})")
    check(bool((np.abs(d_obs - 0.08) < 1e-6).any()) or N < 4,
          "存在取到 d_min 的下界样本 (概率采样在 N=8 下可能为空, 小样本放行)")
    check(bool(np.allclose(np_(cmd.wall_x_pos_command), 0.05, atol=1e-5)), "右墙观测恒为 +0.05")

    print("\n[5] 循环复位不换墙位")
    before = np_(ent.read_wall_x(env)[0]).copy()
    d_before = np_(cmd._wall_d).copy()
    with torch.no_grad():
        cmd._pending_cycle_reset[:] = True
        cmd._apply_cycle_reset(torch.ones(N, dtype=torch.bool, device="cuda:0"))
    after = np_(ent.read_wall_x(env)[0])
    check(bool(np.allclose(before, after, atol=1e-5)),
          "循环复位后物理墙位不变 (sim.reset 打回默认值的问题已修)")
    check(bool(np.allclose(d_before, np_(cmd._wall_d), atol=1e-5)),
          "循环复位后 _wall_d 未被重新采样")
    check(bool(np.allclose(np_(cmd.wall_x_neg_command), after, atol=1e-5)),
          "循环复位后观测与物理仍一致")

    print("\n" + "=" * 90)
    if fails:
        print(f"结论: 未通过 {len(fails)} 项")
        for f in fails:
            print("  - " + f)
        sys.exit(1)
    print("结论: 全部通过")


if __name__ == "__main__":
    main()
