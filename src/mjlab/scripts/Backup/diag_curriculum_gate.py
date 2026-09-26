# 诊断: 用已有检查点评估墙位课程的门控量是否达标 (只读, 不落盘)
#   p_stood : 回合"站起来"率 = 站姿维持满确认窗口且当步严格几何, **不含关节速度** <- 课程推进量
#   p_onset : 回合内出现过站立窗口 (单帧, 备用/诊断)
#   p_done  : 回合稳定站立成功率 (现行判据, 含 V/T <= 门限)
# 训练时的门控量是在**采样动作**下统计的, 所以默认两种动作模式都跑, 以 sample 为准。
# 用法:
#   uv run python -B -m mjlab.scripts.Backup.diag_curriculum_gate <checkpoint> \
#       --wall_x_neg -0.20 --wall_x_pos 0.08
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.SQuRo_Backup.mdp import entity as mdp_entity
from mjlab.tasks.SQuRo_Backup.mdp.curriculums import (
    CURRICULUM_GATE_P_STOOD,
    _STEPS_PER_ITER,
)
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES

TASK_NAME = "Mjlab-SQuRo-Backup"


@dataclass
class Cfg:
    checkpoint: str
    wall_x_neg: float = -0.20          # 墙位必须显式给 (自动课程的墙位无法按轮次反推)
    wall_x_pos: float = 0.08
    enable_collision: bool = True
    mode: str = "both"                 # sample | deterministic | both
    num_envs: int = 128
    steps: int = 3900                  # 约 3 个 12s 回合 (首个回合无效)
    device: str = "cuda:0"
    seed: int = 0


# 包络统计桶: 基座/足端的世界 X、Y、Z 极值 + 计入的环境样本数
def _new_box() -> dict:
    d = {"n": 0}
    for k in ("base_x", "base_y", "base_z", "foot_x", "foot_y", "foot_z"):
        d[k] = [float("inf"), float("-inf")]
    return d


# 把一帧快照并入包络 (mask 给定时只统计被选中的环境)
def _acc_box(dst: dict, b_xyz, s_xyz, mask=None) -> None:
    if mask is not None:
        if not bool(mask.any()):
            return
        b_xyz, s_xyz = b_xyz[mask], s_xyz[mask]
    for tag, arr in (("base", b_xyz), ("foot", s_xyz)):
        for i, ax in enumerate("xyz"):
            v = arr[..., i]
            dst[f"{tag}_{ax}"] = [min(dst[f"{tag}_{ax}"][0], float(v.min())),
                                  max(dst[f"{tag}_{ax}"][1], float(v.max()))]
    dst["n"] += int(b_xyz.shape[0])


# 跑一种动作模式, 统计回合级的三个率 (与训练侧 _ingest_episode_results 同一套发布机制)
def run_mode(name: str, env, policy, cmd, steps: int, num_envs: int) -> dict:
    obs = env.get_observations()
    # 只统计本模式开始之后发布的回合: 起点取当前序号快照, **不能**对推理张量做就地清零
    seq_seen = cmd._ep_seq.detach().to("cpu").clone()
    stat = {"episodes": 0, "valid": 0, "onset": 0, "stood": 0, "success": 0}
    vt_sum, vt_n = 0.0, 0
    robot = env.unwrapped.scene.entities["robot"]
    ent = env.unwrapped.scene.entities["restricted_space"]
    foot_ids = _MODEL_INDICES.foot_site_ids
    neg, pos = ent.cfg.wall_x_neg, ent.cfg.wall_x_pos
    half_t = ent.cfg.wall_half_thickness
    half_l = ent.cfg.wall_half_length
    wh = ent.cfg.wall_height
    stand_box, init_box = _new_box(), _new_box()
    inside_wall, stand_samples = 0, 0
    out_x_beside, out_x_beyond = 0, 0
    base_beside, base_beyond = 0, 0
    for k in range(steps):
        with torch.inference_mode():
            # 先取状态再步进: k=0 取到的就是重置后的初始姿态
            b_xyz = robot.data.root_link_pos_w[:, :3]
            s_xyz = robot.data.site_pos_w[:, foot_ids, :3]
            if k == 0:
                _acc_box(init_box, b_xyz, s_xyz)
            # 站立窗口内的包络: 墙在"已经站起来"之后是否还限制得住姿态 (见技术细节 §7.14)
            active = cmd._stand_elapsed > 0
            if bool(active.any()):
                _acc_box(stand_box, b_xyz, s_xyz, active)
                stand_samples += int(active.sum())
                sx, sy, sz = s_xyz[..., 0], s_xyz[..., 1], s_xyz[..., 2]
                in_x = ((sx > neg - half_t) & (sx < neg + half_t)) | \
                       ((sx > pos - half_t) & (sx < pos + half_t))
                # 足端落在墙体 X 区间、低于墙顶且在墙的 Y 范围内 => 与墙体重叠
                hit = in_x & (sz < wh) & (sy.abs() < half_l) & active.unsqueeze(-1)
                inside_wall += int(hit.sum())
                # 足端已经在墙内侧之外: 分"在墙的 Y 范围内(只能越顶)"与"越过墙端(绕行)"
                beyond = (sx < neg + half_t) | (sx > pos - half_t)
                beside = beyond & (sy.abs() < half_l) & active.unsqueeze(-1)
                out_x_beside += int(beside.sum())
                out_x_beyond += int((beyond & ~(sy.abs() < half_l) & active.unsqueeze(-1)).sum())
                # 基座同理: 判断"墙有没有真的把躯干限制在走廊里"
                b_x, b_y = b_xyz[..., 0], b_xyz[..., 1]
                b_out = (b_x < neg + half_t) | (b_x > pos - half_t)
                base_beside += int((b_out & (b_y.abs() < half_l) & active).sum())
                base_beyond += int((b_out & ~(b_y.abs() < half_l) & active).sum())
            act = policy(obs, stochastic_output=True) if name == "sample" else policy(obs)
            obs, _, _, _ = env.step(act.to(env.device))
        seq = cmd._ep_seq.detach().to("cpu")
        new = (seq > seq_seen).nonzero(as_tuple=False).squeeze(-1)
        if len(new) > 0:
            valid = cmd._last_ep_valid.detach().to("cpu")[new]
            onset = cmd._last_ep_stood_onset.detach().to("cpu")[new]
            stood = cmd._last_ep_stood_pose.detach().to("cpu")[new]
            succ = cmd._last_ep_success.detach().to("cpu")[new]
            for i in range(len(new)):
                stat["episodes"] += 1
                if bool(valid[i]):
                    stat["valid"] += 1
                    stat["onset"] += int(onset[i])
                    stat["stood"] += int(stood[i])
                    stat["success"] += int(succ[i])
            seq_seen[new] = seq[new]
        # 判据量本身: 站立窗口活跃环境上的 V/T
        active = cmd._stand_elapsed > 0
        if bool(active.any()):
            vt_sum += float(cmd.windowed_mean_vel()[active].mean())
            vt_n += 1
    stat["mean_vt"] = (vt_sum / vt_n) if vt_n else float("nan")
    stat["stand_box"] = stand_box
    stat["init_box"] = init_box
    stat["stand_samples"] = stand_samples
    stat["inside_wall"] = inside_wall
    stat["out_x_beside"] = out_x_beside
    stat["out_x_beyond"] = out_x_beyond
    stat["base_beside"] = base_beside
    stat["base_beyond"] = base_beyond
    stat["walls"] = (neg, pos, half_t, half_l, wh)
    return stat


def main() -> None:
    cfg = tyro.cli(Cfg)
    torch.manual_seed(cfg.seed)
    m = re.search(r"model_(\d+)", Path(cfg.checkpoint).name)
    it = int(m.group(1)) if m else 0

    env_cfg = load_env_cfg(TASK_NAME)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.events.pop("init_restricted_space", None)
    mdp_entity.configure_restricted_space(env_cfg, wall_x_neg=cfg.wall_x_neg,
                                         wall_x_pos=cfg.wall_x_pos,
                                         enable_collision=cfg.enable_collision)
    raw = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
    raw.common_step_counter = it * _STEPS_PER_ITER      # 让 λ 课程落在训练同段
    env = RslRlVecEnvWrapper(raw)
    cmd = env.unwrapped.command_manager.get_term("backup_cmd")

    runner = load_runner_cls(TASK_NAME)(env, asdict(load_rl_cfg(TASK_NAME)), None,
                                        device=cfg.device)
    runner.load(cfg.checkpoint, load_cfg={"actor": True}, strict=True,
                map_location=cfg.device)
    policy = runner.get_inference_policy(device=cfg.device)

    net = cfg.wall_x_pos - cfg.wall_x_neg - 0.02
    print("=" * 92)
    print(f"检查点 {Path(cfg.checkpoint).name} (iter {it})  墙位 x_neg={cfg.wall_x_neg:+.4f} "
          f"x_pos={cfg.wall_x_pos:+.4f} (净宽 {net:.4f} m)  碰撞={cfg.enable_collision}  "
          f"环境数={cfg.num_envs}  步数={cfg.steps}")
    print(f"课程门控: p_stood >= {CURRICULUM_GATE_P_STOOD}")
    print("=" * 92)
    modes = ["deterministic", "sample"] if cfg.mode == "both" else [cfg.mode]
    for name in modes:
        s = run_mode(name, env, policy, cmd, cfg.steps, cfg.num_envs)
        v = max(1, s["valid"])
        print(f"\n[{name} 动作]")
        print(f"  收到回合 {s['episodes']} 个 (有效 {s['valid']} 个)")
        print(f"  p_onset (站立窗口出现)   = {s['onset'] / v:.3f}")
        print(f"  p_stood (站起来并维持)   = {s['stood'] / v:.3f}"
              f"   {'>= 门控, 会推进' if s['stood'] / v >= CURRICULUM_GATE_P_STOOD else '< 门控, 不推进'}")
        print(f"  p_done  (稳定站立成功)   = {s['success'] / v:.3f}")
        print(f"  站立窗口内平均 V/T       = {s['mean_vt']:.2f} rad/s")
        neg, pos, half_t, half_l, wh = s["walls"]
        ib = s["init_box"]
        print(f"  初始姿态 基座 X [{ib['base_x'][0]:+.4f}, {ib['base_x'][1]:+.4f}]"
              f" Y [{ib['base_y'][0]:+.4f}, {ib['base_y'][1]:+.4f}]"
              f" Z [{ib['base_z'][0]:+.4f}, {ib['base_z'][1]:+.4f}]")
        print(f"  初始姿态 足端 X [{ib['foot_x'][0]:+.4f}, {ib['foot_x'][1]:+.4f}]"
              f" Y [{ib['foot_y'][0]:+.4f}, {ib['foot_y'][1]:+.4f}]"
              f" Z [{ib['foot_z'][0]:+.4f}, {ib['foot_z'][1]:+.4f}]")
        print(f"  墙盒   X [{neg - half_t:+.4f}, {neg + half_t:+.4f}] 与 "
              f"[{pos - half_t:+.4f}, {pos + half_t:+.4f}]  Y ±{half_l:.3f}  Z 0~{wh:.3f}"
              f"  (净宽 {pos - neg - 2 * half_t:.4f} m)")
        sb = s["stand_box"]
        if s["stand_samples"] == 0:
            print("  站立窗口内无样本, 无法给出包络")
        else:
            print(f"  站立窗口 {s['stand_samples']} 个环境样本: "
                  f"基座 X [{sb['base_x'][0]:+.4f}, {sb['base_x'][1]:+.4f}]"
                  f" Y [{sb['base_y'][0]:+.4f}, {sb['base_y'][1]:+.4f}]"
                  f" Z [{sb['base_z'][0]:+.4f}, {sb['base_z'][1]:+.4f}]")
            print(f"  站立窗口内足端 X [{sb['foot_x'][0]:+.4f}, {sb['foot_x'][1]:+.4f}]"
                  f" Y [{sb['foot_y'][0]:+.4f}, {sb['foot_y'][1]:+.4f}]"
                  f" Z [{sb['foot_z'][0]:+.4f}, {sb['foot_z'][1]:+.4f}]")
            n = max(1, s["stand_samples"])
            nf = 4 * n            # 足端统计的样本数是 4 只脚 × 环境样本数
            print(f"  站立窗口内足端落入墙盒 = {s['inside_wall']} / {nf}"
                  f"  ({100.0 * s['inside_wall'] / nf:.3f}%)")
            print(f"  站立窗口内足端已在墙内侧之外: 在墙 Y 范围内(含与墙体重叠) = "
                  f"{s['out_x_beside']} ({100.0 * s['out_x_beside'] / nf:.3f}%)  "
                  f"越过墙端(绕行) = {s['out_x_beyond']} ({100.0 * s['out_x_beyond'] / nf:.3f}%)")
            print(f"  站立窗口内基座已在墙内侧之外: 在墙 Y 范围内(与墙体重叠) = "
                  f"{s['base_beside']} ({100.0 * s['base_beside'] / n:.2f}%)  "
                  f"越过墙端(在走廊之外) = {s['base_beyond']} ({100.0 * s['base_beyond'] / n:.2f}%)")
        # 每次换模式前重置环境, 避免上一次的回合尾巴混进统计 (序号快照在 run_mode 里取)。
        # 必须在 inference_mode 内: 上一步 step 已把管理器里的缓冲张量标成推理张量,
        # 在外面做就地写会直接抛错。
        with torch.inference_mode():
            env.reset()


if __name__ == "__main__":
    main()
