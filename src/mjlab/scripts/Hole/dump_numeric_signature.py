# 数值签名: 冻结当前分支的全部可观测数值, 供重构前后逐项比对
# 用法: uv run python -B -m mjlab.scripts.Hole.dump_numeric_signature [输出json]
# 设计: 只读, 不写仿真状态以外的任何东西; 所有量都来自公开接口

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-SQuRo-Hole"
CPU = "cpu"
# 固定种子, 保证两次运行的采样与仿真初态一致
SEED = 12345


# 把张量压成可比对的紧凑结构 (保留全部数值精度)
def _t(x) -> list:
    return np.asarray(x, dtype=np.float64).reshape(-1).tolist()


# 参考表: 三个模式 × 四个高度档 × 500 相位的完整数值
def dump_reference_tables() -> dict:
    from mjlab.tasks.SQuRo_Hole.mdp.reference import Initialize_Tables

    tables = Initialize_Tables(CPU)
    out = {"keys": sorted(tables.keys())}
    for key in ("front_pos", "hind_pos", "spine_pos", "front_vel", "hind_vel", "spine_vel",
                "phase_range", "mode_periods"):
        if key in tables:
            out[key] = _t(tables[key])
    for key in ("front_T", "hind_T"):
        if key in tables:
            out[key] = float(tables[key])
    out["height_list"] = [float(v) for v in tables["height_list"]]
    return out


# 位置表与阶段函数: 纯函数, 直接逐点取值
def dump_schedule_and_stages() -> dict:
    import mjlab.tasks.SQuRo_Hole.mdp.command as cmd
    import mjlab.tasks.SQuRo_Hole.mdp.config as cfg

    out = {
        "STAGE3_POSITION_SCHEDULE": [list(row) for row in cmd.STAGE3_POSITION_SCHEDULE],
    }
    # 阶段判定: 扫遍所有阶段边界前后各 2 步
    bounds = [cfg.STAGE1_END, cfg.STAGE2_END, cfg.STAGE3_END]
    steps = sorted({0, 1, *[b + d for b in bounds for d in (-2, -1, 0, 1, 2)]})
    out["get_current_stage"] = {str(s): int(cfg.get_current_stage(s)) for s in steps}
    out["stage_uses_schedule"] = {str(s): bool(cfg.stage_uses_schedule(s)) for s in range(0, 6)}
    out["stage_requires_collision"] = {str(s): bool(cfg.stage_requires_collision(s)) for s in range(0, 6)}
    out["heights_for_stage"] = {
        str(s): (None if cfg.heights_for_stage(s) is None else [float(v) for v in cfg.heights_for_stage(s)])
        for s in range(0, 6)
    }
    # 速度公式: 网格扫描
    hf = torch.tensor([0.02, 0.0395, 0.04, 0.05, 0.055], dtype=torch.float32)
    hh = torch.tensor([0.02, 0.0395, 0.04, 0.05, 0.055], dtype=torch.float32)
    grid = torch.cartesian_prod(hf, hh)
    out["compute_command_velocity"] = _t(cmd.compute_command_velocity(grid[:, 0], grid[:, 1]))
    out["command_velocity_grid"] = _t(grid)
    return out


# 课程: 每条曲线在每个阶段的值 + 权重表
def dump_curriculum() -> dict:
    from mjlab.tasks.SQuRo_Hole.mdp.curriculums import reward_weight_curriculum, _CURVES

    out = {"curves": {k: [float(v) for v in vals] for k, vals in sorted(_CURVES.items())}}
    probes = [0, 1, 47999, 48000, 95999, 96000, 143999, 144000, 191999]
    out["weights_at"] = {str(s): reward_weight_curriculum.get_reward_weights(s) for s in probes}
    out["stage_info_at"] = {str(s): reward_weight_curriculum.get_current_stage_info(s) for s in probes}
    return out


# 几何与物理常量: 板几何 / 接触开关 / 仿真参数
def dump_geometry_and_sim() -> dict:
    import mjlab.tasks.SQuRo_Hole.mdp.config as cfg
    from mjlab.tasks.SQuRo_Hole.mdp.entity import hole_geometry

    out = {"hole_geometry": hole_geometry()}
    for name in ("BASE_HEIGHT", "BASE_FREQ", "BASE_SPEED", "THRESHOLD_HEIGHT", "HEIGHT_LIST",
                 "TABLE_RESOLUTION", "USE_SPINE_CSV", "NECK_REF_POS", "NECK_REF_VEL",
                 "STAGE1_END_ITER", "STAGE2_END_ITER", "STAGE3_END_ITER", "NUM_STAGES",
                 "STEPS_PER_ITER", "STAGE1_END", "STAGE2_END", "STAGE3_END",
                 "RANDOM_HEIGHT_VALUES", "FULL_HEIGHT_VALUES", "STAGE_COLLISION", "ANGLE_VALUES"):
        if hasattr(cfg, name):
            v = getattr(cfg, name)
            out[name] = list(v) if isinstance(v, (list, tuple)) else v

    import mjlab.tasks.SQuRo_Hole.mdp.indices as idx
    for name in ("_ACT_JOINT_NAMES", "REF_TABLE_ORDER", "REF_SLOT_OF_ACT", "_ACT_LEG_IDS",
                 "_ACT_SPN_IDS", "_ACT_NECK_IDS", "_ACT_FL_LEG_IDS", "_ACT_FR_LEG_IDS",
                 "_ACT_HL_LEG_IDS", "_ACT_HR_LEG_IDS", "FRONT_SEG_SITE_NAMES",
                 "REAR_SEG_SITE_NAMES", "FOOT_SITE_NAMES"):
        v = getattr(idx, name, None)
        if v is not None:
            out["indices." + name] = [list(x) if isinstance(x, (list, tuple)) else x for x in v] \
                if isinstance(v, tuple) and v and isinstance(v[0], (list, tuple)) else list(v)
    return out


# 环境级: 同一初态下推 N 步, 逐项记录观测/奖励/命令
def dump_env_rollout(steps: int = 40) -> dict:
    from mjlab.tasks.SQuRo_Hole.mdp.reference import Initialize_Tables

    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = 8
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    env = ManagerBasedRlEnv(cfg=cfg, device=CPU)
    Initialize_Tables(CPU)

    obs = env.reset()
    obs_dict = obs[0] if isinstance(obs, tuple) else obs
    out = {
        "obs_shapes": {k: list(v.shape) for k, v in obs_dict.items()},
        "action_dim": int(env.action_space.shape[-1]),
        "max_episode_length": int(env.max_episode_length),
        "reward_term_names": sorted(env.reward_manager.active_terms),
        "observation_term_names": {k: list(env.observation_manager.active_terms[k])
                                   for k in env.observation_manager.active_terms},
        "obs_reset": {k: _t(v) for k, v in obs_dict.items()},
        "reward_terms_reset": _t(env.reward_manager._step_reward),
    }

    # 零动作固定推步: 记录每步的总奖励、各奖励项、命令、关节状态
    zeros = torch.zeros(env.num_envs, env.action_space.shape[-1], device=env.device)
    traj_reward, traj_cmd, traj_joint, traj_body = [], [], [], []
    with torch.inference_mode():
        for _ in range(steps):
            env.step(zeros)
            traj_reward.append(_t(env.reward_manager._step_reward))
            cmd = env.command_manager._terms["hole_cmd"].command
            traj_cmd.append(_t(cmd))
            asset = env.scene["robot"]
            traj_joint.append(_t(asset.data.joint_pos))
            traj_body.append(_t(asset.data.body_link_pos_w))
    out["traj_step_reward"] = traj_reward
    out["traj_command"] = traj_cmd
    out["traj_joint_pos"] = traj_joint
    out["traj_body_pos"] = traj_body
    env.close()
    return out


# 参考状态: 命令固定在若干工况下, 取参考关节位置/速度
def dump_reference_states() -> dict:
    from mjlab.tasks.SQuRo_Hole.mdp.config import CMD_HEIGHT_F_IDS, CMD_HEIGHT_H_IDS
    from mjlab.tasks.SQuRo_Hole.mdp.reference import Initialize_Tables, get_reference_joint_state

    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device=CPU)
    Initialize_Tables(CPU)
    env.reset()
    cmd_term = env.command_manager._terms["hole_cmd"]
    out = {}
    # 覆盖四个高度模式 + 位置表全部 8 段
    cases = [(0.055, 0.055), (0.02, 0.05), (0.055, 0.02), (0.02, 0.02), (0.04, 0.04), (0.06, 0.02)]
    with torch.inference_mode():
        for hf, hh in cases:
            cmd_term.command[:, CMD_HEIGHT_F_IDS] = hf
            cmd_term.command[:, CMD_HEIGHT_H_IDS] = hh
            for phase_step in (0, 25, 50, 125, 250, 375, 499):
                env.episode_length_buf[:] = phase_step
                pos, vel = get_reference_joint_state(env)
                key = f"hF{hf:.3f}_hH{hh:.3f}_phase{phase_step}"
                out[key] = {"pos": _t(pos), "vel": _t(vel)}
    env.close()
    return out


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tmp_branch_audit/signature.json")
    sig = {
        "reference_tables": dump_reference_tables(),
        "schedule_and_stages": dump_schedule_and_stages(),
        "curriculum": dump_curriculum(),
        "geometry_and_constants": dump_geometry_and_sim(),
        "reference_states": dump_reference_states(),
        "env_rollout": dump_env_rollout(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sig, f, allow_nan=False)
    n = sum(len(v) if isinstance(v, list) else 1 for v in sig.values())
    print(f"[签名] 已写出 {out_path} ({out_path.stat().st_size} bytes, {n} 个顶层数据块)")


if __name__ == "__main__":
    main()
