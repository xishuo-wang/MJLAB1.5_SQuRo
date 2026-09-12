from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES, resolve_model_indices
from mjlab.tasks.SQuRo_Backup.mdp.reference import get_reference_joint_state
from mjlab.utils.torch import configure_torch_backends


# 在环境步之后采样真实姿态，记录的是刚才实际下发的参考与动作。
def sample_state(env, action, target, time):
    asset = env.scene.entities["robot"]
    body_ids = [_MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id]
    quat = asset.data.body_link_quat_w[0, body_ids]
    w, x, y, z = quat.unbind(dim=1)
    up = 2 * (y * z + w * x) * torch.tensor([1.0, -1.0], device=env.device)
    heights = asset.data.body_link_pos_w[0, body_ids, 2]
    fu, hu = up.tolist()
    fz, hz = heights.tolist()
    checks = [fu > 0.5, hu < -0.5, fz < 0.03, hz < 0.03]
    term = env.command_manager.get_term("backup_cmd")
    return {
        "t": time,
        "fu": fu, "hu": hu, "fz": fz, "hz": hz,
        "s1_checks": checks, "s1": all(checks),
        "q": asset.data.joint_pos[0, _MODEL_INDICES.joint_ids].tolist(),
        "q_vel": asset.data.joint_vel[0, _MODEL_INDICES.joint_ids].tolist(),
        "reference": target[0].tolist(), "action": action[0].tolist(),
        "body_quat": quat.tolist(),
        "body_pos": asset.data.body_link_pos_w[0, body_ids].tolist(),
        "phase": int(term.phase[0]), "stage_t": float(term.t_phase[0]),
    }


# 仅比较首次 P1：强制执行完整参考后保持端点，不允许重试或提前进入 P2 混淆终态。
def manual_trial(env, lam, recover, inverse_scale, hind_hip, buffer):
    from mjlab.scripts.SQuRo_backup_Replay import slow1_target

    env.reset()
    asset = env.scene.entities["robot"]
    default = asset.data.default_joint_pos[:, _MODEL_INDICES.joint_ids]
    end_nom = 0.65 + recover
    end = end_nom * lam
    steps = math.ceil((end + buffer) / env.step_dt)
    records = []
    with torch.no_grad():
        for i in range(steps):
            t_nom = min(i * env.step_dt / lam, end_nom)
            target = torch.tensor(
                [slow1_target(1.0 + t_nom * lam, lam, recover, hind_hip)],
                device=env.device,
            )
            action = (target - default) / inverse_scale
            env.step(action)
            records.append(sample_state(env, action, target, (i + 1) * env.step_dt))
    return records


# 汇总前半段末、P1 末和缓冲末姿态，以及具备切换时间资格的 S1 检测。
def summarize(records, lam, recover):
    end = (0.65 + recover) * lam
    snapshots = {}
    for label, time in [("build_end", 0.65 * lam), ("p1_end", end), ("buffer_end", records[-1]["t"])]:
        row = min(records, key=lambda r: abs(r["t"] - time))
        snapshots[label] = {k: row[k] for k in ["t", "fu", "hu", "fz", "hz", "s1", "s1_checks"]}
        snapshots[label]["spine"] = [row["q"][j] for j in [0, 1, 8, 9]]
    eligible = [r for r in records if r["t"] >= end - 1e-6]
    return {
        "snapshots": snapshots,
        "eligible_s1_count": sum(r["s1"] for r in eligible),
        "eligible_count": len(eligible),
        "condition_pass_counts": [sum(r["s1_checks"][j] for r in eligible) for j in range(4)],
    }


# 在同一物理环境、相同重置条件下分离时间、动作放大和后腿参考的影响。
def run_comparison(args):
    # 与训练策略可视化入口采用相同的矩阵运算精度。
    configure_torch_backends()
    folder = Path(args.video_dir) / ("p1_compare_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "diagnostics.json"
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    cfg.scene.num_envs = 1
    cfg.commands["backup_cmd"].fixed_time_scale = args.time_scale
    # 保留 stand 项名称供里程碑奖励查询，但关闭诊断期间的自动重置。
    for term_cfg in cfg.terminations.values():
        term_cfg.func = lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    cfg.episode_length_s = max(args.duration, 1.3 * args.time_scale + args.buffer + 1)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    resolve_model_indices(env.scene.entities["robot"])
    env_scale = float(cfg.actions["joint_pos"].scale)
    output = {
        "time_scale": args.time_scale, "env_action_scale": env_scale,
        "step_dt": env.step_dt, "buffer": args.buffer,
        "protocol": "每例独立重置；强制执行首次 P1 后保持端点；不提前切换、不重试、不自动重置",
        "cases": [],
    }
    try:
        for name, inverse_scale, hip in [
            ("legacy_mapping", 0.3, -1.5),
            ("correct_mapping", env_scale, -1.5),
            ("training_reference", env_scale, -1.4),
        ]:
            for recover in [0.15, 0.65]:
                for repeat in range(args.compare_repeats):
                    rows = manual_trial(env, args.time_scale, recover, inverse_scale, hip, args.buffer)
                    case = {"name": name, "repeat": repeat, "recover": recover, "inverse_scale": inverse_scale, "hind_hip": hip,
                            "summary": summarize(rows, args.time_scale, recover), "records": rows}
                    output["cases"].append(case)
                    print("[对照] " + json.dumps({k: v for k, v in case.items() if k != "records"}, ensure_ascii=False), flush=True)

        if args.checkpoint_file:
            rl_cfg = load_rl_cfg("Mjlab-SQuRo-Backup")
            wrapper = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
            runner_cls = load_runner_cls("Mjlab-SQuRo-Backup") or MjlabOnPolicyRunner
            runner = runner_cls(wrapper, asdict(rl_cfg), log_dir=None, device=device)
            runner.load(args.checkpoint_file, load_cfg={"actor": True}, strict=True, map_location=device)
            policy = runner.get_inference_policy(device=device)
            for repeat in range(args.compare_repeats):
                obs, _ = wrapper.reset()
                records = []
                steps = math.ceil((1.3 * args.time_scale + args.buffer) / env.step_dt)
                with torch.no_grad():
                    for i in range(steps):
                        target, _ = get_reference_joint_state(env)
                        action = policy(obs)
                        obs, _, _, _ = wrapper.step(action)
                        records.append(sample_state(env, action, target, (i + 1) * env.step_dt))
                case = {"name": "trained_policy", "repeat": repeat, "recover": 0.65, "checkpoint": args.checkpoint_file,
                        "summary": summarize(records, args.time_scale, 0.65), "records": records}
                output["cases"].append(case)
                print("[策略] " + json.dumps({k: v for k, v in case.items() if k != "records"}, ensure_ascii=False), flush=True)
    finally:
        env.close()
        # 即使后续试验出错，也保留已完成的独立对照数据。
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[结果] 逐步姿态和关节数据：{path.resolve()}", flush=True)
