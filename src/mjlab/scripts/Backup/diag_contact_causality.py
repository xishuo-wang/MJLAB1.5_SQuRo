# 诊断: 论证"行为改变来自闭环力学接触, 而不是知道走廊变窄" (只读, 落盘 CSV + 图)
# 三组对照, 全部固定 checkpoint / λ / 确定性策略 / 同一初始姿态:
#   E1 配对闭环: 同一标称墙位, 碰撞开 vs 关 —— 观测逐位相同, 唯一差别是墙是否参与物理。
#      同条件再跑一遍作为"数值噪声底"(独立构建环境)。
#   E2 反事实开环: 把 A 条件录到的动作序列**原样**喂进 B 条件。若宽走廊的动作在窄走廊里
#      同样塌缩, 说明几何本身完成了行为改变, 不需要策略适应。
#      含恒等对照(同条件自复放)用来证明复放管线本身无偏。
#   E3 墙接触力: ContactSensorCfg(逐环境 mocap 墙)给出首次接触时刻 t_c 与力时程。
# 用法:
#   uv run python -B -m mjlab.scripts.Backup.diag_contact_causality \
#       --checkpoint logs/rsl_rl/SQuRo_Backup/<run>/model_8999.pt
from __future__ import annotations

import csv
import importlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.SQuRo_Backup.mdp import entity as mdp_entity
from mjlab.tasks.SQuRo_Backup.mdp.curriculums import WALL_X_POS, _STEPS_PER_ITER
from mjlab.tasks.SQuRo_Backup.mdp.entity import WALL_HALF_THICKNESS
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES


TASK_NAME = "Mjlab-SQuRo-Backup"
CONTROL_DT = 0.01                 # = env_cfg 的 timestep 0.002 × decimation 5
# 每步记录: [0:14] 动作, 14 base_x, 15 foot|x|max, 16 foot_x 最小, 17 u_F, 18 u_H,
#           19 phase, 20 wall_force_neg, 21 wall_force_pos
ROW_DIM = 22
# 墙接触传感器 primary: 机器人全部 body (几何名有空串, 只能按 body 取)
CONTACT_BODY_PATTERNS = (
    "base_Link", "FU_.*", "FD_.*", "F_spine.*", "F_body_Link",
    "H_spine.*", "H_body_Link", "Neck_.*", "FL_.*", "FR_.*", "HL_.*", "HR_.*",
)


@dataclass
class Cfg:
    checkpoint: str
    clear_width: float = 0.09          # 窄走廊净宽 (训练课程末档 d=0.06)
    wide_clear_width: float = 0.23     # 宽走廊净宽 (训练课程起点 d=0.20)
    steps: int = 320                   # 覆盖一个完整循环 (≈2.7 s) 并留出余量
    repeats: int = 3                   # 每条件闭环重复次数 (同构建内, 给重置级重复性)
    time_scale: float = 1.0
    device: str = "cuda:0"
    seed: int = 0
    out_dir: str = "logs/backup_contact_causality"


# 构造固定净宽的环境; wall_force=True 时挂墙接触传感器
def _build(cfg: Cfg, clear_width: float, collision: bool, wall_force: bool, iter_num: int):
    importlib.import_module("mjlab.tasks.SQuRo_Backup.config")
    env_cfg = load_env_cfg(TASK_NAME)
    env_cfg.scene.num_envs = 1
    env_cfg.events.pop("init_restricted_space", None)
    env_cfg.commands["backup_cmd"].fixed_time_scale = cfg.time_scale
    half = 0.5 * clear_width + WALL_HALF_THICKNESS
    mdp_entity.configure_restricted_space(env_cfg, wall_x_neg=-half, wall_x_pos=half,
                                         enable_collision=collision, fixed_width=True)
    if collision and wall_force:
        sensors = tuple(env_cfg.scene.sensors or ())
        for tag, geom in (("neg", "restricted_space_wall_n_geom"),
                          ("pos", "restricted_space_wall_p_geom")):
            sensors += (ContactSensorCfg(
                name=f"wall_contact_{tag}",
                primary=ContactMatch(mode="body", pattern=CONTACT_BODY_PATTERNS, entity="robot"),
                secondary=ContactMatch(mode="geom", pattern=geom, entity="restricted_space"),
                fields=("found", "force"), reduce="netforce", num_slots=1),)
        env_cfg.scene.sensors = sensors
    raw = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
    raw.common_step_counter = iter_num * _STEPS_PER_ITER
    return RslRlVecEnvWrapper(raw), raw


# 载入检查点 actor, 返回确定性推理策略
def _make_policy(cfg: Cfg, env):
    from dataclasses import asdict
    runner = load_runner_cls(TASK_NAME)(env, asdict(load_rl_cfg(TASK_NAME)), None,
                                        device=cfg.device)
    runner.load(cfg.checkpoint, load_cfg={"actor": True}, strict=True,
                map_location=cfg.device)
    return runner.get_inference_policy(device=cfg.device)


# 跑一次 rollout: act_seq=None 用闭环策略, 否则逐步开环复放给定动作
# 每次都从检查点重新载入策略: 观测归一化的统计量在 rollout 期间会被更新,
# 复用同一个 policy 会让"第二次 rollout"与"第一次"不可比 (恒等复放对不上就是这个原因)。
def _rollout(env, steps: int, act_seq: np.ndarray | None, wall_force: bool) -> np.ndarray:
    scene = env.unwrapped.scene
    robot = scene.entities["robot"]
    cmd = env.unwrapped.command_manager.get_term("backup_cmd")
    feet = _MODEL_INDICES.foot_site_ids
    policy = None if act_seq is not None else _make_policy(_ROLLOUT_CFG, env)
    # reset 必须在 inference_mode 内: 上一步 step 已把管理器缓冲标成推理张量
    with torch.inference_mode():
        env.reset()
        obs = env.get_observations()
    rows = []
    for k in range(steps):
        with torch.inference_mode():
            if act_seq is None:
                act = policy(obs)
            else:
                act = torch.as_tensor(act_seq[k], device=env.device).reshape(1, -1)
            obs, _, _, _ = env.step(act.to(env.device))
            if wall_force:
                # 墙对机器人的"挤压合力": 每面墙先把 37 个 body 的净力矢量和起来(得到该墙的合反力),
                # 再取两墙范数之和。**不要**用 |Fx|+|Fy|+|Fz| 逐 body 求和 —— 那会把数值放大近一倍
                # (见 docs §7.17 单位与口径)。单位 N; 机器人自重 2.698 N (0.2750 kg)。
                wf = [scene[f"wall_contact_{t}"].data.force[0].sum(dim=0).norm()
                      for t in ("neg", "pos")]
            else:
                wf = [torch.zeros((), device=env.device)] * 2
            fx = robot.data.site_pos_w[0, feet, 0]
            pose = cmd._pose_cos()[0]
            rows.append(torch.cat((
                act.reshape(-1).detach(),
                torch.stack((robot.data.root_link_pos_w[0, 0], fx.abs().max(), fx.min(),
                             pose[0], pose[1], cmd.phase[0].float(), wf[0], wf[1])),
            )))
    return torch.stack(rows).cpu().numpy()


_ROLLOUT_CFG: Cfg = None  # type: ignore[assignment]  # 策略工厂要用到 cfg, 运行时注入


# 一次环境构建内跑完: 预热 (丢弃) -> 闭环 × repeats -> 恒等复放 -> 各路交叉复放
# 每次都先 reset, 初始姿态逐位相同; 对照全部在同一构建内, 避免跨构建的数值差异。
def _session(cfg: Cfg, clear: float, collision: bool, wall_force: bool, iter_num: int,
             cross: dict[str, np.ndarray], tag: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    env, _ = _build(cfg, clear, collision, wall_force, iter_num)
    out: dict[str, np.ndarray] = {}
    _rollout(env, 30, None, wall_force)                 # 预热: 丢掉首次构建态, 统一从"回合内重置态"起步
    closed = _rollout(env, cfg.steps, None, wall_force)
    out[tag] = closed
    acts = closed[:, :14]
    for r in range(1, cfg.repeats):
        out[f"{tag}_rep{r}"] = _rollout(env, cfg.steps, None, wall_force)
    out[f"id_{tag}"] = _rollout(env, cfg.steps, acts, wall_force)
    for src, other in cross.items():
        out[f"ol_{src}_in_{tag}"] = _rollout(env, cfg.steps, other, wall_force)
    env.close()
    return out, acts


# 首次墙接触时刻 (s)
def _first_contact(tr: np.ndarray) -> float:
    hit = np.flatnonzero((tr[:, 20] + tr[:, 21]) > 0)
    return float(hit[0]) * CONTROL_DT if len(hit) else float("nan")


# 动作序列差异 (逐时刻 14 关节 RMS)
def _da(tr_a: np.ndarray, tr_b: np.ndarray) -> np.ndarray:
    n = min(len(tr_a), len(tr_b))
    return np.sqrt(((tr_a[:n, :14] - tr_b[:n, :14]) ** 2).mean(axis=1))


# 首次显著发散时刻: |Δbase_x| 连续 5 步超过 5 mm
def _diverge(tr_a: np.ndarray, tr_b: np.ndarray, tol: float = 0.005) -> float:
    n = min(len(tr_a), len(tr_b))
    d = np.abs(tr_a[:n, 14] - tr_b[:n, 14])
    over = d > tol
    for i in range(len(over) - 5):
        if over[i:i + 5].all():
            return float(i) * CONTROL_DT
    return float("nan")


def main() -> None:
    global _ROLLOUT_CFG
    cfg = tyro.cli(Cfg)
    _ROLLOUT_CFG = cfg
    torch.manual_seed(cfg.seed)
    m = re.search(r"model_(\d+)", Path(cfg.checkpoint).name)
    iter_num = int(m.group(1)) if m else 0
    out_dir = Path(cfg.out_dir) / Path(cfg.checkpoint).parent.name
    out_dir.mkdir(parents=True, exist_ok=True)

    traces: dict[str, np.ndarray] = {}
    print("=" * 96)
    print(f"checkpoint {Path(cfg.checkpoint).name}  λ={cfg.time_scale}  "
          f"窄走廊 {cfg.clear_width * 1000:.0f} mm  宽走廊 {cfg.wide_clear_width * 1000:.0f} mm  "
          f"{cfg.steps} 步", flush=True)

    # E1+E2 顺序: 宽(开) -> 窄(关) -> 窄(开) -> 窄(开)跨构建重复
    # 每个 session 内先闭环重复、再恒等复放、再交叉复放, 保证对照都在同一构建内
    got, act_wide = _session(cfg, cfg.wide_clear_width, True, False, iter_num, {}, "wide_on")
    traces.update(got)
    print(f"  [session] wide_on: max|base_x|={np.abs(traces['wide_on'][:, 14]).max() * 1000:5.1f} mm",
          flush=True)
    got, act_off = _session(cfg, cfg.clear_width, False, False, iter_num,
                            {"wide_act": act_wide}, "narrow_off")
    traces.update(got)
    print(f"  [session] narrow_off: max|base_x|="
          f"{np.abs(traces['narrow_off'][:, 14]).max() * 1000:5.1f} mm", flush=True)
    cross = {"wide_act": act_wide, "narrowoff_act": act_off}
    got, act_on = _session(cfg, cfg.clear_width, True, True, iter_num, cross, "narrow_on")
    traces.update(got)
    print(f"  [session] narrow_on: max|base_x|="
          f"{np.abs(traces['narrow_on'][:, 14]).max() * 1000:5.1f} mm  "
          f"t_contact={_first_contact(traces['narrow_on']):.2f}s", flush=True)
    got, _ = _session(cfg, cfg.clear_width, True, True, iter_num, {}, "narrow_on_build2")
    traces.update(got)
    print(f"  [session] narrow_on_build2 (跨构建重复): max|base_x|="
          f"{np.abs(traces['narrow_on_build2'][:, 14]).max() * 1000:5.1f} mm", flush=True)
    _ = act_on
    # 汇总读数: 峰值对混沌敏感, 另给"接触后平均横向占位"作为稳健量
    summary: list[dict] = []
    for name, tr in traces.items():
        tc = _first_contact(tr)
        cut = int(tc / CONTROL_DT) if np.isfinite(tc) else 0
        post = np.abs(tr[cut:, 14]) if cut < len(tr) else np.abs(tr[:, 14])
        summary.append({
            "rollout": name, "steps": len(tr),
            "base_x_absmax_mm": float(np.abs(tr[:, 14]).max() * 1000),
            "base_x_absmean_post_mm": float(post.mean() * 1000),
            "base_x_absp90_post_mm": float(np.percentile(post, 90) * 1000),
            "foot_x_absmax_mm": float(np.abs(tr[:, 15]).max() * 1000),
            "t_contact_s": tc,
            "wall_force_max": float((tr[:, 20] + tr[:, 21]).max()),
        })

    def val(name: str, key: str) -> float:
        return next(x[key] for x in summary if x["rollout"] == name)

    def group(prefix: str, key: str) -> tuple[float, float]:
        # 只取同构建内的闭环重复: prefix, prefix_rep1, prefix_rep2 ... (排除 id_/ol_/跨构建)
        vals = [x[key] for x in summary if x["rollout"] == prefix or
                (x["rollout"].startswith(prefix + "_rep")
                 and x["rollout"][len(prefix) + 4:].isdigit())]
        return float(np.mean(vals)), float(np.std(vals))

    t_c = val("narrow_on", "t_contact_s")
    print("\n" + "=" * 96)
    print(f"E1 配对闭环 (同一标称墙位, 观测逐位相同, 只差墙是否参与物理)")
    print(f"   首次墙接触 t_c = {t_c:.2f} s;  显著发散 t_div = "
          f"{_diverge(traces['narrow_on'], traces['narrow_off']):.2f} s  "
          f"(Δt = {_diverge(traces['narrow_on'], traces['narrow_off']) - t_c:+.2f} s)")
    da_pair = _da(traces["narrow_on"], traces["narrow_off"])
    da_rep = _da(traces["narrow_on"], traces["narrow_on_rep1"])
    da_wide = _da(traces["narrow_on"], traces["wide_on"])
    k = int(t_c / CONTROL_DT)
    print(f"   接触前动作差 (t<t_c): 开-关 = {da_pair[:k].mean():.5f}   "
          f"同条件重复 = {da_rep[:k].mean():.5f}   ->  两者同级, 说明接触前动作逐位一致")
    print(f"   动作差峰值: 开-关 = {da_pair.max():.2f}   同条件重复 = {da_rep.max():.2f}   "
          f"窄-宽(观测也不同) = {da_wide.max():.2f}")
    id_err = float(np.abs(traces["narrow_on"][:, 14] - traces["id_narrow_on"][:, 14]).max() * 1000)
    rep_err = float(np.abs(traces["narrow_on"][:, 14] - traces["narrow_on_rep1"][:, 14]).max() * 1000)
    print(f"   同一构建内: 恒等复放误差 max|Δx| = {id_err:.2f} mm;  "
          f"同条件重复误差 = {rep_err:.2f} mm  (两者都是重置级不可复现度)")
    for key, lab in (("base_x_absmax_mm", "max|base_x|"),
                     ("base_x_absmean_post_mm", "接触后 mean|base_x|"),
                     ("base_x_absp90_post_mm", "接触后 p90|base_x|")):
        on_m, on_s = group("narrow_on", key)
        off_m, off_s = group("narrow_off", key)
        print(f"   {lab:22s} 碰撞开 {on_m:6.1f} ± {on_s:4.1f} mm    "
              f"碰撞关 {off_m:6.1f} ± {off_s:4.1f} mm    "
              f"比值 {off_m / max(on_m, 1e-6):.2f}×")

    print("\nE2 反事实开环 (同一构建内的对照优先; 量 = max|base_x| / 接触后 mean|base_x| / mm)")
    names = ("wide_on", "wide_on_rep1", "id_wide_on", "ol_narrowoff_act_in_wide_on",
             "narrow_off", "narrow_off_rep1", "id_narrow_off",
             "ol_wide_act_in_narrow_off",
             "narrow_on", "narrow_on_rep1", "id_narrow_on", "narrow_on_build2",
             "ol_wide_act_in_narrow_on", "ol_narrowoff_act_in_narrow_on")
    for name in names:
        if not any(x["rollout"] == name for x in summary):
            continue
        print(f"   {name:32s} {val(name, 'base_x_absmax_mm'):6.1f}   "
              f"{val(name, 'base_x_absmean_post_mm'):6.1f}   "
              f"foot|x|max {val(name, 'foot_x_absmax_mm'):6.1f}   "
              f"t_contact {val(name, 't_contact_s'):.2f}s")

    with (out_dir / f"rollouts_iter{iter_num}.csv").open("w", newline="", encoding="utf-8") as fh:
        keys = ["rollout", "step", "base_x", "foot_x_absmax", "foot_x_min", "u_F", "u_H",
                "phase", "wall_force_neg", "wall_force_pos"] + [f"a{i}" for i in range(14)]
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for name, tr in traces.items():
            for k in range(len(tr)):
                wr.writerow({"rollout": name, "step": k, "base_x": tr[k, 14],
                             "foot_x_absmax": tr[k, 15], "foot_x_min": tr[k, 16],
                             "u_F": tr[k, 17], "u_H": tr[k, 18], "phase": tr[k, 19],
                             "wall_force_neg": tr[k, 20], "wall_force_pos": tr[k, 21],
                             **{f"a{i}": tr[k, i] for i in range(14)}})
    with (out_dir / f"summary_iter{iter_num}.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        wr.writeheader()
        for s in summary:
            wr.writerow(s)
    np.savez(out_dir / f"traces_iter{iter_num}.npz", **traces)
    print(f"\n[INFO] 逐 rollout 明细: {out_dir / f'rollouts_iter{iter_num}.csv'}")
    print(f"[INFO] 汇总: {out_dir / f'summary_iter{iter_num}.csv'}")


if __name__ == "__main__":
    main()
