from __future__ import annotations
import sys
import torch
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES, resolve_model_indices
from mjlab.tasks.SQuRo_Backup.mdp.reference import get_reference_joint_state


# 测量: 若策略完美跟踪训练参考表 (纯开环下发参考角), S1/S2 能否成立、窗口多宽
#
# 与 measure_s1_s2_conditions.py 的区别:
#   那个测的是"手调脚本"的动作 (其自身时序 T2=T3=0.3)
#   这个测的是"训练参考表"的动作 (T2=T3=0.15) —— 即 RL 策略被要求跟踪的目标
#
# 判据直接调用 BackupCommand._check_S1 / _check_S2 / _check_both_inverted

COS45 = 0.7071067811865476


def main() -> None:
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0 * lam

    cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    cfg.scene.num_envs = 1
    cfg.commands["backup_cmd"].fixed_time_scale = lam  # type: ignore[attr-defined]
    for tc in cfg.terminations.values():
        tc.func = lambda e: torch.zeros(e.num_envs, dtype=torch.bool, device=e.device)
    cfg.episode_length_s = dur + 1.0

    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
    env.reset()
    asset = env.unwrapped.scene.entities["robot"]
    resolve_model_indices(asset)
    cmd = env.unwrapped.command_manager.get_term("backup_cmd")

    default = asset.data.default_joint_pos[:, _MODEL_INDICES.joint_ids]
    scale = float(env.unwrapped.cfg.actions["joint_pos"].scale)
    pairs = _MODEL_INDICES.segment_belly_back_ids
    assert pairs is not None

    def u_norm(idx: int) -> float:
        b, k = pairs[idx]
        d = (asset.data.site_pos_w[0, k] - asset.data.site_pos_w[0, b]).detach().cpu().numpy()
        n = float(np.linalg.norm(d))
        return float(d[2] / n) if n > 1e-12 else float("nan")

    rows = []
    with torch.no_grad():
        for _ in range(int(dur / env.step_dt)):
            # 纯开环: 直接下发参考角对应的动作, 不经过任何策略
            ref, _ = get_reference_joint_state(env.unwrapped)
            action = (ref - default) / scale
            env.step(action)
            d = asset.data
            rows.append({
                "t": float(env.unwrapped.episode_length_buf[0]) * float(env.step_dt),
                "uF": u_norm(0), "uH": u_norm(1),
                "zF": float(d.body_link_pos_w[0, _MODEL_INDICES.f_body_id, 2]),
                "zH": float(d.body_link_pos_w[0, _MODEL_INDICES.h_body_id, 2]),
                "S1": bool(cmd._check_S1()[0]),
                "S2": bool(cmd._check_S2()[0]),
                "INV": bool(cmd._check_both_inverted()[0]),
            })

    n = len(rows)
    print(f"纯开环跟踪参考表, λ={lam}, 时长 {rows[-1]['t']:.2f}s, {n} 步")
    print(f"参考表时序: T1=0.65 T2=0.15 T3=0.15 T4=0.50, P1_END=0.80 P2_END=0.95 (名义秒)")
    print()
    print(f"{'t':>6} {'t/λ':>6} | {'uF':>7} {'uH':>7} | {'zF':>7} {'zH':>7} | {'S1':>3} {'S2':>3} {'INV':>4}")
    step = max(1, n // 40)
    for i in range(0, n, step):
        r = rows[i]
        print(f"{r['t']:6.2f} {r['t']/lam:6.2f} | {r['uF']:+7.3f} {r['uH']:+7.3f} | "
              f"{r['zF']:7.4f} {r['zH']:7.4f} | {int(r['S1']):>3} {int(r['S2']):>3} {int(r['INV']):>4}")

    def ever(pred, key=None):
        hits = [r for r in rows if (pred(r) if key is None else r[key])]
        if not hits:
            return "从未成立"
        return f"{len(hits):4d}/{n} 帧, t∈[{hits[0]['t']:.2f}, {hits[-1]['t']:.2f}]s"

    # 连续成立段: 确认计时要求连续, 故真正决定门控的是最长连续段
    def runs(key_or_pred) -> list[tuple[float, float, int]]:
        out = []
        cur = 0
        start = None
        for r in rows:
            hit = (r[key_or_pred] if isinstance(key_or_pred, str)
                   else key_or_pred(r))
            if hit:
                if cur == 0:
                    start = r["t"]
                cur += 1
            else:
                if cur:
                    out.append((start, rows[rows.index(r) - 1]["t"], cur))
                cur = 0
        if cur:
            out.append((start, rows[-1]["t"], cur))
        return out

    dt = float(env.step_dt)
    print()
    print("=== 结果 ===")
    print(f"  S1 四条件同时: {ever(None, 'S1')}")
    print(f"  S2 四条件同时: {ever(None, 'S2')}")
    print(f"  双倒        : {ever(None, 'INV')}")
    print()
    print(f"  连续成立段 (step_dt={dt}s; pose_confirm_s=0.10s → 需连续 {int(0.1/dt)+1} 步):")
    for key in ("S1", "S2", "INV"):
        rs = runs(key)
        if not rs:
            print(f"    {key}: 无")
            continue
        longest = max(rs, key=lambda x: x[2])
        print(f"    {key}: 共 {len(rs)} 段, 最长 {longest[2]} 步 = {longest[2]*dt:.2f}s "
              f"(t∈[{longest[0]:.2f}, {longest[1]:.2f}])"
              f"  {'够 0.10s 确认' if longest[2]*dt >= 0.10 else '** 不足 0.10s, 无法确认 **'}")

    # 进度奖励可发放上限: 每个确认计时器在整段轨迹里累积的总步数
    print()
    print("  [进度奖励上限] 各候选成立的总步数 (完美跟踪下):")
    tot = {"S1": 0, "S2": 0, "INV": 0}
    for r in rows:
        for k in tot:
            if r[k]:
                tot[k] += 1
    for k, v in tot.items():
        print(f"    {k:4s}: {v:4d} 步 × dt({dt}) = {v*dt:.2f}s 等效连续奖励时长")

    print()
    print("  分项:")
    print(f"    F倒 (uF<=-0.707): {ever(lambda r: r['uF'] <= -COS45)}")
    print(f"    H正 (uH>=+0.707): {ever(lambda r: r['uH'] >= COS45)}")
    print(f"    F正 (uF>=+0.707): {ever(lambda r: r['uF'] >= COS45)}")
    print(f"    zF<0.03         : {ever(lambda r: r['zF'] < 0.03)}")
    print(f"    zH<0.03         : {ever(lambda r: r['zH'] < 0.03)}")
    print(f"    zF<0.04         : {ever(lambda r: r['zF'] < 0.04)}")
    print(f"    zH<0.04         : {ever(lambda r: r['zH'] < 0.04)}")

    env.close()


if __name__ == "__main__":
    main()
