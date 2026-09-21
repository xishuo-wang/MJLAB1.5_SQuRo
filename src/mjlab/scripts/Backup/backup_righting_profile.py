from __future__ import annotations
import tyro
import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.SQuRo_Backup_Replay import StateMachinePolicy
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES, resolve_model_indices


# 只读诊断: 不落盘, 全部结果打到 stdout (供沙箱/CI 环境使用)
# 采集开环手调状态机每一步的 关节角 + body 位姿 + 真实包络, 用于量化翻正行为


# 由 XML 载入的独立模型 + 逐 geom 顶点缓存
class EnvelopeProbe:
    def __init__(self) -> None:
        import mujoco
        from mjlab.asset_zoo.robots.SQuRo import __file__ as sq_init
        xml = Path(sq_init).parent / "xmls" / "SQuRo.xml"
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.gids = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] > 0]
        self.cache: dict[int, np.ndarray] = {}
        # 每个 geom 的顶点在"所属 body 局部系"中的表达 (MuJoCo 编译后 geom_pos/quat 已相对 body)
        self.local: dict[int, np.ndarray] = {}
        for g in self.gids:
            did = self.m.geom_dataid[g]
            adr, num = self.m.mesh_vertadr[did], self.m.mesh_vertnum[did]
            v = self.m.mesh_vert[adr:adr + num] * self.m.mesh_scale[did]
            self.local[g] = v
        self.body_of = {g: int(self.m.geom_bodyid[g]) for g in self.gids}
        self._compose()

    # 预计算每个 geom 顶点相对其 body 原点的偏移
    def _compose(self) -> None:
        import mujoco
        self.offsets: dict[int, np.ndarray] = {}
        self.d.qpos[:] = 0.0
        self.d.qpos[3] = 1.0
        mujoco.mj_kinematics(self.m, self.d)
        for g in self.gids:
            R = self.d.geom_xmat[g].reshape(3, 3)
            p = self.d.geom_xpos[g]
            self.offsets[g] = (R @ self.local[g].T).T + p  # body 在原点时的顶点

    # 给定各 body 的世界旋转 R 与位置 p, 输出整机 AABB
    def aabb(self, R_by_body: dict[int, np.ndarray], p_by_body: dict[int, np.ndarray]):
        pts = []
        for g in self.gids:
            b = self.body_of[g]
            pts.append((R_by_body[b] @ self.offsets[g].T).T + p_by_body[b])
        P = np.vstack(pts)
        return P.min(axis=0), P.max(axis=0)


@dataclass(frozen=True)
class ProfileCfg:
    time_scale: float = 3.0
    p1_recover_duration: float = 0.65
    hind_hip: float = -1.5
    duration: float = 8.0
    device: str | None = None


def main() -> None:
    args = tyro.cli(ProfileCfg)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    env_cfg.scene.num_envs = 1
    env_cfg.commands["backup_cmd"].fixed_time_scale = args.time_scale  # type: ignore[attr-defined]
    for term_cfg in env_cfg.terminations.values():
        term_cfg.func = lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env_cfg.episode_length_s = args.duration + 1.0

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()
    asset = env.unwrapped.scene.entities["robot"]
    resolve_model_indices(asset)

    probe = EnvelopeProbe()
    # 仿真 body 顺序与 XML 一致; 按名称建立 名称->id 映射做交叉校验
    import mujoco
    name_to_id = {mujoco.mj_id2name(probe.m, mujoco.mjtObj.mjOBJ_BODY, b): b for b in range(probe.m.nbody)}
    fb_xml, hb_xml = name_to_id["F_body_Link"], name_to_id["H_body_Link"]

    policy = StateMachinePolicy(
        env, args.time_scale, max_retry=5, buffer=0.3,
        p1_recover_duration=args.p1_recover_duration,
        hind_hip=args.hind_hip, quiet=True,
    )

    fb, hb = _MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id
    nbody = asset.data.body_link_quat_w.shape[1]
    rows = []
    n = int(args.duration / env.step_dt)
    with torch.no_grad():
        for _ in range(n):
            action = policy(env.unwrapped.get_observations())
            env.step(action)
            d = asset.data
            # 全部 body 的位姿 (相对世界)
            qa = d.body_link_quat_w[0]                     # [B,4]
            pa = d.body_link_pos_w[0]                      # [B,3]
            R_by_body, p_by_body = {}, {}
            for b in range(min(nbody, probe.m.nbody)):
                w, x, y, z = (float(qa[b, i]) for i in range(4))
                R_by_body[b] = np.array([
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                ])
                p_by_body[b] = pa[b].cpu().numpy().astype(np.float64)
            # 平移到 base 原点, 只保留"相对 base 的包络"
            p0 = p_by_body[0].copy()
            for b in p_by_body:
                p_by_body[b] = p_by_body[b] - p0
            lo, hi = probe.aabb(R_by_body, p_by_body)

            qf, qh = qa[fb], qa[hb]
            up_f = float(2.0 * (qf[2] * qf[3] + qf[0] * qf[1]))
            up_h = float(-2.0 * (qh[2] * qh[3] + qh[0] * qh[1]))
            base_q = d.root_link_quat_w[0]
            bw, bx, by, bz = (float(base_q[i]) for i in range(4))
            roll = float(np.degrees(np.arctan2(2 * (bw * bx + by * bz), 1 - 2 * (bx * bx + by * by))))
            jp = d.joint_pos[0, _MODEL_INDICES.joint_ids]
            rows.append({
                "t": float(policy._elapsed),
                "phase": policy.phase,
                "q": [float(v) for v in jp.tolist()],
                "up_f": up_f, "up_h": up_h,
                "yF": float(pa[fb, 1]), "zF": float(pa[fb, 2]),
                "yH": float(pa[hb, 1]), "zH": float(pa[hb, 2]),
                "base": [float(v) for v in d.root_link_pos_w[0].tolist()],
                "roll_deg": roll,
                "env_y": float(hi[1] - lo[1]), "env_z": float(hi[2] - lo[2]),
                "env_ymin": float(lo[1]), "env_ymax": float(hi[1]),
                "env_zmin": float(lo[2]), "env_zmax": float(hi[2]),
            })
            if policy.phase == "DONE":
                break
    step_dt = float(env.step_dt)
    env.close()

    print("=" * 78)
    print(f"翻正开环诊断  λ={args.time_scale}  恢复段={args.p1_recover_duration}s  step_dt={step_dt:.4f}s")
    print(f"阶段结束: {policy.phase}  重试 P1={policy.retry['P1']} P2={policy.retry['P2']}  步数={len(rows)}")
    print("=" * 78)

    t = np.array([r["t"] for r in rows])
    def arr(k): return np.array([r[k] for r in rows])

    marks = [0.0]
    for i in range(1, len(t)):
        if rows[i]["phase"] != rows[i - 1]["phase"]:
            marks.append(t[i])
    marks.append(t[-1])
    print("\n[阶段时序] " + "  ".join(
        f"{rows[max(0, int(np.searchsorted(t, m)) - 1)]['phase']}@{m:.2f}s" for m in marks))

    print("\n[关键姿态]")
    print(f"{'t':>6} {'phase':>5} {'up_F':>7} {'up_H':>7} {'yF':>8} {'zF':>7} {'yH':>8} {'zH':>7} "
          f"{'roll':>7} {'envW':>7} {'envH':>7}")
    for tt in np.linspace(0, t[-1], 16):
        i = int(np.argmin(np.abs(t - tt)))
        r = rows[i]
        print(f"{r['t']:6.2f} {r['phase']:>5} {r['up_f']:+7.3f} {r['up_h']:+7.3f} "
              f"{r['yF']:+8.4f} {r['zF']:7.4f} {r['yH']:+8.4f} {r['zH']:7.4f} "
              f"{r['roll_deg']:+7.1f} {r['env_y']:7.4f} {r['env_z']:7.4f}")

    print("\n[横向漂移]")
    print(f"  F 端 y: {rows[0]['yF']:+.4f} -> {rows[-1]['yF']:+.4f}   净漂移 {rows[-1]['yF'] - rows[0]['yF']:+.4f} m")
    print(f"  H 端 y: {rows[0]['yH']:+.4f} -> {rows[-1]['yH']:+.4f}   净漂移 {rows[-1]['yH'] - rows[0]['yH']:+.4f} m")
    print(f"  base y: {rows[0]['base'][1]:+.4f} -> {rows[-1]['base'][1]:+.4f}   "
          f"净漂移 {rows[-1]['base'][1] - rows[0]['base'][1]:+.4f} m")
    print(f"  脊柱两端横向错位 |yF-yH|: 初始 {abs(rows[0]['yF'] - rows[0]['yH']):.4f} "
          f"-> 终值 {abs(rows[-1]['yF'] - rows[-1]['yH']):.4f}")

    ey, ez = arr("env_y"), arr("env_z")
    ymin, ymax = arr("env_ymin"), arr("env_ymax")
    print("\n[真实包络需求] (含腿, 相对 base 的 AABB)")
    print(f"  横向宽度 W: 全程 [{ey.min():.4f}, {ey.max():.4f}]  峰值@t={t[int(np.argmax(ey))]:.2f}s")
    print(f"  竖直高度 H: 全程 [{ez.min():.4f}, {ez.max():.4f}]  峰值@t={t[int(np.argmax(ez))]:.2f}s")
    print(f"  横向左/右极值: {ymin.min():+.4f} / {ymax.max():+.4f}")
    print(f"  → 窄缝下限 = {ey.max():.4f} m ; 低顶棚下限 = {ez.max():.4f} m")

    from mjlab.tasks.SQuRo_Backup.mdp.config import P2_END
    T = P2_END
    mask = t <= T
    names = ["F_body(前扭转)", "F_spine1(侧摆)", "H_spine1(俯仰)", "H_body(后扭转)"]
    thmax = np.array([1.57, 0.60, 0.60, 1.57])
    Q = np.array([r["q"][j] for r in rows for j in [0, 1, 8, 9]]).reshape(len(rows), 4)
    ts = t[mask]
    S = np.array([np.trapezoid(np.abs(Q[mask, i]), ts) for i in range(4)])
    A = S / (thmax * T)
    tau = np.array([
        np.trapezoid(ts * np.abs(Q[mask, i]), ts) / max(np.trapezoid(np.abs(Q[mask, i]), ts), 1e-12) / T
        for i in range(4)
    ])
    print(f"\n[脊柱基元指标]  T = P2_END = {T:.2f}s (动作段)")
    for i, nm in enumerate(names):
        print(f"  {nm:16s} S={S[i]:7.4f} rad·s   A={A[i]:6.4f}   tau={tau[i]:6.4f}")
    C4 = S / S.sum()
    print(f"  四通道 C = [{', '.join(f'{v:.4f}' for v in C4)}]  和={C4.sum():.4f}")
    S3 = np.array([S[1], S[2], S[0] + S[3]])
    C3 = S3 / S3.sum()
    print(f"  三基元 C(侧摆,俯仰,扭转) = [{', '.join(f'{v:.4f}' for v in C3)}]  和={C3.sum():.4f}")
    print(f"  三基元 A(侧摆,俯仰,扭转) = "
          f"[{A[1]:.4f}, {A[2]:.4f}, {S3[2] / ((1.57 + 1.57) * T):.4f}]")
    print(f"  扭转差动 ΔS = S(F_body) - S(H_body) = {S[0] - S[3]:+.4f} rad·s")

    print("\n[JSON]")
    print(json.dumps({"rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
