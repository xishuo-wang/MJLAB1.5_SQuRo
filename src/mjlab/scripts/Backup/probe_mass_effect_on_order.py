# 探针: 质量/惯量分布对"哪一端先翻正"的影响 (只读, 不落盘)
# 做法: 用 asset_zoo 的 spec_fn 在内存里给 F(前)/H(后) body 加配重 (不动任何仓库文件),
# 跑同一检查点的确定性策略, 用腹背标记 site 的世界 Z 余弦判定**每一段**何时变为正置,
# 从而得到"前段先翻 / 后段先翻"的顺序。
# 用法:
#   uv run python -B -m mjlab.scripts.Backup.probe_mass_effect_on_order --checkpoint <pt>
#   --add-front 0.10 --add-rear 0.10   (可多次: 每次一组变体)
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro

from mjlab.asset_zoo.robots.SQuRo import SQuRo_constants as SC
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.SQuRo_Backup.mdp import entity as mdp_entity
from mjlab.tasks.SQuRo_Backup.mdp import indices as mdp_indices
from mjlab.tasks.SQuRo_Backup.rl.runner import _STEPS_PER_ITER

UPRIGHT_COS = 0.9      # 与判据一致的正置阈值
GROUND_Z = 0.03        # 该段贴地的高度阈值 (与 S1 一致)


@dataclass
class Cfg:
    checkpoint: str
    payloads: tuple[float, ...] = (0.0,)          # 配重质量 (kg); 0 = 基线
    com_fwd_mm: tuple[float, ...] = ()            # 前段质心沿 F_body 局部 +Z 偏移 (mm, + = 往头)
    com_up_mm: tuple[float, ...] = ()             # 前段质心沿 F_body 局部 -Y 偏移 (mm, + = 往上)
    wall_x_neg: float = -0.20
    wall_x_pos: float = 0.05
    num_envs: int = 32
    steps: int = 1500
    device: str = "cuda:0"
    seed: int = 0


# 坐标系实测 (编译后 xmat): F_body_Link 局部 +Z 指向头, 局部 -Y 指向上。
# 注意 H_body_Link 的上下轴符号相反 (+Y 才是朝上) —— 这里只动前段, 不要照搬。
def shifted_body_inertial(body_name: str, d_fwd: float, d_up: float):
    # 显式惯量覆盖: 质量与惯量保持编译值不变, 只平移 ipos -> 干净隔离质心位置
    def spec_fn() -> mujoco.MjSpec:
        m0 = SC.get_spec().compile()
        i = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, body_name)
        mass = float(m0.body_mass[i])
        ipos = np.array(m0.body_ipos[i], dtype=float)
        diag = np.array(m0.body_inertia[i], dtype=float)
        iq = np.array(m0.body_iquat[i], dtype=float)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, iq)
        R = R.reshape(3, 3)
        Ifull = R @ np.diag(diag) @ R.T
        spec = SC.get_spec()
        b = spec.body(body_name)
        b.explicitinertial = True
        b.mass = mass
        b.ipos = [float(v) for v in (ipos + np.array([0.0, -d_up, d_fwd]))]
        b.fullinertia = [float(Ifull[0, 0]), float(Ifull[1, 1]), float(Ifull[2, 2]),
                         float(Ifull[0, 1]), float(Ifull[0, 2]), float(Ifull[1, 2])]
        return spec
    return spec_fn


# 在 F/H body 的原点各加一个"配重"小球 (contype=0 不产生接触, 只改质量与质心)
def make_spec_fn(front_kg: float, rear_kg: float):
    def spec_fn() -> mujoco.MjSpec:
        spec = SC.get_spec()
        for body_name, mass in (("F_body_Link", front_kg), ("H_body_Link", rear_kg)):
            if mass <= 0.0:
                continue
            body = spec.body(body_name)
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.004],
                          mass=float(mass), pos=[0.0, 0.0, 0.0],
                          contype=0, conaffinity=0, rgba=[1, 0, 0, 1])
        return spec
    return spec_fn


def build_env(cfg: Cfg, spec_fn):
    env_cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.events.pop("init_restricted_space", None)
    mdp_entity.configure_restricted_space(env_cfg, wall_x_neg=cfg.wall_x_neg,
                                          wall_x_pos=cfg.wall_x_pos,
                                          enable_collision=True, fixed_width=True)
    robot_cfg = SC.get_squro_robot_cfg()
    robot_cfg.spec_fn = spec_fn
    entities = dict(env_cfg.scene.entities)
    entities["robot"] = robot_cfg
    env_cfg.scene.entities = entities
    env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
    return env


# 每一段的正置余弦: belly->back 的世界 Z 分量 (与判据同源)
def segment_u(robot, pairs) -> torch.Tensor:
    sp = robot.data.site_pos_w
    out = []
    for belly_id, back_id in pairs:
        d = sp[:, back_id, :] - sp[:, belly_id, :]
        out.append(d[:, 2] / torch.linalg.vector_norm(d, dim=-1).clamp_min(1e-9))
    return torch.stack(out, dim=1)


def main() -> None:
    cfg = tyro.cli(Cfg)
    torch.manual_seed(cfg.seed)
    m = re.search(r"model_(\d+)", Path(cfg.checkpoint).name)
    it = int(m.group(1)) if m else 0

    print("=" * 92)
    print(f"检查点 {Path(cfg.checkpoint).name}  环境 {cfg.num_envs}  步数 {cfg.steps}")
    print(f"判定: 正置余弦 >= {UPRIGHT_COS}; 顺序 = 两段各自**首次**正置的先后")
    print("=" * 92)

    variants = []
    if any(k > 0 for k in cfg.payloads):
        variants.append(("基线 (无配重)", make_spec_fn(0.0, 0.0)))
        for kg in cfg.payloads:
            if kg <= 0:
                continue
            variants.append((f"前配重 {kg:.2f} kg", make_spec_fn(kg, 0.0)))
            variants.append((f"后配重 {kg:.2f} kg", make_spec_fn(0.0, kg)))
    else:
        variants.append(("基线 (无变化)", SC.get_spec))
    # 只动前段(F_body_Link)质心: +Z = 往头, -Y = 往上 (坐标系实测, 见 shifted_body_inertial)
    for d in cfg.com_fwd_mm:
        variants.append((f"质心往头 {d:+.0f} mm",
                         shifted_body_inertial("F_body_Link", d / 1000.0, 0.0)))
    for d in cfg.com_up_mm:
        variants.append((f"质心往上 {d:+.0f} mm",
                         shifted_body_inertial("F_body_Link", 0.0, d / 1000.0)))
    rows = []
    for _ in (0,):
        for tag, sfn in variants:
            env = build_env(cfg, sfn)
            robot = env.scene.entities["robot"]
            mdp_indices.resolve_model_indices(robot)
            pairs = mdp_indices._MODEL_INDICES.segment_belly_back_ids
            assert pairs is not None, "segment_belly_back_ids 未解析"
            wrapped = RslRlVecEnvWrapper(env)
            wrapped.unwrapped.common_step_counter = it * _STEPS_PER_ITER
            runner = load_runner_cls("Mjlab-SQuRo-Backup")(
                wrapped, asdict(load_rl_cfg("Mjlab-SQuRo-Backup")), None, device=cfg.device)
            runner.load(cfg.checkpoint, load_cfg={"actor": True}, strict=True,
                        map_location=cfg.device)
            policy = runner.get_inference_policy(device=cfg.device)

            obs = wrapped.get_observations()
            # 逐环境记录**首次**正置的步号。**不能**用"全部环境同时正置": 32 个环境的相位
            # 不同步, 前段就永远不满足 -> 全部判成"数据不足"(实测踩过)。
            first = torch.full((cfg.num_envs, 2), -1, dtype=torch.long, device=env.device)
            with torch.no_grad():
                for k in range(cfg.steps):
                    act = policy(obs)
                    obs, _, _, _ = wrapped.step(act.to(env.device))
                    u = segment_u(env.scene.entities["robot"], pairs)
                    up = u >= UPRIGHT_COS                    # (n, 2) 每环境每段是否正置
                    newly = up & (first < 0)
                    first[newly] = k
                    if bool((first >= 0).all()):
                        break
            ff, fh = first[:, 0], first[:, 1]
            ok = (ff >= 0) & (fh >= 0)
            n_ok = int(ok.sum())
            if n_ok == 0:
                order, med_f, med_h = "无环境两段都正置", None, None
            else:
                front_first = int((ff[ok] < fh[ok]).sum())
                order = (f"前段先翻 {front_first}/{n_ok}"
                         if front_first * 2 > n_ok else
                         f"后段先翻 {n_ok - front_first}/{n_ok}"
                         if front_first * 2 < n_ok else f"持平 {front_first}/{n_ok}")
                med_f = int(ff[ok].float().median().item())
                med_h = int(fh[ok].float().median().item())
            rows.append((tag, med_f, med_h, order))
            print(f"{tag:20s} 有效环境 {n_ok:2d}/{cfg.num_envs}  前段首次正置中位={str(med_f):>5}  "
                  f"后段={str(med_h):>5}  -> {order}")
            env.close()

    print("\n=== 汇总 ===")
    for tag, ff, fh, order in rows:
        print(f"  {tag:20s} {order}")
    print("\n(步数越小越早; 控制步 dt=0.01 s)")


if __name__ == "__main__":
    main()
