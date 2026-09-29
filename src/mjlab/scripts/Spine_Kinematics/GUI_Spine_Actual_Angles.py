import argparse
import csv
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots.SQuRo.SQuRo_constants import INIT_STATE, get_spec
from mjlab.tasks.SQuRo_Backup.mdp.spine_deformation import SQuRoSpineDeformation

SPINE_JOINTS = ("F_spine1_joint", "F_body_joint", "H_spine1_joint", "H_body_joint")
ZERO_SPINE: dict[str, float] = dict.fromkeys(SPINE_JOINTS, 0.0)

# 驱动序列: (标签, 时长 s, 该段结束时四个脊柱关节的目标角), 段内线性插值
PHASES: tuple[tuple[str, float, dict[str, float]], ...] = (
    ("归零", 1.0, dict(ZERO_SPINE)),
    ("纯侧摆 +0.6", 1.5, {**ZERO_SPINE, "F_spine1_joint": 0.6}),
    ("纯侧摆 -0.6", 1.5, {**ZERO_SPINE, "F_spine1_joint": -0.6}),
    ("归零", 1.0, dict(ZERO_SPINE)),
    ("纯俯仰 +0.6", 1.5, {**ZERO_SPINE, "H_spine1_joint": 0.6}),
    ("纯俯仰 -0.6", 1.5, {**ZERO_SPINE, "H_spine1_joint": -0.6}),
    ("归零", 1.0, dict(ZERO_SPINE)),
    ("纯扭转 反向满量程", 2.0, {**ZERO_SPINE, "F_body_joint": -1.57, "H_body_joint": 1.57}),
    ("归零", 1.0, dict(ZERO_SPINE)),
    ("组合满量程 (解析模型 0.821 姿态)", 2.0,
     {"F_spine1_joint": 0.6, "F_body_joint": -1.57, "H_spine1_joint": 0.6, "H_body_joint": 1.57}),
    ("归零", 1.0, dict(ZERO_SPINE)),
    ("扭转对齐·全落入侧摆 (实测上限)", 2.0,
     {"F_spine1_joint": -0.6, "F_body_joint": 0.0, "H_spine1_joint": -0.6, "H_body_joint": -1.57}),
    ("归零", 1.0, dict(ZERO_SPINE)),
    ("扭转对齐·全落入俯仰 (实测上限)", 2.0,
     {"F_spine1_joint": -0.6, "F_body_joint": 0.785, "H_spine1_joint": -0.6, "H_body_joint": 0.785}),
    ("归零", 1.0, dict(ZERO_SPINE)),
)


# 用原生 MjData 伪装 SQuRoSpineDeformation 需要的实体接口, 不依赖 mujoco_warp
def make_reader(model: mujoco.MjModel, data: mujoco.MjData):
  robot = SimpleNamespace(data=SimpleNamespace())
  robot.find_bodies = lambda names, preserve_order: ([model.body(n).id for n in names], list(names))
  refresh(robot, data)
  return SQuRoSpineDeformation(robot), robot


# 把当前仿真位姿搬进接口读取的那两个张量
def refresh(robot, data: mujoco.MjData) -> None:
  robot.data.body_link_pos_w = torch.from_numpy(data.xpos.copy()).unsqueeze(0)
  robot.data.body_link_quat_w = torch.from_numpy(data.xquat.copy()).unsqueeze(0)


# 三点折线的总弯曲角 (前后段朝向夹角), 与解析模型的"等效弯曲角"同口径
def total_bend(robot, body_ids: tuple[int, int, int]) -> float:
  p = robot.data.body_link_pos_w[0, list(body_ids)].numpy()
  u_f, u_h = p[0] - p[1], p[1] - p[2]
  u_f = u_f / max(np.linalg.norm(u_f), 1e-12)
  u_h = u_h / max(np.linalg.norm(u_h), 1e-12)
  return float(np.arccos(np.clip(float(u_f @ u_h), -1.0, 1.0)))


# 把 INIT_STATE 的显式关节角写进 qpos, 未列出的一律归零 (通配项)
def init_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  data.qpos[:] = 0.0
  data.qpos[3] = 1.0
  for name, value in INIT_STATE.joint_pos.items():
    if name != ".*":
      data.qpos[model.jnt_qposadr[model.joint(name).id]] = float(value)


# 非脊柱执行器保持 INIT_STATE 姿态, 脊柱执行器由序列给出
def hold_ctrl(model: mujoco.MjModel) -> np.ndarray:
  ctrl = np.zeros(model.nu)
  for i in range(model.nu):
    joint = model.joint(int(model.actuator_trnid[i, 0])).name
    ctrl[i] = float(INIT_STATE.joint_pos.get(joint, 0.0))
  return ctrl


# 四个脊柱关节在 actuator 张量里的列号
def spine_ctrl_ids(model: mujoco.MjModel) -> list[int]:
  ids = []
  for i in range(model.nu):
    if model.joint(int(model.actuator_trnid[i, 0])).name in SPINE_JOINTS:
      ids.append(i)
  return ids


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="逐段驱动脊柱并打印实际三形变角")
  parser.add_argument("--interval", type=float, default=0.1, help="打印间隔 (仿真秒)")
  parser.add_argument("--no-viewer", action="store_true", help="不开 MuJoCo 窗口")
  parser.add_argument("--free-base", action="store_true", help="不固定基座, 让机器人自由落地")
  parser.add_argument("--base-height", type=float, default=0.15, help="固定基座时的高度 (m)")
  parser.add_argument("--csv", type=str, default="", help="把逐帧读数写入该 CSV")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  model = get_spec().compile()
  data = mujoco.MjData(model)
  init_qpos(model, data)
  mujoco.mj_forward(model, data)

  reader, robot = make_reader(model, data)
  body_ids = reader.body_ids
  spine_ids = spine_ctrl_ids(model)
  base_ctrl = hold_ctrl(model)
  dt = float(model.opt.timestep)

  print(f"模型: nu={model.nu} nq={model.nq} dt={dt}s  脊柱 actuator 列号={spine_ids}")
  print(f"打印间隔 {args.interval}s; 基座{'自由' if args.free_base else f'固定 (z={args.base_height})'}")
  print("名义角 = 关节目标角; 实际角 = 三点折线在平均扭转平面内的投影; 总弯曲 = 前后段朝向夹角")
  print("提示: 解析模型 GUI_Spine_Kinematics.py 的 0.821 rad 是"
        "arccos(cos0.6*cos0.6) 的总弯曲角, 且假设连杆直接绕基座中心转")

  rows: list[dict[str, float | str]] = []
  peaks: dict[str, list[float]] = {}
  viewer = None
  if not args.no_viewer:
    from mujoco import viewer as mujoco_viewer
    viewer = mujoco_viewer.launch_passive(model, data)

  next_wall = time.perf_counter()
  next_print = 0.0
  sim_t = 0.0
  try:
    for label, duration, target in PHASES:
      start = {name: float(data.qpos[model.jnt_qposadr[model.joint(name).id]]) for name in SPINE_JOINTS}
      phase_t = 0.0
      while phase_t < duration:
        mix = min(1.0, phase_t / max(duration, 1e-9))
        data.ctrl[:] = base_ctrl
        for k, name in enumerate(SPINE_JOINTS):
          data.ctrl[spine_ids[k]] = start[name] + mix * (target[name] - start[name])
        if not args.free_base:
          data.qpos[:3] = (0.0, 0.0, args.base_height)
          data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
          data.qvel[:6] = 0.0
        mujoco.mj_step(model, data)

        refresh(robot, data)
        result = reader.compute()
        valid = bool(result.valid[0])
        lateral, sagittal, axial = (float(result.angles[0, i]) for i in range(3))
        unwrapped = float(result.axial_unwrapped[0])
        bend = total_bend(robot, body_ids)
        nominal = [float(data.ctrl[spine_ids[k]]) for k in range(4)]

        if sim_t + 1e-9 >= next_print:
          next_print = sim_t + args.interval
          if valid:
            print(f" t={sim_t:6.2f}s [{label}] "
                  f"名义[侧摆{nominal[0]:+.2f} 俯仰{nominal[2]:+.2f} 前扭{nominal[1]:+.2f} 后扭{nominal[3]:+.2f}] "
                  f"实际 侧摆{lateral:+.4f} 俯仰{sagittal:+.4f} 扭转{unwrapped:+.4f} 总弯曲{bend:.4f}")
          else:
            print(f" t={sim_t:6.2f}s [{label}] 无效样本 (几何退化或扭转分支歧义)")

        peak = peaks.setdefault(label, [0.0, 0.0, 0.0, 0.0])
        if valid:
          for i, value in enumerate((abs(lateral), abs(sagittal), abs(unwrapped), bend)):
            peak[i] = max(peak[i], value)
        rows.append({
            "t": sim_t, "phase": label, "valid": int(valid),
            "nominal_f_spine1": nominal[0], "nominal_h_spine1": nominal[2],
            "nominal_f_body": nominal[1], "nominal_h_body": nominal[3],
            "lateral": lateral, "sagittal": sagittal,
            "axial": axial, "axial_unwrapped": unwrapped, "total_bend": bend,
        })

        sim_t += dt
        phase_t += dt
        if viewer is not None:
          viewer.sync()
          if not viewer.is_running():
            return
          next_wall += dt
          delay = next_wall - time.perf_counter()
          if delay > 0:
            time.sleep(delay)
          else:
            next_wall = time.perf_counter()
  finally:
    if viewer is not None:
      viewer.close()

  print("\n各段实测峰值 (rad); 归零段只是回位过渡, 不计")
  print(f"  {'段':<34}{'侧摆':>9}{'俯仰':>9}{'扭转':>9}{'总弯曲':>9}")
  for label, _duration, _target in PHASES:
    p = peaks.get(label)
    if p is not None and label != "归零":
      print(f"  {label:<34}{p[0]:>9.4f}{p[1]:>9.4f}{p[2]:>9.4f}{p[3]:>9.4f}")

  if args.csv:
    with open(args.csv, "w", newline="", encoding="utf-8") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
      writer.writeheader()
      writer.writerows(rows)
    print(f"\n逐帧读数已写入 {args.csv}")


if __name__ == "__main__":
  main()
