from __future__ import annotations
import sys
import torch
import numpy as np
import mujoco
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.SQuRo_backup_Replay import StateMachinePolicy
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES, resolve_model_indices
import mjlab.asset_zoo.robots.SQuRo as _sq

XML = Path(_sq.__file__).resolve().parent / "xmls" / "SQuRo.xml"
# 各 body 相对 base 的"零位"姿态 (由 SQuRo.xml 固定安装旋转导出)
# 用于把 body 的世界姿态分解成 滚转(绕世界X) / 俯仰(绕世界Y) / 偏航(绕世界Z)
UP_TH = 0.5
GROUND_TH = 0.03


# 把旋转矩阵分解为 ZYX 欧拉角 (度): 返回 (yaw, pitch, roll)
def euler_zyx(R: np.ndarray) -> tuple[float, float, float]:
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = np.degrees(np.arcsin(sy))
    if abs(sy) < 0.9999:
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        yaw = 0.0
    return yaw, pitch, roll


def main() -> None:
    cfg = load_env_cfg("Mjlab-SQuRo-Backup")
    cfg.scene.num_envs = 1
    cfg.commands["backup_cmd"].fixed_time_scale = 1.0  # type: ignore[attr-defined]
    for tc in cfg.terminations.values():
        tc.func = lambda e: torch.zeros(e.num_envs, dtype=torch.bool, device=e.device)
    cfg.episode_length_s = 4.0

    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
    env.reset()
    asset = env.unwrapped.scene.entities["robot"]
    resolve_model_indices(asset)
    pol = StateMachinePolicy(env, 1.0, 5, 0.3, quiet=True)

    # 建立 body 索引 -> 原始 xmat 行
    m = mujoco.MjModel.from_xml_path(str(XML))
    fb = _MODEL_INDICES.f_body_id
    hb = _MODEL_INDICES.h_body_id

    print("说明: 用原始 MuJoCo xmat 计算 body 在世界系的姿态 (无四元数约定歧义)")
    print("      '背腹' = body 背腹轴(局部 +Y, H 取负号) 的世界 Z 分量: +1=腹部朝上, -1=背部朝上")
    print("      '躯干Z' = body 局部 +X (前后轴) 的世界 Z 分量: ±1=躯干竖直, 0=躯干水平")
    print()
    print(f"{'t':>5} {'ph':>3} | {'背腹F':>7} {'背腹H':>7} | {'躯干Fz':>7} {'躯干Hz':>7} | "
          f"{'roll_F':>7} {'roll_H':>7} | {'zF':>7} {'zH':>7} | {'base_roll':>9} | S1_world")
    with torch.no_grad():
        for i in range(200):
            env.step(pol(env.unwrapped.get_observations()))
            # 从 mjlab 的 quat 取 body 世界旋转矩阵 (与 base 复合)
            qb = asset.data.root_link_quat_w[0].cpu().numpy().astype(float)
            qF = asset.data.body_link_quat_w[0, fb].cpu().numpy().astype(float)
            qH = asset.data.body_link_quat_w[0, hb].cpu().numpy().astype(float)

            def quat2mat(q):
                w, x, y, z = q
                return np.array([
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                ])

            RF = quat2mat(qF)
            RH = quat2mat(qH)
            Rbase = quat2mat(qb)
            # 局部 +Y (背腹轴) 的世界 Z; F 直接取, H 取负
            upF = RF[1, 2]
            upH = -RH[1, 2]
            # 前后轴 (局部 +X) 的世界 Z
            axF = RF[0, 2]
            axH = RH[0, 2]
            _, _, rollF = euler_zyx(RF)
            _, _, rollH = euler_zyx(RH)
            _, _, rollB = euler_zyx(Rbase)
            fz = float(asset.data.body_link_pos_w[0, fb, 2])
            hz = float(asset.data.body_link_pos_w[0, hb, 2])
            if i % 10 == 0:
                s1 = (upF > UP_TH) and (upH < -UP_TH) and (fz < GROUND_TH) and (hz < GROUND_TH)
                print(f"{pol._elapsed:5.2f} {pol.phase:>3} | {upF:+7.3f} {upH:+7.3f} | "
                      f"{axF:+7.3f} {axH:+7.3f} | {rollF:+7.1f} {rollH:+7.1f} | "
                      f"{fz:7.4f} {hz:7.4f} | {rollB:+9.1f} | {s1}")
            if pol.phase == "DONE":
                print(f"--- DONE @ t={pol._elapsed:.2f}s  重试 P1={pol.retry['P1']} P2={pol.retry['P2']}")
                break
    env.close()
    _ = m


if __name__ == "__main__":
    main()
