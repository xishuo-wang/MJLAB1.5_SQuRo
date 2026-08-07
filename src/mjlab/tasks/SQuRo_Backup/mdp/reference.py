from __future__ import annotations
import math
import numpy as np
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


REF_TOTAL_TIME = 7.0        # 参考轨迹总时长 (s)
_REF_DT = 0.005             # 参考表分辨率 (s)

# 站立初始腿角 (FL_sh, FL_el, FR_sh, FR_el, HL_hip, HL_knee, HR_hip, HR_knee)
_LEG_INIT = np.array([0.1, -0.3, 0.1, -0.3, -0.1, 0.3, -0.1, 0.3], dtype=np.float64)


# 逆运动学 — 与参考脚本 Import/Inverse_Kinematics.py inverse_kinematics_old 一致
def _ik(target: tuple[float, float], is_front: bool) -> tuple[float, float]:
    L1, L2 = (0.040, 0.040) if is_front else (0.040, 0.036)
    x, y = target
    R = math.sqrt(x * x + y * y)
    K = (L2 * L2 - x * x - y * y - L1 * L1) / (2.0 * L1)
    theta = math.atan2(y, x)
    phi = math.acos(max(-1.0, min(1.0, K / R)))
    if is_front:
        a1 = theta + phi
        a2 = math.atan2(y + L1 * math.sin(a1), -x - L1 * math.cos(a1))
        if a2 > 2.0:
            a2 -= 2.0 * math.pi
        return (a1 - 0.888, -(a2 + 2.648))
    else:
        a1 = theta - phi
        a2 = math.atan2(x + L1 * math.cos(a1), y + L1 * math.sin(a1))
        if a2 > 2.0:
            a2 -= 2.0 * math.pi
        return (a1 + 4.325, -(a2 + 1.794))


# 腿支撑 IK 目标 (参考脚本): 前腿 (0.007,-0.02), 后腿 (-0.07,-0.02)
_FL_HOLD = _ik((0.007, -0.02), True)
_FL_REACH = _ik((0.005, -0.05), True)
_HL_HOLD = _ik((-0.07, -0.02), False)


# 生成参考表: 返回 (t[np], ref[np, 14]) — MJLAB actuator 顺序
# 顺序: [F_spine1, F_body, Neck_yaw, Neck_pitch,
#        FL_shoulder, FL_elbow, FR_shoulder, FR_elbow,
#        H_spine1, H_body, HL_hip, HL_knee, HR_hip, HR_knee]
def _generate_reference_table() -> tuple[np.ndarray, np.ndarray]:
    n = int(REF_TOTAL_TIME / _REF_DT) + 1
    t_grid = np.linspace(0.0, REF_TOTAL_TIME, n)
    ref = np.zeros((n, 14), dtype=np.float64)

    # 腿部列: 0-1=FL, 2-3=FR, 8-9=HL, 10-11=HR (MJLAB 顺序下腿在 4-7 与 10-13)
    # 简化: 先按 Backup 顺序生成 12 维 [FL_sh,FL_el,FR_sh,FR_el,HL_hip,HL_knee,HR_hip,HR_knee,F_spine1,F_body,H_spine1,H_body]
    # 再映射到 MJLAB 顺序
    leg_col = (4, 5, 6, 7, 10, 11, 12, 13)  # MJLAB 中腿列: FL/FR 4-7, HL/HR 10-13
    spn_col = (0, 1, 8, 9)                 # MJLAB 中脊柱列

    for i, t in enumerate(t_grid):
        # 默认: 站立腿
        leg = _LEG_INIT.copy()
        f_sp1, f_bd, h_sp1, h_bd = 0.0, 0.0, 0.0, 0.0

        if t <= 1.0:
            pass  # 躺地
        elif t <= 1.2:
            # 腿伸向远处支撑 (FL0), H_spine1 上升
            leg[0], leg[1], leg[2], leg[3] = _FL_REACH[0], _FL_REACH[1], _FL_REACH[0], _FL_REACH[1]
            leg[4], leg[5], leg[6], leg[7] = _HL_HOLD[0], _HL_HOLD[1], _HL_HOLD[0], _HL_HOLD[1]
            h_sp1 = (t - 1.0) * 4.0
        elif t <= 1.4:
            leg[0], leg[1], leg[2], leg[3] = _FL_HOLD[0], _FL_HOLD[1], _FL_HOLD[0], _FL_HOLD[1]
            leg[4], leg[5], leg[6], leg[7] = _HL_HOLD[0], _HL_HOLD[1], _HL_HOLD[0], _HL_HOLD[1]
            f_sp1 = -(t - 1.2) * 4.0
            h_sp1 = 0.8
            h_bd = -(t - 1.2) * 7.85
        elif t <= 1.6:
            leg[0], leg[1], leg[2], leg[3] = _FL_HOLD[0], _FL_HOLD[1], _FL_HOLD[0], _FL_HOLD[1]
            leg[4], leg[5], leg[6], leg[7] = _HL_HOLD[0], _HL_HOLD[1], _HL_HOLD[0], _HL_HOLD[1]
            f_sp1 = -0.8
            f_bd = -(t - 1.4) * 7.85
            h_sp1 = 0.8
            h_bd = -1.57
        elif t <= 1.8:
            leg[0], leg[1], leg[2], leg[3] = _FL_HOLD[0], _FL_HOLD[1], _FL_HOLD[0], _FL_HOLD[1]
            leg[4], leg[5], leg[6], leg[7] = _HL_HOLD[0], _HL_HOLD[1], _HL_HOLD[0], _HL_HOLD[1]
            f_sp1 = -0.8 + (t - 1.6) * 4.0
            f_bd = 1.57
            h_sp1 = 0.8 - (t - 1.6) * 4.0
            h_bd = -1.57
        elif t <= 2.0:
            leg[0], leg[1], leg[2], leg[3] = _FL_HOLD[0], _FL_HOLD[1], _FL_HOLD[0], _FL_HOLD[1]
            leg[4], leg[5], leg[6], leg[7] = _HL_HOLD[0], _HL_HOLD[1], _HL_HOLD[0], _HL_HOLD[1]
            f_bd = 1.57
            h_bd = -1.57 + (t - 1.8) * 7.85
        elif t <= 2.2:
            leg[0], leg[1], leg[2], leg[3] = _FL_HOLD[0], _FL_HOLD[1], _FL_HOLD[0], _FL_HOLD[1]
            leg[4], leg[5], leg[6], leg[7] = _HL_HOLD[0], _HL_HOLD[1], _HL_HOLD[0], _HL_HOLD[1]
            f_sp1 = -(t - 2.0) * 4.0
            f_bd = 1.57 - (t - 2.0) * 7.85
        elif t <= 2.4:
            leg[0], leg[1], leg[2], leg[3] = _FL_HOLD[0], _FL_HOLD[1], _FL_HOLD[0], _FL_HOLD[1]
            leg[4], leg[5], leg[6], leg[7] = _HL_HOLD[0], _HL_HOLD[1], _HL_HOLD[0], _HL_HOLD[1]
            f_sp1 = -0.8 + (t - 2.2) * 4.0
        elif t <= 4.2:
            # 保持支撑位
            leg[0], leg[1], leg[2], leg[3] = _FL_HOLD[0], _FL_HOLD[1], _FL_HOLD[0], _FL_HOLD[1]
            leg[4], leg[5], leg[6], leg[7] = _HL_HOLD[0], _HL_HOLD[1], _HL_HOLD[0], _HL_HOLD[1]
        else:
            # 过渡到站立腿
            f = min(1.0, (t - 4.2) / 1.2)
            # 前腿列 (0-3): FL/FR 共用 FL_HOLD 两个角; 后腿列 (4-7): HL/HR 共用 HL_HOLD 两个角
            for c in range(4):
                leg[c] = _FL_HOLD[c % 2] + f * (_LEG_INIT[c] - _FL_HOLD[c % 2])
            for c in range(4, 8):
                leg[c] = _HL_HOLD[c % 2] + f * (_LEG_INIT[c] - _HL_HOLD[c % 2])

        # 映射到 MJLAB 顺序
        for c in range(8):
            ref[i, leg_col[c]] = leg[c]
        ref[i, spn_col[0]] = f_sp1   # F_spine1
        ref[i, spn_col[1]] = f_bd    # F_body
        ref[i, spn_col[2]] = h_sp1   # H_spine1
        ref[i, spn_col[3]] = h_bd    # H_body
        ref[i, 2] = 0.0              # Neck_yaw
        ref[i, 3] = 0.0              # Neck_pitch

    return t_grid, ref


_table_cache: dict = {}


def _get_ref_table(device: str) -> dict:
    if device in _table_cache:
        return _table_cache[device]
    t_np, ref_np = _generate_reference_table()
    t_t = torch.tensor(t_np, device=device, dtype=torch.float32)
    pos_t = torch.tensor(ref_np, device=device, dtype=torch.float32)
    vel_t = (pos_t[1:] - pos_t[:-1]) / _REF_DT
    vel_t = torch.cat([vel_t, vel_t[-1:]])
    cache = {"t": t_t, "pos": pos_t, "vel": vel_t}
    _table_cache[device] = cache
    return cache


# 获取当前步的参考关节位置和速度 — 按 episode 时间查参考表
def get_reference_joint_state(env: "ManagerBasedRlEnv") -> tuple[torch.Tensor, torch.Tensor]:
    cache = _get_ref_table(env.device)
    t = env.episode_length_buf.float() * env.step_dt  # [N]
    idx = torch.searchsorted(cache["t"], t).clamp(1, len(cache["t"]) - 1)
    idx_p = idx - 1
    frac = (t - cache["t"][idx_p]) / (cache["t"][idx] - cache["t"][idx_p] + 1e-12)
    pos = cache["pos"][idx_p] + frac.unsqueeze(1) * (cache["pos"][idx] - cache["pos"][idx_p])
    vel = cache["vel"][idx_p]
    return pos, vel
