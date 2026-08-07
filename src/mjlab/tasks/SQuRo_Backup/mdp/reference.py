from __future__ import annotations
import math
import numpy as np
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# =========================================================================================
# 跌倒爬起参考轨迹 — 真实复刻 D:\Code\SQuRo-MuJoCo\Loco\Loco_Backup_fast1.py (左侧起身)
# 原始时间轴 tf∈[1.2,2.1] (动作 0.9s), 此处映射为名义时间 t_nom = tf-1.2 ∈ [0,0.9]
#   0.0-0.1: H_spine1 0->0.8        (腿伸向远处 FL0/HL)
#   0.1-0.3: F_spine1 0.8->1.6, H_spine1=0.8, H_body 1.6->3.2   (腿支撑 FL1/HL)
#   0.3-0.4: F_spine1=0.8, F_body 1.0->0, H_spine1=0.8, H_body=1.57
#   0.4-0.5: F_spine1=0.8, F_body=-1.57, H_spine1=0.8, H_body=1.57
#   0.5-0.6: F_spine1=0,   F_body=-1.57, H_spine1=0,   H_body=1.57
#   0.6-0.7: F_body=-1.57, H_body 1.57->0
#   0.7-0.8: F_spine1=0.8, F_body -1.57->0, H_body=0
#   0.8-0.9: F_spine1=0.8, F_body=0
#   0.9 之后: 回站立角并保持
# 命令系统: time_scale λ (放慢倍数), 查询 t_nom = t_episode / λ
#   λ=1.0: 原始 fast1 速度 (贴地初始实测 ~0.98s 站起); λ=1.5: 放慢 1.5 倍学习起点
# 已验证 (纯 MuJoCo, SQuRo.xml, 初始 base_z=0.024 贴地): λ=1.0 站起 ~0.98s, λ=1.5 站起 ~1.43s
# =========================================================================================

REF_TOTAL_TIME = 2.0        # 名义参考总时长 (s, λ=1 基准: 动作 0.9s + 保持 1.1s)
_REF_DT = 0.005             # 参考表分辨率 (s)
_ACTION_END = 0.9           # 翻身动作结束的名义时间 (s)

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


# 腿支撑 IK 目标 (参考脚本): 前腿伸远 (0.005,-0.05) / 支撑 (0.007,-0.02), 后腿 (-0.07,-0.02)
_FL_REACH = _ik((0.005, -0.05), True)
_FL_HOLD = _ik((0.007, -0.02), True)
_HL_HOLD = _ik((-0.07, -0.02), False)


# 生成参考表: 返回 (t[np], ref[np, 14]) — MJLAB actuator 顺序
# 顺序: [F_spine1, F_body, Neck_yaw, Neck_pitch,
#        FL_shoulder, FL_elbow, FR_shoulder, FR_elbow,
#        H_spine1, H_body, HL_hip, HL_knee, HR_hip, HR_knee]
def _generate_reference_table() -> tuple[np.ndarray, np.ndarray]:
    n = int(REF_TOTAL_TIME / _REF_DT) + 1
    t_grid = np.linspace(0.0, REF_TOTAL_TIME, n)
    ref = np.zeros((n, 14), dtype=np.float64)
    leg_col = (4, 5, 6, 7, 10, 11, 12, 13)  # MJLAB 中腿列: FL/FR 4-7, HL/HR 10-13
    spn_col = (0, 1, 8, 9)                   # MJLAB 中脊柱列

    for i, tn in enumerate(t_grid):
        leg = _LEG_INIT.copy()
        f_sp1, f_bd, h_sp1, h_bd = 0.0, 0.0, 0.0, 0.0
        if tn < 0.1:
            # 腿伸向远处 (FL0), H_spine1 上升
            leg[0], leg[1], leg[2], leg[3] = _FL_REACH[0], _FL_REACH[1], _FL_REACH[0], _FL_REACH[1]
            leg[4], leg[5], leg[6], leg[7] = _HL_HOLD[0], _HL_HOLD[1], _HL_HOLD[0], _HL_HOLD[1]
            h_sp1 = tn * 8.0
        elif tn < 0.3:
            leg[0], leg[1], leg[2], leg[3] = _FL_HOLD[0], _FL_HOLD[1], _FL_HOLD[0], _FL_HOLD[1]
            leg[4], leg[5], leg[6], leg[7] = _HL_HOLD[0], _HL_HOLD[1], _HL_HOLD[0], _HL_HOLD[1]
            f_sp1 = (tn + 0.1) * 4.0
            h_sp1 = 0.8
            h_bd = (tn + 0.1) * 8.0
        elif tn < 0.4:
            leg[0], leg[1], leg[2], leg[3] = _FL_HOLD[0], _FL_HOLD[1], _FL_HOLD[0], _FL_HOLD[1]
            leg[4], leg[5], leg[6], leg[7] = _HL_HOLD[0], _HL_HOLD[1], _HL_HOLD[0], _HL_HOLD[1]
            f_sp1 = 0.8
            f_bd = (0.4 - tn) * 10.0   # F_body 1.0 -> 0
            h_sp1 = 0.8
            h_bd = 1.57
        elif tn < 0.5:
            leg[0], leg[1], leg[2], leg[3] = _FL_HOLD[0], _FL_HOLD[1], _FL_HOLD[0], _FL_HOLD[1]
            leg[4], leg[5], leg[6], leg[7] = _HL_HOLD[0], _HL_HOLD[1], _HL_HOLD[0], _HL_HOLD[1]
            f_sp1 = 0.8
            f_bd = -1.57
            h_sp1 = 0.8
            h_bd = 1.57
        elif tn < 0.6:
            leg[0], leg[1], leg[2], leg[3] = _FL_HOLD[0], _FL_HOLD[1], _FL_HOLD[0], _FL_HOLD[1]
            leg[4], leg[5], leg[6], leg[7] = _HL_HOLD[0], _HL_HOLD[1], _HL_HOLD[0], _HL_HOLD[1]
            f_bd = -1.57
            h_bd = 1.57
        elif tn < 0.7:
            leg[0], leg[1], leg[2], leg[3] = _FL_HOLD[0], _FL_HOLD[1], _FL_HOLD[0], _FL_HOLD[1]
            leg[4], leg[5], leg[6], leg[7] = _HL_HOLD[0], _HL_HOLD[1], _HL_HOLD[0], _HL_HOLD[1]
            f_bd = -1.57
            h_bd = 1.57 - (tn - 0.6) * 15.7
        elif tn < 0.8:
            leg[0], leg[1], leg[2], leg[3] = _FL_HOLD[0], _FL_HOLD[1], _FL_HOLD[0], _FL_HOLD[1]
            leg[4], leg[5], leg[6], leg[7] = _HL_HOLD[0], _HL_HOLD[1], _HL_HOLD[0], _HL_HOLD[1]
            f_sp1 = 0.8
            f_bd = -1.57 + (tn - 0.7) * 15.7
            h_bd = 0.0
        elif tn < _ACTION_END:
            leg[0], leg[1], leg[2], leg[3] = _FL_HOLD[0], _FL_HOLD[1], _FL_HOLD[0], _FL_HOLD[1]
            leg[4], leg[5], leg[6], leg[7] = _HL_HOLD[0], _HL_HOLD[1], _HL_HOLD[0], _HL_HOLD[1]
            f_sp1 = 0.8
        # tn >= 0.9: 回站立角 (默认腿站立角, 脊柱 0)

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


# 获取当前步的参考关节位置和速度 — 按 episode 时间 / λ 查参考表
def get_reference_joint_state(env: "ManagerBasedRlEnv") -> tuple[torch.Tensor, torch.Tensor]:
    cache = _get_ref_table(env.device)
    # 命令 time_scale λ (放慢倍数): λ=1.0 原始速度, λ=1.5 放慢 1.5 倍
    cmd = env.command_manager._terms["backup_cmd"].command  # type: ignore[union-attr]
    lam = cmd[:, 5].clamp(min=0.1)  # [N] 第 6 维 time_scale
    t_nom = (env.episode_length_buf.float() * env.step_dt) / lam  # [N]
    idx = torch.searchsorted(cache["t"], t_nom).clamp(1, len(cache["t"]) - 1)
    idx_p = idx - 1
    frac = (t_nom - cache["t"][idx_p]) / (cache["t"][idx] - cache["t"][idx_p] + 1e-12)
    pos = cache["pos"][idx_p] + frac.unsqueeze(1) * (cache["pos"][idx] - cache["pos"][idx_p])
    # 参考速度: dref/dt = dref/dt_nom * (1/λ)
    vel = cache["vel"][idx_p] / lam.unsqueeze(1)
    return pos, vel
