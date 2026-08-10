from __future__ import annotations
import numpy as np
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# =========================================================================================
# 跌倒爬起参考轨迹 — 复刻 D:\Code\SQuRo-MuJoCo\Loco\Loco_Backup_slow1.py 三段式手调动作
# 名义时间轴 (scale=1, 动作 0.95s):
#   0.00-0.65: 段1 脊柱同时展开  F_spine1 0->0.8, F_body 0->-1.57, H_spine1 0->0.8, H_body 0->1.57
#   0.65-0.80: 段2 F/H_spine1 0.8->0 (F/H_body 保持 ±1.57)
#   0.80-0.95: 段3 F_spine1 0->0.8, F_body -1.57->0, H_body 1.57->0
#   0.95-1.45: time5 平滑过渡 — 腿从支撑位线性转到站立角, F_spine1 0.8->0 (避免生硬切换)
#   1.45 之后: 保持站立 (腿站立角, 脊柱 0)
# 全程腿: 段1-3 支撑位 IK(0.007,-0.02)/(-0.07,-0.02), time5 平滑回站立角
# 命令系统: time_scale λ (= scale), 查询 t_nom = t_episode / λ
#   已验证: 纯 MuJoCo scale=1 站起 1.05s; MJLAB warp scale=2 站起 2.0s, scale=3 站起 2.9s
#   注意: MJLAB 的 position 执行器直接输入期望角 (kp=2.5/kv=0.01), 无需手动 PD
# =========================================================================================

REF_TOTAL_TIME = 2.5        # 名义参考总时长 (s, λ=1 基准: 动作 0.95s + 过渡 0.5s + 保持 1.05s)
_REF_DT = 0.005             # 参考表分辨率 (s)
_ACTION_END = 0.95          # 三段动作结束的名义时间 (s)
_SEG1_END = 0.65            # 段1 结束
_SEG2_END = 0.80            # 段2 结束
_TRANS_END = 1.45           # time5 平滑过渡结束 (0.95 + 0.5)

# 站立初始腿角 (FL_sh, FL_el, FR_sh, FR_el, HL_hip, HL_knee, HR_hip, HR_knee)
_LEG_INIT = np.array([0.1, -0.3, 0.1, -0.3, -0.1, 0.3, -0.1, 0.3], dtype=np.float64)


# 腿支撑期望角 — 直接使用手调 slow1/PD 实际值, 尽量减少与参考脚本的不一致
# (IK 精确值 -0.2833/0.5622/-1.4041/-0.2523 vs PD 实际 -0.271/0.550/-1.388/-0.250)
_FL_HOLD = (-0.28, 0.55)   # FL/FR shoulder, elbow
_HL_HOLD = (-1.40, -0.25)  # HL/HR hip, knee


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
        if tn < _ACTION_END:
            # 腿支撑位 (段1-3)
            leg[0], leg[1], leg[2], leg[3] = _FL_HOLD[0], _FL_HOLD[1], _FL_HOLD[0], _FL_HOLD[1]
            leg[4], leg[5], leg[6], leg[7] = _HL_HOLD[0], _HL_HOLD[1], _HL_HOLD[0], _HL_HOLD[1]
            if tn < _SEG1_END:
                u = tn / _SEG1_END
                f_sp1 = 0.8 * u
                f_bd = -1.57 * u
                h_sp1 = 0.8 * u
                h_bd = 1.57 * u
            elif tn < _SEG2_END:
                u = (tn - _SEG1_END) / (_SEG2_END - _SEG1_END)
                f_sp1 = 0.8 - 0.8 * u
                f_bd = -1.57
                h_sp1 = 0.8 - 0.8 * u
                h_bd = 1.57
            else:
                u = (tn - _SEG2_END) / (_ACTION_END - _SEG2_END)
                f_sp1 = 0.8 * u
                f_bd = -1.57 + 1.57 * u
                h_sp1 = 0.0
                h_bd = 1.57 - 1.57 * u
        elif tn < _TRANS_END:
            # time5: 腿支撑位 -> 站立角, F_spine1 0.8 -> 0 (平滑过渡, 避免生硬切换)
            u = (tn - _ACTION_END) / (_TRANS_END - _ACTION_END)
            for c in range(4):
                leg[c] = _FL_HOLD[c % 2] + u * (_LEG_INIT[c] - _FL_HOLD[c % 2])
            for c in range(4, 8):
                leg[c] = _HL_HOLD[c % 2] + u * (_LEG_INIT[c] - _HL_HOLD[c % 2])
            f_sp1 = 0.8 * (1.0 - u)
        # tn >= 1.45: 保持站立 (默认腿站立角, 脊柱 0)

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
