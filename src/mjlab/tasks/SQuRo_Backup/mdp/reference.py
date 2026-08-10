from __future__ import annotations
import torch
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv



REF_TOTAL_TIME = 2.5        # 名义参考总时长 (s, λ=1 基准: 动作 0.95s + 过渡 0.5s + 保持 1.05s)
_REF_DT = 0.005             # 参考表分辨率 (s)
_ACTION_END = 0.95          # 三段动作结束的名义时间 (s)
_SEG1_END = 0.65            # 段1 结束
_SEG2_END = 0.80            # 段2 结束
_TRANS_END = 1.45           # time5 平滑过渡结束 (0.95 + 0.5)

# 身体轨迹表 (开环重放 λ=1 记录 F/H body 世界 y/z + 站起后理想化):
# 列: [t_nom, yF, zF, yH, zH] — 用作时变期望高度与走廊参考中心
_BODY_TRAJ_PATH = Path(__file__).parent / "Bio_Data" / "backup_body_traj.npy"
_body_traj_cache: dict = {}

# 站立初始腿角 (FL_sh, FL_el, FR_sh, FR_el, HL_hip, HL_knee, HR_hip, HR_knee)
_LEG_INIT = np.array([0.1, -0.3, 0.1, -0.3, -0.1, 0.3, -0.1, 0.3], dtype=np.float64)


# 腿支撑期望角 — 直接使用手调 slow1/PD 实际值, 尽量减少与参考脚本的不一致
# (IK 精确值 -0.2833/0.5622/-1.4041/-0.2523 vs PD 实际 -0.271/0.550/-1.388/-0.250)
_FL_HOLD = (-0.28, 0.55)   # FL/FR shoulder, elbow
_HL_HOLD = (-1.40, -0.25)  # HL/HR hip, knee


# 生成参考表: 返回 (t[np], ref[np, 14]) — MJLAB actuator 顺序
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
                f_sp1 = 0.6 * u
                f_bd = -1.57 * u
                h_sp1 = 0.6 * u
                h_bd = 1.57 * u
            elif tn < _SEG2_END:
                u = (tn - _SEG1_END) / (_SEG2_END - _SEG1_END)
                f_sp1 = 0.6 - 0.6 * u
                f_bd = -1.57
                h_sp1 = 0.6 - 0.6 * u
                h_bd = 1.57
            else:
                u = (tn - _SEG2_END) / (_ACTION_END - _SEG2_END)
                f_sp1 = 0.6 * u
                f_bd = -1.57 + 1.57 * u
                h_sp1 = 0.0
                h_bd = 1.57 - 1.57 * u
        elif tn < _TRANS_END:
            # time5: 腿支撑位 -> 站立角, F_spine1 0.6 -> 0 (平滑过渡, 避免生硬切换)
            u = (tn - _ACTION_END) / (_TRANS_END - _ACTION_END)
            for c in range(4):
                leg[c] = _FL_HOLD[c % 2] + u * (_LEG_INIT[c] - _FL_HOLD[c % 2])
            for c in range(4, 8):
                leg[c] = _HL_HOLD[c % 2] + u * (_LEG_INIT[c] - _HL_HOLD[c % 2])
            f_sp1 = 0.6 * (1.0 - u)
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


# 加载身体轨迹表 (按设备缓存)
def _get_body_traj(device: str) -> dict:
    if device in _body_traj_cache:
        return _body_traj_cache[device]
    arr = np.load(_BODY_TRAJ_PATH)  # [N,5]
    cache = {
        "t": torch.tensor(arr[:, 0], device=device, dtype=torch.float32),
        "yF": torch.tensor(arr[:, 1], device=device, dtype=torch.float32),
        "zF": torch.tensor(arr[:, 2], device=device, dtype=torch.float32),
        "yH": torch.tensor(arr[:, 3], device=device, dtype=torch.float32),
        "zH": torch.tensor(arr[:, 4], device=device, dtype=torch.float32),
    }
    _body_traj_cache[device] = cache
    return cache


# 获取身体参考轨迹 (时变期望高度 + 走廊中心): 返回 (yF, zF, yH, zH) [N]
# 时间缩放与关节参考一致 (t_nom = t_episode / λ), 超出表范围取末值 (站起后理想值)
def get_body_reference(env: "ManagerBasedRlEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = _get_body_traj(env.device)
    cmd = env.command_manager._terms["backup_cmd"].command  # type: ignore[union-attr]
    lam = cmd[:, 5].clamp(min=0.1)  # [N]
    t_nom = _stage_t_nom(env)
    idx = torch.searchsorted(cache["t"], t_nom).clamp(1, len(cache["t"]) - 1)
    idx_p = idx - 1
    frac = ((t_nom - cache["t"][idx_p]) / (cache["t"][idx] - cache["t"][idx_p] + 1e-12)).clamp(0.0, 1.0)
    out = []
    for key in ("yF", "zF", "yH", "zH"):
        v = cache[key][idx_p] + frac * (cache[key][idx] - cache[key][idx_p])
        out.append(v)
    return out[0], out[1], out[2], out[3]


# 获取当前步的参考关节位置和速度 — 按 episode 时间 / λ 查参考表
# ???? -> ??????: P1 t_phase/lam, P2 0.8+t_phase/lam, P3 0.95+t_phase/lam
def _stage_t_nom(env: "ManagerBasedRlEnv") -> torch.Tensor:
    cmd_term = env.command_manager._terms["backup_cmd"]  # type: ignore[union-attr]
    lam = cmd_term.command[:, 5].clamp(min=0.1)
    phase = cmd_term.phase  # type: ignore[attr-defined]
    t_phase = cmd_term.stage_t  # type: ignore[attr-defined]
    z = torch.zeros_like(lam)
    phase_start = torch.where(phase == 0, z,
                   torch.where(phase == 1, torch.full_like(lam, 0.8), torch.full_like(lam, 0.95)))
    return phase_start + t_phase / lam


def get_reference_joint_state(env: "ManagerBasedRlEnv") -> tuple[torch.Tensor, torch.Tensor]:
    cache = _get_ref_table(env.device)
    # 命令 time_scale λ (放慢倍数): λ=1.0 原始速度, λ=1.5 放慢 1.5 倍
    cmd = env.command_manager._terms["backup_cmd"].command  # type: ignore[union-attr]
    lam = cmd[:, 5].clamp(min=0.1)  # [N] 第 6 维 time_scale
    t_nom = _stage_t_nom(env)  # [N]
    idx = torch.searchsorted(cache["t"], t_nom).clamp(1, len(cache["t"]) - 1)
    idx_p = idx - 1
    frac = (t_nom - cache["t"][idx_p]) / (cache["t"][idx] - cache["t"][idx_p] + 1e-12)
    pos = cache["pos"][idx_p] + frac.unsqueeze(1) * (cache["pos"][idx] - cache["pos"][idx_p])
    # 参考速度: dref/dt = dref/dt_nom * (1/λ)
    vel = cache["vel"][idx_p] / lam.unsqueeze(1)
    return pos, vel
