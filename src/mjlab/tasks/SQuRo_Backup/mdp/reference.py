from __future__ import annotations
import torch
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING
from .timing import (
    BODY_TRAJ_BUILD_END,
    BODY_TRAJ_P1_END,
    P1_BUILD_DURATION,
    P1_END,
    P2_END,
    REFERENCE_TOTAL_TIME,
    STAND_TRANSITION_END,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv



REF_TOTAL_TIME = REFERENCE_TOTAL_TIME  # 动作 1.45s + 过渡 0.5s + 保持 1.05s
_REF_DT = 0.005             # 参考表分辨率 (s)
_ACTION_END = P2_END        # 三段动作结束
_SEG1_END = P1_BUILD_DURATION
_SEG2_END = P1_END
_TRANS_END = STAND_TRANSITION_END

# 身体轨迹表 (开环重放 λ=1 记录 F/H body 世界 y/z + 站起后理想化):
# 列: [t_nom, yF, zF, yH, zH] — 用作时变期望高度与走廊参考中心
_BODY_TRAJ_PATH = Path(__file__).parent / "Bio_Data" / "backup_body_traj.npy"
_body_traj_cache: dict = {}

# 站立初始腿角 (FL_sh, FL_el, FR_sh, FR_el, HL_hip, HL_knee, HR_hip, HR_knee)
_LEG_INIT = np.array([0.1, -0.3, 0.1, -0.3, -0.1, 0.3, -0.1, 0.3], dtype=np.float64)


# 腿支撑期望角 — 与手调脚本 (SQuRo_backup_Replay.FL_HOLD / HL_HOLD) 完全一致
_FL_HOLD = (-0.28, 0.55)   # FL/FR shoulder, elbow
_HL_HOLD = (-1.50, -0.25)  # HL/HR hip, knee


# 生成参考表: 返回 (t[np], ref[np, 14]) — MJLAB actuator 顺序
# T2/T3 的系数必须与 scripts/SQuRo_backup_Replay.py 的 slow1_target 一致, 见踩坑记录
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
                f_sp1 = 0.6 - 0.4 * u
                f_bd = -1.57
                h_sp1 = 0.6 - 0.8 * u
                h_bd = 1.57
            else:
                u = (tn - _SEG2_END) / (_ACTION_END - _SEG2_END)
                f_sp1 = 0.2 + 0.4 * u
                f_bd = -1.57 + 1.57 * u
                h_sp1 = -0.2 + 0.2 * u
                h_bd = 1.57 - 1.57 * u
        elif tn < _TRANS_END:
            # T4: 腿支撑位 -> 站立角, 脊柱保持全零，避免前段侧摆在 T3 末再次跳变
            u = (tn - _ACTION_END) / (_TRANS_END - _ACTION_END)
            for c in range(4):
                leg[c] = _FL_HOLD[c % 2] + u * (_LEG_INIT[c] - _FL_HOLD[c % 2])
            for c in range(4, 8):
                leg[c] = _HL_HOLD[c % 2] + u * (_LEG_INIT[c] - _HL_HOLD[c % 2])
            f_sp1 = 0.0
        # 站立过渡结束后保持站立 (默认腿站立角, 脊柱 0)

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


# 将新动作时间映射回旧身体轨迹的采集时间，只拉伸 P1 回收段。
# 保留原始轨迹文件与空间数值；这是参考重定时，不是重新仿真得到的身体轨迹。
def _body_traj_source_time(t_nom: torch.Tensor) -> torch.Tensor:
    recover_fraction = (t_nom - _SEG1_END) / (_SEG2_END - _SEG1_END)
    recover_t = BODY_TRAJ_BUILD_END + recover_fraction * (BODY_TRAJ_P1_END - BODY_TRAJ_BUILD_END)
    return torch.where(
        t_nom < _SEG1_END,
        t_nom * (BODY_TRAJ_BUILD_END / _SEG1_END),
        torch.where(t_nom < _SEG2_END, recover_t, BODY_TRAJ_P1_END + t_nom - _SEG2_END),
    )


# 获取身体参考轨迹 (时变期望高度 + 走廊中心)，按阶段时间重定时并冻结缓冲期参考。
def get_body_reference(env: "ManagerBasedRlEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = _get_body_traj(env.device)
    t_nom = _body_traj_source_time(_stage_t_nom(env))
    idx = torch.searchsorted(cache["t"], t_nom).clamp(1, len(cache["t"]) - 1)
    idx_p = idx - 1
    frac = ((t_nom - cache["t"][idx_p]) / (cache["t"][idx] - cache["t"][idx_p] + 1e-12)).clamp(0.0, 1.0)
    out = []
    for key in ("yF", "zF", "yH", "zH"):
        v = cache[key][idx_p] + frac * (cache[key][idx] - cache[key][idx_p])
        out.append(v)
    return out[0], out[1], out[2], out[3]


# 阶段时间映射到统一名义时间；P2/P3 起点与动作表共享时间常量。
def _stage_t_nom(env: "ManagerBasedRlEnv") -> torch.Tensor:
    cmd_term = env.command_manager._terms["backup_cmd"]  # type: ignore[union-attr]
    lam = cmd_term.command[:, 5].clamp(min=0.1)
    phase = cmd_term.phase  # type: ignore[attr-defined]
    t_phase = cmd_term.stage_t  # type: ignore[attr-defined]
    t_local_nom = t_phase / lam

    # 与手调状态机一致：等待状态门控判定时固定在当前阶段端点，
    # 不让缓冲期参考继续泄漏到下一段动作。
    p1_t = t_local_nom.clamp(max=_SEG2_END)
    p2_t = _SEG2_END + t_local_nom.clamp(max=_ACTION_END - _SEG2_END)
    p3_t = _ACTION_END + t_local_nom
    return torch.where(phase == 0, p1_t, torch.where(phase == 1, p2_t, p3_t))


def get_reference_joint_state(env: "ManagerBasedRlEnv") -> tuple[torch.Tensor, torch.Tensor]:
    cache = _get_ref_table(env.device)
    # 命令 time_scale λ (放慢倍数): λ=1.0 原始速度, λ=1.5 放慢 1.5 倍
    cmd_term = env.command_manager._terms["backup_cmd"]  # type: ignore[union-attr]
    cmd = cmd_term.command
    lam = cmd[:, 5].clamp(min=0.1)  # [N] 第 6 维 time_scale
    t_nom = _stage_t_nom(env)  # [N]
    idx = torch.searchsorted(cache["t"], t_nom).clamp(1, len(cache["t"]) - 1)
    idx_p = idx - 1
    frac = (t_nom - cache["t"][idx_p]) / (cache["t"][idx] - cache["t"][idx_p] + 1e-12)
    pos = cache["pos"][idx_p] + frac.unsqueeze(1) * (cache["pos"][idx] - cache["pos"][idx_p])
    # 参考速度: dref/dt = dref/dt_nom * (1/λ)
    vel = cache["vel"][idx_p] / lam.unsqueeze(1)
    phase = cmd_term.phase  # type: ignore[attr-defined]
    endpoint_eps = 1e-6
    holding = ((phase == 0) & (t_nom >= _SEG2_END - endpoint_eps)) | (
        (phase == 1) & (t_nom >= _ACTION_END - endpoint_eps)
    )
    vel = torch.where(holding.unsqueeze(1), torch.zeros_like(vel), vel)
    return pos, vel
