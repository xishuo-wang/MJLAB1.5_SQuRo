from __future__ import annotations
import torch
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING
from .config import (
    FL_HOLD,
    HL_HOLD,
    LEG_INIT,
    P1_END,
    P1_SPAN,
    P2_SPAN,
    REFERENCE_TOTAL_TIME,
    STAND_GROUND_HEIGHT,
    STAND_TARGET_HEIGHT,
    T0,
    T1,
    T2,
    T3,
    T4,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# 前置收腿段在参考表时间轴上的绝对位置 (本文件与 verify_backup_config 用)。
P1_ONSET = T0                                # T1 起点
P2_ONSET = T0 + T1 + T2                      # T3 起点
P3_ONSET = T0 + T1 + T2 + T3                 # T4 起点

# 躯干姿态参考的端点与翻正时刻 (量 = 背腹轴世界 Z 余弦 u, -1 仰卧 / +1 俯卧)。
ATTITUDE_SUPINE_U = -1.0              # 仰卧时的姿态余弦
ATTITUDE_PRONE_U = 1.0                # 俯卧时的姿态余弦
ATTITUDE_H_HOLD_T = 0.0               # 后段开始翻正的名义时刻
ATTITUDE_H_RIGHTED_T = 0.30           # 后段翻正完成 (须早于 P1 段末)
ATTITUDE_F_HOLD_T = 0.78              # 前段开始翻正 (窗口窄是实测如此)
ATTITUDE_F_RIGHTED_T = P2_SPAN         # 前段翻正完成 = P2 段末(段内), 使 P1 末参考正好是 S1 姿态

# 录制身体轨迹表的采集时序 (只在本文件用于把新动作时间映射回旧采集时间)。
BODY_TRAJ_BUILD_END = 0.65            # 录制表中 T1 末端
BODY_TRAJ_P1_END = 0.80               # 录制表中 P1 末端

# 参考表总长 = 累计口径的 REFERENCE_TOTAL_TIME (前置收腿 0.50 + 动作 0.95 + 过渡 0.50 + 保持 1.05)。
REF_TOTAL_TIME = REFERENCE_TOTAL_TIME
_REF_DT = 0.005             # 参考表分辨率 (s)
_ACTION_END = P3_ONSET      # 三段动作结束 (含前置段后的绝对时刻)
_SEG1_END = P1_ONSET                        # T1 起点 = 0.50
_SEG1_TAIL = P1_ONSET + T1                   # T1 末端 = T2 起点
_SEG2_END = P2_ONSET                        # T2 末端 = T3 起点 = 1.30
_TRANS_END = P3_ONSET + T4

# 身体轨迹表 (开环重放 λ=1 记录 F/H body 世界 y/z + 站起后理想化):
# 列: [t_nom, yF, zF, yH, zH] — 用作时变期望高度与走廊参考中心
# 注意: 该表的 z 只在 P1~P2 使用; P3(起立+站立)的 z 由下方解析斜坡取代, 理由见
# 技术细节"2026-09-19 训练诊断" §结论 3: 录制表在 T4 中段把期望高度拉回 0.0248(=趴平),
# 到 1.365 才升到 0.055, 于是"保持趴姿"在该段几乎是最优解、而要求站立的那一小段
# 反而给出最低核值 —— 参考方向与任务目标相反。y 仍取录制值(目前无人消费)。
_BODY_TRAJ_PATH = Path(__file__).parent / "Bio_Data" / "backup_body_traj.npy"
_body_traj_cache: dict = {}




# 生成参考表
def _generate_reference_table(*, p2_endpoint: bool = False) -> tuple[np.ndarray, np.ndarray]:
    n = int(REF_TOTAL_TIME / _REF_DT) + 1
    t_grid = np.linspace(0.0, REF_TOTAL_TIME, n)
    ref = np.zeros((n, 14), dtype=np.float64)
    leg_col = (4, 5, 6, 7, 10, 11, 12, 13)      # MJLAB 中腿列: FL/FR 4-7, HL/HR 10-13
    spn_col = (0, 1, 8, 9)                      # MJLAB 中脊柱列
    leg_init = np.array(LEG_INIT, dtype=np.float64)

    for i, tn in enumerate(t_grid):
        leg = leg_init.copy()
        f_sp1, f_bd, h_sp1, h_bd = 0.0, 0.0, 0.0, 0.0
        if tn < _SEG1_END:
            # 前置收腿段: 脊柱与颈部保持 0, 腿从 LEG_INIT 线性插到支撑角。
            u = (tn / _SEG1_END) if _SEG1_END > 0.0 else 1.0
            for c in range(8):
                hold = FL_HOLD[c % 2] if c < 4 else HL_HOLD[c % 2]
                leg[c] = leg_init[c] + u * (hold - leg_init[c])
        # P2 专用表在边界保留 T3 左极限；普通表在同一时间点取 T4 起点。
        at_p2_end = p2_endpoint and abs(tn - _ACTION_END) < 1e-12
        # 下界 _SEG1_END 不可省: 否则前置段会掉进 T4 分支被腿过渡覆盖(实测腿变成 HL_HOLD[1])。
        if _SEG1_END <= tn < _ACTION_END or at_p2_end:
            leg[0], leg[1], leg[2], leg[3] = FL_HOLD[0], FL_HOLD[1], FL_HOLD[0], FL_HOLD[1]
            leg[4], leg[5], leg[6], leg[7] = HL_HOLD[0], HL_HOLD[1], HL_HOLD[0], HL_HOLD[1]
            if tn < _SEG1_TAIL:
                u = (tn - _SEG1_END) / (_SEG1_TAIL - _SEG1_END)
                f_sp1 = 0.6 * u
                f_bd = -1.57 * u
                h_sp1 = 0.6 * u
                h_bd = 1.57 * u
            elif tn < _SEG2_END:
                u = (tn - _SEG1_TAIL) / (_SEG2_END - _SEG1_TAIL)
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
        elif _ACTION_END <= tn < _TRANS_END:
            u = (tn - _ACTION_END) / (_TRANS_END - _ACTION_END)
            for c in range(4):
                leg[c] = FL_HOLD[c % 2] + u * (leg_init[c] - FL_HOLD[c % 2])
            for c in range(4, 8):
                leg[c] = HL_HOLD[c % 2] + u * (leg_init[c] - HL_HOLD[c % 2])
            f_sp1 = 0.6 * (1.0 - u)
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
    _, p2_np = _generate_reference_table(p2_endpoint=True)
    p2_pos = torch.tensor(p2_np, device=device, dtype=torch.float32)
    p2_vel = (p2_pos[1:] - p2_pos[:-1]) / _REF_DT
    p2_vel = torch.cat([p2_vel, p2_vel[-1:]])
    cache = {"t": t_t, "pos": pos_t, "vel": vel_t, "p2_pos": p2_pos, "p2_vel": p2_vel}
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



# 将新动作时间映射回旧身体轨迹的采集时间, 只拉伸 P1 回收段 (参考重定时, 不是重新仿真)。
def _body_traj_source_time(t_nom: torch.Tensor) -> torch.Tensor:
    # 传入的是参考表绝对时间; 录制表以 T1 起点为 0, 故先减去 P1_ONSET。
    t_rel = (t_nom - P1_ONSET).clamp(min=0.0)
    tail = T1                                # T1 时长
    recover_fraction = (t_rel - tail) / (P1_SPAN - tail)
    recover_t = BODY_TRAJ_BUILD_END + recover_fraction * (BODY_TRAJ_P1_END - BODY_TRAJ_BUILD_END)
    return torch.where(
        t_rel < tail,
        t_rel * (BODY_TRAJ_BUILD_END / tail),
        torch.where(t_rel < P1_SPAN, recover_t, BODY_TRAJ_P1_END + t_rel - P1_SPAN),
    )



# 获取身体参考轨迹 (时变期望高度 + 走廊中心)，按阶段时间重定时并冻结缓冲期参考。
# P3 的 z 用解析斜坡取代录制值, 见文件头对 _BODY_TRAJ_PATH 的说明。
def get_body_reference(env: "ManagerBasedRlEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cmd_term = env.command_manager._terms["backup_cmd"]  # type: ignore[union-attr]
    lam = cmd_term.command[:, 5].clamp(min=0.1)
    phase = cmd_term.phase  # type: ignore[attr-defined]
    t_phase = cmd_term.stage_t  # type: ignore[attr-defined]
    cache = _get_body_traj(env.device)
    t_nom = _body_traj_source_time(_stage_t_nom(env))
    idx = torch.searchsorted(cache["t"], t_nom).clamp(1, len(cache["t"]) - 1)
    idx_p = idx - 1
    frac = ((t_nom - cache["t"][idx_p]) / (cache["t"][idx] - cache["t"][idx_p] + 1e-12)).clamp(0.0, 1.0)
    out = []
    for key in ("yF", "zF", "yH", "zH"):
        v = cache[key][idx_p] + frac * (cache[key][idx] - cache[key][idx_p])
        out.append(v)
    y_f, z_f, y_h, z_h = out
    # P3 起立段: z 从趴平高度线性抬到站立目标, 斜坡时长 = T4 名义时长。
    # 归 P3 必须用 **phase == 2** 判定, 不能用 source_t >= ACTION_END: P2 播完后进入
    # S2 确认等待段时 _stage_t_nom 会把 source_t 冻结在 0.95, 那样 P2 的末端与等待段
    # 也会被斜坡覆盖(实测 F/H 参考从 0.0445/0.0503 被改成 0.024/0.024), 污染 P2 奖励
    # 并破坏归因。斜坡进度同样取自段内时钟, 与录制表查询时间解耦。
    u = (t_phase / (lam * T4)).clamp(0.0, 1.0)
    z_ramp = STAND_GROUND_HEIGHT + (STAND_TARGET_HEIGHT - STAND_GROUND_HEIGHT) * u
    in_p3 = phase == 2
    z_f = torch.where(in_p3, z_ramp, z_f)
    z_h = torch.where(in_p3, z_ramp, z_h)
    return y_f, z_f, y_h, z_h



# 获取参考躯干姿态 (两段背腹轴的世界 Z 余弦, 与 command._pose_cos 同一量)。
# 不能由关节表直接 cos 得到, 且在余弦域线性插值(非等角速度); 理由见技术细节 §7.8。
def get_reference_body_attitude(env: "ManagerBasedRlEnv") -> torch.Tensor:
    # _stage_t_nom 已是参考表绝对时间; 姿态参考的时间基准是"P1 起点", 故减去 P1_ONSET。
    t_nom = _stage_t_nom(env) - P1_ONSET  # [N], 已按 λ 缩放
    climb = ATTITUDE_PRONE_U - ATTITUDE_SUPINE_U

    def ramp(hold: float, righted: float) -> torch.Tensor:
        # 先保持仰卧到 hold, 再在 [hold, righted] 内升到俯卧, 之后保持。
        span = max(righted - hold, 1e-6)
        prog = ((t_nom - hold) / span).clamp(0.0, 1.0)
        return ATTITUDE_SUPINE_U + climb * prog

    u_f = ramp(ATTITUDE_F_HOLD_T, ATTITUDE_F_RIGHTED_T)
    u_h = ramp(ATTITUDE_H_HOLD_T, ATTITUDE_H_RIGHTED_T)
    return torch.stack((u_f, u_h), dim=1)



# 阶段时间映射到参考表绝对时间(含前置收腿段), 各段加自己的表起点。语义见技术细节 §5.4。
def _stage_t_nom(env: "ManagerBasedRlEnv") -> torch.Tensor:
    cmd_term = env.command_manager._terms["backup_cmd"]  # type: ignore[union-attr]
    lam = cmd_term.command[:, 5].clamp(min=0.1)
    phase = cmd_term.phase  # type: ignore[attr-defined]
    t_phase = cmd_term.stage_t  # type: ignore[attr-defined]
    t_local_nom = t_phase / lam
    # 缓冲期参考不得泄漏到下一段动作; float32 的起点+段长可能超边界一个 ulp, 需再限幅。
    # 相位 0 的时钟就是参考表时间: P0 收腿[0,T0] -> T1 -> T2; 播完后冻结在 P1 段末,
    # 余下的时钟用来等 S1 确认。不可再加 P1_ONSET —— 那会整段跳过 P0, 腿在相位 0 起点
    # 就直接跳到支撑角, 又变成"边收腿边拧脊柱"(实测确认过)。
    p1_t = t_local_nom.clamp(max=P1_END)
    p2_t = P2_ONSET + t_local_nom.clamp(max=T3)
    p3_t = P3_ONSET + t_local_nom
    return torch.where(phase == 0, p1_t, torch.where(phase == 1, p2_t, p3_t))



# 获取参考关节状态
def get_reference_joint_state(env: "ManagerBasedRlEnv") -> tuple[torch.Tensor, torch.Tensor]:
    cache = _get_ref_table(env.device)
    cmd_term = env.command_manager._terms["backup_cmd"]  # type: ignore[union-attr]
    cmd = cmd_term.command
    lam = cmd[:, 5].clamp(min=0.1)  # [N] 第 6 维 time_scale (放慢倍数)
    # _stage_t_nom 已是参考表绝对时间。
    t_nom = _stage_t_nom(env).clamp(0.0, REF_TOTAL_TIME)  # [N]
    # 精确落在节点时取右侧导数, 尤其 P3 起点不能读到 T3->T4 的跳变速度。
    idx = torch.searchsorted(cache["t"], t_nom, right=True).clamp(1, len(cache["t"]) - 1)
    idx_p = idx - 1
    frac = (t_nom - cache["t"][idx_p]) / (cache["t"][idx] - cache["t"][idx_p] + 1e-12)
    pos = cache["pos"][idx_p] + frac.unsqueeze(1) * (cache["pos"][idx] - cache["pos"][idx_p])
    # 参考速度: dref/dt = dref/dt_nom * (1/λ)
    vel = cache["vel"][idx_p] / lam.unsqueeze(1)
    phase = cmd_term.phase  # type: ignore[attr-defined]
    # 同一名义时间有两个边界值: P2 保持 T3 末端, P3 用全零脊柱。
    p2_pos = cache["p2_pos"][idx_p] + frac.unsqueeze(1) * (cache["p2_pos"][idx] - cache["p2_pos"][idx_p])
    p2_vel = cache["p2_vel"][idx_p] / lam.unsqueeze(1)
    pos = torch.where((phase == 1).unsqueeze(1), p2_pos, pos)
    vel = torch.where((phase == 1).unsqueeze(1), p2_vel, vel)
    endpoint_eps = 1e-6
    holding = ((phase == 0) & (t_nom >= _SEG2_END - endpoint_eps)) | ((phase == 1) & (t_nom >= _ACTION_END - endpoint_eps))
    vel = torch.where(holding.unsqueeze(1), torch.zeros_like(vel), vel)
    return pos, vel
