from __future__ import annotations
import torch
import numpy as np
from mjlab.entity import Entity
from typing import TYPE_CHECKING, Optional
from .indices import _MODEL_INDICES
from .curriculums import (
    BASE_VEL,
    CURVATURE_TARGET,
    SMOOTH_TIME,
    SMOOTH_VEL,
    STRAIGHT_VEL_SCALE,
    VEL_MIN,
    Rmin,
)
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv



# SQuRo 身体尺寸参数
F_BODY_HALF_LENGTH = 0.04       # F_body_Link 在XoY平面上沿身体前后轴半长(m)
F_BODY_HALF_WIDTH  = 0.035      # F_body_Link 在XoY平面左右方向半宽(m)
H_BODY_HALF_LENGTH = 0.04       # H_body_Link 在XoY平面沿身体前后轴半长(m)
H_BODY_HALF_WIDTH  = 0.035      # H_body_Link 在XoY平面左右方向半宽(m)
BODY_REF_OFFSET = 0.04          # F_body/H_body 中心距 base 中心的X轴偏移量
CORRIDOR_HALF_WIDTH = 0.04      # 走廊半宽（基元阶段 = 身体半宽 + 控制余量）
_INIT_DIST = 0.03               # 初始接近段弧长 (m): 机器人从圆弧起点走 _INIT_DIST 到路径起点 (0,0)



_APPROACH_CACHE: dict = {}
_SMOOTH_XSW_CACHE: dict = {}



# 获取指定 body_link 的偏航角
def get_body_heading(env: "ManagerBasedRlEnv", body_id: int | None = None) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    if body_id is None:
        quat = asset.data.root_link_quat_w
    else:
        quat = asset.data.body_link_quat_w[:, body_id]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sin_h = 2.0 * (w * z + x * y)
    cos_h = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(sin_h, cos_h)



# F_body_Link 物理前向 heading（body+X ≠ 物理前向，需减 π/2）
def get_f_body_physical_heading(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return get_body_heading(env, _MODEL_INDICES.f_body_id) - (torch.pi / 2)



# H_body_Link 物理前向 heading（body+X → world -Y，需加 π/2 修正）
def get_h_body_physical_heading(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return get_body_heading(env, _MODEL_INDICES.h_body_id) + (torch.pi / 2)



# 反推接近段轨迹表; platform_at_end=True 时正向终点落在 κ=-K 平台 (无直行 LUT 的首段)
def _approach_rev_table(s0: float, tr: float, platform_at_end: bool = False) -> dict:
    key = (round(s0, 6), round(tr, 6), bool(platform_at_end))
    if key in _APPROACH_CACHE:
        return _APPROACH_CACHE[key]
    K = CURVATURE_TARGET
    ds = 1e-4
    n = max(2, int(s0 / ds))
    u = np.linspace(0.0, s0, n, endpoint=False)
    trp = min(s0, tr)
    if platform_at_end:
        # 反向 κ: 平台 +K (末端) + 过渡 →0; 正向 = 过渡 0→-K + 平台 -K
        ku = np.where(u < s0 - trp, K, K * (s0 - u) / trp)
    else:
        # 反向 κ: 过渡 0→+K + 平台 +K; 正向 = 平台 -K + 过渡 -K→0 (接直行段)
        ku = np.where(u < trp, K * u / trp, K)
    h = np.cumsum(ku) * ds
    x = -np.cumsum(np.cos(h)) * ds
    y = -np.cumsum(np.sin(h)) * ds
    tbl = {"s": u, "x": x, "y": y, "h": h, "k": ku}
    _APPROACH_CACHE[key] = tbl
    return tbl



# 无直行模式判定 (与 _generate_slalom_lut_smooth_period 的选支口径一致)
def is_no_straight_spacing(spacing: float) -> bool:
    tr = SMOOTH_VEL * SMOOTH_TIME
    _, x_sw_full = _get_smooth_xsw(tr)
    return spacing < 2 * x_sw_full - 1e-6



# 机器人初始位置/朝向 (名义 vel), 供 events.py 重置
def get_approach_start(spacing: float = 0.0) -> tuple[float, float, float]:
    tr = SMOOTH_VEL * SMOOTH_TIME
    tbl = _approach_rev_table(_INIT_DIST, tr, platform_at_end=is_no_straight_spacing(spacing))
    # spacing: 杆间距 (新几何起点右移, 接近段终点 = (spacing, 0))
    return float(tbl["x"][-1]) + spacing, float(tbl["y"][-1]), float(tbl["h"][-1])



# Phase 0 圆弧接近段
def get_phase0_approach(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s0 = _INIT_DIST
    mask = torch.abs(k) > 1e-6
    k_s = torch.where(mask, k, torch.ones_like(k))
    x = torch.where(mask, -torch.sin(k * s0) / k_s, torch.full_like(k, -s0))
    y = torch.where(mask, (1.0 - torch.cos(k * s0)) / k_s, torch.zeros_like(k))
    h = torch.where(mask, -k * s0, torch.zeros_like(k))
    return x, y, h



# 机器人初始位置/朝向 (名义 vel), 与 get_approach_start 同义
def get_phase1_approach(spacing: float = 0.0) -> tuple[float, float, float]:
    return get_approach_start(spacing)



# 路径参考调度 — 根据训练阶段自动选择圆弧或绕杆路径
def compute_path_ref(env: "ManagerBasedRlEnv"):
    cmd_term = env.command_manager._terms["slalom_cmd"]
    if cmd_term.slalom_mode_active:  # type: ignore[union-attr]
        return compute_slalom_path_ref(env)
    return compute_arc_path_ref(env)



# 路径参考计算 — 基元阶段：恒定曲率圆弧
def compute_arc_path_ref(env: "ManagerBasedRlEnv"):
    cmd_term = env.command_manager._terms["slalom_cmd"]
    curvature = cmd_term.command[:, 4]      # [N]
    vel_cmd = cmd_term.command[:, 0]        # [N]
    t = env.episode_length_buf.float() * env.step_dt  # [N]

    # 接近段: 圆弧 (κ=curvature 反推), 距 (0,0) 弧长 u = _INIT_DIST - total_dist, 之后圆弧从原点出发
    total_dist = vel_cmd * t
    in_approach = total_dist < _INIT_DIST
    u = (_INIT_DIST - total_dist).clamp(min=0.0)
    mask = torch.abs(curvature) > 1e-6
    k_s = torch.where(mask, curvature, torch.ones_like(curvature))
    approach_x = torch.where(mask, -torch.sin(curvature * u) / k_s, -u)
    approach_y = torch.where(mask, (1.0 - torch.cos(curvature * u)) / k_s, torch.zeros_like(u))
    approach_heading = torch.where(mask, -curvature * u, torch.zeros_like(u))

    arc_t = torch.clamp(t - _INIT_DIST / vel_cmd.clamp(min=1e-6), min=0.0)
    start_heading = 0.0  # 物理前向 = world +X
    omega = curvature * vel_cmd
    dtheta = omega * arc_t
    arc_heading = start_heading + dtheta

    # 弦长公式：chord = v·t·sinc(dθ/2π)
    chord = vel_cmd * arc_t * torch.sinc(dtheta / (2 * torch.pi))
    arc_x = chord * torch.cos(start_heading + dtheta / 2)   # start_x = 0
    arc_y = chord * torch.sin(start_heading + dtheta / 2)   # start_y = 0

    x_ref = torch.where(in_approach, approach_x, arc_x)
    y_ref = torch.where(in_approach, approach_y, arc_y)
    path_heading = torch.where(in_approach, approach_heading, arc_heading)

    # 世界系期望速度（路径切线方向）
    vx_des = vel_cmd * torch.cos(path_heading)
    vy_des = vel_cmd * torch.sin(path_heading)

    return x_ref, y_ref, vx_des, vy_des, path_heading



# 绕杆路径查找表（LUT）— 使用圆弧拼接方式，与 slalom_path_viz.py 逻辑一致
def _generate_slalom_lut_one_period(X: float, n_arc_pts: int = 15):
    def _arc_np(start, end, r, clockwise, steps=n_arc_pts):
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        mid = (start + end) / 2
        chord_vec = end - start
        chord_len = np.linalg.norm(chord_vec)
        d = np.sqrt(max(0, r**2 - (chord_len / 2)**2))
        perp = np.array([-chord_vec[1], chord_vec[0]]) / chord_len
        sign = -1 if clockwise else 1
        center = mid + sign * d * perp
        v_s = start - center; v_e = end - center
        a_s = np.arctan2(v_s[1], v_s[0]); a_e = np.arctan2(v_e[1], v_e[0])
        if clockwise:
            if a_e > a_s: a_e -= 2 * np.pi
        else:
            if a_e < a_s: a_e += 2 * np.pi
        theta = np.linspace(a_s, a_e, steps)
        return center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)

    r = Rmin
    x0, y0 = 0.0, 0.0
    pts_x, pts_y = [x0], [y0]

    # 直行段采样点数 — 与弧段密度一致
    arc_len = r * np.pi / 2            # 单个 1/4 弧的弧长
    straight_len = X - 2 * r           # 单个直行段长度
    n_straight = max(2, int(straight_len / arc_len * n_arc_pts)) if straight_len > 1e-9 else 0

    # S1: CW ¼ arc (0,0) → (r, -r)  [跳过首点，因与 waypoint 重复]
    ax, ay = _arc_np((x0, y0), (x0 + r, y0 - r), r, clockwise=True)
    pts_x.extend(ax[1:]); pts_y.extend(ay[1:])
    # S2: CCW ¼ arc (r, -r) → (2r, -2r)
    ax, ay = _arc_np((x0 + r, y0 - r), (x0 + 2*r, y0 - 2*r), r, clockwise=False)
    pts_x.extend(ax[1:]); pts_y.extend(ay[1:])
    # S3: straight → (X, -2r)  [多点插值]
    if n_straight > 0:
        sx = np.linspace(x0 + 2*r, x0 + X, n_straight)[1:]  # 跳过首点
        sy = np.full(n_straight - 1, y0 - 2*r)
        pts_x.extend(sx); pts_y.extend(sy)
    # S4: CCW ¼ arc (X, -2r) → (X+r, -r)
    ax, ay = _arc_np((x0 + X, y0 - 2*r), (x0 + X + r, y0 - r), r, clockwise=False)
    pts_x.extend(ax[1:]); pts_y.extend(ay[1:])
    # S5: CW ¼ arc (X+r, -r) → (X+2r, 0)
    ax, ay = _arc_np((x0 + X + r, y0 - r), (x0 + X + 2*r, y0), r, clockwise=True)
    pts_x.extend(ax[1:]); pts_y.extend(ay[1:])
    # S6: straight → (2X, 0)  [多点插值]
    if n_straight > 0:
        sx = np.linspace(x0 + X + 2*r, x0 + 2*X, n_straight)[1:]  # 跳过首点
        sy = np.zeros(n_straight - 1)
        pts_x.extend(sx); pts_y.extend(sy)

    # 累积弧长 + heading + curvature (前向差分)
    pts = np.column_stack([pts_x, pts_y])
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lens)])

    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    headings = np.concatenate([headings, headings[-1:]])  # 最后一点复用

    # 曲率: κ = Δheading / Δarc (弧度差需 wrap 到 [-π, π])
    dtheta = np.diff(headings)
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    kappa_vals = dtheta / (seg_lens + 1e-12)
    kappa_vals = np.concatenate([kappa_vals, kappa_vals[-1:]])

    return arc, pts_x, pts_y, headings, kappa_vals



# 积分 S1 弧段, 返回结束处 x 位移 (κ 首次到 0)
def _measure_s1_xsw(segs, ds=1e-4) -> float:
    s_pts, k_pts, s = [], [], 0.0
    for L, k0, k1 in segs:
        n = max(2, int(L / ds))
        s_pts.extend(np.linspace(s, s + L, n, endpoint=False))
        k_pts.extend(np.linspace(k0, k1, n, endpoint=False))
        s += L
    s_pts.append(s)
    k_pts.append(k_pts[-1])
    k_g = np.array(k_pts)
    h = np.cumsum(k_g) * ds
    return float(np.sum(np.cos(h)) * ds)



# 返回 (x_sw_half, x_sw_full) (按过渡弧长缓存)
def _get_smooth_xsw(tr: float) -> tuple[float, float]:
    key = round(tr, 6)
    if key in _SMOOTH_XSW_CACHE:
        return _SMOOTH_XSW_CACHE[key]
    K = CURVATURE_TARGET
    L90 = np.pi / 2 * Rmin
    platform = L90 - tr
    platform_half = L90 - tr / 2
    x_sw_half = _measure_s1_xsw([(platform_half, -K, -K), (tr, -K, 0.0)])
    x_sw_full = _measure_s1_xsw([(tr, 0.0, -K), (platform, -K, -K), (tr, -K, 0.0)])
    _SMOOTH_XSW_CACHE[key] = (x_sw_half, x_sw_full)
    return x_sw_half, x_sw_full



# 平滑模式下的有效杆间距: 不兼容区间 (2×x_sw_half, 2×x_sw_full) → 无直行最小间距
def get_effective_pole_spacing(spacing: float) -> float:
    tr = SMOOTH_VEL * SMOOTH_TIME
    x_sw_half, x_sw_full = _get_smooth_xsw(tr)
    min_sp = 2 * x_sw_half
    if spacing <= min_sp + 1e-6:
        return min_sp                       # 无直行 (同向连续)
    if spacing >= 2 * x_sw_full - 1e-6:
        return spacing                      # 有直行 (含边界)
    return min_sp



def _generate_slalom_lut_smooth_period(X: float, smooth_time: Optional[float] = None, vel: Optional[float] = None):
    t = SMOOTH_TIME if smooth_time is None else smooth_time
    v = SMOOTH_VEL if vel is None else vel
    K = CURVATURE_TARGET
    tr = v * t
    L90 = np.pi / 2 * Rmin
    platform = L90 - tr             # 全过渡弧段平台 (含进出过渡各 tr)
    platform_half = L90 - tr / 2    # 半过渡弧段平台 (仅单侧过渡)

    def integrate(segs, ds=1e-4):
        s_pts, k_pts, s = [], [], 0.0
        for L, k0, k1 in segs:
            n = max(2, int(L / ds))
            s_pts.extend(np.linspace(s, s + L, n, endpoint=False))
            k_pts.extend(np.linspace(k0, k1, n, endpoint=False))
            s += L
        s_pts.append(s)
        k_pts.append(k_pts[-1])
        s_g = np.array(s_pts)
        k_g = np.array(k_pts)
        h = np.cumsum(k_g) * ds
        x = np.cumsum(np.cos(h)) * ds
        y = np.cumsum(np.sin(h)) * ds
        return s_g, x, y, h, k_g

    # 模式选择: 无直行 (同向连续) vs 有直行 (全过渡)
    x_sw_half, x_sw_full = _get_smooth_xsw(tr)
    if X >= 2 * x_sw_full - 1e-6:
        straight = max(0.0, X - 2 * x_sw_full)   # 边界浮点防御
        segs = [
            (straight / 2, 0.0, 0.0),                            # S_a 直行 y=0 (杆1 上方直行右半, 起点=杆1正上方)
            (tr, 0.0, -K), (platform, -K, -K), (tr, -K, 0.0),   # S1
            (tr, 0.0, K), (platform, K, K), (tr, K, 0.0),       # S2
            (straight, 0.0, 0.0),                                # S_b 直行 y=-2*x_sw (杆2 下方, 中点=杆2 x)
            (tr, 0.0, K), (platform, K, K), (tr, K, 0.0),       # S4
            (tr, 0.0, -K), (platform, -K, -K), (tr, -K, 0.0),   # S5
            (straight / 2, 0.0, 0.0),                            # S_c 直行 y=0 (杆3 上方直行左半, 终点=杆3正上方)
        ]
    else:
        segs = [
            (platform_half, -K, -K), (tr, -K, 0.0),   # S1 (起点直接 -20)
            (tr, 0.0, K), (platform_half, K, K),      # S2 (无出, 同向接 S4)
            (platform_half, K, K), (tr, K, 0.0),      # S4 (无进, 同向接 S2)
            (tr, 0.0, -K), (platform_half, -K, -K),   # S5 (无出, 同向接 S1')
        ]
    return integrate(segs)



# 平滑后名义弧段 x 位移 (有直行模式 S1: 进+平台+出), 用于杆 Y 定位
def get_smooth_x_sw(smooth_time: Optional[float] = None, vel: Optional[float] = None) -> float:
    t = SMOOTH_TIME if smooth_time is None else smooth_time
    v = SMOOTH_VEL if vel is None else vel
    _, x_sw_full = _get_smooth_xsw(v * t)
    return float(x_sw_full)



# 路径参考计算 — 绕杆阶段：圆弧拼接路径（LUT 查表）
def compute_slalom_path_ref(env: "ManagerBasedRlEnv"):
    cmd_term = env.command_manager._terms["slalom_cmd"]
    pole_spacing = cmd_term.active_pole_spacing      # type: ignore[union-attr]
    cmd_tensor = cmd_term.command                    # type: ignore[union-attr]
    dt = env.step_dt
    t = env.episode_length_buf.float() * dt         # [N]

    # 名义时间 τ = t × 自身步频: dt = dτ/gait, 故 τ(s) 表与步频解耦 (推导见 PROJECT.md 绕杆一节)
    base_vel = float(getattr(env, "_slalom_base_vel_scalar", BASE_VEL))
    tau = t * cmd_tensor[:, 3]                      # [N]

    # LUT 几何固定 (平滑弧长与 vel 解耦); τ(s) 表只随 (间距, 基础速度) 变化
    key = (pole_spacing, base_vel)
    cache = getattr(env, "_slalom_lut_cache", None)
    if cache is None or cache.get("key") != key:
        arc_np, xs_np, ys_np, hd_np, kp_np = _generate_slalom_lut_smooth_period(pole_spacing)
        dev = env.device
        # 名义速度剖面 (gait=1): scale(κ) 直行 STRAIGHT_VEL_SCALE → 弯道 VEL_MIN
        scale_np = STRAIGHT_VEL_SCALE - (STRAIGHT_VEL_SCALE - VEL_MIN) * np.abs(kp_np) / CURVATURE_TARGET
        v_nom_np = base_vel * scale_np
        v_mid = 0.5 * (v_nom_np[:-1] + v_nom_np[1:])
        tau_lut_np = np.concatenate([[0.0], np.cumsum(np.diff(arc_np) / np.maximum(v_mid, 1e-6))])
        cache = {
            "key": key,
            "spacing": pole_spacing,
            "arc": torch.tensor(arc_np, device=dev, dtype=torch.float32),
            "x": torch.tensor(xs_np, device=dev, dtype=torch.float32),
            "y": torch.tensor(ys_np, device=dev, dtype=torch.float32),
            "heading": torch.tensor(hd_np, device=dev, dtype=torch.float32),
            "kappa": torch.tensor(kp_np, device=dev, dtype=torch.float32),
            "tau": torch.tensor(tau_lut_np, device=dev, dtype=torch.float32),
            "tau_period": float(tau_lut_np[-1]),
            "period": float(arc_np[-1]),
        }
        env._slalom_lut_cache = cache  # type: ignore[attr-defined]

    arc_lut = cache["arc"]
    x_lut = cache["x"]
    y_lut = cache["y"]
    hd_lut = cache["heading"]
    kappa_lut = cache["kappa"]
    tau_lut = cache["tau"]
    s_period = cache["period"]
    tau_period = cache["tau_period"]

    # 新几何: 周期起点 = 第一根杆 (x=pole_spacing) 正上方, 路径整体右移 pole_spacing
    x_offset = pole_spacing

    # 接近段: 圆弧 (平台 -K + 过渡 -K→0), 名义 τ(u) 表 (正向从起点 u=s0 到终点 u=0)
    app_cache = getattr(env, "_slalom_app_tbl", None)
    if app_cache is None or app_cache.get("key") != key:
        app_tbl = _approach_rev_table(_INIT_DIST, SMOOTH_VEL * SMOOTH_TIME,
                                      platform_at_end=is_no_straight_spacing(pole_spacing))
        app_s_np = app_tbl["s"]                    # 0 → _INIT_DIST (距终点弧长)
        app_k_fwd = -app_tbl["k"]                  # 正向 κ: 起点 -K → 终点 0
        app_scale = STRAIGHT_VEL_SCALE - (STRAIGHT_VEL_SCALE - VEL_MIN) * np.abs(app_k_fwd) / CURVATURE_TARGET
        app_v = base_vel * app_scale
        u_desc = app_s_np[::-1].copy()             # _INIT_DIST → 0 (正向行进方向; copy 避免负 stride)
        v_desc = app_v[::-1].copy()
        v_mid = 0.5 * (v_desc[:-1] + v_desc[1:])
        tau_desc_np = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(u_desc)) / np.maximum(v_mid, 1e-6))])
        app_cache = {name: torch.tensor(arr, device=env.device, dtype=torch.float32)
                     for name, arr in app_tbl.items()}
        app_cache["u_desc"] = torch.tensor(u_desc, device=env.device, dtype=torch.float32)
        app_cache["tau_desc"] = torch.tensor(tau_desc_np, device=env.device, dtype=torch.float32)
        app_cache["tau_total"] = float(tau_desc_np[-1])
        app_cache["key"] = key
        env._slalom_app_tbl = app_cache  # type: ignore[attr-defined]
    app_s, app_x = app_cache["s"], app_cache["x"]
    app_y, app_h = app_cache["y"], app_cache["h"]
    app_k = app_cache["k"]
    app_u_desc = app_cache["u_desc"]               # 距终点弧长 (s0→0)
    app_tau_desc = app_cache["tau_desc"]
    tau_app_total = app_cache["tau_total"]

    # 名义时间推进: τ → 弧长 (直行段快, 弯道慢, 由名义速度剖面积分)
    in_approach = tau < tau_app_total

    # 接近段: τ → u (距终点弧长), 再查位置表
    idx = torch.searchsorted(app_tau_desc, tau).clamp(1, len(app_tau_desc) - 1)
    idx_p = idx - 1
    frac = (tau - app_tau_desc[idx_p]) / (app_tau_desc[idx] - app_tau_desc[idx_p] + 1e-12)
    u_approach = app_u_desc[idx_p] + frac * (app_u_desc[idx] - app_u_desc[idx_p])
    idx2 = torch.searchsorted(app_s, u_approach).clamp(1, len(app_s) - 1)
    idx2_p = idx2 - 1
    frac2 = (u_approach - app_s[idx2_p]) / (app_s[idx2] - app_s[idx2_p] + 1e-12)
    approach_x = app_x[idx2_p] + frac2 * (app_x[idx2] - app_x[idx2_p]) + x_offset
    approach_y = app_y[idx2_p] + frac2 * (app_y[idx2] - app_y[idx2_p])
    approach_h = app_h[idx2_p] + frac2 * (app_h[idx2] - app_h[idx2_p])
    approach_k = -(app_k[idx2_p] + frac2 * (app_k[idx2] - app_k[idx2_p]))   # 正向 κ = -反推 κ

    # LUT 段: 按周期名义时间推进 (每周期 = tau_period)
    tau_lut_sec = torch.clamp(tau - tau_app_total, min=0.0)
    num_periods = (tau_lut_sec / tau_period).floor().long()  # [N]
    tau_rem = tau_lut_sec - num_periods * tau_period

    idx = torch.searchsorted(tau_lut, tau_rem).clamp(1, len(tau_lut) - 1)
    idx_prev = idx - 1
    frac = (tau_rem - tau_lut[idx_prev]) / (tau_lut[idx] - tau_lut[idx_prev] + 1e-12)  # [N]
    x_lut_val = x_lut[idx_prev] + frac * (x_lut[idx] - x_lut[idx_prev])
    y_lut_val = y_lut[idx_prev] + frac * (y_lut[idx] - y_lut[idx_prev])
    h_lut_val = hd_lut[idx_prev]  # 分段常值 heading
    k_lut_val = kappa_lut[idx_prev] + frac * (kappa_lut[idx] - kappa_lut[idx_prev])
    x_lut_ref = x_offset + x_lut_val + num_periods.float() * (2 * pole_spacing)

    # 合成: 接近段用圆弧 (反推表正向), LUT 段用周期路径
    x_ref = torch.where(in_approach, approach_x, x_lut_ref)
    y_ref = torch.where(in_approach, approach_y, y_lut_val)
    path_heading = torch.where(in_approach, approach_h, h_lut_val)
    kappa_ref = torch.where(in_approach, approach_k, k_lut_val)

    # 期望速度 (变速): v = base*gait*scale(|κ|) — 直行段快, 弯道慢
    scale_ref = STRAIGHT_VEL_SCALE - (STRAIGHT_VEL_SCALE - VEL_MIN) * kappa_ref.abs() / CURVATURE_TARGET
    v_cur = base_vel * cmd_tensor[:, 3] * scale_ref
    vx_des = v_cur * torch.cos(path_heading)
    vy_des = v_cur * torch.sin(path_heading)

    # 存储瞬时曲率供脊柱参考使用
    env._path_kappa = kappa_ref  # type: ignore[attr-defined]

    return x_ref, y_ref, vx_des, vy_des, path_heading



# 获取当前路径瞬时曲率 κ(t) — Phase 0 返回静态命令值, Phase 1 返回 LUT 插值
def get_path_curvature(env: "ManagerBasedRlEnv") -> torch.Tensor:
    cmd_term = env.command_manager._terms["slalom_cmd"]
    if not cmd_term.slalom_mode_active:  # type: ignore[union-attr]
        return cmd_term.command[:, 4]  # type: ignore[union-attr]
    return getattr(env, "_path_kappa", cmd_term.command[:, 4])  # type: ignore[union-attr]



# 走廊超额计算
def compute_corridor_excess(
    body_pos_xy: torch.Tensor,     # [N, 2] 身体中心在投影平面上的位置
    body_heading: torch.Tensor,    # [N]    身体物理前向朝向
    ref_pos_xy: torch.Tensor,      # [N, 2] 参考点位置
    path_heading: torch.Tensor,    # [N]    路径切线方向
    half_length: float,            # 身体半长
    half_width: float,             # 身体半宽
) -> torch.Tensor:
    # 路径法向量（指向路径左侧）
    n_x = -torch.sin(path_heading)
    n_y = torch.cos(path_heading)

    # 侧向距离（带符号）
    dx = body_pos_xy[:, 0] - ref_pos_xy[:, 0]
    dy = body_pos_xy[:, 1] - ref_pos_xy[:, 1]
    d_lat = dx * n_x + dy * n_y  # [N]

    # 朝向误差导致的半宽膨胀
    delta_theta = body_heading - path_heading
    eff_hw = torch.abs(half_length * torch.sin(delta_theta)) + torch.abs(half_width * torch.cos(delta_theta))

    return torch.abs(d_lat) + eff_hw  # [N]
