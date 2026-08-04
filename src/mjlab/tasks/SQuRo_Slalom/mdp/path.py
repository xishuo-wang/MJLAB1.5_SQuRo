from __future__ import annotations
import torch
import numpy as np
from mjlab.entity import Entity
from typing import TYPE_CHECKING, Optional
from .indices import _MODEL_INDICES
from .curriculums import CURVATURE_TARGET, SMOOTH_TIME, SMOOTH_VEL
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# SQuRo 身体尺寸参数
F_BODY_HALF_LENGTH = 0.04       # F_body_Link 在XoY平面上沿身体前后轴半长(m)
F_BODY_HALF_WIDTH  = 0.035      # F_body_Link 在XoY平面左右方向半宽(m)
H_BODY_HALF_LENGTH = 0.04       # H_body_Link 在XoY平面沿身体前后轴半长(m)
H_BODY_HALF_WIDTH  = 0.035      # H_body_Link 在XoY平面左右方向半宽(m)
BODY_REF_OFFSET = 0.04          # F_body/H_body 中心距 base 中心的X轴偏移量
CORRIDOR_HALF_WIDTH = 0.04      # 走廊半宽（基元阶段 = 身体半宽 + 控制余量）
_INIT_DIST = 0.02               # 初始接近段弧长 (m): 机器人从圆弧起点走 _INIT_DIST 到路径起点 (0,0)


# =========================================================================================
# 接近段 (圆弧) — 从 LUT 起点 (0,0) 反推 s0=_INIT_DIST:
#   正向接近段 = 平台(-K) + 过渡(-K→0), 终点 (0,0) κ=0 heading=0, 与 LUT 进过渡衔接
_APPROACH_CACHE: dict = {}


def _approach_rev_table(s0: float, tr: float) -> dict:
    """反推接近段轨迹表: 从 (0,0) heading=0 反推 s0。
    返回 {s, x, y, h, k}: 反推弧长 u∈[0,s0] 对应的路径 (反向)。
    机器人正向接近段 = 反推表的正向 (从 u=s0 走向 u=0, 到 (0,0) heading=0)。
    """
    key = (round(s0, 6), round(tr, 6))
    if key in _APPROACH_CACHE:
        return _APPROACH_CACHE[key]
    K = CURVATURE_TARGET
    ds = 1e-4
    n = max(2, int(s0 / ds))
    u = np.linspace(0.0, s0, n, endpoint=False)
    trp = min(s0, tr)
    ku = np.where(u < trp, K * u / trp, K)     # 反向 κ: 过渡 0→+K, 平台 +K
    h = np.cumsum(ku) * ds
    x = -np.cumsum(np.cos(h)) * ds
    y = -np.cumsum(np.sin(h)) * ds
    tbl = {"s": u, "x": x, "y": y, "h": h, "k": ku}
    _APPROACH_CACHE[key] = tbl
    return tbl


# 机器人初始位置/朝向 (名义 vel), 供 events.py 重置
def get_approach_start() -> tuple[float, float, float]:
    tr = SMOOTH_VEL * SMOOTH_TIME
    tbl = _approach_rev_table(_INIT_DIST, tr)
    return float(tbl["x"][-1]), float(tbl["y"][-1]), float(tbl["h"][-1])


# Phase 0 圆弧接近段: 从 (0,0) 反推 s0, κ=curvature 恒定 (解析解, 向量化)
#   x(u) = -sin(k·u)/k, y(u) = (1-cos(k·u))/k, h(u) = -k·u  (u=距(0,0)弧长)
def get_arc_approach_start_xyh(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s0 = _INIT_DIST
    mask = torch.abs(k) > 1e-6
    k_s = torch.where(mask, k, torch.ones_like(k))
    x = torch.where(mask, -torch.sin(k * s0) / k_s, torch.full_like(k, -s0))
    y = torch.where(mask, (1.0 - torch.cos(k * s0)) / k_s, torch.zeros_like(k))
    h = torch.where(mask, -k * s0, torch.zeros_like(k))
    return x, y, h


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


# =========================================================================================
# 绕杆路径查找表（LUT）— 使用圆弧拼接方式，与 slalom_path_viz.py 逻辑一致
_RMIN = 1.0 / CURVATURE_TARGET      # 最小转弯半径 (= 1/κ_arc)


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

    r = _RMIN
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


# =========================================================================================
# 平滑 LUT 生成 — κ 曲线 (弧段 = 进过渡 + 平台 + 出过渡) 数值积分重建路径
_SMOOTH_XSW_CACHE: dict = {}


def _measure_s1_xsw(segs, ds=1e-4) -> float:
    """积分 S1 弧段, 返回结束处 x 位移 (κ 首次到 0)"""
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


def _get_smooth_xsw(tr: float) -> tuple[float, float]:
    """返回 (x_sw_half, x_sw_full) (按过渡弧长缓存):
    x_sw_half = 无直行模式 S1 (平台half+出) 的 x 位移
    x_sw_full = 有直行模式 S1 (进+平台+出) 的 x 位移"""
    key = round(tr, 6)
    if key in _SMOOTH_XSW_CACHE:
        return _SMOOTH_XSW_CACHE[key]
    K = CURVATURE_TARGET
    L90 = np.pi / 2 * _RMIN
    platform = L90 - tr
    platform_half = L90 - tr / 2
    x_sw_half = _measure_s1_xsw([(platform_half, -K, -K), (tr, -K, 0.0)])
    x_sw_full = _measure_s1_xsw([(tr, 0.0, -K), (platform, -K, -K), (tr, -K, 0.0)])
    _SMOOTH_XSW_CACHE[key] = (x_sw_half, x_sw_full)
    return x_sw_half, x_sw_full


# 平滑模式下的有效杆间距: 不兼容区间 (2×x_sw_half, 2×x_sw_full) → 无直行最小间距
# 平滑弧长固定 (与 vel 解耦): tr = SMOOTH_VEL × SMOOTH_TIME, 保证任意步频下几何一致
def get_effective_pole_spacing(spacing: float) -> float:
    tr = SMOOTH_VEL * SMOOTH_TIME
    x_sw_half, x_sw_full = _get_smooth_xsw(tr)
    min_sp = 2 * x_sw_half
    if spacing <= min_sp + 1e-6:
        if spacing < min_sp - 1e-6:
            print(f"[WARN] spacing={spacing:.4f} 低于平滑最小间距 {min_sp:.4f}, "
                  f"采用无直行模式, 有效间距={min_sp:.4f}")
        return min_sp                       # 无直行 (同向连续)
    if spacing >= 2 * x_sw_full - 1e-6:
        return spacing                      # 有直行 (含边界)
    print(f"[WARN] spacing={spacing:.4f} 处于平滑不兼容区间 ({min_sp:.4f}, {2*x_sw_full:.4f}), "
          f"采用无直行模式, 有效间距={min_sp:.4f}")
    return min_sp


def _generate_slalom_lut_smooth_period(X: float, smooth_time: Optional[float] = None, vel: Optional[float] = None):
    t = SMOOTH_TIME if smooth_time is None else smooth_time
    v = SMOOTH_VEL if vel is None else vel
    K = CURVATURE_TARGET
    tr = v * t
    L90 = np.pi / 2 * _RMIN
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
            (tr, 0.0, -K), (platform, -K, -K), (tr, -K, 0.0),   # S1
            (tr, 0.0, K), (platform, K, K), (tr, K, 0.0),       # S2
            (straight, 0.0, 0.0),                                # S3 直行
            (tr, 0.0, K), (platform, K, K), (tr, K, 0.0),       # S4
            (tr, 0.0, -K), (platform, -K, -K), (tr, -K, 0.0),   # S5
            (straight, 0.0, 0.0),                                # S6 直行
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
    vel_cmd = cmd_term.command[:, 0]                # [N]
    pole_spacing = cmd_term.active_pole_spacing      # type: ignore[union-attr]
    dt = env.step_dt
    t = env.episode_length_buf.float() * dt         # [N]

    # 首次调用或杆间距变化时重建 LUT (平滑弧长固定, 与 vel 解耦)
    cache = getattr(env, "_slalom_lut_cache", None)
    if cache is None or cache["spacing"] != pole_spacing:
        arc_np, xs_np, ys_np, hd_np, kp_np = _generate_slalom_lut_smooth_period(pole_spacing)
        dev = env.device
        cache = {
            "spacing": pole_spacing,
            "arc": torch.tensor(arc_np, device=dev, dtype=torch.float32),
            "x": torch.tensor(xs_np, device=dev, dtype=torch.float32),
            "y": torch.tensor(ys_np, device=dev, dtype=torch.float32),
            "heading": torch.tensor(hd_np, device=dev, dtype=torch.float32),
            "kappa": torch.tensor(kp_np, device=dev, dtype=torch.float32),
            "period": float(arc_np[-1]),
        }
        env._slalom_lut_cache = cache  # type: ignore[attr-defined]

    arc_lut = cache["arc"]
    x_lut = cache["x"]
    y_lut = cache["y"]
    hd_lut = cache["heading"]
    s_period = cache["period"]

    # 当前弧长 s = v·t mod period + 周期偏移(保证 x_ref 连续不跳变)
    # 起始偏移 -_INIT_DIST: 前 _INIT_DIST 米为直行接近段 (机器人 X=-_INIT_DIST → LUT 起点 X=0)
    total_s = vel_cmd * t - _INIT_DIST            # [N], 负值=接近段
    in_approach = total_s < 0.0

    # 接近段: 圆弧 (平台 -K + 过渡 -K→0), 从起点正向走 s0=_INIT_DIST 到 (0,0)
    # 效率: torch 表缓存到 env (避免每步 tensor 创建)
    app_cache = getattr(env, "_slalom_app_tbl", None)
    if app_cache is None:
        app_tbl = _approach_rev_table(_INIT_DIST, SMOOTH_VEL * SMOOTH_TIME)
        app_cache = {key: torch.tensor(val, device=env.device, dtype=torch.float32)
                     for key, val in app_tbl.items()}
        env._slalom_app_tbl = app_cache  # type: ignore[attr-defined]
    app_s, app_x = app_cache["s"], app_cache["x"]
    app_y, app_h = app_cache["y"], app_cache["h"]
    app_k = app_cache["k"]
    u = (-total_s).clamp(min=0.0)                     # 机器人正向距 (0,0) 的弧长
    idx = torch.searchsorted(app_s, u).clamp(1, len(app_s) - 1)
    idx_p = idx - 1
    frac = (u - app_s[idx_p]) / (app_s[idx] - app_s[idx_p] + 1e-12)
    approach_x = app_x[idx_p] + frac * (app_x[idx] - app_x[idx_p])
    approach_y = app_y[idx_p] + frac * (app_y[idx] - app_y[idx_p])
    approach_h = app_h[idx_p] + frac * (app_h[idx] - app_h[idx_p])
    approach_k = -(app_k[idx_p] + frac * (app_k[idx] - app_k[idx_p]))   # 正向 κ = -反推 κ

    s_pos = torch.clamp(total_s, min=0.0)         # [N], ≥0 进入 LUT
    num_periods = (s_pos / s_period).floor().long()  # [N]
    s = s_pos - num_periods * s_period            # [N], = s_pos % period

    # searchsorted 查找索引 + 线性插值
    idx = torch.searchsorted(arc_lut, s).clamp(1, len(arc_lut) - 1)  # [N]
    idx_prev = idx - 1
    s_prev = arc_lut[idx_prev]
    s_next = arc_lut[idx]

    frac = (s - s_prev) / (s_next - s_prev + 1e-12)  # [N]
    x_lut_val = x_lut[idx_prev] + frac * (x_lut[idx] - x_lut[idx_prev])
    y_lut_val = y_lut[idx_prev] + frac * (y_lut[idx] - y_lut[idx_prev])
    h_lut_val = hd_lut[idx_prev]  # 分段常值 heading
    kappa_lut = cache["kappa"]
    k_lut_val = kappa_lut[idx_prev] + frac * (kappa_lut[idx] - kappa_lut[idx_prev])
    # 叠加已完成周期偏移: 每周期 X 前进 2*pole_spacing
    x_lut_ref = x_lut_val + num_periods.float() * (2 * pole_spacing)

    # 合成: 接近段用圆弧 (反推表正向), LUT 段用周期路径
    x_ref = torch.where(in_approach, approach_x, x_lut_ref)
    y_ref = torch.where(in_approach, approach_y, y_lut_val)
    path_heading = torch.where(in_approach, approach_h, h_lut_val)
    kappa_ref = torch.where(in_approach, approach_k, k_lut_val)

    vx_des = vel_cmd * torch.cos(path_heading)
    vy_des = vel_cmd * torch.sin(path_heading)

    # 存储瞬时曲率供脊柱参考使用
    env._path_kappa = kappa_ref  # type: ignore[attr-defined]

    return x_ref, y_ref, vx_des, vy_des, path_heading


# 获取当前路径瞬时曲率 κ(t) — Phase 0 返回静态命令值, Phase 1 返回 LUT 插值
def get_path_curvature(env: "ManagerBasedRlEnv") -> torch.Tensor:
    cmd_term = env.command_manager._terms["slalom_cmd"]
    if not cmd_term.slalom_mode_active:  # type: ignore[union-attr]
        return cmd_term.command[:, 4]  # type: ignore[union-attr]
    return getattr(env, "_path_kappa", cmd_term.command[:, 4])  # type: ignore[union-attr]


# 走廊超额计算 — 纯数学函数，不依赖 env
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
