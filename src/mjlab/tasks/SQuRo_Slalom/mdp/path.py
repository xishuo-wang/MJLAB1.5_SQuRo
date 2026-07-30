from __future__ import annotations
import torch
import numpy as np
from mjlab.entity import Entity
from typing import TYPE_CHECKING
from .indices import _MODEL_INDICES
from .curriculums import CURVATURE_TARGET
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# SQuRo 身体尺寸参数
F_BODY_HALF_LENGTH = 0.04           # F_body_Link 在XoY平面上沿身体前后轴半长(m)
F_BODY_HALF_WIDTH  = 0.035          # F_body_Link 在XoY平面左右方向半宽(m)
H_BODY_HALF_LENGTH = 0.04           # H_body_Link 在XoY平面沿身体前后轴半长(m)
H_BODY_HALF_WIDTH  = 0.035          # H_body_Link 在XoY平面左右方向半宽(m)
BODY_REF_OFFSET = 0.04              # F_body/H_body 中心距 base 中心的X轴偏移量
CORRIDOR_HALF_WIDTH = 0.04          # 走廊半宽（基元阶段 = 身体半宽 + 控制余量）

_RMIN = 1.0 / CURVATURE_TARGET  # 最小转弯半径 (= 0.05m for κ=20)


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

    # 固定世界坐标系起点 — 与 reset_model 中的初始位置一致
    start_heading = 0.0  # 物理前向 = world +X
    omega = curvature * vel_cmd
    dtheta = omega * t
    path_heading = start_heading + dtheta

    # 弦长公式：chord = v·t·sinc(dθ/2π)
    chord = vel_cmd * t * torch.sinc(dtheta / (2 * torch.pi))
    x_ref = chord * torch.cos(start_heading + dtheta / 2)   # start_x = 0
    y_ref = chord * torch.sin(start_heading + dtheta / 2)   # start_y = 0

    # 世界系期望速度（路径切线方向）
    vx_des = vel_cmd * torch.cos(path_heading)
    vy_des = vel_cmd * torch.sin(path_heading)

    return x_ref, y_ref, vx_des, vy_des, path_heading


# =========================================================================================
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


# 路径参考计算 — 绕杆阶段：圆弧拼接路径（LUT 查表）
def compute_slalom_path_ref(env: "ManagerBasedRlEnv"):
    cmd_term = env.command_manager._terms["slalom_cmd"]
    vel_cmd = cmd_term.command[:, 0]                # [N]
    pole_spacing = cmd_term.active_pole_spacing      # type: ignore[union-attr]
    dt = env.step_dt

    # 首次调用或杆间距变化时重建 LUT
    cache = getattr(env, "_slalom_lut_cache", None)
    if cache is None or abs(cache["spacing"] - pole_spacing) > 1e-6:
        arc_np, xs_np, ys_np, hd_np, kp_np = _generate_slalom_lut_one_period(pole_spacing)
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

    # 累积弧长 (∫ vel dt) — 适配速度动态变化
    if getattr(env, "_slalom_arc_len", None) is None:
        env._slalom_arc_len = torch.zeros(env.num_envs, device=env.device)  # type: ignore[attr-defined]
    env._slalom_arc_len += vel_cmd * dt  # type: ignore[attr-defined]
    # 重置已终止环境
    reset_ids = getattr(env, "reset_terminated", None)
    if reset_ids is not None:
        ids = reset_ids.nonzero(as_tuple=False).flatten()
        if len(ids) > 0:
            env._slalom_arc_len[ids] = 0.0  # type: ignore[attr-defined]

    total_s = env._slalom_arc_len  # type: ignore[attr-defined]  # [N], = ∫vel dt
    num_periods = (total_s / s_period).floor().long()  # [N]
    s = total_s - num_periods * s_period          # [N], = total_s % period

    # searchsorted 查找索引 + 线性插值
    idx = torch.searchsorted(arc_lut, s).clamp(1, len(arc_lut) - 1)  # [N]
    idx_prev = idx - 1
    s_prev = arc_lut[idx_prev]
    s_next = arc_lut[idx]

    frac = (s - s_prev) / (s_next - s_prev + 1e-12)  # [N]
    x_lut_val = x_lut[idx_prev] + frac * (x_lut[idx] - x_lut[idx_prev])
    y_ref = y_lut[idx_prev] + frac * (y_lut[idx] - y_lut[idx_prev])
    # 叠加已完成周期偏移: 每周期 X 前进 2*pole_spacing
    x_ref = x_lut_val + num_periods.float() * (2 * pole_spacing)
    path_heading = hd_lut[idx_prev]  # 分段常值 heading

    vx_des = vel_cmd * torch.cos(path_heading)
    vy_des = vel_cmd * torch.sin(path_heading)

    # 存储瞬时曲率供脊柱参考使用
    kappa_lut = cache["kappa"]
    kappa_ref = kappa_lut[idx_prev] + frac * (kappa_lut[idx] - kappa_lut[idx_prev])
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


# =========================================================================================
# 点到旋转矩形的最短距离 [N]
def _point_to_rect_dist(
    px: torch.Tensor, py: torch.Tensor,        # [N] 杆心 XY
    cx: torch.Tensor, cy: torch.Tensor,         # [N] 身体中心 XY
    heading: torch.Tensor,                      # [N] 身体朝向
    half_L: float, half_W: float,
) -> torch.Tensor:
    dx = px - cx; dy = py - cy
    local_x =  dx * torch.cos(-heading) + dy * torch.sin(-heading)
    local_y = -dx * torch.sin(-heading) + dy * torch.cos(-heading)
    cx_c = torch.clamp(local_x, -half_L, half_L)
    cy_c = torch.clamp(local_y, -half_W, half_W)
    return torch.sqrt((local_x - cx_c)**2 + (local_y - cy_c)**2)


# 点到线段的最短距离 [N]
def _point_to_segment_dist(
    px: torch.Tensor, py: torch.Tensor,        # [N] 杆心 XY
    x1: torch.Tensor, y1: torch.Tensor,         # [N] 线段起点 (附着点)
    x2: torch.Tensor, y2: torch.Tensor,         # [N] 线段终点 (足端)
) -> torch.Tensor:
    dx = x2 - x1; dy = y2 - y1
    t = ((px - x1)*dx + (py - y1)*dy) / (dx*dx + dy*dy + 1e-12)
    t = torch.clamp(t, 0.0, 1.0)
    near_x = x1 + t * dx; near_y = y1 + t * dy
    return torch.sqrt((px - near_x)**2 + (py - near_y)**2)


# 每个身体环节到最近杆的距离
def compute_body_pole_dist(
    body_center: torch.Tensor,        # [N, 2] 身体中心
    body_heading: torch.Tensor,       # [N] 身体朝向
    half_L: float, half_W: float,
    pole_pos: torch.Tensor,           # [M, 2] 所有杆 XY
) -> torch.Tensor:
    pole_r = 0.005  # 杆半径
    min_dist = torch.full((body_center.shape[0],), 1e9, device=body_center.device)
    for j in range(pole_pos.shape[0]):
        d = _point_to_rect_dist(
            pole_pos[j, 0].expand(body_center.shape[0]),
            pole_pos[j, 1].expand(body_center.shape[0]),
            body_center[:, 0], body_center[:, 1],
            body_heading, half_L, half_W,
        )
        min_dist = torch.minimum(min_dist, d - pole_r)
    return min_dist  # [N], negative = collision


# 单条腿到最近杆的距离 [N]
def compute_leg_pole_dist(
    body_center: torch.Tensor,        # [N, 2]
    body_heading: torch.Tensor,       # [N]
    half_W: float,                    # 半宽 (附着点在 (0, ±W))
    sign: float,                      # +1=左侧(FR/HR), -1=右侧(FL/HL)
    foot_pos: torch.Tensor,           # [N, 2] 足端 site XY
    pole_pos: torch.Tensor,           # [M, 2]
) -> torch.Tensor:
    pole_r = 0.005
    # 附着点: body_center + (0, sign*half_W) 旋转 body_heading
    attach_x = body_center[:, 0] - sign * half_W * torch.sin(body_heading)
    attach_y = body_center[:, 1] + sign * half_W * torch.cos(body_heading)
    min_dist = torch.full((body_center.shape[0],), 1e9, device=body_center.device)
    for j in range(pole_pos.shape[0]):
        d = _point_to_segment_dist(
            pole_pos[j, 0].expand(body_center.shape[0]),
            pole_pos[j, 1].expand(body_center.shape[0]),
            attach_x, attach_y,
            foot_pos[:, 0], foot_pos[:, 1],
        )
        min_dist = torch.minimum(min_dist, d - pole_r)
    return min_dist
