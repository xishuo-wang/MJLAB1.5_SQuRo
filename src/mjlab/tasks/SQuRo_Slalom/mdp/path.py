from __future__ import annotations
import torch
import numpy as np
from mjlab.entity import Entity
from typing import TYPE_CHECKING
from .indices import _MODEL_INDICES
from .curriculums import (
    BASE_VEL,
    CURVATURE_TARGET, Rmin,
    PHASE0_EPISODE_SECONDS,
    get_velocity_scale,
    phase0_kappa_at,
)
from .slalom_geometry import APPROACH_LENGTH, curvature_from_spacing, get_slalom_geometry, get_timed_slalom_tables

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv



# SQuRo 身体尺寸参数
F_BODY_HALF_LENGTH = 0.04       # F_body_Link 在XoY平面上沿身体前后轴半长(m)
F_BODY_HALF_WIDTH  = 0.035      # F_body_Link 在XoY平面左右方向半宽(m)
H_BODY_HALF_LENGTH = 0.04       # H_body_Link 在XoY平面沿身体前后轴半长(m)
H_BODY_HALF_WIDTH  = 0.035      # H_body_Link 在XoY平面左右方向半宽(m)
BODY_REF_OFFSET = 0.04          # F_body/H_body 中心距 base 中心的X轴偏移量
CORRIDOR_HALF_WIDTH = 0.04      # 走廊半宽（基元阶段 = 身体半宽 + 控制余量）
_INIT_DIST = APPROACH_LENGTH               # 初始接近段弧长 (m): 机器人从圆弧起点走 _INIT_DIST 到路径起点 (0,0)



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



# 按整数曲率返回接近段起点，指定杆距时选择最近的可实现条件。
def get_approach_start(spacing: float | None = None, curvature: int | None = None) -> tuple[float, float, float]:
    k = curvature if curvature is not None else (int(CURVATURE_TARGET) if spacing is None else curvature_from_spacing(spacing))
    approach = get_slalom_geometry(k)["approach"]
    return tuple(float(approach[name][0]) for name in ("x", "y", "heading"))


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
def get_phase1_approach(spacing: float | None = None) -> tuple[float, float, float]:
    return get_approach_start(spacing)



# 路径参考调度 — 根据训练阶段自动选择圆弧或绕杆路径
def compute_path_ref(env: "ManagerBasedRlEnv"):
    cmd_term = env.command_manager._terms["slalom_cmd"]
    # 逐环境阶段: 两条路径都算再按阶段选取, 阶段切换后的一个回合内会同时存在两种环境
    phase1 = cmd_term.phase1_mask  # type: ignore[union-attr]
    sx, sy, svx, svy, sh = compute_slalom_path_ref(env)
    kappa_slalom = env._path_kappa  # type: ignore[attr-defined]
    ax, ay, avx, avy, ah = compute_arc_path_ref(env)
    env._path_kappa = torch.where(phase1, kappa_slalom,  # type: ignore[attr-defined]
                                  getattr(env, "_phase0_path_kappa", cmd_term.command[:, 4]))  # type: ignore[union-attr]
    return (
        torch.where(phase1, sx, ax),
        torch.where(phase1, sy, ay),
        torch.where(phase1, svx, avx),
        torch.where(phase1, svy, avy),
        torch.where(phase1, sh, ah),
    )



# 预计算一阶段回合路径: 按曲率日程积分成逐控制步位置/朝向数组, 长度按一阶段有效回合
def build_phase0_episode_path(env: "ManagerBasedRlEnv", env_ids, start_x, start_y, start_heading,
                              schedule, velocity) -> None:
    n = len(env_ids)
    dt = float(env.step_dt)
    steps = min(int(env.max_episode_length), int(round(PHASE0_EPISODE_SECONDS / dt)))
    segments = schedule["segments"][env_ids]
    bounds = schedule["bounds"][env_ids]
    path_xy = getattr(env, "_phase0_path_xy", None)
    if path_xy is None or path_xy.shape[1] != steps + 1:
        env._phase0_path_xy = torch.zeros(env.num_envs, steps + 1, 2, device=env.device)
        env._phase0_path_heading = torch.zeros(env.num_envs, steps + 1, device=env.device)
    env._phase0_schedule = schedule
    # 逐控制步中点曲率, 用圆弧弦增量累积: 常数曲率下与解析圆弧一致
    mid_t = (torch.arange(steps, device=env.device).float() + 0.5) * dt
    kappa = phase0_kappa_at(mid_t[None, :].expand(n, -1),
                            {"segments": segments, "bounds": bounds})
    delta_theta = kappa * velocity[:, None] * dt
    heading = torch.cat((start_heading[:, None],
                         start_heading[:, None] + torch.cumsum(delta_theta, dim=1)), dim=1)
    previous, following = heading[:, :-1], heading[:, 1:]
    turning = kappa.abs() > 1e-6
    safe = torch.where(turning, kappa, torch.ones_like(kappa))
    chord_x = torch.where(turning, (torch.sin(following) - torch.sin(previous)) / safe,
                          velocity[:, None] * dt * torch.cos(previous))
    chord_y = torch.where(turning, -(torch.cos(following) - torch.cos(previous)) / safe,
                          velocity[:, None] * dt * torch.sin(previous))
    env._phase0_path_xy[env_ids, :, 0] = torch.cat(
        (start_x[:, None], start_x[:, None] + torch.cumsum(chord_x, dim=1)), dim=1)
    env._phase0_path_xy[env_ids, :, 1] = torch.cat(
        (start_y[:, None], start_y[:, None] + torch.cumsum(chord_y, dim=1)), dim=1)
    env._phase0_path_heading[env_ids, :] = heading


# 路径参考计算 — 基元阶段: 查本回合预计算路径, 动态切换时路径保持连续
def compute_arc_path_ref(env: "ManagerBasedRlEnv"):
    cmd_term = env.command_manager._terms["slalom_cmd"]
    device = env.device
    path_xy = getattr(env, "_phase0_path_xy", None)
    if path_xy is None:
        env._phase0_path_xy = torch.zeros(env.num_envs, 1, 2, device=device)
        env._phase0_path_heading = torch.zeros(env.num_envs, 1, device=device)
        path_xy = env._phase0_path_xy
    steps = path_xy.shape[1] - 1
    idx = env.episode_length_buf.long().clamp(0, steps)
    rows = torch.arange(env.num_envs, device=device)
    xy = path_xy[rows, idx]
    path_heading = env._phase0_path_heading[rows, idx]
    schedule = getattr(env, "_phase0_schedule", None)
    if schedule is None:
        schedule = getattr(cmd_term, "phase0_schedule", None)
    if schedule is None:
        schedule = {"segments": cmd_term.command[:, 4:5],
                    "bounds": torch.full_like(cmd_term.command[:, :1], 1e9),
                    "peak": cmd_term.command[:, 4].abs()}
    env._phase0_path_kappa = phase0_kappa_at(idx.float() * env.step_dt, schedule)  # type: ignore[attr-defined]
    vel_cmd = cmd_term.command[:, 0]
    vx_des = vel_cmd * torch.cos(path_heading)
    vy_des = vel_cmd * torch.sin(path_heading)
    origins = getattr(getattr(env, "scene", None), "env_origins", None)
    x_ref, y_ref = xy[:, 0], xy[:, 1]
    if origins is not None:
        x_ref, y_ref = x_ref + origins[:, 0], y_ref + origins[:, 1]
    return x_ref, y_ref, vx_des, vy_des, path_heading



# 历史纯圆弧绘图对照；实际训练使用下方无直行平滑路径。
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


# 将输入杆距映射到整数峰值曲率对应的真实杆距。
def get_effective_pole_spacing(spacing: float) -> float:
    return get_slalom_geometry(curvature_from_spacing(spacing))["spacing"]


# 保留诊断脚本入口，当前绕杆路径全部采用无直行结构。
def is_no_straight_spacing(spacing: float) -> bool:
    del spacing
    return True


# 返回无直行周期的几何表，曲率与杆距由同一份几何确定。
def _generate_slalom_lut_smooth_period(X: float, smooth_time=None, vel=None, curvature: int | None = None):
    if smooth_time is not None or vel is not None:
        raise ValueError("无直行路径使用集中配置的固定平滑弧长")
    k = curvature_from_spacing(X) if curvature is None else curvature
    period = get_slalom_geometry(k)["period"]
    return tuple(period[name] for name in ("arc", "x", "y", "heading", "kappa"))


# 把全部整数曲率的名义时间表缓存到当前设备。
def get_slalom_tables(env, base_velocity: float) -> dict[str, torch.Tensor]:
    cached = getattr(env, "_slalom_tables", None)
    if cached is None or cached[0] != base_velocity:
        cached = (base_velocity, {name: torch.as_tensor(value, device=env.device, dtype=torch.float32)
                                for name, value in get_timed_slalom_tables(base_velocity).items()})
        env._slalom_tables = cached
    return cached[1]


# 按逐环境曲率行和归一化时间插值，避免每步复制完整路径表。
def _sample_slalom_table(table: torch.Tensor, rows: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
    coordinate = progress.clamp(0, 1) * (table.shape[1] - 1)
    left = coordinate.long().clamp(max=table.shape[1] - 2)
    fraction = (coordinate - left).unsqueeze(-1)
    return torch.lerp(table[rows, left], table[rows, left + 1], fraction)


# 绕杆参考按本回合的整数曲率和自身步频推进，杆距不会随其他环境重置改变。
def compute_slalom_path_ref(env: "ManagerBasedRlEnv"):
    cmd = env.command_manager._terms["slalom_cmd"]
    base_velocity = float(getattr(env, "_slalom_base_vel_scalar", BASE_VEL))
    tables = get_slalom_tables(env, base_velocity)
    rows = cmd.slalom_curvature_index
    tau = env.episode_length_buf.float() * env.step_dt * cmd.command[:, 3]
    approach_time = tables["approach_time"][rows]
    period_time = tables["period_time"][rows]
    in_approach = tau < approach_time
    approach = _sample_slalom_table(tables["approach"], rows, tau / approach_time)
    elapsed = (tau - approach_time).clamp_min(0)
    cycles = torch.floor(elapsed / period_time)
    period = _sample_slalom_table(tables["period"], rows, elapsed / period_time - cycles)
    x_period = period[:, 0] + cmd.pole_spacing * (1 + 2 * cycles)
    x = torch.where(in_approach, approach[:, 0], x_period)
    y = torch.where(in_approach, approach[:, 1], period[:, 1])
    heading = torch.where(in_approach, approach[:, 2], period[:, 2])
    kappa = torch.where(in_approach, approach[:, 3], period[:, 3])
    origins = getattr(getattr(env, "scene", None), "env_origins", None)
    if origins is not None:
        x, y = x + origins[:, 0], y + origins[:, 1]
    velocity = base_velocity * cmd.command[:, 3] * get_velocity_scale(cmd.slalom_peak_curvature)
    env._path_kappa = kappa
    return x, y, velocity * heading.cos(), velocity * heading.sin(), heading


# 获取当前路径瞬时曲率 κ(t) — 由 compute_path_ref 逐环境写入 (Phase 0 为命令值, Phase 1 为 LUT 插值)
def get_path_curvature(env: "ManagerBasedRlEnv") -> torch.Tensor:
    cmd_term = env.command_manager._terms["slalom_cmd"]
    return getattr(env, "_path_kappa", cmd_term.command[:, 4])  # type: ignore[union-attr]



# 走廊超额计算
def _compute_corridor_projection(
    body_pos_xy: torch.Tensor,     # [N, 2] 身体中心在投影平面上的位置
    body_heading: torch.Tensor,    # [N]    身体物理前向朝向
    ref_pos_xy: torch.Tensor,      # [N, 2] 参考点位置
    path_heading: torch.Tensor,    # [N]    路径切线方向
    half_length: float,            # 身体半长
    half_width: float,             # 身体半宽
) -> tuple[torch.Tensor, torch.Tensor]:
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

    return d_lat, eff_hw


# 返回沿路径法向的最大身体包络距离，保留原有诊断口径。
def compute_corridor_excess(
    body_pos_xy: torch.Tensor,
    body_heading: torch.Tensor,
    ref_pos_xy: torch.Tensor,
    path_heading: torch.Tensor,
    half_length: float,
    half_width: float,
) -> torch.Tensor:
    d_lat, eff_hw = _compute_corridor_projection(
        body_pos_xy, body_heading, ref_pos_xy, path_heading, half_length, half_width
    )
    return d_lat.abs() + eff_hw


# 正曲率以内侧左法向计量，负曲率相反；直行使用对称走廊。
def compute_corridor_side_excess(
    body_pos_xy: torch.Tensor,
    body_heading: torch.Tensor,
    ref_pos_xy: torch.Tensor,
    path_heading: torch.Tensor,
    half_length: float,
    half_width: float,
    curvature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_lat, eff_hw = _compute_corridor_projection(
        body_pos_xy, body_heading, ref_pos_xy, path_heading, half_length, half_width
    )
    signed_lat = curvature.sign() * d_lat
    turning = curvature.abs() > 1e-6
    inner = torch.clamp(signed_lat + eff_hw - CORRIDOR_HALF_WIDTH, min=0.0)
    outer = torch.clamp(-signed_lat + eff_hw - CORRIDOR_HALF_WIDTH, min=0.0)
    straight = torch.clamp(d_lat.abs() + eff_hw - CORRIDOR_HALF_WIDTH, min=0.0)
    return torch.where(turning, inner, 0.0), torch.where(turning, outer, straight)
