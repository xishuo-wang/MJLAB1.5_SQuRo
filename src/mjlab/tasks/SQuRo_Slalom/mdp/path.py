from __future__ import annotations
import torch
from mjlab.entity import Entity
from typing import TYPE_CHECKING
from .indices import _MODEL_INDICES
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# SQuRo 身体尺寸参数
F_BODY_HALF_LENGTH = 0.04       # F_body_Link 在XoY平面上沿身体前后轴半长(m)
F_BODY_HALF_WIDTH  = 0.035      # F_body_Link 在XoY平面左右方向半宽(m)
H_BODY_HALF_LENGTH = 0.04       # H_body_Link 在XoY平面沿身体前后轴半长(m)
H_BODY_HALF_WIDTH  = 0.035      # H_body_Link 在XoY平面左右方向半宽(m)
BODY_REF_OFFSET = 0.04          # F_body/H_body 中心距 base 中心的X轴偏移量
CORRIDOR_HALF_WIDTH = 0.04      # 走廊半宽（基元阶段 = 身体半宽 + 控制余量）


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
# 路径参考调度 — 根据命令模式自动选择圆弧或绕杆路径
def compute_path_ref(env: "ManagerBasedRlEnv"):
    """路径参考统一入口

    slalom_mode=False → 基元圆弧路径 (compute_arc_path_ref)
    slalom_mode=True  → 正弦绕杆路径 (compute_slalom_path_ref)
    """
    cmd_term = env.command_manager._terms["slalom_cmd"]
    if getattr(cmd_term.cfg, "slalom_mode", False):
        return compute_slalom_path_ref(env)
    return compute_arc_path_ref(env)


# =========================================================================================
# 路径参考计算 — 绕杆阶段：正弦曲率剖面
def compute_slalom_path_ref(env: "ManagerBasedRlEnv"):
    """正弦曲率绕杆路径 — 曲率连续光滑变化，无阶跃

    κ(t) = κ_peak · cos(ωt)
    θ(t) = κ_peak·v/ω · sin(ωt)           (零均值对称振荡)
    ω = 2π/T, T = 2·pole_spacing/v

    返回 (x_ref, y_ref, vx_des, vy_des, path_heading)
    """
    cmd_term = env.command_manager._terms["slalom_cmd"]
    vel_cmd = cmd_term.command[:, 0]            # [N]
    kappa_peak = cmd_term.command[:, 4]          # [N] — 曲率幅值
    pole_spacing = cmd_term.cfg.pole_spacing     # type: ignore[attr-defined]
    dt = env.step_dt
    t = env.episode_length_buf.float() * dt     # [N]

    # ω = 2π / T, T = 2·pole_spacing / v
    period = 2.0 * pole_spacing / (vel_cmd + 1e-8)   # [N]
    omega = 2 * torch.pi / period                     # [N]

    # κ(t), θ(t) 解析计算
    omega_t = omega * t
    kappa_t = kappa_peak * torch.cos(omega_t)          # 当前曲率
    theta_t = kappa_peak * vel_cmd / omega * torch.sin(omega_t)  # 当前 heading

    # 梯形积分累积位置（从上一时刻到当前时刻）
    if getattr(env, "_slalom_ref_state", None) is None:
        env._slalom_ref_state = {  # type: ignore[attr-defined]
            "x": torch.zeros(env.num_envs, device=env.device),
            "y": torch.zeros(env.num_envs, device=env.device),
        }
    state = env._slalom_ref_state  # type: ignore[attr-defined]

    t_prev = torch.clamp(t - dt, min=0.0)
    omega_t_prev = omega * t_prev
    theta_prev = kappa_peak * vel_cmd / omega * torch.sin(omega_t_prev)

    # 梯形法则：Δx = v · (cos(θ_prev) + cos(θ_curr)) / 2 · dt
    dx = vel_cmd * (torch.cos(theta_prev) + torch.cos(theta_t)) / 2 * dt
    dy = vel_cmd * (torch.sin(theta_prev) + torch.sin(theta_t)) / 2 * dt
    state["x"] += dx
    state["y"] += dy

    # 重置已终止环境
    reset_ids = getattr(env, "reset_terminated", None)
    if reset_ids is not None:
        ids = reset_ids.nonzero(as_tuple=False).flatten()
        if len(ids) > 0:
            state["x"][ids] = 0.0
            state["y"][ids] = 0.0

    x_ref = state["x"]
    y_ref = state["y"]
    vx_des = vel_cmd * torch.cos(theta_t)
    vy_des = vel_cmd * torch.sin(theta_t)

    return x_ref, y_ref, vx_des, vy_des, theta_t


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
