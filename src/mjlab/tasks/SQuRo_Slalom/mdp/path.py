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
