from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import matrix_from_quat
from .indices import _MODEL_INDICES, resolve_body_envelope_bounds
from .path import CORRIDOR_HALF_WIDTH
from .pole import POLE_RADIUS

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_BOX_SIGNS = tuple(product((-1.0, 1.0), repeat=3))
_BOX_EDGES = tuple((i, j) for i in range(8) for j in range(i + 1, 8)
                   if sum(a != b for a, b in zip(_BOX_SIGNS[i], _BOX_SIGNS[j], strict=True)) == 1)


# 将完整三维包围盒随躯干实际姿态投影到水平面。
def project_body_envelopes(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    local_vertices = getattr(env, "_slalom_body_envelope_vertices", None)
    if local_vertices is None:
        bounds = torch.as_tensor(resolve_body_envelope_bounds(robot, env.sim.mj_model),
                                 device=env.device, dtype=torch.float32)
        signs = torch.tensor(_BOX_SIGNS, device=env.device, dtype=bounds.dtype)
        center = bounds.mean(dim=1)
        half_size = (bounds[:, 1] - bounds[:, 0]) / 2
        local_vertices = center[:, None] + signs[None] * half_size[:, None]
        env._slalom_body_envelope_vertices = local_vertices
    body_ids = (_MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id)
    positions = robot.data.body_link_pos_w[:, body_ids]
    rotations = matrix_from_quat(robot.data.body_link_quat_w[:, body_ids])
    vertices = torch.einsum("nbij,bvj->nbvi", rotations, local_vertices) + positions[:, :, None]
    return vertices[..., :2]


# 稳定计算包络点相对参考圆半径的径向偏差。
def _radial_offset(points: torch.Tensor, normal: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    scaled = curvature[:, None, None] * points - normal[:, None]
    numerator = (curvature.abs()[:, None] * points.square().sum(dim=-1)
                 - 2 * curvature.sign()[:, None] * (points * normal[:, None]).sum(dim=-1))
    return numerator / (torch.linalg.vector_norm(scaled, dim=-1) + 1)


# 检查整个投影盒到圆心的最近边界和最远角点，详见一阶段训练文档。
def compute_annulus_side_excess(
    vertices_xy: torch.Tensor,
    ref_xy: torch.Tensor,
    path_heading: torch.Tensor,
    curvature: torch.Tensor,
    pole_clearance: float = POLE_RADIUS,
) -> tuple[torch.Tensor, torch.Tensor]:
    normal = torch.stack((-path_heading.sin(), path_heading.cos()), dim=-1)
    relative = vertices_xy - ref_xy[:, None]
    turning = curvature.abs() > 1e-6
    safe_curvature = torch.where(turning, curvature, torch.ones_like(curvature))
    radius = safe_curvature.abs().reciprocal()

    # 圆心到十二条投影边的最近点包含多边形边界的最近点。
    starts = relative[:, [edge[0] for edge in _BOX_EDGES]]
    edges = relative[:, [edge[1] for edge in _BOX_EDGES]] - starts
    scaled_center_delta = normal[:, None] - safe_curvature[:, None, None] * starts
    denominator = safe_curvature[:, None] * edges.square().sum(dim=-1).clamp_min(1e-20)
    fraction = ((scaled_center_delta * edges).sum(dim=-1) / denominator).clamp(0, 1)
    nearest_points = starts + fraction[..., None] * edges
    inner_offset = _radial_offset(nearest_points, normal, safe_curvature).amin(dim=-1)
    outer_offset = _radial_offset(relative, normal, safe_curvature).amax(dim=-1)

    # 圆心落入完整投影多边形时，最近距离为零。
    generators = (relative[:, (4, 2, 1)] - relative[:, :1]) / 2
    axes = torch.stack((-generators[..., 1], generators[..., 0]), dim=-1)
    center_delta = normal - safe_curvature[:, None] * relative.mean(dim=1)
    projections = torch.einsum("nai,ni->na", axes, center_delta).abs()
    support = torch.einsum("nai,nbi->nab", axes, generators).abs().sum(dim=-1)
    inside = (projections <= safe_curvature.abs()[:, None] * support + 1e-12).all(dim=-1)
    inner_offset = torch.where(inside, -radius, inner_offset)
    min_distance = torch.where(inside, torch.zeros_like(radius), radius + inner_offset).clamp_min(0)
    inner = torch.maximum((-CORRIDOR_HALF_WIDTH - inner_offset).clamp_min(0),
                          (pole_clearance - min_distance).clamp_min(0))
    outer = (outer_offset - CORRIDOR_HALF_WIDTH).clamp_min(0)

    lateral = (relative * normal[:, None]).sum(dim=-1)
    straight = (lateral.abs().amax(dim=-1) - CORRIDOR_HALF_WIDTH).clamp_min(0)
    return torch.where(turning, inner, 0.0), torch.where(turning, outer, straight)


# 按实际前后体包络计算一阶段圆环越界，直行退化为对称带。
def compute_phase0_corridor_excess(
    env: ManagerBasedRlEnv,
    ref_xy: torch.Tensor,
    path_heading: torch.Tensor,
    curvature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    vertices = project_body_envelopes(env)
    batch_size = vertices.shape[0]
    inner, outer = compute_annulus_side_excess(
        vertices.reshape(-1, 8, 2),
        ref_xy[:, None].expand(-1, 2, -1).reshape(-1, 2),
        path_heading[:, None].expand(-1, 2).reshape(-1),
        curvature[:, None].expand(-1, 2).reshape(-1),
    )
    return inner.reshape(batch_size, 2), outer.reshape(batch_size, 2)
