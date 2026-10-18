from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import matrix_from_quat
from .indices import _MODEL_INDICES, resolve_body_envelope_bounds
from .path import CORRIDOR_HALF_WIDTH
from .pole import POLE_RADIUS
from .slalom_geometry import PROJECTION_STRIDE, get_spatial_slalom_tables

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_BOX_SIGNS = tuple(product((-1.0, 1.0), repeat=3))
_BOX_EDGES = tuple((i, j) for i in range(8) for j in range(i + 1, 8)
                   if sum(a != b for a, b in zip(_BOX_SIGNS[i], _BOX_SIGNS[j], strict=True)) == 1)
_ENVELOPE_SAMPLE_STEP = 0.002
_PROJECTION_ENV_BLOCK = 64
_PHASE0_PROJECTION_STRIDE = 8
_PHASE0_ENVELOPE_SAMPLE_STEP = 0.006   # 一阶段动态回合的包络边采样间距 (m)


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


# 以固定形状分块查询，避免阶段切换时不同大批量形状累积显存预留。
def project_slalom_points(
    points: torch.Tensor, tables: torch.Tensor, rows: torch.Tensor, spacing: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = points.shape[0]
    outputs = tuple(points.new_empty(points.shape[:2]) for _ in range(3))
    if count == 0:
        return outputs
    block_size = min(_PROJECTION_ENV_BLOCK, 1 << (count - 1).bit_length())
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        block_points, block_rows, block_spacing = points[start:stop], rows[start:stop], spacing[start:stop]
        padding = block_size - (stop - start)
        if padding:
            block_points = torch.cat((block_points, block_points[-1:].expand(padding, -1, -1)))
            block_rows = torch.cat((block_rows, block_rows[-1:].expand(padding)))
            block_spacing = torch.cat((block_spacing, block_spacing[-1:].expand(padding)))
        values = _project_slalom_points_block(block_points, tables, block_rows, block_spacing)
        for output, value in zip(outputs, values, strict=True):
            output[start:stop].copy_(value[:stop - start])
    return outputs


# 粗定位后在线段上细化投影，接近段起点向后延伸为切线射线。
def _project_slalom_points_block(
    points: torch.Tensor, tables: torch.Tensor, rows: torch.Tensor, spacing: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points = points.clone()
    period_x = 2 * spacing[:, None]
    cycles = torch.floor((points[..., 0] - period_x) / period_x).clamp_min(0)
    points[..., 0] -= cycles * period_x
    coarse = tables[rows, ::PROJECTION_STRIDE, :2]
    distance_sq = (points.square().sum(-1, keepdim=True) + coarse.square().sum(-1)[:, None]
                   - 2 * torch.bmm(points, coarse.transpose(1, 2)))
    nearest = distance_sq.argmin(dim=-1) * PROJECTION_STRIDE
    offsets = torch.arange(-PROJECTION_STRIDE, PROJECTION_STRIDE, device=points.device)
    indices = (nearest[..., None] + offsets).clamp(0, tables.shape[1] - 2)
    start = tables[rows[:, None, None], indices]
    end = tables[rows[:, None, None], indices + 1]
    edges = end[..., :2] - start[..., :2]
    delta = points[:, :, None] - start[..., :2]
    fraction = ((delta * edges).sum(-1) / edges.square().sum(-1).clamp_min(1e-20)).clamp(0, 1)
    residual = delta - fraction[..., None] * edges
    best = residual.square().sum(-1).argmin(-1, keepdim=True)
    chosen_residual = residual.gather(2, best[..., None].expand(-1, -1, -1, 2)).squeeze(2)
    chosen_edge = edges.gather(2, best[..., None].expand(-1, -1, -1, 2)).squeeze(2)
    curvature = torch.lerp(start[..., 3], end[..., 3], fraction).gather(2, best).squeeze(2)
    distance = torch.linalg.vector_norm(chosen_residual, dim=-1)
    lateral = chosen_edge[..., 0] * chosen_residual[..., 1] - chosen_edge[..., 1] * chosen_residual[..., 0]

    origin = tables[rows, 0]
    tangent = torch.stack((origin[:, 2].cos(), origin[:, 2].sin()), dim=-1)
    ray_delta = points - origin[:, None, :2]
    ray_residual = ray_delta - (ray_delta * tangent[:, None]).sum(-1).clamp_max(0)[..., None] * tangent[:, None]
    ray_distance = torch.linalg.vector_norm(ray_residual, dim=-1)
    use_ray = ray_distance < distance
    ray_lateral = tangent[:, None, 0] * ray_residual[..., 1] - tangent[:, None, 1] * ray_residual[..., 0]
    return (torch.minimum(distance, ray_distance), torch.where(use_ray, ray_lateral, lateral),
            torch.where(use_ray, 0.0, curvature))


# 沿投影盒十二条边采样，边界相邻点间距不超过给定步长 (按步长分别缓存)
def sample_body_envelopes(env: ManagerBasedRlEnv, vertices: torch.Tensor,
                          step: float = _ENVELOPE_SAMPLE_STEP) -> torch.Tensor:
    cache = getattr(env, "_slalom_envelope_samples", None)
    if cache is None:
        cache = {}
        env._slalom_envelope_samples = cache
    cached = cache.get(float(step))
    if cached is None:
        local = env._slalom_body_envelope_vertices
        starts, ends, fractions = list(range(8)), list(range(8)), [0.0] * 8
        for start, end in _BOX_EDGES:
            count = max(1, int(torch.ceil(torch.linalg.vector_norm(local[:, end] - local[:, start], dim=-1).max()
                                         / step).item()))
            starts.extend([start] * (count - 1))
            ends.extend([end] * (count - 1))
            fractions.extend(i / count for i in range(1, count))
        cached = (torch.tensor(starts, device=env.device), torch.tensor(ends, device=env.device),
                  torch.tensor(fractions, device=env.device, dtype=vertices.dtype))
        cache[float(step)] = cached
    starts, ends, fractions = cached
    return torch.lerp(vertices[:, :, starts], vertices[:, :, ends], fractions[None, None, :, None])


# 二阶段按各包络点的最近实际轨迹计算内外越界，不依赖参考时钟的位置。
def compute_phase1_corridor_excess(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    vertices = project_body_envelopes(env)[env_ids]
    points = sample_body_envelopes(env, vertices)
    origins = getattr(env.scene, "env_origins", None)
    if origins is not None:
        points = points - origins[env_ids, None, None, :2]
    tables = getattr(env, "_slalom_spatial_tables", None)
    if tables is None:
        tables = torch.as_tensor(get_spatial_slalom_tables(), device=env.device, dtype=points.dtype)
        env._slalom_spatial_tables = tables
    cmd = env.command_manager._terms["slalom_cmd"]
    distance, lateral, curvature = project_slalom_points(
        points.flatten(1, 2), tables, cmd.slalom_curvature_index[env_ids], cmd.pole_spacing[env_ids])
    excess = (distance - CORRIDOR_HALF_WIDTH).clamp_min(0)
    inner_side = curvature * lateral > 0
    inner = torch.where(inner_side, excess, 0.0).reshape(points.shape[:-1]).amax(-1)
    outer = torch.where(inner_side, 0.0, excess).reshape(points.shape[:-1]).amax(-1)
    return inner, outer

# 一阶段路径投影: 返回包络点到本回合实际路径 (含起终点切线延伸) 的距离、侧向叉积与局部曲率
def project_phase0_path_points(points: torch.Tensor, path_xy: torch.Tensor,
                               path_heading: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stride = _PHASE0_PROJECTION_STRIDE
    coarse = path_xy[:, ::stride]
    distance_sq = (points.square().sum(-1, keepdim=True) + coarse.square().sum(-1)[:, None]
                   - 2.0 * torch.bmm(points, coarse.transpose(1, 2)))
    nearest = distance_sq.argmin(dim=-1) * stride
    offsets = torch.arange(-stride, stride, device=points.device)
    indices = (nearest[..., None] + offsets).clamp(0, path_xy.shape[1] - 2)
    rows = torch.arange(points.shape[0], device=points.device)[:, None, None]
    start = path_xy[rows, indices]
    end = path_xy[rows, indices + 1]
    edges = end - start
    delta = points[:, :, None] - start
    fraction = ((delta * edges).sum(-1) / edges.square().sum(-1).clamp_min(1e-20)).clamp(0.0, 1.0)
    residual = delta - fraction[..., None] * edges
    best = residual.square().sum(-1).argmin(-1, keepdim=True)
    chosen_residual = residual.gather(2, best[..., None].expand(-1, -1, -1, 2)).squeeze(2)
    chosen_edge = edges.gather(2, best[..., None].expand(-1, -1, -1, 2)).squeeze(2)
    distance = torch.linalg.vector_norm(chosen_residual, dim=-1)
    lateral = chosen_edge[..., 0] * chosen_residual[..., 1] - chosen_edge[..., 1] * chosen_residual[..., 0]
    head_start = path_heading[rows, indices]
    head_end = path_heading[rows, indices + 1]
    segment_length = torch.linalg.vector_norm(chosen_edge, dim=-1).clamp_min(1e-9)
    curvature = (head_end - head_start).gather(2, best).squeeze(2) / segment_length
    # 起终点切线延伸: 起点向后、终点向前各作一条射线, 身体跨出路径端点时按射线量距
    for position, extend_back in ((0, True), (-1, False)):
        tangent = torch.stack((torch.cos(path_heading[:, position]),
                               torch.sin(path_heading[:, position])), dim=-1)
        offset = points - path_xy[:, position][:, None, :]
        along = (offset * tangent[:, None]).sum(-1)
        along = along.clamp_max(0.0) if extend_back else along.clamp_min(0.0)
        ray_residual = offset - along[..., None] * tangent[:, None]
        ray_distance = torch.linalg.vector_norm(ray_residual, dim=-1)
        ray_lateral = tangent[:, None, 0] * ray_residual[..., 1] - tangent[:, None, 1] * ray_residual[..., 0]
        use_ray = ray_distance < distance
        distance = torch.where(use_ray, ray_distance, distance)
        lateral = torch.where(use_ray, ray_lateral, lateral)
        curvature = torch.where(use_ray, torch.zeros_like(curvature), curvature)
    return distance, lateral, curvature


# 一阶段动态回合: 按各包络采样点到本回合实际路径的最近距离计算内外越界
def compute_phase0_dynamic_corridor_excess(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    vertices = project_body_envelopes(env)[env_ids]
    points = sample_body_envelopes(env, vertices, _PHASE0_ENVELOPE_SAMPLE_STEP)
    origins = getattr(env.scene, "env_origins", None)
    if origins is not None:
        points = points - origins[env_ids, None, None, :2]
    path_xy = getattr(env, "_phase0_path_xy")[env_ids]
    path_heading = getattr(env, "_phase0_path_heading")[env_ids]
    flat = points.flatten(1, 2)
    # 分块投影, 限制大中间张量的峰值显存
    distances, laterals, curvatures = [], [], []
    for block in range(0, len(env_ids), _PROJECTION_ENV_BLOCK):
        section = slice(block, min(block + _PROJECTION_ENV_BLOCK, len(env_ids)))
        distance, lateral, curvature = project_phase0_path_points(
            flat[section], path_xy[section], path_heading[section])
        distances.append(distance)
        laterals.append(lateral)
        curvatures.append(curvature)
    distance = torch.cat(distances)
    lateral = torch.cat(laterals)
    curvature = torch.cat(curvatures)
    excess = (distance - CORRIDOR_HALF_WIDTH).clamp_min(0.0)
    inner_side = curvature * lateral > 0.0
    inner = torch.where(inner_side, excess, torch.zeros_like(excess)).reshape(points.shape[:-1]).amax(-1)
    outer = torch.where(inner_side, torch.zeros_like(excess), excess).reshape(points.shape[:-1]).amax(-1)
    return inner, outer
