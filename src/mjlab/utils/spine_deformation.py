from __future__ import annotations

import math
from dataclasses import dataclass

import torch

GEOMETRY_EPS = 1e-6


@dataclass(frozen=True)
class SpineDeformationState:
  axial_unwrapped: torch.Tensor
  valid: torch.Tensor


@dataclass(frozen=True)
class SpineDeformation:
  lateral: torch.Tensor
  sagittal: torch.Tensor
  axial: torch.Tensor
  axial_unwrapped: torch.Tensor
  frame_w: torch.Tensor
  bending_valid: torch.Tensor
  axial_valid: torch.Tensor
  branch_ambiguous: torch.Tensor
  state: SpineDeformationState

  # 三列依次为侧摆、俯仰、主值扭转，单位为弧度。
  @property
  def angles(self) -> torch.Tensor:
    return torch.stack((self.lateral, self.sagittal, self.axial), dim=-1)

  # 三个角均有定义时才将该样本标记为有效。
  @property
  def valid(self) -> torch.Tensor:
    return self.bending_valid & self.axial_valid


# 单位化向量，同时保留零长度和非有限输入的有效性。
def _unit(vector: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
  length = torch.linalg.vector_norm(vector, dim=-1)
  valid = torch.isfinite(vector).all(dim=-1) & (length > eps)
  return vector / length.clamp_min(eps).unsqueeze(-1), valid


# 把横向向量随纵轴的最短旋转平行搬运到共同纵轴。
def _transport(
  vector: torch.Tensor, source: torch.Tensor, target: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
  cross = torch.linalg.cross(source, target)
  cosine = (source * target).sum(dim=-1).clamp(-1.0, 1.0)
  moved = vector + torch.linalg.cross(cross, vector)
  moved = moved + torch.linalg.cross(cross, torch.linalg.cross(cross, vector)) / (
    1.0 + cosine
  ).clamp_min(eps).unsqueeze(-1)
  moved = moved - (moved * target).sum(dim=-1, keepdim=True) * target
  moved, valid = _unit(moved, eps)
  return moved, valid & (1.0 + cosine > eps)


# 验证输入为右手正交旋转矩阵，避免把非单位四元数误差当作形变。
def _rotation_valid(rotation: torch.Tensor, eps: float) -> torch.Tensor:
  eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
  error = (rotation.transpose(-1, -2) @ rotation - eye).abs().amax(dim=(-1, -2))
  handed = (
    torch.linalg.cross(rotation[..., :, 0], rotation[..., :, 1]) * rotation[..., :, 2]
  ).sum(dim=-1)
  return (
    torch.isfinite(rotation).all(dim=-1).all(dim=-1)
    & (error < 100 * eps)
    & (handed > 0)
  )


# 计算三点折线在平均扭转身体平面中的形变；完整约定见 docs/SQuRo_实际脊柱形变角.md。
def compute_spine_deformation(
  front_pos_w: torch.Tensor,
  center_pos_w: torch.Tensor,
  hind_pos_w: torch.Tensor,
  front_frame_w: torch.Tensor,
  hind_frame_w: torch.Tensor,
  state: SpineDeformationState | None = None,
  *,
  eps: float = GEOMETRY_EPS,
) -> SpineDeformation:
  shape = front_pos_w.shape
  if (
    not shape
    or shape[-1] != 3
    or front_pos_w.dtype not in (torch.float32, torch.float64)
  ):
    raise ValueError("位置必须是 float32/float64 的 (..., 3) 张量")
  if not math.isfinite(eps) or eps <= 0:
    raise ValueError("eps 必须为正的有限数")
  for value, expected in (
    (center_pos_w, shape),
    (hind_pos_w, shape),
    (front_frame_w, shape[:-1] + (3, 3)),
    (hind_frame_w, shape[:-1] + (3, 3)),
  ):
    if (
      value.shape != expected
      or value.dtype != front_pos_w.dtype
      or value.device != front_pos_w.device
    ):
      raise ValueError("位置、姿态的批量维度、精度和设备必须一致")
  if state is not None:
    if (
      state.axial_unwrapped.shape != shape[:-1]
      or state.valid.shape != shape[:-1]
      or state.axial_unwrapped.dtype != front_pos_w.dtype
      or state.valid.dtype != torch.bool
      or state.axial_unwrapped.device != front_pos_w.device
      or state.valid.device != front_pos_w.device
    ):
      raise ValueError("历史状态必须匹配当前输入的批量维度、精度和设备")

  # 前后连线都朝向头部，中间纵轴取两条单位连线的角平分方向。
  forward_f, valid_f = _unit(front_pos_w - center_pos_w, eps)
  forward_h, valid_h = _unit(center_pos_w - hind_pos_w, eps)
  middle_x, valid_x = _unit(forward_f + forward_h, eps)
  front_back, transported_f = _transport(
    front_frame_w[..., :, 2], front_frame_w[..., :, 0], middle_x, eps
  )
  hind_back, transported_h = _transport(
    hind_frame_w[..., :, 2], hind_frame_w[..., :, 0], middle_x, eps
  )
  axial_valid = valid_f & valid_h & valid_x & transported_f & transported_h
  axial_valid = (
    axial_valid
    & _rotation_valid(front_frame_w, eps)
    & _rotation_valid(hind_frame_w, eps)
  )
  axial = torch.atan2(
    (middle_x * torch.linalg.cross(hind_back, front_back)).sum(dim=-1),
    (hind_back * front_back).sum(dim=-1),
  )

  # 相对扭转连续展开后取半角，不对相反的两个背向向量直接求和。
  previous_valid = (
    torch.zeros_like(axial_valid)
    if state is None
    else state.valid & torch.isfinite(state.axial_unwrapped)
  )
  previous = (
    torch.zeros_like(axial)
    if state is None
    else torch.where(previous_valid, state.axial_unwrapped, 0.0)
  )
  delta = torch.atan2(torch.sin(axial - previous), torch.cos(axial - previous))
  unwrapped = torch.where(previous_valid, previous + delta, axial)
  ambiguous = axial_valid & (
    (math.pi - torch.where(previous_valid, delta.abs(), axial.abs())).abs() <= eps
  )
  frame_valid = axial_valid & ~ambiguous
  half = unwrapped.unsqueeze(-1) / 2
  middle_z = hind_back * torch.cos(half) + torch.linalg.cross(
    middle_x, hind_back
  ) * torch.sin(half)
  middle_y = torch.linalg.cross(middle_z, middle_x)
  frame = torch.stack((middle_x, middle_y, middle_z), dim=-1)

  # 各投影的有符号夹角：左弯为正，前段朝背侧弯为正。
  f = (frame.transpose(-1, -2) @ forward_f.unsqueeze(-1)).squeeze(-1)
  h = (frame.transpose(-1, -2) @ forward_h.unsqueeze(-1)).squeeze(-1)
  lateral = torch.atan2(
    h[..., 0] * f[..., 1] - h[..., 1] * f[..., 0],
    h[..., 0] * f[..., 0] + h[..., 1] * f[..., 1],
  )
  sagittal = torch.atan2(
    h[..., 0] * f[..., 2] - h[..., 2] * f[..., 0],
    h[..., 0] * f[..., 0] + h[..., 2] * f[..., 2],
  )
  projections_valid = torch.ones_like(frame_valid)
  for vector in (f, h):
    for axis in (1, 2):
      projections_valid = projections_valid & (
        vector[..., 0].square() + vector[..., axis].square() > eps * eps
      )
  bending_valid = frame_valid & projections_valid
  nan = float("nan")
  next_state = SpineDeformationState(
    torch.where(frame_valid, unwrapped, 0.0).detach().clone(),
    frame_valid.detach().clone(),
  )
  return SpineDeformation(
    lateral=torch.where(bending_valid, lateral, nan),
    sagittal=torch.where(bending_valid, sagittal, nan),
    axial=torch.where(axial_valid, axial, nan),
    axial_unwrapped=torch.where(frame_valid, unwrapped, nan),
    frame_w=torch.where(frame_valid[..., None, None], frame, nan),
    bending_valid=bending_valid,
    axial_valid=axial_valid,
    branch_ambiguous=ambiguous,
    state=next_state,
  )


class SpineDeformationTracker:
  # 一个实例对应一批有固定顺序的环境，历史不会在实例之间共享。
  def __init__(self) -> None:
    self.state: SpineDeformationState | None = None

  # 物理复位后清理全部或指定环境的历史；不修改之前返回的结果。
  def reset(self, env_ids: torch.Tensor | list[int] | slice | None = None) -> None:
    if env_ids is None:
      self.state = None
    elif self.state is not None:
      axial = self.state.axial_unwrapped.clone()
      valid = self.state.valid.clone()
      axial[env_ids] = 0.0
      valid[env_ids] = False
      self.state = SpineDeformationState(axial, valid)

  # 按时间顺序输入实际位姿，每步返回角度及可用于诊断的有效性。
  def compute(
    self,
    front_pos_w: torch.Tensor,
    center_pos_w: torch.Tensor,
    hind_pos_w: torch.Tensor,
    front_frame_w: torch.Tensor,
    hind_frame_w: torch.Tensor,
  ) -> SpineDeformation:
    result = compute_spine_deformation(
      front_pos_w, center_pos_w, hind_pos_w, front_frame_w, hind_frame_w, self.state
    )
    self.state = result.state
    return result
