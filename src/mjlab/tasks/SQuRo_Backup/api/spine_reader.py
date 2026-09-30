from __future__ import annotations

import torch

from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.utils.spine_deformation import SpineDeformation, SpineDeformationTracker

from ..mdp.indices import (
  SPINE_FRONT_FRAME,
  SPINE_HIND_FRAME,
  resolve_spine_deformation_indices,
)


class SQuRoSpineDeformation:
  # 对接 SQuRo 实体的实际位姿；偏置以各刚体原生局部坐标表达。
  def __init__(
    self,
    robot,
    *,
    front_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    hind_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
  ) -> None:
    self.robot = robot
    self.body_ids = resolve_spine_deformation_indices(robot)
    sample = robot.data.body_link_pos_w
    self.offsets = sample.new_tensor((front_offset, center_offset, hind_offset))
    if self.offsets.shape != (3, 3) or not bool(torch.isfinite(self.offsets).all()):
      raise ValueError("三个局部参考点偏置必须各含三个有限数")
    self.front_frame = sample.new_tensor(SPINE_FRONT_FRAME)
    self.hind_frame = sample.new_tensor(SPINE_HIND_FRAME)
    self.tracker = SpineDeformationTracker()

  # 完整回合复位和成功后的中途物理复位都需要由调用方通知。
  def reset(self, env_ids: torch.Tensor | list[int] | slice | None = None) -> None:
    self.tracker.reset(env_ids)

  # 只读仿真状态，返回每个环境的弧度角和有效性，不自动写入日志。
  def compute(self) -> SpineDeformation:
    data = self.robot.data
    positions = data.body_link_pos_w[:, list(self.body_ids)]
    rotations = matrix_from_quat(data.body_link_quat_w[:, list(self.body_ids)])
    points = positions + (rotations @ self.offsets[..., None]).squeeze(-1)
    return self.tracker.compute(
      points[:, 0],
      points[:, 1],
      points[:, 2],
      rotations[:, 0] @ self.front_frame,
      rotations[:, 2] @ self.hind_frame,
    )
