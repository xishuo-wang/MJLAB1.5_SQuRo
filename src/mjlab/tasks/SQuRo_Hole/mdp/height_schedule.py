from __future__ import annotations

import torch

from .config import BASE_HEIGHT

# 高度课程规则参数；可视化与训练共用此处定义。
OBSTACLE_HEIGHT_SPLIT = 0.070
ENTRY_MARGIN = 0.150
EXIT_MARGIN = 0.100
SEGMENT_CENTER_DISTANCE = 0.0935
BODY_HEIGHT_OFFSET = 0.030
LOW_TARGET_HEIGHT = 0.020
HEIGHT_MIN = 0.048
HEIGHT_MAX = 0.090
HEIGHT_INITIAL_MIN = 0.070
HEIGHT_STEP = 0.002
HEIGHT_MIN_SAMPLE_PROBABILITY = 0.30
HEIGHT_MIN_SUCCESS_BATCH = 256
HEIGHT_MIN_SUCCESS_RATE = 0.60
HEIGHT_REQUIRED_BATCHES = 2
HEIGHT_MIN_DWELL_ITERATIONS = 100


# 从单板几何计算查表边界。
def rule_bounds(center_x: float, half_length: float) -> tuple[float, float, float, float]:
    x1 = center_x - half_length
    x2 = center_x + half_length
    left = x1 - ENTRY_MARGIN
    middle = (x1 + x2) / 2.0
    right = x2 + EXIT_MARGIN
    return left, middle, right, middle + SEGMENT_CENTER_DISTANCE / 2.0


# 按两个躯干段的实际世界 X 计算目标高度。
def sample_body_heights(
    front_x: torch.Tensor,
    hind_x: torch.Tensor,
    center_x: float,
    half_length: float,
    bottom_z: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    left, middle, right, front_end = rule_bounds(center_x, half_length)
    hind_start = middle - SEGMENT_CENTER_DISTANCE / 2.0
    bottom = torch.as_tensor(bottom_z, device=front_x.device, dtype=front_x.dtype)
    high_board = bottom >= OBSTACLE_HEIGHT_SPLIT
    front = torch.full_like(front_x, BASE_HEIGHT)
    hind = torch.full_like(hind_x, BASE_HEIGHT)
    high_target = bottom - BODY_HEIGHT_OFFSET
    front_high = (front_x >= left) & (front_x <= right)
    hind_high = (hind_x >= left) & (hind_x <= right)
    front_low = (front_x >= left) & (front_x < front_end)
    hind_low = (hind_x >= hind_start) & (hind_x <= right)
    front = torch.where(high_board & front_high, high_target, front)
    hind = torch.where(high_board & hind_high, high_target, hind)
    front = torch.where((~high_board) & front_low, LOW_TARGET_HEIGHT, front)
    hind = torch.where((~high_board) & hind_low, LOW_TARGET_HEIGHT, hind)
    return front, hind
