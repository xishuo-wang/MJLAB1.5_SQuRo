from __future__ import annotations

import torch

from .config import BASE_HEIGHT

# 高度课程规则参数；可视化与训练共用此处定义。
OBSTACLE_HEIGHT_SPLIT = 0.070
ENTRY_MARGIN = 0.150
EXIT_MARGIN = 0.100
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
def rule_bounds(center_x: float, half_length: float) -> tuple[float, float, float]:
    x1 = center_x - half_length
    x2 = center_x + half_length
    left = x1 - ENTRY_MARGIN
    middle = (x1 + x2) / 2.0
    right = x2 + EXIT_MARGIN
    return left, middle, right


# 按 base_Link 的世界 X 统一决定前后端高度，低板两段互斥。
def sample_base_heights(
    base_x: torch.Tensor,
    center_x: float,
    half_length: float,
    bottom_z: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    left, middle, right = rule_bounds(center_x, half_length)
    bottom = torch.as_tensor(bottom_z, device=base_x.device, dtype=base_x.dtype)
    high_board = bottom >= OBSTACLE_HEIGHT_SPLIT
    front = torch.full_like(base_x, BASE_HEIGHT)
    hind = torch.full_like(base_x, BASE_HEIGHT)
    high_target = bottom - BODY_HEIGHT_OFFSET
    in_range = (base_x >= left) & (base_x <= right)
    front_low = in_range & (base_x < middle)
    hind_low = in_range & (base_x >= middle)
    front = torch.where(high_board & in_range, high_target, front)
    hind = torch.where(high_board & in_range, high_target, hind)
    front = torch.where((~high_board) & front_low, LOW_TARGET_HEIGHT, front)
    hind = torch.where((~high_board) & hind_low, LOW_TARGET_HEIGHT, hind)
    return front, hind
