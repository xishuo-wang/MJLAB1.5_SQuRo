from functools import lru_cache
from typing import Any

import numpy as np

from .curriculums import SLALOM_CURVATURES, SMOOTH_TIME, SMOOTH_VEL, get_velocity_scale


_SEGMENT_SAMPLES = 256
_APPROACH_SAMPLES = 256
APPROACH_LENGTH = 0.03


# 按每段真实步长积分曲率，保留位置、切线和曲率的连续端点。
def integrate_segments(segments: list[tuple[float, float, float]]) -> dict[str, np.ndarray]:
    lengths = np.concatenate([np.full(_SEGMENT_SAMPLES, length / _SEGMENT_SAMPLES)
                              for length, _, _ in segments])
    pieces = [np.linspace(start, end, _SEGMENT_SAMPLES + 1) for _, start, end in segments]
    kappa = np.concatenate([pieces[0]] + [piece[1:] for piece in pieces[1:]])
    delta_heading = 0.5 * (kappa[:-1] + kappa[1:]) * lengths
    heading = np.concatenate(([0.0], np.cumsum(delta_heading)))
    mid_heading = heading[:-1] + delta_heading / 2
    return {
        "arc": np.concatenate(([0.0], np.cumsum(lengths))),
        "x": np.concatenate(([0.0], np.cumsum(np.cos(mid_heading) * lengths))),
        "y": np.concatenate(([0.0], np.cumsum(np.sin(mid_heading) * lengths))),
        "heading": heading,
        "kappa": kappa,
    }


# 构建整数峰值对应的无直行路径；杆心距取半周期的真实纵向位移。
@lru_cache(maxsize=len(SLALOM_CURVATURES))
def get_slalom_geometry(curvature: int) -> dict[str, Any]:
    if curvature not in SLALOM_CURVATURES:
        raise ValueError(f"绕杆峰值曲率必须是 {SLALOM_CURVATURES[0]}～{SLALOM_CURVATURES[-1]} 的整数")
    k = float(curvature)
    transition = SMOOTH_TIME * SMOOTH_VEL
    platform = np.pi / (2 * k) - transition / 2
    period = integrate_segments([
        (platform, -k, -k), (transition, -k, 0.0),
        (transition, 0.0, k), (platform, k, k),
        (platform, k, k), (transition, k, 0.0),
        (transition, 0.0, -k), (platform, -k, -k),
    ])
    approach = integrate_segments([(transition, 0.0, -k), (APPROACH_LENGTH - transition, -k, -k)])
    approach["heading"] -= approach["heading"][-1]
    ds = np.diff(approach["arc"])
    mid_heading = (approach["heading"][:-1] + approach["heading"][1:]) / 2
    approach["x"] = np.concatenate(([0.0], np.cumsum(np.cos(mid_heading) * ds)))
    approach["y"] = np.concatenate(([0.0], np.cumsum(np.sin(mid_heading) * ds)))
    spacing = float(period["x"][-1] / 2)
    approach["x"] += spacing - approach["x"][-1]
    approach["y"] -= approach["y"][-1]
    return {"curvature": curvature, "spacing": spacing,
            "pole_y": float((period["y"].min() + period["y"].max()) / 2),
            "period": period, "approach": approach}


# 将指定杆距映射到最近的整数曲率条件，实际杆距必须向调用者回显。
def curvature_from_spacing(spacing: float) -> int:
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("杆心距必须是有限正数")
    return min(SLALOM_CURVATURES, key=lambda k: abs(get_slalom_geometry(k)["spacing"] - spacing))


# 将空间路径重采样到等名义时间网格，运行时只需按行插值。
def timed_samples(path: dict[str, np.ndarray], base_velocity: float, count: int) -> tuple[np.ndarray, float]:
    velocity = base_velocity * get_velocity_scale(path["kappa"])
    time = np.concatenate(([0.0], np.cumsum(np.diff(path["arc"]) / ((velocity[:-1] + velocity[1:]) / 2))))
    samples = np.stack([np.interp(np.linspace(0, time[-1], count), time, path[name])
                        for name in ("x", "y", "heading", "kappa")], axis=-1)
    return samples, float(time[-1])


# 预计算全部整数曲率，步频仅在运行时缩放时间。
@lru_cache(maxsize=8)
def get_timed_slalom_tables(base_velocity: float) -> dict[str, np.ndarray]:
    if not np.isfinite(base_velocity) or base_velocity <= 0:
        raise ValueError("基础速度必须是有限正数")
    geometries = [get_slalom_geometry(k) for k in SLALOM_CURVATURES]
    periods = [timed_samples(g["period"], base_velocity, 8 * _SEGMENT_SAMPLES + 1) for g in geometries]
    approaches = [timed_samples(g["approach"], base_velocity, _APPROACH_SAMPLES + 1) for g in geometries]
    return {
        "period": np.stack([p[0] for p in periods]),
        "period_time": np.array([p[1] for p in periods]),
        "approach": np.stack([a[0] for a in approaches]),
        "approach_time": np.array([a[1] for a in approaches]),
        "spacing": np.array([g["spacing"] for g in geometries]),
        "pole_y": np.array([g["pole_y"] for g in geometries]),
        "start": np.array([[g["approach"][name][0] for name in ("x", "y", "heading")] for g in geometries]),
    }
