import numpy as np
import torch

from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import (
    CURVATURE_TARGET_MAX,
    REFERENCE_CURVATURE_BLEND,
    get_velocity_scale,
)
from mjlab.tasks.SQuRo_Slalom.mdp.reference import smooth_curvature_magnitude
from mjlab.tasks.SQuRo_Slalom.mdp.slalom_geometry import get_slalom_geometry, get_timed_slalom_tables


# 检查后脊柱过零参考的连续性、解析导数和端点形态。
def verify_reference() -> None:
    width = REFERENCE_CURVATURE_BLEND
    curvature = torch.linspace(-20.0, 20.0, 40001, dtype=torch.float64)
    magnitude, slope = smooth_curvature_magnitude(curvature)
    delta = 1e-5
    right, _ = smooth_curvature_magnitude(curvature + delta)
    left, _ = smooth_curvature_magnitude(curvature - delta)
    torch.testing.assert_close((right - left) / (2 * delta), slope, atol=2e-8, rtol=2e-6)
    assert slope.abs().max() <= 1.0
    assert torch.all(magnitude >= curvature.abs())
    outside = curvature.abs() >= width
    torch.testing.assert_close(magnitude[outside], curvature[outside].abs())
    zero = torch.tensor([-1e-6, 0.0, 1e-6], dtype=torch.float64)
    _, zero_slope = smooth_curvature_magnitude(zero)
    assert zero_slope.abs().max() < 1e-5
    for endpoint in (-20.0, -15.0, 15.0, 20.0):
        value, derivative = smooth_curvature_magnitude(torch.tensor(endpoint))
        assert float(value) == abs(endpoint)
        assert float(derivative) == (1.0 if endpoint > 0 else -1.0)
    zero_angle = -0.65 * (3 * width / 8) / CURVATURE_TARGET_MAX
    print(f"后脊柱参考验证通过：位置及一阶、二阶导数连续，零点角度 {zero_angle:.6f} rad，十五／二十端点不变。")


# 检查全部几何的时钟积分与峰值限速一致，过零不再加速。
def verify_velocity() -> None:
    for base in (0.07, 0.1):
        tables = get_timed_slalom_tables(base)
        for row, peak in enumerate(range(5, 16)):
            speed = base * get_velocity_scale(peak)
            geometry = get_slalom_geometry(peak)
            for part in ("approach", "period"):
                expected = geometry[part]["arc"][-1] / speed
                np.testing.assert_allclose(tables[f"{part}_time"][row], expected, rtol=1e-12)
                points = tables[part][row]
                actual = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
                actual /= expected / (len(points) - 1)
                np.testing.assert_allclose(actual, speed, rtol=2e-4)
    print("绕杆速度验证通过：全部整数曲率的接近段、周期时间积分同源，过零保持本回合峰值速度。")


if __name__ == "__main__":
    verify_reference()
    verify_velocity()
