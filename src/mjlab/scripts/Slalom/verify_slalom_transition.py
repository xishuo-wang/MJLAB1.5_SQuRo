import torch

from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import (
    CURVATURE_TARGET_MAX,
    REFERENCE_CURVATURE_BLEND,
)
from mjlab.tasks.SQuRo_Slalom.mdp.reference import smooth_curvature_magnitude


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


if __name__ == "__main__":
    verify_reference()
