# 独立凸包几何验证圆环边界、完整包络、左右对称和小曲率稳定性。
from itertools import product
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
import torch

from mjlab.asset_zoo.robots.SQuRo.SQuRo_constants import get_spec
from mjlab.tasks.SQuRo_Slalom.mdp import reference
from mjlab.tasks.SQuRo_Slalom.mdp.config import CORRIDOR_HALF_WIDTH
from mjlab.tasks.SQuRo_Slalom.mdp.entity import POLE_RADIUS
from mjlab.tasks.SQuRo_Slalom.mdp.indices import BODY_NAMES, resolve_body_envelope_bounds
from mjlab.tasks.SQuRo_Slalom.mdp.reference import _BOX_SIGNS, project_body_envelopes
from mjlab.tasks.SQuRo_Slalom.mdp.rewards import compute_annulus_side_excess
from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import CURVATURE_TARGET_MAX


# 生成可旋转的完整矩形投影，保留三维盒的顶点排列。
def rectangle(center, heading=0.0, length=0.04, width=0.035):
    points = torch.tensor(_BOX_SIGNS, dtype=torch.float64)
    points = points[:, :2] * torch.tensor((length, width))
    rotation = torch.tensor(((np.cos(heading), -np.sin(heading)),
                             (np.sin(heading), np.cos(heading))), dtype=torch.float64)
    return points @ rotation.T + torch.tensor(center, dtype=torch.float64)


# 使用独立凸包和线段距离给出圆环越界基准。
def oracle(vertices, reference, heading, curvature):
    normal = np.array((-np.sin(heading), np.cos(heading)))
    if abs(curvature) <= 1e-6:
        return 0.0, max(0.0, np.abs((vertices - reference) @ normal).max() - CORRIDOR_HALF_WIDTH)
    radius = 1.0 / abs(curvature)
    center = reference + normal / curvature
    hull = ConvexHull(vertices)
    polygon = vertices[hull.vertices]
    distances = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        edge = end - start
        fraction = np.clip((center - start) @ edge / (edge @ edge), 0, 1)
        distances.append(np.linalg.norm(start + fraction * edge - center))
    inside = bool((hull.equations[:, :2] @ center + hull.equations[:, 2] <= 1e-12).all())
    d_min = 0.0 if inside else min(distances)
    d_max = np.linalg.norm(vertices - center, axis=1).max()
    return max(0.0, max(radius - CORRIDOR_HALF_WIDTH, POLE_RADIUS) - d_min), max(0.0, d_max - radius - CORRIDOR_HALF_WIDTH)


class TestSlalomAnnulus(unittest.TestCase):
    # 边缘穿入内孔时，即使四个角点均在孔外也必须处罚。
    def test_edge_intrusion(self):
        vertices = rectangle((0, 0)).unsqueeze(0)
        inner, outer = compute_annulus_side_excess(vertices, torch.zeros(1, 2), torch.zeros(1), torch.tensor([25.0]))
        self.assertGreater(float(torch.linalg.vector_norm(vertices[0] - torch.tensor([0, 0.04]), dim=-1).min()), POLE_RADIUS)
        self.assertAlmostEqual(float(inner[0]), 0.005, places=7)
        self.assertAlmostEqual(float(outer[0]), 0.005, places=7)

    # 圆心落入完整矩形时，最近距离必须为零。
    def test_center_inside(self):
        inner, _ = compute_annulus_side_excess(rectangle((0, 0.04)).unsqueeze(0),
            torch.zeros(1, 2), torch.zeros(1), torch.tensor([25.0]))
        self.assertAlmostEqual(float(inner[0]), POLE_RADIUS, places=7)

    # 圆弧切向姿态不再承受共同直线带的内侧误罚。
    def test_arc_tangent(self):
        for curvature in (15.0, 20.0):
            radius = 1.0 / curvature
            angle = curvature * 0.04
            center = (radius * np.sin(angle), radius * (1 - np.cos(angle)))
            inner, outer = compute_annulus_side_excess(rectangle(center, angle).unsqueeze(0),
                torch.zeros(1, 2), torch.zeros(1), torch.tensor([curvature]))
            self.assertAlmostEqual(float(inner[0]), 0.0, places=7)
            self.assertGreater(float(outer[0]), 0.0)

    # 左右镜像必须给出相同内外越界。
    def test_turn_mirror(self):
        vertices = rectangle((0.01, 0.005), 0.4).unsqueeze(0)
        positive = compute_annulus_side_excess(vertices, torch.zeros(1, 2), torch.zeros(1), torch.tensor([20.0]))
        mirrored = vertices.clone()
        mirrored[..., 1] *= -1
        negative = compute_annulus_side_excess(mirrored, torch.zeros(1, 2), torch.zeros(1), torch.tensor([-20.0]))
        for left, right in zip(positive, negative, strict=True):
            torch.testing.assert_close(left, right)

    # 路径与躯干整体平移旋转后，圆环度量保持一致。
    def test_rigid_transform(self):
        vertices = rectangle((0.01, 0.007), 0.3).unsqueeze(0)
        base = compute_annulus_side_excess(vertices, torch.zeros(1, 2), torch.zeros(1), torch.tensor([20.0]))
        angle = 1.2
        rotation = torch.tensor(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))), dtype=torch.float64)
        translation = torch.tensor([[2.0, -3.0]], dtype=torch.float64)
        changed = compute_annulus_side_excess(vertices @ rotation.T + translation[:, None],
            translation, torch.tensor([angle], dtype=torch.float64), torch.tensor([20.0]))
        for left, right in zip(base, changed, strict=True):
            torch.testing.assert_close(left, right)

    # 直行保持两侧对称，极小曲率单精度不会丢失毫米级越界。
    def test_straight_and_small_curvature(self):
        vertices = torch.stack((rectangle((0, 0.03)), rectangle((0, -0.03)))).float()
        reference, heading = torch.zeros(2, 2), torch.zeros(2)
        inner, outer = compute_annulus_side_excess(vertices, reference, heading, torch.zeros(2))
        torch.testing.assert_close(inner, torch.zeros(2))
        torch.testing.assert_close(outer, torch.full((2,), 0.025))
        for curvature in (2e-6, 1e-5, -2e-6):
            inner, outer = compute_annulus_side_excess(vertices, reference, heading, torch.full((2,), curvature))
            torch.testing.assert_close(torch.maximum(inner, outer), torch.full((2,), 0.025), atol=1e-6, rtol=0)

    # 随机三维旋转的投影盒与独立二维凸包计算一致。
    def test_random_projected_boxes(self):
        rng = np.random.default_rng(42)
        signs = np.asarray(tuple(product((-1.0, 1.0), repeat=3)))
        vertices, references, headings, curvatures, expected = [], [], [], [], []
        for _ in range(200):
            quaternion = rng.normal(size=4)
            quaternion /= np.linalg.norm(quaternion)
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, quaternion)
            points = ((signs * rng.uniform(0.01, 0.04, size=3)) @ rotation.reshape(3, 3).T)[:, :2]
            points += rng.uniform(-0.08, 0.08, size=2)
            reference = rng.uniform(-0.02, 0.02, size=2)
            heading = rng.uniform(-np.pi, np.pi)
            curvature = rng.uniform(-25, 25)
            expected.append(oracle(points, reference, heading, curvature))
            vertices.append(points)
            references.append(reference)
            headings.append(heading)
            curvatures.append(curvature)
        actual = compute_annulus_side_excess(torch.tensor(np.array(vertices), dtype=torch.float32),
            torch.tensor(np.array(references), dtype=torch.float32), torch.tensor(headings), torch.tensor(curvatures))
        torch.testing.assert_close(torch.stack(actual, dim=1), torch.tensor(expected, dtype=torch.float32), atol=2e-7, rtol=1e-5)

    # 网格解析使用全局编号，平放躯干可放入当前最大曲率圆环。
    def test_model_bounds_and_high_curvature_fit(self):
        model = get_spec().compile()
        ids = [model.body(name).id for name in BODY_NAMES]
        robot = NS(find_bodies=lambda names, **kwargs: ([0, 1], names), indexing=NS(body_ids=torch.tensor(ids)))
        bounds = resolve_body_envelope_bounds(robot, model)
        half = (bounds[:, 1] - bounds[:, 0]) / 2
        for size in half:
            width, length = size[0], size[2]
            center = (0.0, 1 / CURVATURE_TARGET_MAX - (POLE_RADIUS + width))
            inner, outer = compute_annulus_side_excess(rectangle(center, length=length, width=width).unsqueeze(0),
                torch.zeros(1, 2), torch.zeros(1), torch.tensor([CURVATURE_TARGET_MAX]))
            self.assertLess(float(inner[0]), 1e-7)
            self.assertLess(float(outer[0]), 1e-7)

    # 真实网格的世界坐标投影必须完全包含在姿态变换后的包围盒中。
    def test_world_projection_contains_mesh(self):
        model = get_spec().compile()
        data = mujoco.MjData(model)
        data.qpos[3:7] = np.array((0.8, 0.2, -0.3, 0.45)) / np.linalg.norm((0.8, 0.2, -0.3, 0.45))
        mujoco.mj_forward(model, data)
        ids = [model.body(name).id for name in BODY_NAMES]
        robot = NS(find_bodies=lambda names, **kwargs: ([0, 1], names), indexing=NS(body_ids=torch.tensor(ids)),
                   data=NS(body_link_pos_w=torch.tensor(data.xpos[ids])[None].float(),
                           body_link_quat_w=torch.tensor(data.xquat[ids])[None].float()))
        env = NS(scene={"robot": robot}, device="cpu", sim=NS(mj_model=model))
        with patch.object(reference, "_MODEL_INDICES", NS(f_body_id=0, h_body_id=1)):
            boxes = project_body_envelopes(env)[0].numpy()  # type: ignore
        for box, body_id in zip(boxes, ids, strict=True):
            hull = ConvexHull(box)
            for geom_id in np.flatnonzero(model.geom_bodyid == body_id):
                mesh_id = model.geom_dataid[geom_id]
                start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
                vertices = model.mesh_vert[start:start + count] @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
                violation = vertices[:, :2] @ hull.equations[:, :2].T + hull.equations[:, 2]
                self.assertLess(float(violation.max()), 1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
