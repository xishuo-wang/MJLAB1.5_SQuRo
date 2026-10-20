# 用原始高密度路径独立验证二阶段曲线距离、盒边越界及阶段隔离。
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest

import numpy as np
import torch

from mjlab.tasks.SQuRo_Slalom.mdp import reference, rewards
from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import SLALOM_CURVATURES
from mjlab.tasks.SQuRo_Slalom.mdp.reference import get_slalom_geometry, get_spatial_slalom_tables
from mjlab.tasks.SQuRo_Slalom.mdp.rewards import compute_phase1_corridor_side_excess


# 用原始积分节点逐线段穷举最近距离，并包含接近段起点的向后射线。
def oracle(points, curvature):
    geometry = get_slalom_geometry(curvature)
    approach, period = geometry["approach"], geometry["period"]
    pieces = [np.column_stack((approach["x"], approach["y"]))]
    for cycle in range(2):
        pieces.append(np.column_stack((period["x"] + (1 + 2 * cycle) * geometry["spacing"], period["y"]))[1:])
    path = np.concatenate(pieces)
    starts, edges = path[:-1], np.diff(path, axis=0)
    delta = points[:, None] - starts
    fraction = np.clip(np.sum(delta * edges, axis=-1) / np.sum(edges * edges, axis=-1), 0, 1)
    distance = np.linalg.norm(delta - fraction[..., None] * edges, axis=-1).min(axis=-1)
    heading = approach["heading"][0]
    tangent = np.array((np.cos(heading), np.sin(heading)))
    delta = points - path[0]
    ray = np.linalg.norm(delta - np.minimum(delta @ tangent, 0)[:, None] * tangent, axis=-1)
    return np.minimum(distance, ray)


# 用三维盒的编号顺序构造水平矩形投影。
def rectangle(center, heading, length=0.03, width=0.02):
    points = torch.tensor(reference._BOX_SIGNS, dtype=torch.float64)[:, :2]
    rotation = torch.tensor(((np.cos(heading), -np.sin(heading)),
                             (np.sin(heading), np.cos(heading))), dtype=torch.float64)
    return points * torch.tensor((length, width)) @ rotation.T + torch.as_tensor(center)


class TestSlalomCurveCorridor(unittest.TestCase):
    # 分块、尾块补齐及跨块曲率混排必须与原几何查询一致。
    def test_block_projection_preserves_geometry(self):
        tables = torch.tensor(get_spatial_slalom_tables(), dtype=torch.float64)
        rows = torch.arange(129) % len(tables)
        spacing = tables[rows, -1, 0] / 5
        generator = torch.Generator().manual_seed(42)
        points = torch.rand(129, 11, 2, generator=generator, dtype=torch.float64) * 0.2
        expected = rewards._project_phase1_path_points_block(points, tables, rows, spacing)
        for count in (0, 1, 63, 64, 65, 129):
            with patch.object(rewards, "_project_phase1_path_points_block", wraps=rewards._project_phase1_path_points_block) as project:
                actual = rewards.project_phase1_path_points(points[:count], tables, rows[:count], spacing[:count])
            for call in project.call_args_list:
                size = call.args[0].shape[0]
                self.assertLessEqual(size, rewards._PROJECTION_ENV_BLOCK)
                self.assertEqual(size & (size - 1), 0)
            for value, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(value, reference[:count])

    # 正负曲率的内外侧分别随最近轨迹点判定，不沿用基座所在处的转向。
    def test_local_turn_side(self):
        geometry = get_slalom_geometry(15)
        path = geometry["period"]
        for arc in (0.04, path["arc"][-1] / 2 + 0.04):
            center = np.array((np.interp(arc, path["arc"], path["x"]) + geometry["spacing"],
                               np.interp(arc, path["arc"], path["y"])))
            heading = np.interp(arc, path["arc"], path["heading"])
            sign = np.sign(np.interp(arc, path["arc"], path["kappa"]))
            normal = np.array((-np.sin(heading), np.cos(heading)))
            points = torch.tensor(np.stack((center + sign * 0.045 * normal, center - sign * 0.045 * normal)))
            _, lateral, curvature = rewards.project_phase1_path_points(points[None], torch.tensor(get_spatial_slalom_tables()),
                torch.tensor([len(SLALOM_CURVATURES) - 1]), torch.tensor([geometry["spacing"]]))
            self.assertGreater(float(lateral[0, 0] * curvature[0, 0]), 0)
            self.assertLess(float(lateral[0, 1] * curvature[0, 1]), 0)

    # 全部整数曲率及换向邻域的距离与原始高密度线段穷举对照。
    def test_dense_geometry_oracle(self):
        tables = torch.tensor(get_spatial_slalom_tables(), dtype=torch.float64)
        rng = np.random.default_rng(42)
        for row, curvature in enumerate(SLALOM_CURVATURES):
            geometry = get_slalom_geometry(curvature)
            spacing = geometry["spacing"]
            points = np.column_stack((rng.uniform(geometry["approach"]["x"][0] - 0.08, 4 * spacing, 120),
                                      rng.uniform(geometry["period"]["y"].min() - 0.06, 0.06, 120)))
            actual = rewards.project_phase1_path_points(torch.tensor(points)[None], tables,
                torch.tensor([row]), torch.tensor([spacing]))[0][0].numpy()
            np.testing.assert_allclose(actual, oracle(points, curvature), atol=0.0001, rtol=0)

    # 圆弧内沿各自切向的躯干不再被同一条切线误罚。
    def test_arc_tangent_avoids_old_strip_penalty(self):
        geometry = get_slalom_geometry(15)
        path = geometry["period"]
        center = np.array((np.interp(0.04, path["arc"], path["x"]) + geometry["spacing"],
                           np.interp(0.04, path["arc"], path["y"])))
        heading = np.interp(0.04, path["arc"], path["heading"])
        vertices = rectangle(center, heading)
        distance = rewards.project_phase1_path_points(vertices[None], torch.tensor(get_spatial_slalom_tables()),
                    torch.tensor([len(SLALOM_CURVATURES) - 1]), torch.tensor([geometry["spacing"]]))[0]
        self.assertLess(float(distance.max()), 0.04)
        old = compute_phase1_corridor_side_excess(torch.tensor(center)[None], torch.tensor([heading]),
            torch.tensor([[geometry["spacing"], 0.0]]), torch.zeros(1), 0.03, 0.02, torch.tensor([-15.0]))
        self.assertGreater(float(torch.maximum(*old)[0]), 0.001)

    # 盒边中点侵入弯道内侧时，即使角点无内侧越界也能检出。
    def test_edge_sampling_and_origin_isolation(self):
        geometry = get_slalom_geometry(15)
        path = geometry["period"]
        center = (np.interp(0.05, path["arc"], path["x"]) + geometry["spacing"],
                  np.interp(0.05, path["arc"], path["y"]))
        heading = np.interp(0.05, path["arc"], path["heading"])
        vertices = rectangle(center, heading, length=0.04, width=0.045)[None, None].expand(2, 2, -1, -1).clone()
        origins = torch.tensor([[0.0, 0.0, 0.0], [3.0, -2.0, 0.0]], dtype=torch.float64)
        vertices += origins[:, None, None, :2]
        cmd = NS(slalom_curvature_index=torch.full((2,), len(SLALOM_CURVATURES) - 1),
                 obstacle_spacing=torch.full((2,), geometry["spacing"]))
        env = NS(device="cpu", scene=NS(env_origins=origins), command_manager=NS(_terms={"slalom_cmd": cmd}),
                 _slalom_body_envelope_vertices=rectangle((0.0, 0.0), 0.0, length=0.04, width=0.045)[None].expand(2, -1, -1))
        with patch.object(rewards, "project_body_envelopes", return_value=vertices):
            inner, outer = rewards.compute_phase1_corridor_excess(env, torch.arange(2))  # type: ignore
        self.assertGreater(float(inner.min()), 0.004)
        torch.testing.assert_close(inner[0], inner[1], atol=1e-7, rtol=0)
        torch.testing.assert_close(outer[0], outer[1], atol=1e-7, rtol=0)

    # 后躯干位于接近段起点之后方时不承受有限路径端盖误罚，后续周期无接缝跳变。
    def test_approach_ray_and_period_seam(self):
        for row, curvature in enumerate(SLALOM_CURVATURES):
            geometry = get_slalom_geometry(curvature)
            approach = geometry["approach"]
            tangent = np.array((np.cos(approach["heading"][0]), np.sin(approach["heading"][0])))
            point = np.array((approach["x"][0], approach["y"][0])) - 0.08 * tangent
            query = np.array([point, [4 * geometry["spacing"] - 1e-7, geometry["obstacle_y"]],
                              [4 * geometry["spacing"] + 1e-7, geometry["obstacle_y"]]])
            distance = rewards.project_phase1_path_points(torch.tensor(query)[None], torch.tensor(get_spatial_slalom_tables()),
                torch.tensor([row]), torch.tensor([geometry["spacing"]]))[0][0]
            self.assertLess(float(distance[0]), 1e-7)
            self.assertLess(float(abs(distance[1] - distance[2])), 1e-6)

    # 同一批环境跨越阶段边界时，仅替换已经进入绕杆的环境。
    def test_mixed_phase_reward_dispatch(self):
        env = NS(command_manager=NS(_terms={"slalom_cmd": NS(phase1_mask=torch.tensor([False, True]))}), extras={"log": {}})
        zeros = torch.zeros(2)
        with patch.object(rewards, "compute_path_ref", return_value=(zeros,) * 5), \
             patch.object(rewards, "get_path_curvature", return_value=zeros), \
             patch.object(rewards, "get_curriculum_value", return_value=1.0), \
             patch.object(rewards, "compute_phase0_corridor_excess", return_value=(torch.zeros(2, 2), torch.zeros(2, 2))), \
             patch.object(rewards, "compute_phase1_corridor_excess", return_value=(torch.full((1, 2), 0.02), torch.zeros(1, 2))) as phase1:
            value = rewards.compute_corridor_reward(env)  # type: ignore
        torch.testing.assert_close(phase1.call_args.args[1], torch.tensor([1]))
        self.assertEqual(float(value[0]), 1.0)
        self.assertLess(float(value[1]), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
