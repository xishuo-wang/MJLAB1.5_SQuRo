# 验证名称索引、一阶段训练课程、Hole 参数对齐及内外侧包络奖励。
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from mjlab.scripts.SQuRo_Slalom_play import read_saved_config
from mjlab.tasks.SQuRo_Hole.config.rl_cfg import SQuRo_Hole_PPO_Runner_Cfg
from mjlab.tasks.SQuRo_Hole.SQuRo_Hole_env_cfg import SQuRo_Hole_Env_Cfg
from mjlab.tasks.SQuRo_Slalom.config.rl_cfg import SQuRo_Slalom_PPO_Runner_Cfg
from mjlab.tasks.SQuRo_Slalom.SQuRo_Slalom_env_cfg import SQuRo_Slalom_Env_Cfg
from mjlab.tasks.SQuRo_Slalom.mdp import events, indices
from mjlab.tasks.SQuRo_Slalom.mdp.command import SlalomCommand
from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import (
    CURVATURE_TARGET, CURVATURE_TARGET_MAX, Rmin,
    _STEPS_PER_ITER, get_curriculum_value, get_phase0_curvature_range,
    reward_weight_curriculum, SpacingCurriculumState,
)
from mjlab.tasks.SQuRo_Slalom.mdp.rewards import compute_phase1_corridor_side_excess


# 用路径局部坐标生成带符号的身体横向偏移。
def side_excess(offsets, kappas, heading=0.0, body_delta=0.0):
    heading_t = torch.full((len(offsets),), heading)
    normal = torch.stack((-heading_t.sin(), heading_t.cos()), dim=1)
    xy = normal * torch.tensor(offsets).unsqueeze(1)
    return compute_phase1_corridor_side_excess(
        xy, heading_t + body_delta, torch.zeros_like(xy), heading_t,
        0.04, 0.035, torch.tensor(kappas),
    )


class TestSlalomTrainingSetup(unittest.TestCase):
    # 训练从五至十起步：最低档固定 30%，其余档按考核缺口加权。
    def test_integer_slalom_sampling(self):
        torch.manual_seed(42)
        cmd = object.__new__(SlalomCommand)
        cmd._env = NS(device=torch.device("cpu"))
        cmd.cfg = SQuRo_Slalom_Env_Cfg().commands["slalom_cmd"]
        cmd.spacing_curriculum = SpacingCurriculumState()
        cmd.adaptive_sampling = True
        peaks = cmd._sample_slalom_indices(4096) + 5
        self.assertEqual(set(peaks.tolist()), set(range(5, 11)))
        self.assertAlmostEqual(float((peaks == 10).float().mean()), 0.30 + 0.70 / 6, delta=0.03)
        self.assertAlmostEqual(float((peaks == 5).float().mean()), 0.70 / 6, delta=0.03)
        # 考核缺口加权后该档采样概率升高
        cmd.spacing_curriculum.bin_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 3.0]
        peaks = cmd._sample_slalom_indices(8192) + 5
        self.assertAlmostEqual(float((peaks == 10).float().mean()), 0.30 + 0.70 * 3 / 8, delta=0.02)
        # 最终档: 考核环境 adaptive_sampling=False 时十一档等概率
        cmd.spacing_curriculum.final_level_passed = True
        cmd.adaptive_sampling = False
        peaks = cmd._sample_slalom_indices(8192) + 5
        self.assertEqual(set(peaks.tolist()), set(range(5, 16)))
        counts = [(peaks == peak).float().mean().item() for peak in range(5, 16)]
        self.assertLess(max(counts) - min(counts), 0.02)
        # 关闭课程后回放覆盖全部整数条件
        cmd.cfg.enable_spacing_curriculum = False
        peaks = cmd._sample_slalom_indices(4096) + 5
        self.assertEqual(set(peaks.tolist()), set(range(5, 16)))
        cmd.cfg.fixed_slalom_curvature = 12
        self.assertEqual(cmd._sample_slalom_indices(3).tolist(), [7, 7, 7])
        cmd.cfg.fixed_obstacle_spacing = 0.2
        with self.assertRaises(ValueError):
            cmd._sample_slalom_indices(1)

    # 曲率速度曲线保留端点、左右对称，并满足批准的十五曲率速度。
    def test_velocity_curve(self):
        from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import get_velocity_scale
        values = get_velocity_scale(torch.tensor([0.0, 5.0, 10.0, 15.0, 20.0]))
        self.assertTrue(bool((values[:-1] > values[1:]).all()))
        self.assertAlmostEqual(get_velocity_scale(0.0), 1.0)
        self.assertAlmostEqual(get_velocity_scale(20.0), 0.15)
        self.assertAlmostEqual(0.1 * 2 * get_velocity_scale(15.0), 0.05125)
        torch.testing.assert_close(get_velocity_scale(torch.tensor([-15.0])), values[3:4])

    # PPO 算法、网络、采样步数和动作裁剪与 Hole 对拍。
    def test_ppo_matches_hole(self):
        hole = SQuRo_Hole_PPO_Runner_Cfg()
        slalom = SQuRo_Slalom_PPO_Runner_Cfg()
        for name in ("algorithm", "actor", "critic"):
            self.assertEqual(asdict(getattr(hole, name)), asdict(getattr(slalom, name)))
        self.assertEqual(slalom.num_steps_per_env, hole.num_steps_per_env)
        self.assertEqual(slalom.num_steps_per_env, _STEPS_PER_ITER)
        self.assertEqual(slalom.clip_actions, hole.clip_actions)

    # 控制周期与物理步长均对齐 Hole，训练允许自动进入绕杆。
    def test_policy_frequency_matches_hole(self):
        hole = SQuRo_Hole_Env_Cfg()
        slalom = SQuRo_Slalom_Env_Cfg()
        self.assertEqual(slalom.sim.mujoco.timestep, hole.sim.mujoco.timestep)
        self.assertEqual(slalom.decimation, hole.decimation)
        self.assertAlmostEqual(slalom.decimation * slalom.sim.mujoco.timestep, 0.01)
        self.assertFalse(slalom.commands["slalom_cmd"].phase0_only)  # type: ignore

    # 跨过原绕杆边界后，路径与奖励仍使用一阶段。
    def test_phase0_lock_keeps_late_curriculum(self):
        cmd = object.__new__(SlalomCommand)
        cmd.cfg = SQuRo_Slalom_Env_Cfg().commands["slalom_cmd"]
        cmd.cfg.phase0_only = True
        env = NS(command_manager=NS(_terms={"slalom_cmd": cmd}))
        cmd._env = env
        for iteration in (2000, 3999, 4000, 6000, 8000):
            env.common_step_counter = iteration * _STEPS_PER_ITER
            env.episode_length_buf = torch.tensor([0, 17])
            self.assertFalse(cmd.slalom_mode_active)
            self.assertFalse(bool(cmd.phase1_mask.any()))
            self.assertEqual(get_curriculum_value(env, "sigma_spn_pos"), 10.0)
            self.assertEqual(get_curriculum_value(env, "weight_corridor"), 5.0)

    # 一阶段保持二十归一化，绕杆峰值上限独立为十五。
    def test_curvature_range_reaches_20(self):
        self.assertEqual(CURVATURE_TARGET_MAX, 20.0)
        self.assertEqual(CURVATURE_TARGET, 15.0)
        self.assertAlmostEqual(Rmin, 1 / 15)
        for iteration in (2000, 2999, 4000):
            self.assertEqual(get_phase0_curvature_range(iteration * _STEPS_PER_ITER), (-20.0, 20.0))
        self.assertLess(get_phase0_curvature_range(0)[1], 15.0)

    # 左右转内侧互换，同时验证外侧不受内侧系数影响。
    def test_turn_side_and_reward_strength(self):
        inner, outer = side_excess([0.03, -0.03, -0.03, 0.03], [25.0, 25.0, -25.0, -25.0])
        torch.testing.assert_close(inner, torch.tensor([0.025, 0.0, 0.025, 0.0]))
        torch.testing.assert_close(outer, torch.tensor([0.0, 0.025, 0.0, 0.025]))
        weights = reward_weight_curriculum.get_curriculum_values(2000 * _STEPS_PER_ITER, True)
        reward = torch.exp(-weights["sigma_corridor"] * (
            outer.square() + weights["corridor_inner_multiplier"] * inner.square()
        ))
        self.assertLess(float(reward[0]), float(reward[1]))
        torch.testing.assert_close(reward[:2], reward[2:])

    # 直行两侧相同，中心身体没有越界。
    def test_straight_is_symmetric(self):
        inner, outer = side_excess([0.03, -0.03, 0.0], [0.0, 0.0, 25.0])
        torch.testing.assert_close(inner, torch.zeros(3))
        torch.testing.assert_close(outer, torch.tensor([0.025, 0.025, 0.0]))

    # 同时旋转路径和身体，横向包络度量不变。
    def test_corridor_is_rotation_invariant(self):
        expected = side_excess([0.02, -0.02], [20.0, -20.0], body_delta=0.7)
        actual = side_excess([0.02, -0.02], [20.0, -20.0], heading=1.3, body_delta=0.7)
        for x, y in zip(actual, expected, strict=True):
            torch.testing.assert_close(x, y)

    # 误差尺度平滑收紧，课程边界后保持 20 毫米。
    def test_corridor_scale_tightens_continuously(self):
        sigma = [reward_weight_curriculum.get_curriculum_values(i * _STEPS_PER_ITER, True)["sigma_corridor"]
                 for i in (0, 1, 1000, 1999, 2000, 4000, 8000)]
        self.assertEqual(sigma, sorted(sigma))
        self.assertAlmostEqual(sigma[0] ** -0.5, 0.05)
        self.assertAlmostEqual(sigma[-1] ** -0.5, 0.02)
        self.assertLess(sigma[4] - sigma[3], 2.0)

    # 打乱模型关节顺序后，复位仍绑定正确部位和颈部默认角。
    def test_named_reset_survives_joint_reordering(self):
        names = ("unused_joint",) + tuple(reversed(indices._ACT_JOINT_NAMES + indices.LINKAGE_JOINT_NAMES))
        joint_map = {name: i for i, name in enumerate(names)}
        recorded = {}
        robot = NS(
            num_joints=len(names),
            find_joints=lambda requested, **kw: ([joint_map[name] for name in requested], requested),
            find_bodies=lambda requested, **kw: ([3, 1], requested),
            find_sites=lambda requested, **kw: ([7, 2, 9, 4], requested),
            write_root_state_to_sim=lambda *a, **kw: None,
            write_joint_state_to_sim=lambda pos, vel, **kw: recorded.update(pos=pos, vel=vel),
        )
        env = NS(device="cpu", scene=NS(entities={"robot": robot}),
                 command_manager=NS(_terms={"slalom_cmd": NS(slalom_mode_active=False)}))
        model_indices = indices.ModelIndices()
        with patch.object(indices, "_MODEL_INDICES", model_indices), patch.object(events, "_MODEL_INDICES", model_indices):
            events.reset_model(env, torch.tensor([0, 1]))
        for name, expected in (("Neck_pitch_joint", -0.3), ("FL_shoulder_joint", 0.1),
                               ("HL_hip_joint", -0.1), ("FL_elbow1_joint", 0.3867)):
            torch.testing.assert_close(recorded["pos"][:, joint_map[name]], torch.full((2,), expected))
        self.assertEqual(model_indices.joint_ids, tuple(joint_map[name] for name in indices._ACT_JOINT_NAMES))

    # 足端力列按传感器名称重排，拒绝依赖自然列序。
    def test_foot_force_order_uses_names(self):
        sensor = NS(primary_names=list(reversed(indices.FOOT_GEOM_NAMES)))
        self.assertEqual(indices.resolve_foot_force_ids(sensor), (3, 2, 1, 0))

    # 保存配置含 Python 标签时只读取字符串字段。
    def test_saved_config_reads_without_constructing_objects(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "params").mkdir()
            (root / "params" / "env.yaml").write_text(
                "commands:\n  slalom_cmd:\n    phase0_only: true\nextra: !!python/tuple [1, 2]\n", encoding="utf-8"
            )
            data = read_saved_config(root / "model_6000.pt", "env")
            self.assertEqual(data["commands"]["slalom_cmd"]["phase0_only"], "true")
            self.assertEqual(read_saved_config(root / "model_6000.pt", "agent"), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
