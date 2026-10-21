# 验证四脊柱参考的时序一致性与针对异常折腿的软惩罚。
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from mjlab.tasks.SQuRo_Slalom.mdp import reference, rewards
from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import (
    PHASE1_END_ITER, SPINE_REFERENCE_TIME_CONSTANT, STEPS_PER_ITER,
)
from mjlab.tasks.SQuRo_Slalom.mdp.indices import (
    _ACT_JOINT_NAMES, _ACT_SPN_IDS, _ACT_LEG_IDS, _ACT_FL_SHOULDER_ID,
    _ACT_FR_SHOULDER_ID, _ACT_HL_HIP_ID, _ACT_HR_HIP_ID,
)
from mjlab.scripts.Slalom.verify_slalom_ref_consistency import make_ref_state_env


# 构造二阶段参考桩并让曲率取测试序列。
def make_env():
    env = make_ref_state_env([15.0, -15.0], gait=0.0)
    env.step_dt = 0.01
    env.common_step_counter = PHASE1_END_ITER * STEPS_PER_ITER + 100
    return env


class TestSpineReferenceDynamics(unittest.TestCase):
    # 固定路径读取，隔离关节参考的状态更新。
    def setUp(self):
        self.patches = (
            patch.object(reference, "compute_path_ref", lambda env: None),
            patch.object(reference, "get_path_curvature",
                         lambda env: env.command_manager._terms["slalom_cmd"].command[:, 4]),
        )
        for item in self.patches:
            item.start()

    # 恢复生产函数。
    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()

    # 所有四个脊柱关节的速度都来自实际平滑位置差分，腿颈保持原参考。
    def test_position_velocity_consistent_through_turn(self):
        env = make_env()
        old, velocity = reference.get_reference_joint_state(env)
        torch.testing.assert_close(velocity[:, _ACT_SPN_IDS], torch.zeros(2, 4))
        for curvature in (10.0, 2.0, 0.0, -2.0, -10.0, -15.0):
            env.common_step_counter += 1
            env.episode_length_buf += 1
            env.command_manager._terms["slalom_cmd"].command[:, 4] = torch.tensor([curvature, -curvature])
            with patch.object(reference, "_smooth_spine_reference", lambda *args: None):
                raw_env = make_env()
                raw_env.command_manager._terms["slalom_cmd"].command[:, 4] = torch.tensor([curvature, -curvature])
                raw, _ = reference.get_reference_joint_state(raw_env)
            position, velocity = reference.get_reference_joint_state(env)
            alpha = env.step_dt / (SPINE_REFERENCE_TIME_CONSTANT + env.step_dt)
            torch.testing.assert_close(position[:, _ACT_SPN_IDS],
                                       old[:, _ACT_SPN_IDS] + alpha * (raw[:, _ACT_SPN_IDS] - old[:, _ACT_SPN_IDS]))
            torch.testing.assert_close(velocity[:, _ACT_SPN_IDS],
                                       (position[:, _ACT_SPN_IDS] - old[:, _ACT_SPN_IDS]) / env.step_dt)
            untouched = tuple(i for i in range(len(_ACT_JOINT_NAMES)) if i not in _ACT_SPN_IDS)
            torch.testing.assert_close(position[:, untouched], raw[:, untouched])
            repeated, repeated_vel = reference.get_reference_joint_state(env)
            torch.testing.assert_close(repeated, position)
            torch.testing.assert_close(repeated_vel, velocity)
            old = position.clone()

    # 奖励后局部重置再计算观测时，其他环境不得多走一次滤波。
    def test_partial_reset_and_same_step_recompute(self):
        env = make_env()
        reference.get_reference_joint_state(env)
        env.common_step_counter += 1
        env.episode_length_buf += 1
        env.command_manager._terms["slalom_cmd"].command[:, 4] *= -1
        old, old_velocity = reference.get_reference_joint_state(env)
        old, old_velocity = old.clone(), old_velocity.clone()
        env.episode_length_buf[0] = 0
        env.command_manager._terms["slalom_cmd"].command[0, 4] = 0
        reference.invalidate_reference_cache(env, torch.tensor([0]))
        position, velocity = reference.get_reference_joint_state(env)
        torch.testing.assert_close(position[1, _ACT_SPN_IDS], old[1, _ACT_SPN_IDS])
        torch.testing.assert_close(velocity[1, _ACT_SPN_IDS], old_velocity[1, _ACT_SPN_IDS])
        torch.testing.assert_close(velocity[0, _ACT_SPN_IDS], torch.zeros(4))
        self.assertAlmostEqual(float(position[0, _ACT_SPN_IDS[1]]), 0.0)
        reference.invalidate_reference_cache(env)
        repeated, _ = reference.get_reference_joint_state(env)
        torch.testing.assert_close(repeated[:, _ACT_SPN_IDS], position[:, _ACT_SPN_IDS])

    # 长时间恒定输入最终回到原角度；新回合首帧直接初始化而非继承旧滤波状态。
    def test_constant_target_and_full_reset(self):
        env = make_env()
        reference.get_reference_joint_state(env)
        env.command_manager._terms["slalom_cmd"].command[:, 4] = 20.0
        for _ in range(100):
            env.common_step_counter += 1
            env.episode_length_buf += 1
            position, _ = reference.get_reference_joint_state(env)
        expected = torch.tensor([[-0.6, -0.9, -0.6, -0.7]]).expand(2, -1)
        torch.testing.assert_close(position[:, _ACT_SPN_IDS], expected)
        env.episode_length_buf.zero_()
        env.command_manager._terms["slalom_cmd"].command[:, 4] = -20.0
        reference.invalidate_reference_cache(env, torch.arange(2))
        position, velocity = reference.get_reference_joint_state(env)
        expected = torch.tensor([[0.6, 0.9, -0.6, 0.7]]).expand(2, -1)
        torch.testing.assert_close(position[:, _ACT_SPN_IDS], expected)
        torch.testing.assert_close(velocity[:, _ACT_SPN_IDS], torch.zeros(2, 4))


class TestAbnormalLegPose(unittest.TestCase):
    # 用置乱动作列和非零编码偏置检查物理请求目标的读取语义。
    def setUp(self):
        count = len(_ACT_JOINT_NAMES)
        self.reference = torch.zeros(2, count)
        self.actual = self.reference.clone()
        self.offset = torch.linspace(-0.2, 0.1, count).expand(2, -1)
        self.bias = torch.full_like(self.offset, 0.02)
        self.raw = torch.zeros_like(self.offset)
        self.order = tuple(reversed(range(count)))
        term = NS(raw_action=self.raw, scale=0.3, offset=(self.offset + self.bias)[:, self.order],
                  target_ids=torch.tensor(self.order), target_names=[_ACT_JOINT_NAMES[i] for i in self.order])
        command = NS(phase1_mask=torch.tensor([True, True]), cfg=NS(phase0_only=False))
        self.env = NS(common_step_counter=PHASE1_END_ITER * STEPS_PER_ITER,
                      scene={"robot": NS(data=NS(joint_pos=self.actual, encoder_bias=self.bias))},
                      action_manager=NS(get_term=lambda name: term),
                      command_manager=NS(_terms={"slalom_cmd": command}), extras={"log": {}})
        self.patches = (
            patch.object(rewards, "get_reference_joint_state", lambda env: (self.reference, self.reference)),
            patch.object(rewards._MODEL_INDICES, "joint_ids", tuple(range(count))),
        )
        for item in self.patches:
            item.start()
        self.set_requested(self.reference)

    # 按动作项自身比例和偏置写入指定物理角度。
    def set_requested(self, target):
        self.raw[:] = ((target - self.offset) / 0.3)[:, self.order]

    # 恢复共享索引与参考函数。
    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()

    # 当前异常姿态有代价，参考附近和脊柱自由运动不受此项惩罚。
    def test_actual_fold_and_shoulder_lift(self):
        self.assertAlmostEqual(float(rewards.compute_abnormal_leg_pose_penalty(self.env).sum()), 0.0)  # type: ignore
        self.actual[:, _ACT_SPN_IDS] = 2.0
        self.assertAlmostEqual(float(rewards.compute_abnormal_leg_pose_penalty(self.env).sum()), 0.0)  # type: ignore
        self.actual[0, _ACT_HL_HIP_ID] = -1.5
        self.actual[1, _ACT_FL_SHOULDER_ID] = 1.1
        penalty = rewards.compute_abnormal_leg_pose_penalty(self.env)  # type: ignore
        torch.testing.assert_close(penalty, torch.tensor([-5 * 0.8**2, -5 * 0.4**2]))

    # 深度超出 XML 范围的请求仍有更大代价，不按十四维平均稀释单腿异常。
    def test_request_cost_increases_before_control_limits(self):
        target = self.reference.clone()
        target[0, _ACT_HL_HIP_ID] = -1.5
        target[1, _ACT_HR_HIP_ID] = -2.0
        self.set_requested(target)
        penalty = rewards.compute_abnormal_leg_pose_penalty(self.env)  # type: ignore
        torch.testing.assert_close(penalty, torch.tensor([-5 * 0.8**2, -5 * 1.3**2]))
        self.assertTrue(bool((penalty < 0).all()))

    # 左右同类关节规则相同，只在绕杆回合启用。
    def test_symmetry_phase_gate_and_tolerance(self):
        self.actual[0, _ACT_FL_SHOULDER_ID] = 1.2
        self.actual[1, _ACT_FR_SHOULDER_ID] = 1.2
        penalty = rewards.compute_abnormal_leg_pose_penalty(self.env)  # type: ignore
        torch.testing.assert_close(penalty[0], penalty[1])
        self.env.command_manager._terms["slalom_cmd"].phase1_mask[0] = False
        self.assertEqual(float(rewards.compute_abnormal_leg_pose_penalty(self.env)[0]), 0.0)  # type: ignore
        self.env.command_manager._terms["slalom_cmd"].cfg.phase0_only = True
        self.assertEqual(float(rewards.compute_abnormal_leg_pose_penalty(self.env).sum()), 0.0)  # type: ignore


if __name__ == "__main__":
    unittest.main(verbosity=2)
