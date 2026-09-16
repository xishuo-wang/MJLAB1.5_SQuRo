# 阶段奖励、参考边界与站立判定回归；不创建环境、不修改训练数据。
import unittest
from math import cos, radians
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from mjlab.tasks.SQuRo_Backup.mdp.command import BackupCommand, BackupCommandCfg
from mjlab.tasks.SQuRo_Backup.mdp import reference, rewards, terminations, events
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES, _ACTUATED_JOINT_NAMES
from mjlab.tasks.SQuRo_Backup.mdp.curriculums import _CURVES


# 构造只含判定所需数据的批量环境，直接调用生产函数。
def make_env(phases):
    n = len(phases)
    cmd = object.__new__(BackupCommand)
    env = NS(device='cpu', num_envs=n, common_step_counter=0, step_dt=0.01,
             episode_length_buf=torch.ones(n, dtype=torch.long), extras={'log': {}})
    cmd._env = env
    cmd.cfg = BackupCommandCfg()
    cmd.phase = torch.tensor(phases)
    cmd.t_phase = torch.zeros(n)
    cmd.retry = torch.zeros(n, dtype=torch.long)
    cmd.command_tensor = torch.zeros(n, 7)
    cmd.command_tensor[:, 5] = 3.0
    cmd.time_scale_command = cmd.command_tensor[:, 5]
    cmd.phase_command = cmd.command_tensor[:, 6]
    cmd._update_dt = env.step_dt
    cmd._pose_cos_threshold = cos(radians(45))
    cmd._pose_cache = None
    cmd._pose_cos_cache = None
    cmd.test_u = torch.ones(n, 2)
    cmd.test_heights = torch.full((n, 2), .055)
    cmd._pose_cos = lambda: cmd.test_u
    cmd._body_height = lambda idx: cmd.test_heights[:, 0 if idx == _MODEL_INDICES.f_body_id else 1]
    env.command_manager = NS(get_term=lambda _: cmd, _terms={'backup_cmd': cmd})
    return env, cmd


class StageRewardTests(unittest.TestCase):
    def setUp(self):
        self.ids = (_MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id)
        _MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id = 0, 1

    def tearDown(self):
        _MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id = self.ids

    def test_exclusive_stage_rewards(self):
        env, cmd = make_env([0, 1, 2])
        cmd.test_u[:] = torch.tensor([-1., 1.])
        torch.testing.assert_close(rewards.compute_s1_progress_reward(env), torch.tensor([3., 0., 0.]))
        self.assertEqual(rewards.compute_s2_progress_reward(env).sum(), 0.)
        self.assertEqual(rewards.compute_s3_progress_reward(env).sum(), 0.)
        cmd.test_u[:] = 1.
        self.assertEqual(rewards.compute_s1_progress_reward(env).sum(), 0.)
        torch.testing.assert_close(rewards.compute_s2_progress_reward(env), torch.tensor([0., 3., 0.]))
        torch.testing.assert_close(rewards.compute_s3_progress_reward(env), torch.tensor([0., 0., 3.]))

    def test_standing_requires_both_segments(self):
        env, cmd = make_env([2, 2, 2, 2])
        cmd.test_heights[0] = torch.tensor([.08, .03])
        cmd.test_heights[1] = .024
        cmd.test_u[2, 0] = float('nan')
        cmd.test_u[3, 0] = -1.
        standing, progress = cmd.standing_state()
        self.assertFalse(standing.any())
        self.assertTrue(torch.isfinite(progress).all())
        self.assertEqual(progress[1:].sum(), 0.)
        cmd.test_heights[:] = .055
        cmd.test_u[:] = 1.
        self.assertTrue(cmd.standing_state()[0].all())

    def test_stand_confirmation_and_phase_gate(self):
        env, cmd = make_env([2, 1])
        for _ in range(49):
            self.assertFalse(terminations.check_stand_success(env).any())
        torch.testing.assert_close(terminations.check_stand_success(env), torch.tensor([True, False]))
        cmd.test_u[0, 0] = -1.
        self.assertFalse(terminations.check_stand_success(env).any())
        self.assertEqual(env._stand_elapsed[0], 0.)
        cmd.test_u[:] = 1.
        env.step_dt = .03
        for _ in range(16):
            self.assertFalse(terminations.check_stand_success(env).any())
        self.assertTrue(terminations.check_stand_success(env)[0])

    def test_reset_clears_only_selected_stand_timer(self):
        env, _ = make_env([2, 2])
        env._stand_elapsed = torch.tensor([.4, .3])
        robot = NS(num_joints=36, write_root_state_to_sim=lambda *a, **kw: None,
                   write_joint_state_to_sim=lambda *a, **kw: None)
        env.scene = NS(entities={'robot': robot})
        with patch.object(events, 'resolve_model_indices'):
            events.reset_model(env, torch.tensor([0]))
        torch.testing.assert_close(env._stand_elapsed, torch.tensor([0., .3]))

    def test_p2_boundary_and_p3_zero(self):
        env, cmd = make_env([1, 1, 1, 2])
        cmd.t_phase[:] = torch.tensor([.43, .44, .75, 0.])
        pos, vel = reference.get_reference_joint_state(env)
        torch.testing.assert_close(pos[:, 0], torch.tensor([.5822222, .5911111, .6, 0.]), atol=2e-6, rtol=0.)
        torch.testing.assert_close(vel[:2, 0], torch.full((2,), .4/.45), atol=1e-5, rtol=0.)
        self.assertEqual(vel[2].abs().sum(), 0.)
        self.assertEqual(pos[3, [0, 1, 8, 9]].abs().sum(), 0.)
        self.assertEqual(vel[3, [0, 1, 8, 9]].abs().sum(), 0.)
        self.assertGreater(vel[3, 10], 0.)
        torch.testing.assert_close(pos[2, [1, 8, 9]], torch.zeros(3), atol=1e-6, rtol=0.)
        cmd.phase[:] = 0
        cmd.t_phase[:] = 3.
        pos, vel = reference.get_reference_joint_state(env)
        torch.testing.assert_close(pos[0, [0, 1, 8, 9]], torch.tensor([.2, -1.57, -.2, 1.57]))
        self.assertEqual(vel.abs().sum(), 0.)

    def test_p2_endpoint_across_time_scales(self):
        env, cmd = make_env([1, 1, 1, 1])
        cmd.time_scale_command[:] = torch.tensor([1., 2., 3., 4.])
        for fraction in [.25, .75, .99, 1., 1.5]:
            cmd.t_phase[:] = .15 * cmd.time_scale_command * fraction
            pos, vel = reference.get_reference_joint_state(env)
            u = min(fraction, 1.)
            expected = torch.tensor([.2+.4*u, -1.57+1.57*u, -.2+.2*u, 1.57-1.57*u])
            torch.testing.assert_close(pos[:, [0,1,8,9]], expected.expand(4,4), atol=3e-6, rtol=0.)
            if fraction >= 1.:
                self.assertEqual(vel.abs().sum(), 0.)
            else:
                self.assertTrue((vel[:,0]>0).all())

    def test_target_weights_and_name_alignment(self):
        env, cmd = make_env([0, 1, 2])
        # 动作名称反序，验证成本没有依赖动作列与参考列同序。
        action = NS(raw_action=torch.ones(3, 14), scale=1., offset=0.,
                    target_names=list(reversed(_ACTUATED_JOINT_NAMES)))
        env.action_manager = NS(get_term=lambda _: action)
        with patch.object(rewards, 'get_reference_joint_state', return_value=(torch.zeros(3,14), torch.zeros(3,14))):
            torch.testing.assert_close(rewards.compute_spine_target_cost(env), torch.full((3,), -2.))
            torch.testing.assert_close(rewards.compute_leg_target_cost(env), torch.tensor([0., 0., -1.]))
            with patch.dict(_CURVES, weight_spine_target=(4.,)):
                torch.testing.assert_close(rewards.compute_spine_target_cost(env), torch.full((3,), -4.))
            action.raw_action.zero_()
            action.raw_action[:, action.target_names.index('F_body_joint')] = 2.
            torch.testing.assert_close(rewards.compute_spine_target_cost(env), torch.full((3,), -2.))
            self.assertEqual(rewards.compute_leg_target_cost(env).sum(), 0.)

    def test_pose_transitions_and_milestone_deduplication(self):
        _, cmd = make_env([0])
        cmd.test_heights[:] = .024
        cmd.test_u[:] = torch.tensor([-1., 1.])
        for _ in range(10):
            cmd._update_command()
        self.assertEqual(cmd.phase.item(), 1)
        self.assertTrue(cmd.s1_milestone_pulse.item())
        cmd.test_u[:] = 1.
        for _ in range(10):
            cmd._update_command()
        self.assertEqual(cmd.phase.item(), 2)
        self.assertTrue(cmd.s2_milestone_pulse.item())
        cmd.test_u[:] = torch.tensor([-1., 1.])
        for _ in range(10):
            cmd._update_command()
        self.assertEqual(cmd.phase.item(), 1)
        self.assertTrue(cmd._last_back_to_p2.item())
        cmd.test_u[:] = 1.
        for _ in range(10):
            cmd._update_command()
        self.assertEqual(cmd.phase.item(), 2)
        self.assertTrue(cmd.s2_transition_pulse.item())
        self.assertFalse(cmd.s2_milestone_pulse.item())
        cmd.test_u[:] = -1.
        for _ in range(15):
            cmd._update_command()
        self.assertEqual(cmd.phase.item(), 0)
        self.assertTrue(cmd._last_back_to_p1.item())

    def test_confirmation_wins_over_timeout(self):
        _, cmd = make_env([1, 1])
        cmd.test_heights[:] = .024
        cmd.test_u[1] = 0.
        cmd._update_dt = 0.
        cmd._update_command()
        cmd.t_phase[:] = .74
        cmd._s2_confirm_elapsed[0] = .09
        cmd._update_dt = .01
        cmd._update_command()
        torch.testing.assert_close(cmd.phase, torch.tensor([2, 1]))
        torch.testing.assert_close(cmd.retry, torch.tensor([0, 1]))
        torch.testing.assert_close(cmd._last_retry_mask, torch.tensor([False, True]))


if __name__ == '__main__':
    unittest.main()
