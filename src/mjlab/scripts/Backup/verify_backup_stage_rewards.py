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

    def test_progress_terrain_is_monotone(self):
        env, cmd = make_env([0, 1, 2])

        def total(u):
            cmd.test_u[:] = torch.tensor(u)
            return (rewards.compute_s1_progress_reward(env)
                    + rewards.compute_s2_progress_reward(env))

        supine = total([-1., -1.])          # 完全仰卧
        rear_only = total([-1., 1.])        # S1: 后段已翻正
        side = total([0., 1.])              # 正侧立: 前段翻到一半
        prone = total([1., 1.])             # S2: 两段都翻正
        wrong_order = total([1., -1.])      # 前段先翻(错误顺序)

        for t in (supine, rear_only, side, prone, wrong_order):
            # 区间项不按阶段门控: 三个环境读数必须相同
            torch.testing.assert_close(t, t[:1].expand(3))
            self.assertTrue(torch.isfinite(t).all())

        # 躺平与错误顺序都不给分, 不构成"不动也拿分"的底分
        self.assertEqual(supine[0].item(), 0.)
        self.assertEqual(wrong_order[0].item(), 0.)
        # 沿目标轨迹严格单调: 仰卧 < S1 < 侧立 < S2, 且 uF<0 半程不许出现零梯度平台
        self.assertLess(supine[0].item(), rear_only[0].item())
        self.assertLess(rear_only[0].item(), side[0].item())
        self.assertLess(side[0].item(), prone[0].item())
        # 数值锚点直接从 _CURVES 推导, 之后调权重不必改测试
        w1 = _CURVES["weight_progress_s1"][0]
        w2 = _CURVES["weight_progress_s2"][0]
        torch.testing.assert_close(rear_only, torch.full((3,), w1))
        torch.testing.assert_close(side, torch.full((3,), w1 + w2 / 2))
        torch.testing.assert_close(prone, torch.full((3,), w1 + w2))
        # S2 必须严格优于 S1: 两者相等时"停在 S1"就是并列最优解
        self.assertGreater(w2, 0.0)

    def test_s3_progress_is_p3_only(self):
        env, cmd = make_env([0, 1, 2])
        cmd.test_u[:] = -1.                 # 仰面: 朝向门控关闭 -> 三段全 0
        self.assertEqual(rewards.compute_s3_progress_reward(env).sum(), 0.)
        cmd.test_u[:] = 1.                  # 双段正置 + 高度达标 -> 只有 P3 拿到
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

    def test_standing_progress_is_gated_and_monotone(self):
        _, cmd = make_env([2])
        cmd.test_heights[:] = .055          # 高度已达标, 只考察朝向门控
        # 任一段落在 45 度锥外 -> 进度必须为 0
        cmd.test_u[:] = torch.tensor([[.5, 1.]])
        self.assertEqual(cmd.standing_state()[1].item(), 0.)
        # 锥内随朝向单调增长, 完全正置时为 1
        cmd.test_u[:] = torch.tensor([[.85, 1.]])
        low = cmd.standing_state()[1].item()
        cmd.test_u[:] = torch.tensor([[.95, 1.]])
        high = cmd.standing_state()[1].item()
        self.assertGreater(high, low)
        self.assertGreater(low, 0.)
        cmd.test_u[:] = 1.
        self.assertAlmostEqual(cmd.standing_state()[1].item(), 1., places=6)
        # 高度不达标时用"较低的一段"决定进度: 只抬起一端拿不到分
        cmd.test_heights[:] = torch.tensor([[.055, .024]])
        self.assertEqual(cmd.standing_state()[1].item(), 0.)

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
            # 计时器尚未被 termination 创建过时也要能建立并清零, 不能静默跳过
            del env._stand_elapsed
            events.reset_model(env, torch.tensor([1]))
            torch.testing.assert_close(env._stand_elapsed, torch.tensor([0., 0.]))

    def test_p2_boundary_and_p3_is_continuous(self):
        env, cmd = make_env([1, 1, 1, 2])
        cmd.t_phase[:] = torch.tensor([.43, .44, .75, 0.])
        pos, vel = reference.get_reference_joint_state(env)
        # 前三个环境在 P2: 保持段读到 T3 左端极限 0.6; P3 起点也承接 0.6, 不再跳回 0
        torch.testing.assert_close(pos[:, 0], torch.tensor([.5822222, .5911111, .6, .6]), atol=2e-6, rtol=0.)
        torch.testing.assert_close(vel[:2, 0], torch.full((2,), .4/.45), atol=1e-5, rtol=0.)
        self.assertEqual(vel[2].abs().sum(), 0.)
        self.assertEqual(pos[3, [1, 8, 9]].abs().sum(), 0.)
        # P3 入口 F_spine1 以 0.6/0.5/λ 的速率线性回零, 其余脊柱速度为 0
        torch.testing.assert_close(vel[3, 0], torch.tensor(-0.6 / 0.5 / 3.0), atol=1e-5, rtol=0.)
        self.assertEqual(vel[3, [1, 8, 9]].abs().sum(), 0.)
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
            with patch.dict(_CURVES, weight_spine_target=(4.,)):
                torch.testing.assert_close(rewards.compute_spine_target_cost(env), torch.full((3,), -4.))
            action.raw_action.zero_()
            action.raw_action[:, action.target_names.index('F_body_joint')] = 2.
            torch.testing.assert_close(rewards.compute_spine_target_cost(env), torch.full((3,), -2.))

    def test_action_ctrl_excess_penalty(self):
        env, cmd = make_env([0])
        action = NS(raw_action=torch.zeros(1, 14), scale=1., offset=0.,
                    target_names=list(reversed(_ACTUATED_JOINT_NAMES)))
        env.action_manager = NS(get_term=lambda _: action)
        with patch.dict(_CURVES, weight_action_excess=(1.,)):
            # 命令落在 ctrlrange 内 -> 零成本 (贴住限位撑地不受罚)
            action.raw_action[:, action.target_names.index('F_body_joint')] = 1.0
            self.assertEqual(rewards.compute_action_ctrl_excess_penalty(env).item(), 0.)
            # 只对被丢弃的那一段计成本: HL_hip 上限 0.8, 目标 5.0 -> 超出 4.2, 均值除以 14 列
            action.raw_action.zero_()
            action.raw_action[:, action.target_names.index('HL_hip_joint')] = 5.0
            self.assertAlmostEqual(rewards.compute_action_ctrl_excess_penalty(env).item(),
                                   -4.2 / 14, places=6)
            # 必须计入 scale: 同样 action=6.0, scale 0.3 时 F_body 目标 1.8 才刚超出 1.57
            action.scale = 0.3
            action.raw_action.zero_()
            action.raw_action[:, action.target_names.index('F_body_joint')] = 6.0
            self.assertAlmostEqual(rewards.compute_action_ctrl_excess_penalty(env).item(),
                                   -0.23 / 14, places=6)
            # 负方向同样计成本
            action.raw_action.zero_()
            action.raw_action[:, action.target_names.index('F_spine1_joint')] = -6.0
            self.assertAlmostEqual(rewards.compute_action_ctrl_excess_penalty(env).item(),
                                   -(1.8 - 0.6) / 14, places=6)

    def test_phase_is_monotone_and_milestones_once_per_episode(self):
        _, cmd = make_env([0])
        cmd.test_heights[:] = .024
        cmd.test_u[:] = torch.tensor([-1., 1.])          # S1 候选
        for _ in range(10):
            cmd._update_command()
        self.assertEqual(cmd.phase.item(), 1)
        self.assertTrue(cmd.s1_milestone_pulse.item())
        # P2 里持续保持 S1 姿态不再触发任何回退 (取消 back_to_p2 的套利通道)
        for _ in range(30):
            cmd._update_command()
        self.assertEqual(cmd.phase.item(), 1)
        self.assertFalse(cmd._last_back_to_p1.any())
        self.assertFalse(cmd._last_back_to_p2.any())
        cmd.test_u[:] = 1.                                # S2 候选
        for _ in range(10):
            cmd._update_command()
        self.assertEqual(cmd.phase.item(), 2)
        self.assertTrue(cmd.s2_milestone_pulse.item())
        # P3 里摆回 S1 姿态或双倒都不再退回, 里程碑也不重复发放
        cmd.test_u[:] = torch.tensor([-1., 1.])
        for _ in range(30):
            cmd._update_command()
        self.assertEqual(cmd.phase.item(), 2)
        self.assertFalse(cmd.s1_milestone_pulse.item())
        cmd.test_u[:] = -1.
        for _ in range(30):
            cmd._update_command()
        self.assertEqual(cmd.phase.item(), 2)
        self.assertFalse(cmd.s2_milestone_pulse.item())
        self.assertTrue(cmd._last_inverted_confirmed.item())   # 双倒仍作为诊断指标保留

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
