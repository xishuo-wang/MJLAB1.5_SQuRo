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
    cmd.test_vel = torch.zeros(n)
    cmd._pose_cos = lambda: cmd.test_u
    cmd._body_height = lambda idx: cmd.test_heights[:, 0 if idx == _MODEL_INDICES.f_body_id else 1]
    cmd._joint_vel_rms = lambda: cmd.test_vel
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
        for _ in range(149):
            self.assertFalse(terminations.check_stand_success(env).any())
        # 第 150 步 (0.01×150=1.5s) 才确认; 非 P3 的环境永远不确认
        torch.testing.assert_close(terminations.check_stand_success(env), torch.tensor([True, False]))
        cmd.test_u[0, 0] = -1.
        self.assertFalse(terminations.check_stand_success(env).any())
        self.assertEqual(env._stand_elapsed[0], 0.)
        cmd.test_u[:] = 1.
        env.step_dt = .05
        for _ in range(29):
            self.assertFalse(terminations.check_stand_success(env).any())
        self.assertTrue(terminations.check_stand_success(env)[0])

    def test_stand_requires_low_joint_velocity(self):
        env, cmd = make_env([2, 2, 2])
        # 0.3 进入阈值内 / 4.5 落在迟滞带 (3.5 与 5.5 之间) / 6.0 超过退出阈值
        cmd.test_vel[:] = torch.tensor([.3, 4.5, 6.])
        for _ in range(200):
            terminations.check_stand_success(env)
        self.assertAlmostEqual(env._stand_elapsed[0].item(), 2., places=4)   # 一直达标 -> 持续累计
        self.assertAlmostEqual(env._stand_elapsed[1].item(), 0., places=4)   # 迟滞带只侵蚀, 不累计
        self.assertAlmostEqual(env._stand_elapsed[2].item(), 0., places=4)   # 超过退出阈值 -> 立即清零

    def test_stand_dropout_erodes_instead_of_pausing(self):
        env, cmd = make_env([2, 2])
        cmd.test_vel[:] = .3
        for _ in range(50):                                   # 先攒 0.5s
            terminations.check_stand_success(env)
        self.assertAlmostEqual(env._stand_elapsed[0].item(), .5, places=4)
        cmd.test_vel[:] = 4.5                                 # 掉进迟滞带: 按 2 倍速率侵蚀
        for _ in range(10):                                   # 0.1s 掉出 -> 只退 0.2s
            terminations.check_stand_success(env)
        self.assertAlmostEqual(env._stand_elapsed[0].item(), .3, places=4)
        for _ in range(20):                                   # 再掉 0.2s -> 侵蚀到 0
            terminations.check_stand_success(env)
        self.assertAlmostEqual(env._stand_elapsed[0].item(), 0., places=4)
        # 侵蚀到 0 后必须重新从 0 攒满 1.5s, 不能把多次短暂达标拼成一次连续站稳
        cmd.test_vel[:] = .3
        for _ in range(149):
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

    def test_s2_accepts_grounded_or_both_segments_raised(self):
        _, cmd = make_env([1] * 10)
        cmd.test_heights[:] = torch.tensor([
            [.024, .034], [.0535, .0568], [.055, .055], [.055, .03],
            [.045, .045], [.055, .055], [.024, .024], [.024, .024],
            [.024, float('-inf')], [.055, float('inf')],
        ])
        cmd.test_u[:] = torch.tensor([
            [.75, .75], [.9525, .9976], [.8, 1.], [1., 1.],
            [1., 1.], [-1., 1.], [0., 1.], [float('nan'), 1.],
            [1., 1.], [1., 1.],
        ])
        torch.testing.assert_close(cmd._check_S2(), torch.tensor([True, True] + [False] * 8))
        # 抬身路径严格复用站立几何边界，不扩大朝向容差或放宽到单端抬起。
        cmd.test_u[:] = 1.
        cmd.test_heights[:] = .05
        self.assertFalse(cmd._check_S2().any())
        cmd.test_heights[:] = .055
        cmd.test_u[:, 0] = .9
        self.assertFalse(cmd._check_S2().any())

    def test_raised_s2_requires_continuity_and_does_not_skip_p1(self):
        env, cmd = make_env([0, 1])
        for _ in range(9):
            cmd._update_command()
            self.assertFalse(cmd.s2_transition_pulse.any())
        cmd.test_u[1, 0] = 0.
        cmd._update_command()
        self.assertEqual(cmd._s2_confirm_elapsed[1].item(), 0.)
        cmd.test_u[:] = 1.
        for _ in range(9):
            cmd._update_command()
            self.assertEqual(cmd.phase[1].item(), 1)
        cmd._update_command()
        torch.testing.assert_close(cmd.phase, torch.tensor([0, 2]))
        torch.testing.assert_close(cmd.s2_milestone_pulse, torch.tensor([False, True]))
        # 进入 P3 不是任务成功：仍须独立满足 1.5 秒的"站稳"确认 (含关节速度条件)。
        for _ in range(149):
            self.assertFalse(terminations.check_stand_success(env).any())
        torch.testing.assert_close(terminations.check_stand_success(env), torch.tensor([False, True]))
        cmd._update_command()
        self.assertFalse(cmd.s2_milestone_pulse.any())

    def test_s2_confirmation_grace_across_time_scales(self):
        env, cmd = make_env([1, 1, 1, 1])
        cmd.time_scale_command[:] = torch.tensor([1., 2., 3., 4.])
        deadline = .15 * cmd.time_scale_command + cmd.cfg.p2_buffer_s
        cmd.t_phase[:] = deadline - .01
        # 首个候选发生在截止边界附近：不能立刻清零重播，也不能当步过关。
        for _ in range(9):
            cmd._update_command()
            self.assertTrue((cmd.phase == 1).all())
            self.assertFalse(cmd._last_retry_mask.any())
            pos, vel = reference.get_reference_joint_state(env)
            torch.testing.assert_close(pos[:, [0, 1, 8, 9]],
                                       torch.tensor([.6, 0., 0., 0.]).expand(4, 4), atol=1e-6, rtol=0.)
            self.assertEqual(vel.abs().sum(), 0.)
        cmd._update_command()
        self.assertTrue((cmd.phase == 2).all())
        self.assertTrue(cmd.s2_milestone_pulse.all())
        self.assertEqual(cmd.retry.sum(), 0)

    def test_s2_grace_breaks_immediately_on_invalid_candidate(self):
        _, cmd = make_env([1, 1, 1])
        cmd.time_scale_command[:] = 1.
        cmd.t_phase[:] = .45
        cmd._update_command()
        self.assertFalse(cmd._last_retry_mask.any())
        cmd.test_u[0, 0] = 0.
        cmd.test_heights[1] = .045
        cmd.test_u[2, 1] = float('nan')
        cmd._update_command()
        self.assertTrue(cmd._last_retry_mask.all())
        self.assertEqual(cmd.retry.sum(), 3)
        self.assertEqual(cmd.t_phase.sum(), 0.)
        self.assertEqual(cmd._s2_confirm_elapsed.sum(), 0.)

    def test_s2_grace_has_hard_deadline_and_does_not_change_p1(self):
        _, cmd = make_env([1, 1, 0])
        cmd.time_scale_command[:] = 1.
        cmd._update_dt = 0.
        cmd._update_command()
        cmd._update_dt = .01
        cmd.t_phase[:] = torch.tensor([.56, .56, 1.81])
        # 第一个环境刚形成候选但已超过宽限上限；第二个恰好确认，仍应成功优先。
        cmd._s2_confirm_elapsed[1] = .09
        cmd.test_u[2] = torch.tensor([-1., 1.])
        cmd.test_heights[2] = .024
        cmd._update_command()
        torch.testing.assert_close(cmd.phase, torch.tensor([1, 2, 0]))
        torch.testing.assert_close(cmd.retry, torch.tensor([1, 0, 1]))

    def test_manual_replay_shares_s2_grace(self):
        from mjlab.scripts.SQuRo_backup_Replay import StateMachinePolicy

        _, cmd = make_env([1])
        cmd.time_scale_command[:] = 1.
        cmd.t_phase[:] = .44
        hand = object.__new__(StateMachinePolicy)
        hand.phase, hand.t_phase, hand.lam = 'P2', .44, 1.
        hand.p2_buffer_s, hand.pose_confirm_s, hand.inverted_confirm_s = .3, .1, .15
        hand._s1_confirm_t = hand._s2_confirm_t = hand._inverted_confirm_t = 0.
        hand.retry, hand._milestones = {'P1': 0, 'P2': 0}, {'S1': True, 'S2': False}
        hand.max_retry, hand.log_events, hand.events = 0, False, []
        hand._is_S1 = lambda: False
        hand._is_S2 = lambda: bool(cmd._check_S2()[0])
        hand._is_both_inverted = lambda: False
        hand._state = lambda: (1., 1.)
        hand._fz = lambda _: .055
        hand.fb, hand.hb = 0, 1
        for _ in range(10):
            hand._advance_state(.01)
            cmd._update_command()
            self.assertEqual(hand.phase, ['P1', 'P2', 'P3'][cmd.phase.item()])
            self.assertEqual(hand.retry['P2'], cmd.retry.item())
        self.assertEqual(hand.phase, 'P3')
        # 候选中断时，手调与训练同样立即重试。
        hand.phase, hand.t_phase, hand._s2_confirm_t = 'P2', .47, .03
        cmd.phase[:] = 1
        cmd.t_phase[:] = .47
        cmd._s2_confirm_elapsed[:] = .03
        cmd.test_u[:] = 0.
        hand._advance_state(.01)
        cmd._update_command()
        self.assertEqual(hand.retry['P2'], 1)
        self.assertEqual(cmd.retry.item(), 1)


if __name__ == '__main__':
    unittest.main()
