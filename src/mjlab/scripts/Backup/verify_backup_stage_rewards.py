# 阶段奖励、参考边界与站立判定回归；不创建环境、不修改训练数据。
import unittest
from math import cos, radians
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from mjlab.tasks.SQuRo_Backup.mdp.command import BackupCommand, BackupCommandCfg
from mjlab.tasks.SQuRo_Backup.mdp import reference, rewards, terminations, events
from mjlab.tasks.SQuRo_Backup.mdp.rewards import _milestone_time_quality
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

    def test_stand_window_uses_mean_velocity(self):
        env, cmd = make_env([2, 2, 2])
        cmd.test_vel[:] = torch.tensor([.3, 4.5, 1.])
        for _ in range(50):                                   # 攒 0.5s 窗口
            self.assertFalse(terminations.check_stand_success(env).any())
        # 判据量就是窗口平均速度: 恒定的 0.3 与 4.5 分别远离门限两侧
        self.assertAlmostEqual(env._stand_vel_integral[0].item() / env._stand_elapsed[0].item(), .3, places=4)
        self.assertAlmostEqual(env._stand_vel_integral[1].item() / env._stand_elapsed[1].item(), 4.5, places=4)
        cmd.test_vel[2] = .3                                  # 后半程变静 -> 窗口均值被拉低
        for _ in range(100):
            terminations.check_stand_success(env)
        self.assertLess((env._stand_vel_integral[2] / env._stand_elapsed[2]).item(), .75)  # (0.5×1.0+1.0×0.3)/1.5
        self.assertGreater(env._stand_elapsed[2].item(), 1.4)
        # 窗口攒满 1.5s 且均值远低于 3.5 -> 结算; 一直抖的 4.5 永远不结算
        torch.testing.assert_close(terminations.check_stand_success(env), torch.tensor([True, False, True]))

    def test_stand_window_resets_instead_of_pausing(self):
        env, cmd = make_env([2])
        cmd.test_vel[:] = .3
        for _ in range(50):
            terminations.check_stand_success(env)
        self.assertAlmostEqual(env._stand_elapsed[0].item(), .5, places=4)
        cmd.test_u[:] = torch.tensor([[-1., 1.]])             # 几何掉出 -> T 与 V 一起清零, 不是暂停累计
        terminations.check_stand_success(env)
        self.assertEqual(env._stand_elapsed[0].item(), 0.)
        self.assertEqual(env._stand_vel_integral[0].item(), 0.)
        # 窗口重启后必须重新攒满 1.5s, 不能把前后两段拼接成一次站稳
        cmd.test_u[:] = 1.
        for _ in range(149):
            self.assertFalse(terminations.check_stand_success(env).any())
        self.assertTrue(terminations.check_stand_success(env)[0])

    def test_stand_still_reward_is_p3_gated_and_linear(self):
        env, cmd = make_env([2, 1, 2])
        cmd.test_vel[:] = torch.tensor([0., 0., 3.])
        reward = rewards.compute_stand_still_reward(env)
        weight = _CURVES["weight_stand_still"][0]
        self.assertAlmostEqual(reward[0].item(), weight, places=6)      # 完全静止 -> 满分
        self.assertEqual(reward[1].item(), 0.)                          # 非 P3 -> 0
        self.assertAlmostEqual(reward[2].item(), weight * .5, places=6)  # 3 rad/s -> 线性核一半
        # 姿态不达标时不给分: 不存在"不进锥就不被罚"的反向作弊路线(惩罚形式才有)
        cmd.test_u[:] = torch.tensor([-1., 1.])
        self.assertEqual(rewards.compute_stand_still_reward(env).abs().sum().item(), 0.)

    def test_reset_clears_only_selected_stand_timer(self):
        env, _ = make_env([2, 2])
        env._stand_elapsed = torch.tensor([.4, .3])
        env._stand_vel_integral = torch.tensor([1.2, .9])
        robot = NS(num_joints=36, write_root_state_to_sim=lambda *a, **kw: None,
                   write_joint_state_to_sim=lambda *a, **kw: None)
        env.scene = NS(entities={'robot': robot})
        with patch.object(events, 'resolve_model_indices'):
            events.reset_model(env, torch.tensor([0]))
            torch.testing.assert_close(env._stand_elapsed, torch.tensor([0., .3]))
            torch.testing.assert_close(env._stand_vel_integral, torch.tensor([0., .9]))
            # 计时器尚未被 termination 创建过时也要能建立并清零, 不能静默跳过
            del env._stand_elapsed
            del env._stand_vel_integral
            events.reset_model(env, torch.tensor([1]))
            torch.testing.assert_close(env._stand_elapsed, torch.tensor([0., 0.]))
            torch.testing.assert_close(env._stand_vel_integral, torch.tensor([0., 0.]))

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
        # 早侧门控 = 名义段末 − 0.20λ = 2.40 − 0.60 = 1.80: 门控之前候选成立也不起算确认。
        cmd.t_phase[:] = 1.80
        for _ in range(10):
            cmd._update_command()
        # 段末门控: 姿态在门控后第 0.10s 就确认了, 但 P1 名义时长 0.80×λ=2.40s, 不得提前推进。
        self.assertEqual(cmd.phase.item(), 0)
        self.assertFalse(cmd.s1_milestone_pulse.item())
        # 脉冲起点记录的是"门控内候选的上升沿", 与名义段末之差以实际秒计 (早到为负);
        # 阶段时钟在每步开头先加一个 dt, 故起点落在 1.81。
        self.assertAlmostEqual(cmd._s1_onset.item(), 1.81, places=5)
        self.assertAlmostEqual(cmd._s1_dev_early.item(), 1.81 - .80 * 3., places=5)
        self.assertAlmostEqual(cmd._s1_dev_late.item(), 1.81 - .80 * 3., places=5)
        cmd.t_phase[:] = .80 * cmd.time_scale_command
        cmd._update_command()
        self.assertEqual(cmd.phase.item(), 1)
        self.assertTrue(cmd.s1_milestone_pulse.item())   # 到点才结算 => 早到不再有折扣红利
        # P2 里持续保持 S1 姿态不再触发任何回退 (取消 back_to_p2 的套利通道)
        for _ in range(30):
            cmd._update_command()
        self.assertEqual(cmd.phase.item(), 1)
        self.assertFalse(cmd._last_back_to_p1.any())
        self.assertFalse(cmd._last_back_to_p2.any())
        cmd.test_u[:] = 1.                                # S2 候选
        # t_phase 是"段内"时钟: 进入 P2 必须把 phase 与 t_phase 一起归零, 否则这个值会被
        # 拿去和 P2 的窗界(0.15λ+0.50=0.95)比较, 当步就判过窗。
        cmd.phase[:] = 1
        cmd.t_phase[:] = 0.
        for _ in range(10):
            cmd._update_command()
            self.assertEqual(cmd.phase.item(), 1)          # 段末与窗界都未到
            self.assertFalse(cmd._last_retry_mask.any())
        cmd._update_command()
        self.assertEqual(cmd.phase.item(), 1)             # 同样要等 P2 段末
        cmd.t_phase[:] = .15 * cmd.time_scale_command
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

    def test_reference_is_continuous_across_phase_switch(self):
        # 段末门控的直接产物: 推进只发生在参考段末, 而段末两侧的参考值本应相同。
        # 逐步检查参考变化量, 覆盖切换那一刻 —— 去掉门控后切换会发生在第 10 步(t_nom 仅 0.03),
        # 参考一步跳约 1.37 rad, 这里立刻超界。只查位置: 段末保持(ref_vel=0)与下一段起点
        # (取右侧导数)之间速度本来有台阶, 属既有设计。
        env, cmd = make_env([0])
        cmd.test_heights[:] = .024
        cmd.test_u[:] = torch.tensor([-1., 1.])
        prev, _ = reference.get_reference_joint_state(env)
        prev_phase = cmd.phase.item()
        advanced = False
        worst = 0.0
        for _ in range(260):                       # 覆盖 P1 段末(0.80×λ=2.40s=240 步)与切换
            cmd._update_command()
            pos, _ = reference.get_reference_joint_state(env)
            worst = max(worst, (pos - prev).abs().max().item())
            prev = pos
            advanced |= cmd.phase.item() != prev_phase
            prev_phase = cmd.phase.item()
        self.assertTrue(advanced)
        self.assertLess(worst, .1)

    def test_confirmation_wins_before_window_closes(self):
        # 窗界之内确认完成 -> 成功优先, 不因为接近截止而作废。
        # P2 窗 = [max(0, 0.15λ−0.20λ), 0.15λ+0.50]; λ=1 时门控被 clamp 到段首, 窗界 0.65 才是约束。
        _, cmd = make_env([1, 1])
        cmd.time_scale_command[:] = 1.
        cmd.test_heights[:] = .024
        cmd.test_u[:] = torch.tensor([1., 1.])           # 两环境都满足 S2 贴地路径
        cmd.test_u[1] = 0.
        cmd._update_dt = 0.
        cmd._update_command()
        cmd.t_phase[:] = .64                             # 窗内最后一格 (.65 为窗界)
        cmd._s2_confirm_elapsed[0] = .09
        cmd._update_dt = .01
        cmd._update_command()
        torch.testing.assert_close(cmd.phase, torch.tensor([2, 1]))
        torch.testing.assert_close(cmd.retry, torch.tensor([0, 0]))

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
        # 门控按"距段末 0.20λ 的提前量": P1 门控 = 1.80, P2 全长 0.45 故门控被 clamp 到段首。
        # env1 的高度要留在站立几何(>0.05)内, 才能验证进 P3 之后的站稳判据;
        # env0 保持贴地高度, 使其构造不出 S1 候选 (仍是 P1)。
        cmd.test_heights[:] = .055
        cmd.test_heights[0] = .024
        cmd.test_u[1] = 0.                                 # env1 先不满足 S2 候选
        for _ in range(50):
            cmd._update_command()
            self.assertFalse(cmd.s2_transition_pulse.any())
        self.assertEqual(cmd._s2_confirm_elapsed[1].item(), 0.)
        # 窗内形成候选 -> 起算; 中断一次必须立刻清零重来。
        cmd.test_u[1] = torch.tensor([1., 1.])
        for _ in range(5):
            cmd._update_command()
        self.assertAlmostEqual(cmd._s2_confirm_elapsed[1].item(), .05, places=4)
        cmd.test_u[1, 0] = 0.                              # 候选中断
        cmd._update_command()
        self.assertEqual(cmd._s2_confirm_elapsed[1].item(), 0.)
        cmd.phase[1] = 1
        cmd.t_phase[1] = 0.
        cmd.test_u[:] = 1.
        for _ in range(9):
            cmd._update_command()
            self.assertEqual(cmd.phase[1].item(), 1)
        self.assertAlmostEqual(cmd._s2_confirm_elapsed[1].item(), .09, places=4)
        # 段末门控: 确认成立, 但 P2 名义 0.15×λ=0.45s 未到, 不得提前推进。
        cmd._update_command()
        self.assertEqual(cmd.phase[1].item(), 1)
        cmd.t_phase[1] = .15 * cmd.time_scale_command[1]
        cmd._update_command()
        self.assertEqual(cmd.phase[1].item(), 2)
        self.assertTrue(cmd.s2_milestone_pulse[1].item())
        self.assertFalse(cmd.s2_milestone_pulse[0].item())
        # 进入 P3 不是任务成功：仍须独立满足 1.5 秒的"站稳"确认 (含关节速度条件)。
        for _ in range(149):
            self.assertFalse(terminations.check_stand_success(env).any())
        torch.testing.assert_close(terminations.check_stand_success(env), torch.tensor([False, True]))
        cmd._update_command()
        self.assertFalse(cmd.s2_milestone_pulse.any())

    def test_window_boundary_and_invalid_candidate_handling(self):
        # λ=1: P2 门控 = max(0, 0.15−0.20) = 0 (clamp 到段首), 窗界 = 0.15+0.50 = 0.65。
        # 阶段时钟在每步开头先加一个 dt, 所以"设 t_phase=X 再调一步"实际是在 X+dt 上判定。
        _, cmd = make_env([1, 1])
        cmd.time_scale_command[:] = 1.
        cmd.test_u[:] = torch.tensor([1., 1.])
        for _ in range(5):
            cmd._update_command()                        # 段首即门控内 -> 正常累积
        self.assertAlmostEqual(cmd._s2_confirm_elapsed.sum().item(), .10, places=5)
        self.assertFalse(cmd._last_retry_mask.any())
        # 窗内候选无效(单段抬起/NaN)只消耗窗口, 不起算也不作废。
        _, cmd = make_env([1, 1, 1])
        cmd.time_scale_command[:] = 1.
        cmd.test_u[:] = torch.tensor([1., 1.])
        cmd.t_phase[:] = .60
        cmd.test_u[1, 0] = 0.                            # 环境1: 前段侧立, 非 S2 候选
        cmd.test_u[2, 1] = float('nan')                  # 环境2: 退化向量
        cmd._update_command()
        torch.testing.assert_close(cmd.phase, torch.ones(3, dtype=torch.long))
        self.assertEqual(cmd.retry.sum().item(), 0)
        self.assertAlmostEqual(cmd._s2_confirm_elapsed[0].item(), .01, places=5)
        self.assertEqual(cmd._s2_confirm_elapsed[1:].sum().item(), 0.)
        # 过窗仍未确认 -> 一起作废重试, 阶段时钟与确认计时清零。
        cmd.t_phase[:] = .66
        cmd._update_command()
        self.assertTrue(cmd._last_retry_mask.all())
        torch.testing.assert_close(cmd.retry, torch.tensor([1, 1, 1]))
        self.assertEqual(cmd.t_phase.sum().item(), 0.)
        self.assertEqual(cmd._s2_confirm_elapsed.sum().item(), 0.)

    def test_p1_gate_blocks_early_arrival(self):
        # λ=3: P1 门控 = 2.40 − 0.60 = 1.80。0.30λ=0.90 就摆出 S1 也不起算确认。
        _, cmd = make_env([0])
        cmd.test_heights[:] = .024
        cmd.test_u[:] = torch.tensor([-1., 1.])
        for _ in range(100):                             # 推到 1.00s, 仍早于门控
            cmd._update_command()
        self.assertAlmostEqual(cmd.t_phase.item(), 1.00, places=4)
        self.assertEqual(cmd._s1_confirm_elapsed.item(), 0.)
        self.assertFalse(cmd._last_retry_mask.any())
        # 晚侧窗界: 一直不达成 -> 过 0.80λ+0.50=2.90 才作废。
        _, cmd = make_env([0])
        cmd.test_u[:] = torch.tensor([1., 1.])           # 一直不是 S1 候选
        cmd.t_phase[:] = .80 * 3. + .50 + .01
        cmd._update_command()
        torch.testing.assert_close(cmd.retry, torch.tensor([1]))
        torch.testing.assert_close(cmd._last_retry_mask, torch.tensor([True]))

    def test_p2_window_and_p1_independence(self):
        # P2 窗界 = 0.15λ + window_late_s; 窗内确认即成功, P1 环境不受影响。
        _, cmd = make_env([1, 1, 0])
        cmd.time_scale_command[:] = 1.
        cmd.test_u[1] = torch.tensor([1., 1.])
        cmd._update_dt = .01
        cmd._update_command()
        # 环境0: 已过窗界(.65) 且未确认 -> 重试; 环境1: 窗内最后一格, 确认刚好完成 -> 成功;
        # 环境2: P1 且 t_phase 仅 1.00 < 1.30(P1 窗界), 既不推进也不重试。
        cmd.t_phase[:] = torch.tensor([.70, .63, 1.00])
        cmd._s2_confirm_elapsed[1] = .09
        cmd._update_command()
        torch.testing.assert_close(cmd.phase, torch.tensor([1, 2, 0]))
        torch.testing.assert_close(cmd.retry, torch.tensor([1, 0, 0]))

    def test_manual_replay_shares_p2_window(self):
        from mjlab.scripts.SQuRo_Backup_Replay import StateMachinePolicy

        _, cmd = make_env([1])
        cmd.time_scale_command[:] = 1.
        cmd.test_u[:] = torch.tensor([1., 1.])
        cmd.t_phase[:] = .40
        hand = object.__new__(StateMachinePolicy)
        hand.phase, hand.t_phase, hand.lam = 'P2', .40, 1.
        hand.window_late_s, hand.early_lead_fraction = .50, .20
        hand.pose_confirm_s, hand.inverted_confirm_s = .1, .15
        hand._s1_confirm_t = hand._s2_confirm_t = hand._inverted_confirm_t = 0.
        hand.retry, hand._milestones = {'P1': 0, 'P2': 0}, {'S1': True, 'S2': False}
        hand.max_retry, hand.log_events, hand.events = 0, False, []
        hand._is_S1 = lambda: False
        hand._is_S2 = lambda: bool(cmd._check_S2()[0])
        hand._is_both_inverted = lambda: False
        hand._state = lambda: (1., 1.)
        hand._fz = lambda _: .055
        hand.fb, hand.hb = 0, 1
        # 窗内确认完成: 两侧同时推进到 P3。λ=1 时 P2 门控被 clamp 到段首(0.15−0.20<0),
        # 窗界 0.65 才是约束。两侧都预置 0.09 的确认进度, 于同一步 (0.64→0.65) 一起确认。
        hand._s2_confirm_t = .09
        cmd._update_command()                             # 先跑一步, 惰性建立确认计时张量
        hand.t_phase = cmd.t_phase[:] = torch.tensor([.64])
        cmd._s2_confirm_elapsed[:] = .09
        hand._advance_state(.01)
        cmd._update_command()
        self.assertEqual(hand.phase, 'P3')
        self.assertEqual(cmd.phase.item(), 2)
        self.assertEqual(hand.retry['P2'], cmd.retry.item())
        # 过窗仍未确认: 两侧在同一时刻一起作废重试。
        hand.phase = 'P2'
        hand._s2_confirm_t = 0.
        cmd.phase[:] = 1
        cmd._s2_confirm_elapsed[:] = 0.
        hand.t_phase = cmd.t_phase[:] = torch.tensor([.66])
        hand._advance_state(.01)
        cmd._update_command()
        self.assertEqual(hand.retry['P2'], 1)
        self.assertEqual(cmd.retry.item(), 1)

    def test_milestone_quality_kernel_calibration(self):
        # 时间质量核 q = q_early × q_late 的标定锚点, 全部取自实测的"手调参考 S1 脉冲起点"。
        lam = torch.tensor([1., 2., 3., 4.])
        # 手调参考实测起点 = 1.00/1.69/2.41/3.18 (阶段内实际秒), 名义段末 = 0.80λ。
        hand_onset = torch.tensor([1.00, 1.69, 2.41, 3.18])
        hand_dev = hand_onset - .80 * lam
        q_hand = _milestone_time_quality(hand_dev, hand_dev, lam)
        # λ≥2 必须几乎满分; λ=1 的 +0.20s 是手调参考自身的仿射滞后, 只要求不被判死。
        self.assertTrue((q_hand[1:] > .9).all())
        self.assertGreater(q_hand[0].item(), .7)
        # 名义时刻达成 -> 满分 (核在目标点取 1, 不是"门控一开就满分"的台阶)。
        zero = torch.zeros(4)
        torch.testing.assert_close(_milestone_time_quality(zero, zero, lam), torch.ones(4))
        # 早侧极早 (0.30λ 型的抄近路) -> 只剩约 1/5 的钱。σ_e=0.40λ 下这是可达的最强区分度:
        # 再收紧就会先误杀手调参考在 λ=4 的 −0.015λ 提前量。
        far_early = -.50 * lam
        q_far = _milestone_time_quality(far_early, far_early, lam)
        self.assertTrue((q_far < .25).all())
        self.assertTrue((q_far > .1).all())
        # 早侧门控刚开即达成 (dev = −0.20λ) -> 明显低于满分, 但也不是零: 留出连续梯度。
        gated = -.20 * lam
        q_gated = _milestone_time_quality(gated, gated, lam)
        self.assertTrue((q_gated < .85).all())
        self.assertTrue((q_gated > .2).all())
        # 单调性: 同一 λ 下越接近名义时刻, 质量单调不降。
        dev = torch.tensor([-.60, -.30, -.10, 0.0])
        q = _milestone_time_quality(dev, dev, torch.full((4,), 3.))
        self.assertTrue((q[1:] >= q[:-1]).all())
        # 晚侧只罚迟到: 同样的正偏差不得被早侧窄 σ 二次惩罚。
        self.assertAlmostEqual(_milestone_time_quality(torch.zeros(1), torch.tensor([.10]), torch.tensor([1.])).item(),
                               _milestone_time_quality(torch.zeros(1), torch.tensor([.10]), torch.tensor([3.])).item(), places=6)
        # NaN(尚未达成) 不得污染: 核必须有限。
        self.assertTrue(torch.isfinite(_milestone_time_quality(
            torch.tensor([float('nan')]), torch.tensor([float('nan')]), torch.tensor([3.]))).all())

    def test_milestone_weight_is_not_idle(self):
        # "提前翻完干等"必须真的被扣钱: 同一 pulse 下, 0.30λ 达成远低于 0.80λ 达成。
        env, cmd = make_env([0])
        env.step_dt = .01
        cmd.test_heights[:] = .024
        cmd.test_u[:] = torch.tensor([-1., 1.])
        w = _CURVES['weight_milestone_s1'][0]

        def full_reward_at(dev):
            cmd._last_s1_milestone = torch.tensor([True])
            cmd._s1_dev_early = torch.tensor([dev])
            cmd._s1_dev_late = torch.tensor([dev])
            return rewards.compute_s1_milestone_reward(env).item() * env.step_dt

        nominal = full_reward_at(0.)
        torch.testing.assert_close(torch.tensor(nominal), torch.tensor(w))
        early = full_reward_at(0.30 * 3. - 0.80 * 3.)      # 0.30λ 达成
        self.assertLess(early, nominal * .25)

    def test_gate_and_weight_are_both_effective(self):
        # 两者必须各自独立起作用, 不能互相掩盖。
        # (a) 只靠核: dev=0 时核为 1 -> 台阶仍在, 说明"必须有门控"。
        lam = torch.tensor([3.])
        self.assertAlmostEqual(_milestone_time_quality(torch.tensor([0.]), torch.tensor([0.]), lam).item(), 1., places=6)
        # (b) 只靠门控: 门控把起算推到 0.60λ, 但"门一开就满分"是台阶; 核把它压成斜坡。
        env, cmd = make_env([0])
        cmd.test_heights[:] = .024
        cmd.test_u[:] = torch.tensor([-1., 1.])
        cmd.t_phase[:] = 1.80                              # 门控刚开
        for _ in range(10):
            cmd._update_command()
        onset = cmd._s1_onset.item()
        self.assertAlmostEqual(onset, 1.81, places=5)
        self.assertLess(_milestone_time_quality(cmd._s1_dev_early, cmd._s1_dev_late,
                                                cmd.time_scale_command).item(), .85)
        # 门控前不累积确认: 只把时间放到门控之前, 确认计时必须为 0。
        _, cmd2 = make_env([0])
        cmd2.test_heights[:] = .024
        cmd2.test_u[:] = torch.tensor([-1., 1.])
        cmd2.t_phase[:] = 1.00                             # < 门控 1.80
        for _ in range(20):
            cmd2._update_command()
        self.assertEqual(cmd2._s1_confirm_elapsed.item(), 0.)


if __name__ == '__main__':
    unittest.main()
