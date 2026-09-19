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
    # _resample_command 会写这些视图; 单测直接调它, 必须预置。
    cmd.vel_command = cmd.command_tensor[:, 0]
    cmd.height_f_command = cmd.command_tensor[:, 1]
    cmd.height_h_command = cmd.command_tensor[:, 2]
    cmd.gait_freq_command = cmd.command_tensor[:, 3]
    cmd.curvature_command = cmd.command_tensor[:, 4]
    cmd.fixed_time_scale = None          # _resample_command 会读它决定 λ 来源
    cmd._update_dt = env.step_dt
    cmd._pose_cos_threshold = cos(radians(45))
    cmd._pose_cache = None
    cmd._pose_cos_cache = None
    # 站立窗口现在归 command 所有 (判定必须发生在复位之前)。
    cmd._stand_elapsed = torch.zeros(n)
    cmd._stand_vel_integral = torch.zeros(n)
    cmd._pending_cycle_reset = torch.zeros(n, dtype=torch.bool)
    cmd._last_cycle_reset = torch.zeros(n, dtype=torch.bool)
    cmd._last_cycle_end_pulse = torch.zeros(n, dtype=torch.bool)
    # 未加掩码的真实首次到达时刻 (里程碑时间质量核的输入)
    cmd._s1_criterion_first = torch.full((n,), float('nan'))
    cmd._s2_criterion_first = torch.full((n,), float('nan'))
    # 每循环一次的首次达成锁存 + 本回合循环数
    cmd._s1_cycle_latched = torch.zeros(n, dtype=torch.bool)
    cmd._s2_cycle_latched = torch.zeros(n, dtype=torch.bool)
    cmd._cycles_this_episode = torch.zeros(n, dtype=torch.long)
    cmd._pending_episode_reset = torch.zeros(n, dtype=torch.bool)
    cmd._s1_onset = torch.full((n,), float('nan'))
    cmd._s2_onset = torch.full((n,), float('nan'))
    env.reset_buf = torch.zeros(n, dtype=torch.bool)
    # _resample_command 会写这些诊断锁存
    cmd._last_s1_ok = torch.zeros(n, dtype=torch.bool)
    cmd._last_s2_ok = torch.zeros(n, dtype=torch.bool)
    cmd._last_advance1 = torch.zeros(n, dtype=torch.bool)
    cmd._last_advance2 = torch.zeros(n, dtype=torch.bool)
    cmd._last_retry_mask = torch.zeros(n, dtype=torch.bool)
    cmd._last_s1_milestone = torch.zeros(n, dtype=torch.bool)
    cmd._last_s2_milestone = torch.zeros(n, dtype=torch.bool)
    cmd._last_back_to_p1 = torch.zeros(n, dtype=torch.bool)
    cmd._last_back_to_p2 = torch.zeros(n, dtype=torch.bool)
    cmd._last_both_inverted = torch.zeros(n, dtype=torch.bool)
    cmd._last_s1_confirmed = torch.zeros(n, dtype=torch.bool)
    cmd._last_s2_confirmed = torch.zeros(n, dtype=torch.bool)
    cmd._last_inverted_confirmed = torch.zeros(n, dtype=torch.bool)
    cmd._last_s1_gated_ok = torch.zeros(n, dtype=torch.bool)
    cmd._last_s2_gated_ok = torch.zeros(n, dtype=torch.bool)
    cmd._inverted_confirm_elapsed = torch.zeros(n)
    cmd._s1_confirm_elapsed = torch.zeros(n)
    cmd._s2_confirm_elapsed = torch.zeros(n)
    cmd._s1_awarded = torch.zeros(n, dtype=torch.bool)
    cmd._s2_awarded = torch.zeros(n, dtype=torch.bool)
    cmd.test_u = torch.ones(n, 2)
    cmd.test_heights = torch.full((n, 2), .055)
    cmd.test_vel = torch.zeros(n)
    cmd._pose_cos = lambda: cmd.test_u
    cmd._body_height = lambda idx: cmd.test_heights[:, 0 if idx == _MODEL_INDICES.f_body_id else 1]
    cmd._joint_vel_rms = lambda: cmd.test_vel
    env.command_manager = NS(get_term=lambda _: cmd, _terms={'backup_cmd': cmd})
    # 循环复位会碰这些接口; 记录调用以便断言"只复位选中的环境、不结束回合"。
    env.reset_calls = {'sim': [], 'scene': [], 'obs': [], 'act': []}
    env.sim = NS(reset=lambda ids: env.reset_calls['sim'].append(list(ids)),
                 forward=lambda: None)
    robot = NS(write_root_state_to_sim=lambda *a, **kw: None,
               write_joint_state_to_sim=lambda *a, **kw: None, num_joints=36,
               # apply_fallen_state 会调 resolve_model_indices; 单测里 f_body_id 已被 setUp
               # 预置成 0/1, 该函数会提前 return, 所以这里只需保证被调用时不抛异常。
               find_bodies=lambda *a, **kw: ([], []),
               find_sites=lambda *a, **kw: ([], []),
               find_joints=lambda *a, **kw: ([], []))

    # scene 替身: 既支持 env.scene["robot"] 也支持 env.scene.reset(ids)/.entities,
    # 因为生产代码两条路径都会走 (apply_fallen_state 用 entities, 复位用 reset)。
    class _FakeScene(dict):
        def __init__(self, mapping, on_reset):
            super().__init__(mapping)
            self.entities = mapping
            self._on_reset = on_reset

        def reset(self, ids):
            self._on_reset(ids)

    env.scene = _FakeScene({'robot': robot},
                           lambda ids: env.reset_calls['scene'].append(list(ids)))
    env.make_robot = robot          # 供个别测试替换 data 时保留其余接口
    env.observation_manager = NS(reset=lambda ids: env.reset_calls['obs'].append(list(ids)))
    env.action_manager = NS(reset=lambda ids: env.reset_calls['act'].append(list(ids)))
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
        # 站立窗口现在由 command 拥有, 判定入口是 stand_reward_and_pulse (返回 旧T, 旧V, 均值, 本步完成脉冲)。
        env, cmd = make_env([2, 1])
        for _ in range(149):
            self.assertFalse(cmd.stand_reward_and_pulse()[3].any())
        # 第 150 步 (0.01×150=1.5s) 才确认; 非 P3 的环境永远不确认
        torch.testing.assert_close(cmd.stand_reward_and_pulse()[3], torch.tensor([True, False]))
        self.assertTrue(cmd.consume_cycle_end_pulse()[0])       # 完成脉冲可被消费一次
        self.assertFalse(cmd.consume_cycle_end_pulse().any())   # 第二次读到的是空
        self.assertTrue(cmd._pending_cycle_reset[0].item())     # 复位待执行
        cmd._pending_cycle_reset[:] = False                     # 模拟下一步的复位(测试不跑完整 step)
        cmd.test_u[0, 0] = -1.
        self.assertFalse(cmd.stand_reward_and_pulse()[3].any())
        self.assertEqual(cmd._stand_elapsed[0], 0.)
        cmd.test_u[:] = 1.
        env.step_dt = .05
        cmd._update_dt = .05
        for _ in range(29):
            self.assertFalse(cmd.stand_reward_and_pulse()[3].any())
        self.assertTrue(cmd.stand_reward_and_pulse()[3][0])

    def test_stand_window_uses_mean_velocity(self):
        env, cmd = make_env([2, 2, 2])
        cmd.test_vel[:] = torch.tensor([.3, 4.5, 1.])
        for _ in range(50):                                   # 攒 0.5s 窗口
            self.assertFalse(cmd.stand_reward_and_pulse()[3].any())
        # 判据量就是窗口平均速度: 恒定的 0.3 与 4.5 分别远离门限两侧
        self.assertAlmostEqual(cmd._stand_vel_integral[0].item() / cmd._stand_elapsed[0].item(), .3, places=4)
        self.assertAlmostEqual(cmd._stand_vel_integral[1].item() / cmd._stand_elapsed[1].item(), 4.5, places=4)
        cmd.test_vel[2] = .3                                  # 后半程变静 -> 窗口均值被拉低
        for _ in range(100):
            cmd.stand_reward_and_pulse()
        self.assertLess((cmd._stand_vel_integral[2] / cmd._stand_elapsed[2]).item(), .75)  # (0.5×1.0+1.0×0.3)/1.5
        self.assertGreater(cmd._stand_elapsed[2].item(), 1.4)
        # 窗口攒满 1.5s 且均值远低于 3.5 -> 结算; 一直抖的 4.5 永远不结算。
        # 先清掉前面已完成的那个循环的待复位标志(测试不跑完整 step), 否则会被"每循环一次"的锁挡住。
        cmd._pending_cycle_reset[:] = False
        torch.testing.assert_close(cmd.stand_reward_and_pulse()[3], torch.tensor([True, False, True]))

    def test_stand_window_resets_instead_of_pausing(self):
        env, cmd = make_env([2])
        cmd.test_vel[:] = .3
        for _ in range(50):
            cmd.stand_reward_and_pulse()
        self.assertAlmostEqual(cmd._stand_elapsed[0].item(), .5, places=4)
        cmd.test_u[:] = torch.tensor([[-1., 1.]])             # 几何掉出 -> T 与 V 一起清零, 不是暂停累计
        cmd.stand_reward_and_pulse()
        self.assertEqual(cmd._stand_elapsed[0].item(), 0.)
        self.assertEqual(cmd._stand_vel_integral[0].item(), 0.)
        # 窗口重启后必须重新攒满 1.5s, 不能把前后两段拼接成一次站稳
        cmd.test_u[:] = 1.
        for _ in range(149):
            self.assertFalse(cmd.stand_reward_and_pulse()[3].any())
        self.assertTrue(cmd.stand_reward_and_pulse()[3][0])

    def test_stand_still_reward_is_p3_gated_and_linear(self):
        # 锚点从常量推导, 阈值调整时不必改测试(此前硬编码 "3 rad/s -> 半值",
        # 把 STAND_STILL_FULL_SPEED 从 6.0 改成 4.5 后立刻失效)。
        from mjlab.tasks.SQuRo_Backup.mdp import timing as T
        env, cmd = make_env([2, 1, 2])
        half = T.STAND_STILL_FULL_SPEED * .5
        cmd.test_vel[:] = torch.tensor([0., 0., half])
        reward = rewards.compute_stand_still_reward(env)
        weight = _CURVES["weight_stand_still"][0]
        self.assertAlmostEqual(reward[0].item(), weight, places=6)      # 完全静止 -> 满分
        self.assertEqual(reward[1].item(), 0.)                          # 非 P3 -> 0
        self.assertAlmostEqual(reward[2].item(), weight * .5, places=6)  # 半速 -> 线性核一半
        # 线性核必须在工作区间内可分辨: 站定实测 vel_rms 中位约 1.16, 核值不得已饱和
        cmd.test_vel[:] = torch.tensor([0., 0., 1.16])
        r = rewards.compute_stand_still_reward(env)[2].item()
        self.assertLess(r, weight)
        self.assertGreater(r, weight * .6)
        # 姿态不达标时不给分: 不存在"不进锥就不被罚"的反向作弊路线(惩罚形式才有)
        cmd.test_u[:] = torch.tensor([-1., 1.])
        self.assertEqual(rewards.compute_stand_still_reward(env).abs().sum().item(), 0.)

    def test_reset_model_writes_fallen_state_only(self):
        # reset_model 现在只写仰卧初态; 站立窗口/循环状态的归属地已迁到 BackupCommand,
        # 由 command_manager.reset → _resample_command → _clear_cycle_state 清理。
        # 这里同时断言它**不再**去碰 env._stand_* (旧字段已废弃, 碰了会让人误以为清干净了)。
        env, cmd = make_env([2, 2])
        called = {}
        with patch.object(events, 'resolve_model_indices'), \
             patch.object(events, 'apply_fallen_state',
                          lambda e, ids: called.setdefault('ids', list(ids))):
            events.reset_model(env, torch.tensor([0]))
            self.assertEqual(called['ids'], [0])
            events.reset_model(env, torch.tensor([]))          # 空集合必须安全返回
            self.assertEqual(called['ids'], [0])
        self.assertFalse(hasattr(env, '_stand_elapsed'))       # 不再创建旧字段
        # 循环状态的清理入口是 _clear_cycle_state
        cmd._stand_elapsed[1] = .4
        cmd._cycles_this_episode[1] = 5
        cmd._clear_cycle_state(torch.tensor([0]))
        torch.testing.assert_close(cmd._stand_elapsed, torch.tensor([0., .4]))
        torch.testing.assert_close(cmd._cycles_this_episode, torch.tensor([0, 5]))

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
        self.assertAlmostEqual(cmd.s1_dev_early.item(), 1.81 - .80 * 3., places=5)
        self.assertAlmostEqual(cmd.s1_dev_late.item(), 1.81 - .80 * 3., places=5)
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

    def test_body_height_reference_is_monotone_ramp_in_p3(self):
        # P3 的期望身体高度改为解析斜坡: 从趴平 0.024 线性抬到站立目标 0.055, 时长 = T4 名义。
        # 动机(技术细节 2026-09-19 §结论 3): 录制表在 T4 中段把期望拉回 0.0248, 使"保持趴姿"
        # 几乎最优、而要求站立的那段给出最低核值 —— 参考方向与任务目标相反。
        from mjlab.tasks.SQuRo_Backup.mdp import timing as T
        env, cmd = make_env([2])                      # phase=2 -> P3
        lam = float(cmd.time_scale_command[0])
        t4 = T.STAND_TRANSITION_DURATION * lam
        zs, kern_prone = [], []
        for frac in (0.0, .1, .25, .5, .75, .9, 1.0):
            cmd.t_phase[:] = frac * t4
            _, zF, _, zH = reference.get_body_reference(env)
            # F/H 必须给同一条斜坡, 否则 min(zF,zH) 仍被另一段拖住
            torch.testing.assert_close(zF, zH, atol=1e-5, rtol=0.)
            zs.append(float(zF[0]))
            kern_prone.append(float(torch.exp(-500.0 * (zF[0] - T.STAND_GROUND_HEIGHT) ** 2)))
        # 起点=趴平高度, 终点=站立目标
        self.assertAlmostEqual(zs[0], T.STAND_GROUND_HEIGHT, places=5)
        self.assertAlmostEqual(zs[-1], T.STAND_TARGET_HEIGHT, places=5)
        # 单调抬升
        for a, b in zip(zs, zs[1:]):
            self.assertLess(a, b)
        # 核心断言: 趴姿的核值必须**单调下降** —— 旧参考在中段反而最高(0.966~0.999)
        for a, b in zip(kern_prone, kern_prone[1:]):
            self.assertGreater(a, b)
        self.assertAlmostEqual(kern_prone[0], 1.0, places=4)
        self.assertLess(kern_prone[-1], .65)

    def test_body_height_reference_leaves_p1_p2_untouched(self):
        # P1/P2 的 z 必须仍取录制值 —— 斜坡只在 **phase == 2** 生效。
        # 关键回归: 早先用 source_t >= 0.95 判定, 而 P2 播完后 _stage_t_nom 会把 source_t
        # 冻结在 0.95, 于是 P2 的末端与 S2 确认等待段也被斜坡覆盖(实测 F/H 参考从
        # 0.0445/0.0503 被改成 0.024/0.024)。这里逐 λ 检查 P2 全段(含等待段)不被改。
        from mjlab.tasks.SQuRo_Backup.mdp import timing as T
        env, cmd = make_env([0])
        lam = float(cmd.time_scale_command[0])
        # P1: 录制值特征 zF != zH(两段独立录制), 斜坡会强制相等
        for tp in (0.0, .2 * lam, .5 * lam, .79 * lam):
            cmd.t_phase[:] = tp
            _, zF, _, zH = reference.get_body_reference(env)
            self.assertNotAlmostEqual(float(zF[0]), float(zH[0]), places=4,
                                      msg="P1 的 z 不应被斜坡覆盖(此时 zF/zH 应各取录制值)")
        # P2 全段: 名义末端(0.15λ) + 超时等待段(t_phase 超过 0.15λ, source_t 冻结在 0.95)。
        # 注意 P2 名义时长是 0.15λ(λ=3 时 = 0.45s), 不是 0.15s。
        cmd.phase[:] = 1
        p2_end = .15 * lam
        for tp in (0.0, .5 * p2_end, p2_end, p2_end + .30, p2_end + .50 - 1e-3):
            cmd.t_phase[:] = tp
            _, zF, _, zH = reference.get_body_reference(env)
            self.assertNotAlmostEqual(float(zF[0]), T.STAND_GROUND_HEIGHT, places=4,
                                      msg=f"P2 的 t_phase={tp} 不应被斜坡覆盖")
            if tp >= p2_end - 1e-6:      # 名义末端与等待段: source_t 冻结在 0.95
                self.assertAlmostEqual(float(zF[0]), 0.04451, places=3)
                self.assertAlmostEqual(float(zH[0]), 0.05025, places=3)
        # P3 才是斜坡: 入口 = 趴平高度, 末端 = 站立目标
        cmd.phase[:] = 2
        cmd.t_phase[:] = 0.
        _, zF3, _, zH3 = reference.get_body_reference(env)
        self.assertAlmostEqual(float(zF3[0]), T.STAND_GROUND_HEIGHT, places=5)
        self.assertAlmostEqual(float(zH3[0]), T.STAND_GROUND_HEIGHT, places=5)
        cmd.t_phase[:] = T.STAND_TRANSITION_DURATION * lam
        _, zF3b, _, zH3b = reference.get_body_reference(env)
        self.assertAlmostEqual(float(zF3b[0]), T.STAND_TARGET_HEIGHT, places=5)
        self.assertAlmostEqual(float(zH3b[0]), T.STAND_TARGET_HEIGHT, places=5)

    def test_leg_target_cost_is_p3_gated_and_pre_clip(self):
        # P3 腿部目标跟踪代价: 只在 P3 生效; 且比较的是**限幅前**的目标角(否则超限指令
        # 会被 MuJoCo 裁到同一位置、彼此不可区分, 这正是它要解决的病)。
        env, cmd = make_env([0, 1, 2])
        action = NS(raw_action=torch.zeros(3, 14), scale=1., offset=0.,
                    target_names=list(_ACTUATED_JOINT_NAMES))
        env.action_manager = NS(get_term=lambda _: action)
        with patch.object(rewards, 'get_reference_joint_state',
                          return_value=(torch.zeros(3, 14), torch.zeros(3, 14))):
            with patch.dict(_CURVES, weight_leg_target=(1.,)):
                # 目标全 0 = 参考 -> 代价 0
                out = rewards.compute_leg_target_cost(env)
                torch.testing.assert_close(out, torch.tensor([0., 0., 0.]))
                # 只给 P3(phase==2) 的环境一个偏大的腿目标
                leg_col = action.target_names.index('HL_hip_joint')
                action.raw_action[2, leg_col] = 1.5
                out = rewards.compute_leg_target_cost(env)
                # env2 在 P3 且腿目标偏离 -> 负值; env0/env1 不在 P3 -> 恒 0
                self.assertLess(out[2].item(), 0.)
                self.assertEqual(out[0].item(), 0.)
                self.assertEqual(out[1].item(), 0.)
                # 均值除以 8 条腿
                self.assertAlmostEqual(out[2].item(), -1.5 ** 2 / 8, places=6)
                # 超出 ctrlrange 的指令必须被计入(限幅前口径): HL_hip 上限 0.8
                action.raw_action[2, leg_col] = 5.0
                self.assertAlmostEqual(rewards.compute_leg_target_cost(env)[2].item(),
                                       -5.0 ** 2 / 8, places=6)

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
        # 窗口推进入口是 stand_reward_and_pulse (奖励项调用), 不再是终止项。
        cmd._update_dt = env.step_dt
        for _ in range(149):
            self.assertFalse(cmd.stand_reward_and_pulse()[3].any())
        torch.testing.assert_close(cmd.stand_reward_and_pulse()[3], torch.tensor([False, True]))
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

    def test_early_arrival_advances_only_at_segment_end(self):
        # 早侧地板已删除: 0.30λ=0.90 摆出 S1 会正常起算确认(不再被拦), 但推进仍必须等
        # 参考播完门 0.80λ=2.40 —— 于是"提前到达并保持"的达成时刻恒为 2.40, 与恰好到点
        # 达成者同刻推进。真正的惩罚在里程碑时间质量核(按真实到达时刻打折), 见下一测试。
        _, cmd = make_env([0])
        cmd.test_heights[:] = .024
        cmd.test_u[:] = torch.tensor([-1., 1.])
        for _ in range(10):                              # 0.10s -> 确认已成立
            cmd._update_command()
        self.assertAlmostEqual(cmd._s1_confirm_elapsed.item(), .10, places=4)
        self.assertEqual(cmd.phase.item(), 0)            # 但不得推进
        self.assertFalse(cmd._last_retry_mask.any())
        self.assertAlmostEqual(cmd._s1_criterion_first.item(), .01, places=4)   # 首步即到达(时钟先加 dt)
        for _ in range(230):
            cmd._update_command()
        self.assertAlmostEqual(cmd.t_phase.item(), 2.40, places=4)
        cmd._update_command()
        self.assertEqual(cmd.phase.item(), 1)            # 播完才推进
        self.assertTrue(cmd.s1_milestone_pulse.item())
        # 晚侧余量: 一直不达成 -> 过 0.80λ+0.50=2.90 才作废。
        _, c2 = make_env([0])
        c2.test_u[:] = torch.tensor([1., 1.])            # 一直不是 S1 候选
        c2.t_phase[:] = .80 * 3. + .50 + .01
        c2._update_command()
        torch.testing.assert_close(c2.retry, torch.tensor([1]))
        torch.testing.assert_close(c2._last_retry_mask, torch.tensor([True]))

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
            # dev 是相对名义段末的偏差; 核的输入是"真实首次到达时刻", 所以反解成时刻写入。
            cmd._last_s1_milestone = torch.tensor([True])
            cmd._s1_criterion_first = torch.tensor([.80 * 3. + dev])
            return rewards.compute_s1_milestone_reward(env).item() * env.step_dt

        nominal = full_reward_at(0.)
        torch.testing.assert_close(torch.tensor(nominal), torch.tensor(w))
        early = full_reward_at(0.30 * 3. - 0.80 * 3.)      # 0.30λ 真实到达
        self.assertLess(early, nominal * .25)

    def test_arrival_time_kernel_penalizes_early_completion(self):
        # 核必须对"真实到达时刻"敏感, 而不是被"参考播完门"夹住 —— 后者会让它恒等于 1.0。
        env, cmd = make_env([0])
        env.step_dt = .01
        cmd.test_heights[:] = .024
        cmd.test_u[:] = torch.tensor([-1., 1.])
        w = _CURVES['weight_milestone_s1'][0]
        lam = cmd.time_scale_command
        onset = cmd._s1_criterion_first
        # (a) 真实到达 0.30λ -> 核打到 1/5 以下
        cmd._last_s1_milestone = torch.tensor([True])
        cmd._s1_criterion_first = torch.tensor([.30 * 3.])
        r_early = rewards.compute_s1_milestone_reward(env).item() * env.step_dt
        self.assertLess(r_early, w * .25)
        # (b) 真实到达正好名义时刻 -> 满分
        cmd._s1_criterion_first = torch.tensor([.80 * 3.])
        r_nominal = rewards.compute_s1_milestone_reward(env).item() * env.step_dt
        torch.testing.assert_close(torch.tensor(r_nominal), torch.tensor(w))
        # (c) 关键回归: 若 dev 由"被门夹住的确认起算时刻"给出(旧实现), 核会恒等于 1.0。
        #     这里断言 dev 确实来自 criterion_first, 而不是 onset。
        self.assertAlmostEqual(cmd.s1_dev_early.item(), .80 * 3. - .80 * 3., places=5)
        del onset, lam

    def test_first_arrival_is_latched_once_per_attempt(self):
        # [P1 绕过路径 1] 提前达成 -> 短暂跨出判据 -> 重新进入, 不得把首次到达记录覆盖成
        # 临近段末的时刻。旧实现每次上升沿都覆盖 _s1_criterion_first, 于是核从 0.019 跳到 0.993。
        env, cmd = make_env([0])
        env.step_dt = .01
        cmd.test_heights[:] = .024
        cmd.test_u[:] = torch.tensor([-1., 1.])
        cmd._update_command()                              # 首步即达成 -> 记录在 0.01
        self.assertAlmostEqual(cmd._s1_criterion_first.item(), .01, places=4)
        # 短暂跨出判据, 再于 2.30s 重新进入
        cmd.test_u[:] = torch.tensor([1., 1.])             # 不再是 S1 候选
        cmd.t_phase[:] = 2.20
        cmd._update_command()
        cmd.test_u[:] = torch.tensor([-1., 1.])
        cmd.t_phase[:] = 2.30
        cmd._update_command()
        # 记录必须仍是 0.01, 不得被覆盖成 2.30
        self.assertAlmostEqual(cmd._s1_criterion_first.item(), .01, places=4)
        q = _milestone_time_quality(cmd.s1_dev_early, cmd.s1_dev_late, cmd.time_scale_command)
        self.assertLess(q.item(), .05)                     # 而不是 0.99

    def test_retry_reopens_first_arrival_latch(self):
        # [P1 绕过路径 2] 重试后必须能重新锁存首次达成, 否则记录一直是 NaN,
        # 而旧实现把 NaN 当零偏差 -> 满分。现在 NaN 一律给 0(见 quality kernel),
        # 且重试会清 latch, 使"候选跨重试持续成立"也能重新记录。
        env, cmd = make_env([0])
        env.step_dt = .01
        cmd.test_heights[:] = .024
        cmd.test_u[:] = torch.tensor([-1., 1.])            # 候选持续成立, 跨重试不清零
        cmd.t_phase[:] = .80 * 3. + .50                    # 正好到 P1 窗界
        cmd._update_command()                              # 这一步触发重试(未确认)
        self.assertEqual(cmd.retry.item(), 1)
        self.assertTrue(torch.isnan(cmd._s1_criterion_first).item())
        cmd._update_command()                              # 重试后重新锁存
        self.assertFalse(torch.isnan(cmd._s1_criterion_first).item())

    def test_nan_arrival_record_never_pays_full_milestone(self):
        # 记录缺失时不得静默按零偏差发满额: 脉冲为真但 dev 为 NaN -> 核必须为 0。
        env, cmd = make_env([0])
        env.step_dt = .01
        nan = torch.tensor([float('nan')])
        q = _milestone_time_quality(nan, nan, torch.tensor([3.]))
        self.assertEqual(q.item(), 0.)
        cmd._last_s1_milestone = torch.tensor([True])
        cmd._s1_criterion_first = nan                      # 记录缺失
        reward = rewards.compute_s1_milestone_reward(env).item()
        self.assertEqual(reward, 0.)

    def test_segment_end_gate_holds_advance(self):
        # 推进门仍然必须等参考播完 —— 它保证参考连续, 与"早到是否受罚"是两件事。
        env, cmd = make_env([0])
        cmd.test_heights[:] = .024
        cmd.test_u[:] = torch.tensor([-1., 1.])
        for _ in range(10):
            cmd._update_command()
        self.assertAlmostEqual(cmd._s1_confirm_elapsed.item(), .10, places=4)   # 确认已成立
        self.assertEqual(cmd.phase.item(), 0)              # 但段末未到, 不得推进
        for _ in range(230):
            cmd._update_command()
        cmd._update_command()
        self.assertEqual(cmd.phase.item(), 1)
        self.assertTrue(cmd.s1_milestone_pulse.item())

    def test_cycle_reset_is_isolated_to_selected_envs(self):
        # 只复位凑满站立窗口的环境; 其他环境的阶段、时钟与累计量必须原封不动。
        env, cmd = make_env([2, 2])
        cmd.test_u[1, 0] = -1.                            # env1 姿态掉出 -> 窗口清零
        cmd.phase[:] = 2
        cmd.t_phase[:] = torch.tensor([.20, .35])
        cmd._stand_elapsed[0] = 1.5
        cmd._stand_vel_integral[0] = 1.5 * 0.3
        cmd._update_dt = 0.
        cmd.stand_reward_and_pulse()                       # 不推进窗口, 只取上一步已攒的状态
        self.assertTrue(cmd._pending_cycle_reset[0].item())
        self.assertFalse(cmd._pending_cycle_reset[1].item())
        cmd._update_command()
        # env0 被复位: 阶段归零, 且 sim/scene/obs/act 都只收到 [0]
        self.assertEqual(cmd.phase.tolist(), [0, 2])
        self.assertEqual(cmd.t_phase[0].item(), 0.)
        self.assertAlmostEqual(cmd.t_phase[1].item(), .35, places=5)
        for key in ('sim', 'scene', 'obs', 'act'):
            self.assertEqual(env.reset_calls[key], [[0]], f"{key} 复位范围错误")

    def test_cycle_completion_settles_exactly_once(self):
        # 完成脉冲恰好结算一次: 首次读到 True, 随后必须为空 (不重复发钱)。
        env, cmd = make_env([2])
        cmd.phase[:] = 2
        cmd._stand_elapsed[0] = 1.5
        cmd._stand_vel_integral[0] = 0.
        cmd._update_dt = 0.
        cmd.stand_reward_and_pulse()
        self.assertTrue(cmd.consume_cycle_end_pulse()[0].item())
        self.assertFalse(cmd.consume_cycle_end_pulse().any())
        # 完成奖励只能结算一次: 窗口在复位前一直保持"已攒满", 必须有"每循环一次"的锁,
        # 否则完成帧与复位帧之间会反复结算同一个循环。
        env2, cmd2 = make_env([2])
        cmd2._stand_elapsed[0] = 1.5
        cmd2._stand_vel_integral[0] = 0.
        cmd2._update_dt = 0.                              # 不推进窗口, 直接结算已攒满的状态
        with patch.dict(_CURVES, weight_milestone_success=(35.,)):
            paid = rewards.compute_task_success_milestone_reward(env2)
            self.assertGreater(paid.item(), 0.)
            again = rewards.compute_task_success_milestone_reward(env2)
            self.assertEqual(again.item(), 0.)
            # 下一步执行复位后, 下一个循环重新可以结算。
            cmd2._update_command()
            self.assertEqual(cmd2._stand_elapsed.item(), 0.)
            self.assertFalse(cmd2._pending_cycle_reset.any())

    def test_cycle_reset_does_not_end_episode_and_reopens_milestones(self):
        # 关键语义: 循环复位不是回合结束 —— episode_length_buf 不动、奖励累计不清,
        # 且下一个循环的 S1/S2/完成里程碑重新有效。
        env, cmd = make_env([2])
        env.episode_length_buf = torch.tensor([137])
        cmd._s1_awarded = torch.tensor([True])
        cmd._s2_awarded = torch.tensor([True])
        cmd._stand_elapsed[0] = 1.5
        cmd._update_dt = 0.
        cmd.stand_reward_and_pulse()
        cmd._update_command()
        self.assertEqual(env.episode_length_buf.item(), 137)        # 回合没结束
        self.assertFalse(cmd._s1_awarded.item())                     # 里程碑锁存已重开
        self.assertFalse(cmd._s2_awarded.item())
        self.assertFalse(cmd._pending_cycle_reset.any())             # 待复位标志已消费
        self.assertEqual(cmd._stand_elapsed.item(), 0.)              # 站立窗口已清零

    def test_cycle_reset_writes_all_state_before_forward(self):
        # 物理一致性: 写入 qpos/qvel 之后必须 forward, 否则观测里的 site/body 派生量
        # 仍是复位前状态 (entity/data.py 明确要求"写后读前先 forward")。
        from mjlab.tasks.SQuRo_Backup.mdp import events as ev
        env, cmd = make_env([2])
        order: list[str] = []
        env.sim = NS(reset=lambda ids: order.append('sim.reset'), forward=lambda: order.append('sim.forward'))
        env.scene.reset = lambda ids: order.append('scene.reset')
        env.observation_manager.reset = lambda ids: order.append('obs.reset')
        env.action_manager.reset = lambda ids: order.append('act.reset')
        with patch.object(ev, 'apply_fallen_state', lambda e, i: order.append('fallen')):
            cmd._stand_elapsed[0] = 1.5
            cmd._update_dt = 0.
            cmd.stand_reward_and_pulse()
            cmd._update_command()
        # 仰卧初态必须写在 forward 之前, 否则派生量不刷新
        self.assertLess(order.index('fallen'), order.index('sim.forward'))
        # 且 forward 必须是最后一步
        self.assertEqual(order[-1], 'sim.forward')
        self.assertEqual(order[0], 'sim.reset')

    def test_stand_window_writes_calibration_logs(self):
        # 仪表守卫: 判据量本身必须被记录 —— §7.2.2 要求"先看 stand_mean_vel 离门限多远,
        # 再决定调阈值还是改结构"。这几个量曾在重写 terminations 时被漏掉, 只能盲猜。
        env, cmd = make_env([2, 1])
        log = {}
        env.extras['log'] = log
        cmd.test_vel[:] = torch.tensor([2.0, 0.0])        # env0 抖动, env1 静止
        cmd._stand_elapsed[0] = 1.6
        cmd._stand_vel_integral[0] = 1.6 * 2.0
        cmd._update_dt = .01
        cmd.stand_reward_and_pulse()
        for key in ("Progress/standing", "Progress/stand_hold", "Progress/stand_mean_vel"):
            self.assertIn(key, log, f"{key} 必须被记录")
        # stand_mean_vel 是判据量本身: env0 的窗口均值应为约 2.0
        self.assertAlmostEqual(log["Progress/stand_mean_vel"], 2.0, places=3)
        # stand_hold 是所有环境的均值: env1 从未建窗, 故约为 env0(1.6+dt) 的一半
        self.assertGreater(log["Progress/stand_hold"], .8)
        self.assertLess(log["Progress/stand_hold"], .82)

    def test_full_episode_reset_clears_cycle_state(self):
        # [P2] 完整回合重置必须清循环级状态: 旧实现只动 env._stand_*, 且 _resample_command
        # 不清 _pending_cycle_reset -> 成功与超时同帧时, 重置后仍会多执行一次部分复位。
        env, cmd = make_env([2])
        cmd._stand_elapsed[0] = 1.5
        cmd._stand_vel_integral[0] = .4
        cmd._update_dt = 0.
        cmd.stand_reward_and_pulse()
        self.assertTrue(cmd._pending_cycle_reset[0].item())
        env.reset_buf = torch.tensor([True])
        cmd._update_metrics()                                      # 快照回合重置掩码
        self.assertTrue(cmd._pending_episode_reset[0].item())
        cmd._update_command()
        self.assertFalse(cmd._pending_cycle_reset.any())
        self.assertFalse(cmd._last_cycle_end_pulse.any())
        self.assertEqual(cmd._stand_elapsed.item(), 0.)
        self.assertEqual(cmd._stand_vel_integral.item(), 0.)
        self.assertEqual(cmd._cycles_this_episode.item(), 0)      # 上报之后才归零

    def test_completion_event_survives_until_consumed(self):
        # [P1 回归] _apply_cycle_reset 不得把"本步完成事件"清掉 —— 旧实现写完立刻被
        # _clear_cycle_state 清成 False, 导致 Cycle/* 恒为 0、回放判不出 DONE。
        env, cmd = make_env([2])
        cmd._stand_elapsed[0] = 1.5
        cmd._stand_vel_integral[0] = 0.
        cmd._update_dt = 0.
        cmd.stand_reward_and_pulse()
        cmd._update_command()                                      # 执行循环复位
        self.assertEqual(cmd.phase.item(), 0)                      # 确实复位了
        self.assertTrue(cmd.cycle_completed_pulse[0].item())       # 但完成事件仍在
        # 消费者读过之后才被清
        log = {}
        cmd._env.extras['log'] = log
        cmd._update_metrics()
        self.assertEqual(log['Cycle/completed'], 1.0)
        self.assertEqual(log['Cycle/per_episode'], 1.0)            # 一次成功 -> 计数 0→1
        self.assertFalse(cmd.cycle_completed_pulse.any())          # 已消费

    def test_one_completion_increments_counter_exactly_once(self):
        # 每个循环恰好计一次: 完成事件被消费后不得再计。
        env, cmd = make_env([2])
        log = {}
        env.extras['log'] = log
        cmd._stand_elapsed[0] = 1.5
        cmd._stand_vel_integral[0] = 0.
        cmd._update_dt = 0.
        cmd.stand_reward_and_pulse()
        cmd._update_command()
        cmd._update_metrics()
        self.assertEqual(cmd._cycles_this_episode.item(), 1)
        for _ in range(5):                                          # 无新完成 -> 不再增加
            cmd._update_metrics()
        self.assertEqual(cmd._cycles_this_episode.item(), 1)
        self.assertEqual(log['Cycle/completed'], 0.0)

    def test_timeout_reports_final_cycle_count_before_clearing(self):
        # [P2] 超时回合: 框架顺序是 奖励 → 完整回合重置 → command.compute → _update_metrics,
        # 所以计数必须在 metrics 里先上报、再归零 —— 否则超时回合读成 0。
        env, cmd = make_env([2])
        log = {}
        env.extras['log'] = log
        cmd._cycles_this_episode[0] = 3
        env.reset_buf = torch.tensor([True])                       # 本步超时
        cmd._update_metrics()
        self.assertEqual(log['Cycle/per_episode'], 3.0)            # 上报的是最终次数
        cmd._update_command()
        self.assertEqual(cmd._cycles_this_episode.item(), 0)       # 之后才归零

    def test_deviation_log_matches_reward_input(self):
        # [P2] 日志与奖励核必须同源: 旧日志读的是会被后续候选覆盖的缓冲,
        # 把"严重提前"显示成"接近准时"(实测 -2.39s 显示成 -0.09s)。
        env, cmd = make_env([0])
        env.step_dt = .01
        log = {}
        env.extras['log'] = log
        cmd.test_heights[:] = .024
        cmd.test_u[:] = torch.tensor([-1., 1.])
        cmd._update_command()                                      # 首步即达成 -> 0.01
        cmd.test_u[:] = torch.tensor([1., 1.])                     # 离开
        cmd.t_phase[:] = 2.20
        cmd._update_command()
        cmd.test_u[:] = torch.tensor([-1., 1.])                    # 重新进入
        cmd.t_phase[:] = 2.30
        cmd._update_command()
        cmd._update_metrics()
        # 奖励核用的偏差
        reward_dev = cmd.s1_dev_early.item()
        self.assertAlmostEqual(reward_dev, .01 - .80 * 3., places=4)
        # 日志必须给出同一个数, 不得显示成 -0.09
        self.assertAlmostEqual(log['Progress/s1_dev_s'], reward_dev, places=4)
        # 首次到达时刻也必须是 0.01
        self.assertAlmostEqual(log['Data/s1_first_s'], .01, places=4)

    def test_lifecycle_boundaries_are_distinct(self):
        # 三种边界的生命周期必须可区分:
        # 重试(清 latch) / 循环复位(清 latch 与窗口, 保留完成事件与循环数) / 回合重置(全清)。
        env, cmd = make_env([0])
        cmd._s1_cycle_latched[0] = True
        cmd._clear_cycle_state(torch.tensor([0]))
        self.assertFalse(cmd._s1_cycle_latched.item())
        # 循环复位保留完成事件
        cmd._last_cycle_reset[0] = True
        cmd._cycles_this_episode[0] = 4
        cmd._clear_cycle_state(torch.tensor([0]))
        self.assertTrue(cmd._last_cycle_reset.item())
        self.assertEqual(cmd._cycles_this_episode.item(), 4)

    def test_joint_track_cost_is_order_invariant(self):
        # [P2] 实际关节/参考/权重必须同序。旧实现把实际关节按 actions 名称顺序重排,
        # 而权重也按 actions 顺序构造 -> 名称反序时代价从 0 变成 0.239。
        import torch as _t
        from mjlab.tasks.SQuRo_Backup.mdp import rewards as R
        n = 2
        ref = _t.zeros(n, 14)
        env, cmd = make_env([0, 0])
        env.step_dt = .01
        saved = _MODEL_INDICES.joint_ids
        _MODEL_INDICES.joint_ids = tuple(range(14))
        try:
            # 只替换 data, 保留 write_*/find_*/num_joints —— 否则会破坏后续测试(它们是同一个 env)
            env.make_robot.data = NS(joint_pos=ref.clone())
            env.scene['robot'] = env.make_robot
            with patch.object(R, 'get_reference_joint_state', return_value=(ref.clone(), ref.clone())):
                cost = R.compute_joint_track_cost(env)
        finally:
            _MODEL_INDICES.joint_ids = saved
        # 完全等于参考 -> 代价必须是 0, 与动作项列序无关
        torch.testing.assert_close(cost, _t.zeros(n))
        # 权重向量按参考表顺序构造: 脊柱位为 1.57、颈位为 0.3、其余 1.0
        w = R._joint_group_weights(_t.device('cpu'), _t.float32)
        self.assertAlmostEqual(w[0].item(), R.TRACK_W_SPN, places=6)    # F_spine1
        self.assertAlmostEqual(w[1].item(), R.TRACK_W_SPN, places=6)    # F_body
        self.assertAlmostEqual(w[2].item(), R.TRACK_W_NECK, places=6)   # Neck_yaw
        self.assertAlmostEqual(w[4].item(), R.TRACK_W_LEG, places=6)    # FL_shoulder
        self.assertAlmostEqual(w[8].item(), R.TRACK_W_SPN, places=6)    # H_spine1


if __name__ == '__main__':
    unittest.main()
