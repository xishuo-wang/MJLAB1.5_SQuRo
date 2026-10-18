# SQuRo_Slalom 参考路径一致性回归；不创建环境、不修改训练数据。
# 覆盖: 局部重置隔离、参考推进与自身步频一致、gait=1 几何不变、命令计时器、
#       固定间距的单一真源、一阶段预算与最大形变、腿部参考过零连续。
import math
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS

import torch

from mjlab.tasks.SQuRo_Slalom.mdp import reference as ref_mod
from mjlab.tasks.SQuRo_Slalom.mdp.command import SlalomCommand, SlalomCommandCfg
from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import (
    _STEPS_PER_ITER,
    BASE_VEL,
    CURVATURE_TARGET_MAX,
    PHASE1_END_ITER,
    PHASE1_MID_ITER,
    PHASE2_END_ITER,
    POLE_SPACING_START,
    VEL_MIN, get_velocity_scale,
)
from mjlab.tasks.SQuRo_Slalom.mdp.events import reset_model
from mjlab.tasks.SQuRo_Slalom.mdp.indices import (
    _ACT_LEG_IDS, _ACT_FL_LEG_IDS, _ACT_FR_LEG_IDS, _ACT_HL_LEG_IDS, _ACT_HR_LEG_IDS,
    _ACT_F_SPINE1_ID, _ACT_H_SPINE1_ID, _ACT_NECK_YAW_ID,
    _ACT_F_BODY_ID, _ACT_H_BODY_ID, _ACT_NECK_PITCH_ID,
)
from mjlab.tasks.SQuRo_Slalom.mdp.path import (
    compute_slalom_path_ref,
    get_approach_start,
    get_effective_pole_spacing,
)
from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import get_pole_spacing_bounds
from mjlab.tasks.SQuRo_Slalom.mdp.slalom_geometry import curvature_from_spacing, get_slalom_geometry, get_timed_slalom_tables

# 几何黄金数据仍采用历史时间步，真实频率由端到端自检检查。
STEP_DT = 0.02
# Phase 1 期间的 common_step_counter (绕杆模式生效)
PHASE1_STEP = PHASE1_END_ITER * _STEPS_PER_ITER
# 腿部关节列 (FL/FR/HL/HR) 与内侧腿列
_LEG_COLS = _ACT_LEG_IDS
_LEFT_COLS = _ACT_FL_LEG_IDS + _ACT_HL_LEG_IDS
_RIGHT_COLS = _ACT_FR_LEG_IDS + _ACT_HR_LEG_IDS
_TABLE_RES = 50

# 为桩命令绑定当前整数曲率对应的逐环境几何。
def bind_geometry(cmd, n, spacing=None):
    peak = 15 if spacing is None else curvature_from_spacing(spacing)
    cmd.slalom_curvature_index = torch.full((n,), peak - 5, dtype=torch.long)
    geometry = get_slalom_geometry(peak)
    cmd.pole_spacing = torch.full((n,), geometry["spacing"])
    cmd.pole_y = torch.full((n,), geometry["pole_y"])


# 构造只含参考计算所需字段的桩环境
def make_ref_env(gait_freqs, t_secs, spacing=POLE_SPACING_START, counter=PHASE1_STEP,
                 gait_scalar=None, base_vel=BASE_VEL):
    n = len(gait_freqs)
    cmd = object.__new__(SlalomCommand)
    env = NS(
        device=torch.device("cpu"),
        num_envs=n,
        step_dt=STEP_DT,
        episode_length_buf=torch.tensor(
            [round(t / STEP_DT) for t in t_secs], dtype=torch.long
        ),
        common_step_counter=counter,
    )
    cmd._env = env
    cfg = SlalomCommandCfg()
    cfg.fixed_pole_spacing = spacing
    cfg.fixed_velocity = base_vel
    cmd.cfg = cfg
    # 生产代码在 __init__ 里从 cfg 拷贝的固定值 + 命令视图, 桩环境需手动补齐
    cmd.fixed_velocity = None
    cmd.fixed_height_f = None
    cmd.fixed_height_h = None
    cmd.fixed_gait_freq = None
    cmd.fixed_curvature = None
    cmd.fixed_pole_spacing = spacing
    bind_geometry(cmd, n, getattr(cmd.cfg, "fixed_pole_spacing", None))
    cmd.command_tensor = torch.zeros(n, 5)
    cmd.command_tensor[:, 3] = torch.tensor(gait_freqs, dtype=torch.float32)
    cmd.vel_command = cmd.command_tensor[:, 0]
    cmd.height_f_command = cmd.command_tensor[:, 1]
    cmd.height_h_command = cmd.command_tensor[:, 2]
    cmd.gait_freq_command = cmd.command_tensor[:, 3]
    cmd.curvature_command = cmd.command_tensor[:, 4]
    cmd._start_recorded = torch.zeros(n, dtype=torch.bool)
    env.command_manager = NS(_terms={"slalom_cmd": cmd})
    # 旧实现从全局槽读步频; 桩里默认模拟"最后重置者"写入
    env._slalom_gait_scalar = float(gait_freqs[-1] if gait_scalar is None else gait_scalar)
    env._slalom_base_vel_scalar = base_vel
    return env


# 构造只含计时器所需字段的桩命令项
def make_timer_cmd(n=4, time_left=25.0):
    cmd = object.__new__(SlalomCommand)
    env = NS(device=torch.device("cpu"), num_envs=n, step_dt=STEP_DT,
             common_step_counter=0,
             episode_length_buf=torch.full((n,), 10, dtype=torch.long))
    cmd._env = env
    cmd.cfg = SlalomCommandCfg()
    cmd.time_left = torch.full((n,), float(time_left))
    cmd.command_counter = torch.zeros(n, dtype=torch.long)
    cmd.metrics = {}
    cmd._update_metrics = lambda: None
    cmd._update_command = lambda: None
    bind_geometry(cmd, n)
    cmd.command_tensor = torch.zeros(n, 5)
    cmd.command_tensor[:, 3] = 1.0
    cmd.fixed_velocity = None
    cmd.vel_command = cmd.command_tensor[:, 0]
    cmd.gait_freq_command = cmd.command_tensor[:, 3]
    cmd.curvature_command = cmd.command_tensor[:, 4]
    env._path_kappa = cmd.command_tensor[:, 4]
    env.command_manager = NS(_terms={"slalom_cmd": cmd})
    return cmd


# 参考路径解算 (返回顺序为 x, y, vx_des, vy_des, heading)
def ref_of(env):
    x, y, _, _, heading = compute_slalom_path_ref(env)
    return x, y, heading


# 构造只含参考关节表所需字段的桩环境 (Phase 0 语义: 曲率取静态命令值)
def make_ref_state_env(curvatures, phases=None, gait=1.0, step_count=5):
    n = len(curvatures)
    cmd = object.__new__(SlalomCommand)
    env = NS(
        device=torch.device("cpu"),
        num_envs=n,
        step_dt=STEP_DT,
        common_step_counter=0,
        episode_length_buf=torch.full((n,), step_count, dtype=torch.long),
        reset_terminated=torch.zeros(n, dtype=torch.bool),
        scene={"robot": None},
    )
    cmd._env = env
    cmd.cfg = SlalomCommandCfg()
    bind_geometry(cmd, n)
    cmd.command_tensor = torch.zeros(n, 5)
    cmd.command_tensor[:, 3] = gait
    cmd.command_tensor[:, 4] = torch.tensor(curvatures, dtype=torch.float32)
    cmd._shared_gait_freq = gait
    env.command_manager = NS(_terms={"slalom_cmd": cmd})
    if phases is None:
        env._ref_phase = torch.zeros(n)
    else:
        env._ref_phase = torch.tensor(phases, dtype=torch.float32)
    return env


# 取一条腿在一个步态周期内的关节行程 (max-min), 用于判断内侧腿是否被压缩
def leg_travel(kappa, cols):
    n = _TABLE_RES
    env = make_ref_state_env([kappa] * n, phases=[i / (n - 1) for i in range(n)])
    pos, _ = ref_mod.get_reference_joint_state(env)
    return (pos[:, cols].max(dim=0).values - pos[:, cols].min(dim=0).values)


class TestSlalomRefConsistency(unittest.TestCase):

    # 替换模型索引解析, 桩环境没有真实实体
    def setUp(self):
        self._orig_resolve = ref_mod.resolve_model_indices
        ref_mod.resolve_model_indices = lambda entity: None

    def tearDown(self):
        ref_mod.resolve_model_indices = self._orig_resolve

    # 局部重置 (某个环境重抽步频) 不得改变其他环境的参考
    def test_local_reset_does_not_move_others(self):
        env = make_ref_env([1.0, 1.0], [2.0, 2.0], gait_scalar=1.0)
        x0, y0, h0 = ref_of(env)

        # 模拟 env1 重置: 步频改为 2.0, 全局槽随之被覆盖
        env.command_manager._terms["slalom_cmd"].command_tensor[:, 3] = torch.tensor(
            [1.0, 2.0]
        )
        env._slalom_gait_scalar = 2.0
        x1, y1, h1 = ref_of(env)

        self.assertAlmostEqual(x1[0].item(), x0[0].item(), places=6,
                               msg="未重置环境的参考 x 被其他环境的重置改动")
        self.assertAlmostEqual(y1[0].item(), y0[0].item(), places=6,
                               msg="未重置环境的参考 y 被其他环境的重置改动")
        self.assertAlmostEqual(h1[0].item(), h0[0].item(), places=6,
                               msg="未重置环境的参考朝向被其他环境的重置改动")
        # 被重置的环境自身必须发生推进 (否则说明改动没生效)
        self.assertGreater(float(torch.hypot(x1[1] - x0[1], y1[1] - y0[1])), 1e-3)

    # 参考推进必须只由自身步频决定: ref(t, gait=g) == ref(t*g, gait=1)
    def test_reference_scales_with_own_gait(self):
        # 全局槽停在 1.0 (模拟 env0 最后重置), env1 自身步频为 2.0
        env = make_ref_env([1.0, 2.0], [2.0, 2.0], gait_scalar=1.0)
        x, y, h = ref_of(env)
        env_b = make_ref_env([1.0], [4.0], gait_scalar=1.0)
        xb, yb, hb = ref_of(env_b)

        # 同一时刻两个不同步频的环境必须落在不同位置 (证明用的是各自步频)
        self.assertGreater(abs(x[1].item() - x[0].item()), 1e-4,
                           msg="不同步频的环境拿到了同一参考")
        # 步频 2.0 在 t 处 == 步频 1.0 在 2t 处
        self.assertAlmostEqual(x[1].item(), xb[0].item(), places=6,
                               msg="步频 2.0 的参考未落在 2 倍时间处")
        self.assertAlmostEqual(y[1].item(), yb[0].item(), places=6,
                               msg="步频 2.0 的参考未落在 2 倍时间处")
        self.assertAlmostEqual(h[1].item(), hb[0].item(), places=6,
                               msg="步频 2.0 的参考朝向未落在 2 倍时间处")

    # 周期边界必须连续，位移必须等于两倍杆心距。
    def test_gait1_geometry_matches_baseline(self):
        for peak in range(5, 16):
            geometry = get_slalom_geometry(peak)
            path = geometry["period"]
            self.assertAlmostEqual(float(path["x"][-1]), 2*geometry["spacing"], places=10)
            self.assertAlmostEqual(float(path["y"][-1]), 0.0, places=8)
            self.assertAlmostEqual(float(path["heading"][-1]), 0.0, places=8)
            self.assertEqual(path["kappa"][0], path["kappa"][-1])
            self.assertLessEqual(float(abs(path["kappa"]).max()), peak)

    # 命令计时器每个控制步只允许扣一次
    def test_timer_decrements_once_per_step(self):
        cmd = make_timer_cmd(time_left=25.0)
        for _ in range(100):
            cmd.compute(STEP_DT)
        # float32 累加误差 ~1e-5, 用 delta 而非 places
        self.assertAlmostEqual(cmd.time_left[0].item(), 25.0 - 100 * STEP_DT, delta=1e-3,
                               msg="time_left 每步被扣了多次")

    # 一个回合 (20 s) 内不得重采样命令, 步频必须在回合内固定
    def test_no_mid_episode_gait_resample(self):
        cmd = make_timer_cmd(time_left=25.0)
        resampled = []
        cmd._resample_command = lambda ids: resampled.append(int(len(ids)))
        for _ in range(1000):
            cmd.compute(STEP_DT)
        self.assertEqual(resampled, [],
                         msg="回合内触发了命令重采样, 步频会中途被其他批次改写")
        self.assertAlmostEqual(cmd.time_left[0].item(), 25.0 - 1000 * STEP_DT, delta=1e-3)

    # 固定杆距映射到最近整数曲率，并可反查到同一几何。
    def test_active_pole_spacing_applies_effective(self):
        for peak in range(5, 16):
            spacing = get_slalom_geometry(peak)["spacing"]
            self.assertEqual(curvature_from_spacing(spacing), peak)
            self.assertAlmostEqual(get_effective_pole_spacing(spacing), spacing, places=10)

    # 总预算必须包含三千轮转圈和五千轮绕杆。
    def test_default_budget_covers_phase0_range(self):
        from mjlab.tasks.SQuRo_Slalom.config.rl_cfg import SQuRo_Slalom_PPO_Runner_Cfg
        self.assertEqual(PHASE1_END_ITER, 3000)
        self.assertEqual(PHASE1_END_ITER - PHASE1_MID_ITER, 1000)
        self.assertEqual(SQuRo_Slalom_PPO_Runner_Cfg().max_iterations, 8000)

    # 曲率过零时腿部参考必须连续 (不允许整条腿的相位被换掉)
    def test_leg_ref_continuous_across_zero_curvature(self):
        worst = 0.0
        for i in range(20):
            phase = i / 20
            env_a = make_ref_state_env([-1e-6], [phase])
            pos_a, _ = ref_mod.get_reference_joint_state(env_a)
            env_b = make_ref_state_env([1e-6], [phase])
            pos_b, _ = ref_mod.get_reference_joint_state(env_b)
            jump = float((pos_a[0, _LEG_COLS] - pos_b[0, _LEG_COLS]).abs().max())
            worst = max(worst, jump)
        self.assertLess(worst, 0.05,
                        msg=f"κ 过零时腿部参考跳变 {worst:.4f} rad (相位被交换)")

    # 内侧腿必须随转向侧收缩, 外侧腿行程随本回合速度缩放
    def test_inner_leg_is_on_turn_side(self):
        travel_left_pos = leg_travel(CURVATURE_TARGET_MAX, _LEFT_COLS)
        travel_right_pos = leg_travel(CURVATURE_TARGET_MAX, _RIGHT_COLS)
        travel_left_neg = leg_travel(-CURVATURE_TARGET_MAX, _LEFT_COLS)
        travel_right_neg = leg_travel(-CURVATURE_TARGET_MAX, _RIGHT_COLS)

        # 左转 (κ>0): 左腿为内侧 → 行程相对外侧压缩; 右转 (κ<0) 反之
        for i in range(len(_LEFT_COLS)):
            self.assertLess(float(travel_left_pos[i]), float(travel_right_pos[i]) * 0.95,
                            msg="左转时左腿 (内侧) 行程未被压缩")
        for i in range(len(_RIGHT_COLS)):
            self.assertLess(float(travel_right_neg[i]), float(travel_left_neg[i]) * 0.95,
                            msg="右转时右腿 (内侧) 行程未被压缩")

    # 腿部参考必须落在关节范围与执行器范围的交集内
    def test_leg_reference_within_mechanical_limits(self):
        from mjlab.asset_zoo.robots.SQuRo.SQuRo_constants import get_spec
        from mjlab.tasks.SQuRo_Slalom.mdp.indices import _ACT_JOINT_NAMES
        model = get_spec().compile()
        limits = []
        for name in _ACT_JOINT_NAMES:
            joint = model.joint(name)
            actuator = model.actuator(name.removesuffix("_joint"))
            low = max(model.actuator_ctrlrange[actuator.id, 0], model.jnt_range[joint.id, 0])
            high = min(model.actuator_ctrlrange[actuator.id, 1], model.jnt_range[joint.id, 1])
            limits.append((float(low), float(high)))
        limits = torch.tensor(limits)
        ref_mod._init_tables("cpu")
        table = ref_mod._pos_table
        bins = ref_mod._CURVATURE_BINS
        index = torch.arange(len(bins))
        rows = {
            "绕杆": torch.cat([table[p, :, :p + 1] for p, peak in enumerate(bins) if peak >= 5], dim=1),
            "转圈": table[index, :, index],
        }
        for name, values in rows.items():
            excess = (values - limits[:, 1]).clamp(min=0.0) + (limits[:, 0] - values).clamp(min=0.0)
            self.assertLess(float(excess.max()), 1e-4,
                            msg=f"{name}参考超出机械限位 {float(torch.rad2deg(excess.max())):.2f}°")

    # 腿部参考角度沿相位连续, 速度有界
    def test_leg_reference_continuous_and_bounded(self):
        ref_mod._init_tables("cpu")
        table, vel_table = ref_mod._pos_table, ref_mod._vel_table
        bins = ref_mod._CURVATURE_BINS
        slalom = torch.cat([table[p, :, :p + 1] for p, peak in enumerate(bins) if peak >= 5], dim=1)
        slalom_vel = torch.cat([vel_table[p, :, :p + 1] for p, peak in enumerate(bins) if peak >= 5], dim=1)
        jump = torch.diff(slalom[..., _ACT_LEG_IDS], dim=-2).abs().max()
        self.assertLess(float(jump), 1.0, msg=f"腿部参考相位跳变 {float(jump):.3f} rad")
        self.assertLess(float(slalom_vel[..., _ACT_LEG_IDS].abs().max() * 2.0), 40.0,
                        msg="腿部参考速度超过 40 rad/s (2 Hz)")

    # 基础速度必须进入参考表: 同一峰值不同基础速度对应不同步幅
    def test_reference_tracks_base_velocity(self):
        def leg_travel(base):
            ref_mod._init_tables("cpu", base)
            row = ref_mod._pos_table[ref_mod._CURVATURE_BINS.index(15.0), :, 0]
            leg = row[:, :, _ACT_LEG_IDS]
            return float((leg.max(dim=1).values - leg.min(dim=1).values).abs().max())

        low, high = leg_travel(0.05), leg_travel(0.15)
        ref_mod._init_tables("cpu", BASE_VEL)
        # 关节角行程对足端步幅是非线性的, 只要求明显随基础速度变化
        self.assertGreater(high / max(low, 1e-6), 1.5, msg="基础速度未进入参考步幅")

    # 脊柱/颈参考不得超过模型关节与执行器限位
    def test_spine_ref_within_joint_limits(self):
        kappas = torch.linspace(-CURVATURE_TARGET_MAX, CURVATURE_TARGET_MAX, 21).tolist()
        env = make_ref_state_env(kappas, phases=[0.3] * len(kappas), gait=0.0)
        pos, _ = ref_mod.get_reference_joint_state(env)
        max_f = float(pos[:, _ACT_F_SPINE1_ID].abs().max())
        max_h = float(pos[:, _ACT_H_SPINE1_ID].abs().max())
        max_yaw = float(pos[:, _ACT_NECK_YAW_ID].abs().max())
        self.assertLessEqual(max_f, 0.6 + 1e-5, msg=f"F_spine1 参考超出限位: {max_f:.4f}")
        self.assertLessEqual(max_h, 0.6 + 1e-5, msg=f"H_spine1 参考超出限位: {max_h:.4f}")
        self.assertLessEqual(max_yaw, 0.8 + 1e-5, msg=f"Neck_yaw 参考超出限位: {max_yaw:.4f}")
        # 高曲率处必须真的贴到限位 (否则说明 clamp 没生效)
        self.assertAlmostEqual(max_f, 0.6, places=4)
        self.assertAlmostEqual(max_h, 0.6, places=4)

    # 新上限正负二十对应原最大曲率的完整脊柱和颈部参考。
    def test_max_curvature_keeps_full_deformation(self):
        env = make_ref_state_env([-20.0, 20.0], gait=0.0)
        pos, _ = ref_mod.get_reference_joint_state(env)
        columns = (_ACT_F_SPINE1_ID, _ACT_F_BODY_ID, _ACT_H_SPINE1_ID,
                   _ACT_H_BODY_ID, _ACT_NECK_YAW_ID, _ACT_NECK_PITCH_ID)
        expected = torch.tensor([[0.6, 0.9, -0.6, 0.7, -0.8, -0.3],
                                 [-0.6, -0.9, -0.6, -0.7, 0.8, -0.3]])
        torch.testing.assert_close(pos[:, columns], expected)

    # 速度参考必须等于位置参考的差分 (位置动而速度说不动即为不一致)
    def test_ref_vel_matches_position_difference(self):
        # 冻结相位 (gait=0) 以隔离"曲率变化"这一项; |κ| 取在脊柱限位以内
        env = make_ref_state_env([-0.8 * CURVATURE_TARGET_MAX], phases=[0.3], gait=0.0)
        pos_a, _ = ref_mod.get_reference_joint_state(env)
        cmd = env.command_manager._terms["slalom_cmd"]
        cmd.command_tensor[:, 4] = -0.816 * CURVATURE_TARGET_MAX  # 同一曲率区间内差分
        env.common_step_counter = 1
        env.episode_length_buf = torch.tensor([6])
        pos_b, vel_b = ref_mod.get_reference_joint_state(env)
        expected = (pos_b - pos_a) / STEP_DT
        diff = (vel_b - expected).abs().max()
        # 峰值轴与瞬时轴双线性插值的交叉项会带来千分级残差
        self.assertLess(float(diff), 1e-2,
                        msg=f"速度参考与位置参考差分不符, 最大差 {float(diff):.4f} rad/s")
        # 脊柱确实在动 (排除"两边都是 0"的假通过)
        self.assertGreater(float(vel_b[:, _ACT_F_SPINE1_ID].abs().max()), 1e-3)
        # 被限位截住时导数必须为 0 (与位置恒定一致)
        env_hi = make_ref_state_env([-0.96 * CURVATURE_TARGET_MAX], phases=[0.3], gait=0.0)
        pos_c, _ = ref_mod.get_reference_joint_state(env_hi)
        env_hi.command_manager._terms["slalom_cmd"].command_tensor[:, 4] = -0.976 * CURVATURE_TARGET_MAX
        env_hi.common_step_counter = 1
        env_hi.episode_length_buf = torch.tensor([6])
        pos_d, vel_d = ref_mod.get_reference_joint_state(env_hi)
        self.assertLess(float((vel_d[:, _ACT_F_SPINE1_ID] - (pos_d[:, _ACT_F_SPINE1_ID] - pos_c[:, _ACT_F_SPINE1_ID]) / STEP_DT).abs().max()),
                        1e-3, msg="限位区内速度参考与位置不一致")

    # 回合首步不得因重置前的 κ 产生假的速度尖峰
    def test_ref_vel_zero_on_first_episode_step(self):
        env = make_ref_state_env([-CURVATURE_TARGET_MAX], phases=[0.0], gait=0.0, step_count=1)
        env._ref_kappa_prev = torch.tensor([0.0])   # 重置前后的曲率不同
        _, vel = ref_mod.get_reference_joint_state(env)
        self.assertLess(float(vel.abs().max()), 1e-6,
                        msg="回合首步出现了重置导致的速度参考尖峰")

    # 超时重置的回合也必须把步态相位清零 (原判据只看终止, 漏掉超时)
    def test_gait_phase_reset_covers_timeout(self):
        env = make_ref_state_env([0.0], phases=[0.7], gait=1.0)
        env.reset_buf = torch.tensor([True])        # 本步超时重置
        ref_mod.get_reference_joint_state(env)
        self.assertAlmostEqual(float(env._ref_phase[0]), 0.0, places=6,
                               msg="超时重置的回合没有清零步态相位")

        # 未重置环境的相位必须继续推进 (gait=1 Hz, dt=20 ms)
        env_keep = make_ref_state_env([0.0], phases=[0.7], gait=1.0)
        env_keep.reset_buf = torch.tensor([False])
        ref_mod.get_reference_joint_state(env_keep)
        self.assertAlmostEqual(float(env_keep._ref_phase[0]), 0.72, places=5,
                               msg="未重置环境的相位被误清零")

    # 命令更新必须按逐环境阶段: 未结束的旧阶段回合不得套用新阶段的速度公式
    def test_command_update_respects_episode_phase(self):
        env = make_ref_env([1.0, 1.0], [10.0, 10.0])
        cmd = env.command_manager._terms["slalom_cmd"]
        env.common_step_counter = PHASE1_STEP + 130
        env.episode_length_buf = torch.tensor([200, 50])   # env0 旧阶段, env1 新阶段
        cmd.command_tensor[:, 0] = 0.10                    # 旧阶段写下的命令速度
        cmd.command_tensor[:, 4] = 0.0
        env._path_kappa = torch.tensor([0.0, -CURVATURE_TARGET_MAX])

        cmd._update_command()

        # env0: 仍属旧阶段 → 命令必须保持不变 (曾经从 0.10 掉到 0.05, 参考随之后跳)
        self.assertAlmostEqual(cmd.command_tensor[0, 0].item(), 0.10, places=6,
                               msg="未结束的旧阶段回合被套用了新阶段速度公式")
        self.assertAlmostEqual(cmd.command_tensor[0, 4].item(), 0.0, places=6,
                               msg="未结束的旧阶段回合的曲率被改写")
        # 新阶段的命令与本步路径及共用速度曲线一致。
        self.assertAlmostEqual(cmd.command_tensor[1, 4].item(), float(env._path_kappa[1]), places=5)
        self.assertAlmostEqual(cmd.command_tensor[1, 0].item(),
                               BASE_VEL * get_velocity_scale(float(env.command_manager._terms["slalom_cmd"].slalom_peak_curvature[1])), places=6)

    # 重置后同一控制步内的首帧观测必须用新回合的参考 (框架顺序: 奖励 → 重置 → 观测)
    def test_reset_first_frame_uses_new_episode_reference(self):
        env = make_ref_state_env([20.0], phases=[0.5], gait=1.0, step_count=500)
        env.reset_buf = torch.tensor([True])            # 框架在算奖励前就已标出本步重置
        cmd = env.command_manager._terms["slalom_cmd"]

        pos_reward, _ = ref_mod.get_reference_joint_state(env)
        self.assertAlmostEqual(float(pos_reward[0, 1]), -0.9 * 20.0 / CURVATURE_TARGET_MAX,
                               places=5, msg="奖励趟的参考不是旧回合曲率")
        self.assertAlmostEqual(float(env._ref_phase[0]), 0.0, places=6,
                               msg="重置环境的相位未在重置趟清零")

        # 重置: 曲率重采样 (+20 → -20) + 回合计数归零 + 参考代次 +1
        cmd.command_tensor[:, 4] = -20.0
        env.episode_length_buf = torch.tensor([0])
        ref_mod.invalidate_reference_cache(env)

        pos_obs, _ = ref_mod.get_reference_joint_state(env)
        self.assertAlmostEqual(float(pos_obs[0, 1]), 0.9 * 20.0 / CURVATURE_TARGET_MAX,
                               places=5, msg="首帧观测仍返回上一回合的参考")
        self.assertAlmostEqual(float(env._ref_phase[0]), 0.0, places=6,
                               msg="重置后的重算把相位又推进了一次")

    # 接近段和主路径在位置、朝向、曲率上连续。
    def test_approach_joins_lut_at_matching_curvature(self):
        for peak in range(5, 16):
            geometry = get_slalom_geometry(peak)
            approach, period = geometry["approach"], geometry["period"]
            self.assertAlmostEqual(approach["x"][-1], geometry["spacing"], places=10)
            self.assertAlmostEqual(approach["y"][-1], 0.0, places=10)
            self.assertAlmostEqual(approach["heading"][-1], period["heading"][0], places=10)
            self.assertAlmostEqual(approach["kappa"][-1], period["kappa"][0], places=10)

    # 阶段必须按回合起点锁定, 不得在回合中途随全局计数器翻转
    def test_phase_locked_at_episode_start(self):
        from mjlab.tasks.SQuRo_Slalom.mdp.path import (
            compute_arc_path_ref,
            compute_path_ref,
            compute_slalom_path_ref,
        )

        # env0 起步于边界前 (仍 Phase 0), env1 起步于边界后 (Phase 1)
        env = make_ref_env([1.0, 1.0], [1.0, 1.0], spacing=POLE_SPACING_START)
        env.common_step_counter = PHASE1_STEP + 130
        env.episode_length_buf = torch.tensor([200, 50])
        cmd = env.command_manager._terms["slalom_cmd"]

        self.assertEqual(cmd.phase1_mask.tolist(), [False, True],
                         msg="阶段未按回合起点判定")

        x, _, _, _, _ = compute_path_ref(env)
        arc_x, _, _, _, _ = compute_arc_path_ref(env)
        slalom_x, _, _, _, _ = compute_slalom_path_ref(env)
        self.assertAlmostEqual(x[0].item(), arc_x[0].item(), places=6,
                               msg="边界前起步的回合被切到了绕杆路径")
        self.assertAlmostEqual(x[1].item(), slalom_x[1].item(), places=6,
                               msg="边界后起步的回合没有走绕杆路径")

        # 回合继续推进 (计数器与回合步数同步增加) → 阶段不得翻转
        env.common_step_counter += 1
        env.episode_length_buf += 1
        self.assertEqual(cmd.phase1_mask.tolist(), [False, True],
                         msg="阶段在回合中途翻转了")
        x2, _, _, _, _ = compute_path_ref(env)
        arc_x2, _, _, _, _ = compute_arc_path_ref(env)
        self.assertAlmostEqual(x2[0].item(), arc_x2[0].item(), places=6,
                               msg="回合中途被切到了绕杆路径")

    # 全部整数曲率的杆距严格递减，慢步频也能在回合内完成一个周期。
    def test_curriculum_spacing_avoids_incompatible_window(self):
        spacings = [get_slalom_geometry(k)["spacing"] for k in range(5, 16)]
        self.assertTrue(all(a > b for a, b in zip(spacings, spacings[1:])))
        tables = get_timed_slalom_tables(BASE_VEL)
        self.assertLess(float((tables["period_time"] + tables["approach_time"]).max()), 20.0)


# 一阶段动态曲率日程与连续路径回归。
class TestPhase0DynamicSchedule(unittest.TestCase):

    # 动态起始前只采恒定曲率; 起始后按概率混入动态回合, 曲率在当前范围内。
    def test_schedule_mix_and_range(self):
        from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import (
            PHASE0_DYNAMIC_PROBABILITY, PHASE0_DYNAMIC_START_ITER, PHASE0_MAX_SEGMENTS,
            PHASE0_MIN_SEGMENTS, get_phase0_curvature_range, sample_phase0_schedule,
        )
        torch.manual_seed(0)
        before = sample_phase0_schedule(PHASE0_DYNAMIC_START_ITER - 1, 256, torch.device("cpu"))
        self.assertFalse(bool(before["dynamic"].any()), msg="动态起始前出现了动态回合")
        self.assertTrue(bool((before["bounds"] > 20.0).all()), msg="恒定回合存在段边界")

        iteration = PHASE0_DYNAMIC_START_ITER + 500
        after = sample_phase0_schedule(iteration, 512, torch.device("cpu"))
        ratio = float(after["dynamic"].float().mean())
        self.assertAlmostEqual(ratio, PHASE0_DYNAMIC_PROBABILITY, delta=0.08,
                               msg=f"动态回合占比 {ratio:.2f} 偏离目标")
        dynamic = after["dynamic"]
        segments = after["segments"][dynamic]
        lower, upper = get_phase0_curvature_range(iteration * _STEPS_PER_ITER)
        self.assertGreaterEqual(float(segments.min()), lower - 1e-6)
        self.assertLessEqual(float(segments.max()), upper + 1e-6)
        counts = (after["bounds"][dynamic] < 20.0).sum(dim=1)
        self.assertGreaterEqual(int(counts.min()), PHASE0_MIN_SEGMENTS - 1)
        self.assertLessEqual(int(counts.max()), PHASE0_MAX_SEGMENTS - 1)
        peak = after["peak"][dynamic]
        self.assertTrue(torch.allclose(peak, segments.abs().amax(dim=1)))

    # 段间过渡连续: 两端到达目标曲率, 单步变化有界。
    def test_kappa_transition_is_smooth(self):
        from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import phase0_kappa_at
        segments = torch.zeros(1, 6)
        segments[0, :2] = torch.tensor([8.0, -6.0])
        bounds = torch.full((1, 6), 1e9)
        bounds[0, 0] = 6.0
        t = torch.linspace(0.0, 10.0, 1001)
        kappa = phase0_kappa_at(t, {"segments": segments, "bounds": bounds})
        self.assertAlmostEqual(float(kappa[0]), 8.0, places=5)
        self.assertAlmostEqual(float(kappa[-1]), -6.0, places=5)
        jump = float(torch.diff(kappa).abs().max())
        self.assertLess(jump, 0.6, msg=f"曲率切换单步跳变 {jump:.3f}")

    # 动态曲率路径连续: 单步位移有界, 恒定曲率段落在精确圆弧上。
    def test_dynamic_path_is_continuous(self):
        from mjlab.tasks.SQuRo_Slalom.mdp.path import compute_arc_path_ref
        segments = torch.zeros(2, 6)
        segments[0, :2] = torch.tensor([8.0, -8.0])
        bounds = torch.full((2, 6), 1e9)
        bounds[0, 0] = 5.0
        command = torch.zeros(2, 5)
        command[:, 0] = 0.1
        cmd = NS(command=command, phase1_mask=torch.zeros(2, dtype=torch.bool),
                 phase0_schedule={"dynamic": torch.tensor([True, False]),
                                  "segments": segments, "bounds": bounds,
                                  "peak": torch.tensor([8.0, 8.0])})
        env = NS(device=torch.device("cpu"), num_envs=2, step_dt=0.01,
                 episode_length_buf=torch.zeros(2, dtype=torch.long),
                 command_manager=NS(_terms={"slalom_cmd": cmd}), scene=NS(env_origins=None))
        env._phase0_path_state = {"xy": torch.zeros(2, 2), "heading": torch.zeros(2),
                                  "step": torch.zeros(2)}
        previous = None
        for step in range(1, 1001):
            env.episode_length_buf.fill_(step)
            x, y, _, _, heading = compute_arc_path_ref(env)
            if previous is not None:
                move = float(torch.maximum((x - previous[0]).abs(), (y - previous[1]).abs()).max())
                self.assertLess(move, 2e-3, msg=f"第 {step} 步路径跳变 {move:.5f} m")
                self.assertTrue(bool(torch.isfinite(heading).all()))
            previous = (x.clone(), y.clone())
        # 前 5 s 恒定曲率 8: 位置落在半径 0.125 m 的圆弧上
        radius = 1.0 / 8.0
        env.episode_length_buf.fill_(100)  # 1 s
        env._phase0_path_state = {"xy": torch.zeros(2, 2), "heading": torch.zeros(2),
                                  "step": torch.zeros(2)}
        env._phase0_path_cache = None
        x, y, _, _, _ = compute_arc_path_ref(env)
        theta = 8.0 * 0.1 * 1.0
        expected = 2.0 * radius * math.sin(theta / 2.0)
        self.assertAlmostEqual(float(torch.hypot(x[0], y[0])), expected, places=4)

    # 一阶段动态回合的参考步幅按回程峰值缩放, 而不是瞬时曲率。
    def test_reference_strides_follow_episode_peak(self):
        count = 50
        phases = [index / (count - 1) for index in range(count)]
        env = make_ref_state_env([0.0] * count, phases=phases, gait=0.0)
        cmd = env.command_manager._terms["slalom_cmd"]
        cmd.curvature_peak = torch.full((count,), 15.0)
        peak_pos, _ = ref_mod.get_reference_joint_state(env)
        ref_mod.invalidate_reference_cache(env)
        del cmd.curvature_peak
        inst_pos, _ = ref_mod.get_reference_joint_state(env)
        leg = _ACT_LEG_IDS
        peak_travel = float((peak_pos[:, leg].max(dim=0).values - peak_pos[:, leg].min(dim=0).values).abs().max())
        inst_travel = float((inst_pos[:, leg].max(dim=0).values - inst_pos[:, leg].min(dim=0).values).abs().max())
        self.assertGreater(inst_travel, peak_travel * 1.5,
                           msg="参考步幅没有按回程峰值缩放")


if __name__ == "__main__":
    unittest.main(verbosity=2)
