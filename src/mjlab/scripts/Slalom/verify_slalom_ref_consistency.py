# SQuRo_Slalom 参考路径一致性回归；不创建环境、不修改训练数据。
# 覆盖: 局部重置隔离、参考推进与自身步频一致、gait=1 几何不变、命令计时器、
#       固定间距的单一真源、默认训练预算覆盖课程终点、腿部参考过零连续。
import unittest
from types import SimpleNamespace as NS

import torch

from mjlab.tasks.SQuRo_Slalom.mdp import reference as ref_mod
from mjlab.tasks.SQuRo_Slalom.mdp.command import SlalomCommand, SlalomCommandCfg
from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import (
    _STEPS_PER_ITER,
    BASE_VEL,
    CURVATURE_TARGET_MAX,
    PHASE1_END_ITER,
    PHASE2_END_ITER,
    POLE_SPACING_START,
)
from mjlab.tasks.SQuRo_Slalom.mdp.events import reset_model
from mjlab.tasks.SQuRo_Slalom.mdp.path import (
    compute_slalom_path_ref,
    get_approach_start,
    get_effective_pole_spacing,
)

STEP_DT = 0.02
# Phase 1 期间的 common_step_counter (绕杆模式生效)
PHASE1_STEP = PHASE1_END_ITER * _STEPS_PER_ITER
# 腿部关节列 (FL/FR/HL/HR) 与内侧腿列
_LEG_COLS = [4, 5, 6, 7, 10, 11, 12, 13]
_LEFT_COLS = [4, 5, 10, 11]
_RIGHT_COLS = [6, 7, 12, 13]
_TABLE_RES = 50

# 重构前 (旧实现) 在 gait=1、间距 0.20 下的绕杆参考, 用于证明几何语义未变
_GOLDEN_GAIT1 = {
    0.5: (0.177975863, -0.001805203, 0.244365886, -22.057279587),
    1.0: (0.191688433, -0.000095714, 0.034134317, -8.212568283),
    2.0: (0.240696758, 0.000000000, 0.000000000, 0.000000000),
    3.0: (0.271876156, -0.002690654, -0.318749994, -25.000000000),
    5.0: (0.294352561, -0.021453112, -1.068750024, -25.000000000),
    8.0: (0.307691991, -0.088275515, -0.968750000, 25.000000000),
}


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
    cmd.command_tensor = torch.zeros(n, 5)
    cmd.command_tensor[:, 3] = torch.tensor(gait_freqs, dtype=torch.float32)
    env.command_manager = NS(_terms={"slalom_cmd": cmd})
    # 旧实现从全局槽读步频; 桩里默认模拟"最后重置者"写入
    env._slalom_gait_scalar = float(gait_freqs[-1] if gait_scalar is None else gait_scalar)
    env._slalom_base_vel_scalar = base_vel
    return env


# 构造只含计时器所需字段的桩命令项
def make_timer_cmd(n=4, time_left=25.0):
    cmd = object.__new__(SlalomCommand)
    cmd._env = NS(device=torch.device("cpu"), num_envs=n, step_dt=STEP_DT,
                  common_step_counter=0)
    cmd.cfg = SlalomCommandCfg()
    cmd.time_left = torch.full((n,), float(time_left))
    cmd.command_counter = torch.zeros(n, dtype=torch.long)
    cmd.metrics = {}
    cmd._update_metrics = lambda: None
    cmd.command_tensor = torch.zeros(n, 5)
    cmd.command_tensor[:, 3] = 1.0
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

    # gait=1 时绕杆几何必须与重构前逐点一致
    def test_gait1_geometry_unchanged(self):
        for t, (gx, gy, gh, gk) in _GOLDEN_GAIT1.items():
            env = make_ref_env([1.0], [t], gait_scalar=1.0)
            x, y, _, _, h = compute_slalom_path_ref(env)
            k = env._path_kappa
            self.assertAlmostEqual(x[0].item(), gx, places=6, msg=f"t={t} 参考 x 漂移")
            self.assertAlmostEqual(y[0].item(), gy, places=6, msg=f"t={t} 参考 y 漂移")
            self.assertAlmostEqual(h[0].item(), gh, places=6, msg=f"t={t} 参考朝向漂移")
            self.assertAlmostEqual(k[0].item(), gk, places=4, msg=f"t={t} 参考曲率漂移")

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

    # 固定间距覆盖值也必须经过有效间距转换
    def test_active_pole_spacing_applies_effective(self):
        cmd = object.__new__(SlalomCommand)
        cmd._env = NS(common_step_counter=PHASE1_STEP)
        cmd.cfg = SlalomCommandCfg()

        # 不兼容区间内的覆盖值必须 clamp 到无直行最小间距
        cmd.cfg.fixed_pole_spacing = 0.10
        self.assertAlmostEqual(cmd.active_pole_spacing, get_effective_pole_spacing(0.10),
                               places=9, msg="覆盖值绕过了有效间距转换")
        self.assertLess(cmd.active_pole_spacing, 0.10)

        # 兼容区间内的覆盖值原样返回
        cmd.cfg.fixed_pole_spacing = 0.15
        self.assertAlmostEqual(cmd.active_pole_spacing, 0.15, places=9)
        self.assertAlmostEqual(get_effective_pole_spacing(0.15), 0.15, places=9)

    # 出生点必须与参考路径使用同一个杆间距
    def test_reset_model_uses_command_spacing(self):
        cmd = object.__new__(SlalomCommand)
        env = NS(device=torch.device("cpu"), common_step_counter=PHASE1_STEP)
        cmd._env = env
        cmd.cfg = SlalomCommandCfg()
        cmd.cfg.fixed_pole_spacing = 0.15

        recorded = {}

        def record_root(root_state, env_ids=None):
            recorded["root"] = root_state.clone()

        robot = NS(
            num_joints=0,
            write_root_state_to_sim=record_root,
            write_joint_state_to_sim=lambda *a, **k: None,
        )
        env.scene = NS(entities={"robot": robot})
        env.command_manager = NS(_terms={"slalom_cmd": cmd})

        reset_model(env, torch.tensor([0, 1]))

        expected_x = get_approach_start(spacing=cmd.active_pole_spacing)[0]
        for i in (0, 1):
            self.assertAlmostEqual(recorded["root"][i, 0].item(), expected_x, places=6,
                                   msg="出生点与参考路径的杆间距不一致")

    # 默认训练预算必须覆盖课程终点
    def test_default_budget_covers_curriculum(self):
        from mjlab.tasks.SQuRo_Slalom.config.rl_cfg import SQuRo_Slalom_PPO_Runner_Cfg

        cfg = SQuRo_Slalom_PPO_Runner_Cfg()
        self.assertGreaterEqual(cfg.max_iterations, PHASE2_END_ITER,
                                msg="默认训练轮数覆盖不到课程终点")

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

    # 内侧腿必须随转向侧收缩, 外侧腿参考不得随曲率改变
    def test_inner_leg_is_on_turn_side(self):
        travel_left_0 = leg_travel(0.0, _LEFT_COLS)
        travel_right_0 = leg_travel(0.0, _RIGHT_COLS)
        travel_left_pos = leg_travel(CURVATURE_TARGET_MAX, _LEFT_COLS)
        travel_right_pos = leg_travel(CURVATURE_TARGET_MAX, _RIGHT_COLS)
        travel_left_neg = leg_travel(-CURVATURE_TARGET_MAX, _LEFT_COLS)
        travel_right_neg = leg_travel(-CURVATURE_TARGET_MAX, _RIGHT_COLS)

        # 左转 (κ>0): 左腿为内侧 → 行程被压缩; 右腿为外侧 → 与直行完全一致
        for i in range(len(_LEFT_COLS)):
            self.assertLess(float(travel_left_pos[i]), float(travel_left_0[i]) * 0.95,
                            msg="左转时左腿 (内侧) 行程未被压缩")
        for i in range(len(_RIGHT_COLS)):
            self.assertAlmostEqual(float(travel_right_pos[i]), float(travel_right_0[i]),
                                   places=6, msg="左转时右腿 (外侧) 参考被改动")

        # 右转 (κ<0): 右腿为内侧 → 压缩; 左腿为外侧 → 与直行完全一致
        for i in range(len(_RIGHT_COLS)):
            self.assertLess(float(travel_right_neg[i]), float(travel_right_0[i]) * 0.95,
                            msg="右转时右腿 (内侧) 行程未被压缩")
        for i in range(len(_LEFT_COLS)):
            self.assertAlmostEqual(float(travel_left_neg[i]), float(travel_left_0[i]),
                                   places=6, msg="右转时左腿 (外侧) 参考被改动")

    # 脊柱/颈参考不得超过模型关节与执行器限位
    def test_spine_ref_within_joint_limits(self):
        kappas = [(-25 + 2.5 * i) for i in range(21)]
        env = make_ref_state_env(kappas, phases=[0.3] * len(kappas), gait=0.0)
        pos, _ = ref_mod.get_reference_joint_state(env)
        max_f = float(pos[:, 0].abs().max())
        max_h = float(pos[:, 8].abs().max())
        max_yaw = float(pos[:, 2].abs().max())
        self.assertLessEqual(max_f, 0.6 + 1e-5, msg=f"F_spine1 参考超出限位: {max_f:.4f}")
        self.assertLessEqual(max_h, 0.6 + 1e-5, msg=f"H_spine1 参考超出限位: {max_h:.4f}")
        self.assertLessEqual(max_yaw, 0.8 + 1e-5, msg=f"Neck_yaw 参考超出限位: {max_yaw:.4f}")
        # 高曲率处必须真的贴到限位 (否则说明 clamp 没生效)
        self.assertAlmostEqual(max_f, 0.6, places=4)
        self.assertAlmostEqual(max_h, 0.6, places=4)

    # 速度参考必须等于位置参考的差分 (位置动而速度说不动即为不一致)
    def test_ref_vel_matches_position_difference(self):
        # 冻结相位 (gait=0) 以隔离"曲率变化"这一项; |κ| 取在脊柱限位以内
        env = make_ref_state_env([-20.0], phases=[0.3], gait=0.0)
        pos_a, _ = ref_mod.get_reference_joint_state(env)
        cmd = env.command_manager._terms["slalom_cmd"]
        cmd.command_tensor[:, 4] = -20.4      # 同一曲率 bin 内, 保证插值线性
        env.common_step_counter = 1
        env.episode_length_buf = torch.tensor([6])
        pos_b, vel_b = ref_mod.get_reference_joint_state(env)
        expected = (pos_b - pos_a) / STEP_DT
        diff = (vel_b - expected).abs().max()
        self.assertLess(float(diff), 1e-3,
                        msg=f"速度参考与位置参考差分不符, 最大差 {float(diff):.4f} rad/s")
        # 脊柱确实在动 (排除"两边都是 0"的假通过)
        self.assertGreater(float(vel_b[:, 0].abs().max()), 1e-3)
        # 被限位截住时导数必须为 0 (与位置恒定一致)
        env_hi = make_ref_state_env([-24.0], phases=[0.3], gait=0.0)
        pos_c, _ = ref_mod.get_reference_joint_state(env_hi)
        env_hi.command_manager._terms["slalom_cmd"].command_tensor[:, 4] = -24.4
        env_hi.common_step_counter = 1
        env_hi.episode_length_buf = torch.tensor([6])
        pos_d, vel_d = ref_mod.get_reference_joint_state(env_hi)
        self.assertLess(float((vel_d[:, 0] - (pos_d[:, 0] - pos_c[:, 0]) / STEP_DT).abs().max()),
                        1e-3, msg="限位区内速度参考与位置不一致")

    # 回合首步不得因重置前的 κ 产生假的速度尖峰
    def test_ref_vel_zero_on_first_episode_step(self):
        env = make_ref_state_env([-25.0], phases=[0.0], gait=0.0, step_count=1)
        env._ref_kappa_prev = torch.tensor([0.0])   # 上一回合末尾 κ=0, 重置后 κ=-25
        _, vel = ref_mod.get_reference_joint_state(env)
        self.assertLess(float(vel.abs().max()), 1e-6,
                        msg="回合首步出现了重置导致的速度参考尖峰")

    # 接近段与绕杆主路径衔接处, 曲率与期望速度都必须连续
    def test_approach_joins_lut_at_matching_curvature(self):
        # 有直行模式 LUT 首段是直行 (κ=0), 无直行模式首段是 κ=-K 平台
        cases = ((POLE_SPACING_START, 0.0),
                 (get_effective_pole_spacing(0.08), -CURVATURE_TARGET_MAX))
        for spacing, junction_kappa in cases:
            env = make_ref_env([1.0], [0.0], spacing=spacing)
            compute_slalom_path_ref(env)
            tau_total = env._slalom_app_tbl["tau_total"]
            self.assertGreater(tau_total, 2 * STEP_DT)

            samples = []
            for sign in (-1, 1):
                t = tau_total + sign * 0.5 * STEP_DT      # 步频 1.0 → tau == t
                e = make_ref_env([1.0], [t], spacing=spacing)
                _, _, vx, vy, _ = compute_slalom_path_ref(e)
                samples.append((float(e._path_kappa[0]), float(torch.hypot(vx[0], vy[0]))))
            k_before, v_before = samples[0]
            k_after, v_after = samples[1]

            # 主路径起始曲率必须与接近段终点一致 (无直行时曾是 0 vs -K 的硬跳变)
            self.assertAlmostEqual(k_after, junction_kappa, places=1,
                                   msg=f"间距 {spacing:.4f}: 主路径起始曲率 {k_after:.3f} "
                                       f"与期望 {junction_kappa:.3f} 不符")
            self.assertLess(abs(k_after - k_before), 0.1 * CURVATURE_TARGET_MAX,
                            msg=f"间距 {spacing:.4f}: 衔接处曲率跳变 {abs(k_after - k_before):.3f}")
            self.assertLess(abs(v_after - v_before), 0.2 * max(v_before, v_after),
                            msg=f"间距 {spacing:.4f}: 衔接处期望速度跳变 "
                                f"{v_before:.4f} → {v_after:.4f} m/s")


if __name__ == "__main__":
    unittest.main(verbosity=2)
