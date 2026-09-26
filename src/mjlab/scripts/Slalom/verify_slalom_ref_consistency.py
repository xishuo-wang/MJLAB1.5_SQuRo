# SQuRo_Slalom 参考路径一致性回归；不创建环境、不修改训练数据。
# 覆盖: 局部重置隔离、参考推进与自身步频一致、gait=1 几何不变、命令计时器、
#       固定间距的单一真源、默认训练预算覆盖课程终点。
import unittest
from types import SimpleNamespace as NS

import torch

from mjlab.tasks.SQuRo_Slalom.mdp.command import SlalomCommand, SlalomCommandCfg
from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import (
    _STEPS_PER_ITER,
    BASE_VEL,
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


class TestSlalomRefConsistency(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
