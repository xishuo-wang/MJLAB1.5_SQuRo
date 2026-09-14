from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mjlab.scripts.SQuRo_backup_Replay import (
    StateMachinePolicy,
    VisConfig,
    _configure_command,
    slow1_target,
)
from mjlab.tasks.SQuRo_Backup.mdp.command import BackupCommandCfg


# slow1_target 的名义分段: T1 0→0.65 / T2 0.65→0.80 / T3 0.80→0.95 / T4 0.95→1.45
# 内部 time1=1.0 为 padding, 由 current_time = 1.0 + tn*λ 抵消, 故 tn = (current_time-1.0)/λ
@pytest.mark.parametrize("lam", [1.0, 3.0])
def test_replay_keyframes(lam):
    for t, expected in [
        (0.00, [0.0, 0.0, 0.0, 0.0]),          # 起点: 全部归零
        (0.325, [0.3, -0.785, 0.3, 0.785]),    # T1 中
        (0.65, [0.6, -1.57, 0.6, 1.57]),       # T1 末: 扭转拧满 + 侧摆/俯仰 0.6
        (0.725, [0.3, -1.57, 0.3, 1.57]),      # T2 中: 扭转保持, 侧摆/俯仰回收一半
        (0.80, [0.0, -1.57, 0.0, 1.57]),       # T2 末 = S1 姿态
        (0.875, [0.3, -0.785, 0.0, 0.785]),    # T3 中: 解扭 + F 侧摆回加
        (0.95, [0.6, 0.0, 0.0, 0.0]),          # T3 末 = S2 姿态
        (1.20, [0.3, 0.0, 0.0, 0.0]),          # T4 中
        (1.45, [0.0, 0.0, 0.0, 0.0]),          # T4 末: 回到站立零位
        (3.00, [0.0, 0.0, 0.0, 0.0]),          # T5 保持
    ]:
        q = slow1_target(1.0 + t * lam, lam)
        np.testing.assert_allclose([q[j] for j in [0, 1, 8, 9]], expected, atol=1e-12)


# T4 段腿角从支撑位平滑回到站立位, 脊柱只有 F_spine1 在动
@pytest.mark.parametrize("lam", [1.0, 3.0])
def test_leg_transition_in_t4(lam):
    fl_hold, hl_hold = (-0.28, 0.55), (-1.50, -0.25)
    leg_init = [0.1, -0.3, 0.1, -0.3, -0.1, 0.3, -0.1, 0.3]
    mid = slow1_target(1.0 + 1.20 * lam, lam)
    for c in range(4):
        assert mid[4 + c] == pytest.approx((fl_hold[c % 2] + leg_init[c]) / 2, abs=1e-12)
        assert mid[10 + c] == pytest.approx((hl_hold[c % 2] + leg_init[4 + c]) / 2, abs=1e-12)
    end = slow1_target(1.0 + 1.45 * lam, lam)
    np.testing.assert_allclose(end[4:8], leg_init[0:4], atol=1e-12)
    np.testing.assert_allclose(end[10:14], leg_init[4:8], atol=1e-12)


# T1~T3 段腿固定在支撑位 (脊柱形变期不换腿)
@pytest.mark.parametrize("tn", [0.1, 0.65, 0.80, 0.95])
def test_legs_hold_during_spine_phases(tn):
    q = slow1_target(1.0 + tn, 1.0)
    np.testing.assert_allclose(q[4:8], [-0.28, 0.55, -0.28, 0.55], atol=1e-12)
    np.testing.assert_allclose(q[10:14], [-1.50, -0.25, -1.50, -0.25], atol=1e-12)
    assert q[2] == 0.0 and q[3] == 0.0  # 颈部始终为零


# 构造无仿真的回放环境，独立控制 S1/S2 判据。
@pytest.fixture
def replay_env(monkeypatch):
    from mjlab.scripts import SQuRo_backup_Replay as replay

    monkeypatch.setattr(replay, "resolve_model_indices", lambda _: None)
    monkeypatch.setattr(replay._MODEL_INDICES, "joint_ids", tuple(range(14)))
    monkeypatch.setattr(replay._MODEL_INDICES, "f_body_id", 0)
    monkeypatch.setattr(replay._MODEL_INDICES, "h_body_id", 1)
    monkeypatch.setattr(replay._MODEL_INDICES, "segment_belly_back_ids", ((0, 1), (2, 3)))
    asset = SimpleNamespace(data=SimpleNamespace(
        default_joint_pos=torch.zeros(1, 14),
        joint_pos=torch.ones(1, 14),
        site_pos_w=torch.zeros(1, 4, 3),
        body_link_pos_w=torch.zeros(1, 2, 3),
    ))
    gates = SimpleNamespace(s1=False, s2=False)
    command = SimpleNamespace(
        _check_S1=lambda: torch.tensor([gates.s1]),
        _check_S2=lambda: torch.tensor([gates.s2]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(entities={"robot": asset}),
        cfg=SimpleNamespace(
            actions={"joint_pos": SimpleNamespace(scale=0.3)},
            commands={"backup_cmd": BackupCommandCfg()},
        ),
        command_manager=SimpleNamespace(get_term=lambda _: command),
        step_dt=0.01,
        gates=gates,
    )
    env.unwrapped = env
    return env


# 动作反算默认跟随环境 scale, 也允许显式覆盖
def test_action_scale_follows_env_and_override(replay_env):
    env = replay_env
    policy = StateMachinePolicy(env, 3.0, 5)
    assert policy.action_scale == 0.3

    # T2 末目标 F_body=-1.57, 默认角 0 → action = -1.57/0.3
    q = slow1_target(1.0 + 0.80 * 3.0, 3.0)
    action = (torch.tensor(q) - policy.default[0]) / policy.action_scale
    assert float(action[1]) == pytest.approx(-1.57 / 0.3, abs=1e-6)

    # 显式覆盖仍然生效 (复现旧脚本行为)
    env.cfg.actions["joint_pos"].scale = 0.5
    assert StateMachinePolicy(env, 3.0, 5).action_scale == 0.5
    assert StateMachinePolicy(env, 3.0, 5, action_scale=0.3).action_scale == 0.3


# 默认等待参数与训练一致，环境覆盖和旧共同覆盖均继续生效。
def test_wait_buffers_follow_env_and_legacy_override(replay_env):
    policy = StateMachinePolicy(replay_env, 3.0, 5)
    assert policy.p1_buffer_s == 1.0
    assert policy.p2_buffer_s == 0.3
    replay_env.cfg.commands["backup_cmd"].p1_buffer_s = 1.2
    replay_env.cfg.commands["backup_cmd"].p2_buffer_s = 0.4
    policy = StateMachinePolicy(replay_env, 3.0, 5)
    assert policy.p1_buffer_s == 1.2
    assert policy.p2_buffer_s == 0.4
    policy = StateMachinePolicy(replay_env, 3.0, 5, buffer=0.3)
    assert policy.p1_buffer_s == 0.3
    assert policy.p2_buffer_s == 0.3


# 旧截止时间之后仍保持末端姿态，新截止时间才重试，等待秒数不乘 λ。
@pytest.mark.parametrize("lam", [1.0, 3.0])
def test_p1_holds_until_extended_deadline(replay_env, lam):
    policy = StateMachinePolicy(replay_env, lam, 5, log_events=False)
    actual_before = policy.asset.data.joint_pos.clone()
    policy.t_phase = 0.8 * lam + 0.31
    policy(None)
    assert policy.phase == "P1"
    assert policy.retry["P1"] == 0
    assert policy.t_phase > 0.8 * lam + 0.3
    np.testing.assert_allclose(policy._spn_ref[-1], [0.0, -1.57, 0.0, 1.57], atol=1e-12)

    policy.t_phase = 0.8 * lam + 1.0 - replay_env.step_dt / 2
    policy(None)
    assert policy.phase == "P1"
    assert policy.retry["P1"] == 1
    assert policy.t_phase == 0.0
    np.testing.assert_allclose(policy._spn_ref[-1], [0.0, -1.57, 0.0, 1.57], atol=1e-12)
    policy(None)
    np.testing.assert_allclose(policy._spn_ref[-1], [0.0, 0.0, 0.0, 0.0], atol=1e-12)
    torch.testing.assert_close(policy.asset.data.joint_pos, actual_before)


# P1 未结束时不越过参考表；等待中 S1 达成即推进，截止时达成也优先推进。
@pytest.mark.parametrize("wait_elapsed", [0.1, 0.995])
def test_s1_advances_during_hold_and_at_deadline(replay_env, wait_elapsed):
    policy = StateMachinePolicy(replay_env, 3.0, 5, log_events=False)
    replay_env.gates.s1 = True
    policy.t_phase = 1.0
    policy(None)
    assert policy.phase == "P1"
    policy.t_phase = 0.8 * 3.0 + wait_elapsed
    policy(None)
    assert policy.phase == "P2"
    assert policy.t_phase == 0.0
    assert policy.retry["P1"] == 0


# P2 保留原有 0.3 秒等待，没有被 P1 的延长连带改变。
def test_p2_keeps_original_wait_deadline(replay_env):
    policy = StateMachinePolicy(replay_env, 3.0, 5, log_events=False)
    policy.phase = "P2"
    policy.t_phase = 0.15 * 3.0 + 0.28
    policy(None)
    assert policy.phase == "P2"
    assert policy.retry["P2"] == 0
    np.testing.assert_allclose(policy._spn_ref[-1], [0.6, 0.0, 0.0, 0.0], atol=1e-12)
    policy.t_phase = 0.15 * 3.0 + 0.3 - replay_env.step_dt / 2
    policy(None)
    assert policy.phase == "P2"
    assert policy.retry["P2"] == 1
    assert policy.t_phase == 0.0


# CLI 先写共同覆盖，再写分阶段覆盖，回放和内置 RL 状态机读取同一份配置。
@pytest.mark.parametrize("args, expected", [
    (VisConfig(), (1.0, 0.3)),
    (VisConfig(buffer=0.5), (0.5, 0.5)),
    (VisConfig(buffer=0.5, p1_buffer_s=1.2), (1.2, 0.5)),
    (VisConfig(buffer=0.5, p2_buffer_s=0.2), (0.5, 0.2)),
    (VisConfig(p1_buffer_s=1.2, p2_buffer_s=0.2), (1.2, 0.2)),
])
def test_cli_wait_overrides_share_command_config(replay_env, args, expected):
    command_cfg = replay_env.cfg.commands["backup_cmd"]
    _configure_command(command_cfg, args)
    assert command_cfg.fixed_time_scale == args.time_scale
    policy = StateMachinePolicy(replay_env, args.time_scale, 5)
    assert (policy.p1_buffer_s, policy.p2_buffer_s) == expected
