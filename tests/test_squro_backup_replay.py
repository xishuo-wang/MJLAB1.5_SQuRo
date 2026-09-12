from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mjlab.scripts.SQuRo_backup_Replay import StateMachinePolicy, slow1_target


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


# 动作反算默认跟随环境 scale, 也允许显式覆盖
def test_action_scale_follows_env_and_override(monkeypatch):
    from mjlab.scripts import SQuRo_backup_Replay as replay

    monkeypatch.setattr(replay, "resolve_model_indices", lambda _: None)
    monkeypatch.setattr(replay._MODEL_INDICES, "joint_ids", tuple(range(14)))
    asset = SimpleNamespace(data=SimpleNamespace(default_joint_pos=torch.zeros(1, 14)))
    env = SimpleNamespace(
        scene=SimpleNamespace(entities={"robot": asset}),
        cfg=SimpleNamespace(actions={"joint_pos": SimpleNamespace(scale=0.3)}),
    )
    env.unwrapped = env

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
