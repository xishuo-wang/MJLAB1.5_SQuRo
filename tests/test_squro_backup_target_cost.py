from types import SimpleNamespace

import pytest
import torch

from mjlab.managers.reward_manager import RewardManager
from mjlab.tasks.SQuRo_Backup.mdp import rewards
from mjlab.tasks.SQuRo_Backup.mdp.indices import _ACTUATED_JOINT_NAMES
from mjlab.tasks.SQuRo_Backup.SQuRo_Backup_env_cfg import SQuRo_Backup_Env_Cfg

SPINE = (0, 1, 8, 9)


# 构造动作项而不是实际关节状态，确保成本只依赖本步指令及本步参考。
def make_env(monkeypatch, reference, target=None, scale=0.3, offset=0.0, order=None):
  reference = torch.as_tensor(reference, dtype=torch.float64)
  target = (
    reference.clone()
    if target is None
    else torch.as_tensor(target, dtype=torch.float64)
  )
  scale = torch.as_tensor(scale, dtype=torch.float64).expand_as(reference).clone()
  offset = torch.as_tensor(offset, dtype=torch.float64).expand_as(reference).clone()
  order = list(range(14)) if order is None else order
  term = SimpleNamespace(
    raw_action=((target - offset) / scale)[:, order],
    scale=scale[:, order],
    offset=offset[:, order],
    target_names=[_ACTUATED_JOINT_NAMES[i] for i in order],
  )

  def get_term(name):
    assert name == "joint_pos"
    return term

  env = SimpleNamespace(
    action_manager=SimpleNamespace(get_term=get_term), reference=reference
  )
  monkeypatch.setattr(
    rewards,
    "get_reference_joint_state",
    lambda e: (e.reference, torch.zeros_like(e.reference)),
  )
  return env, term


def test_exact_reference_is_zero_with_scale_offset_and_reordered_actions(monkeypatch):
  reference = torch.arange(28, dtype=torch.float64).reshape(2, 14) / 20
  scale = torch.linspace(0.1, 0.8, 14)
  offset = torch.linspace(-0.2, 0.2, 14)
  env, _ = make_env(
    monkeypatch, reference, scale=scale, offset=offset, order=list(reversed(range(14)))
  )
  torch.testing.assert_close(
    rewards.compute_spine_target_cost(env),
    torch.zeros(2, dtype=torch.float64),
    atol=1e-14,
    rtol=0,
  )


# 默认标量配置和按关节张量配置都应使用同一动作映射。
@pytest.mark.parametrize("scale,offset", [(0.3, 0.0), (0.5, 0.1)])
def test_scalar_scale_and_offset(monkeypatch, scale, offset):
  reference = torch.zeros(2, 14, dtype=torch.float64)
  reference[:, SPINE] = torch.tensor([0.6, -1.57, 0.6, 1.57], dtype=torch.float64)
  env, term = make_env(monkeypatch, reference, scale=scale, offset=offset)
  term.scale, term.offset = scale, offset
  torch.testing.assert_close(
    rewards.compute_spine_target_cost(env),
    torch.zeros(2, dtype=torch.float64),
    atol=1e-14,
    rtol=0,
  )


# 即使另一个字段已被限幅成正确目标，仍应识别 1.46 rad 的过量指令。
def test_overshoot_before_clipping_is_penalized(monkeypatch):
  reference = torch.zeros(2, 14, dtype=torch.float64)
  reference[:, 0] = 0.6
  target = reference.clone()
  target[1, 0] = 1.46
  env, term = make_env(monkeypatch, reference, target)
  term._processed_actions = reference.clone()
  cost = rewards.compute_spine_target_cost(env)
  assert cost[0] == pytest.approx(0)
  assert cost[1] == pytest.approx(-((1.46 - 0.6) ** 2) / 4)


# T2 正确保持、部分释放、完全释放分别给出递增成本，不使用指数饱和。
def test_t2_twist_release_has_continuous_cost(monkeypatch):
  reference = torch.zeros(3, 14, dtype=torch.float64)
  reference[:, 1], reference[:, 9] = -1.57, 1.57
  target = reference.clone()
  target[1, 1], target[1, 9] = -0.8, 0.8
  target[2, 1], target[2, 9] = 0, 0
  env, _ = make_env(monkeypatch, reference, target)
  cost = rewards.compute_spine_target_cost(env)
  assert cost[0] == pytest.approx(0)
  assert cost[0] > cost[1] > cost[2]
  assert cost[2] == pytest.approx(-2 * 1.57**2 / 4)


@pytest.mark.parametrize("column", SPINE)
def test_four_spine_joints_have_equal_weight(monkeypatch, column):
  reference = torch.zeros(1, 14, dtype=torch.float64)
  target = reference.clone()
  target[0, column] = 0.4
  env, _ = make_env(monkeypatch, reference, target)
  assert rewards.compute_spine_target_cost(env)[0] == pytest.approx(-(0.4**2) / 4)


def test_leg_and_neck_commands_are_not_penalized(monkeypatch):
  reference = torch.zeros(1, 14, dtype=torch.float64)
  target = torch.ones_like(reference)
  target[:, SPINE] = 0
  env, _ = make_env(monkeypatch, reference, target)
  assert rewards.compute_spine_target_cost(env)[0] == 0


def test_current_reference_is_queried_each_step(monkeypatch):
  reference = torch.zeros(1, 14, dtype=torch.float64)
  env, _ = make_env(monkeypatch, reference)
  assert rewards.compute_spine_target_cost(env)[0] == 0
  env.reference[:, 1] = -1.57
  assert rewards.compute_spine_target_cost(env)[0] == pytest.approx(-(1.57**2) / 4)


# 用实际 RewardManager 验证外部权重 2.0 和 dt 只乘一次。
@pytest.mark.parametrize("dt", [0.005, 0.01, 0.02])
def test_reward_manager_applies_weight_and_dt_once(monkeypatch, dt):
  reference = torch.zeros(1, 14, dtype=torch.float64)
  target = reference.clone()
  target[:, SPINE] = 0.5
  env, _ = make_env(monkeypatch, reference, target)
  env.num_envs, env.device, env.max_episode_length_s = 1, "cpu", 10.0
  cfg = SQuRo_Backup_Env_Cfg()
  manager = RewardManager({"spine_target": cfg.rewards["spine_target"]}, env)
  assert manager.compute(dt)[0] == pytest.approx(-2.0 * 0.25 * dt)
  assert manager._step_reward[0, 0] == pytest.approx(-0.5)


def test_configuration_only_adds_target_cost():
  cfg = SQuRo_Backup_Env_Cfg()
  assert cfg.rewards["spine_target"].weight == 2.0
  assert cfg.rewards["spine_target"].func is rewards.compute_spine_target_cost
  for name in ("mimic_pos", "mimic_vel", "action_L1", "action_L2", "energy"):
    assert cfg.rewards[name].weight == 1.0
  assert cfg.actions["joint_pos"].scale == 0.3
