from types import SimpleNamespace

import pytest
import torch

from mjlab.scripts.Backup.baseline_trivial_policy import (
  EpisodeAccumulator,
  TerminalSnapshot,
  action_metrics,
  summarize,
)


def state(phase=(0, 0), success=(False, False)):
  return {
    "phase": torch.tensor(phase),
    "success": torch.tensor(success),
    "retry": torch.zeros(2),
  }


# 不同环境异步终止；保留末步奖励，超时也入账，不接收第二个 episode。
def test_complete_episodes_include_terminal_and_timeout_rewards():
  stats = EpisodeAccumulator(2, ["mimic", "cost"], 0.01, 0.5, "cpu")
  stats.update(
    torch.tensor([2.0, 3.0]),
    torch.tensor([[3.0, -1.0], [4.0, -1.0]]),
    torch.tensor([True, False]),
    torch.tensor([False, False]),
    torch.tensor([2, 0]),
    state((2, 0), (True, False)),
    {"test": torch.tensor([1.0, 2.0])},
  )
  stats.update(
    torch.tensor([100.0, 5.0]),
    torch.tensor([[101.0, -1.0], [7.0, -2.0]]),
    torch.tensor([False, False]),
    torch.tensor([False, True]),
    torch.tensor([0, 1]),
    state((0, 2)),
    {"test": torch.tensor([100.0, 4.0])},
  )
  a, b = stats.rows
  assert a["return"] == 2
  assert a["steps"] == 1
  assert a["success"] is True
  assert b["return"] == 8
  assert b["discounted_return"] == 5.5
  assert b["steps"] == 2
  assert b["timeout"] is True
  assert b["reward_parts"] == {"mimic": 11, "cost": -3}
  assert b["metrics_mean"]["test"] == 3
  assert b["s2_time_s"] == pytest.approx(0.02)
  assert not stats.active.any()
  for row in stats.rows:
    assert sum(row["reward_parts"].values()) == row["return"]
    assert sum(row["discounted_reward_parts"].values()) == row["discounted_return"]
    assert (
      sum(sum(p.values()) for p in row["phase_reward_parts"].values()) == row["return"]
    )
  summary = summarize(stats.rows)
  assert summary["success_rate"] == 0.5
  assert summary["success_duration_s"]["mean"] == 0.01
  assert summary["failure_return"]["mean"] == 8


# 同时成功和超时只生成一条记录，不丢失任何终止原因。
def test_simultaneous_termination_and_timeout():
  stats = EpisodeAccumulator(2, ["reward"], 0.01, 0.99, "cpu")
  stats.update(
    torch.ones(2),
    torch.ones(2, 1),
    torch.ones(2, dtype=torch.bool),
    torch.ones(2, dtype=torch.bool),
    torch.zeros(2),
    state(),
    {},
  )
  assert len(stats.rows) == 2
  assert all(r["terminated"] and r["timeout"] for r in stats.rows)


def test_reward_mismatch_is_rejected():
  stats = EpisodeAccumulator(2, ["reward"], 0.01, 0.99, "cpu")
  with pytest.raises(AssertionError):
    stats.update(
      torch.ones(2),
      torch.zeros(2, 1),
      torch.zeros(2, dtype=torch.bool),
      torch.zeros(2, dtype=torch.bool),
      torch.zeros(2),
      state(),
      {},
    )


# 终止回合的状态来自重置前，其余环境来自当前步，快照不得混入下一步。
def test_terminal_snapshot_survives_auto_reset():
  recorder = TerminalSnapshot(SimpleNamespace(), SimpleNamespace(device="cpu"))
  recorder.snapshot = lambda: {"phase": torch.tensor([2, 1])}
  recorder.record_pre_reset(torch.tensor([0]))
  recorder.snapshot = lambda: {"phase": torch.tensor([0, 2])}
  assert recorder.after_step()["phase"].tolist() == [2, 2]
  assert recorder.after_step()["phase"].tolist() == [0, 2]


def test_action_metrics_use_clipped_targets_and_separate_initial_jump():
  raw = torch.tensor([[10.0, -2.0], [2.0, 2.0]])
  applied = raw.clamp(-6, 6)
  previous = torch.zeros_like(raw)
  metric = action_metrics(raw, applied, previous, previous, torch.tensor([0, 1]), 0.3)
  assert metric["clip_fraction"].tolist() == [0.5, 0.0]
  assert metric["action_delta_l2_sum"].tolist() == [40, 8]
  assert metric["initial_action_delta_l2_sum"].tolist() == [40, 0]
  assert metric["ongoing_action_delta_l2_sum"].tolist() == [0, 8]
  torch.testing.assert_close(
    metric["target_delta_l2_rad2_sum"], torch.tensor([3.6, 0.72])
  )


# 成功轨迹的重计分只移除两项平滑成本，不能误删能耗或里程碑。
def test_same_trajectory_rescoring_only_removes_smoothing():
  stats = EpisodeAccumulator(
    2, ["milestone_success", "action_L1", "action_L2", "energy"], 0.01, 0.99, "cpu"
  )
  parts = torch.tensor([[10.0, -1.0, -2.0, -3.0]]).expand(2, -1)
  stats.update(
    parts.sum(-1),
    parts,
    torch.ones(2, dtype=torch.bool),
    torch.zeros(2, dtype=torch.bool),
    torch.full((2,), 2),
    state((2, 2), (True, True)),
    {},
  )
  assert all(row["return"] == 4 for row in stats.rows)
  assert all(row["same_trajectory_return_without_smoothing"] == 7 for row in stats.rows)
