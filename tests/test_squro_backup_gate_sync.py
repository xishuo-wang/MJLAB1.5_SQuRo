from types import SimpleNamespace

import pytest
import torch

from mjlab.scripts.SQuRo_backup_Replay import StateMachinePolicy
from mjlab.tasks.SQuRo_Backup.mdp.command import BackupCommand
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES


# 同时覆盖正置、倒置、侧立零边界和贴地高度边界；不提供四元数数据。
@pytest.mark.parametrize(
    "front_delta,hind_delta,front_z,hind_z,s1,s2",
    [
        (-0.02, -0.02, 0.02, 0.02, False, False),
        (-0.02, 0.02, 0.02, 0.02, True, False),
        (0.02, 0.02, 0.02, 0.02, False, True),
        (0.02, -0.02, 0.02, 0.02, False, False),
        (0.0, 0.02, 0.02, 0.02, True, False),
        (-0.02, 0.0, 0.02, 0.02, False, False),
        (0.02, 0.0, 0.02, 0.02, False, False),
        (-0.02, 0.02, 0.03, 0.02, False, False),
        (-0.02, 0.02, 0.02, 0.03, False, False),
        (0.02, 0.02, 0.039, 0.039, False, True),
        (0.02, 0.02, 0.04, 0.02, False, False),
        (0.02, 0.02, 0.02, 0.04, False, False),
    ],
)
def test_replay_uses_training_site_gates(
    monkeypatch, front_delta, hind_delta, front_z, hind_z, s1, s2
):
    monkeypatch.setattr(_MODEL_INDICES, "segment_belly_back_ids", ((0, 1), (2, 3)))
    monkeypatch.setattr(_MODEL_INDICES, "f_body_id", 0)
    monkeypatch.setattr(_MODEL_INDICES, "h_body_id", 1)
    sites = torch.zeros(1, 4, 3, dtype=torch.float64)
    sites[0, 1, 2] = front_delta
    sites[0, 3, 2] = hind_delta
    bodies = torch.zeros(1, 2, 3, dtype=torch.float64)
    bodies[0, :, 2] = torch.tensor([front_z, hind_z], dtype=torch.float64)
    asset = SimpleNamespace(data=SimpleNamespace(site_pos_w=sites, body_link_pos_w=bodies))
    command = BackupCommand.__new__(BackupCommand)
    command._asset = asset

    def get_term(name):
        assert name == "backup_cmd"
        return command

    policy = StateMachinePolicy.__new__(StateMachinePolicy)
    policy.asset = asset
    policy.env = SimpleNamespace(
        unwrapped=SimpleNamespace(command_manager=SimpleNamespace(get_term=get_term))
    )
    assert policy._state() == pytest.approx((front_delta, hind_delta))
    assert policy._is_S1() == bool(command._check_S1()[0]) == s1
    assert policy._is_S2() == bool(command._check_S2()[0]) == s2


# 训练侧判据变化时，手调侧应立即跟随，不维护独立阈值。
def test_replay_delegates_gate_decisions():
    command = SimpleNamespace(
        _check_S1=lambda: torch.tensor([True]),
        _check_S2=lambda: torch.tensor([False]),
    )
    policy = StateMachinePolicy.__new__(StateMachinePolicy)
    policy.env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda _: command)
        )
    )
    assert policy._is_S1() is True
    assert policy._is_S2() is False
    command._check_S1 = lambda: torch.tensor([False])
    command._check_S2 = lambda: torch.tensor([True])
    assert policy._is_S1() is False
    assert policy._is_S2() is True
