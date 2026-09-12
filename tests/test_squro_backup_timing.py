from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mjlab.tasks.SQuRo_Backup.mdp import reference
from mjlab.tasks.SQuRo_Backup.mdp.command import BackupCommand
from mjlab.tasks.SQuRo_Backup.mdp.timing import (
    P1_END,
    P2_END,
    STAND_TRANSITION_END,
    TIME_COMPARISON_SCALE,
)
from mjlab.tasks.SQuRo_Backup.SQuRo_Backup_env_cfg import SQuRo_Backup_Env_Cfg


# 构造不依赖物理仿真的参考查询环境。
def make_reference_env(phase, stage_t):
    count = len(phase)
    command = torch.zeros(count, 7)
    command[:, 5] = TIME_COMPARISON_SCALE
    term = SimpleNamespace(
        command=command,
        phase=torch.tensor(phase),
        stage_t=torch.tensor(stage_t, dtype=torch.float32),
    )
    return SimpleNamespace(
        device="cpu", command_manager=SimpleNamespace(_terms={"backup_cmd": term})
    )


# 检查新阶段端点、回收中点以及参考范围之外的站立保持。
# t_phase 为缩放后时间: t_nom = t_phase / λ
def test_reference_keyframes():
    env = make_reference_env(
        [0, 0, 0, 0, 1, 1, 2, 2],
        [0, 1.95, 2.175, 2.4, 0, 0.45, 0, 3.15],
    )
    pos, _ = reference.get_reference_joint_state(env)
    expected = torch.tensor([
        [0, 0, 0, 0],             # 起点
        [0.6, -1.57, 0.6, 1.57],  # T1 末 (t_nom=0.65)
        [0.3, -1.57, 0.3, 1.57],  # T2 中点 (t_nom=0.725)
        [0, -1.57, 0, 1.57],      # T2 末 = S1 姿态 (t_nom=0.80)
        [0, -1.57, 0, 1.57],      # P2 起点冻结在 S1 姿态
        [0.6, 0, 0, 0],           # T3 末 = S2 姿态 (t_nom=0.95)
        [0.6, 0, 0, 0],           # P3 起点
        [0, 0, 0, 0],             # T4 末 (t_nom=1.45)
    ])
    torch.testing.assert_close(pos[:, [0, 1, 8, 9]], expected, atol=2e-5, rtol=0)


# T1 前半段与 T2 回收段速度方向相反、幅值不同（两段名义时长不同）；缓冲期冻结端点参考。
def test_recovery_speed_and_buffer_hold():
    #   0.75 → t_nom=0.25  T1 前半, 速度 +0.6/(0.65λ)
    #   2.1  → t_nom=0.70  T2 回收中, 速度 -0.6/(0.15λ)
    #   2.4/2.7 → t_nom=0.80/0.90  端点冻结
    #   0.3  → t_nom=1.00  T3 保持 +0.6 ; 0.6 → t_nom=1.20 T4 末端归零
    env = make_reference_env([0, 0, 0, 0, 1, 1], [0.75, 2.1, 2.4, 2.7, 0.3, 0.6])
    pos, vel = reference.get_reference_joint_state(env)
    spine = [0, 8]
    speed_t1 = 0.6 / (0.65 * TIME_COMPARISON_SCALE)
    speed_t2 = 0.6 / (0.15 * TIME_COMPARISON_SCALE)
    torch.testing.assert_close(vel[0, spine], torch.full((2,), speed_t1), atol=2e-5, rtol=0)
    torch.testing.assert_close(vel[1, spine], torch.full((2,), -speed_t2), atol=2e-5, rtol=0)
    # 缓冲期: 参考冻结 → 速度为零、位置停端点
    torch.testing.assert_close(vel[2], torch.zeros_like(vel[2]), atol=0, rtol=0)
    torch.testing.assert_close(vel[3], torch.zeros_like(vel[3]), atol=0, rtol=0)
    torch.testing.assert_close(pos[2], pos[3])
    # 端点 = T2 末姿态 (S1)
    torch.testing.assert_close(pos[2, [0, 1, 8, 9]], torch.tensor([0.0, -1.57, 0.0, 1.57]))
    # P2/T3 解扭中: F_spine1 上扬、扭转回零, 且 H_spine1 保持 0
    torch.testing.assert_close(pos[4, [0, 1, 8, 9]], torch.tensor([0.4, -0.5233, 0.0, 0.5233]), atol=2e-4, rtol=0)
    torch.testing.assert_close(vel[4, [0, 8]], torch.tensor([0.6 / (0.15 * TIME_COMPARISON_SCALE), 0.0]))
    # T4 末端 F_spine1 归零、腿回到站立位
    torch.testing.assert_close(pos[5, [0, 1, 8, 9]], torch.tensor([0.6, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(vel[5], torch.zeros_like(vel[5]), atol=0, rtol=0)


# 身体轨迹重定时只拉伸旧回收区间，后续轨迹平移且保持连续。
# 旧文件采集于 0.65→0.80，新时序 T2 恰为同一区间，故前段为恒等映射。
def test_body_reference_retiming():
    new_t = torch.tensor([0, 0.5, 0.65, 0.725, P1_END, P2_END, STAND_TRANSITION_END, 3.0])
    old_t = torch.tensor([0, 0.5, 0.65, 0.725, 0.8, 0.95, 1.45, 3.0])
    torch.testing.assert_close(reference._body_traj_source_time(new_t), old_t)
    env = make_reference_env([0, 0, 0, 1, 2, 2], [1.95, 2.175, 4.1, 0.45, 1.5, 10])
    actual = torch.stack(reference.get_body_reference(env), dim=1).numpy()
    source = np.load(reference._BODY_TRAJ_PATH)
    lookup = [0.65, 0.725, 0.8, 0.95, 1.45, source[-1, 0]]
    expected = np.column_stack([np.interp(lookup, source[:, 0], source[:, c]) for c in range(1, 5)])
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)


# 直接调用真实状态机更新，隔离物理状态以验证时间门槛与重试。
# 门限: P1_END*λ = 2.40, P2_DURATION*λ = 0.45 (每步先加 dt 再判定)
@pytest.mark.parametrize("phase,start,ok,expected_phase,expected_retry", [
    (0, 2.38, False, 0, 0),
    (0, 2.38, True, 0, 0),
    (0, 2.39, True, 1, 0),
    (0, 2.39, False, 0, 0),
    (0, 2.69, False, 0, 1),
    (1, 0.43, True, 1, 0),
    (1, 0.45, True, 2, 0),
    (1, 0.75, False, 1, 1),
])
def test_stage_time_gates(phase, start, ok, expected_phase, expected_retry):
    term = SimpleNamespace(
        _update_dt=0.01,
        time_scale_command=torch.tensor([TIME_COMPARISON_SCALE]),
        t_phase=torch.tensor([start]),
        phase=torch.tensor([phase]),
        retry=torch.zeros(1, dtype=torch.long),
        phase_command=torch.zeros(1),
        _env=SimpleNamespace(episode_length_buf=torch.ones(1, dtype=torch.long)),
        _check_S1=lambda: torch.tensor([ok]),
        _check_S2=lambda: torch.tensor([ok]),
    )
    BackupCommand._update_command(term)
    assert term.phase.item() == expected_phase
    assert term.retry.item() == expected_retry
    if expected_phase != phase or expected_retry:
        assert term.t_phase.item() == 0
    else:
        assert term.t_phase.item() == pytest.approx(start + 0.01)


# 训练和回放配置使用固定速度，单次完整动作及等待可容纳于原回合上限。
@pytest.mark.parametrize("play", [False, True])
def test_fixed_scale_and_episode_budget(play):
    cfg = SQuRo_Backup_Env_Cfg(play=play)
    assert cfg.commands["backup_cmd"].fixed_time_scale == TIME_COMPARISON_SCALE
    assert STAND_TRANSITION_END * TIME_COMPARISON_SCALE + 2 * 0.3 + 0.5 < cfg.episode_length_s


# 动作 scale 与手调脚本同口径，且能覆盖期望轨迹极值并留出 clip 冗余。
# 最紧的是两个扭转关节 (期望 ±1.57): 需 action 5.233, clip=6.0 → 余量 1.15x
# 后腿髋角 _HL_HOLD=-1.50 → 需 4.667, 余量 1.29x
def test_action_scale_covers_reference():
    import mjlab.tasks.SQuRo_Backup.mdp.indices as idx_mod
    from mjlab.tasks.SQuRo_Backup.mdp.reference import _generate_reference_table

    cfg = SQuRo_Backup_Env_Cfg()
    scale = float(cfg.actions["joint_pos"].scale)
    clip = 6.0
    assert scale == 0.3

    # 默认关节角: 站立姿态 (events.py reset_model 写入值)
    default = {
        "F_spine1_joint": 0.0, "F_body_joint": 0.0,
        "Neck_yaw_joint": 0.0, "Neck_pitch_joint": 0.0,
        "FL_shoulder_joint": 0.1, "FL_elbow_joint": -0.3,
        "FR_shoulder_joint": 0.1, "FR_elbow_joint": -0.3,
        "H_spine1_joint": 0.0, "H_body_joint": 0.0,
        "HL_hip_joint": -0.1, "HL_knee_joint": 0.3,
        "HR_hip_joint": -0.1, "HR_knee_joint": 0.3,
    }
    order = list(idx_mod._ACTUATED_JOINT_NAMES)
    d = np.array([default[n] for n in order])

    _, ref = _generate_reference_table()
    need = np.abs(np.stack([ref.min(axis=0), ref.max(axis=0)]) - d).max(axis=0) / scale

    # 每个关节都必须落在 clip 之内, 且保留至少 10% 冗余
    assert need.max() <= clip, f"动作超限: {order[int(need.argmax())]} 需 {need.max():.3f} > {clip}"
    assert clip / need.max() >= 1.1, f"冗余不足: {clip / need.max():.2f}x"
    # 两个扭转关节是最紧的, 单独断言防止后续改动悄悄吃掉余量
    for name in ("F_body_joint", "H_body_joint"):
        j = order.index(name)
        assert clip / need[j] >= 1.1, f"{name} 冗余不足: {clip / need[j]:.2f}x"
