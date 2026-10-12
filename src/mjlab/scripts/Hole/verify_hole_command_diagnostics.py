# 用法：uv run python -B -m mjlab.scripts.Hole.verify_hole_command_diagnostics
# 验证模式日志、物理方向修正及空模式的有限数输出。

from __future__ import annotations

import torch

from mjlab.scripts.Hole.verify_hole_command_reset import make_env, write_positions
from mjlab.tasks.SQuRo_Hole.mdp.config import CMD_HEIGHT_F_IDS, CMD_HEIGHT_H_IDS
from mjlab.utils.lab_api.math import quat_from_angle_axis, quat_mul


# 比较日志值与独立构造的预期值。
def check_value(log: dict, key: str, expected: float, atol: float = 1e-4) -> None:
    value = log[key]
    assert torch.is_tensor(value) and torch.isfinite(value), f"{key} 不是有限张量"
    torch.testing.assert_close(value, value.new_tensor(expected), atol=atol, rtol=0, msg=key)


# 用四种已知朝向和速度的状态检查分组日志。
def check_modes_and_directions(env) -> None:
    command = env.command_manager.get_term("hole_cmd")
    command.cfg.stage_schedule_fallback = False
    robot = env.scene["robot"]
    state = torch.zeros(env.num_envs, 13, device=env.device)
    state[:, :7] = robot.data.root_link_pose_w
    angles = torch.deg2rad(torch.tensor([0.0, 30.0, -45.0, 179.0]))
    axes = torch.zeros(env.num_envs, 3)
    axes[:, 2] = 1
    state[:, 3:7] = quat_mul(quat_from_angle_axis(angles, axes), state[:, 3:7])
    state[:, 0] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    state[:, 7] = torch.tensor([0.01, 0.02, 0.03, 0.04])
    state[:, 8] = torch.tensor([0.03, -0.04, 0.05, -0.06])
    robot.write_root_state_to_sim(state)
    env.sim.forward()
    command.command[:, CMD_HEIGHT_F_IDS] = torch.tensor([0.055, 0.02, 0.055, 0.02])
    command.command[:, CMD_HEIGHT_H_IDS] = torch.tensor([0.055, 0.055, 0.02, 0.02])
    command._update_command()
    log = env.extras["log"]
    names = ("双高", "前低后高", "前高后低", "双低")
    for i, name in enumerate(names):
        prefix = f"Metrics/{name}"
        check_value(log, f"{prefix}/占比", 0.25)
        check_value(log, f"{prefix}/样本数", 1.0)
        check_value(log, f"{prefix}/相对X均值", float(state[i, 0]))
        check_value(log, f"{prefix}/前向速度均值", float(state[i, 7]))
        check_value(log, f"{prefix}/侧向速度均值", float(state[i, 8]))
        check_value(log, f"{prefix}/侧向速度绝对值均值", float(state[i, 8].abs()))
        for segment in ("前", "后"):
            check_value(log, f"{prefix}/{segment}躯干方向误差均值_度", float(torch.rad2deg(angles[i]).abs()))
    check_value(log, "Metrics/位移/相对X均值", 0.25)
    check_value(log, "Metrics/位移/相对X最大值", 0.4)
    print("[通过] 四模式分组、位移、速度和前后段朝向修正正确")


# 空模式输出零且保持有限，必须用占比判断是否有样本。
def check_empty_modes(env) -> None:
    command = env.command_manager.get_term("hole_cmd")
    command.command[:, CMD_HEIGHT_F_IDS] = 0.055
    command.command[:, CMD_HEIGHT_H_IDS] = 0.055
    command._update_command()
    log = env.extras["log"]
    check_value(log, "Metrics/双高/占比", 1.0)
    for name in ("前低后高", "前高后低", "双低"):
        prefix = f"Metrics/{name}/"
        for key in (key for key in log if key.startswith(prefix)):
            check_value(log, key, 0.0)
    assert all(torch.isfinite(value) for key, value in log.items() if key.startswith("Metrics/")), "日志存在非有限数"
    print("[通过] 空模式的占比和诊断值均为零，不产生 NaN/Inf")


# 日志必须在位置表更新命令后记录，并保留观测动作维度。
def check_schedule_logging(env) -> None:
    command = env.command_manager.get_term("hole_cmd")
    command.cfg.stage_schedule_fallback = True
    env.reset()
    write_positions(env, (0.199, 0.201, 0.319, 0.321))
    env.command_manager.compute(dt=0.0)
    log = env.extras["log"]
    for name, expected in (("前低后高", 0.25), ("前高后低", 0.5), ("双高", 0.25), ("双低", 0.0)):
        check_value(log, f"Metrics/{name}/占比", expected)
    assert env.observation_manager.compute()["actor"].shape[-1] == 113, "诊断改变了观测维度"
    assert env.action_space.shape[-1] == 14, "诊断改变了动作维度"
    print("[通过] 模式日志反映当帧位置表更新，观测 113 维、动作 14 维")


# 执行真实环境诊断日志验收。
def main() -> None:
    torch.set_num_threads(1)
    env = make_env()
    try:
        check_modes_and_directions(env)
        check_empty_modes(env)
        check_schedule_logging(env)
    finally:
        env.close()
    print("[结论] Hole 模式与运动方向日志验收全部通过")


if __name__ == "__main__":
    main()
