# uv run python -B -m mjlab.scripts.Slalom.verify_slalom_env_smoke
# 绕杆端到端自检: 真实环境构造 + 两阶段运行 + 阶段混合窗口 + 重置边界语义
# 与 verify_slalom_ref_consistency (纯桩环境, 不建 env) 互补, 都通过才算基线可用

import sys

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Slalom.mdp import reference as ref_mod
from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import _STEPS_PER_ITER, PHASE1_END_ITER
from mjlab.tasks.SQuRo_Slalom.mdp.path import compute_path_ref, get_approach_start

TASK_NAME = "Mjlab-SQuRo-Slalom"
PHASE1_STEP = PHASE1_END_ITER * _STEPS_PER_ITER
NUM_ENVS = 2
FAILURES: list[str] = []


# 记录一条失败 (脚本末尾统一给结论)
def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"    ✗ {msg}")


# 检查一批张量是否全部有限
def check_finite(tag: str, *tensors) -> None:
    for i, t in enumerate(tensors):
        if t is None or not bool(torch.isfinite(t).all()):
            fail(f"{tag}: 第 {i} 个张量出现非有限值")
            return


# 建立 2 环境裸环境 (不渲染, 不录视频)
def build_env():
    cfg = load_env_cfg(TASK_NAME, play=True)
    cfg.scene.num_envs = NUM_ENVS
    cfg.viewer.height = 240
    cfg.viewer.width = 320
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
    u = env.unwrapped if hasattr(env, "unwrapped") else env
    print(f"    device={device}, num_envs={u.num_envs}")
    return env, u


# 单步并回读观测/奖励/参考的有限性
def step_and_check(u, act, tag: str):
    obs, rew, dones, timeouts, extras = u.step(act)
    actor_obs = obs["actor"] if isinstance(obs, dict) else obs
    check_finite(f"{tag} obs", actor_obs)
    check_finite(f"{tag} reward", rew)
    check_finite(f"{tag} path_kappa", getattr(u, "_path_kappa", None))
    ref_pos, ref_vel = ref_mod.get_reference_joint_state(u)
    check_finite(f"{tag} ref_pos", ref_pos)
    check_finite(f"{tag} ref_vel", ref_vel)
    return obs, rew, extras


# [1] Phase 0: 圆弧基元阶段能构造并稳定运行
def check_phase0(u, act) -> None:
    print("\n[1] Phase 0 (圆弧基元)")
    cmd_term = u.command_manager._terms["slalom_cmd"]
    if cmd_term.slalom_mode_active or bool(cmd_term.phase1_mask.any()):
        fail("初始计数器下不应处于 Phase 1")
    step_and_check(u, act, "Phase0 首步")
    for _ in range(20):
        obs, rew, extras = step_and_check(u, act, "Phase0")
    print(f"    kappa={u._path_kappa.tolist()} | vel_cmd={cmd_term.command[0, 0].item():.4f} "
          f"| obs 维数={tuple(obs['actor'].shape) if isinstance(obs, dict) else 'n/a'}")


# [2] Phase 1: 绕杆阶段能进入并沿路径推进
def check_phase1(u, act) -> None:
    print("\n[2] Phase 1 (绕杆)")
    u.common_step_counter = PHASE1_STEP
    u.reset()
    cmd_term = u.command_manager._terms["slalom_cmd"]
    if not bool(cmd_term.phase1_mask.all()):
        fail("对齐到 Phase 1 后 phase1_mask 不全为真")
    print(f"    杆间距={cmd_term.active_pole_spacing:.4f} m")

    x_first = None
    for i in range(60):
        obs, rew, extras = step_and_check(u, act, "Phase1")
        if i == 0:
            x_ref, _, _, _, _ = compute_path_ref(u)
            x_first = float(x_ref[0])
        if i == 59:
            x_ref, _, vx, vy, _ = compute_path_ref(u)
            print(f"    kappa=[{u._path_kappa.min().item():.2f}, {u._path_kappa.max().item():.2f}] "
                  f"| x_ref: {x_first:.5f} → {float(x_ref[0]):.5f} "
                  f"| vel_cmd={cmd_term.command[0, 0].item():.4f} "
                  f"| |v_des|={float(torch.hypot(vx[0], vy[0])):.4f}")
            if float(x_ref[0]) <= x_first:
                fail("Phase 1 参考位置未推进")
            # 参考推进速度必须与自身速度命令一致 (名义时间推进的正确性)
            if abs(float(torch.hypot(vx[0], vy[0])) - float(cmd_term.command[0, 0])) > 1e-4:
                fail("期望速度与速度命令不一致")


# [3] 重置边界: 超时重置的那一步, 观测必须已属于新回合
def check_reset_boundary(u, act) -> None:
    print("\n[3] 重置边界 (超时重置)")
    cmd_term = u.command_manager._terms["slalom_cmd"]
    u.reset()
    u.episode_length_buf[:] = u.max_episode_length - 1
    step_and_check(u, act, "重置步")

    if not bool(u.reset_buf.all()):
        fail("强制超时后 reset_buf 未标记全体环境")
    expected_x = get_approach_start(spacing=cmd_term.active_pole_spacing)[0]
    x_ref, _, _, _, _ = compute_path_ref(u)
    got_x = float(x_ref[0])
    if abs(got_x - expected_x) > 1e-4:
        fail(f"重置首帧参考不属新回合: x_ref={got_x:.5f}, 应为接近段起点 {expected_x:.5f}")
    phase = u._ref_phase
    if float(phase.abs().max()) > 1e-6:
        fail(f"重置首帧步态相位未清零: {phase.tolist()}")
    print(f"    步态相位={phase.tolist()} | x_ref={got_x:.5f} (接近段起点 {expected_x:.5f})")

    # 之后再走两步: 相位每步只推进一格, 不得被重算推进两次
    step_and_check(u, act, "重置后第 2 步")
    step_and_check(u, act, "重置后第 3 步")
    gait = float(cmd_term.command[0, 3])
    step_phase = gait * float(u.step_dt)
    expect_phase = (2 * step_phase) % 1.0
    # _ref_phase 每步是重新绑定的新张量, 必须重新读取而不是复用上面的引用
    phase_now = u._ref_phase
    if abs(float(phase_now[0]) - expect_phase) > 1e-5:
        fail(f"重置后相位推进异常: {float(phase_now[0]):.5f}, 应为 {expect_phase:.5f} (每步 {step_phase:.5f})")
    print(f"    重置后 2 步相位={float(phase_now[0]):.5f} (期望 {expect_phase:.5f})")


# [4] 阶段混合窗口: 旧阶段回合与新阶段回合同批时, 各自走自己的路径
def check_mixed_phase(u, act) -> None:
    print("\n[4] 阶段混合窗口")
    cmd_term = u.command_manager._terms["slalom_cmd"]
    u.common_step_counter = PHASE1_STEP + 130
    # 人为制造失同步: env0 的回合始于边界之前, env1 始于边界之后
    u.episode_length_buf = torch.tensor([200, 50], dtype=torch.long, device=u.device)
    mask = cmd_term.phase1_mask.tolist()
    print(f"    phase1_mask={mask} (env0 边界前起步, env1 边界后起步)")
    if mask != [False, True]:
        fail(f"混合窗口的 phase1_mask 不正确: {mask}")

    step_and_check(u, act, "混合窗口")
    # 旧阶段回合保持重置时写入的命令, 新阶段回合跟随路径曲率
    print(f"    曲率命令={[round(v, 3) for v in cmd_term.command[:, 4].tolist()]} "
          f"| 速度命令={[round(v, 4) for v in cmd_term.command[:, 0].tolist()]}")
    if mask == [False, True]:
        # 旧阶段的速度不应等于新阶段公式 (直行段 base*gait*0.5) 与旧公式 (base*gait*1.0) 的差
        if abs(float(cmd_term.command[0, 0]) - float(cmd_term.command[1, 0])) < 1e-9:
            fail("混合窗口中两个环境的命令完全相同, 逐环境阶段可能未生效")


def main() -> int:
    print("绕杆端到端自检")
    env, u = build_env()
    act = torch.zeros(u.num_envs, 14, device=u.device)
    try:
        check_phase0(u, act)
        check_phase1(u, act)
        check_reset_boundary(u, act)
        check_mixed_phase(u, act)
    finally:
        env.close()

    print("\n[结论]", "全部通过" if not FAILURES else f"{len(FAILURES)} 项失败")
    for msg in FAILURES:
        print("  -", msg)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
