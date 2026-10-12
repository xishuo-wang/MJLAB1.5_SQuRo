# 用法：uv run python -B -m mjlab.scripts.Hole.verify_hole_index_equivalence
# 对照第一批提交验证索引重构后的参考、复位与全部已注册奖励。

from __future__ import annotations

import argparse
import ast
import builtins
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Hole.mdp import events, reference, rewards
from mjlab.tasks.SQuRo_Hole.mdp import curriculums
from mjlab.tasks.SQuRo_Hole.mdp.config import (
    CMD_HEIGHT_F_IDS,
    CMD_HEIGHT_H_IDS,
    FULL_HEIGHT_VALUES,
    STAGE1_END,
    STAGE2_END,
    STAGE3_END,
)
from mjlab.tasks.SQuRo_Hole.mdp.indices import (
    _ACT_F_LEG_IDS,
    _ACT_NECK_IDS,
    _ACT_SPN_IDS,
    _MODEL_INDICES,
    FOOT_GEOM_NAMES,
    resolve_actuator_joint_ids,
    resolve_foot_force_ids,
)

BASELINE_COMMIT = "929a2a3"
_TASK = "Mjlab-SQuRo-Hole"
_REPO_ROOT = Path(__file__).resolve().parents[4]


# 从指定提交装载对照模块，保留当前工作区。
def load_baseline(name: str, revision: str, dependencies: dict[str, ModuleType]) -> ModuleType:
    path = f"src/mjlab/tasks/SQuRo_Hole/mdp/{name}.py"
    source = subprocess.check_output(
        ["git", "show", f"{revision}:{path}"], cwd=_REPO_ROOT,
        text=True, encoding="utf-8",
    )
    module = ModuleType(f"mjlab.tasks.SQuRo_Hole.mdp._baseline_{name}")
    module.__package__ = "mjlab.tasks.SQuRo_Hole.mdp"
    module.__file__ = str(_REPO_ROOT / path)
    # 对照模块的任务内导入绑定到同一历史版本。
    def import_baseline(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 1 and name in dependencies:
            return dependencies[name]
        return builtins.__import__(name, globals, locals, fromlist, level)

    module.__dict__["__builtins__"] = {**vars(builtins), "__import__": import_baseline}
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


# 要求张量逐位一致，非有限数直接失败。
def check_equal(label: str, before: torch.Tensor, after: torch.Tensor) -> None:
    assert before.shape == after.shape, f"{label}：形状不同"
    assert torch.isfinite(before).all() and torch.isfinite(after).all(), f"{label}：存在非有限数"
    assert torch.equal(before, after), f"{label}：最大差 {(before - after).abs().max().item()}"


# 比较完整参考表及每条现有课程参数的四阶段取值。
def check_tables(old_reference: ModuleType, old_curriculums: ModuleType) -> None:
    before = old_reference.Initialize_Tables("cpu")
    after = reference.Initialize_Tables("cpu")
    count = 0
    for name, value in before.items():
        if torch.is_tensor(value):
            check_equal(f"参考表 {name}", value, after[name])
            count += value.numel()
        else:
            assert value == after[name], f"参考表元信息 {name} 不同"
    probes = (1, STAGE1_END + 1, STAGE2_END + 1, STAGE3_END + 1)
    for step in probes:
        env = SimpleNamespace(common_step_counter=step)
        keys = set(old_curriculums._CURVES) | set(curriculums._CURVES)
        for key in keys:
            assert (old_curriculums.get_curriculum_reward_weight(env, key)
                    == curriculums.get_curriculum_reward_weight(env, key)), f"课程参数 {key} 不同"
    print(f"[通过] 完整参考表 {count} 个标量及四阶段课程参数逐位一致")


# 检查所有奖励查询键，并模拟脊柱参数被删或拼错。
def check_curriculum_keys() -> None:
    tree = ast.parse(Path(rewards.__file__).read_text(encoding="utf-8"))
    keys = {
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "get_curriculum_reward_weight"
        and len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
    }
    assert keys <= curriculums._CURVES.keys(), f"奖励缺少课程定义：{keys - curriculums._CURVES.keys()}"
    env = SimpleNamespace(common_step_counter=0)
    original = curriculums._CURVES.copy()
    for key in ("sigma_spn_pos_typo", "sigma_spn_pos", "sigma_spn_vel"):
        try:
            curriculums._CURVES.pop(key, None)
            try:
                curriculums.get_curriculum_reward_weight(env, key)
            except KeyError as error:
                assert key in str(error), "缺键错误没有指出参数名称"
            else:
                raise AssertionError(f"缺少 {key} 时仍静默返回数值")
        finally:
            curriculums._CURVES.clear()
            curriculums._CURVES.update(original)
    print(f"[通过] 全部 {len(keys)} 个奖励查询键有定义；脊柱 σ 缺失或拼错时明确报错")


# 比较真实模型上的复位写入和足端力名称映射。
def check_reset(env: ManagerBasedRlEnv, old_events: ModuleType) -> None:
    robot = env.scene["robot"]
    env_ids = torch.arange(env.num_envs)
    old_events.reset_model(env, env_ids)
    env.sim.forward()
    before = (robot.data.joint_pos.clone(), robot.data.joint_vel.clone(),
              robot.data.root_link_pos_w.clone(), robot.data.root_link_quat_w.clone())
    events.reset_model(env, env_ids)
    env.sim.forward()
    after = (robot.data.joint_pos, robot.data.joint_vel,
             robot.data.root_link_pos_w, robot.data.root_link_quat_w)
    for i, (left, right) in enumerate(zip(before, after, strict=True)):
        check_equal(f"复位状态 {i}", left, right)
    sensor = env.scene["feet_ground_contact"]
    ids = resolve_foot_force_ids(sensor)
    assert tuple(sensor.primary_names[i] for i in ids) == FOOT_GEOM_NAMES, "足端力列名错位"
    reordered = SimpleNamespace(primary_names=list(reversed(sensor.primary_names)))
    ids = resolve_foot_force_ids(reordered)
    assert tuple(reordered.primary_names[i] for i in ids) == FOOT_GEOM_NAMES, "足端力依赖传感器列序"
    print("[通过] 真实模型复位状态逐位一致，足端力支持传感器列序变化")


# 在非对称关节状态及全部高度组合下对照每个已注册奖励。
def check_rewards(env: ManagerBasedRlEnv, old_reference: ModuleType, old_rewards: ModuleType) -> None:
    robot = env.scene["robot"]
    command = env.command_manager.get_command("hole_cmd")
    heights = torch.tensor(FULL_HEIGHT_VALUES)
    pairs = torch.cartesian_prod(heights, heights)
    base_pos = robot.data.joint_pos.clone()
    term_names = [term.func.__name__ for term in env.cfg.rewards.values()]
    phases = (0, 1, 25, 49, 125, 499)
    old_rewards.get_reference_joint_state = old_reference.get_reference_joint_state
    count = 0
    for profile in ("前腿运动", "脊柱颈部运动", "非对称运动"):
        velocity = torch.zeros_like(base_pos)
        if profile == "前腿运动":
            velocity[:, resolve_actuator_joint_ids(robot, _ACT_F_LEG_IDS)] = 1.0
        elif profile == "脊柱颈部运动":
            velocity[:, resolve_actuator_joint_ids(robot, _ACT_SPN_IDS + _ACT_NECK_IDS)] = 1.0
        else:
            velocity[:] = torch.linspace(-1.3, 1.7, robot.num_joints)
        position = base_pos + torch.linspace(-0.02, 0.02, robot.num_joints)
        robot.write_joint_state_to_sim(position, velocity)
        root = torch.zeros(env.num_envs, 13)
        root[:, 0] = torch.linspace(0.0, 1.3, env.num_envs)
        root[:, 2] = 0.06
        root[:, 4:6] = -0.707107
        robot.write_root_state_to_sim(root)
        env.sim.forward()
        command[:, CMD_HEIGHT_F_IDS] = pairs[:, 0]
        command[:, CMD_HEIGHT_H_IDS] = pairs[:, 1]
        for step in (1, STAGE1_END + 1, STAGE2_END + 1, STAGE3_END + 1):
            env.common_step_counter = step
            for phase in phases:
                env.episode_length_buf[:] = phase
                old_pos, old_vel = old_reference.get_reference_joint_state(env)
                new_pos, new_vel = reference.get_reference_joint_state(env)
                check_equal("参考位置", old_pos, new_pos)
                check_equal("参考速度", old_vel, new_vel)
                for name in term_names:
                    before = getattr(old_rewards, name)(env)
                    after = getattr(rewards, name)(env)
                    check_equal(f"{profile}/{step}/{phase}/{name}", before, after)
                    count += before.numel()
        command[:, CMD_HEIGHT_F_IDS] = 0.055
        command[:, CMD_HEIGHT_H_IDS] = 0.02
        expected = 0.0 if profile == "前腿运动" else -4.0
        if profile != "非对称运动":
            assert torch.all(rewards.compute_stop_reward(env) == expected), "stop 未正确区分前腿与脊柱颈部"
    print(f"[通过] 三种非对称状态、四阶段、25 高度组合、六相位：奖励 {count} 个标量逐位一致")


# 重排身体数组后验证奖励不依赖固定身体编号。
def check_body_permutation(env: ManagerBasedRlEnv) -> None:
    robot = env.scene["robot"]
    pairs = torch.cartesian_prod(torch.tensor(FULL_HEIGHT_VALUES), torch.tensor(FULL_HEIGHT_VALUES))
    command = env.command_manager.get_command("hole_cmd")
    command[:, CMD_HEIGHT_F_IDS] = pairs[:, 0]
    command[:, CMD_HEIGHT_H_IDS] = pairs[:, 1]
    original_ids = (_MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id)
    functions = (rewards.compute_angle_reward, rewards.compute_height_reward,
                 rewards.compute_body_contact_reward)
    before = [function(env) for function in functions]
    order = torch.arange(robot.data.body_link_pos_w.shape[1] - 1, -1, -1)
    data = SimpleNamespace(
        body_link_pos_w=robot.data.body_link_pos_w[:, order],
        body_link_quat_w=robot.data.body_link_quat_w[:, order],
        site_pos_w=robot.data.site_pos_w,
    )
    proxy = SimpleNamespace(
        num_envs=env.num_envs, device=env.device, scene={"robot": SimpleNamespace(data=data)},
        command_manager=env.command_manager, common_step_counter=env.common_step_counter,
        extras={"log": {}},
    )
    try:
        _MODEL_INDICES.f_body_id = int((order == original_ids[0]).nonzero()[0])
        _MODEL_INDICES.h_body_id = int((order == original_ids[1]).nonzero()[0])
        for function, expected in zip(functions, before, strict=True):
            check_equal(f"身体编号重排/{function.__name__}", expected, function(proxy))
    finally:
        _MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id = original_ids
    print("[通过] 身体编号重排不改变高度、角度与虚拟碰撞奖励")


# 执行基线数值对照。
def main() -> None:
    parser = argparse.ArgumentParser(description="Hole 索引与奖励基线等价验证")
    parser.add_argument("--baseline", default=BASELINE_COMMIT)
    args = parser.parse_args()
    modules: dict[str, ModuleType] = {}
    for name in ("config", "indices", "reference", "curriculums", "events", "rewards"):
        modules[name] = load_baseline(name, args.baseline, modules)
    old_reference = modules["reference"]
    old_curriculums = modules["curriculums"]
    old_events = modules["events"]
    old_rewards = modules["rewards"]
    check_tables(old_reference, old_curriculums)
    check_curriculum_keys()
    cfg = load_env_cfg(_TASK)
    cfg.scene.num_envs = len(FULL_HEIGHT_VALUES) ** 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        with torch.inference_mode():
            env.reset()
            check_reset(env, old_events)
            check_rewards(env, old_reference, old_rewards)
            check_body_permutation(env)
    finally:
        env.close()
    print(f"[结论] 与基线 {args.baseline} 的已注册奖励及参考数值逐位一致")


if __name__ == "__main__":
    main()
