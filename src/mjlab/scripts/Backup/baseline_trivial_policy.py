from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import tyro
from tensordict import TensorDict

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.SQuRo_Backup.mdp.curriculums import (
  _STEPS_PER_ITER,
  reward_weight_curriculum,
)
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES, resolve_model_indices
from mjlab.tasks.SQuRo_Backup.mdp.reference import get_reference_joint_state

TASK = "Mjlab-SQuRo-Backup"
SPINE_NAMES = ("F_spine1", "F_body", "H_spine1", "H_body")


@dataclass
class EvaluationConfig:
  # 所有策略在当前环境配置下重评估，不声称复现旧训练环境。
  checkpoint: Path | None = None
  cases: tuple[Literal["zero", "random", "reference", "trained", "sampled"], ...] = (
    "zero",
    "random",
    "reference",
  )
  num_envs: int = 16
  seeds: tuple[int, ...] = (42, 43, 44)
  episodes_per_env: int = 1
  time_scale: float = 3.0
  curriculum_iteration: int = 0
  random_amplitude: float = 1.0
  device: str = "cuda:0"
  output_dir: Path = Path("logs/backup_evaluation")


# 保存终止步的物理状态，避免 env.step 返回时已被自动重置覆盖。
class TerminalSnapshot(RecorderTerm):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.terminal: dict[str, torch.Tensor] = {}
    self.ids = torch.empty(0, dtype=torch.long, device=env.device)

  def snapshot(self):
    env = self._env
    asset = env.scene.entities["robot"]
    cmd = env.command_manager.get_term("backup_cmd")
    return {
      "joint_pos": asset.data.joint_pos[:, _MODEL_INDICES.joint_ids].clone(),
      "phase": cmd.phase.clone(),
      "retry": cmd.retry.clone(),
      "success": env.termination_manager.get_term("stand").clone(),
    }

  def record_pre_reset(self, env_ids):
    self.ids = env_ids.clone()
    self.terminal = {k: v[env_ids].clone() for k, v in self.snapshot().items()}

  def after_step(self):
    result = self.snapshot()
    for key, value in self.terminal.items():
      result[key][self.ids] = value
    self.terminal = {}
    return result


# 每个并行环境只接收首个完整 episode；结束后忽略后续自动重置回合。
# 每轮等所有环境完成，避免快结束的策略贡献更多样本。
class EpisodeAccumulator:
  def __init__(self, num_envs, names, dt, gamma, device):
    self.names = list(names)
    self.dt, self.gamma = dt, gamma
    self.active = torch.ones(num_envs, dtype=torch.bool, device=device)
    self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.total = torch.zeros(num_envs, dtype=torch.float64, device=device)
    self.discounted = torch.zeros_like(self.total)
    self.parts = torch.zeros(num_envs, len(names), dtype=torch.float64, device=device)
    self.discounted_parts = torch.zeros_like(self.parts)
    self.stage_parts = torch.zeros(
      num_envs, 3, len(names), dtype=torch.float64, device=device
    )
    self.stage_steps = torch.zeros(num_envs, 3, dtype=torch.long, device=device)
    self.first = torch.full((num_envs, 2), -1.0, device=device)
    self.metrics: dict[str, torch.Tensor] = {}
    self.rows: list[dict] = []

  def update(self, reward, parts, terminated, truncated, phase_before, state, metrics):
    torch.testing.assert_close(parts.sum(-1), reward, atol=1e-5, rtol=1e-5)
    active = self.active
    self.discounted += active * (self.gamma**self.steps) * reward
    self.discounted_parts += (active * (self.gamma**self.steps))[:, None] * parts
    self.total += active * reward
    self.parts += active[:, None] * parts
    self.steps += active.long()
    for stage in range(3):
      mask = active & (phase_before == stage)
      self.stage_parts[:, stage] += mask[:, None] * parts
      self.stage_steps[:, stage] += mask.long()
    for stage in (1, 2):
      reached = active & (state["phase"] >= stage) & (self.first[:, stage - 1] < 0)
      self.first[reached, stage - 1] = self.steps[reached] * self.dt
    for name, value in metrics.items():
      if name not in self.metrics:
        self.metrics[name] = torch.zeros_like(self.total)
      self.metrics[name] += active * value
    finished = active & (terminated | truncated)
    for index in finished.nonzero().flatten().tolist():
      length = int(self.steps[index])
      duration = length * self.dt
      row = {
        "env_id": index,
        "steps": length,
        "duration_s": duration,
        "return": float(self.total[index]),
        "discounted_return": float(self.discounted[index]),
        "return_per_second": float(self.total[index]) / duration,
        "success": bool(state["success"][index]),
        "terminated": bool(terminated[index]),
        "timeout": bool(truncated[index]),
        "retry_count": int(state["retry"][index]),
        "s1_time_s": None if self.first[index, 0] < 0 else float(self.first[index, 0]),
        "s2_time_s": None if self.first[index, 1] < 0 else float(self.first[index, 1]),
        "reward_parts": dict(zip(self.names, self.parts[index].tolist(), strict=True)),
        "discounted_reward_parts": dict(
          zip(self.names, self.discounted_parts[index].tolist(), strict=True)
        ),
        "phase_reward_parts": {
          f"P{s + 1}": dict(
            zip(self.names, self.stage_parts[index, s].tolist(), strict=True)
          )
          for s in range(3)
        },
        "phase_duration_s": {
          f"P{s + 1}": int(self.stage_steps[index, s]) * self.dt for s in range(3)
        },
        "metrics_mean": {k: float(v[index]) / length for k, v in self.metrics.items()},
      }
      row["spine_rmse_rad"] = (
        {
          name: math.sqrt(row["metrics_mean"][f"sq_error_{name}"])
          for name in SPINE_NAMES
        }
        if all(f"sq_error_{n}" in self.metrics for n in SPINE_NAMES)
        else {}
      )
      # 只对同一条轨迹重计分，不代表取消平滑后重新训练的结果。
      row["same_trajectory_return_without_smoothing"] = row["return"] - sum(
        row["reward_parts"].get(k, 0.0) for k in ("action_L1", "action_L2")
      )
      self.rows.append(row)
    self.active = active & ~finished


def distribution(values):
  if not values:
    return {"n": 0, "mean": None, "std": None}
  data = torch.tensor(values, dtype=torch.float64)
  return {
    "n": len(values),
    "mean": float(data.mean()),
    "std": float(data.std(unbiased=False)),
  }


def summarize(rows):
  return {
    "episodes": len(rows),
    "success_rate": sum(r["success"] for r in rows) / len(rows),
    "s1_rate": sum(r["s1_time_s"] is not None for r in rows) / len(rows),
    "s2_rate": sum(r["s2_time_s"] is not None for r in rows) / len(rows),
    "timeout_rate": sum(r["timeout"] for r in rows) / len(rows),
    **{
      key: distribution([r[key] for r in rows])
      for key in (
        "return",
        "discounted_return",
        "duration_s",
        "return_per_second",
        "retry_count",
        "same_trajectory_return_without_smoothing",
      )
    },
    "success_duration_s": distribution([r["duration_s"] for r in rows if r["success"]]),
    "success_return": distribution([r["return"] for r in rows if r["success"]]),
    "failure_return": distribution([r["return"] for r in rows if not r["success"]]),
    "s1_time_s": distribution(
      [r["s1_time_s"] for r in rows if r["s1_time_s"] is not None]
    ),
    "s2_time_s": distribution(
      [r["s2_time_s"] for r in rows if r["s2_time_s"] is not None]
    ),
    "reward_parts": {
      k: distribution([r["reward_parts"][k] for r in rows])
      for k in rows[0]["reward_parts"]
    },
    "metrics_mean": {
      k: distribution([r["metrics_mean"][k] for r in rows])
      for k in rows[0]["metrics_mean"]
    },
    "spine_rmse_rad": {
      k: distribution([r["spine_rmse_rad"][k] for r in rows])
      for k in rows[0]["spine_rmse_rad"]
    },
  }


# 首步跳变单独记录，避免与持续颤抖混淆；其余变化量使用已裁剪动作。
def action_metrics(raw, applied, previous, before_previous, step, scale):
  delta = applied - previous
  running = (step > 0).float()
  return {
    "clip_fraction": (raw != applied).float().mean(-1),
    "action_delta_l1_sum": delta.abs().sum(-1),
    "action_delta_l2_sum": delta.square().sum(-1),
    "target_delta_l1_rad_sum": (scale * delta).abs().sum(-1),
    "target_delta_l2_rad2_sum": (scale * delta).square().sum(-1),
    "ongoing_action_delta_l2_sum": running * delta.square().sum(-1),
    "initial_action_delta_l2_sum": (1 - running) * delta.square().sum(-1),
    "action_second_difference_l2_sum": (step > 1).float()
    * (applied - 2 * previous + before_previous).square().sum(-1),
  }


@torch.inference_mode()
def evaluate(cfg: EvaluationConfig):
  if cfg.num_envs < 1 or cfg.episodes_per_env < 1 or not cfg.seeds or not cfg.cases:
    raise ValueError("环境数、每环境回合数、种子和策略列表不能为空或小于 1")
  if cfg.time_scale <= 0 or cfg.random_amplitude < 0 or cfg.curriculum_iteration < 0:
    raise ValueError("时间缩放必须为正，随机幅度和课程轮次必须非负")
  if any(c in cfg.cases for c in ("trained", "sampled")) and cfg.checkpoint is None:
    raise ValueError("评估训练策略必须传入 --checkpoint")
  if cfg.checkpoint is not None and not cfg.checkpoint.is_file():
    raise FileNotFoundError(cfg.checkpoint)
  agent_cfg = load_rl_cfg(TASK)
  env_cfg = load_env_cfg(TASK)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seeds[0]
  env_cfg.commands["backup_cmd"].fixed_time_scale = cfg.time_scale
  env_cfg.recorders["evaluation_terminal"] = RecorderTermCfg(func=TerminalSnapshot)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  try:
    asset = env.scene.entities["robot"]
    resolve_model_indices(asset)
    recorder = env.recorder_manager.get_term("evaluation_terminal")
    scale = float(env_cfg.actions["joint_pos"].scale)
    clip = agent_cfg.clip_actions
    gamma = agent_cfg.algorithm.gamma
    dimension = env.action_manager.action.shape[-1]
    default = asset.data.default_joint_pos[:, _MODEL_INDICES.joint_ids]
    actor = None
    if any(c in cfg.cases for c in ("trained", "sampled")):
      wrapper = RslRlVecEnvWrapper(env, clip_actions=clip)
      runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
      runner = runner_cls(wrapper, asdict(agent_cfg), log_dir=None, device=cfg.device)
      runner.load(
        str(cfg.checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=cfg.device,
      )
      actor = runner.get_inference_policy(device=cfg.device)
      actor.eval()
      normalizer_before = {
        k: v.clone() for k, v in actor.obs_normalizer.state_dict().items()
      }
    fixed_counter = cfg.curriculum_iteration * _STEPS_PER_ITER
    results = {}
    for case in cfg.cases:
      rows = []
      for seed in cfg.seeds:
        for repeat in range(cfg.episodes_per_env):
          run_seed = seed + repeat * 100003
          env.common_step_counter = fixed_counter
          obs, _ = env.reset(seed=run_seed)
          # 站立计数器不是 manager buffer；评估轮次间显式清零。
          if hasattr(env, "_stand_steps"):
            env._stand_steps.zero_()
          generator = torch.Generator(device=cfg.device).manual_seed(run_seed)
          stats = EpisodeAccumulator(
            cfg.num_envs,
            env.reward_manager.active_terms,
            env.step_dt,
            gamma,
            cfg.device,
          )
          previous = torch.zeros(cfg.num_envs, dimension, device=cfg.device)
          before_previous = previous.clone()
          mean_previous = previous.clone()
          recorder.terminal = {}
          for _ in range(env.max_episode_length):
            env.common_step_counter = fixed_counter
            cmd = env.command_manager.get_term("backup_cmd")
            phase_before = cmd.phase.clone()
            reference, _ = get_reference_joint_state(env)
            reference = reference.clone()
            if case == "zero":
              raw = torch.zeros_like(previous)
            elif case == "random":
              raw = cfg.random_amplitude * (
                2 * torch.rand(previous.shape, device=cfg.device, generator=generator)
                - 1
              )
            elif case == "reference":
              raw = (reference - default) / scale
            else:
              # 刷新分布以读取当前熵/std；确定性评估仅下发均值，丢弃样本。
              raw = actor(
                TensorDict(obs, batch_size=[cfg.num_envs]), stochastic_output=True
              )
              if case == "trained":
                raw = actor.output_mean
            applied = raw.clamp(-clip, clip) if clip is not None else raw
            metrics = action_metrics(
              raw, applied, previous, before_previous, stats.steps, scale
            )
            if actor is not None and case in ("trained", "sampled"):
              metrics["policy_std_mean"] = actor.output_std.mean(-1).expand(
                cfg.num_envs
              )
              metrics["policy_entropy"] = actor.output_entropy.expand(cfg.num_envs)
              mean_action = actor.output_mean
              if clip is not None:
                mean_action = mean_action.clamp(-clip, clip)
              metrics["policy_mean_delta_l2_sum"] = (
                (mean_action - mean_previous).square().sum(-1)
              )
              mean_previous = mean_action.clone()
              for j, name in zip((0, 1, 8, 9), SPINE_NAMES, strict=True):
                metrics[f"policy_std_{name}"] = actor.output_std[:, j]
            obs, reward, terminated, truncated, _ = env.step(applied)
            state = recorder.after_step()
            for j, name in zip((0, 1, 8, 9), SPINE_NAMES, strict=True):
              metrics[f"sq_error_{name}"] = (
                state["joint_pos"][:, j] - reference[:, j]
              ).square()
            # _step_reward 保留终止步值，且尚未乘 dt；不可累计 Episode_Reward 日志。
            manager = env.reward_manager
            parts = manager._step_reward.clone() * (
              env.step_dt if manager._scale_by_dt else 1.0
            )
            stats.update(
              reward, parts, terminated, truncated, phase_before, state, metrics
            )
            before_previous, previous = previous, applied.clone()
            if not stats.active.any():
              break
          if stats.active.any():
            raise RuntimeError("到达最大评估时长仍有未结束 episode，不输出不完整统计")
          for row in stats.rows:
            row.update(seed=seed, repeat=repeat, reset_seed=run_seed)
          rows.extend(stats.rows)
      results[case] = {"summary": summarize(rows), "episodes": rows}
      summary = results[case]["summary"]
      print(
        f"[{case}] 回合={len(rows)} S1={summary['s1_rate']:.1%} S2={summary['s2_rate']:.1%} "
        f"成功={summary['success_rate']:.1%} 回报={summary['return']['mean']:.3f}",
        flush=True,
      )
    if actor is not None:
      for key, value in actor.obs_normalizer.state_dict().items():
        torch.testing.assert_close(value, normalizer_before[key], rtol=0, atol=0)
    return {
      "metadata": {
        "evaluation": asdict(cfg),
        "environment": asdict(env_cfg),
        "agent": asdict(agent_cfg),
        "step_dt": env.step_dt,
        "action_scale": scale,
        "clip_actions": clip,
        "gamma": gamma,
        "frozen_curriculum_counter": fixed_counter,
        "curriculum_weights": reward_weight_curriculum.get_reward_weights(
          fixed_counter
        ),
        "note": "当前配置重评估；固定仰面初态，无初态随机化；多种子不等于姿态泛化测试。阶段统计按动作下发前阶段归属。",
      },
      "results": results,
    }
  finally:
    env.close()


def json_safe(value):
  # 配置中的无限边界用字符串保留；评估数值本身仍通过严格 JSON 检查。
  if isinstance(value, dict):
    return {str(k): json_safe(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [json_safe(v) for v in value]
  if isinstance(value, float) and not math.isfinite(value):
    return str(value)
  if callable(value):
    return f"{value.__module__}.{value.__qualname__}"
  return value


def main():
  cfg = tyro.cli(EvaluationConfig)
  # 仿真前确认目录可写，避免长时间评估后才暴露权限问题。
  output = cfg.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
  output.mkdir(parents=True, exist_ok=False)
  result = evaluate(cfg)
  result["metadata"] = json_safe(result["metadata"])
  path = output / "evaluation.json"
  path.write_text(
    json.dumps(result, ensure_ascii=False, indent=2, default=str, allow_nan=False),
    encoding="utf-8",
  )
  print(f"评估结果：{path.resolve()}")


if __name__ == "__main__":
  main()
