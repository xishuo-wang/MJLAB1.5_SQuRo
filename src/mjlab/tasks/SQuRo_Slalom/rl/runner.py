import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from rsl_rl.utils import check_nan

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import (
  SLALOM_CURVATURE_MIN, SPACING_EVALUATION_ENVS, SPACING_SUCCESS_BATCH,
  SPACING_REQUIRED_BATCHES, PHASE0_EPISODE_SECONDS,
)
from mjlab.tasks.SQuRo_Slalom.mdp.pole import configure_pole_collision, pole_collision_enabled
from mjlab.tasks.SQuRo_Slalom.mdp.slalom_geometry import get_slalom_geometry


class SQuRoOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  # 训练环境保持无杆碰撞，独立考核环境在第一次考核时创建并复用。
  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    super().__init__(env, train_cfg, log_dir, device)
    self._evaluation_env = None
    self._command = env.unwrapped.command_manager._terms["slalom_cmd"]

  # 显式固定条件与回放不自动推进杆距课程。
  def _course_enabled(self) -> bool:
    cfg = self._command.cfg
    return (cfg.enable_spacing_curriculum and not cfg.phase0_only
            and cfg.fixed_slalom_curvature is None and cfg.fixed_pole_spacing is None)

  # 独立场景保留当前机器人和命令参数，只打开杆碰撞并固定考核档位。
  def _get_evaluation_env(self):
    if self._evaluation_env is None:
      cfg = copy.deepcopy(self.env.unwrapped.cfg)
      cfg.scene.num_envs = SPACING_EVALUATION_ENVS
      cfg.commands["slalom_cmd"].enable_spacing_curriculum = False
      cfg.commands["slalom_cmd"].phase0_only = False
      cfg.commands["slalom_cmd"].debug_vis = False
      configure_pole_collision(cfg, True)
      raw = ManagerBasedRlEnv(cfg, device=str(self.env.device), render_mode=None)
      raw.common_step_counter = self.env.unwrapped.common_step_counter
      self._evaluation_env = RslRlVecEnvWrapper(raw, clip_actions=self.env.clip_actions)
      if not pole_collision_enabled(raw):
        raise RuntimeError("碰撞考核场景未真正开启杆碰撞")
    return self._evaluation_env

  # 每个环境每轮只计第一个完整回合，避免快速跌倒者被重复计分。
  def _run_collision_evaluation(self) -> tuple[list[dict], float]:
    started = time.perf_counter()
    numpy_state, python_state = np.random.get_state(), random.getstate()
    devices = [self.env.device.index or 0] if self.env.device.type == "cuda" else []
    try:
      with torch.random.fork_rng(devices=devices), torch.no_grad():
        evaluation = self._get_evaluation_env()
        env = evaluation.unwrapped
        command = env.command_manager._terms["slalom_cmd"]
        command.cfg.fixed_slalom_curvature = self._command.spacing_curriculum.max_curvature
        torch.manual_seed(int(self.cfg.get("seed", 42)) + self.current_learning_iteration)
        policy = self.get_inference_policy(device=str(evaluation.device))
        results: list[dict] = []
        while len(results) < SPACING_SUCCESS_BATCH:
          env.common_step_counter = self.env.unwrapped.common_step_counter
          obs, _ = evaluation.reset()
          seen: set[int] = set()
          monitor = getattr(env, "_slalom_passage_monitor", None)
          if monitor is not None:
            monitor.consume()
          for _ in range(env.max_episode_length + 1):
            obs, _, _, _ = evaluation.step(policy(obs.to(self.device)).to(evaluation.device))
            monitor = getattr(env, "_slalom_passage_monitor", None)
            for item in monitor.consume() if monitor is not None else []:
              if item["env_id"] not in seen:
                seen.add(item["env_id"])
                if len(results) < SPACING_SUCCESS_BATCH:
                  results.append(item)
            if len(seen) == evaluation.num_envs:
              break
          if len(seen) != evaluation.num_envs:
            raise RuntimeError("真实碰撞考核缺少完整回合，拒绝推进课程")
          print(f"[绕杆考核] 已完成 {len(results)}/{SPACING_SUCCESS_BATCH} 个真实碰撞回合", flush=True)
        return results, time.perf_counter() - started
    finally:
      np.random.set_state(numpy_state)
      random.setstate(python_state)
      self.alg.train_mode()

  # 把真实碰撞成绩用于课程；考核轨迹不会送入 PPO 的 rollout 缓冲。
  def _maybe_evaluate_curriculum(self, iteration: int) -> None:
    course = self._command.spacing_curriculum
    if not self._course_enabled() or not course.should_evaluate(iteration):
      return
    if self.is_distributed:
      raise RuntimeError("当前 Slalom 独立碰撞考核仅支持单卡训练")
    peak = course.max_curvature
    print(f"[绕杆考核] iter {iteration}，峰值曲率 {peak}，训练无碰撞／考核有碰撞", flush=True)
    results, seconds = self._run_collision_evaluation()
    advanced = course.record_evaluation(results, iteration, pole_collision_enabled(self._evaluation_env.unwrapped))
    rate = course.history[-1]["success_rate"]
    print(f"[绕杆考核] 通过率 {rate:.1%}，连续达标 {course.history[-1]['consecutive_batches']}/"
          f"{SPACING_REQUIRED_BATCHES}，耗时 {seconds:.1f}s，课程上限 {peak}→{course.max_curvature}", flush=True)
    if self.logger.writer is not None:
      metrics = {
        "考核通过率": rate, "考核耗时秒": seconds, "峰值曲率上限": course.max_curvature,
        "最小杆心距米": get_slalom_geometry(course.max_curvature)["spacing"],
        "最大杆心距米": get_slalom_geometry(SLALOM_CURVATURE_MIN)["spacing"],
        "连续达标批次": course.consecutive_success_batches, "最终档达标": int(course.final_level_passed),
      }
      for name, value in metrics.items():
        self.logger.writer.add_scalar(f"Curriculum/{name}", value, iteration)
    if self.logger.log_dir is not None:
      directory = Path(self.logger.log_dir) / "collision_evaluations"
      directory.mkdir(parents=True, exist_ok=True)
      payload = {"iteration": iteration, "curvature": peak, "collision": True,
                 "success_rate": rate, "advanced": advanced, "seconds": seconds, "episodes": results}
      (directory / f"evaluation_{iteration:05d}_k{peak}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

  # 消费训练回合仅用于诊断，训练成绩不能推进真实碰撞课程。
  def _consume_training_results(self, extras) -> None:
    monitor = getattr(self.env.unwrapped, "_slalom_passage_monitor", None)
    results = monitor.consume() if monitor is not None else []
    if results:
      log = extras.setdefault("log", {})
      log["Metrics/无碰撞绕杆通过率"] = sum(item["success"] for item in results) / len(results)
      log["Metrics/实际通过杆数"] = sum(item["passed_poles"] for item in results) / len(results)

  # 保持 RSL-RL 的采样和 PPO 更新顺序，仅在更新完成后插入冻结策略考核。
  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    if self._course_enabled() and self.is_distributed:
      raise ValueError("当前 Slalom 独立碰撞考核仅支持单卡训练")
    if self._course_enabled() and pole_collision_enabled(self.env.unwrapped):
      raise ValueError("本课程要求训练环境关闭杆碰撞，仅独立考核开启")
    if init_at_random_ep_len and not self._command.slalom_mode_active:
      initial_limit = min(self.env.max_episode_length,
                          round(PHASE0_EPISODE_SECONDS / self.env.unwrapped.step_dt))
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=initial_limit)
    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()
    if self.is_distributed:
      self.alg.broadcast_parameters()
    self.logger.init_logging_writer()
    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    try:
      for iteration in range(start_it, total_it):
        started = time.time()
        with torch.inference_mode():
          for _ in range(self.cfg["num_steps_per_env"]):
            actions = self.alg.act(obs)
            obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
            self._consume_training_results(extras)
            if self.cfg.get("check_for_nan", True):
              check_nan(obs, rewards, dones)
            obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
            self.alg.process_env_step(obs, rewards, dones, extras)
            intrinsic = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
            self.logger.process_env_step(rewards, dones, extras, intrinsic)
          stopped = time.time()
          collect_time = stopped - started
          started = stopped
          self.alg.compute_returns(obs)
        loss_dict = self.alg.update()
        learn_time = time.time() - started
        self.current_learning_iteration = iteration
        self.logger.log(
          it=iteration, start_it=start_it, total_it=total_it,
          collect_time=collect_time, learn_time=learn_time, loss_dict=loss_dict,
          learning_rate=self.alg.learning_rate, action_std=self.alg.get_policy().output_std,
          rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
        )
        self._maybe_evaluate_curriculum(iteration)
        if self.logger.writer is not None and iteration % self.cfg["save_interval"] == 0:
          self.save(os.path.join(self.logger.log_dir, f"model_{iteration}.pt"))
      if self.logger.writer is not None:
        self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
    finally:
      if self._evaluation_env is not None:
        self._evaluation_env.close()
        self._evaluation_env = None
      if self.logger.writer is not None:
        self.logger.stop_logging_writer()

  # 训练恢复课程与新回合；回放保留入口指定的固定条件和碰撞开关。
  def load(self, path: str, load_cfg=None, strict=True, map_location=None):
    infos = super().load(path, load_cfg, strict, map_location)
    saved = (infos or {}).get("slalom_state", {}).get("spacing_course")
    if saved is not None:
      self._command.spacing_curriculum.load_state_dict(saved)
    if not (load_cfg or {}).get("actor"):
      self.env.reset()
    return infos

  # 保存课程档位、批次和实际训练碰撞开关，并沿用策略导出接口。
  def save(self, path: str, infos=None):
    state = {"spacing_course": self._command.spacing_curriculum.state_dict(),
             "training_collision": pole_collision_enabled(self.env.unwrapped)}
    super().save(path, {**(infos or {}), "slalom_state": state})
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name if wandb.run else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if wandb.run and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as e:
      print(f"[警告] ONNX 导出失败，训练继续：{e}")

