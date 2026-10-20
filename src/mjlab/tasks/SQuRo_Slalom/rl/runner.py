import copy
import json
import os
import random
import time
from pathlib import Path
from typing import cast

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
from mjlab.tasks.SQuRo_Slalom.mdp.command import SlalomCommand, SlalomCommandCfg
from mjlab.tasks.SQuRo_Slalom.mdp.config import (
  PHASE0_EPISODE_SECONDS, SPACING_EVALUATION_ENVS,
  SPACING_EVALUATION_EPISODE_SECONDS, SPACING_REQUIRED_BATCHES,
)
from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import SLALOM_BIN_COUNT
from mjlab.tasks.SQuRo_Slalom.mdp.entity import configure_obstacle_collision, obstacle_collision_enabled
from mjlab.tasks.SQuRo_Slalom.mdp.reference import get_slalom_geometry


class SQuRoSlalomOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  # 训练环境保持无杆碰撞，独立考核环境在第一次考核时创建并复用。
  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    super().__init__(env, train_cfg, log_dir, device)
    self._evaluation_env = None
    self._command: SlalomCommand = cast(SlalomCommand, env.unwrapped.command_manager._terms["slalom_cmd"])

  # 显式固定条件与回放不自动推进杆距课程。
  def _evaluation_enabled(self) -> bool:
    cfg = self._command.cfg
    return (cfg.enable_spacing_curriculum and not cfg.phase0_only
            and cfg.fixed_slalom_curvature is None and cfg.fixed_obstacle_spacing is None)

  # 独立场景保留当前机器人和命令参数，只打开杆碰撞并按课程档位采样。
  def _get_evaluation_env(self):
    if self._evaluation_env is None:
      cfg = copy.deepcopy(self.env.unwrapped.cfg)
      cfg.scene.num_envs = SPACING_EVALUATION_ENVS
      cfg.episode_length_s = SPACING_EVALUATION_EPISODE_SECONDS
      command_cfg = cast(SlalomCommandCfg, cfg.commands["slalom_cmd"])
      command_cfg.enable_spacing_curriculum = True
      command_cfg.phase0_only = False
      command_cfg.debug_vis = False
      command_cfg.fixed_slalom_curvature = None
      command_cfg.fixed_obstacle_spacing = None
      command_cfg.spacing_curriculum_state = None
      configure_obstacle_collision(cfg, enable_collision=True)
      raw = ManagerBasedRlEnv(cfg, device=str(self.env.device), render_mode=None)
      raw.common_step_counter = self.env.unwrapped.common_step_counter
      self._evaluation_env = RslRlVecEnvWrapper(raw, clip_actions=self.env.clip_actions)
      if not obstacle_collision_enabled(raw):
        raise RuntimeError("碰撞考核场景未真正开启杆碰撞")
      print(f"[INFO] 独立真实碰撞考核环境已创建: num_envs={SPACING_EVALUATION_ENVS}", flush=True)
    return self._evaluation_env

  # 考核沿用训练侧课程档位，但按挡位等概率采样且不继承训练权重。
  def _sync_evaluation_course(self) -> None:
    assert self._evaluation_env is not None
    env = self._evaluation_env.unwrapped
    command = cast(SlalomCommand, env.command_manager._terms["slalom_cmd"])
    course = command.spacing_curriculum
    train = self._command.spacing_curriculum
    course.max_curvature = int(train.max_curvature)
    course.final_level_passed = bool(train.final_level_passed)
    course.bin_weights = [1.0] * SLALOM_BIN_COUNT
    command.adaptive_sampling = False
    env.common_step_counter = int(self.env.unwrapped.common_step_counter)

  # 冻结策略在接触环境中跑一整轮，每环境只计第一个完整回合。
  def _run_evaluation_sweep(self) -> list[dict]:
    evaluation = self._get_evaluation_env()
    env = evaluation.unwrapped
    self._sync_evaluation_course()
    policy = self.get_inference_policy(device=str(evaluation.device))
    results: list[dict] = []
    seen: set[int] = set()
    obs, _ = evaluation.reset()
    monitor = getattr(env, "_slalom_passage_monitor", None)
    if monitor is not None:
      monitor.consume()
    for _ in range(env.max_episode_length + 1):
      obs, _, _, _ = evaluation.step(policy(obs.to(self.device)).to(evaluation.device))
      monitor = getattr(env, "_slalom_passage_monitor", None)
      for item in monitor.consume() if monitor is not None else []:
        if item["env_id"] not in seen:
          seen.add(item["env_id"])
          results.append(item)
      if len(seen) == env.num_envs:
        break
    if len(seen) != env.num_envs:
      raise RuntimeError("真实碰撞考核缺少完整回合，拒绝推进课程")
    return results

  # 每 100 轮做两批独立考核：普通模式推进/进入最终档，最终模式全挡位验收。
  def _maybe_evaluate_curriculum(self, iteration: int) -> None:
    course = self._command.spacing_curriculum
    if not self._evaluation_enabled() or not course.should_evaluate(iteration):
      return
    if self.is_distributed:
      raise RuntimeError("当前 Slalom 独立碰撞考核仅支持单卡训练")
    final_mode = bool(course.final_level_passed)
    measure = course.measure_final_batch if final_mode else course.measure_batch
    numpy_state, python_state = np.random.get_state(), random.getstate()
    devices = [self.env.device.index or 0] if self.env.device.type == "cuda" else []  # type: ignore
    started = time.perf_counter()
    batch_stats: list[dict] = []
    result: dict | None = None
    try:
      with torch.random.fork_rng(devices=devices), torch.no_grad():
        for sweep in range(SPACING_REQUIRED_BATCHES):
          torch.manual_seed(int(self.cfg.get("seed", 42)) + iteration + sweep)
          batch_stats.append(measure(self._run_evaluation_sweep()))
        assert self._evaluation_env is not None
        collision = obstacle_collision_enabled(self._evaluation_env.unwrapped)
        result = course.record_evaluation(batch_stats, iteration, collision)
    finally:
      np.random.set_state(numpy_state)
      random.setstate(python_state)
      self.alg.train_mode()
    seconds = time.perf_counter() - started
    for index, batch in enumerate(batch_stats):
      label = "全档位" if final_mode else "最低档"
      print(f"[绕杆考核] iter {iteration} 第{index + 1}批: {label} {batch['count']} 回合, "
            f"通过率 {batch['success_rate']:.1%}, 达标={batch['passed']}", flush=True)
    print(f"[绕杆考核] iter {iteration} 本次考核({result['mode']}): 达标={result['passed']}, "
          f"合计 {result['sample_count']} 回合, 通过率 {result['success_rate']:.1%}, "
          f"上限峰值 {course.max_curvature}", flush=True)
    if result["advanced"]:
      print(f"[绕杆考核] 课程推进: 峰值曲率 {result['previous_curvature']}->{course.max_curvature}", flush=True)
    if result["entered_final_mode"]:
      print("[绕杆考核] 峰值十五档通过, 进入最终档全挡位等概率采样", flush=True)
    if result.get("accepted"):
      print("[绕杆考核] 最终验收通过: 全挡位两批整体通过率均不低于 80%", flush=True)
    if self.logger.writer is not None:
      metrics = {
        "Curriculum/eval_pass_rate": result["success_rate"],
        "Curriculum/eval_samples": result["sample_count"],
        "Curriculum/eval_passed": int(result["passed"]),
        "Curriculum/max_curvature": course.max_curvature,
        "Curriculum/min_obstacle_spacing_mm": get_slalom_geometry(course.max_curvature)["spacing"] * 1000.0,
        "Curriculum/consecutive_passed_batches": course.consecutive_success_batches,
        "Curriculum/final_mode": int(final_mode),
        "Curriculum/final_accepted": int(course.final_accepted),
        "Curriculum/eval_seconds": seconds,
      }
      if result.get("bin_rate"):
        for index, rate in enumerate(result["bin_rate"]):
          metrics[f"Curriculum/bin{index}_pass_rate"] = rate
      for name, value in metrics.items():
        self.logger.writer.add_scalar(name, value, iteration)
    if self.logger.log_dir is not None:
      directory = Path(self.logger.log_dir) / "collision_evaluations"
      directory.mkdir(parents=True, exist_ok=True)
      payload = {"iteration": iteration, "mode": result["mode"], "collision": True,
                 "passed": bool(result["passed"]), "advanced": bool(result["advanced"]),
                 "entered_final_mode": bool(result["entered_final_mode"]),
                 "final_accepted": bool(course.final_accepted),
                 "max_curvature": course.max_curvature,
                 "bin_rate": result.get("bin_rate", []), "bin_count": result.get("bin_count", []),
                 "bin_weights": result.get("bin_weights", []),
                 "batches": batch_stats, "seconds": seconds}
      (directory / f"evaluation_{iteration:05d}_k{course.max_curvature}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

  # 消费训练回合仅用于诊断，训练成绩不能推进真实碰撞课程。
  def _consume_passage_results(self, extras) -> None:
    monitor = getattr(self.env.unwrapped, "_slalom_passage_monitor", None)
    results = monitor.consume() if monitor is not None else []
    if results:
      log = extras.setdefault("log", {})
      log["Curriculum/train_pass_rate"] = sum(item["success"] for item in results) / len(results)
      log["Metrics/passed_obstacles_mean"] = sum(item["passed_obstacles"] for item in results) / len(results)

  # 保持 RSL-RL 的采样和 PPO 更新顺序，仅在更新完成后插入冻结策略考核。
  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    if self._evaluation_enabled() and self.is_distributed:
      raise ValueError("当前 Slalom 独立碰撞考核仅支持单卡训练")
    if self._evaluation_enabled() and obstacle_collision_enabled(self.env.unwrapped):
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
            self._consume_passage_results(extras)
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
          rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,  # type: ignore
        )
        self._maybe_evaluate_curriculum(iteration)
        if self.logger.writer is not None and iteration % self.cfg["save_interval"] == 0:
          assert self.logger.log_dir is not None
          self.save(os.path.join(self.logger.log_dir, f"model_{iteration}.pt"))
      if self.logger.writer is not None:
        assert self.logger.log_dir is not None
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
             "training_collision": obstacle_collision_enabled(self.env.unwrapped)}
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


