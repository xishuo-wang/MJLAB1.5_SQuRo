import os
import copy
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import wandb
from rsl_rl.utils import check_nan
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.SQuRo_Hole.mdp.curriculums import (
    STEPS_PER_ITER,
    HEIGHT_EVALUATION_ENVS,
    HEIGHT_REQUIRED_BATCHES,
    RETREAT_FAILURE_LIMIT,
    get_current_stage,
    stage_requires_collision,
    stage_uses_height_course,
)
from mjlab.tasks.SQuRo_Hole.mdp.entity import (
    configure_obstacle_entity,
    obstacle_collision_enabled,
    configure_obstacle_collision,
)



# 读取 checkpoint 里记录的 hole_state / env_state
def read_hole_state(checkpoint_path) -> dict:
    try:
        loaded = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"[WARN] 读取检查点信息失败, 将按文件名轮次推算: {exc}")
        return {}
    infos = loaded.get("infos") or {}
    state = infos.get("hole_state") if isinstance(infos, dict) else None
    return state if isinstance(state, dict) else {}



# 读取 checkpoint 里记录的全局步数
def read_env_step(checkpoint_path) -> int | None:
    try:
        loaded = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except Exception:
        return None
    infos = loaded.get("infos") or {}
    env_state = infos.get("env_state") if isinstance(infos, dict) else None
    if isinstance(env_state, dict) and env_state.get("common_step_counter") is not None:
        return int(env_state["common_step_counter"])
    return None



class SQuRoHoleOnPolicyRunner(MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper
    def __init__(self, env, train_cfg, log_dir=None, device="cpu") -> None:
        super().__init__(env, train_cfg, log_dir, device)
        self._obstacle_device = device
        self._obstacle_env_cfg = copy.deepcopy(env.unwrapped.cfg)  # type: ignore
        self._obstacle_num_envs = int(env.unwrapped.scene.num_envs)  # type: ignore
        self._obstacle_clip_actions = getattr(env, "clip_actions", None)
        self._obstacle_render_mode = getattr(env.unwrapped, "render_mode", None)  # type: ignore
        self._obstacle_command = env.unwrapped.command_manager._terms.get("hole_cmd")  # type: ignore
        self._obstacle_height_course_enabled = bool(self._obstacle_command is not None and self._obstacle_command.enable_height_course)
        self._height_curriculum_state = (self._obstacle_command.height_curriculum_state if self._obstacle_command is not None else None)
        self._episode_lengths_randomized = False
        self._evaluation_env = None


    # 当前实际编译进仿真的碰撞开关
    def _obstacle_compiled_collision(self) -> bool:
        return obstacle_collision_enabled(self.env.unwrapped)


    # 课程在当前位置要求的碰撞开关
    def _obstacle_target_collision(self) -> bool:
        step = int(self.env.unwrapped.common_step_counter)
        return stage_requires_collision(get_current_stage(step))


    # 是否需要重建: 目标与实际编译值不一致
    def _obstacle_change_needed(self) -> bool:
        compiled = self._obstacle_compiled_collision()
        if self._obstacle_target_collision() != compiled:
            return True
        if (stage_uses_height_course(get_current_stage(int(self.env.unwrapped.common_step_counter))) and self._obstacle_height_course_enabled):
            obstacle_keys = [
                key for key, entity in self.env.unwrapped.scene.entities.items()
                if key.startswith("obstacle") and hasattr(entity, "cfg")
            ]
            return set(obstacle_keys) != {"obstacle"}
        return False


    # 清掉未结束回合的累计, 避免跨重建拼接统计
    def _clear_logger_episode_state(self) -> None:
        logger = self.logger
        try:
            logger.ep_extras.clear()
        except AttributeError:
            pass
        for name in ("cur_reward_sum", "cur_episode_length", "cur_ereward_sum", "cur_ireward_sum"):
            buf = getattr(logger, name, None)
            if buf is not None:
                buf.zero_()


    # 错开首次超时，截短的首回合不进入高度课程统计。
    def _randomize_episode_lengths(self) -> None:
        self.env.episode_length_buf.random_(0, int(self.env.max_episode_length))
        if self._obstacle_command is not None:
            self._obstacle_command.course_episode_eligible[:] = self.env.episode_length_buf == 0  # type: ignore
        self._episode_lengths_randomized = True


    # 重建环境, 并按需重新取观测
    def _apply_obstacle_rebuild(self, refresh_obs: bool):
        step_counter = int(self.env.unwrapped.common_step_counter)
        collision = self._obstacle_target_collision()
        old = self.env
        env_cfg = copy.deepcopy(self._obstacle_env_cfg)
        env_cfg.scene.num_envs = self._obstacle_num_envs
        stage = get_current_stage(step_counter)
        if stage_uses_height_course(stage):
            configure_obstacle_entity(env_cfg, enable_collision=collision)
            if self._obstacle_height_course_enabled:
                env_cfg.commands["hole_cmd"].height_curriculum_state = (self._height_curriculum_state.state_dict())  # type: ignore
        else:
            configure_obstacle_collision(env_cfg, enable_collision=collision)
        new_env = ManagerBasedRlEnv(cfg=env_cfg, device=self._obstacle_device, render_mode=self._obstacle_render_mode)
        new_env.common_step_counter = step_counter
        self.env = RslRlVecEnvWrapper(new_env, clip_actions=self._obstacle_clip_actions)
        self.env.unwrapped.common_step_counter = step_counter
        self._obstacle_command = new_env.command_manager._terms.get("hole_cmd")
        self._height_curriculum_state = (
            self._obstacle_command.height_curriculum_state  # type: ignore
            if self._obstacle_command is not None else None
        )
        self._clear_logger_episode_state()
        self._randomize_episode_lengths()
        try:
            old.close()
        except Exception as exc:
            print(f"[WARN] 旧环境关闭失败: {exc}")
        print(f"[INFO] 限高板重建: 碰撞={'开' if collision else '关'}, "
              f"stage={get_current_stage(step_counter)}, iter {step_counter // STEPS_PER_ITER}, "
              f"num_envs={self._obstacle_num_envs}")
        return self.env.get_observations().to(self.device) if refresh_obs else None


    # 当前是否由独立真实碰撞考核推进高度课程
    def _course_enabled(self) -> bool:
        command = self._obstacle_command
        if command is None or not command.enable_height_course:
            return False
        cfg = command.cfg
        if cfg.fixed_velocity is not None or cfg.fixed_height_f is not None or cfg.fixed_height_h is not None:
            return False
        step = int(self.env.unwrapped.common_step_counter)
        return stage_uses_height_course(get_current_stage(step))


    # 独立考核环境: 复制训练配置, 只打开板体碰撞
    def _get_evaluation_env(self):
        if self._evaluation_env is None:
            cfg = copy.deepcopy(self._obstacle_env_cfg)
            cfg.scene.num_envs = HEIGHT_EVALUATION_ENVS
            cmd_cfg = cfg.commands["hole_cmd"]
            cmd_cfg.enable_height_course = True
            cmd_cfg.enable_course_progress = False
            cmd_cfg.debug_vis = False
            cmd_cfg.fixed_velocity = None
            cmd_cfg.fixed_height_f = None
            cmd_cfg.fixed_height_h = None
            cmd_cfg.height_curriculum_state = {}
            if self._obstacle_command is not None:
                cmd_cfg.high_obstacle_equal_height = self._obstacle_command.cfg.high_obstacle_equal_height  # type: ignore
            configure_obstacle_entity(cfg, enable_collision=True)
            raw = ManagerBasedRlEnv(cfg=cfg, device=self._obstacle_device, render_mode=None)
            raw.common_step_counter = int(self.env.unwrapped.common_step_counter)
            self._evaluation_env = RslRlVecEnvWrapper(raw, clip_actions=self._obstacle_clip_actions)
            if not obstacle_collision_enabled(raw):
                raise RuntimeError("考核环境未真正开启板体碰撞")
            print(f"[INFO] 独立真实碰撞考核环境已创建: num_envs={HEIGHT_EVALUATION_ENVS}", flush=True)
        return self._evaluation_env


    # 考核环境沿用训练当前的高度范围与高板策略
    def _sync_evaluation_course(self) -> None:
        env = self._evaluation_env.unwrapped
        command = env.command_manager._terms["hole_cmd"]
        state = command.height_curriculum_state
        state.z_min = float(self._height_curriculum_state.z_min)
        state.z_max = float(self._height_curriculum_state.z_max)
        state.final_level_passed = bool(self._height_curriculum_state.final_level_passed)
        # 考核始终按挡位等概率采样, 不继承训练侧的加权采样
        state.bin_weights = [1.0] * len(self._height_curriculum_state.bin_weights)
        if self._obstacle_command is not None:
            command.cfg.high_obstacle_equal_height = self._obstacle_command.cfg.high_obstacle_equal_height  # type: ignore
        env.common_step_counter = int(self.env.unwrapped.common_step_counter)


    # 冻结策略在接触环境中跑一批完整回合, 每环境只计首个完成回合
    def _run_evaluation_sweep(self, seed: int) -> list[dict]:
        evaluation = self._get_evaluation_env()
        env = evaluation.unwrapped
        self._sync_evaluation_course()
        torch.manual_seed(int(seed))
        policy = self.get_inference_policy(device=str(evaluation.device))
        results: list[dict] = []
        seen: set[int] = set()
        obs, _ = evaluation.reset()
        monitor = getattr(env, "_hole_passage_monitor", None)
        if monitor is not None:
            monitor.consume()
        for _ in range(env.max_episode_length + 1):
            actions = policy(obs.to(self.device)).to(evaluation.device)
            obs, _, _, _ = evaluation.step(actions)
            monitor = getattr(env, "_hole_passage_monitor", None)
            for item in monitor.consume() if monitor is not None else []:
                if item["env_id"] not in seen:
                    seen.add(item["env_id"])
                    results.append(item)
            if len(seen) == env.num_envs:
                break
        if len(seen) != env.num_envs:
            raise RuntimeError("独立真实碰撞考核缺少完整回合, 拒绝推进课程")
        return results


    # 每 100 iter 做两批独立考核: 普通模式推进/退档, 最终模式全挡位验收
    def _maybe_evaluate_curriculum(self, iteration: int) -> None:
        course = self._height_curriculum_state
        if course is None or not self._course_enabled() or not course.should_evaluate(iteration):
            return
        if self.is_distributed:
            raise RuntimeError("独立真实碰撞考核仅支持单卡训练")
        final_mode = bool(course.final_level_passed)
        measure = course.measure_final_batch if final_mode else course.measure_batch
        numpy_state, python_state = np.random.get_state(), random.getstate()
        devices = [self.env.device.index or 0] if self.env.device.type == "cuda" else []
        started = time.perf_counter()
        batch_stats: list[dict] = []
        result: dict | None = None
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                for sweep in range(HEIGHT_REQUIRED_BATCHES):
                    results = self._run_evaluation_sweep(int(self.cfg.get("seed", 42)) + iteration + sweep)
                    batch_stats.append(measure(results))
                result = course.record_evaluation(batch_stats, iteration)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            self.alg.train_mode()
        seconds = time.perf_counter() - started
        for index, batch in enumerate(batch_stats):
            print(f"[高度考核] iter {iteration} 第{index + 1}批: {batch['count']} 回合, "
                  f"通过率 {batch['success_rate']:.1%}, 达标={batch['passed']}", flush=True)
        print(f"[高度考核] iter {iteration} 本次考核({result['mode']}): 达标={result['passed']}, "
              f"合计 {result['sample_count']} 回合, 通过率 {result['success_rate']:.1%}, "
              f"Zmin {course.z_min:.3f} m", flush=True)
        if result["advanced"]:
            print(f"[高度考核] 课程降档: Zmin {result['previous_z_min']:.3f}->{course.z_min:.3f} m", flush=True)
        if result["retreated"]:
            print(f"[高度考核] 连续 {RETREAT_FAILURE_LIMIT} 次失败退档: "
                  f"Zmin {result['previous_z_min']:.3f}->{course.z_min:.3f} m", flush=True)
        if result["entered_final_mode"]:
            print("[高度考核] 47mm 档通过, 进入最终档全挡位均匀采样", flush=True)
        if self.logger.writer is not None:
            values = {
                "Curriculum/考核通过率": result["success_rate"],
                "Curriculum/考核样本数": result["sample_count"],
                "Curriculum/本次考核达标": int(result["passed"]),
                "Curriculum/最低板高米": course.z_min,
                "Curriculum/连续达标批次": course.consecutive_success_batches,
                "Curriculum/连续失败次数": course.consecutive_failures,
                "Curriculum/退档次数": course.retreats,
                "Curriculum/考核耗时秒": seconds,
                "Curriculum/最终模式": int(final_mode),
                "Curriculum/最终验收": int(course.final_accepted),
                "Curriculum/各挡位最低通过率": course.metrics()["min_bin_rate"],
            }
            if result.get("bin_rate"):
                for index, rate in enumerate(result["bin_rate"]):
                    values[f"Curriculum/挡位{index}_通过率"] = rate
            for name, value in values.items():
                self.logger.writer.add_scalar(name, value, iteration)
        if self.logger.log_dir is not None:
            directory = Path(self.logger.log_dir) / "collision_evaluations"
            directory.mkdir(parents=True, exist_ok=True)
            payload = {
                "iteration": int(iteration),
                "z_min": float(course.z_min),
                "mode": result["mode"],
                "passed": bool(result["passed"]),
                "advanced": bool(result["advanced"]),
                "retreated": bool(result["retreated"]),
                "entered_final_mode": bool(result["entered_final_mode"]),
                "final_accepted": bool(course.final_accepted),
                "bin_rate": result.get("bin_rate", []),
                "bin_count": result.get("bin_count", []),
                "bin_weights": result.get("bin_weights", []),
                "batches": batch_stats,
                "seconds": seconds,
                "collision": True,
            }
            (directory / f"evaluation_{iteration:05d}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


    # 训练循环: 复制 rsl_rl 主循环, 在每轮采样前插入碰撞开关重建
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if self._obstacle_change_needed():
            self._apply_obstacle_rebuild(refresh_obs=False)
        if init_at_random_ep_len and not self._episode_lengths_randomized:
            self._randomize_episode_lengths()
        if self._course_enabled() and self.is_distributed:
            raise ValueError("独立真实碰撞考核仅支持单卡训练")
        if self._course_enabled() and self._obstacle_compiled_collision():
            raise ValueError("高度课程要求训练环境关闭板碰撞, 仅独立考核开启")

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        try:
            for it in range(start_it, total_it):
                if self._obstacle_change_needed():
                    obs = self._apply_obstacle_rebuild(refresh_obs=True)
                start = time.time()
                with torch.inference_mode():
                    for _ in range(self.cfg["num_steps_per_env"]):
                        actions = self.alg.act(obs) # type: ignore
                        obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                        self._consume_obstacle_results(extras)
                        if self.cfg.get("check_for_nan", True):
                            check_nan(obs, rewards, dones)
                        obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                        self.alg.process_env_step(obs, rewards, dones, extras)
                        intrinsic_rewards = (self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None)
                        self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                    stop = time.time()
                    collect_time = stop - start
                    start = stop
                    self.alg.compute_returns(obs) # type: ignore

                loss_dict = self.alg.update()

                stop = time.time()
                learn_time = stop - start
                self.current_learning_iteration = it

                self.logger.log(
                    it=it,
                    start_it=start_it,
                    total_it=total_it,
                    collect_time=collect_time,
                    learn_time=learn_time,
                    loss_dict=loss_dict,
                    learning_rate=self.alg.learning_rate,
                    action_std=self.alg.get_policy().output_std,
                    rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,  # type: ignore
                )
                self._maybe_evaluate_curriculum(it)

                if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                    self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore
        finally:
            try:
                if self._evaluation_env is not None:
                    try:
                        self._evaluation_env.close()
                    except Exception as exc:
                        print(f"[WARN] 考核环境关闭失败: {exc}")
                    self._evaluation_env = None
            finally:
                if self.logger.writer is not None:
                    try:
                        self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")) # type: ignore
                    except Exception as exc:
                        print(f"[WARN] 最终检查点保存失败: {exc}")
                    finally:
                        self.logger.stop_logging_writer()


    # 续训: 恢复环境状态, 首次采样前把碰撞开关对齐到课程目标
    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True,
             map_location: str | None = None) -> dict:
        infos = super().load(path, load_cfg, strict, map_location)
        if self._obstacle_command is None:
            return infos
        if (load_cfg or {}).get("actor"):
            print("[INFO] 回放加载 (load_cfg.actor=True): 保留入口已编译好的配置, 不重建环境")
            return infos

        saved = (infos or {}).get("hole_state") or {}
        if saved.get("height_course") and self._height_curriculum_state is not None:
            self._height_curriculum_state.load_state_dict(saved["height_course"])
        if saved:
            step = int(self.env.unwrapped.common_step_counter)
            print(f"[INFO] 检查点 hole_state: stage={saved.get('stage')}, "
                  f"实际碰撞={saved.get('collision')}, "
                  f"当前恢复步数={step} (iter {step // STEPS_PER_ITER})")
            mismatch = []
            current = {d["name"]: d for d in self._obstacle_scene_state()}
            for d in saved.get("layout") or []:
                cur = current.get(d.get("name"))
                if cur is None:
                    continue
                for f in ("position", "size", "solref", "solimp"):
                    a = d.get(f)
                    b = cur.get(f)
                    if a is None and b is None:
                        continue
                    if a != b:
                        mismatch.append(f"{d.get('name')}.{f}: 保存={a} 当前={b}")
            if mismatch:
                print("[WARN] 检查点记录的板几何/接触参数与当前场景不一致, 精度口径可能变化:")
                for line in mismatch:
                    print(f"       {line}")

        if self._obstacle_change_needed():
            print(f"[INFO] 续训: 实际碰撞={self._obstacle_compiled_collision()} "
                  f"与课程目标 {self._obstacle_target_collision()} 不一致, 首次采样前重建")
            self._apply_obstacle_rebuild(refresh_obs=False)
        return infos


    # 记录实际场景里的板几何与碰撞开关 (不是全局默认表), 供续训/回放复现
    def _obstacle_scene_state(self) -> list[dict]:
        scene = self.env.unwrapped.scene
        layout = []
        for key in ("obstacle",):
            ent = scene.entities.get(key)
            if ent is None:
                continue
            layout.append({
                "name": ent.cfg.name,  # type: ignore
                "position": list(ent.cfg.position),  # type: ignore
                "size": list(ent.cfg.size),  # type: ignore
                "contype": int(ent.cfg.contype),  # type: ignore
                "conaffinity": int(ent.cfg.conaffinity),  # type: ignore
                "solref": (list(ent.cfg.solref) if getattr(ent.cfg, "solref", None) is not None else None), # type: ignore
                "solimp": (list(ent.cfg.solimp) if getattr(ent.cfg, "solimp", None) is not None else None), # type: ignore
            })
        return layout


    # 保存模型
    def save(self, path: str, infos=None):
        step = int(self.env.unwrapped.common_step_counter)
        extra = {
            "stage": get_current_stage(step),
            "collision": self._obstacle_compiled_collision(),
            "training_collision": self._obstacle_compiled_collision(),
            "evaluation_envs": HEIGHT_EVALUATION_ENVS,
            "layout": self._obstacle_scene_state(),
            "high_obstacle_equal_height": (
                self._obstacle_command.cfg.high_obstacle_equal_height
                if self._obstacle_command is not None else True
            ),
            "height_course": (
                self._height_curriculum_state.state_dict()
                if self._height_curriculum_state is not None else None
            ),
        }
        super().save(path, infos={**(infos or {}), "hole_state": extra})
        try:
            policy_dir, filename, onnx_path = self._get_export_paths(path)
            self.export_policy_to_onnx(str(policy_dir), filename)
            run_name: str = wandb.run.name if wandb.run else "local"  # type: ignore[assignment]
            metadata = get_base_metadata(self.env.unwrapped, run_name)
            attach_metadata_to_onnx(str(onnx_path), metadata)
            if wandb.run and self.cfg["upload_model"]:
                wandb.save(str(onnx_path), base_path=str(policy_dir))
        except Exception as e:
            print(f"[WARN] ONNX export failed (training continues): {e}")


    # 每个环境步消费终止阶段发布的回合结果快照
    def _consume_obstacle_results(self, extras) -> None:
        env = self.env.unwrapped
        monitor = getattr(env, "_hole_passage_monitor", None)
        results = monitor.consume() if monitor is not None else []
        log = extras.setdefault("log", {}) if isinstance(extras, dict) else {}
        command = env.command_manager._terms.get("hole_cmd")
        if (
            command is None
            or not command.enable_height_course  # type: ignore
            or not stage_uses_height_course(get_current_stage(env.common_step_counter))
        ):
            return
        if results:
            log["Metrics/训练无碰撞几何通过率"] = torch.tensor(
                sum(int(item["success"]) for item in results) / len(results), device=env.device)
            log["Metrics/训练无碰撞通过高度均值"] = torch.tensor(
                sum(float(item["height"]) for item in results) / len(results), device=env.device)
        stats = command.height_curriculum_state.metrics()  # type: ignore
        log["Metrics/Zmin_m"] = torch.tensor(stats["z_min"], device=env.device)
        log["Metrics/success_rate"] = torch.tensor(stats["success_rate"], device=env.device)
        log["Metrics/failure_rate"] = torch.tensor(stats["failure_rate"], device=env.device)
        log["Metrics/level_progress"] = torch.tensor(stats["level_progress"], device=env.device)
