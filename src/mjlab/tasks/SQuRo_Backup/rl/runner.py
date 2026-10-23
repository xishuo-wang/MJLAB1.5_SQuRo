import os
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import wandb
from typing import Any, cast
from rsl_rl.utils import check_nan
from mjlab.rl.exporter_utils import (
    get_base_metadata,
    attach_metadata_to_onnx,
)
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.SQuRo_Backup.mdp import entity as mdp_entity
from mjlab.tasks.SQuRo_Backup.mdp.command import BackupCommand, BackupCommandCfg
from mjlab.tasks.SQuRo_Backup.mdp.config import P1_END, T3
from mjlab.tasks.SQuRo_Backup.mdp.curriculums import (
    _STEPS_PER_ITER,
    WALL_D_MIN_END,
    CURRICULUM_BATCHES_REQUIRED,
    CURRICULUM_LEVELS,
    CURRICULUM_WINDOW_EPISODES,
    WALL_D_MAX,
    WALL_D_MIN_FRAC,
    WALL_X_NEG_LEVELS,
    WALL_X_POS,
    WALL_BIN_COUNT,
    WALL_EVALUATION_ENVS,
    WALL_CURRICULUM_STATE_VERSION,
    P1_SETTLE_MAX,
    TIME_SCALE_MAX,
    WallCurriculumState,
    get_level_for_wall_x_neg,
    get_training_phase,
    get_wall_positions,
)


class SQuRoBackupOnPolicyRunner(MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper

    # 初始化训练、场景重建和独立考核状态。
    def __init__(self, env, train_cfg, log_dir=None, device="cpu") -> None:
        super().__init__(env, train_cfg, log_dir, device)
        self._corridor_device = device
        self._corridor_env_cfg = copy.deepcopy(env.unwrapped.cfg)  # type: ignore
        self._corridor_num_envs = int(env.unwrapped.scene.num_envs)  # type: ignore
        self._corridor_clip_actions = getattr(env, "clip_actions", None)
        self._corridor_render_mode = getattr(env.unwrapped, "render_mode", None)  # type: ignore
        entity = env.unwrapped.scene.entities.get("restricted_space")  # type: ignore
        self._corridor_start_walls = (
            None if entity is None else (float(entity.cfg.wall_x_neg), float(entity.cfg.wall_x_pos))
        )
        self._corridor_fixed = entity is not None and entity.cfg.fixed_width
        # 显式指定宽度
        self._corridor_fixed_by_cli = self._corridor_fixed
        self._randomize_ep_len = False
        # 墙位课程状态 (技术细节 §7.11 第 11 条)
        self._cur_level = 0
        self._cur_level_iter = 0
        self._cur_level_history: list[dict] = []
        self._wall_random = self._corridor_start_walls is not None and not self._corridor_fixed
        self._wall_d_min = WALL_D_MAX
        self._dmin_batch_ep = 0
        self._dmin_batch_stood = 0
        self._dmin_batch_pass = 0
        self._wall_course = WallCurriculumState()
        self._evaluation_env = None
        self._reset_curriculum_window()
        self._push_wall_curriculum()
        self._force_wall_resample()

    # 懒初始化墙位课程状态
    def _ensure_wall_state(self) -> None:
        if not hasattr(self, "_evaluation_env"):
            self._evaluation_env = None
        if not hasattr(self, "_wall_random"):
            self._wall_random = False
        if not hasattr(self, "_wall_d_min"):
            self._wall_d_min = WALL_D_MAX
        if not hasattr(self, "_wall_course"):
            self._wall_course = WallCurriculumState()
            self._wall_course.d_min = self._wall_d_min
        if not hasattr(self, "_dmin_batch_ep"):
            self._dmin_batch_ep = 0
            self._dmin_batch_stood = 0
            self._dmin_batch_pass = 0

    # 清空当前档位的门控窗口
    def _reset_curriculum_window(self) -> None:
        n, w = self._corridor_num_envs, CURRICULUM_WINDOW_EPISODES
        self._w_succ = torch.zeros(n, w, dtype=torch.bool)  # 稳定站立成功 (现行判据)
        self._w_stood = torch.zeros(n, w, dtype=torch.bool)  # 站姿维持满窗口 (去速度项) <- 门控量
        self._w_onset = torch.zeros(n, w, dtype=torch.bool)  # 站立窗口出现 (单帧, 诊断)
        self._w_head = torch.zeros(n, dtype=torch.long)
        self._w_fill = torch.zeros(n, dtype=torch.long)
        self._ep_seq_seen = torch.zeros(n, dtype=torch.long)
        self._dmin_batch_ep = 0
        self._dmin_batch_stood = 0
        self._dmin_batch_pass = 0

    # 阶段二 (开碰撞) 是否已生效
    def _collision_phase_on(self) -> bool:
        try:
            step_counter = int(self.env.unwrapped.common_step_counter)
        except Exception:
            return False
        return get_training_phase(step_counter) == 1

    # 拉取本轮新发布的回合结果, 更新每环境的滑动窗口
    def _ingest_episode_results(self) -> None:
        self._ensure_wall_state()
        if not hasattr(self, "_w_succ"):
            self._reset_curriculum_window()
        cmd = self._command_term()
        seq = cmd._ep_seq.detach().to("cpu")
        new = seq > self._ep_seq_seen
        if not bool(new.any()):
            return
        idx = new.nonzero(as_tuple=False).squeeze(-1)
        keep = idx[cmd._last_ep_valid.detach().to("cpu")[idx]]
        if len(keep) > 0:
            head = self._w_head[keep]
            stood = cmd._last_ep_stood_pose.detach().to("cpu")[keep]
            self._w_succ[keep, head] = cmd._last_ep_success.detach().to("cpu")[keep]
            self._w_stood[keep, head] = stood
            self._w_onset[keep, head] = cmd._last_ep_stood_onset.detach().to("cpu")[keep]
            self._w_head[keep] = (head + 1) % CURRICULUM_WINDOW_EPISODES
            self._w_fill[keep] = torch.clamp(self._w_fill[keep] + 1, max=CURRICULUM_WINDOW_EPISODES)
        self._ep_seq_seen[idx] = seq[idx]

    # 三个训练诊断率共用有效回合窗口, 不参与课程推进
    def _curriculum_metrics(self) -> tuple[float, float, float, bool, int]:
        n_valid = int(self._w_fill.sum())
        ready = bool((self._w_fill >= CURRICULUM_WINDOW_EPISODES).all())
        if n_valid == 0:
            return float("nan"), float("nan"), float("nan"), ready, 0
        return (
            int(self._w_stood.sum()) / n_valid,
            int(self._w_onset.sum()) / n_valid,
            int(self._w_succ.sum()) / n_valid,
            ready,
            n_valid,
        )

    # 兼容旧诊断入口: 训练回合不再推进墙位课程
    def _maybe_promote(self, it: int) -> bool:
        return False

    # 兼容旧诊断入口: 下界仅由独立考核结算
    def _maybe_lower_dmin(self, it: int) -> bool:
        return False

    # 把当前 d_min / 范围 / 下界采样比例推给命令项
    def _push_wall_curriculum(self, enabled: bool | None = None) -> None:
        self._ensure_wall_state()
        cmd = self._command_term()
        if cmd is None:
            return
        setter = getattr(cmd, "set_wall_curriculum", None)
        if setter is None:
            return
        if enabled is False or not self._wall_random:
            setter(None, None, None)
            return
        course = self._wall_course
        setter(
            course.d_min,
            course.d_max,
            WALL_D_MIN_FRAC,
            bin_weights=list(course.bin_weights),
            final_mode=course.final_level_passed,
        )

    # 获取命令项
    def _command_term(self) -> Any:
        try:
            return self.env.unwrapped.command_manager.get_term("backup_cmd")
        except Exception:
            return None

    # 按当前 (_corridor_fixed, _corridor_start_walls) 重新确定课程模式
    def _refresh_wall_mode(self) -> None:
        self._ensure_wall_state()
        if self._corridor_fixed or self._corridor_start_walls is None:
            self._wall_random = False
        if not self._wall_random:
            self._push_wall_curriculum(enabled=False)
            self._force_wall_resample()

    # 让正在进行的回合立刻改用当前课程范围
    def _force_wall_resample(self) -> None:
        cmd = self._command_term()
        if cmd is None:
            return
        try:
            ids = torch.arange(self._corridor_num_envs, device=cmd.device)
            cmd._resample_command(ids)
        except Exception as exc:
            print(f"[WARN] 墙位范围同步失败: {exc}")

    # 每轮把课程状态写进 tensorboard; 无样本时不写 (NaN 会污染曲线)
    def _log_curriculum(self, it: int) -> None:
        writer = getattr(self.logger, "writer", None)
        if writer is None:
            return
        self._ensure_wall_state()
        neg, pos = get_wall_positions(self._cur_level)
        p_stood, p_onset, p_done, ready, n_valid = self._curriculum_metrics()
        if self._wall_random:
            neg, pos = -self._wall_d_min, WALL_X_POS
            writer.add_scalar("Curriculum/wall_d_min", self._wall_d_min, it)
            stats = self._wall_course.metrics()
            writer.add_scalar("Curriculum/final_mode", int(stats["final_mode"]), it)
            writer.add_scalar("Curriculum/final_accepted", int(stats["final_accepted"]), it)
            if self._wall_course.history:
                writer.add_scalar("Curriculum/eval_first_attempt_righting_rate", stats["success_rate"], it)
        thickness = self.env.unwrapped.scene.entities["restricted_space"].cfg.wall_half_thickness
        writer.add_scalar("Curriculum/wall_width", pos - neg - 2 * thickness, it)
        writer.add_scalar("Curriculum/window_ready", float(ready), it)
        if n_valid > 0:
            writer.add_scalar("Curriculum/n_valid_episodes", n_valid, it)
            writer.add_scalar("Curriculum/p_stood", p_stood, it)
            writer.add_scalar("Curriculum/p_onset", p_onset, it)
            writer.add_scalar("Curriculum/p_done", p_done, it)

    # 读取当前编译场景的墙位与碰撞状态。
    def _corridor_compiled_state(
        self,
    ) -> tuple[tuple[float, float] | None, bool | None]:
        entity = self.env.unwrapped.scene.entities.get("restricted_space")
        if entity is None:
            return None, None
        return (
            (float(entity.cfg.wall_x_neg), float(entity.cfg.wall_x_pos)),  # type: ignore
            bool(entity.collision_enabled),
        )  # type: ignore

    # 目标课程值
    def _corridor_target(self) -> tuple[tuple[float, float] | None, bool]:
        self._ensure_wall_state()
        step_counter = int(self.env.unwrapped.common_step_counter)
        collision = get_training_phase(step_counter) == 1
        if self._corridor_fixed:
            return self._corridor_start_walls, collision
        if self._wall_random:
            return self._corridor_compiled_state()[0], collision
        return get_wall_positions(self._cur_level), collision

    # 判定是否需要重建
    def _corridor_change_needed(self) -> bool:
        walls, collision = self._corridor_target()
        compiled_walls, compiled_collision = self._corridor_compiled_state()
        if compiled_collision is None:
            return False
        if collision != compiled_collision:
            return True
        if walls is None or compiled_walls is None:
            return False
        return max(abs(a - b) for a, b in zip(walls, compiled_walls, strict=True)) > 1e-9

    # 清掉未结束回合的累计
    def _clear_logger_episode_state(self) -> None:
        logger = self.logger
        try:
            logger.ep_extras.clear()
        except AttributeError:
            pass
        for name in (
            "cur_reward_sum",
            "cur_episode_length",
            "cur_ereward_sum",
            "cur_ireward_sum",
        ):
            buf = getattr(logger, name, None)
            if buf is not None:
                buf.zero_()

    # 随机化各环境的回合相位
    def _randomize_episode_phase(self) -> None:
        self.env.episode_length_buf = torch.randint_like(
            self.env.episode_length_buf, high=int(self.env.max_episode_length)
        )

    # 重建环境
    def _apply_corridor_rebuild(self, refresh_obs: bool) -> "torch.Tensor | None":
        step_counter = int(self.env.unwrapped.common_step_counter)
        walls, collision = self._corridor_target()
        if walls is None:
            return None
        neg, pos = walls
        old = self.env
        env_cfg = copy.deepcopy(self._corridor_env_cfg)
        env_cfg.scene.num_envs = self._corridor_num_envs
        env_cfg.events.pop("init_restricted_space", None)
        env_cfg.seed = None
        mdp_entity.configure_restricted_space(env_cfg, wall_x_neg=neg, wall_x_pos=pos, enable_collision=collision)
        new_env = ManagerBasedRlEnv(
            cfg=env_cfg,
            device=self._corridor_device,
            render_mode=self._corridor_render_mode,
        )
        new_env.common_step_counter = step_counter
        self.env = RslRlVecEnvWrapper(new_env, clip_actions=self._corridor_clip_actions)
        self.env.unwrapped.common_step_counter = step_counter
        if self._randomize_ep_len:
            self._randomize_episode_phase()
        self._reset_curriculum_window()
        self._cur_level_iter = step_counter // _STEPS_PER_ITER
        self._clear_logger_episode_state()
        self._push_wall_curriculum()
        self._force_wall_resample()

        try:
            old.close()
        except Exception as exc:
            print(f"[WARN] 旧环境关闭失败: {exc}")
        if self._wall_random:
            print(
                f"[INFO] 受限空间重建: 碰撞={'开' if collision else '关'}, "
                f"随机墙位 d∈[{self._wall_d_min:.3f}, {WALL_D_MAX:.2f}] "
                f"x_pos={WALL_X_POS:.3f}, iter {step_counter // _STEPS_PER_ITER} "
                f"num_envs={self._corridor_num_envs}"
            )
            return self.env.get_observations().to(self.device) if refresh_obs else None
        print(
            f"[INFO] 受限空间重建: 档位 {self._cur_level}/{CURRICULUM_LEVELS - 1}, "
            f"墙 x_neg={neg:.4f} x_pos={pos:.4f} (净宽 {pos - neg - 0.02:.4f} m), "
            f"碰撞={'开' if collision else '关'}, iter {step_counter // _STEPS_PER_ITER} "
            f"num_envs={self._corridor_num_envs}"
        )
        return self.env.get_observations().to(self.device) if refresh_obs else None

    # 仅自动随机墙位课程在二阶段执行独立考核
    def _evaluation_enabled(self) -> bool:
        self._ensure_wall_state()
        command = self._command_term()
        return (
            self._wall_random
            and not self._corridor_fixed
            and self._collision_phase_on()
            and command is not None
            and getattr(command.cfg, "fixed_time_scale", None) is None
        )

    # 独立考核复制训练场景与物理参数, 墙碰撞始终开启
    def _get_evaluation_env(self):
        if self._evaluation_env is None:
            cfg = copy.deepcopy(self._corridor_env_cfg)
            cfg.scene.num_envs = WALL_EVALUATION_ENVS
            cfg.events.pop("init_restricted_space", None)
            cfg.seed = None
            command_cfg = cast(BackupCommandCfg, cfg.commands["backup_cmd"])
            command_cfg.debug_vis = False
            command_cfg.single_attempt_evaluation = True
            # 覆盖最大 λ 下两段首次尝试, 为离散时间边界各保留一个控制步。
            step_dt = cfg.sim.mujoco.timestep * cfg.decimation
            attempt_duration = (
                (P1_END + P1_SETTLE_MAX + T3) * TIME_SCALE_MAX + 2 * command_cfg.window_late_s + 2 * step_dt
            )
            cfg.episode_length_s = min(cfg.episode_length_s, attempt_duration)
            walls, _ = self._corridor_compiled_state()
            if walls is None:
                raise RuntimeError("墙位考核缺少受限空间实体")
            mdp_entity.configure_restricted_space(
                cfg,
                wall_x_neg=walls[0],
                wall_x_pos=walls[1],
                enable_collision=True,
            )
            raw = ManagerBasedRlEnv(cfg=cfg, device=self._corridor_device, render_mode=None)
            raw.common_step_counter = int(self.env.unwrapped.common_step_counter)
            self._evaluation_env = RslRlVecEnvWrapper(raw, clip_actions=self._corridor_clip_actions)
            if not raw.scene.entities["restricted_space"].collision_enabled:
                raise RuntimeError("独立考核环境未开启真实墙碰撞")
            print(
                f"[墙位考核] 已创建真实碰撞环境: {WALL_EVALUATION_ENVS} 个, "
                f"首次尝试时长上限 {raw.max_episode_length * raw.step_dt:.2f} s",
                flush=True,
            )
        return self._evaluation_env

    # 冻结确定性策略, 每环境只取首次尝试, 全批结果收齐即结束。
    def _run_evaluation_sweep(self, seed: int) -> list[dict]:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32))
        torch.random.default_generator.manual_seed(int(seed))
        device = torch.device(self.env.device)
        if device.type == "cuda":
            torch.cuda.default_generators[device.index or 0].manual_seed(int(seed))
        evaluation = self._get_evaluation_env()
        raw = evaluation.unwrapped
        raw.common_step_counter = int(self.env.unwrapped.common_step_counter)
        command = cast(BackupCommand, raw.command_manager.get_term("backup_cmd"))
        course = self._wall_course
        command.set_wall_curriculum(
            course.d_min,
            course.d_max,
            WALL_D_MIN_FRAC,
            bin_weights=[1.0] * WALL_BIN_COUNT,
            final_mode=course.final_level_passed,
        )
        policy = self.get_inference_policy(device=str(self.device))
        obs, _ = evaluation.reset()
        if bool((raw.episode_length_buf != 0).any()):
            raise RuntimeError("独立考核必须从完整回合起点开始")
        sequence = command._ep_seq.clone()
        seen = torch.zeros(raw.num_envs, device=raw.device, dtype=torch.bool)
        results: list[dict] = []
        for step in range(raw.max_episode_length + 1):
            actions = policy(obs.to(self.device)).to(evaluation.device)
            obs, rewards, dones, _ = evaluation.step(actions)
            if self.cfg.get("check_for_nan", True):
                check_nan(obs, rewards, dones)
            completed = (command._ep_seq > sequence) & dones.bool() & ~seen
            if bool((dones.bool() & ~completed & ~seen).any()):
                raise RuntimeError("独立真实碰撞考核出现异常提前复位, 拒绝推进课程")
            righted = torch.where(completed, command._last_ep_righted, command._ep_righted)
            restarted = (
                command._last_retry_mask
                | command._last_back_to_p1
                | command._last_back_to_p2
                | command._last_cycle_reset
            )
            # 首次结果立即收集; 同步出现翻正确认与重试时按失败计。
            finished = ~seen & (righted | restarted | completed)
            distances = torch.where(completed, command._last_ep_wall_d, command._wall_d)
            for env_id in finished.nonzero(as_tuple=False).flatten().tolist():
                success = bool(righted[env_id] and not restarted[env_id])
                results.append(
                    {
                        "env_id": int(env_id),
                        "valid": True,
                        "success": success,
                        "wall_d": float(distances[env_id]),
                        "duration_s": (step + 1) * float(raw.step_dt),
                        "end_reason": "翻正确认"
                        if success
                        else "阶段重试或重新开始"
                        if bool(restarted[env_id])
                        else "超时",
                    }
                )
            seen |= finished
            if bool(seen.all()):
                break
        if len(results) != raw.num_envs:
            raise RuntimeError("独立真实碰撞考核缺少首次尝试结果, 拒绝推进课程")
        return results

    # 两批同档考核全部完成后一次结算, 恢复训练随机状态与归一化模式
    def _maybe_evaluate_curriculum(self, iteration: int) -> None:
        self._ensure_wall_state()
        course = self._wall_course
        if not self._evaluation_enabled() or not course.should_evaluate(iteration):
            return
        if self.is_distributed:
            raise RuntimeError("独立真实碰撞考核仅支持单卡训练")
        if self._corridor_change_needed():
            raise RuntimeError("训练场景物理配置尚未同步, 拒绝结算墙位课程")
        numpy_state, python_state = np.random.get_state(), random.getstate()
        device = torch.device(self.env.device)
        devices = [device.index or 0] if device.type == "cuda" else []
        batches: list[dict] = []
        started = time.perf_counter()
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                for sweep in range(CURRICULUM_BATCHES_REQUIRED):
                    results = self._run_evaluation_sweep(int(self.cfg.get("seed", 42)) + iteration + sweep)
                    batches.append(course.measure_batch(results))
                result = course.record_evaluation(batches, iteration)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            self.alg.train_mode()
        self._wall_d_min = course.d_min
        self._cur_level_iter = course.level_start_iteration
        self._push_wall_curriculum()
        seconds = time.perf_counter() - started
        for index, batch in enumerate(batches):
            print(
                f"[墙位考核] iter {iteration} 第 {index + 1} 批: "
                f"{batch['count']} 个计分回合, 首次尝试翻正率 {batch['success_rate']:.1%}, "
                f"达标={batch['passed']}",
                flush=True,
            )
        print(
            f"[墙位考核] 模式={result['mode']}, 达标={result['passed']}, "
            f"d_min {result['previous_d_min']:.3f}->{course.d_min:.3f} m, "
            f"推进={result['advanced']}, 退档={result['retreated']}, "
            f"最终验收={course.final_accepted}",
            flush=True,
        )
        if self.logger.writer is not None:
            values = {
                "Curriculum/eval_first_attempt_righting_rate": result["success_rate"],
                "Curriculum/eval_passed": int(result["passed"]),
                "Curriculum/eval_seconds": seconds,
                "Curriculum/retreat_count": course.retreats,
                "Curriculum/final_mode": int(course.final_level_passed),
                "Curriculum/final_accepted": int(course.final_accepted),
            }
            values.update(
                {f"Curriculum/bin{i}_first_attempt_righting_rate": rate for i, rate in enumerate(result["bin_rate"])}
            )
            for name, value in values.items():
                self.logger.writer.add_scalar(name, value, iteration)
        if self.logger.log_dir is not None:
            assert self._evaluation_env is not None, "独立考核环境缺失, 无法记录时长"
            directory = Path(self.logger.log_dir) / "collision_evaluations"
            directory.mkdir(parents=True, exist_ok=True)
            payload = {
                **result,
                "batches": batches,
                "seconds": seconds,
                "training_collision": True,
                "evaluation_collision": True,
                "success_criterion": "首次翻正确认前无阶段重试或重新开始; S2 连续确认, 不要求站立或关节速度",
                "evaluation_episode_length_s": self._evaluation_env.unwrapped.max_episode_length
                * self._evaluation_env.unwrapped.step_dt,
                "course": course.state_dict(),
            }
            (directory / f"evaluation_{iteration:05d}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # 训练循环: PPO 更新后独立考核, 考核样本不写入训练存储
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        self._randomize_ep_len = bool(init_at_random_ep_len)
        if self._randomize_ep_len:
            self._randomize_episode_phase()

        if self._corridor_start_walls is not None and self._corridor_change_needed():
            self._apply_corridor_rebuild(refresh_obs=False)
        else:
            self._push_wall_curriculum()
            self._force_wall_resample()

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
                self._ingest_episode_results()
                if self._corridor_start_walls is not None and self._corridor_change_needed():
                    obs = self._apply_corridor_rebuild(refresh_obs=True)
                if self.is_distributed and self._evaluation_enabled():
                    raise ValueError("独立真实碰撞考核仅支持单卡训练")
                self._log_curriculum(it)
                start = time.time()
                with torch.inference_mode():
                    for _ in range(self.cfg["num_steps_per_env"]):
                        actions = self.alg.act(obs)  # type: ignore
                        obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                        if self.cfg.get("check_for_nan", True):
                            check_nan(obs, rewards, dones)
                        obs, rewards, dones = (
                            obs.to(self.device),
                            rewards.to(self.device),
                            dones.to(self.device),
                        )
                        self.alg.process_env_step(obs, rewards, dones, extras)
                        intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                        self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
                    stop = time.time()
                    collect_time = stop - start
                    start = stop
                    self.alg.compute_returns(obs)  # type: ignore
                loss_dict = self.alg.update()
                learn_time = time.time() - start
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
                    self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
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
                        self.save(
                            os.path.join(
                                self.logger.log_dir,
                                f"model_{self.current_learning_iteration}.pt",
                            )
                        )
                    except Exception as exc:
                        print(f"[WARN] 最终检查点保存失败: {exc}")
                    finally:
                        self.logger.stop_logging_writer()

    # 继承训练
    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        self._ensure_wall_state()
        infos = super().load(path, load_cfg, strict, map_location)
        if self._corridor_start_walls is None:
            return infos
        if (load_cfg or {}).get("actor"):
            print("[INFO] 回放加载 (load_cfg.actor=True): 保留入口已编译好的墙体配置, 不重建环境")
            return infos
        saved = (infos or {}).get("corridor_state") or {}
        recorded_neg = saved.get("wall_x_neg")
        recorded_pos = saved.get("wall_x_pos")
        if saved.get("curriculum_level") is not None:
            level = min(max(int(saved["curriculum_level"]), 0), CURRICULUM_LEVELS - 1)
            pair_matches = (recorded_neg is None or abs(WALL_X_NEG_LEVELS[level] - float(recorded_neg)) <= 1e-9) and (
                recorded_pos is None or abs(WALL_X_POS - float(recorded_pos)) <= 1e-9
            )
            if not pair_matches and recorded_neg is not None:
                level = get_level_for_wall_x_neg(float(recorded_neg))
                print(
                    f"[WARN] 检查点档位 {saved['curriculum_level']} 与记录的墙位 "
                    f"({float(recorded_neg):+.4f}, "
                    f"{'?' if recorded_pos is None else format(float(recorded_pos), '+.4f')}) "
                    f"不自洽 (档位表已改), 按墙位反查为第 {level} 档"
                )
            self._cur_level = level
        elif recorded_neg is not None:
            self._cur_level = get_level_for_wall_x_neg(float(recorded_neg))
        elif saved.get("corridor_width") is not None:
            half = 0.5 * float(saved["corridor_width"])
            self._cur_level = get_level_for_wall_x_neg(-half)
            print(
                f"[WARN] 旧格式检查点只记录对称间距 {saved['corridor_width']}, "
                f"已折算到第 {self._cur_level} 档 (x_neg={WALL_X_NEG_LEVELS[self._cur_level]})"
            )
        else:
            print("[WARN] 检查点没有墙位记录, 课程档位按 0 处理; 若这是自动课程 run, 回放请显式指定墙位")
        self._cur_level_iter = int(saved.get("curriculum_iter", self.current_learning_iteration))
        self._cur_level_history = list(saved.get("curriculum_history") or [])

        # 新检查点恢复完整课程, 旧检查点只迁移墙位下界与驻留起点
        if self._wall_random:
            if saved.get("wall_course"):
                self._wall_course.load_state_dict(saved["wall_course"])
                if int(saved["wall_course"]["version"]) < WALL_CURRICULUM_STATE_VERSION:
                    self._wall_course.level_start_iteration = self.current_learning_iteration
                    print("[INFO] 旧考核仅迁移当前墙位, 首次尝试翻正标准需重新考核; 旧通过状态与成绩不继承")
            else:
                distance = saved.get("wall_d_min")
                if distance is None and recorded_neg is not None:
                    distance = -float(recorded_neg)
                if distance is None and saved.get("corridor_width") is not None:
                    distance = float(saved["corridor_width"]) / 2
                if distance is not None:
                    lower = min(WALL_D_MAX, max(WALL_D_MIN_END, float(distance)))
                    if abs(lower - float(distance)) > 1e-9:
                        print(f"[WARN] 旧墙位下界 {float(distance):.3f} m 超出当前课程范围, 调整为 {lower:.3f} m")
                    self._wall_course.d_min = lower
                self._wall_course.level_start_iteration = self._cur_level_iter
                print("[INFO] 旧检查点保留模型与训练进度, 墙位课程改由独立考核推进; 旧批次成绩不继承")
            self._wall_d_min = self._wall_course.d_min
            self._cur_level_iter = self._wall_course.level_start_iteration

        # 锁死模式: 命令行显式指定墙位优先; 否则沿用检查点记录的标记
        if not self._corridor_fixed_by_cli and saved.get("corridor_fixed") is not None:
            self._corridor_fixed = bool(saved["corridor_fixed"])
            if self._corridor_fixed and saved.get("wall_x_neg") is not None:
                self._corridor_start_walls = (
                    float(saved["wall_x_neg"]),
                    float(saved["wall_x_pos"]),
                )
        self._refresh_wall_mode()
        self._reset_curriculum_window()
        if self._corridor_change_needed():
            compiled = self._corridor_compiled_state()
            print(f"[INFO] 续训: 实际编译配置 {compiled} 与课程目标 {self._corridor_target()} 不一致, 首次采样前重建")
            self._apply_corridor_rebuild(refresh_obs=False)
        else:
            self._push_wall_curriculum()
            self._force_wall_resample()
        return infos

    # 保存模型
    def save(self, path: str, infos=None):
        self._ensure_wall_state()
        walls, collision = self._corridor_compiled_state()
        neg, pos = walls if walls is not None else (None, None)
        extra = {
            "wall_x_neg": neg,
            "wall_x_pos": pos,
            "corridor_width": (None if walls is None else pos - neg),  # type: ignore
            "corridor_collision": collision,
            "corridor_fixed": self._corridor_fixed,
            "curriculum_level": self._cur_level,
            "curriculum_iter": self._cur_level_iter,
            "curriculum_history": list(self._cur_level_history),
            "wall_d_min": (self._wall_d_min if self._wall_random else None),
            "wall_course": (self._wall_course.state_dict() if self._wall_random else None),
            "evaluation_envs": WALL_EVALUATION_ENVS,
        }
        super().save(path, infos={**(infos or {}), "corridor_state": extra})
        try:
            policy_dir, filename, onnx_path = self._get_export_paths(path)
            self.export_policy_to_onnx(str(policy_dir), filename)
            run_name: str = wandb.run.name if wandb.run else "local"  # type: ignore[assignment]
            metadata = get_base_metadata(self.env.unwrapped, run_name)
            attach_metadata_to_onnx(str(onnx_path), metadata)
            if wandb.run and self.cfg["upload_model"]:
                wandb.save(str(onnx_path), base_path=str(policy_dir))
        except Exception as e:
            print(f"[WARN] ONNX 导出失败, 训练继续: {e}")
