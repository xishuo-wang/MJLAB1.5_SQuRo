import copy
import os
import time

import torch
import wandb
from rsl_rl.utils import check_nan

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.SQuRo_Hole.mdp.curriculums import (
    STEPS_PER_ITER,
    get_current_stage,
    stage_requires_collision,
    stage_uses_height_course,
)
from mjlab.tasks.SQuRo_Hole.mdp.entity import (
    configure_hole_entities,
    configure_single_hole_entities,
    hole_collision_enabled,
)

# 几何/状态格式版本: 换代时旧 checkpoint 需要显式迁移
HOLE_STATE_VERSION = 3


# 读取 checkpoint 里记录的 hole_state / env_state (回放侧在建环境前要用)
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
        # 进入高度课程时重建环境并创建单板。
        # 留一份本次启动的 env_cfg 副本作重建模板, 否则命令行覆盖会退回注册值。
        self._hole_device = device
        self._hole_env_cfg = copy.deepcopy(env.unwrapped.cfg)  # type: ignore
        self._hole_num_envs = int(env.unwrapped.scene.num_envs)  # type: ignore
        self._hole_clip_actions = getattr(env, "clip_actions", None)
        self._hole_render_mode = getattr(env.unwrapped, "render_mode", None)  # type: ignore
        self._hole_command = env.unwrapped.command_manager._terms.get("hole_cmd")  # type: ignore
        self._hole_height_course_enabled = bool(
            self._hole_command is not None
            and self._hole_command.enable_height_course
        )
        self._height_curriculum_state = (
            self._hole_command.height_curriculum_state
            if self._hole_command is not None else None
        )

    # 当前实际编译进仿真的碰撞开关 (唯一来源是场景里的实体本身)
    def _hole_compiled_collision(self) -> bool:
        return hole_collision_enabled(self.env.unwrapped)

    # 课程在当前位置要求的碰撞开关
    def _hole_target_collision(self) -> bool:
        step = int(self.env.unwrapped.common_step_counter)
        return stage_requires_collision(get_current_stage(step))

    # 是否需要重建: 目标与**实际编译值**不一致
    def _hole_change_needed(self) -> bool:
        compiled = self._hole_compiled_collision()
        if self._hole_target_collision() != compiled:
            return True
        if (
            stage_uses_height_course(get_current_stage(int(self.env.unwrapped.common_step_counter)))
            and self._hole_height_course_enabled
        ):
            holes = [
                key for key, entity in self.env.unwrapped.scene.entities.items()
                if key.startswith("hole") and hasattr(entity, "cfg")
            ]
            return set(holes) != {"hole1"}
        return False

    # 清掉"未结束回合"的累计, 避免跨重建拼接统计
    def _clear_logger_episode_state(self) -> None:
        logger = self.logger
        try:
            logger.ep_extras.clear()
        except AttributeError:
            pass
        for name in ("cur_reward_sum", "cur_episode_length",
                     "cur_ereward_sum", "cur_ireward_sum"):
            buf = getattr(logger, name, None)
            if buf is not None:
                buf.zero_()

    # 重建环境 (换碰撞开关), 并按需重新取观测
    def _apply_hole_rebuild(self, refresh_obs: bool):
        step_counter = int(self.env.unwrapped.common_step_counter)
        collision = self._hole_target_collision()
        old = self.env
        env_cfg = copy.deepcopy(self._hole_env_cfg)
        env_cfg.scene.num_envs = self._hole_num_envs
        stage = get_current_stage(step_counter)
        if stage_uses_height_course(stage):
            configure_single_hole_entities(env_cfg, enable_collision=collision)
            if self._hole_height_course_enabled:
                env_cfg.commands["hole_cmd"].height_curriculum_state = (
                    self._height_curriculum_state.state_dict()  # type: ignore
                )
        else:
            configure_hole_entities(env_cfg, enable_collision=collision)
        new_env = ManagerBasedRlEnv(cfg=env_cfg, device=self._hole_device,
                                    render_mode=self._hole_render_mode)
        # 新环境从 0 起算, 必须接管全局步数, 否则下一轮判据又读到阶段 1 而反复重建
        new_env.common_step_counter = step_counter
        self.env = RslRlVecEnvWrapper(new_env, clip_actions=self._hole_clip_actions)
        self.env.unwrapped.common_step_counter = step_counter
        self._hole_command = new_env.command_manager._terms.get("hole_cmd")
        self._height_curriculum_state = (
            self._hole_command.height_curriculum_state  # type: ignore
            if self._hole_command is not None else None
        )
        self._clear_logger_episode_state()
        try:
            old.close()
        except Exception as exc:
            print(f"[WARN] 旧环境关闭失败: {exc}")
        print(f"[INFO] 限高板重建: 碰撞={'开' if collision else '关'}, "
              f"stage={get_current_stage(step_counter)}, iter {step_counter // STEPS_PER_ITER}, "
              f"num_envs={self._hole_num_envs}")
        return self.env.get_observations().to(self.device) if refresh_obs else None

    # 训练循环: 复制 rsl_rl 主循环, 在每轮采样前插入碰撞开关重建
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length))

        if self._hole_change_needed():
            self._apply_hole_rebuild(refresh_obs=False)

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            if self._hole_change_needed():
                obs = self._apply_hole_rebuild(refresh_obs=True)
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    self._consume_hole_results(extras)
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device),
                                           dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = (self.alg.intrinsic_rewards
                                         if self.cfg["algorithm"]["rnd_cfg"] else None)
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

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

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir,  # type: ignore
                                   f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    # 续训: 恢复环境状态, 首次采样前把碰撞开关对齐到课程目标
    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True,
             map_location: str | None = None) -> dict:
        infos = super().load(path, load_cfg, strict, map_location)
        if self._hole_command is None:
            return infos
        if (load_cfg or {}).get("actor"):
            print("[INFO] 回放加载 (load_cfg.actor=True): 保留入口已编译好的配置, 不重建环境")
            return infos

        saved = (infos or {}).get("hole_state") or {}
        if saved.get("height_course") and self._height_curriculum_state is not None:
            self._height_curriculum_state.load_state_dict(saved["height_course"])
        if not saved:
            print("[WARN] 检查点没有 hole_state 记录 (旧格式): 无法判断它当时的碰撞配置; "
                  "若它是在无真实碰撞下训练的, 不要当作已完成真实碰撞训练")
        else:
            step = int(self.env.unwrapped.common_step_counter)
            print(f"[INFO] 检查点 hole_state: 版本={saved.get('version')}, "
                  f"stage={saved.get('stage')}, 实际碰撞={saved.get('collision')}, "
                  f"当前恢复步数={step} (iter {step // STEPS_PER_ITER})")
            # 校验保存的几何/接触参数与当前场景是否一致 (不一致时训练数据口径会变)
            mismatch = []
            current = {d["name"]: d for d in self._hole_scene_state()}
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

        if self._hole_change_needed():
            print(f"[INFO] 续训: 实际碰撞={self._hole_compiled_collision()} "
                  f"与课程目标 {self._hole_target_collision()} 不一致, 首次采样前重建")
            self._apply_hole_rebuild(refresh_obs=False)
        return infos

    # 记录**实际场景里**的板几何与碰撞开关 (不是全局默认表), 供续训/回放复现
    def _hole_scene_state(self) -> list[dict]:
        scene = self.env.unwrapped.scene
        layout = []
        for key in ("hole1", "hole2", "hole3"):
            ent = scene.entities.get(key)
            if ent is None:
                continue
            layout.append({
                "name": ent.cfg.name,  # type: ignore
                "position": list(ent.cfg.position),  # type: ignore
                "size": list(ent.cfg.size),  # type: ignore
                "contype": int(ent.cfg.contype),  # type: ignore
                "conaffinity": int(ent.cfg.conaffinity),  # type: ignore
                "solref": (list(ent.cfg.solref) if getattr(ent.cfg, "solref", None) is not None  # type: ignore
                           else None),
                "solimp": (list(ent.cfg.solimp) if getattr(ent.cfg, "solimp", None) is not None  # type: ignore
                           else None),
            })
        return layout

    # 保存模型 (记录实际编译生效的碰撞与几何)
    def save(self, path: str, infos=None):
        step = int(self.env.unwrapped.common_step_counter)
        extra = {
            "version": HOLE_STATE_VERSION,
            "stage": get_current_stage(step),
            "collision": self._hole_compiled_collision(),
            "layout": self._hole_scene_state(),
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

    # 每个环境步消费终止阶段发布的回合结果快照。
    def _consume_hole_results(self, extras) -> None:
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
        del results
        stats = command.height_curriculum_state.metrics()  # type: ignore
        log["Metrics/Zmin_m"] = torch.tensor(stats["z_min"], device=env.device)
        log["Metrics/success_rate"] = torch.tensor(stats["success_rate"], device=env.device)
        log["Metrics/failure_rate"] = torch.tensor(stats["failure_rate"], device=env.device)
        log["Metrics/level_progress"] = torch.tensor(stats["level_progress"], device=env.device)
