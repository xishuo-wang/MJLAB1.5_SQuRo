import copy
import os
import time

import torch
import wandb
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.SQuRo_Hole.SQuRo_Hole_env_cfg import (
    configure_hole_collision,
    hole_collision_enabled,
)
from mjlab.tasks.SQuRo_Hole.mdp.command import (
    STAGE3_END_ITER,
    get_current_stage,
)
from rsl_rl.utils import check_nan


_STEPS_PER_ITER = 24


class SQuRoHoleOnPolicyRunner(MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def __init__(self, env, train_cfg, log_dir=None, device="cpu") -> None:
        super().__init__(env, train_cfg, log_dir, device)
        # 限高板的碰撞开关在编译期固化 (见 mdp/hole.py), 阶段 3→4 要开碰撞只能重建环境。
        # 留一份本次启动的 env_cfg 副本做重建模板, 否则命令行覆盖 (num_envs / sim / seed) 会退回注册值。
        self._hole_device = device
        self._hole_env_cfg = copy.deepcopy(env.unwrapped.cfg)
        self._hole_num_envs = int(env.unwrapped.scene.num_envs)
        self._hole_clip_actions = getattr(env, "clip_actions", None)
        self._hole_render_mode = getattr(env.unwrapped, "render_mode", None)
        self._hole_has_gates = "hole1" in env.unwrapped.scene.entities

    # 目标碰撞开关: 阶段 1~3 关, 阶段 4 开
    def _hole_target_collision(self) -> bool:
        step_counter = int(self.env.unwrapped.common_step_counter)
        return get_current_stage(step_counter) >= 4

    # 是否需要重建: 目标与**实际编译值**不一致
    def _hole_change_needed(self) -> bool:
        if not self._hole_has_gates:
            return False
        return self._hole_target_collision() != hole_collision_enabled(self.env.unwrapped)

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

    # 重建环境 (换碰撞开关), 并按需重新取观测 (旧 obs 属于旧环境)
    def _apply_hole_rebuild(self, refresh_obs: bool) -> "torch.Tensor | None":
        step_counter = int(self.env.unwrapped.common_step_counter)
        collision = self._hole_target_collision()
        old = self.env
        env_cfg = copy.deepcopy(self._hole_env_cfg)
        env_cfg.scene.num_envs = self._hole_num_envs
        configure_hole_collision(env_cfg, enable_collision=collision)
        new_env = ManagerBasedRlEnv(cfg=env_cfg, device=self._hole_device,
                                    render_mode=self._hole_render_mode)
        # 新环境从 0 起算, 必须在这里接管计数器, 否则下一轮判据又读到阶段 1 而反复重建
        new_env.common_step_counter = step_counter
        self.env = RslRlVecEnvWrapper(new_env, clip_actions=self._hole_clip_actions)
        self.env.unwrapped.common_step_counter = step_counter
        self._clear_logger_episode_state()
        try:
            old.close()
        except Exception as exc:
            print(f"[WARN] 旧环境关闭失败: {exc}")
        print(f"[INFO] 限高板重建: 碰撞={'开' if collision else '关'}, "
              f"stage={get_current_stage(step_counter)}, "
              f"iter {step_counter // _STEPS_PER_ITER} (边界 {STAGE3_END_ITER}), "
              f"num_envs={self._hole_num_envs}")
        return self.env.get_observations().to(self.device) if refresh_obs else None

    # 训练循环: 复制 rsl_rl 主循环, 只在每轮采样前插入碰撞开关重建
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
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir,
                                   f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    # 续训: 首次采样前把碰撞开关对齐到恢复轮次对应的阶段
    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True,
             map_location: str | None = None) -> dict:
        infos = super().load(path, load_cfg, strict, map_location)
        if not self._hole_has_gates:
            return infos
        if (load_cfg or {}).get("actor"):
            print("[INFO] 回放加载 (load_cfg.actor=True): 保留入口已编译好的配置, 不重建环境")
            return infos
        if self._hole_change_needed():
            print(f"[INFO] 续训: 实际碰撞={hole_collision_enabled(self.env.unwrapped)} "
                  f"与目标 {self._hole_target_collision()} 不一致, 首次采样前重建")
            self._apply_hole_rebuild(refresh_obs=False)
        return infos

    # 检查点记录**实际编译生效**的碰撞开关, 供续训与回放复现
    def save(self, path: str, infos=None):
        extra = {"hole_collision": hole_collision_enabled(self.env.unwrapped)}
        super().save(path, infos={**(infos or {}), "hole_state": extra})
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
            print(f"[WARN] ONNX export failed (training continues): {e}")
