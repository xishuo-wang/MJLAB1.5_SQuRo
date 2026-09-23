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
from mjlab.tasks.SQuRo_Backup.mdp import entity as mdp_entity
from mjlab.tasks.SQuRo_Backup.mdp.curriculums import (
    STAGE1_3_ITER,
    _STEPS_PER_ITER,
    get_corridor_width_for_iter,
    get_training_phase,
)
from rsl_rl.utils import check_nan


class SQuRoBackupOnPolicyRunner(MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def __init__(self, env, train_cfg, log_dir=None, device="cpu") -> None:
        super().__init__(env, train_cfg, log_dir, device)
        # 墙的 a 与碰撞开关在编译期固化, 运行期改不了 (见 mdp/entity.py), 换档只能重建环境。
        # 这里留一份**本次启动的 env_cfg 副本**作为重建模板, 否则重建会把命令行覆盖
        # (fixed_time_scale / episode_length_s / sim 参数 / seed) 全部退回注册配置。
        self._corridor_device = device
        self._corridor_env_cfg = copy.deepcopy(env.unwrapped.cfg)
        self._corridor_num_envs = int(env.unwrapped.scene.num_envs)
        self._corridor_clip_actions = getattr(env, "clip_actions", None)
        self._corridor_render_mode = getattr(env.unwrapped, "render_mode", None)
        entity = env.unwrapped.scene.entities.get("restricted_space")
        self._corridor_width = float(entity.cfg.corridor_width) if entity is not None else None
        self._corridor_collision = bool(entity.collision_enabled) if entity is not None else False
        # 显式指定宽度时训练全程锁死, 不跟随课程
        self._corridor_fixed = entity is not None and entity.cfg.fixed_width

    # 当前轮次该用的 (宽度, 是否开碰撞); 锁死宽度模式下宽度不变
    def _corridor_target(self) -> tuple[float, bool]:
        step_counter = int(self.env.unwrapped.common_step_counter)
        iter_num = step_counter // _STEPS_PER_ITER
        collision = get_training_phase(step_counter) == 1
        width = (self._corridor_width if self._corridor_fixed
                 else get_corridor_width_for_iter(iter_num))
        return float(width), collision

    # 是否需要重建: 碰撞开关变化, 或 (非锁死宽度时) 宽度档位变化。
    # 碰撞判断必须放在 fixed 分支之外 —— 阶段边界处即使宽度锁死也要重建以打开碰撞。
    def _corridor_change_needed(self) -> bool:
        width, collision = self._corridor_target()
        if collision != self._corridor_collision:
            return True
        return (not self._corridor_fixed) and abs(width - self._corridor_width) > 1e-9

    # 清掉"未结束回合"的累计。环境重建会中断所有在跑的回合, 不清就会跨重建拼接统计。
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

    # 重建环境, 并按需重新取观测 (换环境后旧 obs 属于旧环境, 必须作废)
    def _apply_corridor_rebuild(self, refresh_obs: bool) -> "torch.Tensor | None":
        step_counter = int(self.env.unwrapped.common_step_counter)
        width, collision = self._corridor_target()
        old = self.env
        env_cfg = copy.deepcopy(self._corridor_env_cfg)
        env_cfg.scene.num_envs = self._corridor_num_envs
        env_cfg.events.pop("init_restricted_space", None)
        mdp_entity.configure_restricted_space(env_cfg, width, enable_collision=collision)
        new_env = ManagerBasedRlEnv(cfg=env_cfg, device=self._corridor_device,
                                    render_mode=self._corridor_render_mode)
        # 必须在新环境**首次采样前**接管 common_step_counter: 新环境从 0 起算, 否则下一轮
        # _corridor_target() 会读到 iter 0 (阶段一), 判定碰撞又不一致而再次重建把它关回去,
        # 形成"开了又关"的反复重建。课程/阶段只由这个计数器决定。
        new_env.common_step_counter = step_counter
        self.env = RslRlVecEnvWrapper(new_env, clip_actions=self._corridor_clip_actions)
        # 包装器 reset 之后再把计数器写一次 (reset 可能把它归零), 然后才允许取观测
        self.env.unwrapped.common_step_counter = step_counter
        self._corridor_width = width
        self._corridor_collision = collision
        self._clear_logger_episode_state()
        try:
            old.close()
        except Exception as exc:  # 旧环境关不掉不应中断训练
            print(f"[WARN] 旧环境关闭失败: {exc}")
        print(f"[INFO] 受限空间重建: a={width:.4f} m (净宽 {width - 0.02:.4f} m), "
              f"碰撞={'开' if collision else '关'}, iter {step_counter // _STEPS_PER_ITER} "
              f"(阶段边界 STAGE1_3_ITER={STAGE1_3_ITER}), num_envs={self._corridor_num_envs}")
        return self.env.get_observations().to(self.device) if refresh_obs else None

    # 训练循环: 复制 rsl_rl OnPolicyRunner.learn 的主循环, 只在**每轮采样开始前**插入
    # 受限空间重建。不能只重写入口 —— 训练只调用一次 learn(6000), 阶段切换发生在循环内部。
    # 重建后必须重新取观测: obs 是循环里的局部变量, 换环境后旧观测属于旧环境。
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length))

        if self._corridor_width is not None and self._corridor_change_needed():
            self._apply_corridor_rebuild(refresh_obs=False)

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            # 受限空间: 每轮采样前检查课程档位/阶段, 需要就重建并刷新观测
            if self._corridor_width is not None and self._corridor_change_needed():
                obs = self._apply_corridor_rebuild(refresh_obs=True)
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

    # 续训: 父类只恢复 common_step_counter, 这里进一步在**首次采样前**把墙体配置对齐到
    # 检查点记录的实际编译值 (缺失时按恢复后的轮次推算), 而不是等下一轮日志钩子。
    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True,
             map_location: str | None = None) -> dict:
        infos = super().load(path, load_cfg, strict, map_location)
        if self._corridor_width is None:
            return infos
        saved = (infos or {}).get("corridor_state") or {}
        want_width = saved.get("corridor_width")
        want_collision = saved.get("corridor_collision")
        if want_width is not None:
            self._corridor_width = float(want_width)
            self._corridor_fixed = bool(saved.get("corridor_fixed", self._corridor_fixed))
        if want_collision is not None:
            self._corridor_collision = bool(want_collision)
        # 与检查点记录 (或按轮次推算) 不一致时, 在首次采样前重建
        if self._corridor_change_needed():
            print(f"[INFO] 续训: 检查点墙体状态与当前编译配置不一致, 首次采样前重建 "
                  f"(记录 a={want_width}, 碰撞={want_collision})")
            self._apply_corridor_rebuild(refresh_obs=False)
        return infos

    # 检查点里记录实际生效的墙宽与碰撞开关, 供续训与回放精确复现 (不要用轮次反推)
    def save(self, path: str, infos=None):
        extra = {
            "corridor_width": self._corridor_width,
            "corridor_collision": self._corridor_collision,
            "corridor_fixed": self._corridor_fixed,
        }
        super().save(path, infos={**(infos or {}), "corridor_state": extra})
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
