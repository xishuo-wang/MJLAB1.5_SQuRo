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
        # 启动时编译进仿真的墙宽 (锁死宽度模式的模板), 以及该锁死是否来自显式指定
        self._corridor_start_width = (
            float(entity.cfg.corridor_width) if entity is not None else None)
        self._corridor_fixed = entity is not None and entity.cfg.fixed_width
        # 显式指定宽度 (RESTRICTED_SPACE_WIDTH) 属于命令行意图, 优先于检查点记录里的锁死标记
        self._corridor_fixed_by_cli = self._corridor_fixed
        # learn() 入口的回合相位随机化开关: 重建发生在循环内部, 必须遵守同一次调用的语义。
        # 默认 False = 与 learn() 签名默认值一致; load() 触发的重建早于 learn(), 此时不随机化,
        # 随后的 learn(True) 会补上。
        self._randomize_ep_len = False

    # 当前**实际编译进仿真**的 (宽度, 碰撞开关), 唯一来源是场景里的实体本身。
    # 不要另存一份缓存来做比较: 缓存只记录"我们以为改成了什么", 一旦与真实环境脱节
    # (例如加载检查点后直接改了缓存), 判据会认为"无需重建"而让训练跑在错误的物理环境里。
    def _corridor_compiled_state(self) -> tuple[float | None, bool | None]:
        entity = self.env.unwrapped.scene.entities.get("restricted_space")
        if entity is None:
            return None, None
        return float(entity.cfg.corridor_width), bool(entity.collision_enabled)

    # 当前轮次该用的 (宽度, 是否开碰撞); 锁死宽度模式下宽度保持启动值
    def _corridor_target(self) -> tuple[float | None, bool]:
        step_counter = int(self.env.unwrapped.common_step_counter)
        collision = get_training_phase(step_counter) == 1
        if self._corridor_fixed:
            return self._corridor_start_width, collision
        iter_num = step_counter // _STEPS_PER_ITER
        return float(get_corridor_width_for_iter(iter_num)), collision

    # 是否需要重建: 目标与**实际编译值**不一致 (碰撞开关变化, 或宽度不一致)。
    # 必须拿 compiled_width 比, 不能只在非锁死模式下比: 续训从"非锁死档位"切进"锁死某个 a"
    # 时目标宽度也变了, 漏掉就会继续跑在错误的墙宽里 (审查发现的续训问题第二种形态)。
    def _corridor_change_needed(self) -> bool:
        width, collision = self._corridor_target()
        compiled_width, compiled_collision = self._corridor_compiled_state()
        if compiled_collision is None:
            return False
        if collision != compiled_collision:
            return True
        if width is None or compiled_width is None:
            return False
        return abs(width - compiled_width) > 1e-9

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

    # 打散各环境的回合计时 (回合相位随机化), 与 learn() 入口的 init_at_random_ep_len 同一语义。
    # 新环境的 episode_length_buf 全为 0 而本任务只有 timeout 一种终止, 不打散就会让全部环境
    # 永久同进同出: 每轮 rollout 只覆盖一个任务相位, Progress/*、Gate/* 变成随回合周期振荡的
    # 相位抽签值。依据与实测见技术细节 §7.11 第 8 条。
    def _randomize_episode_phase(self) -> None:
        self.env.episode_length_buf = torch.randint_like(
            self.env.episode_length_buf, high=int(self.env.max_episode_length))

    # 重建环境, 并按需重新取观测 (换环境后旧 obs 属于旧环境, 必须作废)
    def _apply_corridor_rebuild(self, refresh_obs: bool) -> "torch.Tensor | None":
        step_counter = int(self.env.unwrapped.common_step_counter)
        width, collision = self._corridor_target()
        if width is None:
            return None
        old = self.env
        env_cfg = copy.deepcopy(self._corridor_env_cfg)
        env_cfg.scene.num_envs = self._corridor_num_envs
        env_cfg.events.pop("init_restricted_space", None)
        # 重建不是新实验的开始: 清掉 seed, 否则 ManagerBasedRlEnv.__init__ 会调 seed_rng ->
        # torch.manual_seed 复位全部设备的 RNG, 影响策略采样与 PPO 抽样。见 §7.11 第 10 条。
        env_cfg.seed = None
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
        # 与计数器同理, 但必须遵守 learn() 入口的开关: 关闭时保持调用方要求的固定回合长度
        if self._randomize_ep_len:
            self._randomize_episode_phase()
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
        # 记住入口开关: 重建发生在循环内部, 必须遵守同一次调用的语义
        self._randomize_ep_len = bool(init_at_random_ep_len)
        if self._randomize_ep_len:
            self._randomize_episode_phase()

        if self._corridor_start_width is not None and self._corridor_change_needed():
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
            if self._corridor_start_width is not None and self._corridor_change_needed():
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
    # 恢复出来的轮次所对应的编译配置 (启动 env 编译的是 iter 0 的配置, 直接续训会跑错物理环境)。
    # load_cfg={"actor": True} 是回放/推理加载: 回放入口已按检查点记录把墙编译好了,
    # 这里再重建会把查看器手里那个 env 换掉 (查看器不接管 runner.env), 必须直接跳过。
    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True,
             map_location: str | None = None) -> dict:
        infos = super().load(path, load_cfg, strict, map_location)
        if self._corridor_start_width is None:
            return infos
        if (load_cfg or {}).get("actor"):
            print("[INFO] 回放加载 (load_cfg.actor=True): 保留入口已编译好的墙体配置, 不重建环境")
            return infos
        saved = (infos or {}).get("corridor_state") or {}
        # 锁死宽度模式: 显式指定宽度 (命令行) 优先; 否则沿用检查点记录的标记, 并同步模型宽度,
        # 因为锁死模式下课程档位恒为"启动宽度", 模板不换就会按错误的 a 重建。
        if not self._corridor_fixed_by_cli and saved.get("corridor_fixed") is not None:
            self._corridor_fixed = bool(saved["corridor_fixed"])
            if self._corridor_fixed and saved.get("corridor_width") is not None:
                self._corridor_start_width = float(saved["corridor_width"])
        # 判据只比较"课程/锁死目标 vs 实际编译值", 不看记录: 记录本身就是当时真实编译的值,
        # 若拿它去覆盖当前值再比较, 就会把"环境其实没编译成这个值"掩盖过去 (曾经的真 bug)。
        # 轮次已由父类从检查点恢复, 目标自然与记录一致; 不一致说明记录与轮次自相矛盾, 以轮次为准。
        if self._corridor_change_needed():
            compiled = self._corridor_compiled_state()
            print(f"[INFO] 续训: 实际编译配置 {compiled} 与课程目标 "
                  f"{self._corridor_target()} 不一致, 首次采样前重建")
            self._apply_corridor_rebuild(refresh_obs=False)
        return infos

    # 检查点里记录**实际编译生效**的墙宽与碰撞开关, 供续训与回放精确复现 (不要用轮次反推)
    def save(self, path: str, infos=None):
        width, collision = self._corridor_compiled_state()
        extra = {
            "corridor_width": width,
            "corridor_collision": collision,
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
