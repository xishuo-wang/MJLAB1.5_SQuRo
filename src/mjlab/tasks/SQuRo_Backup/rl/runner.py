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
    CURRICULUM_GATE_P_STOOD,
    CURRICULUM_LEVELS,
    CURRICULUM_MIN_DWELL_ITER,
    CURRICULUM_START_ITER,
    CURRICULUM_WINDOW_EPISODES,
    STAGE1_3_ITER,
    WALL_X_NEG_LEVELS,
    _STEPS_PER_ITER,
    get_level_for_wall_x_neg,
    get_training_phase,
    get_wall_positions_for_level,
)
from rsl_rl.utils import check_nan


class SQuRoBackupOnPolicyRunner(MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def __init__(self, env, train_cfg, log_dir=None, device="cpu") -> None:
        super().__init__(env, train_cfg, log_dir, device)
        # 墙位与碰撞开关在编译期固化, 运行期改不了 (见 mdp/entity.py), 换档只能重建环境。
        # 这里留一份**本次启动的 env_cfg 副本**作为重建模板, 否则重建会把命令行覆盖
        # (fixed_time_scale / episode_length_s / sim 参数 / seed) 全部退回注册配置。
        self._corridor_device = device
        self._corridor_env_cfg = copy.deepcopy(env.unwrapped.cfg)
        self._corridor_num_envs = int(env.unwrapped.scene.num_envs)
        self._corridor_clip_actions = getattr(env, "clip_actions", None)
        self._corridor_render_mode = getattr(env.unwrapped, "render_mode", None)
        entity = env.unwrapped.scene.entities.get("restricted_space")
        # 启动时编译进仿真的墙位 (锁死模式的模板), 以及该锁死是否来自显式指定
        self._corridor_start_walls = (
            None if entity is None
            else (float(entity.cfg.wall_x_neg), float(entity.cfg.wall_x_pos)))
        self._corridor_fixed = entity is not None and entity.cfg.fixed_width
        # 显式指定宽度 (RESTRICTED_SPACE_WIDTH) 属于命令行意图, 优先于检查点记录里的锁死标记
        self._corridor_fixed_by_cli = self._corridor_fixed
        # learn() 入口的回合相位随机化开关: 重建发生在循环内部, 必须遵守同一次调用的语义。
        # 默认 False = 与 learn() 签名默认值一致; load() 触发的重建早于 learn(), 此时不随机化,
        # 随后的 learn(True) 会补上。
        self._randomize_ep_len = False
        # 墙位课程状态 —— 课程状态全部由 runner 持有并写进检查点; 物理/循环/未结束的回合
        # 重建后重新开始, 不跨环境拼接 (技术细节 §7.11 第 11 条)。
        self._cur_level = 0
        self._cur_level_iter = 0
        self._cur_level_history: list[dict] = []
        self._reset_curriculum_window()

    # 清空当前档位的门控窗口。**每次重建都要清**: 新环境的 _ep_seq 从 0 重新开始, 不清
    # _ep_seq_seen 就再也收不到事件, 门控会永久饿死 (与 common_step_counter 同一类坑);
    # 而且旧墙位的成绩本来也不能用于新墙位升级。
    def _reset_curriculum_window(self) -> None:
        n, w = self._corridor_num_envs, CURRICULUM_WINDOW_EPISODES
        # 三个率共用同一个窗口 (同一批有效回合、同一分母), 所以彼此可直接比较
        self._w_succ = torch.zeros(n, w, dtype=torch.bool)      # 稳定站立成功 (现行判据)
        self._w_stood = torch.zeros(n, w, dtype=torch.bool)     # 站姿维持满窗口 (去速度项) <- 门控量
        self._w_onset = torch.zeros(n, w, dtype=torch.bool)     # 站立窗口出现 (单帧, 诊断)
        self._w_head = torch.zeros(n, dtype=torch.long)
        self._w_fill = torch.zeros(n, dtype=torch.long)
        self._ep_seq_seen = torch.zeros(n, dtype=torch.long)

    # 拉取本轮新发布的回合结果, 更新每环境的滑动窗口。
    # **无效回合只推进已消费序号, 不占窗口槽位、不加填充量** —— 否则"1 个无效 + 2 个有效"
    # 会被判成"3 个有效回合已攒满"(审查 P2 已复现)。窗口只存有效完整回合, 故分母即 _w_fill 合计。
    # 每个环境一轮内最多结束 1 个有效回合 (有效回合恒为 1200 步 > 一轮 96 步), 按序号取增量不漏事件。
    def _ingest_episode_results(self) -> None:
        if not hasattr(self, "_w_succ"):
            self._reset_curriculum_window()
        cmd = self.env.unwrapped.command_manager.get_term("backup_cmd")
        seq = cmd._ep_seq.detach().to("cpu")
        new = seq > self._ep_seq_seen
        if not bool(new.any()):
            return
        idx = new.nonzero(as_tuple=False).squeeze(-1)
        keep = idx[cmd._last_ep_valid.detach().to("cpu")[idx]]
        if len(keep) > 0:
            head = self._w_head[keep]
            self._w_succ[keep, head] = cmd._last_ep_success.detach().to("cpu")[keep]
            self._w_stood[keep, head] = cmd._last_ep_stood_pose.detach().to("cpu")[keep]
            self._w_onset[keep, head] = cmd._last_ep_stood_onset.detach().to("cpu")[keep]
            self._w_head[keep] = (head + 1) % CURRICULUM_WINDOW_EPISODES
            self._w_fill[keep] = torch.clamp(self._w_fill[keep] + 1,
                                            max=CURRICULUM_WINDOW_EPISODES)
        self._ep_seq_seen[idx] = seq[idx]

    # 三个门控量 (分母相同, 都是窗口内的有效回合数):
    #   p_stood : 站姿维持满确认窗口且当步严格几何 (**不含速度**) —— 课程推进读它
    #   p_onset : 站立窗口出现过 (单帧, 备用/诊断)
    #   p_done  : 稳定站立成功 (现行判据, 含 V/T <= 门限) —— 任务进度
    def _curriculum_metrics(self) -> tuple[float, float, float, bool, int]:
        n_valid = int(self._w_fill.sum())
        ready = bool((self._w_fill >= CURRICULUM_WINDOW_EPISODES).all())
        if n_valid == 0:
            return float("nan"), float("nan"), float("nan"), ready, 0
        return (int(self._w_stood.sum()) / n_valid,
                int(self._w_onset.sum()) / n_valid,
                int(self._w_succ.sum()) / n_valid,
                ready, n_valid)

    # 门控达标则推进一档。只升一档、只改索引, 真正的墙位变化交给随后那次重建。
    def _maybe_promote(self, it: int) -> bool:
        if self._corridor_fixed or self._cur_level >= CURRICULUM_LEVELS - 1:
            return False
        if it < CURRICULUM_START_ITER:
            return False
        if it - self._cur_level_iter < CURRICULUM_MIN_DWELL_ITER:
            return False
        # 不变量: 物理环境必须已与当前档位一致才允许推进, 否则会拿"旧物理下的成绩"缩档
        # —— 典型场景就是 iter 3000 刚开碰撞那一步 (审查 P1)。
        if self._corridor_change_needed():
            return False
        p_stood, p_onset, p_done, ready, _ = self._curriculum_metrics()
        if not ready or not (p_stood >= CURRICULUM_GATE_P_STOOD):
            return False
        self._cur_level_history.append({
            "level": self._cur_level, "iter": it,
            "wall_x_neg": WALL_X_NEG_LEVELS[self._cur_level],
            "p_stood": p_stood, "p_onset": p_onset, "p_done": p_done,
        })
        self._cur_level += 1
        self._cur_level_iter = it
        self._reset_curriculum_window()
        neg, pos = get_wall_positions_for_level(self._cur_level)
        print(f"[INFO] 墙位课程推进: 第 {self._cur_level} 档 x_neg={neg:.4f} x_pos={pos:.4f} "
              f"(净宽 {pos - neg - 0.02:.4f} m), iter {it}, "
              f"上档 p_stood={p_stood:.3f} p_onset={p_onset:.3f} p_done={p_done:.3f}")
        return True

    # 每轮把课程状态写进 tensorboard; 无样本时不写 (NaN 会污染曲线)。
    def _log_curriculum(self, it: int) -> None:
        writer = getattr(self.logger, "writer", None)
        if writer is None:
            return
        neg, pos = get_wall_positions_for_level(self._cur_level)
        p_stood, p_onset, p_done, ready, n_valid = self._curriculum_metrics()
        writer.add_scalar("Curriculum/level", self._cur_level, it)
        writer.add_scalar("Curriculum/wall_x_neg", neg, it)
        writer.add_scalar("Curriculum/wall_x_pos", pos, it)
        writer.add_scalar("Curriculum/clear_width", pos - neg - 0.02, it)
        writer.add_scalar("Curriculum/window_ready", float(ready), it)
        if n_valid > 0:
            writer.add_scalar("Curriculum/n_valid_episodes", n_valid, it)
            writer.add_scalar("Curriculum/p_stood", p_stood, it)
            writer.add_scalar("Curriculum/p_onset", p_onset, it)
            writer.add_scalar("Curriculum/p_done", p_done, it)

    # 当前**实际编译进仿真**的 (墙位对, 碰撞开关), 唯一来源是场景里的实体本身。
    # 不要另存一份缓存来做比较: 缓存只记录"我们以为改成了什么", 一旦与真实环境脱节
    # (例如加载检查点后直接改了缓存), 判据会认为"无需重建"而让训练跑在错误的物理环境里。
    def _corridor_compiled_state(self) -> tuple[tuple[float, float] | None, bool | None]:
        entity = self.env.unwrapped.scene.entities.get("restricted_space")
        if entity is None:
            return None, None
        return ((float(entity.cfg.wall_x_neg), float(entity.cfg.wall_x_pos)),
                bool(entity.collision_enabled))

    # 当前该用的 (墙位对, 是否开碰撞)。墙位由课程档位决定; 锁死模式下保持启动墙位。
    # 注意碰撞开关仍按轮次切 (CURRICULUM_START_ITER), 与墙位课程无关。
    def _corridor_target(self) -> tuple[tuple[float, float] | None, bool]:
        step_counter = int(self.env.unwrapped.common_step_counter)
        collision = get_training_phase(step_counter) == 1
        if self._corridor_fixed:
            return self._corridor_start_walls, collision
        return get_wall_positions_for_level(self._cur_level), collision

    # 是否需要重建: 目标与**实际编译值**不一致 (碰撞开关变化, 或任一墙位不一致)。
    # 必须拿 compiled 值比, 不能只在非锁死模式下比: 续训从"锁死某个墙位"切进"课程档位"
    # 时目标也变了, 漏掉就会继续跑在错误的墙位里 (审查发现的续训问题第二种形态)。
    def _corridor_change_needed(self) -> bool:
        walls, collision = self._corridor_target()
        compiled_walls, compiled_collision = self._corridor_compiled_state()
        if compiled_collision is None:
            return False
        if collision != compiled_collision:
            return True
        if walls is None or compiled_walls is None:
            return False
        return max(abs(a - b) for a, b in zip(walls, compiled_walls)) > 1e-9

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
        walls, collision = self._corridor_target()
        if walls is None:
            return None
        neg, pos = walls
        old = self.env
        env_cfg = copy.deepcopy(self._corridor_env_cfg)
        env_cfg.scene.num_envs = self._corridor_num_envs
        env_cfg.events.pop("init_restricted_space", None)
        # 重建不是新实验的开始: 清掉 seed, 否则 ManagerBasedRlEnv.__init__ 会调 seed_rng ->
        # torch.manual_seed 复位全部设备的 RNG, 影响策略采样与 PPO 抽样。见 §7.11 第 10 条。
        env_cfg.seed = None
        mdp_entity.configure_restricted_space(env_cfg, wall_x_neg=neg, wall_x_pos=pos,
                                              enable_collision=collision)
        new_env = ManagerBasedRlEnv(cfg=env_cfg, device=self._corridor_device,
                                    render_mode=self._corridor_render_mode)
        # 必须在新环境**首次采样前**接管 common_step_counter: 新环境从 0 起算, 否则下一轮
        # _corridor_target() 会读到 iter 0 (阶段一), 判定碰撞又不一致而再次重建把它关回去,
        # 形成"开了又关"的反复重建。阶段只由这个计数器决定。
        new_env.common_step_counter = step_counter
        self.env = RslRlVecEnvWrapper(new_env, clip_actions=self._corridor_clip_actions)
        # 包装器 reset 之后再把计数器写一次 (reset 可能把它归零), 然后才允许取观测
        self.env.unwrapped.common_step_counter = step_counter
        # 与计数器同理, 但必须遵守 learn() 入口的开关: 关闭时保持调用方要求的固定回合长度
        if self._randomize_ep_len:
            self._randomize_episode_phase()
        # 新环境的回合发布序号从 0 重新开始, 窗口必须一起清 (否则门控永久饿死);
        # 驻留计时也必须重置到重建这一刻 —— 物理换了 (开碰撞 / 换墙位), 旧物理下的驻留时间
        # 不能算进新配置 (审查 P1: 否则开墙后仅 50 轮就允许缩档)。
        self._reset_curriculum_window()
        self._cur_level_iter = step_counter // _STEPS_PER_ITER
        self._clear_logger_episode_state()
        try:
            old.close()
        except Exception as exc:  # 旧环境关不掉不应中断训练
            print(f"[WARN] 旧环境关闭失败: {exc}")
        print(f"[INFO] 受限空间重建: 档位 {self._cur_level}/{CURRICULUM_LEVELS - 1}, "
              f"墙 x_neg={neg:.4f} x_pos={pos:.4f} (净宽 {pos - neg - 0.02:.4f} m), "
              f"碰撞={'开' if collision else '关'}, iter {step_counter // _STEPS_PER_ITER} "
              f"num_envs={self._corridor_num_envs}")
        return self.env.get_observations().to(self.device) if refresh_obs else None

    # 训练循环: 复制 rsl_rl OnPolicyRunner.learn 的主循环, 只在**每轮采样开始前**插入
    # 受限空间重建。不能只重写入口 —— 训练只调用一次 learn(6000), 阶段切换发生在循环内部。
    # 重建后必须重新取观测: obs 是循环里的局部变量, 换环境后旧观测属于旧环境。
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # 记住入口开关: 重建发生在循环内部, 必须遵守同一次调用的语义
        self._randomize_ep_len = bool(init_at_random_ep_len)
        if self._randomize_ep_len:
            self._randomize_episode_phase()

        if self._corridor_start_walls is not None and self._corridor_change_needed():
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
            # 顺序要紧, 三件事的分工:
            #   1) 先吸收本轮之前结束的回合 (此时物理仍是它们实际经历的那套);
            #   2) 再判门控 —— 由"物理必须已与当前档位一致"这条不变量保证, 开碰撞那一步
            #      (iter 3000) 不可能拿无碰撞成绩缩档 (审查 P1), 因此判门控可以放在重建之前;
            #   3) 判完之后**立刻**重建, 让升级当轮生效 —— 放到下一轮的话本轮采样仍跑在旧墙位,
            #      而 Curriculum/level 已记成新档位 (审查: "升级晚一轮生效")。
            self._ingest_episode_results()
            self._maybe_promote(it)
            if self._corridor_start_walls is not None and self._corridor_change_needed():
                obs = self._apply_corridor_rebuild(refresh_obs=True)
            self._log_curriculum(it)
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
    # 墙位课程不能再按轮次反推 (推进由能力决定), 所以**检查点记录是回放复现墙位的唯一来源**;
    # 记录缺失时必须显式指定, 不能退回"按轮次推算" (那会静默给出错误的墙)。
    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True,
             map_location: str | None = None) -> dict:
        infos = super().load(path, load_cfg, strict, map_location)
        if self._corridor_start_walls is None:
            return infos
        if (load_cfg or {}).get("actor"):
            print("[INFO] 回放加载 (load_cfg.actor=True): 保留入口已编译好的墙体配置, 不重建环境")
            return infos
        saved = (infos or {}).get("corridor_state") or {}
        # 恢复课程档位: 优先用记录里的档位; 只有墙位 (旧格式/手工记录) 时按最近档位反查。
        # 都没有就把档位当作 0 并提示 —— 不猜。
        recorded_neg = saved.get("wall_x_neg")
        if saved.get("curriculum_level") is not None:
            level = min(max(int(saved["curriculum_level"]), 0), CURRICULUM_LEVELS - 1)
            # 档位号必须与记录的墙位自洽: 档位表改过之后, 同一个档位号会指向完全不同的墙位
            # (旧末档 6 = −0.08/+0.08, 新表第 6 档 = −0.050/+0.055), 直接用会把策略瞬移进
            # 最窄的走廊。不自洽时以**实际编译过的墙位**为准反查, 并明确告知。
            if recorded_neg is not None and abs(WALL_X_NEG_LEVELS[level] - float(recorded_neg)) > 1e-9:
                level = get_level_for_wall_x_neg(float(recorded_neg))
                print(f"[WARN] 检查点档位 {saved['curriculum_level']} 与记录的墙位 "
                      f"{float(recorded_neg):+.4f} 不自洽 (档位表已改), 按墙位反查为第 {level} 档")
            self._cur_level = level
        elif recorded_neg is not None:
            self._cur_level = get_level_for_wall_x_neg(float(recorded_neg))
        elif saved.get("corridor_width") is not None:
            # 旧格式: 只记了对称间距。按最近档位折算, 并明确告知几何已经变成不对称。
            half = 0.5 * float(saved["corridor_width"])
            self._cur_level = get_level_for_wall_x_neg(-half)
            print(f"[WARN] 旧格式检查点只记录对称间距 {saved['corridor_width']}, "
                  f"已折算到第 {self._cur_level} 档 (x_neg={WALL_X_NEG_LEVELS[self._cur_level]})")
        else:
            print("[WARN] 检查点没有墙位记录, 课程档位按 0 处理; "
                  "若这是自动课程 run, 回放请显式指定墙位")
        self._cur_level_iter = int(saved.get("curriculum_iter", self.current_learning_iteration))
        self._cur_level_history = list(saved.get("curriculum_history") or [])
        # 锁死模式: 显式指定墙位 (命令行) 优先; 否则沿用检查点记录的标记, 并同步锁死模板。
        if not self._corridor_fixed_by_cli and saved.get("corridor_fixed") is not None:
            self._corridor_fixed = bool(saved["corridor_fixed"])
            if self._corridor_fixed and saved.get("wall_x_neg") is not None:
                self._corridor_start_walls = (float(saved["wall_x_neg"]),
                                              float(saved["wall_x_pos"]))
        # 判据只比较"课程/锁死目标 vs 实际编译值", 不看记录: 记录本身就是当时真实编译的值,
        # 若拿它去覆盖当前值再比较, 就会把"环境其实没编译成这个值"掩盖过去 (曾经的真 bug)。
        self._reset_curriculum_window()
        if self._corridor_change_needed():
            compiled = self._corridor_compiled_state()
            print(f"[INFO] 续训: 实际编译配置 {compiled} 与课程目标 "
                  f"{self._corridor_target()} 不一致, 首次采样前重建")
            self._apply_corridor_rebuild(refresh_obs=False)
        return infos

    # 检查点里记录**实际编译生效**的墙位、碰撞开关与课程档位, 供续训与回放精确复现
    # (不要用轮次反推: 自动课程的墙位推进由能力决定, 与轮次没有函数关系)。
    def save(self, path: str, infos=None):
        walls, collision = self._corridor_compiled_state()
        neg, pos = (walls if walls is not None else (None, None))
        extra = {
            "wall_x_neg": neg,
            "wall_x_pos": pos,
            "corridor_width": (None if walls is None else pos - neg),   # 派生量, 兼容旧读取方
            "corridor_collision": collision,
            "corridor_fixed": self._corridor_fixed,
            "curriculum_level": self._cur_level,
            "curriculum_iter": self._cur_level_iter,
            "curriculum_history": list(self._cur_level_history),
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
