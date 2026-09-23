import copy

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
        self._corridor_pending = False
        original_log = self.logger.log

        # 每轮只做标记, 真正的重建放到下一轮采样**开始之前** (见 learn 的 override):
        # 在 logger.log 里换环境会让父类局部变量 obs 仍是旧环境的观测, 下一步动作
        # 就用错状态的观测去打新环境 (实测可复现), 所以必须等下一次取观测时再换。
        def log_with_corridor_mark(*args, **kwargs):
            if self._corridor_width is not None:
                self._corridor_pending = self._corridor_change_needed()
            return original_log(*args, **kwargs)

        self.logger.log = log_with_corridor_mark

    # 当前轮次该用的 (宽度, 是否开碰撞); 锁死宽度模式下宽度不变
    def _corridor_target(self) -> tuple[float, bool]:
        step_counter = int(self.env.unwrapped.common_step_counter)
        iter_num = step_counter // _STEPS_PER_ITER
        collision = get_training_phase(step_counter) == 1
        width = (self._corridor_width if self._corridor_fixed
                 else get_corridor_width_for_iter(iter_num))
        return float(width), collision

    # 是否需要重建: 碰撞开关变化, 或 (非锁死宽度时) 宽度档位变化。
    # 注意碰撞判断必须放在 fixed 分支之外 —— 阶段边界处即使宽度锁死也要重建以打开碰撞。
    def _corridor_change_needed(self) -> bool:
        width, collision = self._corridor_target()
        if collision != self._corridor_collision:
            return True
        return (not self._corridor_fixed) and abs(width - self._corridor_width) > 1e-9

    # 父类 learn 的开头: 在取观测之前把待重建的环境换掉, 保证 obs 来自新环境
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if self._corridor_pending:
            self._corridor_pending = False
            self._apply_corridor_rebuild()
        super().learn(num_learning_iterations, init_at_random_ep_len)

    # 按本次启动的 env_cfg 模板重建环境, 只替换受限空间实体
    def _apply_corridor_rebuild(self) -> None:
        step_counter = int(self.env.unwrapped.common_step_counter)
        width, collision = self._corridor_target()
        old = self.env
        env_cfg = copy.deepcopy(self._corridor_env_cfg)
        env_cfg.scene.num_envs = self._corridor_num_envs
        env_cfg.events.pop("init_restricted_space", None)
        mdp_entity.configure_restricted_space(env_cfg, width, enable_collision=collision)
        new_env = ManagerBasedRlEnv(cfg=env_cfg, device=self._corridor_device,
                                    render_mode=self._corridor_render_mode)
        new_env.common_step_counter = step_counter
        self.env = RslRlVecEnvWrapper(new_env, clip_actions=self._corridor_clip_actions)
        self._corridor_width = width
        self._corridor_collision = collision
        # 重建后旧回合已中断, 未结束回合的累计必须清掉, 否则会跨重建拼接统计
        try:
            self.logger.ep_infos.clear()
        except Exception:
            pass
        try:
            old.close()
        except Exception as exc:  # 旧环境关不掉不应中断训练
            print(f"[WARN] 旧环境关闭失败: {exc}")
        print(f"[INFO] 受限空间重建: a={width:.4f} m (净宽 {width - 0.02:.4f} m), "
              f"碰撞={'开' if collision else '关'}, iter {step_counter // _STEPS_PER_ITER} "
              f"(阶段边界 STAGE1_3_ITER={STAGE1_3_ITER}), num_envs={self._corridor_num_envs}")

    # 检查点里记录实际生效的墙宽与碰撞开关, 供回放精确复现 (不要用轮次反推)
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
