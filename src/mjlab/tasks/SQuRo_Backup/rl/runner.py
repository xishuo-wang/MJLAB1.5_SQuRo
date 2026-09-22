import wandb
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.SQuRo_Backup.mdp import entity as mdp_entity
from mjlab.tasks.SQuRo_Backup.mdp.curriculums import (
    STAGE1_3_ITER,
    _STEPS_PER_ITER,
    get_curriculum_corridor_width,
    get_training_phase,
)

TASK_NAME = "Mjlab-SQuRo-Backup"
# 课程宽度变化超过这个量就重建环境 (0.40 -> 0.20 线性收缩 => 约 4 次重建)
CORRIDOR_REBUILD_TOL = 0.05


class SQuRoBackupOnPolicyRunner(MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def __init__(self, env, train_cfg, log_dir=None, device="cpu") -> None:
        super().__init__(env, train_cfg, log_dir, device)
        # 受限空间的 a 与碰撞开关在编译期固化, 运行期改不了 (依据见 mdp/entity.py),
        # 所以课程推进只能靠**重建环境**。这里记录当前已编译的取值, 每轮比对课程。
        self._corridor_device = device
        entity = self.env.unwrapped.scene.entities.get("restricted_space")
        self._corridor_width = float(entity.cfg.corridor_width) if entity is not None else None
        self._corridor_collision = bool(entity.collision_enabled) if entity is not None else False
        phase = get_training_phase(int(self.env.unwrapped.common_step_counter))
        original_log = self.logger.log

        # 训练循环的每轮钩子: 阶段切换或课程宽度变化时重建环境
        def log_with_corridor_stage(*args, **kwargs):
            if self._corridor_width is not None:
                step_counter = int(self.env.unwrapped.common_step_counter)
                want_collision = get_training_phase(step_counter) == 1
                want_width = get_curriculum_corridor_width(step_counter)
                if (want_collision != self._corridor_collision
                        or abs(want_width - self._corridor_width) > CORRIDOR_REBUILD_TOL):
                    self._rebuild_corridor(step_counter, want_width, want_collision)
            return original_log(*args, **kwargs)

        self.logger.log = log_with_corridor_stage

    # 按课程重建受限空间: 换 a 或换碰撞开关都必须重建仿真模型。
    # 只换环境是安全的: PPO 的 rollout storage 由 (num_envs × num_steps_per_env) 定尺,
    # 二者不变, 且 PPO 不持有 env 引用 (已确认), 算子与统计量可继续用。
    def _rebuild_corridor(self, step_counter: int, width: float, collision: bool) -> None:
        old = self.env
        num_envs = old.unwrapped.scene.num_envs
        clip_actions = getattr(old, "clip_actions", None)
        render_mode = getattr(old.unwrapped, "render_mode", None)
        env_cfg = load_env_cfg(TASK_NAME)
        env_cfg.scene.num_envs = num_envs
        env_cfg.events.pop("init_restricted_space", None)
        mdp_entity.configure_restricted_space(env_cfg, width, enable_collision=collision)
        new_env = ManagerBasedRlEnv(cfg=env_cfg, device=self._corridor_device,
                                    render_mode=render_mode)
        new_env.common_step_counter = step_counter
        self.env = RslRlVecEnvWrapper(new_env, clip_actions=clip_actions)
        self._corridor_width = width
        self._corridor_collision = collision
        try:
            old.close()
        except Exception as exc:  # 旧环境关不掉不应中断训练
            print(f"[WARN] 旧环境关闭失败: {exc}")
        print(f"[INFO] 受限空间重建: a={width:.4f} m (净宽 {width - 0.02:.4f} m), "
              f"碰撞={'开' if collision else '关'}, iter {step_counter // _STEPS_PER_ITER} "
              f"(阶段边界 STAGE1_3_ITER={STAGE1_3_ITER}), num_envs={num_envs}")

    def save(self, path: str, infos=None):
        super().save(path, infos)
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
