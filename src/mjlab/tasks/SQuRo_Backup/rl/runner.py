import wandb
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.SQuRo_Backup.mdp.curriculums import get_curriculum_corridor_width


class SQuRoBackupOnPolicyRunner(MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def __init__(self, env, train_cfg, log_dir=None, device="cpu") -> None:
        super().__init__(env, train_cfg, log_dir, device)
        # 受限空间的 a 与碰撞开关都在编译期固化, 运行期改不了 (依据见 mdp/entity.py)。
        # 这里只在课程要求的 a 与固化值不一致时告警一次: 那说明该重建环境了。
        unwrapped = self.env.unwrapped
        entity = unwrapped.scene.entities.get("restricted_space")
        state = {"warned": False}
        if entity is not None:
            compiled = float(entity.cfg.corridor_width)
            original_log = self.logger.log

            # 训练循环的每轮钩子
            def log_with_corridor_check(*args, **kwargs):
                wanted = get_curriculum_corridor_width(int(unwrapped.common_step_counter))
                if (not state["warned"]) and abs(wanted - compiled) > 1e-6:
                    state["warned"] = True
                    print(f"[WARN] 受限空间墙间距已固化在 a={compiled:.4f} m (编译期), "
                          f"但课程当前要求 a={wanted:.4f} m。运行期无法改动几何, "
                          f"需要重建环境 (换 env_cfg 里的 corridor_width) 才能生效。")
                return original_log(*args, **kwargs)

            self.logger.log = log_with_corridor_check

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
