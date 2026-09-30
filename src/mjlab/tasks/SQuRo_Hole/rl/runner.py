import os

import wandb
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner

from mjlab.tasks.SQuRo_Hole.rl.exporter import (
    attach_onnx_metadata,
)


class SQuRoHoleOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    """Save the model and training information."""
    super().save(path, infos)
    if self.logger.logger_type in ["wandb"]:
        policy_path = path.split("model")[0]
        filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
        # 1.5 的导出入参已自带 normalizer 处理, 原来的 export_*_policy_as_onnx 由框架方法取代
        self.export_policy_to_onnx(policy_path, filename)
        attach_onnx_metadata(
            self.env.unwrapped,
            wandb.run.name,  # type: ignore
            path=policy_path,
            filename=filename,
        )
        wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
