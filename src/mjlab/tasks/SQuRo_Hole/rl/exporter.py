import os

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_base_metadata,
)


# 给 ONNX 附加基座元数据; 1.5 已提供底层实现, 这里只保留原版的入参形式
def attach_onnx_metadata(
  env: ManagerBasedRlEnv, run_path: str, path: str, filename="policy.onnx"
) -> None:
    onnx_path = os.path.join(path, filename)
    metadata = get_base_metadata(env, run_path)
    attach_metadata_to_onnx(onnx_path, metadata)
