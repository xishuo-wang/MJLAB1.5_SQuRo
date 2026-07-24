"""自定义 W&B 日志写入器，将 wandb 内部文件与训练日志分离存储。"""

import os
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.utils.wandb_log_writer import WandbLogWriter

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


class MjlabWandbLogWriter(WandbLogWriter):
    """WandbLogWriter 的变体，将 wandb 内部文件存放在训练日志目录外部。

    原生 WandbLogWriter 调用 wandb.init(dir=log_dir)，导致 wandb 的内部文件
    （包括媒体副本）存放在训练日志目录下的 wandb/ 子目录中。
    当 logger.py 通过 rglob("*.mp4") 递归扫描 log_dir 时，会扫到 wandb 已拷贝
    的视频副本，再次尝试复制到同一路径时触发 SameFileError。

    本类将 wandb.init(dir=...) 指向 log_dir 的父级目录，使 wandb 内部文件
    与逐次运行的训练日志（videos/params/checkpoints）完全隔离。
    """

    def __init__(self, log_dir: str, project_name: str) -> None:
        if wandb is None:
            raise ModuleNotFoundError(
                "wandb package is required to log to Weights and Biases."
            )
        SummaryWriter.__init__(self, log_dir, flush_secs=10)

        run_name = os.path.split(log_dir)[-1]
        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None

        # wandb 内部文件存放到训练目录的父级，避免 rglob 扫到已拷贝的媒体副本
        wandb_dir = os.path.dirname(log_dir)

        wandb.init(
            project=project_name,
            entity=entity,
            name=run_name,
            id=run_name,
            dir=wandb_dir,
            config={"log_dir": log_dir},
            settings=wandb.Settings(start_method="thread"),
        )
        self.logged_videos: set[str] = set()
