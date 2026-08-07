from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.SQuRo_Backup.rl import SQuRoBackupOnPolicyRunner
from mjlab.tasks.SQuRo_Backup.SQuRo_Backup_env_cfg import SQuRo_Backup_Env_Cfg
from .rl_cfg import SQuRo_Backup_PPO_Runner_Cfg


register_mjlab_task(
    task_id="Mjlab-SQuRo-Backup",
    env_cfg=SQuRo_Backup_Env_Cfg(),
    play_env_cfg=SQuRo_Backup_Env_Cfg(play=True),
    rl_cfg=SQuRo_Backup_PPO_Runner_Cfg(),
    runner_cls=SQuRoBackupOnPolicyRunner,
)
