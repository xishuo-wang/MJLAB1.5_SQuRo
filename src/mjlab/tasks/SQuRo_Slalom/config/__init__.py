from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.SQuRo_Slalom.rl import SQuRoOnPolicyRunner
from mjlab.tasks.SQuRo_Slalom.SQuRo_env_cfg import SQuRo_Env_Cfg
from .rl_cfg import SQuRo_PPO_Runner_Cfg


register_mjlab_task(
    task_id="Mjlab-SQuRo-Slalom",
    env_cfg=SQuRo_Env_Cfg(),
    play_env_cfg=SQuRo_Env_Cfg(play=True),
    rl_cfg=SQuRo_PPO_Runner_Cfg(),
    runner_cls=SQuRoOnPolicyRunner,
)