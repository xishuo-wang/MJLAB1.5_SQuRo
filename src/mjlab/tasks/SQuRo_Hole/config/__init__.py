from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.SQuRo_Hole.rl.runner import SQuRoHoleOnPolicyRunner

from mjlab.tasks.SQuRo_Hole.SQuRo_Hole_env_cfg import SQuRo_Hole_Env_Cfg
from .rl_cfg import SQuRo_Hole_PPO_Runner_Cfg

register_mjlab_task(
    task_id="Mjlab-SQuRo-Hole",
    env_cfg=SQuRo_Hole_Env_Cfg(),
    play_env_cfg=SQuRo_Hole_Env_Cfg(play=True),
    rl_cfg=SQuRo_Hole_PPO_Runner_Cfg(),
    runner_cls=SQuRoHoleOnPolicyRunner,
)
