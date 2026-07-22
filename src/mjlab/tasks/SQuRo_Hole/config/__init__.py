from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.SQuRo_Hole.rl.runner import MouseOnPolicyRunner

from mjlab.tasks.SQuRo_Hole.mouse_env_cfg import Mouse_Env_Cfg
from .rl_cfg import Mouse_PPO_Runner_Cfg

register_mjlab_task(
    task_id="Mjlab-Mouse",
    env_cfg=Mouse_Env_Cfg(),
    play_env_cfg=Mouse_Env_Cfg(play=True),
    rl_cfg=Mouse_PPO_Runner_Cfg(),
    runner_cls=MouseOnPolicyRunner,
)