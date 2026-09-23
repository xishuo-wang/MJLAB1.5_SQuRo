from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.SQuRo_Tunnel.rl.runner import SQuRoOnPolicyRunner
from mjlab.tasks.SQuRo_Tunnel.SQuRo_Tunnel_env_cfg import SQuRo_Tunnel_Env_Cfg
from .rl_cfg import SQuRo_Tunnel_PPO_Runner_Cfg


register_mjlab_task(
    task_id="Mjlab-SQuRo-Tunnel",
    env_cfg=SQuRo_Tunnel_Env_Cfg(),
    play_env_cfg=SQuRo_Tunnel_Env_Cfg(play=True),
    rl_cfg=SQuRo_Tunnel_PPO_Runner_Cfg(),
    runner_cls=SQuRoOnPolicyRunner,
)