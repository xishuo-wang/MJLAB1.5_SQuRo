from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.tasks.SQuRo_Backup.mdp.curriculums import _STEPS_PER_ITER


def SQuRo_Backup_PPO_Runner_Cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            # 控制周期由 0.02 s 减半到 0.01 s；保持每单位实际时间的折扣与 GAE 衰减。
            # 新参数连续作用两步，等于旧参数作用一步；这里 lam 是 GAE 系数，不是参考速度 λ。
            gamma=0.99 ** 0.5,
            lam=0.95 ** 0.5,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="SQuRo_Backup",
        save_interval=100,
        num_steps_per_env=_STEPS_PER_ITER,
        max_iterations=3_000,

        clip_actions=6.0,
        seed=42,

        # resume=True,
        # load_run="...",
        # load_checkpoint="model_XXX.pt",
    )
