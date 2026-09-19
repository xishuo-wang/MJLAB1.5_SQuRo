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
            # 降低探索激励的单变量对照；保持 L1/L2、奖励事件和成功判据不变。
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99 ** 0.5,
            lam=0.95 ** 0.5,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="SQuRo_Backup",
        save_interval=100,
        num_steps_per_env=_STEPS_PER_ITER,
        max_iterations=3_000,

        # 不做外层裁剪: MuJoCo 已按 ctrlrange 限幅, 外层 ±6 只会把真实输出挡在奖励之外。
        # 实测依据见技术细节 §2。
        clip_actions=None,
        seed=42,

        # 从头训练: 本批改了任务语义(成功不再终止、改为循环复位)与达成门控,
        # 旧 checkpoint 的 actor/critic 与课程计数都不再对应同一任务。
        resume=False,
    )
