from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.tasks.SQuRo_Backup.mdp.curriculums import (
    CURRICULUM_MAX_ITER,
    _STEPS_PER_ITER,
)



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
            gamma=0.99 ** 0.5,
            lam=0.95 ** 0.5,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="SQuRo_Backup",
        save_interval=100,
        num_steps_per_env=_STEPS_PER_ITER,
        # 总轮数 = CURRICULUM_MAX_ITER: 墙位课程按能力推进, 走完档位所需轮数不再是常量,
        # 所以预算与 STAGE2_2_ITER (奖励权重/λ 课程的终点) 解耦。
        max_iterations=CURRICULUM_MAX_ITER,

        clip_actions=None,
        seed=42,

        # 继承训练入口 (2026-09-26): 从旧 run 的 model_4000 继续。
        # 为什么是 4000 而不是末档 (3600) 或最后一个 (6700): 3600 刚切进最窄档、actor 还没适应,
        # 6700 已经在旧墙位下跑了 3100 轮; 4000 是"已进入最窄档并稳定下来"的第一个整百轮次,
        # 同时与脊柱侧摆/俯仰的放宽点 (SPN_AXIS_RELAX_ITER) 对齐。
        # 换实验时把下面三行改回 resume=False 即可 (load_run/load_checkpoint 只在 resume 时生效)。
        resume=True,
        load_run="2026-09-26_01-32-03",
        load_checkpoint="model_4000.pt",
    )
