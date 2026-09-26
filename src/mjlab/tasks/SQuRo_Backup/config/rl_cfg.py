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

        # 继承训练入口 (2026-09-26 第二批): 从旧 run 的 model_3000 继续。
        # 为什么取 3000: 它是 STAGE2 的起点 (碰撞与墙位课程从这一轮开始), 此时的策略
        # **从未见过任何墙体**, 因此不会把上一批"卡在墙之间通过判据"的习惯带进来。
        # 墙位由档位表决定, 该检查点记录的 (-0.20, +0.08) 与新表不自洽, runner 会按墙位反查
        # 落到第 0 档 (-0.08, +0.055), 净宽 0.115, 课程再从第 0 档正常往上走。
        # 本批唯一变量: weight_leg_pose 2.0 -> 6.0 (见 mdp/curriculums.py 的说明)。
        # 换实验时把下面三行改回 resume=False 即可 (load_run/load_checkpoint 只在 resume 时生效)。
        resume=True,
        load_run="2026-09-26_01-32-03",
        load_checkpoint="model_3000.pt",
    )
