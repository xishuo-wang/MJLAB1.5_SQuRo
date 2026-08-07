# uv run train Mjlab-SQuRo-Backup

from mjlab.managers import (
    ActionTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.viewer import ViewerConfig
from mjlab.tasks.SQuRo_Backup import mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.asset_zoo.robots.SQuRo_Backup.SQuRo_Backup_constants import (
    get_squro_backup_robot_cfg,
)


def SQuRo_Backup_Env_Cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    # SQuRo 跌倒爬起专用机器人配置
    SQURO_ROBOT_CFG = get_squro_backup_robot_cfg()

    # 观测空间 — 关节状态 + 参考轨迹 + 机身状态（跌倒爬起不需要路径/命令）
    policy_terms = {
        "actions": ObservationTermCfg(func=mdp.last_action, history_length=2),
        "ref_joint_pos": ObservationTermCfg(func=mdp.ref_joint_pos),
        "ref_joint_vel": ObservationTermCfg(func=mdp.ref_joint_vel),
        "actuator_pos": ObservationTermCfg(func=mdp.actuator_pos),
        "actuator_vel": ObservationTermCfg(func=mdp.actuator_vel),
        "actuator_force": ObservationTermCfg(func=mdp.actuator_force),
        "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    }

    critic_terms = {**policy_terms}

    observations = {
        "actor": ObservationGroupCfg(terms=policy_terms, concatenate_terms=True, enable_corruption=False),
        "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True, enable_corruption=False),
    }

    # 动作空间 — 14个执行器位置控制
    # scale=0.5 + clip_actions=3.5 → 最大偏移 ±1.75 rad, 覆盖爬起动作幅度(脊柱 ±1.57)
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,
            use_default_offset=True,
        )
    }

    # 事件
    events = {
        "reset_all": EventTermCfg(func=mdp.reset_model, mode="reset"),
    }

    # 奖励函数 — 第一阶段: 模仿 + 竖直/高度引导 (不加走廊等复杂奖励)
    rewards = {
        "mimic_pos": RewardTermCfg(func=mdp.compute_mimic_pos_reward, weight=1.0),
        "mimic_vel": RewardTermCfg(func=mdp.compute_mimic_vel_reward, weight=1.0),
        "upright": RewardTermCfg(func=mdp.compute_upright_reward, weight=1.0),
        "height": RewardTermCfg(func=mdp.compute_height_reward, weight=1.0),
    }

    # 终止条件 — 仅超时（跌倒不是终止条件）
    terminations = {
        "timeout": TerminationTermCfg(func=lambda env: env.episode_length_buf >= env.max_episode_length, time_out=True),
    }

    # 完整配置
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=512,
            extent=1.0,
            entities={"robot": SQURO_ROBOT_CFG},
            sensors=(),
        ),
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_Link",
            distance=0.5,
            elevation=-30.0,
            azimuth=90.0,
            height=1080,
            width=1920,
        ),
        sim=SimulationCfg(
            nconmax=100,
            njmax=300,
            mujoco=MujocoCfg(
                timestep=0.002,
                iterations=50,
                ls_iterations=50,
                tolerance=1e-9,
            ),
        ),
        decimation=4,
        episode_length_s=6.0,
    )
