# uv run train Mjlab-SQuRo-Backup
# uv run play Mjlab-SQuRo-Backup-Play --checkpoint_file

from mjlab.managers import (
    ActionTermCfg,
    CommandTermCfg,
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
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.asset_zoo.robots.SQuRo.SQuRo_constants import get_squro_robot_cfg


def SQuRo_Backup_Env_Cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    # SQuRo 机器人配置
    SQURO_ROBOT_CFG = get_squro_robot_cfg()

    # 足端碰撞体名称
    foot_names = ("FL", "FR", "HL", "HR")
    geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

    # 观测空间
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
        "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "backup_cmd"}),
    }

    critic_terms = {**policy_terms}

    observations = {
        "actor": ObservationGroupCfg(terms=policy_terms, concatenate_terms=True, enable_corruption=False),
        "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True, enable_corruption=False),
    }

    # 动作空间 — 14个执行器位置控制
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.3,
            use_default_offset=True,
        )
    }

    # 事件
    events = {
        "reset_all": EventTermCfg(func=mdp.reset_model, mode="reset"),
    }

    # 奖励函数
    rewards = {
        "mimic_pos": RewardTermCfg(func=mdp.compute_mimic_pos_reward, weight=1.0),
        "mimic_vel": RewardTermCfg(func=mdp.compute_mimic_vel_reward, weight=1.0),
        "upright": RewardTermCfg(func=mdp.compute_upright_reward, weight=1.0),
        "height": RewardTermCfg(func=mdp.compute_height_reward, weight=1.0),
        "stand": RewardTermCfg(func=mdp.compute_stand_reward, weight=1.0),
    }

    # 终止条件
    terminations = {
        "timeout": TerminationTermCfg(func=lambda env: env.episode_length_buf >= env.max_episode_length, time_out=True),
        "stand": TerminationTermCfg(func=mdp.check_stand_success, time_out=False),
    }

    # 命令系统 — 1D [time_scale λ]: 参考时间缩放 (λ=1.0 最快复位 ~1s, λ=1.5 慢速学习起点)
    commands: dict[str, CommandTermCfg] = {
        "backup_cmd": mdp.BackupCommandCfg(
            asset_name="robot",
            debug_vis=play,
        )
    }

    # 足端接触传感器
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        secondary=ContactMatch(mode="geom", pattern="floor", entity="robot"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    
    # 完整配置
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=1024,
            extent=1.0,
            entities={"robot": SQURO_ROBOT_CFG},
            sensors=(feet_ground_cfg,),
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_Link",
            distance=0.5,
            elevation=-45.0,
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
                ls_iterations=20,
            ),
        ),
        decimation=4,
        episode_length_s=3.0,
    )
