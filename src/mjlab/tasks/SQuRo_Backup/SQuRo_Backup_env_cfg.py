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



ENABLE_RESTRICTED_SPACE = True                  # 受限空间开关
RESTRICTED_SPACE_WIDTH: float | None = None     # 初始墙宽



def SQuRo_Backup_Env_Cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    # SQuRo 机器人配置
    SQURO_ROBOT_CFG = get_squro_robot_cfg()


    # 观测空间配置
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


    # 动作空间配置
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.3,
            use_default_offset=True,
        )
    }


    # 事件配置
    events = {
        "reset_all": EventTermCfg(func=mdp.reset_model, mode="reset"),
        "init_restricted_space": EventTermCfg(func=mdp.init_restricted_space, mode="startup"),
    }


    # 奖励函数配置
    rewards = {
        # 奖励项
        "milestone_s1": RewardTermCfg(func=mdp.compute_s1_milestone_reward, weight=1.0),
        "milestone_s2": RewardTermCfg(func=mdp.compute_s2_milestone_reward, weight=1.0),
        "milestone_success": RewardTermCfg(func=mdp.compute_success_milestone_reward, weight=1.0),
        "progress_s1": RewardTermCfg(func=mdp.compute_s1_progress_reward, weight=1.0),
        "progress_s2": RewardTermCfg(func=mdp.compute_s2_progress_reward, weight=1.0),
        "progress_s3": RewardTermCfg(func=mdp.compute_s3_progress_reward, weight=1.0),
        "s1_shape": RewardTermCfg(func=mdp.compute_s1_shape_reward, weight=1.0),
        "mimic_pos": RewardTermCfg(func=mdp.compute_mimic_pos_reward, weight=1.0),
        "mimic_vel": RewardTermCfg(func=mdp.compute_mimic_vel_reward, weight=1.0),
        "height": RewardTermCfg(func=mdp.compute_height_reward, weight=1.0),
        # 惩罚项
        "joint_track": RewardTermCfg(func=mdp.compute_joint_track_penalty, weight=1.0),
        "spn_track": RewardTermCfg(func=mdp.compute_spn_track_penalty, weight=1.0),
        "body_track": RewardTermCfg(func=mdp.compute_body_track_penalty, weight=1.0),      
        "leg_action": RewardTermCfg(func=mdp.compute_leg_action_penalty, weight=1.0),
        "leg_pose": RewardTermCfg(func=mdp.compute_leg_pos_penalty, weight=1.0),
        "stand_still": RewardTermCfg(func=mdp.compute_stand_reward, weight=1.0),
        "action_excess": RewardTermCfg(func=mdp.compute_action_excess_penalty, weight=1.0),
        "action_L1": RewardTermCfg(func=mdp.compute_action_L1_penalty, weight=1.0),
        "action_L2": RewardTermCfg(func=mdp.compute_action_L2_penalty, weight=1.0),
        "energy": RewardTermCfg(func=mdp.compute_energy_penalty, weight=1.0),
    }


    # 终止条件配置
    terminations = {
        "timeout": TerminationTermCfg(func=lambda env: env.episode_length_buf >= env.max_episode_length, time_out=True),
    }


    # 足端碰撞体配置
    foot_names = ("FR", "FL", "HR", "HL")
    geom_names = tuple(f"{name}_foot_collision" for name in foot_names)
    
        
    # 足部接触传感器
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        secondary=ContactMatch(mode="geom", pattern="floor", entity="robot"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )


    # 命令系统配置
    commands: dict[str, CommandTermCfg] = {
        "backup_cmd": mdp.BackupCommandCfg(
            asset_name="robot",
            debug_vis=play,
        )
    }


    # 受限空间实体
    restricted_space_entities: dict = {}
    if ENABLE_RESTRICTED_SPACE:
        if RESTRICTED_SPACE_WIDTH is not None:
            wall_kwargs: dict = {"corridor_width": float(RESTRICTED_SPACE_WIDTH)}
        else:
            neg0, pos0 = mdp.get_wall_positions(0)
            wall_kwargs = {"wall_x_neg": neg0, "wall_x_pos": pos0}
        restricted_space_entities["restricted_space"] = mdp.build_restricted_space_cfg(
            enable_collision=mdp.get_training_phase(0) == 1,
            fixed_width=RESTRICTED_SPACE_WIDTH is not None,
            **wall_kwargs,
        )


    # 完整配置
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=1024,
            extent=1.0,
            entities={
                "robot": SQURO_ROBOT_CFG,
                **restricted_space_entities,
            },
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
                iterations=100,
                ls_iterations=50,
            ),
        ),
        decimation=5,
        episode_length_s=12.0,
    )
