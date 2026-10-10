# uv run train Mjlab-SQuRo-Hole
# uv run play Mjlab-SQuRo-Hole-Play --checkpoint_file

from mjlab.managers import (
    EventTermCfg,
    ActionTermCfg,
    RewardTermCfg,
    CommandTermCfg,
    ObservationTermCfg,
    TerminationTermCfg,
    ObservationGroupCfg,
)
from mjlab.scene import SceneCfg
from mjlab.viewer import ViewerConfig
from mjlab.tasks.SQuRo_Hole import mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.asset_zoo.robots.SQuRo.SQuRo_constants import get_squro_robot_cfg



def SQuRo_Hole_Env_Cfg(play: bool = False) -> ManagerBasedRlEnvCfg:   
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
        "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "hole_cmd"}),
    }


    critic_terms = {**policy_terms,}


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
    }


    # 奖励函数配置
    rewards = {
        # 奖励项
        "mimic_pos": RewardTermCfg(func=mdp.compute_mimic_pos_reward, weight=1.0),
        "mimic_vel": RewardTermCfg(func=mdp.compute_mimic_vel_reward, weight=1.0),
        "velocity": RewardTermCfg(func=mdp.compute_vel_reward, weight=1.0),
        "height": RewardTermCfg(func=mdp.compute_height_reward1, weight=1.0),
        "angle": RewardTermCfg( func=mdp.compute_angle_reward, weight=1.0),
        "orientation": RewardTermCfg( func=mdp.compute_orientation_reward, weight=1.0),
        "body_contact": RewardTermCfg(func=mdp.compute_body_contact_reward, weight=1.0),
        # 惩罚项
        "smoothness": RewardTermCfg(func=mdp.compute_smoothness_penalty, weight=1.0),
        "stop": RewardTermCfg(func=mdp.compute_stop_reward, weight=1.0),
    }


    # 终止条件配置
    terminations = {
        "timeout": TerminationTermCfg(func=lambda env: env.episode_length_buf >= env.max_episode_length, time_out=True),
        "fallen": TerminationTermCfg(func=mdp.check_fallen, time_out=False),
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


    # 播放模式配置
    if play:
        commands: dict[str, CommandTermCfg] = {
            "hole_cmd": mdp.HoleCommandCfg(
                asset_name="robot",
                debug_vis=True, 
                viz=mdp.HoleCommandCfg.VizCfg(z_offset=0.1, scale=1.0,)
            )
        }
    else:
        commands: dict[str, CommandTermCfg] = {
            "hole_cmd": mdp.HoleCommandCfg(
                asset_name="robot",
                debug_vis=False, 
            )
        }


    # 完整配置
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=1024,
            extent=1.0,
            entities={
                "robot": SQURO_ROBOT_CFG,
                **mdp.build_hole_entities(enable_collision=False)
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
            elevation=0.0,
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
        episode_length_s=20.0,
    )