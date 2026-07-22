# uv run train Mjlab-Mouse
# uv run play Mjlab-Mouse-Play --checkpoint_file

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
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
from mjlab.tasks.mouse import mdp
from mjlab.viewer import ViewerConfig
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.asset_zoo.robots.SQuRo.SQuRo_constants import get_mouse_robot_cfg


def Mouse_Env_Cfg(play: bool = False) -> ManagerBasedRlEnvCfg:   
    # 获取 mouse 机器人配置
    MOUSE_ROBOT_CFG = get_mouse_robot_cfg()

    # Mouse 特定配置
    foot_names = ("FR", "FL", "HR", "HL")
    geom_names = tuple(f"{name}_foot_collision" for name in foot_names)
    

    # 观测空间配置
    policy_terms = {
        "actions": ObservationTermCfg(func=mdp.last_action, history_length=3),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "joint_acc": ObservationTermCfg(func=mdp.joint_acc),
        "base_pos": ObservationTermCfg(func=mdp.base_pos),
        "base_lin_vel_w": ObservationTermCfg(func=mdp.base_lin_vel_w),
        "actuator_force": ObservationTermCfg(func=mdp.actuator_force),     
        "heading": ObservationTermCfg(func=mdp.heading),
        "ref_joint_pos": ObservationTermCfg(func=mdp.ref_joint_pos),
        "ref_joint_vel": ObservationTermCfg(func=mdp.ref_joint_vel),
        "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "mouse_cmd"}),
    }

    critic_terms = {
        **policy_terms,
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
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
        "mimic_pos": RewardTermCfg(func=mdp.compute_mimic_reward, weight=1.0),
        "mimic_vel": RewardTermCfg(func=mdp.compute_mimic_velocity_reward, weight=1.0),
        "velocity": RewardTermCfg(func=mdp.compute_linear_velocity_reward, weight=1.0),
        "height": RewardTermCfg(func=mdp.compute_height_reward, weight=1.0),
        "foot_clearance": RewardTermCfg( func=mdp.compute_foot_clearance_reward, weight=1.0),
        "angle": RewardTermCfg( func=mdp.compute_angle_reward, weight=1.0),
        "orientation": RewardTermCfg( func=mdp.compute_orientation_reward, weight=1.0),
        "smoothness": RewardTermCfg(func=mdp.compute_smoothness_penalty, weight=1.0),
        "body_contact": RewardTermCfg(func=mdp.compute_body_contact_penalty, weight=1.0),
        "update": RewardTermCfg(func=mdp.update_curriculum, weight=0.0),
        "stop": RewardTermCfg(func=mdp.compute_stop_reward, weight=1.0),
        "reached": RewardTermCfg(func=mdp.compute_reached_reward, weight=1.0),
    }


    # 终止条件配置
    terminations = {
        "timeout": TerminationTermCfg(func=lambda env: env.episode_length_buf >= env.max_episode_length, time_out=True),
        "fallen": TerminationTermCfg(func=mdp.check_fallen, time_out=False),
        # "reached": TerminationTermCfg(func=mdp.check_reach_goal, time_out=True),
    }


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
        episode_length_s = 20.0 
        commands: dict[str, CommandTermCfg] = {
            "mouse_cmd": mdp.MouseCommandCfg(
                asset_name="robot",
                resampling_time_range=(2.0, 3.0),  
                use_position_schedule=True,  # 启用位置表
                position_schedule=[
                    (0.0, 0.02, 0.05),
                    (0.2, 0.06, 0.02),
                    (0.32, 0.06, 0.06),
                    (0.4, 0.04, 0.04),
                    (0.8, 0.06, 0.06),
                    (1.0, 0.02, 0.05),
                    (1.2, 0.06, 0.02),
                    (1.32, 0.06, 0.06),
                ],
                debug_vis=False, 
                viz=mdp.MouseCommandCfg.VizCfg(z_offset=0.1, scale=1.0,)
            )
        }
        entities={
            "robot": MOUSE_ROBOT_CFG,
            "hole1": mdp.HoleEntityCfg(name="Hole1", position=(0.2, 0.0, 0.05), size=(0.015, 0.1, 0.005)),
            "hole2": mdp.HoleEntityCfg(name="Hole2", position=(0.6, 0.0, 0.075), size=(0.1, 0.1, 0.005)),
            "hole3": mdp.HoleEntityCfg(name="Hole3", position=(1.2, 0.0, 0.05), size=(0.015, 0.1, 0.005)),
        }
    else:
        episode_length_s = 20.0
        commands: dict[str, CommandTermCfg] = {
            "mouse_cmd": mdp.MouseCommandCfg(
                asset_name="robot",
                debug_vis=False, 
            )
        }
        entities = {
            "robot": MOUSE_ROBOT_CFG,
            "hole1": mdp.HoleEntityCfg(name="Hole1", position=(0.2, 0.0, 0.05), size=(0.015, 0.1, 0.005),
                                       contype=0, conaffinity=0),
            "hole2": mdp.HoleEntityCfg(name="Hole2", position=(0.6, 0.0, 0.075), size=(0.1, 0.1, 0.005),
                                       contype=0, conaffinity=0),
            "hole3": mdp.HoleEntityCfg(name="Hole3", position=(1.2, 0.0, 0.05), size=(0.015, 0.1, 0.005),
                                       contype=0, conaffinity=0),
        }


    # 完整配置
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=1024,
            extent=1.0,
            entities=entities,
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
            distance=0.3,     
            elevation=0.0,
            azimuth=90.0,
            height=1080,
            width=1920,
        ),
        sim=SimulationCfg(
            nconmax=35,
            njmax=300,
            mujoco=MujocoCfg(
                timestep=0.001,  
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=5,
        episode_length_s=episode_length_s,
    )