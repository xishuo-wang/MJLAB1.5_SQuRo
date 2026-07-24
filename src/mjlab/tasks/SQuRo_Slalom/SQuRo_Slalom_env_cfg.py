# uv run train Mjlab-SQuRo-Slalom
# uv run play Mjlab-SQuRo-Slalom-Play --checkpoint_file

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
from mjlab.tasks.SQuRo_Slalom import mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.asset_zoo.robots.SQuRo.SQuRo_constants import get_squro_robot_cfg


def SQuRo_Slalom_Env_Cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    # SQuRo 机器人配置
    SQURO_ROBOT_CFG = get_squro_robot_cfg()

    # 足端碰撞体名称
    foot_names = ("FR", "FL", "HR", "HL")
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
        "heading": ObservationTermCfg(func=mdp.heading),
        "path_ref": ObservationTermCfg(func=mdp.path_ref),
        "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "slalom_cmd"}),
    }

    critic_terms = {**policy_terms}

    observations = {
        "actor": ObservationGroupCfg(terms=policy_terms, concatenate_terms=True, enable_corruption=False),
        "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True, enable_corruption=False),
    }

    # 动作空间 — 12个执行器位置控制
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
        "height": RewardTermCfg(func=mdp.compute_height_reward, weight=1.0),
        "track_vel": RewardTermCfg(func=mdp.compute_vel_track_reward, weight=1.0),
        # "track_omg": RewardTermCfg(func=mdp.compute_omg_track_reward, weight=1.0),
        "track_path": RewardTermCfg(func=mdp.compute_path_track_reward, weight=1.0),
        # "track_head": RewardTermCfg(func=mdp.compute_head_track_reward, weight=1.0),
        "action_L1": RewardTermCfg(func=mdp.compute_action_L1_penalty, weight=1.0),
        "action_L2": RewardTermCfg(func=mdp.compute_action_L2_penalty, weight=1.0),
        "energy": RewardTermCfg(func=mdp.compute_energy_penalty, weight=1.0),
    }

    # 终止条件
    terminations = {
        "timeout": TerminationTermCfg(func=lambda env: env.episode_length_buf >= env.max_episode_length, time_out=True),
        "fallen": TerminationTermCfg(func=mdp.check_fallen, time_out=False),
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

    commands: dict[str, CommandTermCfg] = {
        "slalom_cmd": mdp.SlalomCommandCfg(
            asset_name="robot",
            resampling_time_range=(20.0, 30.0),
            debug_vis=play,
            viz=mdp.SlalomCommandCfg.VizCfg(z_offset=-0.05, scale=1.0),
        )
    }


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
            elevation=-90.0,
            azimuth=90.0,
            height=1080,
            width=1920,
        ),
        sim=SimulationCfg(
            nconmax=35,
            njmax=300,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=4,
        episode_length_s=20.0,
    )
