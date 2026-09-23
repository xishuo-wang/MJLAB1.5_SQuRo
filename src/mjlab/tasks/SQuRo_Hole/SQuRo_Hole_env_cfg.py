
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
from mjlab.tasks.SQuRo_Hole import mdp
from mjlab.viewer import ViewerConfig
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.asset_zoo.robots.SQuRo.SQuRo_constants import get_squro_robot_cfg


# 三块限高板: (x 中心, 板底高度, 半长, 半宽, 半厚)
# 板底 0.05 = 含脊柱情况的最低可通行高度; 0.075 = 不含脊柱(匍匐)的最低可通行高度
HOLE_LAYOUT = (
    ("Hole1", 0.2, 0.050, 0.015),
    ("Hole2", 0.6, 0.075, 0.100),
    ("Hole3", 1.2, 0.050, 0.015),
)
HOLE_HALF_WIDTH = 0.1
HOLE_HALF_THICKNESS = 0.005


# 建三块限高板实体 (contype/conaffinity 在编译期固化, 运行期改无效)
def build_hole_entities(enable_collision: bool) -> dict:
    mask = 1 if enable_collision else 0
    entities: dict = {}
    for name, x_center, bottom, half_len in HOLE_LAYOUT:
        entities[name.lower()] = mdp.HoleEntityCfg(
            name=name,
            position=(x_center, 0.0, bottom),
            size=(half_len, HOLE_HALF_WIDTH, HOLE_HALF_THICKNESS),
            contype=mask,
            conaffinity=mask,
        )
    return entities


def SQuRo_Hole_Env_Cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    SQURO_ROBOT_CFG = get_squro_robot_cfg()

    foot_names = ("FR", "FL", "HR", "HL")
    geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

    # 观测空间: 201 维 (含 joint_acc / base_pos / 全关节位置速度)
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
        "command": ObservationTermCfg(func=mdp.generated_commands,
                                      params={"command_name": "hole_cmd"}),
    }
    critic_terms = {**policy_terms}

    observations = {
        "actor": ObservationGroupCfg(terms=policy_terms, concatenate_terms=True,
                                     enable_corruption=False),
        "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True,
                                      enable_corruption=False),
    }

    # 动作空间: 9 个执行器位置控制 (旧版只有 9 个被控关节)
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.3,
            use_default_offset=True,
        )
    }

    events = {
        "reset_all": EventTermCfg(func=mdp.reset_model, mode="reset"),
    }

    # 奖励: 权重由 curriculums 的 4 段课程给出, 这里一律 1.0
    rewards = {
        "mimic_pos": RewardTermCfg(func=mdp.compute_mimic_reward, weight=1.0),
        "mimic_vel": RewardTermCfg(func=mdp.compute_mimic_velocity_reward, weight=1.0),
        "velocity": RewardTermCfg(func=mdp.compute_linear_velocity_reward, weight=1.0),
        "height": RewardTermCfg(func=mdp.compute_height_reward, weight=1.0),
        "foot_clearance": RewardTermCfg(func=mdp.compute_foot_clearance_reward, weight=1.0),
        "angle": RewardTermCfg(func=mdp.compute_angle_reward, weight=1.0),
        "orientation": RewardTermCfg(func=mdp.compute_orientation_reward, weight=1.0),
        "smoothness": RewardTermCfg(func=mdp.compute_smoothness_penalty, weight=1.0),
        "body_contact": RewardTermCfg(func=mdp.compute_body_contact_penalty, weight=1.0),
        "stop": RewardTermCfg(func=mdp.compute_stop_reward, weight=1.0),
        "reached": RewardTermCfg(func=mdp.compute_reached_reward, weight=1.0),
    }

    terminations = {
        "timeout": TerminationTermCfg(
            func=lambda env: env.episode_length_buf >= env.max_episode_length, time_out=True),
        "fallen": TerminationTermCfg(func=mdp.check_fallen, time_out=False),
    }

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        secondary=ContactMatch(mode="geom", pattern="floor", entity="robot"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    episode_length_s = 20.0
    if play:
        commands: dict[str, CommandTermCfg] = {
            "hole_cmd": mdp.HoleCommandCfg(
                asset_name="robot",
                resampling_time_range=(2.0, 3.0),
                use_position_schedule=True,
                position_schedule=list(mdp.STAGE3_POSITION_SCHEDULE),
                debug_vis=False,
                viz=mdp.HoleCommandCfg.VizCfg(z_offset=0.1, scale=1.0),
            )
        }
    else:
        commands = {
            "hole_cmd": mdp.HoleCommandCfg(
                asset_name="robot",
                debug_vis=False,
            )
        }

    # 完整恢复: 限高板参与碰撞 (mjwarp 在编译期固化碰撞对, 运行期无法开关)
    entities = {"robot": SQURO_ROBOT_CFG, **build_hole_entities(enable_collision=True)}

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
