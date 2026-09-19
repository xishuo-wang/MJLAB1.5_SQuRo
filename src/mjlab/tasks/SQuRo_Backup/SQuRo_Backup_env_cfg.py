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

    # 动作空间
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

    # 奖励函数 — 权重一律 1.0, 实际权重见 mdp/curriculums.py 的 _CURVES
    rewards = {
        "mimic_pos": RewardTermCfg(func=mdp.compute_mimic_pos_reward, weight=1.0),
        "mimic_vel": RewardTermCfg(func=mdp.compute_mimic_vel_reward, weight=1.0),
        "spine_target": RewardTermCfg(func=mdp.compute_spine_target_cost, weight=1.0),
        "leg_target": RewardTermCfg(func=mdp.compute_leg_target_cost, weight=1.0),
        "track_joint": RewardTermCfg(func=mdp.compute_joint_track_cost, weight=1.0),
        "height": RewardTermCfg(func=mdp.compute_height_reward, weight=1.0),
        "milestone_s1": RewardTermCfg(func=mdp.compute_s1_milestone_reward, weight=1.0),
        "milestone_s2": RewardTermCfg(func=mdp.compute_s2_milestone_reward, weight=1.0),
        "milestone_success": RewardTermCfg(func=mdp.compute_task_success_milestone_reward, weight=1.0),
        "progress_s1": RewardTermCfg(func=mdp.compute_s1_progress_reward, weight=1.0),
        "progress_s2": RewardTermCfg(func=mdp.compute_s2_progress_reward, weight=1.0),
        "progress_s3": RewardTermCfg(func=mdp.compute_s3_progress_reward, weight=1.0),
        "stand_still": RewardTermCfg(func=mdp.compute_stand_still_reward, weight=1.0),
        "action_excess": RewardTermCfg(func=mdp.compute_action_ctrl_excess_penalty, weight=1.0),
        "action_L1": RewardTermCfg(func=mdp.compute_action_L1_penalty, weight=1.0),
        "action_L2": RewardTermCfg(func=mdp.compute_action_L2_penalty, weight=1.0),
        "energy": RewardTermCfg(func=mdp.compute_energy_penalty, weight=1.0),
        # 未启用, 需要时取消注释并在 _CURVES 补对应权重
        # "upright":     RewardTermCfg(func=mdp.compute_upright_reward),
        # "stand":       RewardTermCfg(func=mdp.compute_stand_reward),
        # "fallen":      RewardTermCfg(func=mdp.compute_fallen_penalty),
        # "corridor":    RewardTermCfg(func=mdp.compute_corridor_reward),
    }

    # 终止条件 — 只有超时。"站稳"已改为非终止的循环完成事件(见 mdp/terminations.py 注释):
    # 完成即部分复位开始下一次翻正, 回合只在 episode_length_s 上限处截断。
    terminations = {
        "timeout": TerminationTermCfg(func=lambda env: env.episode_length_buf >= env.max_episode_length, time_out=True),
    }

    # 命令系统
    commands: dict[str, CommandTermCfg] = {
        "backup_cmd": mdp.BackupCommandCfg(
            asset_name="robot",
            debug_vis=play,
            fixed_time_scale=mdp.TIME_COMPARISON_SCALE if play else None,
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
                iterations=100,
                ls_iterations=50,
            ),
        ),
        decimation=5,
        episode_length_s=10.0,
    )
