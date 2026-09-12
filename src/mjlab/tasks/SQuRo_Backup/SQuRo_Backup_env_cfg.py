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
from mjlab.tasks.SQuRo_Backup.mdp.timing import TIME_COMPARISON_SCALE
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
    # scale 与手调脚本 (SQuRo_backup_Replay.StateMachinePolicy.action_scale=0.3) 保持一致。
    # 覆盖性核算见 scripts/Backup/check_action_scale_coverage.py:
    #   最紧的是两个扭转关节 F_body/H_body, 期望极值 ±1.57 需 action 5.233, clip=6.0 余量 1.15x
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
        "upright": RewardTermCfg(func=mdp.compute_upright_reward, weight=0.0),
        "height": RewardTermCfg(func=mdp.compute_height_reward, weight=1.0),
        "stand": RewardTermCfg(func=mdp.compute_stand_reward, weight=0.0),
        "milestone_s1": RewardTermCfg(func=mdp.compute_s1_milestone_reward, weight=2.0),
        "milestone_s2": RewardTermCfg(func=mdp.compute_s2_milestone_reward, weight=3.0),
        "milestone_success": RewardTermCfg(func=mdp.compute_task_success_milestone_reward, weight=10.0),
        # "stand_still": RewardTermCfg(func=mdp.compute_stand_still_penalty, weight=0.0),
        # "fallen": RewardTermCfg(func=mdp.compute_fallen_penalty, weight=1.0),
        # "corridor": RewardTermCfg(func=mdp.compute_corridor_reward, weight=0.0),
        # mimic/height 测试阶段: 平滑/能耗惩罚会抑制腿部大动作(站立角->支撑位), 权重置 0
        "action_L1": RewardTermCfg(func=mdp.compute_action_L1_penalty, weight=1.0),
        "action_L2": RewardTermCfg(func=mdp.compute_action_L2_penalty, weight=1.0),
        "energy": RewardTermCfg(func=mdp.compute_energy_penalty, weight=1.0),
    }

    # 终止条件
    terminations = {
        "timeout": TerminationTermCfg(func=lambda env: env.episode_length_buf >= env.max_episode_length, time_out=True),
        "stand": TerminationTermCfg(func=mdp.check_stand_success, time_out=False),
    }

    # 命令系统
    commands: dict[str, CommandTermCfg] = {
        "backup_cmd": mdp.BackupCommandCfg(
            asset_name="robot",
            debug_vis=play,
            fixed_time_scale=TIME_COMPARISON_SCALE,
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
