import re
import tyro
import math
import torch
import pandas as pd
from pathlib import Path
from typing import Literal
from mjlab.envs import ManagerBasedRlEnv
from dataclasses import asdict, dataclass
from mjlab.viewer import NativeMujocoViewer
from mjlab.utils.wrappers import VideoRecorder
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.SQuRo_Hole.mdp.config import (
    CMD_VEL_X_IDS,
    CMD_HEIGHT_F_IDS,
    CMD_HEIGHT_H_IDS,
    CMD_GAIT_FREQ_IDS,
    CMD_CURVATURE_IDS,
    CMD_OBSTACLE_HEIGHT_IDS,
)
from mjlab.tasks.SQuRo_Hole.mdp.curriculums import (
    STEPS_PER_ITER,
    get_current_stage,
    stage_requires_collision,
    stage_uses_height_course,
)
from mjlab.tasks.SQuRo_Hole.mdp.entity import (
    apply_saved_layout,
    set_obstacle_bottom_z,
    set_obstacle_visibility,
    configure_obstacle_entity,
)
from mjlab.tasks.SQuRo_Hole.mdp.indices import (
    FOOT_NAMES,
    _MODEL_INDICES,
    FOOT_SITE_NAMES,
    _ACT_JOINT_NAMES,
    resolve_model_indices,
    resolve_foot_force_ids,
)
from mjlab.tasks.SQuRo_Hole.mdp.command import THRESHOLD_HEIGHT
from mjlab.tasks.SQuRo_Hole.mdp.curriculums import HeightCurriculumState
from mjlab.tasks.SQuRo_Hole.mdp.reference import get_reference_joint_state
from mjlab.tasks.SQuRo_Hole.rl.runner import read_env_step, read_hole_state

TASK_NAME = "Mjlab-SQuRo-Hole"



@dataclass(frozen=True)
class PlayConfig:
    agent: Literal["zero", "random", "trained"] = "trained"
    wandb_run_path: str | None = None
    wandb_checkpoint_name: str | None = None
    checkpoint_file: str | None = None
    num_envs: int | None = 1
    device: str | None = None
    video: bool = True
    video_length: int = 1000
    video_height: int | None = 1080
    video_width: int | None = 1920
    record_data: bool = True
    command_source: Literal["schedule", "fixed", "random"] = "schedule"
    fixed_velocity: float | None = None     # 期望速度
    fixed_height_f: float | None = 0.055    # 期望前肢高度
    fixed_height_h: float | None = 0.020    # 期望后肢高度
    collision: bool | None = None           # 障碍物碰撞
    obstacle_viz: bool = True              # 可视化障碍物
    obstacle_height: float | None = None    # 障碍物板底高度 (m); None = 用检查点记录的课程档



# 从 checkpoint 文件名提取训练轮次
def extract_iter_from_checkpoint(checkpoint_path: Path) -> int:
    m = re.search(r"model_(\d+)", checkpoint_path.name)
    if m:
        return int(m.group(1))
    return 0



# 从检查点路径提取视频名称
def extract_video_name_from_checkpoint(checkpoint_path: Path) -> str:
    checkpoint_name = checkpoint_path.stem
    step = checkpoint_name.split('_')[-1] if '_' in checkpoint_name else checkpoint_name
    run_dir_name = checkpoint_path.parent.parent.name
    if run_dir_name.startswith('run-'):
        timestamp = run_dir_name[4:].split('-')[0]
    else:
        timestamp = run_dir_name
    return f"{timestamp}_{step}"



# 选择可写输出目录
def resolve_output_dir(preferred: Path) -> Path:
    import tempfile
    videos_dir = preferred / "videos"
    try:
        videos_dir.mkdir(parents=True, exist_ok=True)
        probe = videos_dir / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
        return preferred
    except OSError as e:
        fallback = Path(tempfile.gettempdir()) / "mjlab_play" / preferred.name
        (fallback / "videos").mkdir(parents=True, exist_ok=True)
        print(f"[WARN] {preferred} 不可写 ({e}), 输出改到 {fallback}")
        return fallback



# 受限模式: 0 都高 / 1 前低 / 2 后低
def mode_of(height_F: float, height_H: float) -> str:
    low_F = height_F < THRESHOLD_HEIGHT
    low_H = height_H < THRESHOLD_HEIGHT
    if low_F and not low_H:
        return "前低后高"
    if low_H and not low_F:
        return "前高后低"
    if low_F and low_H:
        return "双低"
    return "都高"



class JointDataRecorder:
    def __init__(self, log_dir, video_name, num_envs=1):
        self.log_dir = log_dir
        self.video_name = video_name
        self.num_envs = num_envs
        self.data_records = []
        self.step_count = 0
        self.joint_names = [name.removesuffix("_joint") for name in _ACT_JOINT_NAMES]
        self.action_names = self.joint_names
        self.foot_names = list(FOOT_NAMES)
        self.foot_site_names = list(FOOT_SITE_NAMES)
        self._foot_site_ids = None
        self._joint_ids_resolved = None
        self.ref_names = self.joint_names


    def record_step_data(self, env, actions=None, rewards=None, dones=None):
        record = {'step': float(self.step_count)}
        unwrapped = env.unwrapped
        asset = unwrapped.scene["robot"]
        resolve_model_indices(asset)
        idx = 0

        # 记录动作
        if actions is not None:
            for i in range(actions.shape[1]):
                name = self.action_names[i] if i < len(self.action_names) else f'action_{i}'
                record[f'{name}_action'] = float(actions[0, i].item())

        # 记录足端接触力以及位置
        contact_sensor = unwrapped.scene["feet_ground_contact"]
        feet_contact = contact_sensor.data.force.reshape(unwrapped.num_envs, len(contact_sensor.primary_names), -1, 3).sum(dim=2)
        force_ids = resolve_foot_force_ids(contact_sensor)
        if self._foot_site_ids is None:
            self._foot_site_ids, _ = asset.find_sites(self.foot_site_names, preserve_order=True)
        foot_pos = asset.data.site_pos_w[idx, self._foot_site_ids]
        for i, name in enumerate(self.foot_names):
            force = feet_contact[idx, force_ids[i]]
            record[f'contact_{name}_mag'] = float(torch.norm(force).item())
            record[f'contact_{name}_x'] = float(force[0].item())
            record[f'contact_{name}_y'] = float(force[1].item())
            record[f'contact_{name}_z'] = float(force[2].item())
            record[f'foot_{name}_x'] = float(foot_pos[i, 0].item())
            record[f'foot_{name}_y'] = float(foot_pos[i, 1].item())
            record[f'foot_{name}_z'] = float(foot_pos[i, 2].item())

        # 记录基座位姿、速度、角速度
        base_pos = asset.data.root_link_pos_w[idx]
        base_lin_vel = asset.data.root_link_lin_vel_w[idx]
        base_ang_vel = asset.data.root_link_ang_vel_w[idx]
        record['base_pos_x'] = float(base_pos[0].item())
        record['base_pos_y'] = float(base_pos[1].item())
        record['base_pos_z'] = float(base_pos[2].item())
        record['base_vel_x'] = float(base_lin_vel[0].item())
        record['base_vel_y'] = float(base_lin_vel[1].item())
        record['base_vel_z'] = float(base_lin_vel[2].item())
        # 分析脚本使用的别名
        record['base_lin_vel_x'] = float(base_lin_vel[0].item())
        record['base_lin_vel_y'] = float(base_lin_vel[1].item())
        record['base_lin_vel_z'] = float(base_lin_vel[2].item())
        record['base_ang_vel_x'] = float(base_ang_vel[0].item())
        record['base_ang_vel_y'] = float(base_ang_vel[1].item())
        record['base_ang_vel_z'] = float(base_ang_vel[2].item())

        # 记录朝向
        def _quat_to_yaw(q):
            w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        f_quat = asset.data.body_link_quat_w[idx, _MODEL_INDICES.f_body_id]
        h_quat = asset.data.body_link_quat_w[idx, _MODEL_INDICES.h_body_id]
        f_body_yaw = _quat_to_yaw(f_quat)
        h_body_yaw = _quat_to_yaw(h_quat)
        record['f_body_heading'] = f_body_yaw
        record['h_body_heading'] = h_body_yaw
        # 分析脚本回退分支用的 'heading': 用根链接四元数算; 无该属性时退用 f_body_yaw
        if hasattr(asset.data, 'root_link_quat_w'):
            record['heading'] = _quat_to_yaw(asset.data.root_link_quat_w[idx])
        else:
            record['heading'] = f_body_yaw

        # 记录前后躯干高度
        f_height = float(asset.data.body_link_pos_w[idx, _MODEL_INDICES.f_body_id, 2].item())
        h_height = float(asset.data.body_link_pos_w[idx, _MODEL_INDICES.h_body_id, 2].item())
        record['F_body_height'] = f_height
        record['H_body_height'] = h_height

        # 记录命令
        command = unwrapped.command_manager.get_command("hole_cmd")[idx]
        command_columns = (
            (CMD_VEL_X_IDS, 'vel_command_x'),
            (CMD_HEIGHT_F_IDS, 'height_F_command'),
            (CMD_HEIGHT_H_IDS, 'height_H_command'),
            (CMD_GAIT_FREQ_IDS, 'gait_freq_command'),
            (CMD_CURVATURE_IDS, 'curvature_command'),
            (CMD_OBSTACLE_HEIGHT_IDS, 'obstacle_height_command'),
        )
        for i, name in command_columns:
            record[name] = float(command[i].item())
        record['mode'] = mode_of(float(command[CMD_HEIGHT_F_IDS]), float(command[CMD_HEIGHT_H_IDS])) # type: ignore
        record['height_F_error'] = f_height - float(command[CMD_HEIGHT_F_IDS].item())
        record['height_H_error'] = h_height - float(command[CMD_HEIGHT_H_IDS].item())
        # 步态基频 (Hole 任务参考表固定 2.0 Hz, 与 config.BASE_FREQ 一致)
        record['gait_freq_command'] = 2.0

        # 记录每个关节的 pos / vel / acc / torque
        joint_ids = _MODEL_INDICES.joint_ids
        joint_pos_all = asset.data.joint_pos[idx, joint_ids]
        joint_vel_all = asset.data.joint_vel[idx, joint_ids]
        joint_acc_all = (asset.data.joint_acc[idx, joint_ids] if hasattr(asset.data, 'joint_acc') else None)
        # actuator_force 的顺序也是 actuator 顺序，与 self.joint_names 一一对应
        actuator_force_all = asset.data.actuator_force[idx]

        for i, name in enumerate(self.joint_names):
            record[f'{name}_pos'] = float(joint_pos_all[i].item())
            record[f'{name}_vel'] = float(joint_vel_all[i].item())
            if joint_acc_all is not None:
                record[f'{name}_acc'] = float(joint_acc_all[i].item())
            if i < actuator_force_all.shape[0]:
                record[f'{name}_torque'] = float(actuator_force_all[i].item())

        # 记录参考状态
        ref_pos_all, ref_vel_all = get_reference_joint_state(unwrapped)
        ref_pos = ref_pos_all[idx]
        ref_vel = ref_vel_all[idx]
        for i, name in enumerate(self.ref_names):
            record[f'{name}_ref_pos'] = float(ref_pos[i].item())
            record[f'{name}_ref_vel'] = float(ref_vel[i].item())

        if rewards is not None:
            record['reward'] = float(rewards[0].item()) if torch.is_tensor(rewards) else float(rewards[0])
        if dones is not None:
            record['done'] = float(dones[0].item() if torch.is_tensor(dones) else float(dones[0]))

        self.data_records.append(record)
        self.step_count += 1


    def save_to_csv(self):
        if not self.data_records:
            print("[WARN] 没有数据可保存")
            return
        video_dir = self.log_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        csv_path = video_dir / f"{self.video_name}.csv"
        df = pd.DataFrame(self.data_records)
        df['step'] = df['step'].astype(int)
        df.to_csv(csv_path, index=False)
        print(f"[INFO] 数据已保存到: {csv_path} ({len(self.data_records)} 步 × {len(df.columns)} 列)")



class DataRecordingEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, clip_actions=None, data_recorder=None, action_scale=1.0):
        super().__init__(env, clip_actions)
        self.data_recorder = data_recorder
        self.action_scale = action_scale


    def step(self, actions):
        scaled_actions = actions * self.action_scale
        obs_dict, rew, dones, extras = super().step(scaled_actions)
        if self.data_recorder:
            self.data_recorder.record_step_data(self.env, actions=scaled_actions, rewards=rew, dones=dones)
        return obs_dict, rew, dones, extras



def run_play(cfg: PlayConfig):
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(TASK_NAME, play=True)
    agent_cfg = load_rl_cfg(TASK_NAME)

    DUMMY_MODE = cfg.agent in {"zero", "random"}
    TRAINED_MODE = not DUMMY_MODE

    log_dir: Path | None = None
    resume_path: Path | None = None
    video_name: str | None = None

    if TRAINED_MODE:
        log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
        if cfg.checkpoint_file is not None:
            resume_path = Path(cfg.checkpoint_file)
            if not resume_path.exists():
                raise FileNotFoundError(f"未找到 checkpoint 文件: {resume_path}")
            video_name = extract_video_name_from_checkpoint(resume_path)
        elif cfg.wandb_run_path is not None:
            resume_path, _ = get_wandb_checkpoint_path(
                log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name)
            video_name = extract_video_name_from_checkpoint(resume_path)
        else:
            print("请输入 --checkpoint-file 路径:")
            text = input().strip().strip('"').strip("'")
            if not text:
                raise ValueError("必须提供 checkpoint 文件路径")
            resume_path = Path(text)
            if not resume_path.exists():
                raise FileNotFoundError(f"未找到 checkpoint 文件: {resume_path}")
            video_name = extract_video_name_from_checkpoint(resume_path)
        log_dir = resume_path.parent
    else:
        log_dir = (Path("logs") / "rsl_rl" / TASK_NAME.replace("Mjlab-", "").replace("-", "_") / "dummy").resolve()
        video_name = f"dummy_{cfg.agent}"

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    # 命令来源
    fixed_height_f: float | None = None
    fixed_height_h: float | None = None

    cmd_cfg = env_cfg.commands.get("hole_cmd")
    if cmd_cfg is not None:
        if cfg.command_source == "schedule":
            # 与训练同源：前三阶段随机高度池，第四阶段单板高度规则。
            cmd_cfg.enable_height_course = True      # type: ignore[attr-defined]
            cmd_cfg.enable_course_progress = False   # type: ignore[attr-defined]
            cmd_cfg.fixed_velocity = None            # type: ignore[attr-defined]
            cmd_cfg.fixed_height_f = None            # type: ignore[attr-defined]
            cmd_cfg.fixed_height_h = None            # type: ignore[attr-defined]
            print("[INFO] 命令来源 = schedule（跟随训练：前三阶段随机高度池，第四阶段单板高度规则）")

        elif cfg.command_source == "fixed":
            # 固定和随机命令关闭高度课程
            cmd_cfg.enable_height_course = False     # type: ignore[attr-defined]
            fixed_height_f = 0.055 if cfg.fixed_height_f is None else cfg.fixed_height_f
            fixed_height_h = 0.055 if cfg.fixed_height_h is None else cfg.fixed_height_h
            cmd_cfg.fixed_height_f = fixed_height_f        # type: ignore[attr-defined]
            cmd_cfg.fixed_height_h = fixed_height_h        # type: ignore[attr-defined]
            cmd_cfg.fixed_velocity = cfg.fixed_velocity  # type: ignore[attr-defined]
            print(f"[INFO] 命令来源 = fixed "
                  f"(h_F={fixed_height_f}, h_H={fixed_height_h}, vel={cfg.fixed_velocity})")

        else:  # random
            # 固定和随机命令关闭高度课程
            cmd_cfg.enable_height_course = False     # type: ignore[attr-defined]
            cmd_cfg.fixed_velocity = None            # type: ignore[attr-defined]
            cmd_cfg.fixed_height_f = None            # type: ignore[attr-defined]
            cmd_cfg.fixed_height_h = None            # type: ignore[attr-defined]
            print("[INFO] 命令来源 = random (按阶段随机采样, 忽略所有 fixed_* 字段)")

    saved_state = read_hole_state(resume_path) if resume_path is not None else {}
    if resume_path is not None:
        recorded = read_env_step(resume_path)
        if recorded is not None:
            final_step, stage_src = recorded, "检查点 env_state"
        else:
            train_iter = extract_iter_from_checkpoint(resume_path)
            final_step, stage_src = max(0, train_iter - 10) * STEPS_PER_ITER, "检查点文件名轮次"
    else:
        final_step, stage_src = 0, "默认 (无检查点)"
    final_stage = get_current_stage(final_step)

    # 碰撞优先级: 显式 > 检查点记录 > 按上面解析出的最终阶段推断
    if cfg.collision is not None:
        collision, coll_src = cfg.collision, "命令行显式指定"
    elif saved_state.get("collision") is not None:
        collision, coll_src = bool(saved_state["collision"]), "检查点 hole_state"
    else:
        collision, coll_src = stage_requires_collision(final_stage), f"按阶段推断 (stage {final_stage})"
    # 与训练同源：热身无板，课程使用单板；显式碰撞允许提前诊断。
    course_state = dict(saved_state.get("height_course") or HeightCurriculumState().state_dict())
    if cfg.obstacle_height is not None:
        course_state["z_min"] = float(cfg.obstacle_height)
        course_state["z_max"] = float(cfg.obstacle_height)
    use_course = (
        cfg.command_source == "schedule"
        and cmd_cfg is not None
        and stage_uses_height_course(final_stage)
    )
    show_obstacle = stage_uses_height_course(final_stage) or cfg.collision is True
    if show_obstacle:
        configure_obstacle_entity(env_cfg, enable_collision=collision)
    if use_course:
        # 未显式指定板高时，固定在检查点当前最低档。
        if cfg.obstacle_height is None:
            course_state["z_max"] = course_state["z_min"]
        cmd_cfg.enable_height_course = True          # type: ignore[attr-defined]
        cmd_cfg.height_curriculum_state = course_state  # type: ignore[attr-defined]
    # 回放: 应用检查点保存的板几何与接触参数 (重建时保留自定义几何的同一套口径)
    saved_layout = (saved_state.get("layout") or []) if show_obstacle else []
    applied = apply_saved_layout(env_cfg, saved_layout)
    # 板高优先级：显式值 > 检查点最低档 > 布局或配置默认。
    obstacle_z: float | None = None
    obstacle_src = ""
    if cfg.obstacle_height is not None:
        obstacle_z, obstacle_src = float(cfg.obstacle_height), "命令行显式指定"
    elif (saved_state.get("height_course") and stage_uses_height_course(final_stage)) or use_course:
        obstacle_z = float(course_state["z_min"])
        obstacle_src = "检查点 height_course.z_min"
    n_obstacle = set_obstacle_bottom_z(env_cfg, obstacle_z) if obstacle_z is not None else 0
    # 障碍物可见性: 仅改 rgba, 不改碰撞; 放在几何应用之后以免被覆盖
    n_hidden = set_obstacle_visibility(env_cfg, visible=cfg.obstacle_viz)
    if obstacle_z is not None and n_obstacle:
        print(f"[INFO] 障碍物板底 = {obstacle_z:.4f} m (来源: {obstacle_src}; 共 {n_obstacle} 块)")
    elif obstacle_z is not None:
        print(f"[WARN] 场景里没有障碍物实体, 障碍物高度 {obstacle_z:.4f} 未生效 (来源: {obstacle_src})")
    else:
        print("[INFO] 障碍物板底 = 检查点布局/配置默认值（当前阶段未启用高度课程或未记录课程档）")
    print(f"[INFO] 阶段解析: step {final_step} → stage {final_stage} (来源: {stage_src})")
    print(f"[INFO] 限高板碰撞 = {'开' if collision else '关'} (来源: {coll_src})"
          + (f"; 已应用检查点几何 {applied} 块板" if applied else ""))
    if not cfg.obstacle_viz:
        print(f"[INFO] 限高板已隐藏 (rgba=0,0,0,0, {n_hidden} 块); 碰撞不受影响")
    if saved_state:
        print(f"[INFO] 检查点 hole_state: stage={saved_state.get('stage')}, "
              f"记录碰撞={saved_state.get('collision')}")

    # 构建输出名后缀
    suffix_parts = [cfg.command_source]
    if cfg.command_source == "fixed":
        if fixed_height_f is not None:
            suffix_parts.append(f"hF{fixed_height_f * 1000:.0f}")
        if fixed_height_h is not None:
            suffix_parts.append(f"hH{fixed_height_h * 1000:.0f}")
        if cfg.fixed_velocity is not None:
            suffix_parts.append(f"v{cfg.fixed_velocity:.2f}")
    if cfg.collision is not None:
        suffix_parts.append("col1" if cfg.collision else "col0")
    suffix = f"-{'-'.join(suffix_parts)}" if suffix_parts else ""
    if video_name is not None:
        video_name = f"{video_name}{suffix}"

    render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
    if cfg.video and DUMMY_MODE:
        print("[WARN] 虚拟智能体的视频录制已禁用")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    data_recorder = None
    if cfg.record_data:
        assert log_dir is not None
        log_dir = resolve_output_dir(log_dir)
        data_recorder = JointDataRecorder(log_dir, video_name, num_envs=env_cfg.scene.num_envs)
        print("[INFO] 启用关节数据记录")

    if TRAINED_MODE and cfg.video:
        assert log_dir is not None and video_name is not None
        env = VideoRecorder(
            env,
            video_folder=log_dir / "videos",
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            name_prefix=video_name,
            disable_logger=False,
        )
        print("[INFO] 播放期间录制视频")

    env = DataRecordingEnvWrapper(env, clip_actions=agent_cfg.clip_actions, data_recorder=data_recorder, action_scale=1.0)

    if DUMMY_MODE:
        action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
        if cfg.agent == "zero":
            class PolicyZero:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return torch.zeros(action_shape, device=env.unwrapped.device)
            policy = PolicyZero()
        else:
            class PolicyRandom:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1
            policy = PolicyRandom()
    else:
        runner_cls = load_runner_cls(TASK_NAME) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), str(log_dir), device=device)
        runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)
        env = runner.env

    # 阶段对齐
    env.unwrapped.common_step_counter = final_step
    env.reset()
    print(f"[INFO] 阶段对齐到 step {final_step} → stage {get_current_stage(final_step)} (按检查点记录)")

    try:
        viewer = NativeMujocoViewer(env, policy)
        viewer.run()
    except KeyboardInterrupt:
        print("[INFO] 用户中断播放")
    finally:
        if data_recorder:
            data_recorder.save_to_csv()
        env.close()



def main():
    args = tyro.cli(PlayConfig, description="播放 SQuRo Hole 智能体 (虚拟碰撞版)")
    run_play(args)



if __name__ == "__main__":
    main()
