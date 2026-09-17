from __future__ import annotations
import torch
from mjlab.entity import Entity
from typing import TYPE_CHECKING, cast
from .command import BackupCommand
from .curriculums import get_curriculum_reward_weight
from .reference import get_reference_joint_state, get_body_reference
from .indices import _ACTUATED_JOINT_NAMES, _ACTUATOR_CTRL_RANGE, _MODEL_INDICES
from .timing import STAND_STILL_FULL_SPEED

if TYPE_CHECKING:
    from mjlab.envs.mdp.actions import JointPositionAction
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv



_STAND_UP_THRESHOLD = 0.8     # 站起奖励: 竖直度下限 (身体基本竖直才给站直奖励)
_TARGET_HEIGHT = 0.055        # 站直目标高度 (m, 与命令 height_f/h 一致)
# 跌倒滞留惩罚阈值
_FALLEN_GROUND_H = 0.03       # F/H body 贴地高度阈值 (m, 贴地≈0.024)
_FALLEN_LIN_THRESHOLD = 0.05  # 贴地时水平线速度低于此值视为"不动" (m/s)
_FALLEN_ANG_THRESHOLD = 0.5   # 贴地时角速度低于此值视为"不动" (rad/s)
# 走廊 (YoZ 平面) 参数
_CORRIDOR_HALF = 0.05         # 走廊半宽/死区 (m), 前后肢共用
_BODY_SEG_HALF = 0.025        # 身体段半径 (m, YoZ 截面包络)



# =========================================================================================
# s1里程碑奖励
def compute_s1_milestone_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    # 读取每个 episode 的首次里程碑脉冲；转移脉冲仍保留给诊断使用。
    # 不再单独写日志: 脉冲是 Progress/enter_p2 的导数, 累积量更好读。
    pulse = command.s1_milestone_pulse
    weight = get_curriculum_reward_weight(env, "weight_milestone_s1")
    return weight * pulse.float() / env.step_dt



# =========================================================================================
# s2里程碑奖励
def compute_s2_milestone_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    # 读取每个 episode 的首次里程碑脉冲；重复回退/重试不再重复奖励。
    pulse = command.s2_milestone_pulse
    weight = get_curriculum_reward_weight(env, "weight_milestone_s2")
    return weight * pulse.float() / env.step_dt



def compute_task_success_milestone_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    pulse = env.termination_manager.get_term("stand")
    env.extras["log"]["Progress/success"] = pulse.float().mean().item()
    weight = get_curriculum_reward_weight(env, "weight_milestone_success")
    return weight * pulse.float() / env.step_dt



# =========================================================================================
# s1区间奖励 — 朝 S1 姿态 (前段仰面 + 后段俯卧) 的连续进度
def compute_s1_progress_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    weight = get_curriculum_reward_weight(env, "weight_progress_s1")
    return weight * command.progress_s1



# =========================================================================================
# s2区间奖励 — 朝 S2 姿态 (两段都已俯卧) 的连续进度
def compute_s2_progress_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    weight = get_curriculum_reward_weight(env, "weight_progress_s2")
    return weight * command.progress_s2


# P3 站立进度: 只对 P3 生效；前后段同时背部朝上才解锁, 越接近最终站立越大。
def compute_s3_progress_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    _, progress = command.standing_state()
    weight = get_curriculum_reward_weight(env, "weight_progress_s3")
    return weight * progress * (command.phase == 2)



# =========================================================================================
# 关节位置模仿奖励
def compute_mimic_pos_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    # 计算关节角度误差
    joint_pos = asset.data.joint_pos[:, _MODEL_INDICES.joint_ids]
    ref_pos, _ = get_reference_joint_state(env)
    error = joint_pos - ref_pos
    error_leg = error[:, _MODEL_INDICES.actuator_leg_ids]
    error_spn = error[:, _MODEL_INDICES.actuator_spn_ids]
    error_neck = error[:, _MODEL_INDICES.actuator_neck_ids]
    # 获取课程学习量
    weight = get_curriculum_reward_weight(env, "weight_mimic_pos")
    sigma_leg = get_curriculum_reward_weight(env, "sigma_leg_pos")
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_pos")
    sigma_neck = get_curriculum_reward_weight(env, "sigma_neck_pos")
    alpha_neck = get_curriculum_reward_weight(env, "alpha_neck_pos")
    # 计算奖励
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    mse_neck = torch.mean(error_neck ** 2, dim=1)
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward_neck = torch.exp(-sigma_neck * mse_neck)
    reward = (reward_leg + reward_spn) / 2 + alpha_neck * reward_neck
    return reward * weight



# =========================================================================================
# 按名称对齐关节目标成本；raw_action 已经过 RL 外层 ±6 裁剪，但尚未经过 XML 控制限幅。
def _joint_target_cost(env: "ManagerBasedRlEnv", ref_columns: tuple[int, ...]) -> torch.Tensor:
    action_term = cast("JointPositionAction", env.action_manager.get_term("joint_pos"))
    # 重建限幅前目标；不要读取实际关节角或已经限幅的控制量，否则过量指令会被隐藏。
    target = action_term.raw_action * action_term.scale + action_term.offset
    ref_pos, _ = get_reference_joint_state(env)
    # 动作项按自身关节顺序排列，参考表按固定顺序排列；用名称对齐，避免列序假设。
    target_columns = tuple(action_term.target_names.index(_ACTUATED_JOINT_NAMES[i]) for i in ref_columns)
    error = target[:, target_columns] - ref_pos[:, ref_columns]
    return -torch.mean(error.square(), dim=1)



# 四脊柱等权，课程权重在这里生效；RewardManager 只乘外层 1.0 和 dt。
def compute_spine_target_cost(env: "ManagerBasedRlEnv") -> torch.Tensor:
    weight = get_curriculum_reward_weight(env, "weight_spine_target")
    return weight * _joint_target_cost(env, _MODEL_INDICES.actuator_spn_ids)



# 执行器 ctrlrange 张量 (按动作项自身的关节顺序排列), 按名称解析一次后缓存。
_CTRL_RANGE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}



def _ctrl_range_tensors(names, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    key = (tuple(names), str(device), str(dtype))
    cached = _CTRL_RANGE_CACHE.get(key)
    if cached is None:
        lo = [0.0] * len(names)
        hi = [0.0] * len(names)
        for i, name in enumerate(names):
            lo[i], hi[i] = _ACTUATOR_CTRL_RANGE[name]
        cached = (torch.tensor(lo, device=device, dtype=dtype),
                  torch.tensor(hi, device=device, dtype=dtype))
        _CTRL_RANGE_CACHE[key] = cached
    return cached



# 动作超出执行器 ctrlrange 的成本。
# ctrllimited="true" 时 MuJoCo 会把 ctrl 直接裁到 ctrlrange: 超出部分被完全丢弃,
# 命令写 -28 和写 -4.67 (= 髋的 ctrlrange 下限 / scale) 产生完全一样的力矩。
# 实测策略的确定性输出长期停在 ±26~47, 把 8 条腿和 2 个颈关节永久钉在饱和点上,
# 腿因此完全失去控制权, 站起不可能完成。这里只对被丢弃的那一段计成本:
# 命令落在 ctrlrange 内时为 0, 所以"贴住限位撑地"这个有用行为不受惩罚。
def compute_action_ctrl_excess_penalty(env: "ManagerBasedRlEnv") -> torch.Tensor:
    action_term = cast("JointPositionAction", env.action_manager.get_term("joint_pos"))
    # 与 spine_target 同源: 用裁剪后、XML 控制限幅前的目标, 含 scale 与默认角偏移。
    target = action_term.raw_action * action_term.scale + action_term.offset
    lo, hi = _ctrl_range_tensors(action_term.target_names, target.device, target.dtype)
    excess = (lo - target).clamp(min=0.0) + (target - hi).clamp(min=0.0)
    weight = get_curriculum_reward_weight(env, "weight_action_excess")
    return -weight * excess.mean(dim=1)



# =========================================================================================
# 关节速度模仿奖励
def compute_mimic_vel_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    # 计算关节速度误差
    joint_vel = asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
    _, ref_vel = get_reference_joint_state(env)
    error = joint_vel - ref_vel
    error_leg = error[:, _MODEL_INDICES.actuator_leg_ids]
    error_spn = error[:, _MODEL_INDICES.actuator_spn_ids]
    error_neck = error[:, _MODEL_INDICES.actuator_neck_ids]
    # 获取课程学习量
    weight = get_curriculum_reward_weight(env, "weight_mimic_vel")
    sigma_leg = get_curriculum_reward_weight(env, "sigma_leg_vel")
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_vel")
    sigma_neck = get_curriculum_reward_weight(env, "sigma_neck_vel")
    alpha_neck = get_curriculum_reward_weight(env, "alpha_neck_vel")
    # 计算奖励
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    mse_neck = torch.mean(error_neck ** 2, dim=1)
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward_neck = torch.exp(-sigma_neck * mse_neck)
    reward = (reward_leg + reward_spn) / 2 + alpha_neck * reward_neck
    return reward * weight



# =========================================================================================
# 身体竖直奖励
def compute_upright_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    uprightness = asset.data.projected_gravity_b[:, 2]  # [N]
    sigma = get_curriculum_reward_weight(env, "sigma_upright")
    weight = get_curriculum_reward_weight(env, "weight_upright")
    reward = torch.exp(-sigma * (1.0 - uprightness) ** 2)
    env.extras["log"]["Data/uprightness"] = uprightness.mean().item()
    return reward * weight



# =========================================================================================
# 身体高度跟踪奖励
def compute_height_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    body_pos_w = asset.data.body_link_pos_w
    F_body_height = body_pos_w[:, _MODEL_INDICES.f_body_id, 2]
    H_body_height = body_pos_w[:, _MODEL_INDICES.h_body_id, 2]
    _, z_ref_F, _, z_ref_H = get_body_reference(env)   # [N] 时变期望高度
    height_F_error = torch.abs(z_ref_F - F_body_height)
    height_H_error = torch.abs(z_ref_H - H_body_height)
    sigma_height = get_curriculum_reward_weight(env, "sigma_height")
    w_height = get_curriculum_reward_weight(env, "weight_height")
    r_height_F = torch.exp(-sigma_height * height_F_error ** 2)
    r_height_H = torch.exp(-sigma_height * height_H_error ** 2)   
    Reward_height = w_height * (0.5 * r_height_F + 0.5 * r_height_H)
    env.extras["log"]["Body/height"] = (0.5 * F_body_height + 0.5 * H_body_height).mean().item()
    return Reward_height


# =========================================================================================
# 走廊一致性奖励 (YoZ 平面, 前/后肢独立) — 约束 F/H body 的 (y,z) 贴近手调参考轨迹走廊
def compute_corridor_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    body_pos_w = asset.data.body_link_pos_w
    f_y = body_pos_w[:, _MODEL_INDICES.f_body_id, 1]
    f_z = body_pos_w[:, _MODEL_INDICES.f_body_id, 2]
    h_y = body_pos_w[:, _MODEL_INDICES.h_body_id, 1]
    h_z = body_pos_w[:, _MODEL_INDICES.h_body_id, 2]
    y_ref_F, z_ref_F, y_ref_H, z_ref_H = get_body_reference(env)  # [N]
    # 前肢/后肢走廊超额
    e_f = torch.abs(f_y - y_ref_F) + torch.abs(f_z - z_ref_F) + _BODY_SEG_HALF
    e_h = torch.abs(h_y - y_ref_H) + torch.abs(h_z - z_ref_H) + _BODY_SEG_HALF
    v_f = (e_f - _CORRIDOR_HALF).clamp(min=0.0)
    v_h = (e_h - _CORRIDOR_HALF).clamp(min=0.0)
    sigma = get_curriculum_reward_weight(env, "sigma_corridor")
    weight = get_curriculum_reward_weight(env, "weight_corridor")
    r_f = torch.exp(-sigma * v_f ** 2)
    r_h = torch.exp(-sigma * v_h ** 2)
    reward = (r_f + r_h) / 2
    env.extras["log"]["Data/corridor_excess"] = ((v_f + v_h) / 2).mean().item()
    return reward * weight



# =========================================================================================
# 站起奖励（连续化）— 身体竖直 (uprightness>0.8) 时, 按 F/H body 高度接近站立目标连续给奖励
def compute_stand_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    up = asset.data.projected_gravity_b[:, 2]  # [N]
    body_pos_w = asset.data.body_link_pos_w
    h = 0.5 * (body_pos_w[:, _MODEL_INDICES.f_body_id, 2] + body_pos_w[:, _MODEL_INDICES.h_body_id, 2])
    standing = up > _STAND_UP_THRESHOLD  # [N] bool 身体基本竖直
    weight = get_curriculum_reward_weight(env, "weight_stand")
    sigma = get_curriculum_reward_weight(env, "sigma_height")
    reward = torch.exp(-sigma * (h - _TARGET_HEIGHT) ** 2)  # 连续: 侧立部分奖励, 站直满奖励
    env.extras["log"]["Data/stand_success"] = ((up > 0.9) & (h > 0.05)).float().mean().item()
    return standing.float() * reward * weight



# =========================================================================================
# 站立静止奖励 — P3 内站立几何成立时, 关节速度 RMS 越接近 0 给分越高。
# 与成功判据共用同一个 `standing_metrics()` 门控与同一个速度量, 避免"奖励和判据量两个不同的数"。
# 用奖励而不是"超死区惩罚": 惩罚带死区(低于死区无梯度, 策略没理由从 1.0 降到 0),
# 而且惩罚天然给出一条作弊路线 —— 不进锥内就不被罚; 奖励形式下不进锥只是拿不到钱, 方向相反。
# 线性核而不是 exp: mimic_vel 的指数核在本任务已饱和, 在"接近静止"处几乎没有梯度。
def compute_stand_still_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    _, strict = command.stand_gate()
    weight = get_curriculum_reward_weight(env, "weight_stand_still")
    speed = torch.nan_to_num(command.standing_metrics()[2], nan=STAND_STILL_FULL_SPEED,
                             posinf=STAND_STILL_FULL_SPEED, neginf=STAND_STILL_FULL_SPEED)
    still = (1.0 - speed / STAND_STILL_FULL_SPEED).clamp(0.0, 1.0)
    return weight * (strict & (command.phase == 2)).float() * still



# =========================================================================================
# 跌倒滞留惩罚 — 检测"贴地且不动"的跌倒状态 (翻身过程贴地但在运动, 不惩罚)
def compute_fallen_penalty(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    body_pos_w = asset.data.body_link_pos_w
    h_f = body_pos_w[:, _MODEL_INDICES.f_body_id, 2]
    h_h = body_pos_w[:, _MODEL_INDICES.h_body_id, 2]
    ground = (h_f < _FALLEN_GROUND_H) & (h_h < _FALLEN_GROUND_H)  # [N] 身体贴地
    lin = asset.data.root_link_lin_vel_w[:, :2].norm(dim=1)       # [N] 水平速度
    ang = asset.data.root_link_ang_vel_w[:, 2].abs()              # [N] 偏航角速度
    moving = (lin > _FALLEN_LIN_THRESHOLD) | (ang > _FALLEN_ANG_THRESHOLD)
    fallen_idle = ground & ~moving
    weight = get_curriculum_reward_weight(env, "weight_fallen")
    penalty = -weight * fallen_idle.float()
    env.extras["log"]["Data/fallen_idle"] = fallen_idle.float().mean().item()
    return penalty



# =========================================================================================
# L1 动作平滑惩罚
def compute_action_L1_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 计算动作变化
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    abs_diff = torch.abs(current_action - prev_action)
    leg_cost = torch.sum(abs_diff[:, _MODEL_INDICES.actuator_leg_ids], dim=1)
    spn_cost = torch.sum(abs_diff[:, _MODEL_INDICES.actuator_spn_ids], dim=1)
    error_cost = torch.sum(abs_diff[:, _MODEL_INDICES.actuator_neck_ids], dim=1)
    # 获取课程学习量
    w_leg = get_curriculum_reward_weight(env, "weight_smooth_L1_leg")
    w_spn = get_curriculum_reward_weight(env, "weight_smooth_L1_spn")
    # 计算奖励
    penalty = -w_leg * leg_cost - w_spn * spn_cost - w_spn * error_cost
    return penalty



# =========================================================================================
# L2 动作平滑惩罚
def compute_action_L2_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    # 计算动作变化
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    sq_diff = torch.square(current_action - prev_action)
    leg_cost = torch.sum(sq_diff[:, _MODEL_INDICES.actuator_leg_ids], dim=1)
    spn_cost = torch.sum(sq_diff[:, _MODEL_INDICES.actuator_spn_ids], dim=1)
    error_cost = torch.sum(sq_diff[:, _MODEL_INDICES.actuator_neck_ids], dim=1)
    # 获取课程学习量
    w_leg = get_curriculum_reward_weight(env, "weight_smooth_L2_leg")
    w_spn = get_curriculum_reward_weight(env, "weight_smooth_L2_spn")
    # 计算奖励
    penalty = -w_leg * leg_cost - w_spn * spn_cost - w_spn * error_cost
    return penalty



# =========================================================================================
# 能耗惩罚
def compute_energy_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    # 计算能量消耗
    actuator_vel = asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
    actuator_torque = asset.data.actuator_force
    cost = torch.sum(torch.abs(actuator_vel * actuator_torque), dim=1)
    # 获取课程学习量
    weight = get_curriculum_reward_weight(env, "weight_energy")
    # 计算奖励
    penalty = -weight * cost
    return penalty
