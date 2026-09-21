from __future__ import annotations
import torch
from mjlab.entity import Entity
from typing import TYPE_CHECKING, cast
from .command import BackupCommand
from .curriculums import get_curriculum_reward_weight
from .reference import get_reference_joint_state, get_body_reference, get_reference_body_attitude
from .indices import _ACTUATED_JOINT_NAMES, _ACTUATOR_CTRL_RANGE, _MODEL_INDICES


if TYPE_CHECKING:
    from mjlab.envs.mdp.actions import JointPositionAction
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv



# 只在本文件使用的奖励核参数 (未跨文件, 故不进 config.py)。
QUALITY_SIGMA_EARLY_FRAC = 0.40   # 里程碑时间质量核的早侧 σ (按 λ 缩放)
QUALITY_SIGMA_LATE_S = 0.35       # 里程碑时间质量核的晚侧 σ (绝对秒)
TRACK_W_SPN = 1.57                # 加权二次跟踪代价的脊柱分组权重
TRACK_W_LEG = 1.0                 # 同上, 腿部分组
TRACK_W_NECK = 0.3                # 同上, 颈部
TRACK_REF_MSE_SCALE = 3.0         # 二次跟踪代价的 MSE 归一化尺度
STAND_STILL_FULL_SPEED = 4.5      # 站立静止奖励线性核的归零速度



_STAND_UP_THRESHOLD = 0.8     # 站起奖励: 竖直度下限 (身体基本竖直才给站直奖励)
_TARGET_HEIGHT = 0.055        # 站直目标高度 (m, 与命令 height_f/h 一致)
# 跌倒滞留惩罚阈值
_FALLEN_GROUND_H = 0.03       # F/H body 贴地高度阈值 (m, 贴地≈0.024)
_FALLEN_LIN_THRESHOLD = 0.05  # 贴地时水平线速度低于此值视为"不动" (m/s)
_FALLEN_ANG_THRESHOLD = 0.5   # 贴地时角速度低于此值视为"不动" (rad/s)
# 走廊 (YoZ 平面) 参数
_CORRIDOR_HALF = 0.05         # 走廊半宽/死区 (m), 前后肢共用
_BODY_SEG_HALF = 0.025        # 身体段半径 (m, YoZ 截面包络)



# 里程碑时间质量核: 达成时刻越接近名义段末, 钱越接近满分。
# 早侧按 λ 比例(陡)、晚侧按绝对秒(缓)的理由见技术细节 2026-09-18 一节。
# 关键: dev 取自"导致结算的那次确认脉冲的起点", 不是结算时刻 —— 结算时刻被段末门控压在
# 名义时刻之后, 拿它做核会恒等于满分, 对"提前翻完干等"完全无效。
def _milestone_time_quality(dev_early: torch.Tensor, dev_late: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    # 早侧只吃"提前"那一段、晚侧只吃"迟到"那一段: 否则晚到会被早侧的窄 σ 一并罚掉,
    # 手调参考在 λ=4 的正偏差 (+0.68s) 就会被误杀。传入的 dev 是同一个原值。
    valid = torch.isfinite(dev_early) & torch.isfinite(dev_late)
    e = torch.nan_to_num(dev_early, nan=0.0)
    l = torch.nan_to_num(dev_late, nan=0.0)
    late_positive = l.clamp(min=0.0)
    early_negative = (-e).clamp(min=0.0)
    sigma_early = (QUALITY_SIGMA_EARLY_FRAC * lam).clamp_min(1e-6)
    q_early = torch.exp(-(early_negative / sigma_early) ** 2)
    q_late = torch.exp(-(late_positive / QUALITY_SIGMA_LATE_S) ** 2)
    # 无有效到达记录时必须给 0 而不是 1: 发脉冲说明该循环确实结算了, 此时记录缺失是
    # "重试后没重新锁存"这类内部不一致, 静默按"零偏差"发满额奖励会变成一条绕过路径。
    return torch.where(valid, q_early * q_late, torch.zeros_like(q_early))



# s1里程碑奖励
def compute_s1_milestone_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    pulse = command.s1_milestone_pulse
    weight = get_curriculum_reward_weight(env, "weight_milestone_s1")
    quality = _milestone_time_quality(command.s1_dev_early, command.s1_dev_late, command.time_scale_command)
    return weight * pulse.float() * quality / env.step_dt



# s2里程碑奖励
def compute_s2_milestone_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    pulse = command.s2_milestone_pulse
    weight = get_curriculum_reward_weight(env, "weight_milestone_s2")
    quality = _milestone_time_quality(command.s2_dev_early, command.s2_dev_late, command.time_scale_command)
    return weight * pulse.float() * quality / env.step_dt



def compute_task_success_milestone_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    # 循环完成改为非终止事件: 本项既负责推进站立窗口, 又是完成脉冲的唯一产生点。
    # 时序理由(为什么必须在奖励项里判定)见 command.stand_reward_and_pulse 的注释。
    command.stand_reward_and_pulse()
    pulse = command.consume_cycle_end_pulse()
    # 曾在此上报 Progress/success, 但它与 command 的 Cycle/completed 是同一个脉冲, 纯冗余, 已删。
    weight = get_curriculum_reward_weight(env, "weight_milestone_success")
    return weight * pulse.float() / env.step_dt



# 脊柱跟踪核 (mimic_pos 的脊柱分量同源), 用作姿态进度项的乘法门控。
# 依据: 实测脊柱跟踪奖励在 P1 内只值 0.55/步, 而 progress_s1 值 2.50/步 —— 推动后段翻正的
# 主力是 progress_s1, 但它对"抢在参考之前翻过去"也一视同仁给分, 于是形成抄近路收益。
# 乘上本核后: 跟住参考时给全额(2.67~2.93), 抢跑时衰减到 0.09~0.19。详见技术细节 §7.9。
def _spine_track_kernel(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    # joint_pos 是**全部**关节(36), 而参考表只有 14 个驱动关节: 先用 joint_ids 取到驱动关节,
    # 再按参考表列下标(0/1/8/9)取脊柱列。两个口径不可互换, 缺一都会维度不匹配。
    joint_pos = asset.data.joint_pos[:, _MODEL_INDICES.joint_ids]
    ref_pos, _ = get_reference_joint_state(env)
    error_spn = (joint_pos - ref_pos)[:, _MODEL_INDICES.actuator_spn_ids]
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_pos")
    return torch.exp(-sigma_spn * mse_spn)


# s1区间奖励 — 朝 S1 姿态 (前段仰面 + 后段俯卧) 的连续进度。
# 乘以脊柱跟踪核: 保留"后段必须立起来"的连续激励(不能删, 否则回到"不翻正"局部最优),
# 但只在跟住参考时才给全额, 消除"抢在参考之前翻过去"的收益。
def compute_s1_progress_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    weight = get_curriculum_reward_weight(env, "weight_progress_s1")
    return weight * command.progress_s1 * _spine_track_kernel(env)



# s2区间奖励 — 朝 S2 姿态 (两段都已俯卧) 的连续进度。
# 双重约束: P1 内关闭(phase>=1 才生效, 否则"提前双正置"最优), 且乘脊柱跟踪核(同 s1)。
# 两者理由与实测账见技术细节 §7.9。
def compute_s2_progress_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    weight = get_curriculum_reward_weight(env, "weight_progress_s2")
    return weight * command.progress_s2 * (command.phase >= 1) * _spine_track_kernel(env)


# P3 站立进度: 只对 P3 生效；前后段同时背部朝上才解锁, 越接近最终站立越大。
def compute_s3_progress_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    _, progress = command.standing_state()
    weight = get_curriculum_reward_weight(env, "weight_progress_s3")
    return weight * progress * (command.phase == 2)



# 关节位置模仿奖励
def compute_mimic_pos_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    joint_pos = asset.data.joint_pos[:, _MODEL_INDICES.joint_ids]
    ref_pos, _ = get_reference_joint_state(env)
    error = joint_pos - ref_pos
    error_leg = error[:, _MODEL_INDICES.actuator_leg_ids]
    error_spn = error[:, _MODEL_INDICES.actuator_spn_ids]
    error_neck = error[:, _MODEL_INDICES.actuator_neck_ids]
    weight = get_curriculum_reward_weight(env, "weight_mimic_pos")
    sigma_leg = get_curriculum_reward_weight(env, "sigma_leg_pos")
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_pos")
    sigma_neck = get_curriculum_reward_weight(env, "sigma_neck_pos")
    alpha_neck = get_curriculum_reward_weight(env, "alpha_neck_pos")
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    mse_neck = torch.mean(error_neck ** 2, dim=1)
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward_neck = torch.exp(-sigma_neck * mse_neck)
    reward = (reward_leg + reward_spn) / 2 + alpha_neck * reward_neck
    return reward * weight



# 按名称对齐关节目标成本；raw_action 未经任何外层裁剪 (rl_cfg.clip_actions=None)，也未经 XML 控制限幅。
# 2026-09-19 校正: 此处原注释称"已过 RL 外层 ±6 裁剪", 但 clip_actions=None 使其不成立 ——
# 实测脊柱动作可达 ±5.4, 腿部 ±65, 全部直接来自策略输出。
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



# P3 腿部目标跟踪代价 — 只在 P3 生效的前馈项。
# 为什么需要它(技术细节 2026-09-19): 起立前后腿的实际角约 (-1.500, -0.500), 明显偏离
# 站立目标 (-0.1, 0.3), 且后腿目标指令**持续超出执行器 ctrlrange**(被 MuJoCo 丢弃),
# 故 `action_excess` 单调恶化 15 倍。而现有各项都区分不出这种状态:
#   - mimic_pos 的腿部分量在趴地时只贡献约 0.0009/s, 位置模仿总奖励 7.90/s 几乎全由
#     脊柱与颈部拿到;
#   - track_joint 的二次代价全关节合计也只有约 -0.165/s。
# 与 spine_target 同构: 比较**限幅前**的目标角与参考角, 因此能量化"都被执行器裁到同一
# 位置、但偏离参考程度不同"的指令, 给出"别把腿指令打到饱和"的直接梯度。
def compute_leg_target_cost(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    weight = get_curriculum_reward_weight(env, "weight_leg_target")
    out = weight * _joint_target_cost(env, _MODEL_INDICES.actuator_leg_ids)
    return out * (command.phase == 2)



# 躯干姿态模仿代价 — 跟踪两段背腹轴的世界 Z 余弦 (技术细节 §7.8)。
#
# 为什么关节模仿之外还要它: 实测"关节角推不出整体朝向" —— corr(姿态余弦, 对应 body 角)
# 仅 +0.58, 用 body 角线性拟合姿态余弦的残差 RMS 达 0.48~0.54 (姿态余弦总范围 2.00);
# 且存在"关节几乎不动、姿态余弦却大幅变化"的帧(整机刚体旋转), 关节参考里没有这个信息。
# 该项与关节项互补: 关节项管段内构型, 姿态项管整体朝向。
#
# 局限(必须知道): 整机刚体旋转时关节无法改变姿态, 此时该项部分不可达 —— 标定权重时
# 要同时看"该项扣分中来自刚体旋转的比例", 不能只按 MSE 大小加权重。
def compute_body_attitude_cost(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    ref_u = get_reference_body_attitude(env)
    # 与门控同源的姿态余弦; 退化向量会返回 NaN, 用 0 兜底避免污染奖励和。
    actual_u = command._pose_cos()
    valid = torch.isfinite(actual_u)
    err = torch.where(valid, actual_u - ref_u, torch.zeros_like(ref_u))
    weight = get_curriculum_reward_weight(env, "weight_body_att")
    return -weight * err.square().mean(dim=1)



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



# 动作超出执行器 ctrlrange 的成本: 只对会被 MuJoCo 裁掉的那一段计费。
# 背景(策略被钉在饱和点、腿失去控制权)见技术细节 §2。
def compute_action_ctrl_excess_penalty(env: "ManagerBasedRlEnv") -> torch.Tensor:
    action_term = cast("JointPositionAction", env.action_manager.get_term("joint_pos"))
    # 与 spine_target 同源: 用裁剪后、XML 控制限幅前的目标, 含 scale 与默认角偏移。
    target = action_term.raw_action * action_term.scale + action_term.offset
    lo, hi = _ctrl_range_tensors(action_term.target_names, target.device, target.dtype)
    excess = (lo - target).clamp(min=0.0) + (target - hi).clamp(min=0.0)
    weight = get_curriculum_reward_weight(env, "weight_action_excess")
    return -weight * excess.mean(dim=1)



# 加权二次关节跟踪代价 — 复用 mimic_pos 的分组与关节索引, 但用二次核。
# 与 mimic_pos 的分工: 后者是"跟得准不准"的细粒度形状项, 前者负责在高误差区still有梯度,
# 让"绕过参考表"持续付出与误差成比例的代价。分组权重只在这里生效, 不改 mimic_pos。
_JOINT_W: dict = {}
# 分组按名字判定, 不依赖任何列序 (F_spine1/F_body 在 0-1, H_spine1/H_body 在 8-9)。
_SPN_NAMES = {_ACTUATED_JOINT_NAMES[i] for i in (0, 1, 8, 9)}
_NECK_NAMES = {_ACTUATED_JOINT_NAMES[i] for i in (2, 3)}


def _joint_group_weights(device, dtype) -> torch.Tensor:
    # 权重向量按"参考表顺序"(= _ACTUATED_JOINT_NAMES 顺序) 构造, 与 joint_pos[:, joint_ids]
    # 和 ref_pos 完全同序。不要按动作项名称顺序构造 —— 那会让三者顺序不一致,
    # 一旦动作项列序与参考表不同(名称反序), 代价会从 0 变成非零(已复现 0.23897)。
    key = (str(device), str(dtype))
    w = _JOINT_W.get(key)
    if w is None:
        w = torch.tensor([TRACK_W_SPN if n in _SPN_NAMES
                          else TRACK_W_NECK if n in _NECK_NAMES
                          else TRACK_W_LEG for n in _ACTUATED_JOINT_NAMES],
                         device=device, dtype=dtype)
        _JOINT_W[key] = w
    return w


def compute_joint_track_cost(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    # 实际关节、参考、权重三者统一用参考表顺序; joint_ids 由 find_joints(preserve_order=True)
    # 解析, 顺序即 _ACTUATED_JOINT_NAMES, 所以无需再按动作项名称重排。
    joint_pos = asset.data.joint_pos[:, _MODEL_INDICES.joint_ids]
    ref_pos, _ = get_reference_joint_state(env)
    err = joint_pos - ref_pos
    w = _joint_group_weights(err.device, err.dtype)
    cost = (w * err.square()).mean(dim=1) / TRACK_REF_MSE_SCALE
    weight = get_curriculum_reward_weight(env, "weight_track_joint")
    # 曾在此上报 Data/track_joint_cost, 但它就是 Episode_Reward/track_joint(同值乘权重), 已删。
    return -weight * cost



# 关节速度模仿奖励
def compute_mimic_vel_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    joint_vel = asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
    _, ref_vel = get_reference_joint_state(env)
    error = joint_vel - ref_vel
    error_leg = error[:, _MODEL_INDICES.actuator_leg_ids]
    error_spn = error[:, _MODEL_INDICES.actuator_spn_ids]
    error_neck = error[:, _MODEL_INDICES.actuator_neck_ids]
    weight = get_curriculum_reward_weight(env, "weight_mimic_vel")
    sigma_leg = get_curriculum_reward_weight(env, "sigma_leg_vel")
    sigma_spn = get_curriculum_reward_weight(env, "sigma_spn_vel")
    sigma_neck = get_curriculum_reward_weight(env, "sigma_neck_vel")
    alpha_neck = get_curriculum_reward_weight(env, "alpha_neck_vel")
    mse_leg = torch.mean(error_leg ** 2, dim=1)
    mse_spn = torch.mean(error_spn ** 2, dim=1)
    mse_neck = torch.mean(error_neck ** 2, dim=1)
    reward_leg = torch.exp(-sigma_leg * mse_leg)
    reward_spn = torch.exp(-sigma_spn * mse_spn)
    reward_neck = torch.exp(-sigma_neck * mse_neck)
    reward = (reward_leg + reward_spn) / 2 + alpha_neck * reward_neck
    return reward * weight



# 身体竖直奖励
def compute_upright_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    uprightness = asset.data.projected_gravity_b[:, 2]  # [N]
    sigma = get_curriculum_reward_weight(env, "sigma_upright")
    weight = get_curriculum_reward_weight(env, "weight_upright")
    reward = torch.exp(-sigma * (1.0 - uprightness) ** 2)
    env.extras["log"]["Data/uprightness"] = uprightness.mean().item()
    return reward * weight



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
    # 曾在此上报 Body/height (两段平均高度), 与 Progress/standing + stand_hold 重复, 已删。
    return Reward_height


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



# 站立静止奖励 — P3 内站立几何成立时, 关节速度 RMS 越接近 0 给分越高(线性核)。
# 与成功判据共用同一个门控与速度量; 用奖励而非超死区惩罚、用线性核而非 exp 的理由见技术细节 §7.2.4。
def compute_stand_still_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = cast("BackupCommand", env.command_manager.get_term("backup_cmd"))
    _, strict = command.stand_gate()
    weight = get_curriculum_reward_weight(env, "weight_stand_still")
    speed = torch.nan_to_num(command.standing_metrics()[2], nan=STAND_STILL_FULL_SPEED,
                             posinf=STAND_STILL_FULL_SPEED, neginf=STAND_STILL_FULL_SPEED)
    still = (1.0 - speed / STAND_STILL_FULL_SPEED).clamp(0.0, 1.0)
    return weight * (strict & (command.phase == 2)).float() * still



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



# L1 动作平滑惩罚
def compute_action_L1_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    abs_diff = torch.abs(current_action - prev_action)
    leg_cost = torch.sum(abs_diff[:, _MODEL_INDICES.actuator_leg_ids], dim=1)
    spn_cost = torch.sum(abs_diff[:, _MODEL_INDICES.actuator_spn_ids], dim=1)
    error_cost = torch.sum(abs_diff[:, _MODEL_INDICES.actuator_neck_ids], dim=1)
    w_leg = get_curriculum_reward_weight(env, "weight_smooth_L1_leg")
    w_spn = get_curriculum_reward_weight(env, "weight_smooth_L1_spn")
    penalty = -w_leg * leg_cost - w_spn * spn_cost - w_spn * error_cost
    return penalty



# L2 动作平滑惩罚
def compute_action_L2_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    sq_diff = torch.square(current_action - prev_action)
    leg_cost = torch.sum(sq_diff[:, _MODEL_INDICES.actuator_leg_ids], dim=1)
    spn_cost = torch.sum(sq_diff[:, _MODEL_INDICES.actuator_spn_ids], dim=1)
    error_cost = torch.sum(sq_diff[:, _MODEL_INDICES.actuator_neck_ids], dim=1)
    w_leg = get_curriculum_reward_weight(env, "weight_smooth_L2_leg")
    w_spn = get_curriculum_reward_weight(env, "weight_smooth_L2_spn")
    penalty = -w_leg * leg_cost - w_spn * spn_cost - w_spn * error_cost
    return penalty



# 能耗惩罚
def compute_energy_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    actuator_vel = asset.data.joint_vel[:, _MODEL_INDICES.joint_ids]
    actuator_torque = asset.data.actuator_force
    cost = torch.sum(torch.abs(actuator_vel * actuator_torque), dim=1)
    weight = get_curriculum_reward_weight(env, "weight_energy")
    penalty = -weight * cost
    return penalty
