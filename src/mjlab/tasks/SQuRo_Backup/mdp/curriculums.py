from __future__ import annotations
import torch
from typing import Any


# 100 Hz 初筛配置: 控制周期 0.002 × 5 = 0.01 s, 96 步覆盖 0.96 s。取舍理由见技术细节 §6。
_STEPS_PER_ITER = 96


# 训练阶段边界 (iter)
STAGE1_MID_ITER = 1000             # iter 0-1000:   纯模仿阶段
STAGE2_MID_ITER = 2000             # iter 1000-2000: 强化竖直/高度引导
STAGE2_END_ITER = 4000             # iter 2000-4000: 收敛/泛化 (预留走廊/命令)


# 阶段边界表 (RewardWeightCurriculum 按 iter 取段)
_STAGES = (0, 2000, 4000)


# 命令课程: time_scale λ (放慢倍数) 采样区间
TIME_SCALE_MAX = 4.0
TIME_SCALE_MIN_START = 2.0
TIME_SCALE_MIN_END = 1.0


# 奖励权重课程曲线 — 每阶段一个值, 值数量不足时取末值。
# 权重唯一管理处: env_cfg 中所有奖励项 cfg.weight 一律为 1.0, 生效权重 = 1.0 × 本表的值。
# 本表同时容纳 weight_*(权重) / sigma_*(误差系数) / alpha_*(复合项配比) 三类量。
_CURVES: dict[str, tuple[float, ...]] = {
    "weight_mimic_pos":         (10.0,),
    "weight_mimic_vel":         (5.0,),
    "weight_spine_target":      (2.0,),
    # P3 腿部目标跟踪代价(仅 P3 生效): 比较限幅前目标与腿部参考, 压制"腿指令长期超
    # ctrlrange"的饱和行为。权重从 1.0 起调, 标定依据见技术细节 2026-09-19 一节。
    "weight_leg_target":        (1.0,),
    # 躯干姿态模仿 (技术细节 §7.8): 跟踪两段背腹轴的世界 Z 余弦。
    # 从 1.0 起调; 实测新 run 相对参考的姿态 MSE = 0.1208 (P1 段 0.2536),
    # 故 w=1 时该项平均约 -0.12/步, 与 track_joint(-0.003) 同量级、远小于 mimic_pos(+12)。
    # 标定权重前必须同时看"扣分中来自整机刚体旋转的比例"(该项部分不可达)。
    "weight_body_att":          (1.0,),
    # 加权二次关节跟踪代价: 补上 mimic_pos 的 exp 核在高误差区(MSE≥0.25)梯度归零的缺口,
    # 让"绕过参考表"持续按误差付钱。权重从 1.0 起调, 标定依据见技术细节开头 2026-09-18 一节。
    "weight_track_joint":       (1.0,),
    "weight_height":            (5.0,),
    "weight_milestone_s1":      (10.0,),
    "weight_milestone_s2":      (15.0,),
    "weight_milestone_success": (35.0,),
    # progress_s1 推后段、progress_s2 在此之上承担翻正主推力。两项都必须够大,
    # 否则策略会收敛到"撑起来原地扭"的不翻正解 (标定依据见技术细节 §2)。
    "weight_progress_s1":       (3.0,),
    "weight_progress_s2":       (3.0,),
    "weight_progress_s3":       (3.0,),
    # stand_still 只在 P3 站立几何成立时给分(线性核, vel_rms=0 满分, 6 rad/s 归零);
    # 权重 3.0 与"早到红利"的对冲标定见技术细节 §7.2.4。
    "weight_stand_still":       (3.0,),
    "weight_action_excess":     (0.5,),
    "weight_smooth_L1_leg":     (0.1,),
    "weight_smooth_L1_spn":     (0.1,),
    "weight_smooth_L2_leg":     (0.1,),
    "weight_smooth_L2_spn":     (0.1,),
    "weight_energy":            (0.1,),
    "sigma_leg_pos":      (10.0,),
    "sigma_spn_pos":      (20.0,),
    "sigma_neck_pos":     (10.0,),
    "sigma_leg_vel":      (0.5,),
    "sigma_spn_vel":      (0.5,),
    "sigma_neck_vel":     (0.5,),
    "sigma_height":       (500.0,),
    "alpha_neck_pos":     (0.3,),
    "alpha_neck_vel":     (0.3,),
}


# 奖励权重课程: 按当前训练 iter 返回各奖励项权重/σ
class RewardWeightCurriculum:
    def get_reward_weights(self, current_step: int) -> dict[str, float]:
        current_iter = current_step // _STEPS_PER_ITER
        result: dict[str, float] = {}
        for name, values in _CURVES.items():
            idx = 0
            for i, t in enumerate(_STAGES):
                if current_iter >= t:
                    idx = i
            safe_idx = idx if idx < len(values) else len(values) - 1
            result[name] = values[safe_idx]
        return result

    def get_current_stage_info(self, current_step: int) -> dict[str, Any]:
        current_iter = current_step // _STEPS_PER_ITER
        stage = 0
        for t in _STAGES:
            if current_iter >= t:
                stage = t
        return {
            "current_stage": stage,
            "current_iter": current_iter,
            "current_step": current_step,
            "reward_weights": self.get_reward_weights(current_step),
        }


reward_weight_curriculum = RewardWeightCurriculum()


# 获取课程奖励权重
def get_curriculum_reward_weight(env, reward_name: str) -> float:
    return reward_weight_curriculum.get_reward_weights(env.common_step_counter).get(reward_name, 1.0)


# 采样命令 time_scale λ (episode 内固定): 返回 [n] 张量
def get_curriculum_time_scale(step_counter: int, n: int, device: str) -> torch.Tensor:
    iter_num = step_counter // _STEPS_PER_ITER
    progress = min(1.0, max(0.0, (iter_num - STAGE1_MID_ITER) / (STAGE2_MID_ITER - STAGE1_MID_ITER)))
    lam_min = TIME_SCALE_MIN_START - progress * (TIME_SCALE_MIN_START - TIME_SCALE_MIN_END)
    return lam_min + torch.rand(n, device=device) * (TIME_SCALE_MAX - lam_min)
