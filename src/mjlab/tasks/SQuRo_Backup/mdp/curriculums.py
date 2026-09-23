from __future__ import annotations
import torch
from typing import Any

# 本文件只留课程学习相关内容 (轮数换算、时间缩放课程、奖励权重曲线);
# 其余任务配置常量在 config.py。


# 100 Hz 初筛配置: 控制周期 0.002 × 5 = 0.01 s, 96 步覆盖 0.96 s。取舍理由见技术细节 §6。
_STEPS_PER_ITER = 96


# 训练阶段边界 (iter) — 全任务唯一的阶段划分表, 课程与阶段判断都引用这里。
# 整体两阶段: STAGE1_* = 无受限空间阶段, STAGE2_* = 受限空间阶段。
# 命名规则: STAGE<a>_<b>_ITER = 第 a 阶段第 b 小段的**结束轮次**。
STAGE1_1_ITER = 1000               # iter    0-1000: 纯模仿
STAGE1_2_ITER = 2000               # iter 1000-2000: 时间缩放课程 (λ 下限 2.0 -> 1.0)
STAGE1_3_ITER = 3000               # iter 2000-3000: 动作优化 (平滑权重第 2/3 档)
STAGE2_1_ITER = 5000               # iter 3000-5000: 受限空间课程 (开碰撞, a 0.40 -> 0.20)
STAGE2_2_ITER = 6000               # iter 5000-6000: 受限空间动作优化 (= 总训练轮数)


# 受限空间课程: 两侧墙中心间距 a (m)。实际内侧净宽 = a - 2×墙半厚。
# 注意: mjwarp 在 put_model 时固化碰撞对与几何位置, 运行期都改不了 (实测),
# 所以下面的函数给出的是"当前轮次该用哪个 a", 真正生效要靠按 a 重建环境 (见 mdp/entity.py)。
CORRIDOR_WIDTH_START = 0.40        # STAGE1 期间固定值 = 受限空间课程起点
CORRIDOR_WIDTH_MIN = 0.20          # 课程终点 = STAGE2_2 保持值


# 奖励权重课程的分档边界 (RewardWeightCurriculum 按 iter 取段)。
# 与上面的阶段表保持"前三个边界一致": 0 ~ STAGE1_1_ITER ~ STAGE1_2_ITER。
_STAGES = (0, STAGE1_1_ITER, STAGE1_2_ITER)


# 命令课程: time_scale λ (放慢倍数) 采样区间。λ ~ U[lam_min, TIME_SCALE_MAX],
# lam_min 在 STAGE1_2 (iter 1000~2000) 内由 TIME_SCALE_MIN_START 线性降到 TIME_SCALE_MIN_END。
# 上界 4.0 -> 3.0: 一个循环约 2.37λ + 站立窗口(1.0~1.5s), λ=4 需 ~10.5s, 逼近 episode
# 上限 12s, 且远超价值视野 (~2.9s @ γ=0.99^0.5), 稀疏里程碑项在长 λ 样本上几乎学不到;
# 收紧上界让全部样本都落在"一个回合至少装得下一次完整循环"的区间内。
TIME_SCALE_MAX = 3.0
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
    # 注意: 该权重只在 phase>=1 (P2 及以后) 生效, P1 内被门控关掉。
    # 理由见 rewards.compute_s2_progress_reward 的注释: P1 内它给"提前双正置"发高分,
    # 使抄近路的总收益 (4.99) 高于正确的 S1 姿态 (4.50)。P1 内关掉后地形翻转为 S1 占优。
    "weight_progress_s2":       (3.0,),
    "weight_progress_s3":       (3.0,),
    # stand_still 只在 P3 站立窗口成立时给分(线性核, vel_rms=0 满分, 核的归零速度
    # = config.STAND_VEL_MEAN_MAX); 权重 3.0 与"早到红利"的对冲标定见技术细节 §7.2.4。
    "weight_stand_still":       (3.0,),
    "weight_action_excess":     (0.5,),
    "weight_smooth_L1_leg":     (0.1, 0.2, 0.4),
    "weight_smooth_L1_spn":     (0.2, 0.2, 0.4),
    "weight_smooth_L2_leg":     (0.1, 0.2, 0.4),
    "weight_smooth_L2_spn":     (0.2, 0.2, 0.4),
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


# 采样命令 time_scale λ (episode 内固定): 返回 [n] 张量。
# λ 下限在 STAGE1_2 (iter STAGE1_1_ITER ~ STAGE1_2_ITER) 内线性收紧。
def get_curriculum_time_scale(step_counter: int, n: int, device: str) -> torch.Tensor:
    iter_num = step_counter // _STEPS_PER_ITER
    span = max(1, STAGE1_2_ITER - STAGE1_1_ITER)
    progress = min(1.0, max(0.0, (iter_num - STAGE1_1_ITER) / span))
    lam_min = TIME_SCALE_MIN_START - progress * (TIME_SCALE_MIN_START - TIME_SCALE_MIN_END)
    return lam_min + torch.rand(n, device=device) * (TIME_SCALE_MAX - lam_min)


# 受限空间阶段: 0 = STAGE1 (不开碰撞, a 固定), 1 = STAGE2 (开碰撞, a 按课程收紧)
def get_training_phase(step_counter: int) -> int:
    return 0 if step_counter // _STEPS_PER_ITER < STAGE1_3_ITER else 1


# 受限空间课程: 两侧墙中心间距 a (m) 关于轮次的**连续**取值:
#   STAGE1  (iter 0 ~ STAGE1_3_ITER)               : a = 0.40 固定, 不开碰撞
#   STAGE2_1(iter STAGE1_3_ITER ~ STAGE2_1_ITER)   : a 从 0.40 线性收到 0.20, 开碰撞
#   STAGE2_2(iter STAGE2_1_ITER ~ STAGE2_2_ITER)   : a = 0.20 保持
# 注意: 几何只能在编译期定, 实际生效的宽度是下面 CORRIDOR_WIDTH_LADDER 里的离散档位,
#       **不要**用本函数去推算"训练当时真正的墙宽", 那要用 get_corridor_width_for_iter。
def get_curriculum_corridor_width(step_counter: int) -> float:
    iter_num = step_counter // _STEPS_PER_ITER
    if iter_num < STAGE1_3_ITER:
        return CORRIDOR_WIDTH_START
    span = max(1, STAGE2_1_ITER - STAGE1_3_ITER)
    progress = min(1.0, (iter_num - STAGE1_3_ITER) / span)
    return CORRIDOR_WIDTH_START - progress * (CORRIDOR_WIDTH_START - CORRIDOR_WIDTH_MIN)


# 实际生效的宽度档位 (m)。墙体几何只能在编译期定, 所以课程被量化成这几档;
# 末档**必须**恰好等于 CORRIDOR_WIDTH_MIN, 否则课程永远到不了目标宽度。
CORRIDOR_WIDTH_LEVELS = 5
CORRIDOR_WIDTH_LADDER: tuple[float, ...] = tuple(
    CORRIDOR_WIDTH_MIN
    + (CORRIDOR_WIDTH_START - CORRIDOR_WIDTH_MIN) * (CORRIDOR_WIDTH_LEVELS - 1 - i)
    / (CORRIDOR_WIDTH_LEVELS - 1)
    for i in range(CORRIDOR_WIDTH_LEVELS)
)


# 该轮次实际编译生效的墙间距 (从档位表里选最接近的), 训练与回放必须共用这一个口径。
def get_corridor_width_for_iter(iter_num: int) -> float:
    step_counter = iter_num * _STEPS_PER_ITER
    wanted = get_curriculum_corridor_width(step_counter)
    return min(CORRIDOR_WIDTH_LADDER, key=lambda w: abs(w - wanted))
