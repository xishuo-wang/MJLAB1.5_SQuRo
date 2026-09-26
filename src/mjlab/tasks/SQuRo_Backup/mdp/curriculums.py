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


# 受限空间墙位课程: 两墙中心的世界 x。**墙位是原语, 中心间距是派生量**。
# 2026-09-26 重标定: 控制效果的**唯一决定量是净宽**, 与"哪一侧收"无关 —— 净宽 0.14 时把
# +X 墙收到 +0.055 而放 −X 到 −0.105 (净宽 0.14) 仍是 p_done=0.000; 净宽 0.09 时 (−0.055,
# +0.055) 与 (−0.0618, +0.0482) 都是 p_done=0.055。而**只缩 −X 侧**最多到 −0.049
# (受 reset 姿态足端 X 半宽 0.0382 限制), 净宽 0.099, p_done 仍为 0.000。
# 所以 +X 墙必须同步内收: 固定值取 +0.055 (内侧 +0.045), 课程变量留给 −X 侧。
# 净宽 0.090 是甜点 (p_stood 0.85~0.90, p_done 0.039~0.055, V/T 5.38~5.40);
# 0.085 起 p_stood 塌到 0.203, 0.080 完全站不起来。标定依据见技术细节 §7.14。
# 注意: mjwarp 在 put_model 时固化碰撞对与几何位置, 运行期都改不了 (实测),
# 所以下面的表给出的是"当前档位该用哪对墙位", 真正生效要靠按档位重建环境 (见 mdp/entity.py)。
WALL_X_POS = 0.055                 # +X 墙中心, 全程固定 (内侧 +0.045)
WALL_X_NEG_START = -0.08           # −X 墙中心起点 (= 旧课程的末档, 便于续训对齐)
WALL_X_NEG_END = -0.05             # −X 墙中心终点 = 课程下界 (净宽 0.085)
WALL_X_NEG_STEP = 0.005

# 档位表显式枚举, 课程推进的是**索引**, 不是"变化量超过阈值才重建" ——
# 后者曾因末档差值 0.0497 永不触发, 让课程静默停在 0.2497 而不是目标值。
WALL_X_NEG_LEVELS: tuple[float, ...] = tuple(
    round(WALL_X_NEG_START + i * WALL_X_NEG_STEP, 6)
    for i in range(int(round((WALL_X_NEG_END - WALL_X_NEG_START) / WALL_X_NEG_STEP)) + 1)
)
# 末档必须精确等于下界, 否则课程永远到不了目标墙位
assert abs(WALL_X_NEG_LEVELS[-1] - WALL_X_NEG_END) < 1e-9, WALL_X_NEG_LEVELS
CURRICULUM_LEVELS = len(WALL_X_NEG_LEVELS)


# 按能力推进墙位课程的参数 (门控与预算)
CURRICULUM_START_ITER = STAGE1_3_ITER   # 保留"前 3000 轮无实体碰撞"阶段, 此后才允许推进
CURRICULUM_GATE_P_STOOD = 0.60          # 回合"站起来"率门槛, 见下方口径说明
CURRICULUM_WINDOW_EPISODES = 3          # 每环境保留最近几个有效回合
CURRICULUM_MIN_DWELL_ITER = 100         # 每档最短驻留轮数 (防止一个偶然窗口连跳)
CURRICULUM_MAX_ITER = 9000              # 总预算, 与 STAGE2_2_ITER 解耦

# 门控口径: **回合"站起来"率** = 站姿维持满确认窗口且当步严格几何, **不含关节速度**。
# 为什么不用"稳定站立成功率": 速度项是采样量, 实测开墙前后完全相同 (5.638 vs 5.641 rad/s),
#   即它与墙位无关 —— 拿它当门控会让课程等一个自己影响不了的条件 (2026-09-25 run 实测
#   p_done 全程 0、课程 5100 轮没动一档)。
# 为什么不用"进入 P3": _check_S2 是 grounded | standing 的并集, 翻过来趴平同样满足,
#   那只保证"翻过去了", 不保证"站起来了"。


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


# 脊柱四关节的**误差缩放** (顺序同 actuator_spn_ids = F_spine1, F_body, H_spine1, H_body)。
# 侧摆(F_spine1)与俯仰(H_spine1)是 §2.1 里的"辅助"自由度, 窄走廊里允许它们偏离参考;
# 扭转(F/H_body)保持全额 —— 主导翻正的形变是扭转, 松掉它等于换任务。
# 缩放作用在**误差**上, 所以平方代价按 scale^2 生效 (0.5 => 容差 x2)。
# 脊柱误差一共出现在 5 处 (跟踪核 / mimic_pos / mimic_vel / spine_target / track_joint),
# 全部共用这一份。标定与副作用见技术细节 §7.14。
#
# **到 SPN_AXIS_RELAX_ITER 才切**: 前段的唯一任务是"学会按参考翻正", 而跟踪核同时是
# progress_s1 / progress_s2 / s1_shape 的乘子闸门 (§7.9), 提前放松会一并松开防套利约束,
# 让"抢在参考之前翻过去"重新变得有利。旧 run 在 iter 3600 进入原课程最窄档, 继承训练
# 从 4000 轮开始, 所以切换点与继承点对齐取 4000。
SPN_AXIS_RELAX_ITER = 4000
SPN_AXIS_SCALE_STRICT: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
SPN_AXIS_SCALE_RELAXED: tuple[float, float, float, float] = (0.5, 1.0, 0.5, 1.0)


# 当前该用的脊柱误差缩放 (按公共步计数器换算轮次, 与 get_training_phase 同口径)。
def get_spn_axis_scale(step_counter: int) -> tuple[float, float, float, float]:
    if step_counter // _STEPS_PER_ITER < SPN_AXIS_RELAX_ITER:
        return SPN_AXIS_SCALE_STRICT
    return SPN_AXIS_SCALE_RELAXED


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
    # P3 实际腿构型二次代价 (仅 P3 生效): 约束"实际摆成什么样", 补 leg_target 只管指令的缺口。
    # 起点 2.0: 实测保持段腿 MSE≈0.41 时该步值约 -0.81 (Episode_Reward 口径约 -1.6/s),
    # 与 height(4.87)、mimic_pos(9.38) 同量级但不压过它们。标定依据见技术细节 §7.13。
    # 2026-09-26 提到 6.0: run `2026-09-26_14-59-27` 在净宽 0.085 里学会"卡在墙之间"通过判据,
    # 保持段腿 RMSE 从 0.11 劣化到 0.51 rad。当时 P3 内姿态类代价合计仅 -0.39/s, 而成功是
    # +5.1/s, 相差 13 倍 —— 姿态项没有议价能力。本批先只动这一项 (单变量), 不动判据。
    "weight_leg_pose":          (6.0,),
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
    # S1 姿态塑形 (只作用 P1): 把窄位形"前段仰面+后段俯卧+两段贴地"变成连续地形, 消除
    # S1 只靠随机命中才能发现的瓶颈。S1 处峰值 = 本权重, 与 progress_s1(最大 3.0) 同量级;
    # **不要**调到与里程碑(10.0/step_dt 量级)可比, 否则策略会为了塑形分而拖延 P1。见技术细节 §7.12。
    "weight_s1_shape":          (2.0,),
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


# 受限空间阶段: 0 = STAGE1 (不开实体碰撞), 1 = STAGE2 (开碰撞)。
# 墙位课程不再与轮次绑定 (改为按能力推进), 这里只剩"碰撞开关"这一个时钟量。
def get_training_phase(step_counter: int) -> int:
    return 0 if step_counter // _STEPS_PER_ITER < STAGE1_3_ITER else 1


# 该档位该编译的墙位对 (m)。训练与回放必须共用这一个口径。
# 越界一律 clamp 到端点: 课程停在末档时仍要能算出墙位。
def get_wall_positions_for_level(level: int) -> tuple[float, float]:
    idx = min(max(int(level), 0), CURRICULUM_LEVELS - 1)
    return WALL_X_NEG_LEVELS[idx], WALL_X_POS


# 由墙位反查最近的档位索引 (续训时检查点只记了墙位、没记档位时用)
def get_level_for_wall_x_neg(x_neg: float) -> int:
    return min(range(CURRICULUM_LEVELS),
               key=lambda i: abs(WALL_X_NEG_LEVELS[i] - float(x_neg)))
