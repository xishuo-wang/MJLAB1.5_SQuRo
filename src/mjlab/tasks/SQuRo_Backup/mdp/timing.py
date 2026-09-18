# 翻正参考轨迹各段时长（名义秒，执行时乘以 λ）。取值依据见 docs/SQuRo_Backup_技术细节.md §2。
P1_BUILD_DURATION = 0.65
P1_RECOVER_DURATION = 0.15
P2_DURATION = 0.15
STAND_TRANSITION_DURATION = 0.50
STAND_HOLD_DURATION = 1.05

P1_END = P1_BUILD_DURATION + P1_RECOVER_DURATION
P2_END = P1_END + P2_DURATION
STAND_TRANSITION_END = P2_END + STAND_TRANSITION_DURATION
REFERENCE_TOTAL_TIME = STAND_TRANSITION_END + STAND_HOLD_DURATION

# 阶段末端额外等待（实际秒，不乘 λ）；S1/S2 达成即推进，超时才重试。
P1_BUFFER_DURATION = 1.0
P2_BUFFER_DURATION = 0.3

# 达成时刻容许窗（实际秒，不乘 λ）：只有晚侧。窗上界同时是重试截止。
# 早侧不设窗界, 改由 EARLY_FRACTION 门控接管, 避免同一侧出现两套阈值。
# 取值依据: 手调参考在 λ=1/2/3/4 下的实测 S1 脉冲与结算时刻 (技术细节 §7.2.6);
# λ=1 是最紧的一档 (结算 1.10s vs 窗上界 0.80λ+0.50=1.30s, 余量 +0.18s)。
WINDOW_LATE_S = 0.50
# 早侧门控: 确认计时不得在 (本段名义末端 − EARLY_LEAD_FRACTION·λ) 之前起算。
# 必须用"距段末的提前量"而不是"段长的比例" —— P2 全长只有 0.15λ, 若按段长取 0.60
# 则门控(0.60λ)晚于窗界(0.15λ+0.50), P2 的可确认窗宽度会变成 0, 阶段永远无法推进。
# 取 0.20λ 时: P1 门控落在 0.60λ(与旧 EARLY_FRACTION=0.60 等价, 拦住 0.30λ 型的抄近路),
# P2 在 λ≤1.33 时门控被 clamp 到段首, 窗宽恒为 EARLY_LEAD_FRACTION·λ + 晚侧余量。
EARLY_LEAD_FRACTION = 0.20
# 里程碑时间质量核: q = exp(-(早侧偏差/σ_e)²) · exp(-(晚侧偏差/σ_l)²), 两侧各只吃自己那一半。
# 标定锚点是"手调参考的 S1 脉冲起点"(实测, 阶段内实际秒):
#   λ=1/2/3/4 的起点 = 1.00/1.69/2.41/3.18 ⇒ 相对名义 0.80λ 的偏差 = +0.20/+0.09/+0.01/−0.02。
# 前三个是"迟到", 所以早侧 σ 取 0.40λ 时手调参考全部 ≥0.93; λ=1 的 +0.20s 是手调参考自身的
# 仿射滞后(这一档物理翻正就要 ~1.0s, 名义只有 0.80s), 被晚侧核扣到 ≈0.72 —— 这是"锚在名义
# 时刻"的必然代价, 不是标定缺陷。若日后要把 λ=1 也算进验收, 应改的是手调参考的 P1 节奏。
QUALITY_SIGMA_EARLY_FRAC = 0.40
QUALITY_SIGMA_LATE_S = 0.35

# 站立成功判据（训练侧 terminations 与手调脚本共用）。设计与标定见技术细节 §7.2.1、§7.2.2。
STAND_CONFIRM_DURATION = 1.5
STAND_UPRIGHT_COS = 0.9
STAND_MIN_HEIGHT = 0.05
STAND_TARGET_HEIGHT = 0.055
STAND_GROUND_HEIGHT = 0.024
STAND_VEL_MEAN_MAX = 3.5
# 维持站立窗口的宽松几何阈值，低于它窗口清零重来。
STAND_UPRIGHT_COS_STAY = 0.8
STAND_MIN_HEIGHT_STAY = 0.045
# 站立静止奖励线性核的满值速度：vel_rms=0 得满分，达到该速度归零。
STAND_STILL_FULL_SPEED = 6.0

# 本轮训练和默认策略回放使用相同速度；原速度课程保留但不启用。
TIME_COMPARISON_SCALE = 3.0

# 现有身体轨迹文件的采集时序，不随本轮动作时间修改。
BODY_TRAJ_BUILD_END = 0.65
BODY_TRAJ_P1_END = 0.80
