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
# 早侧门控已删除: 它被"参考播完门 t_phase ≥ 段末"完全吞噬 —— 达成时刻恒为
# max(真实到达时刻, 段末), 所以 0.60λ 的地板从来没有起过作用。可接受区间就写成
# [段末, 段末 + WINDOW_LATE_S] 一个常量即可。要真正拦"提前到达并保持", 只能靠
# 里程碑时间质量核(按真实到达时刻打折), 见下面的 QUALITY_SIGMA_*。
# 里程碑时间质量核: q = exp(-(早侧偏差/σ_e)²) · exp(-(晚侧偏差/σ_l)²), 两侧各只吃自己那一半。
# 输入必须是"真实首次到达时刻"(command._s1_criterion_first), 不能用受门控/播完门夹住的
# 确认起算时刻 —— 后者恒等于段末, 会让核恒等于 1.0(实测确认过), 等于没有。
# 标定锚点仍是手调参考的 S1 脉冲起点 = 1.00/1.69/2.41/3.18s (λ=1/2/3/4)。
QUALITY_SIGMA_EARLY_FRAC = 0.40
QUALITY_SIGMA_LATE_S = 0.35

# 加权二次关节跟踪代价 (新增独立奖励项 weight_track_joint)。
# 为什么必须有: 现有 mimic_pos 的 exp(-σ·MSE) 核在 MSE≥0.25 rad² 后输出与梯度双双恒为 0,
# 而"绕过参考表"恰好生活在该区间 —— 实测脊柱误差从 1.57 rad 劣化到 3.46 rad, mimic_pos
# 只从 3.07 掉到 3.03(差 0.04), 换不来"拧满脊柱"的动机。二次核在整个区间都有梯度。
# 分组权重沿用现有 σ 比(脊柱 20 : 腿 10 : 颈 10, 颈再乘 0.3)的意图, 不改 mimic_pos 本身。
TRACK_W_SPN = 1.57
TRACK_W_LEG = 1.0
TRACK_W_NECK = 0.3
# 归一化尺度: 取"脊柱全错满量程"的参考值 1.57²≈2.46, 圆整到 3.0, 使该项在典型误差下
# 与其他密集项同量级; 权重入口 weight_track_joint 从 1.0 起调。
TRACK_REF_MSE_SCALE = 3.0

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
