# 翻正参考轨迹各段时长（名义秒，执行时乘以 λ）。
# T2 回收段取 0.15s —— 手调实测的最小可行值；放大到 0.65s 等效于对该段单独做时间缩放。
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

# 站起成功所需的连续达标时长（实际秒，不乘 λ）。
# 训练侧 terminations.check_stand_success 与手调脚本 StateMachinePolicy 共用此值。
# 0.5 -> 1.5 s: 原判据只要求"最后 0.5 秒姿态成立", 机器人可以临近末端才站上;
# 加长后必须真正维持住, 同时给"站住"留出可观测窗口。
STAND_CONFIRM_DURATION = 1.5
STAND_UPRIGHT_COS = 0.9
STAND_MIN_HEIGHT = 0.05
STAND_TARGET_HEIGHT = 0.055
STAND_GROUND_HEIGHT = 0.024

# 站立稳定性判据（背部朝上 + 高度达标之外的第三个条件）
# 取 14 个驱动关节速度的瞬时 RMS, 结算要求它在**整个窗口内的平均**低于门限。
# 用窗口均值而不是"连续达标时长": 窗口内任何一次剧烈抖动都会直接抬高均值,
# 因此骗不过去(不存在"多次短暂达标拼接成一次连续站稳"这条路),
# 也没有占空比悬崖 —— 后者会让"低于某个占空比就永远攒不满"变成不可达判据。
# 阈值标定自确定性回放实测的"站立窗口内关节速度 RMS":
#   model_600 = 2.14 / 新 run 800 = 4.21 / 旧 run 900 = 5.33 rad/s
STAND_VEL_MEAN_MAX = 3.5
# 维持站立窗口用的宽松几何阈值(比结算侧松), 低于它窗口清零重来。
STAND_UPRIGHT_COS_STAY = 0.8
STAND_MIN_HEIGHT_STAY = 0.045
# 站立静止奖励的线性核满值速度: vel_rms=0 得满分, 达到该速度归零。
# 取 6.0 使 3.5 门限附近仍保留约 0.42 的梯度, 远高于 6 rad/s 时不再有区分意义。
STAND_STILL_FULL_SPEED = 6.0

# 本轮训练和默认策略回放使用相同速度；原速度课程保留但不启用。
TIME_COMPARISON_SCALE = 3.0

# 现有身体轨迹文件的采集时序，不随本轮动作时间修改。
BODY_TRAJ_BUILD_END = 0.65
BODY_TRAJ_P1_END = 0.80
