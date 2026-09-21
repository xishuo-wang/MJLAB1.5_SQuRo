# 参考轨迹分段时长 (名义秒, 执行时乘以 λ)。相位机只有 P1/P2/P3 三相,
# 与下面的 T0~T5 段不是一一对应: phase0 = T0+T1+T2, phase1 = T3, phase2 = T4+T5。
T0 = 0.50                             # 前置收腿段: 腿 LEG_INIT -> HOLD, 脊柱/颈保持 0
T1 = 0.65                             # P1 一阶段: 脊柱扭转
T2 = 0.15                             # P1 二阶段: 扭转保持
T3 = 0.15                             # P2 一阶段: 脊柱解扭
T4 = 0.50                             # P3 一阶段: 腿从收缩位回直立位
T5 = 1.05                             # P3 二阶段: 站立保持

# 累计边界 (按语义引用, 避免到处写 T0+T1+...)
P1_END = T0 + T1 + T2                 # 1.30  P1 名义段末 = S1 判决时刻
P2_END = P1_END + T3                  # 1.45  P2 名义段末 = S2 判决时刻
STAND_TRANSITION_END = P2_END + T4    # 1.95  起立过渡段末端
REFERENCE_TOTAL_TIME = STAND_TRANSITION_END + T5   # 3.00  参考轨迹总长

# 段内口径 (不含前置收腿段 T0): 姿态参考与录制身体轨迹表都以 T1 起点为 0。
# 累计口径的 P1_END/P2_END 不能直接当段内时刻用 —— 会整整多出 T0。
P1_SPAN = T1 + T2                     # 0.80  P1 段末(段内) = S1 判决时刻
P2_SPAN = T1 + T2 + T3                # 0.95  P2 段末(段内) = 前段翻正完成时刻

PRE_DURATION = T0                     # 前置收腿段时长 (等价别名, 逐步淘汰)
LEG_INIT = (0.1, -0.3, 0.1, -0.3, -0.1, 0.3, -0.1, 0.3)   # 站立状态腿关节角
FL_HOLD = (-0.28, 0.55)               # 前腿收缩状态关节角 (shoulder, elbow)
HL_HOLD = (-1.50, -0.25)              # 后腿收缩状态关节角 (hip, knee)

STAND_CONFIRM_DURATION = 1.0          # 站立成功所需的连续窗口时长
STAND_GROUND_HEIGHT = 0.024           # 趴平贴地时的躯干高度
STAND_TARGET_HEIGHT = 0.055           # 站立目标躯干高度
STAND_VEL_MEAN_MAX = 4.5              # 站立窗口平均关节速度上限 (判据量)
