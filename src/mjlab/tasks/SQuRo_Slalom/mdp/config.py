# 命令字段定义与列索引
COMMAND_FIELDS = ("velocity_x", "height_f", "height_h", "gait_frequency", "curvature")
COMMAND_DIM = len(COMMAND_FIELDS)
CMD_VEL_X_IDS = COMMAND_FIELDS.index("velocity_x")
CMD_HEIGHT_F_IDS = COMMAND_FIELDS.index("height_f")
CMD_HEIGHT_H_IDS = COMMAND_FIELDS.index("height_h")
CMD_GAIT_FREQ_IDS = COMMAND_FIELDS.index("gait_frequency")
CMD_CURVATURE_IDS = COMMAND_FIELDS.index("curvature")


# 命令参数
BASE_VEL = 0.1                      # 基础速度 (m/s, 1Hz 时)
VELOCITY_CURVATURE_POWER = 1.5
VEL_MIN = 0.15                      # 最大曲率下速度缩放比例 (转弯低速)
GAIT_FREQ_MIN = 1.0                 # Phase 0 步频采样下限
GAIT_FREQ_MAX = 2.0                 # Phase 0 步频采样上限
FIXED_HEIGHT_F = 0.055              # 命令默认前肢高度
FIXED_HEIGHT_H = 0.055              # 命令默认后肢高度


# 训练阶段边界与回合时长
STEPS_PER_ITER = 48                 # 与 Hole 的每轮策略步数一致
PHASE1_MID_ITER = 2000              # iter 0-2000: 转弯曲率增大
PHASE1_END_ITER = 3000              # 转圈结束后，新回合进入绕杆
PHASE2_MID_ITER = 6000              # 绕杆奖励课程的后段边界
PHASE2_END_ITER = 6000              # 完整训练预算
PHASE0_EPISODE_SECONDS = 20.0
SLALOM_EPISODE_SECONDS = 30.0
PASSAGE_REQUIRED_OBSTACLES = 3


# 绕杆曲率与参考几何
SLALOM_CURVATURE_MIN = 5
SLALOM_CURVATURE_MAX = 15
CURVATURE_TARGET_MAX = 20.0         # 最大曲率同时用于命令采样与最大形变归一化
OBSTACLE_SPACING_START = 0.20           # 编译占位；运行时杆距由本回合曲率决定
APPROACH_LENGTH = 0.03
TABLE_RESOLUTION = 50               # 预计算表相位分辨率
PROJECTION_STRIDE = 8
PHASE0_ENVELOPE_SAMPLE_STEP = 0.006 # 一阶段动态回合的包络边采样间距 (m)


# 杆与身体几何
OBSTACLE_RADIUS = 0.01
OBSTACLE_HALF_HEIGHT = 0.05
OBSTACLE_NUM = 12
F_BODY_HALF_LENGTH = 0.04           # F_body_Link 在XoY平面上沿身体前后轴半长(m)
F_BODY_HALF_WIDTH = 0.035           # F_body_Link 在XoY平面左右方向半宽(m)
H_BODY_HALF_LENGTH = 0.04           # H_body_Link 在XoY平面上沿身体前后轴半长(m)
H_BODY_HALF_WIDTH = 0.035           # H_body_Link 在XoY平面左右方向半宽(m)
BODY_REF_OFFSET = 0.04              # F_body/H_body 中心距 base 中心的X轴偏移量
CORRIDOR_HALF_WIDTH = 0.04          # 走廊半宽（基元阶段 = 身体半宽 + 控制余量）


# 步态与参考限位
PHASE_LAG = {"FL": 0.0, "FR": 0.5, "HL": 0.5, "HR": 0.0}
STRIDE_MIN = 0.0                    # 最小步幅
MAX_STRIDE_SCALE = 1.0              # 参考不超过生物数据原幅值，更快的命令由策略吸收
F_SPINE1_LIMIT = 0.6                # 取 SQuRo.xml 关节 range 与执行器 ctrlrange 的较小者
H_SPINE1_LIMIT = 0.6
NECK_YAW_LIMIT = 0.8


# 参考平滑与响应时间
SMOOTH_TIME = 1.0                   # 仅用于定义固定平滑弧长，不代表实际过渡耗时
SMOOTH_VEL = 0.025                  # 固定平滑弧长 = SMOOTH_VEL × SMOOTH_TIME
REFERENCE_CURVATURE_BLEND = 2.0     # 二阶段后脊柱在零曲率附近的平滑范围
SPINE_REFERENCE_TIME_CONSTANT = 0.03  # 二阶段四脊柱位置参考的响应时间（秒）


# 一阶段动态曲率
PHASE0_DYNAMIC_START_ITER = 1000
PHASE0_DYNAMIC_PROBABILITY = 0.5
PHASE0_MIN_SEGMENTS = 4
PHASE0_MAX_SEGMENTS = 6
PHASE0_SEGMENT_JITTER = 0.5         # 段时长抖动 (秒), 使切换时机与步态相位独立
PHASE0_TRANSITION_SECONDS = 0.5     # 曲率切换过渡时长 (秒)


# 杆距课程与考核
SPACING_INITIAL_CURVATURE_MAX = 10
SPACING_HARDEST_PROBABILITY = 0.30       # 普通模式最低档固定采样概率
SPACING_SUCCESS_RATE = 0.60              # 普通模式推进门槛：最低档通过率
SPACING_REQUIRED_BATCHES = 2             # 每次考核批数，两批都达标才推进
SPACING_EVALUATION_MIN_SAMPLES = 256     # 普通模式一批内最低档回合数下限
SPACING_MIN_DWELL_ITERATIONS = 100
SPACING_PRETRAIN_ITERATIONS = 500
SPACING_EVALUATION_INTERVAL = 100
SPACING_EVALUATION_ENVS = 1024           # 独立考核环境数，每批取一整轮
SPACING_EVALUATION_EPISODE_SECONDS = 20.0  # 考核回合时长, 比训练短以降低考核耗时
FINAL_ACCEPT_RATE = 0.80                 # 最终档全挡位验收的整体通过率门槛
NORMAL_BIN_TARGET_RATE = 0.60            # 普通模式非最低档加权的目标通过率
FINAL_BIN_WEIGHT_GAIN = 2.0              # 未达标档位的训练采样权重增益
