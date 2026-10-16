# 命令系统相关配置
BASE_HEIGHT = 0.055                                 # 基准高度：正常行走下的躯干中心高度
BASE_FREQ = 2.0                                     # 基准速度：正常行走下的步态频率
BASE_SPEED = 0.2                                    # 基准速度：基准步频和基准高度下的期望速度
BASE_CURVATURE = 0.0                                # 基准曲率
THRESHOLD_HEIGHT = 0.040                            # 高度阈值: 低于该值视为低高度
RANDOM_HEIGHT_VALUES = [0.04, 0.045, 0.05, 0.055]       # 0-1k 轮采样的高度值
FULL_HEIGHT_VALUES = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055]  # 1-3k 轮前后独立均匀采样


# 四种高度模式
MODE_BOTH_HIGH = 0                                  # 前后肢均高
MODE_FRONT_LOW = 1                                  # 前肢低后肢高
MODE_HIND_LOW = 2                                   # 前肢高后肢低
MODE_BOTH_LOW = 3                                   # 前后肢均低
NUM_MODES = 4                                       # 高度模式总数量


# 命令相关索引
COMMAND_FIELDS = ("velocity_x", "height_f", "height_h", "gait_frequency", "curvature", "obstacle_height")
COMMAND_DIM = len(COMMAND_FIELDS)
CMD_VEL_X_IDS = COMMAND_FIELDS.index("velocity_x")
CMD_HEIGHT_F_IDS = COMMAND_FIELDS.index("height_f")
CMD_HEIGHT_H_IDS = COMMAND_FIELDS.index("height_h")
CMD_GAIT_FREQ_IDS = COMMAND_FIELDS.index("gait_frequency")
CMD_CURVATURE_IDS = COMMAND_FIELDS.index("curvature")
CMD_OBSTACLE_HEIGHT_IDS = COMMAND_FIELDS.index("obstacle_height")


# 预计算表相关配置
HEIGHT_LIST = FULL_HEIGHT_VALUES                    # 预计算表第二维；低姿态查询直接使用连续高度 IK
TABLE_RESOLUTION = 500                              # 相位分辨率: 参考表第三维的采样点数
USE_SPINE_CSV = False                               # 是否启用脊柱 CSV 轨迹 (仓库内已打包, 默认不启用)
NECK_REF_POS = (0.0, -0.3)                          # 头颈相关参考姿态


# 受限空间高度课程相关配置
WARMUP_OBSTACLE_HEIGHT = 0.100                      # 无实体时的 100 毫米板高占位
OBSTACLE_HEIGHT_SPLIT = 0.065
ENTRY_MARGIN = 0.150
EXIT_MARGIN = 0.100
BODY_HEIGHT_OFFSET = 0.025
HEIGHT_MIN = 0.046                                  # 受限空间高度最小值
HEIGHT_MAX = 0.090                                  # 受限空间高度最大值
HEIGHT_INITIAL_MIN = 0.070                          # 课程开始时受限空间高度最小值
HEIGHT_FINE_STEP_THRESHOLD = 0.050                  # 课程阈值 50mm
HEIGHT_STEP = 0.002                                 # 大于阈值时课程 step
HEIGHT_FINE_STEP = 0.001                            # 小于阈值时课程 step
HEIGHT_MIN_SAMPLE_PROBABILITY = 0.30                # 最小高度固定采样概率
HEIGHT_MIN_SUCCESS_BATCH = 256                      # 推进课程要求成功批次
HEIGHT_MIN_SUCCESS_RATE = 0.60                      # 推进课程要求成功率
HEIGHT_REQUIRED_BATCHES = 2                         # 推进课程要求连续成功批次
HEIGHT_MIN_DWELL_ITERATIONS = 100
OBSTACLE_POSITION = (0.2, 0.0, HEIGHT_INITIAL_MIN)  # 真实课程的初始板底位置
OBSTACLE_SIZE = (0.015, 0.1, 0.005)                 # 单板半长、半宽、半厚
VIRTUAL_CLEARANCE_MARGIN = 0.0025                   # 板底净空余量
