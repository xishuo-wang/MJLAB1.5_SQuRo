# 命令系统相关配置
BASE_HEIGHT = 0.055                                 # 基准高度：正常行走下的躯干中心高度
BASE_FREQ = 2.0                                     # 基准速度：正常行走下的步态频率
BASE_SPEED = 0.2                                    # 基准速度：基准步频和基准高度下的期望速度
BASE_CURVATURE = 0.0                                # 基准曲率
THRESHOLD_HEIGHT = 0.040                            # 高度阈值: 低于该值视为低高度
LOW_POSE_SPEED_RATIO = 0.75                         # 最低姿态的速度附加缩放，40毫米时恢复为1
RANDOM_HEIGHT_VALUES = [0.04, 0.045, 0.05, 0.055]   # 0-1k 轮采样的高度值
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
HEIGHT_LIST = FULL_HEIGHT_VALUES                    # 预计算表第二维；低姿态使用固定角标定表插值
TABLE_RESOLUTION = 500                              # 相位分辨率: 参考表第三维的采样点数
USE_SPINE_CSV = False                               # 是否启用脊柱 CSV 轨迹 (仓库内已打包, 默认不启用)
NECK_REF_POS = (0.0, -0.3)                          # 头颈相关参考姿态
LOW_POSE_CALIBRATION = (
    (0.020, (-0.36, 0.48), (0.80, -0.15)),
    (0.025, (-0.37, 0.24), (0.78, 0.02)),
    (0.030, (-0.29, 0.14), (0.73, 0.11)),
    (0.035, (-0.35, -0.06), (0.62, 0.15)),
    (0.040, (-0.26, -0.11), (0.59, 0.27)),
)                                                   # 低姿态时高度、前腿角、后腿角标定表
LOW_SPINE_PITCH = -0.65                             # 最大高度差时参考脊柱俯仰角


# 受限空间高度课程相关配置
WARMUP_OBSTACLE_HEIGHT = 0.100                      # 无实体时的 100 毫米板高占位
OBSTACLE_HEIGHT_SPLIT = 0.065
HIGH_OBSTACLE_EQUAL_PROBABILITY = 0.50              # 高板每回合等高／分段各半
OBSTACLE_HEIGHT_EDGES = (0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080)  # 左闭右开升档，目标取 HEIGHT_LIST
ENTRY_MARGIN = 0.150
EXIT_MARGIN = 0.100
BODY_HEIGHT_OFFSET = 0.025
HEIGHT_MIN = 0.047                                  # 受限空间高度最小值
HEIGHT_MAX = 0.090                                  # 受限空间高度最大值
HEIGHT_INITIAL_MIN = 0.070                          # 课程开始时受限空间高度最小值
HEIGHT_FINE_STEP_THRESHOLD = 0.050                  # 课程阈值 50mm
HEIGHT_STEP = 0.002                                 # 大于阈值时课程 step
HEIGHT_FINE_STEP = 0.001                            # 小于阈值时课程 step
HEIGHT_MIN_SAMPLE_PROBABILITY = 0.30                # 普通模式最小高度固定采样概率
HEIGHT_MIN_SUCCESS_RATE = 0.60                      # 推进课程要求成功率
HEIGHT_EVALUATION_BATCHES = 2                       # 每次独立考核执行的批数
HEIGHT_MIN_DWELL_ITERATIONS = 100
FINAL_ACCEPT_RATE = 0.80                            # 最终档全挡位验收的整体通过率门槛
NORMAL_BIN_TARGET_RATE = 0.60                       # 普通模式非最低挡位加权的目标通过率
FINAL_BIN_WEIGHT_GAIN = 2.0                         # 未达标挡位的训练采样权重增益
RETREAT_FAILURE_LIMIT = 5                           # 连续失败考核次数达到该值后退一档
RETREAT_REQUIRED_PASSES = 2                         # 退档后重新推进所需的连续达标考核次数
OBSTACLE_POSITION = (0.2, 0.0, HEIGHT_INITIAL_MIN)  # 真实课程的初始板底位置
OBSTACLE_SIZE = (0.015, 0.3, 0.005)                 # 单板半长、半宽、半厚, 总宽60厘米阻断绕侧
VIRTUAL_CLEARANCE_MARGIN = 0.0025                   # 板底净空余量
VIRTUAL_CONTACT_X_MARGIN = 0.005                    # 虚拟碰撞判定沿 X 前后各扩展 5 毫米
