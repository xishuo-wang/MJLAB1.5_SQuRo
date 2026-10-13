# 命令系统相关配置
BASE_HEIGHT = 0.055         # 基准高度：正常行走下的躯干中心高度
BASE_FREQ = 2.0             # 基准速度：正常行走下的步态频率
BASE_SPEED = 0.2            # 基准速度：基准步频和基准高度下的期望速度
BASE_CURVATURE = 0.0        # 基准曲率
THRESHOLD_HEIGHT = 0.0395   # 高度阈值: 低于该值视为低高度


# 四种高度模式
MODE_BOTH_HIGH = 0          # 前后肢均高
MODE_FRONT_LOW = 1          # 前肢低后肢高
MODE_HIND_LOW = 2           # 前肢高后肢低
MODE_BOTH_LOW = 3           # 前后肢均低


# 命令相关索引
COMMAND_SCHEMA_VERSION = 2
COMMAND_FIELDS = ("velocity_x", "height_f", "height_h", "gait_frequency", "curvature", "obstacle_height")
COMMAND_DIM = len(COMMAND_FIELDS)
CMD_VEL_X_IDS = COMMAND_FIELDS.index("velocity_x")
CMD_HEIGHT_F_IDS = COMMAND_FIELDS.index("height_f")
CMD_HEIGHT_H_IDS = COMMAND_FIELDS.index("height_h")
CMD_GAIT_FREQ_IDS = COMMAND_FIELDS.index("gait_frequency")
CMD_CURVATURE_IDS = COMMAND_FIELDS.index("curvature")
CMD_OBSTACLE_HEIGHT_IDS = COMMAND_FIELDS.index("obstacle_height")





# 预计算表相关配置
HEIGHT_LIST = [0.02, 0.04, 0.05, 0.055]     # 预计算表的高度档 (m): 参考表 [模式][高度档][相位][关节] 的第二维
TABLE_RESOLUTION = 500                      # 相位分辨率: 参考表第三维的采样点数
USE_SPINE_CSV = False                       # 是否启用脊柱 CSV 轨迹 (仓库内已打包, 默认不启用)
NECK_REF_POS = (0.0, -0.3)
NECK_REF_VEL = (0.0, 0.0)
HIND_HOLD_POS = (0.80, -0.15)  # 当前基线的后腿收缩参考角



# 高度课程规则参数；可视化与训练共用此处定义。
WARMUP_OBSTACLE_HEIGHT = 0.100      # 无实体时的100毫米板高占位
OBSTACLE_HEIGHT_SPLIT = 0.070
ENTRY_MARGIN = 0.150
EXIT_MARGIN = 0.100
BODY_HEIGHT_OFFSET = 0.030
LOW_TARGET_HEIGHT = 0.020
HEIGHT_MIN = 0.048
HEIGHT_MAX = 0.090
HEIGHT_INITIAL_MIN = 0.070
HEIGHT_STEP = 0.002
HEIGHT_MIN_SAMPLE_PROBABILITY = 0.30
HEIGHT_MIN_SUCCESS_BATCH = 256
HEIGHT_MIN_SUCCESS_RATE = 0.60
HEIGHT_REQUIRED_BATCHES = 2
HEIGHT_MIN_DWELL_ITERATIONS = 100
HOLE_POSITION = (0.2, 0.0, HEIGHT_INITIAL_MIN)    # 真实课程的初始板底位置
HOLE_SIZE = (0.015, 0.1, 0.005)                  # 单板半长、半宽、半厚
VIRTUAL_CLEARANCE_MARGIN = 0.0025                # 板底净空余量



# 课程阶段：两段原有热身、全位置走廊热身、单板真实碰撞课程。
STAGE1_END_ITER = 1000
STAGE2_END_ITER = 2000
STAGE3_END_ITER = 3000
COURSE_START_ITER = STAGE3_END_ITER
HEIGHT_COURSE_STAGE = 4
NUM_STAGES = 4

STEPS_PER_ITER = 48
STAGE1_END = STAGE1_END_ITER * STEPS_PER_ITER
STAGE2_END = STAGE2_END_ITER * STEPS_PER_ITER
COURSE_START = COURSE_START_ITER * STEPS_PER_ITER
STAGE3_END = STAGE3_END_ITER * STEPS_PER_ITER

# 前三阶段使用随机高度池，第四阶段按板高与基座位置查表。
RANDOM_HEIGHT_VALUES = [0.04, 0.045, 0.05, 0.055]
FULL_HEIGHT_VALUES = [0.02, 0.04, 0.045, 0.05, 0.055]

# 各阶段是否要求限高板实体碰撞 (编译期固化; 切换由 runner 重建环境完成)
STAGE_COLLISION = (False, False, False, True)



# 按全局步数取当前阶段，统一命令、奖励课程与碰撞边界。
def get_current_stage(step_counter: int) -> int:
    if step_counter < STAGE1_END:
        return 1
    if step_counter < STAGE2_END:
        return 2
    if step_counter < STAGE3_END:
        return 3
    return HEIGHT_COURSE_STAGE



# 判断是否已进入真实碰撞高度课程。
def stage_uses_height_course(stage: int) -> bool:
    return stage >= HEIGHT_COURSE_STAGE



# 该阶段是否要求实体碰撞
def stage_requires_collision(stage: int) -> bool:
    idx = min(max(int(stage), 1), NUM_STAGES) - 1
    return STAGE_COLLISION[idx]



# 按阶段获取热身高度池; 课程阶段不使用随机高度命令池。
def heights_for_stage(stage: int):
    if stage == 1:
        return RANDOM_HEIGHT_VALUES
    if stage in (2, 3):
        return FULL_HEIGHT_VALUES
    return None
