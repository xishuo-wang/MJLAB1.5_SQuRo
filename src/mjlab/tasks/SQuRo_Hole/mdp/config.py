# 命令系统相关配置
BASE_HEIGHT = 0.055         # 基准高度：正常行走下的躯干中心高度
BASE_FREQ = 2.0             # 基准速度：正常行走下的步态频率
BASE_SPEED = 0.2            # 基准速度：基准步频和基准高度下的期望速度
THRESHOLD_HEIGHT = 0.0395   # 高度阈值: 低于该值视为低高度 (取 0.0395 而非 0.04, 避免 float32(0.04) 与阈值等值时判定歧义)

# 命令层四种高度模式: 按前后肢各自是否低于阈值切分, 覆盖全部组合
# 注: 参考表的 mode 只有 3 个 (它的 0 同时覆盖"双高"与"双低", 见 reference.py),
# 两者是不同的分层, 不要互相替换
MODE_BOTH_HIGH = 0          # 前后肢均高
MODE_FRONT_LOW = 1          # 前肢低后肢高
MODE_HIND_LOW = 2           # 前肢高后肢低
MODE_BOTH_LOW = 3           # 前后肢均低

# 6 维命令张量的槽位: 顺序与 observations/command 观测列一致
CMD_VEL_X_IDS = 0           # X 方向速度命令
CMD_VEL_Y_IDS = 1           # Y 方向速度命令
CMD_VEL_Z_IDS = 2           # Z 方向速度命令
CMD_HEIGHT_F_IDS = 3        # 前肢高度命令
CMD_HEIGHT_H_IDS = 4        # 后肢高度命令
CMD_ANGLE_IDS = 5           # 期望角度命令
CMD_VEL_IDS = slice(CMD_VEL_X_IDS, CMD_VEL_Z_IDS + 1)




# 预计算表相关配置
HEIGHT_LIST = [0.02, 0.04, 0.05, 0.055]     # 预计算表的高度档 (m): 参考表 [模式][高度档][相位][关节] 的第二维
TABLE_RESOLUTION = 500                      # 相位分辨率: 参考表第三维的采样点数
USE_SPINE_CSV = False                       # 是否启用脊柱 CSV 轨迹 (仓库内已打包, 默认不启用)
# 头颈期望位置与速度 (顺序: Neck_yaw, Neck_pitch)
# 位置同时就是复位姿态, 使 action=0 恰好维持头部俯仰 (见 docs §9.8); 两者必须同值
NECK_REF_POS = (0.0, -0.3)
NECK_REF_VEL = (0.0, 0.0)
HIND_HOLD_POS = (0.80, -0.15)  # 当前基线的后腿收缩参考角



# 命令与课程共用的阶段边界 (iter): 唯一来源, curriculums/command 都从这里取
# 1) 0-1k     正常高度范围内的随机采样
# 2) 1k-2k    含低高度状态的全高度随机采样
# 3) 2k-3k    位置表 (位移时刻表) + 虚拟碰撞软约束
# 4) 3k 以后  位置表 + 真实碰撞
# 注: 阶段 4 的实体碰撞由 rl/runner.py 在阶段边界重建环境实现 (contype 编译期固化)
STAGE1_END_ITER = 1000
STAGE2_END_ITER = 2000
STAGE3_END_ITER = 3000
NUM_STAGES = 4

STEPS_PER_ITER = 48
STAGE1_END = STAGE1_END_ITER * STEPS_PER_ITER
STAGE2_END = STAGE2_END_ITER * STEPS_PER_ITER
STAGE3_END = STAGE3_END_ITER * STEPS_PER_ITER

# 各阶段的随机采样高度池 (阶段 3 起用位置表, 不再随机采样)
RANDOM_HEIGHT_VALUES = [0.04, 0.045, 0.05, 0.055]
FULL_HEIGHT_VALUES = [0.02, 0.04, 0.045, 0.05, 0.055]

# 各阶段是否要求限高板实体碰撞 (编译期固化; 切换由 runner 重建环境完成)
STAGE_COLLISION = (False, False, False, True)

# 角度命令候选 (度)
ANGLE_VALUES = [0.0]



# 按全局步数取当前阶段 (1~4); 阶段边界是命令/课程/碰撞的唯一来源
def get_current_stage(step_counter: int) -> int:
    if step_counter < STAGE1_END:
        return 1
    if step_counter < STAGE2_END:
        return 2
    if step_counter < STAGE3_END:
        return 3
    return 4


# 该阶段是否使用位置表 (阶段 3/4)
def stage_uses_schedule(stage: int) -> bool:
    return stage >= 3


# 该阶段是否要求实体碰撞
def stage_requires_collision(stage: int) -> bool:
    idx = min(max(int(stage), 1), NUM_STAGES) - 1
    return STAGE_COLLISION[idx]


# 按阶段的随机采样高度池; 阶段 3/4 返回 None (用位置表)
def heights_for_stage(stage: int):
    if stage == 1:
        return RANDOM_HEIGHT_VALUES
    if stage == 2:
        return FULL_HEIGHT_VALUES
    return None
