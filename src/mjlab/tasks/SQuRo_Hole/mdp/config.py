# 命令系统相关配置
BASE_HEIGHT = 0.055         # 基准高度：正常行走下的躯干中心高度
BASE_FREQ = 2.0             # 基准速度：正常行走下的步态频率
BASE_SPEED = 0.2            # 基准速度：基准步频和基准高度下的期望速度
THRESHOLD_HEIGHT = 0.0395   # 高度阈值: 低于该值视为低高度 (取 0.0395 而非 0.04, 避免 float32(0.04) 与阈值等值时判定歧义)



# 命令层的四种高度模式: 按前后肢各自是否低于阈值切分, 覆盖全部组合
# 参考表只用其中的 0/1/2 (它的 0 同时覆盖"双高"与"双低")
MODE_BOTH_HIGH = 0
MODE_FRONT_LOW = 1
MODE_HIND_LOW = 2
MODE_BOTH_LOW = 3


# 命令张量槽位: command = [vx, vy, vz, h_F, h_H, angle]
CMD_VEL_X_IDS, CMD_VEL_Y_IDS, CMD_VEL_Z_IDS = 0, 1, 2
CMD_HEIGHT_F_IDS, CMD_HEIGHT_H_IDS, CMD_ANGLE_IDS = 3, 4, 5




# 预计算表相关配置
HEIGHT_LIST = [0.02, 0.04, 0.05, 0.055]     # 预计算表的高度档 (m): 参考表 [模式][高度档][相位][关节] 的第二维
TABLE_RESOLUTION = 500                      # 相位分辨率: 参考表第三维的采样点数
USE_SPINE_CSV = False                       # 是否启用脊柱 CSV 轨迹 (仓库内已打包, 默认不启用)
# 头颈期望位置与速度 (顺序: Neck_yaw, Neck_pitch)
# 位置同时就是复位姿态, 使 action=0 恰好维持头部俯仰 (见 docs §9.8); 两者必须同值
NECK_REF_POS = (0.0, -0.3)
NECK_REF_VEL = (0.0, 0.0)
# 前高后低 (参考表 mode 2) 的后腿收缩状态关节角 (hip, knee), 固定不摆动
HL_HOLD = (-1.50, -0.25)
# 低高度档与单侧模式下参考表的固定后脊柱俯仰角 (第 3 列 H_spine1)
SPINE_LOW_BEND = -0.65



# 各阶段的随机采样高度池 (阶段 3 起用位置表, 不再随机采样)
RANDOM_HEIGHT_VALUES = [0.04, 0.045, 0.05, 0.055]
FULL_HEIGHT_VALUES = [0.02, 0.04, 0.045, 0.05, 0.055]

# 角度命令候选 (度)
ANGLE_VALUES = [0.0]
