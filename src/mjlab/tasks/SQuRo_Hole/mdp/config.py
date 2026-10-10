# 命令系统相关配置
BASE_HEIGHT = 0.055         # 基准高度：正常行走下的躯干中心高度
BASE_FREQ = 2.0             # 基准速度：正常行走下的步态频率
BASE_SPEED = 0.2            # 基准速度：基准步频和基准高度下的期望速度
THRESHOLD_HEIGHT = 0.0395   # 高度阈值: 低于该值视为低高度 (取 0.0395 而非 0.04, 避免 float32(0.04) 与阈值等值时判定歧义)
MODE_BOTH_HIGH = 0          # 前后肢均高
MODE_FRONT_LOW = 1          # 前肢低后肢高
MODE_HIND_LOW = 2           # 前肢高后肢低
MODE_BOTH_LOW = 3           # 前后肢均低
CMD_VEL_X_IDS = 0           # X 方向速度命令索引
CMD_VEL_Y_IDS = 1           # Y 方向速度命令索引
CMD_VEL_Z_IDS = 2           # Z 方向速度命令索引
CMD_HEIGHT_F_IDS = 3        # 前肢高度命令索引
CMD_HEIGHT_H_IDS = 4        # 后肢高度命令索引
CMD_ANGLE_IDS = 5           # 期望角度命令索引
ANGLE_VALUES = [0.0]        # 角度命令采样范围
RANDOM_HEIGHT_VALUES = [0.04, 0.045, 0.05, 0.055]       # 0-1k 高度采样范围
FULL_HEIGHT_VALUES = [0.02, 0.04, 0.045, 0.05, 0.055]   # 1-2k 高度采样范围



# 预计算表相关配置
HEIGHT_LIST = [0.02, 0.04, 0.05, 0.055]     # 预计算表的高度档 (m): 参考表 [模式][高度档][相位][关节] 的第二维
TABLE_RESOLUTION = 500                      # 相位分辨率: 参考表第三维的采样点数
USE_SPINE_CSV = False                       # 是否启用脊柱 CSV 轨迹 (仓库内已打包, 默认不启用)
NECK_REF_POS = (0.0, -0.3)                  # 期望头部位置
NECK_REF_VEL = (0.0, 0.0)                   # 期望头部速度
HL_HOLD = (-1.50, -0.25)                    # 前高后低时的后腿收缩状态关节角
SPINE_LOW_BEND = -0.65                      # 单低情况下的俯仰脊柱



