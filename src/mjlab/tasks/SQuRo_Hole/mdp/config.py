# 任务级全局常量: 多文件共用的基准值集中在这里定义, 其余模块一律 import
# 用法参考 SQuRo_Backup/mdp/config.py


# 正常行走高度 (身体/躯干中心高度)。既是参考表的轨迹缩放基准, 也是速度的高度基准
BASE_HEIGHT = 0.055

# 基准速度 (m/s): 对应"基准高度 + 名义步频"下的前进速度
BASE_SPEED = 0.2

# 高低肢判定阈值: 命令高度低于该值视为"低高度" (该段腿冻结)
HEIGHT_THRESHOLD = 0.04

# 名义步频 (Hz): 参考表内的摆线与 CSV 轨迹都按该步频生成
GAIT_FREQ = 2.0



# 预计算表的高度档 (m): 参考表 [模式][高度档][相位][关节] 的第二维
HEIGHT_LIST = [0.02, 0.04, 0.05, 0.055]

# 相位分辨率: 参考表第三维的采样点数
TABLE_RESOLUTION = 500

# 是否启用脊柱 CSV 轨迹 (仓库内已打包, 默认不启用)
USE_SPINE_CSV = False

# 头颈在参考表里的期望位置与速度: 恒为 0 (期望"头保持不动")
NECK_REF_POS = 0.0
NECK_REF_VEL = 0.0



# 命令阶段边界 (全局步数): 决定命令来源
# 注意前两段都只占 1 个 iter, 第 2 个控制步起即进入阶段 3 (旧版原始行为, 未改)
STEPS_PER_ITER = 24
STAGE1_END = 1 * STEPS_PER_ITER
STAGE2_END = 1 * STEPS_PER_ITER
STAGE3_END = 4000 * STEPS_PER_ITER

# 各阶段可随机采样的高度值 (阶段 3 用位置表, 不随机采样)
STAGE1_HEIGHT_VALUES = [0.04, 0.045, 0.05, 0.055]
STAGE2_HEIGHT_VALUES = [0.02, 0.04, 0.045, 0.05, 0.055]
STAGE3_HEIGHT_VALUES = None

# 角度命令候选 (度)
ANGLE_VALUES = [0.0]
