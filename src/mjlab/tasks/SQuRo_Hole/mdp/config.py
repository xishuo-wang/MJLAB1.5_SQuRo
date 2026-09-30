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



# 命令与课程共用的阶段边界 (iter): 唯一来源, curriculums/command 都从这里取
# 1) 0-1k     正常高度范围内的随机采样
# 2) 1k-2k    含低高度状态的全高度随机采样
# 3) 2k-3k    位置表 (位移时刻表) + 虚拟碰撞软约束
# 4) 3k 以后  位置表 + 真实碰撞
# 注: 阶段 1~3 已生效; 阶段 4 的真实碰撞需要"按阶段重建环境"的机制, 尚未实现
STAGE1_END_ITER = 1000
STAGE2_END_ITER = 2000
STAGE3_END_ITER = 3000
NUM_STAGES = 4

STEPS_PER_ITER = 24
STAGE1_END = STAGE1_END_ITER * STEPS_PER_ITER
STAGE2_END = STAGE2_END_ITER * STEPS_PER_ITER
STAGE3_END = STAGE3_END_ITER * STEPS_PER_ITER

# 各阶段的随机采样高度池 (阶段 3 起用位置表, 不再随机采样)
RANDOM_HEIGHT_VALUES = [0.04, 0.045, 0.05, 0.055]
FULL_HEIGHT_VALUES = [0.02, 0.04, 0.045, 0.05, 0.055]

# 各阶段是否要求限高板实体碰撞 (编译期固化; 阶段 4 的机制未实现, 此处仅声明意图)
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
