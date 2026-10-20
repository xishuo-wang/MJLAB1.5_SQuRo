# SQuRo Slalom 命令与几何静态配置（本文件不得 import 任务内其他模块）

FIXED_HEIGHT_F = 0.055   # 命令默认前肢高度
FIXED_HEIGHT_H = 0.055   # 命令默认后肢高度


# SQuRo 身体尺寸参数
F_BODY_HALF_LENGTH = 0.04       # F_body_Link 在XoY平面上沿身体前后轴半长(m)
F_BODY_HALF_WIDTH  = 0.035      # F_body_Link 在XoY平面左右方向半宽(m)
H_BODY_HALF_LENGTH = 0.04       # H_body_Link 在XoY平面沿身体前后轴半长(m)
H_BODY_HALF_WIDTH  = 0.035      # H_body_Link 在XoY平面左右方向半宽(m)
BODY_REF_OFFSET = 0.04          # F_body/H_body 中心距 base 中心的X轴偏移量
CORRIDOR_HALF_WIDTH = 0.04      # 走廊半宽（基元阶段 = 身体半宽 + 控制余量）


# 杆阵列编译占位；运行时杆距由本回合曲率决定
POLE_SPACING_START = 0.20
