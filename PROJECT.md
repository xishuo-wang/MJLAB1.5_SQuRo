# 脊柱型四足机器人强化学习项目


## 项目概述

基于 MuJoCo + IsaacLab API 实现的 MJLAB 框架，研究脊柱型微小四足机器人 (SQuRo) 的脊腿协同运动。具体分为 SQuRo_Slalom（连续绕杆）与 SQuRo_Hole（分段钻洞）两个子任务。

**核心创新**：走廊一致性 (Corridor Conformance) 奖励框架，将机器人身体建模为两个铰接矩形 (F_body + H_body)，通过可调节宽度的走廊约束统一处理 XoY 平面绕杆和 YoZ 平面钻洞两种场景。


## 机器人模型

### SQuRo 结构

SQuRo 的 XML文件路径为：D:\MuJoCoLab_1.5\src\mjlab\asset_zoo\robots\SQuRo\xmls\SQuRo.xml

从前到后大致分为：头部、前肢躯干及腿部、脊柱、后肢躯干及腿部

腿部具有两个自由度，无法实现内收/外展运动，只能在XoZ平面内运动

脊柱为一个类似于万向节的机构：从前往后有四个关节：F_body_joint、F_spine1_joint、H_spine1_joint、H_body_joint

其中，F_body_joint和H_body_joint为扭转自由度（旋转轴平行于世界坐标系X轴）
F_spine1_joint初始为偏航自由度（初始旋转轴平行于世界坐标系Z轴）
H_spine1_joint初始为俯仰自由度（初始旋转轴平行于世界坐标系Y轴）

随着扭转，F_spine1_joint和H_spine1_joint的旋转轴也会发生变化，但两个旋转轴始终保持正交

SQuRo 共 14 个执行器，按 entity actuator 顺序：

| 序号 | 名称 | 关节 | 功能 |
|------|------|------|------|
| 0 | F_spine1 | F_spine1_joint | 侧摆(偏航) |
| 1 | F_body | F_body_joint | 前肢扭转 |
| 2 | Neck_yaw | Neck_yaw_joint | 头部偏航 |
| 3 | Neck_pitch | Neck_pitch_joint | 头部俯仰 |
| 4 | FL_shoulder | FL_shoulder_joint | 左前腿肩关节 |
| 5 | FL_elbow | FL_elbow_joint | 左前腿肘关节 |
| 6 | FR_shoulder | FR_shoulder_joint | 右前腿肩关节 |
| 7 | FR_elbow | FR_elbow_joint | 右前腿肘关节 |
| 8 | H_spine1 | H_spine1_joint | 俯仰 |
| 9 | H_body | H_body_joint | 后肢扭转 |
| 10 | HL_hip | HL_hip_joint | 左后腿髋关节 |
| 11 | HL_knee | HL_knee_joint | 左后腿膝关节 |
| 12 | HR_hip | HR_hip_joint | 右后腿髋关节 |
| 13 | HR_knee | HR_knee_joint | 右后腿膝关节 |

通过 mdp\indices.py 读取相关的索引

### 坐标系

#### 世界坐标系

世界坐标系 +X 为前、 +Y 为左、 +Z为上

机器人 base link 初始位于世界坐标系原点，朝向世界坐标系 +X

#### 局部坐标系

**body +X ≠ 物理前向**，SQuRo body frame 与世界坐标系有 90° 旋转偏移：

```
F_body_Link: body+X → world +Y (heading=+90°)
H_body_Link: body+X → world -Y (heading=-90°)
```

物理前向 heading 修正量符号相反：

```python
f_body_physical = body_X_heading - π/2  # +90° → 0° (world +X)
h_body_physical = body_X_heading + π/2  # -90° → 0° (world +X)
```

此处已验证，**如果后续设计需要读取其他局部坐标系，需要单独验证！**

### 身体尺寸常量 (`path.py`)

| 常量 | 值 | 含义 |
|------|-----|------|
| F_BODY_HALF_LENGTH | 0.04 m | F_body 半长(沿身体轴) |
| F/H_BODY_HALF_WIDTH | 0.035 m | 半宽(左右方向) |
| BODY_REF_OFFSET | 0.04 m | F/H 中心距 base 偏移 |
| CORRIDOR_HALF_WIDTH | 0.04 m | 走廊半宽(基元阶段) |
| _RMIN | 1/15 ≈ 0.0667 m | 最小转弯半径 (κ=±15) |



# SQuRo_Slalom 任务


## 文件结构

```
src/mjlab/tasks/SQuRo_Slalom/
├── SQuRo_Slalom_env_cfg.py    # 环境配置
├── config/
│   ├── __init__.py            # 任务注册
│   └── rl_cfg.py              # PPO 超参数
├── rl/
│   └── runner.py              # ONNX 导出
├── mdp/
│   ├── command.py             # 命令系统
│   ├── curriculums.py         # 课程学习
│   ├── events.py              # 模型重置
│   ├── indices.py             # 获取索引
│   ├── observations.py        # 观测空间
│   ├── path.py                # 参考路径生成
│   ├── pole.py                # 杆实体
│   ├── reference.py           # 关节参考预计算表
│   ├── rewards.py             # 奖励函数
│   └── terminations.py        # 终止条件
```

关键脚本：
- `scripts/SQuRo_play.py` — 策略回放 (自动 Phase 0/1 识别)


## 课程学习 (mdp/curriculums.py)

整体分为两阶段课程学习框架，为：Phase0 转向基元训练 + Phase1 绕杆场景训练

每个Phase又分为数个小阶段，用于关键变量（如转向曲率、杆间距）的课程学习，以及奖励函数的权重课程

所有阶段边界统一在 `curriculums.py` 定义，其余如 `command.py` 通过导入引用：

```
iter:   0 ─── 2000 ─── 4000 ─── 6000 ─── 8000
        PHASE1_MID  PHASE1_END  PHASE2_MID  PHASE2_END
        κ 0.5→20    基元→绕杆   间距缩窄    (预留)
        ├── Phase 0: 转弯基元 ──┤├── Phase 1: 绕杆 ──┤

Phase 0: 圆弧路径, κ 课程增长 0.5→20, 杆透明
Phase 1: LUT 路径, 杆间距 0.20→0.15m, 杆可见(无碰撞)
```


## 期望轨迹 (mdp/path.py)

期望轨迹是Slalom任务的重点。无论是 Phase0 还是 Phase1，本质上都应该跟随期望轨迹生成命令

### Phase0 的期望轨迹

根据**单个 Episode 内固定的曲率**生成一条弧线（或者直线）

在**train**下，这个曲率由 mdp/command.py 生成，并随着课程增大曲率取值范围直到最大

在**play**下，这个曲率由 scripts/SQuRo_play.py 文件中的  fixed_curvature 给定，覆盖掉 mdp/command.py 生成的曲率

### Phase1 的期望轨迹

根据**单个 Episode 内固定的杆间距**生成弧线与直线的结合的轨迹

在**train**下，这个杆间距由课程学习给定，并逐渐减小直到最小

在**play**下，这个杆间距由 scripts/SQuRo_play.py 文件中的 fixed_pole_spacing 给定，覆盖掉本身课程生成的杆间距

具体而言，**绕杆轨迹** 一个周期由 6 段组成，假设机器人最小转弯半径为 Rmin，杆间距为 X(X>=2Rmim)，机器人从 (0,0) 出发，第一根杆位于 (0,-Rmin),第二根杆位于 (X,-Rmin)……：

1. CW ¼弧: (0,0)→(Rmin,-Rmin), κ=-15
2. CCW ¼弧: →(2Rmin,-2Rmin), κ=+15
3. 直行: →(X,-2Rmin), κ=0
4. CCW ¼弧: →(X+Rmin,-Rmin), κ=+15
5. CW ¼弧: →(X+2Rmin,0), κ=-15
6. 直行: →(2X,0), κ=0

路径跨周期连续通过 `x_ref += num_periods * 2*spacing` 保证。


## 命令系统 (mdp/command.py)

5D 命令张量 `[vel_x, height_f, height_h, gait_freq, curvature]`：

| 字段 | Phase 0 | Phase 1 | 更新频率 |
|------|---------|---------|----------|
| vel_x | `base*(1-0.75*|κ|/20)` | `base*(1-0.75*15/20) ≈ 0.044` | 每ep reset |
| height_f/h | 固定 0.055 | 固定 0.055 | 周期性 |
| gait_freq | 固定 1.0 | 固定 1.0 | 周期性 |
| curvature | 课程采样/fixed_curvature | 每步动态 `get_path_curvature()` | Phase 0: reset; Phase 1: 每步 |


## 预计算表 (mdp/reference.py)

脊柱关节角度由瞬时曲率 κ 驱动：

```
f_spine1 = -0.6 × κ/κ_max     (κ=-max→+0.6, κ=+max→-0.6)
f_body   = -0.9 × κ/κ_max
h_spine1 = -0.6 × |κ|/κ_max   (始终≤0)
h_body   = -0.7 × κ/κ_max
```

Phase 0: κ=静态命令值; Phase 1: κ=`get_path_curvature()` 动态读取 LUT 瞬时值。

腿部参考由 CSV (Trot_F/H) + 逆运动学生成，16 曲率×50 相位×14 关节预计算表，运行时按曲率插值。内侧腿 Y 轨迹按 `|κ|` 缩放实现差速。


## 奖励函数 (mdp/reward.py)

| 奖励项 | 权重 (3阶段) | 功能 |
|--------|-------------|------|
| mimic_pos | 5.0/5.0/5.0 | 关节位置模仿 (腿+脊柱分σ) |
| mimic_vel | 2.5/2.5/2.5 | 关节速度模仿 |
| height | 2.5/2.5/2.5 | 身体高度跟踪 |
| track_vel | 4.0/4.0/4.0 | body-frame 前进速度 |
| corridor | 5.0/5.0/8.0 | 身体包络走廊约束 |
| track_head | 5.0/5.0/5.0 | F_body 朝向对齐 |
| action_L1/L2 | 0.1/0.5/0.5 | 动作平滑性 |
| energy | 0.1/0.5/0.5 | 能耗惩罚 |

### 走廊一致性奖励

```
e_i = |d_lat| + |L·sin(Δθ)| + |W·cos(Δθ)|  (身体在法向占据的半范围)
v_i = max(0, e_i - C)                        (超出走廊的量)
r = exp(-σ·v²)                               (每环节奖励)
```

- `d_lat`: 身体中心到路径的侧向距离
- `Δθ`: 身体朝向 − 路径朝向 (朝向误差膨胀项)
- `C`: 走廊半宽 (死区)
- 同时惩罚位置和朝向偏差，死区内不扣分

相比传统 track_path，corridor 自带朝向耦合且更符合物理直觉。


## 杆模块 (mdp/pole.py)

MuJoCo 圆柱体：直径 1cm, 高 10cm, Y 坐标 `POLE_Y = -1/15 ≈ -0.0667m (= -Rmin)`。

透明度控制：通过 `update_pole_visibility()` 按 `geom_type == CYLINDER` 匹配 WarpBridge 模型修改 alpha。Phase 0 透明，Phase 1 可见。训练全程 `contype=0` (无物理碰撞)。


## 回放脚本 (scripts/SQuRo_play.py)

提取 `train_iter` → `align_iter = max(0, train_iter-10)`
- 避免运行 3999 轮的策略时误加载到 Phase1(4000) 
- 自动识别 Phase：`align_iter < 4000 → Phase 0`，否则 Phase 1
- Phase 0：设 `fixed_curvature`，Phase 1：设 `fixed_pole_spacing` 并清除 `fixed_curvature`
- Phase 1 用正确间距重建杆实体，可视化与路径一致

CSV 记录 关节角度、速度、动作空间输出等信息，并保存视频



# 踩坑记录

## 坐标系

1. **body+X ≠ 物理前向**：F/H body 的 body+X 指向 world ±Y，需分别 ±π/2 修正
2. **不要假设对称身体环节有相同局部坐标系**：F_body 和 H_body 的 XML 局部四元数不同
3. **局部坐标系需要验证朝向**：不要只以 XML 中的旋转轴就假定方向，需要单独验证

## 可视化

1. **WarpBridge 不兼容 mj_name2id**：改用 `geom_type == CYLINDER` 匹配
2. **model.geom_rgba 是 torch tensor**：赋值需 `torch.tensor`，非 numpy
3. **entities 覆盖丢失实体**：用 `{**orig, **pole}` merge

## 变量管理

1. **全局变量**：如果某一个变量在多个文件中重复出现且被硬编码，需要在一个文件中定义为全局变量并在其他文件中导入，避免后续修改时遗漏。例如 SQuRo_Slalom\mdp\curriculums.py 文件中的

```
PHASE1_MID_ITER = 2000       # iter 0-2000: 转弯曲率增大
PHASE1_END_ITER = 4000       # iter 2000-4000: 转弯基元
PHASE2_MID_ITER = 6000       # iter 4000-6000: 绕杆间距缩小阶段
PHASE2_END_ITER = 8000       # iter 6000-8000: 绕杆训练
```

2. **索引**：如关节索引、site点索引，避免硬编码，统一在 mdp/indices 中进行获取，并导入相关索引

## 执行器

1. **动作空间维度变化**：如果涉及动作空间维度变化，如新增 Neck_yaw/Neck_pitch 需更新相关任务以及回放脚本，特别注意更新全部索引常量和预计算参考表的维度、索引