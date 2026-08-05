# 脊柱型四足机器人强化学习项目

## 项目概述

基于 MuJoCo + MJLAB 框架，研究脊柱型微小四足机器人 (SQuRo) 的脊腿协同运动。当前聚焦 SQuRo_Slalom 子任务（连续绕杆），SQuRo_Hole（钻洞）为后续子任务。

**核心创新**：走廊一致性 (Corridor Conformance) 奖励框架 — 将机器人身体建模为两个铰接矩形 (F_body + H_body)，通过可调节宽度的走廊约束统一处理 XoY 平面绕杆和 YoZ 平面钻洞两种场景。


## 机器人模型

### SQuRo 脊柱结构

SQuRo 的 XML文件路径为：D:\MuJoCoLab_1.5\src\mjlab\asset_zoo\robots\SQuRo\xmls\SQuRo.xml

从前到后大致分为：头部、前肢躯干及腿部、脊柱、后肢躯干及腿部

腿部具有两个自由度，无法实现内收/外展运动，只能在XoZ平面内运动

脊柱为一个类似于万向节的机构：从前往后有四个关节：F_body_joint、F_spine1_joint、H_spine1_joint、H_body_joint

其中，F_body_joint和H_body_joint为扭转自由度（旋转轴平行于世界坐标系X轴）
F_spine1_joint初始为偏航自由度（初始旋转轴平行于世界坐标系Z轴）
H_spine1_joint初始为俯仰自由度（初始旋转轴平行于世界坐标系Y轴）

随着扭转，F_spine1_joint和H_spine1_joint的旋转轴也会发生变化，但两个旋转轴始终保持正交
14 个执行器，通过 `mdp/indices.py` 统一管理索引：

| 序号 | 名称 | 功能 | 分组 |
|------|------|------|------|
| 0-1 | F_spine1, F_body | 脊柱(侧摆+扭转) | `actuator_spn_ids` |
| 2-3 | Neck_yaw, Neck_pitch | 颈部 | `actuator_neck_ids` |
| 4-7 | FL/FR shoulder/elbow | 前腿 | `actuator_leg_ids` |
| 8-9 | H_spine1, H_body | 脊柱(俯仰+扭转) | `actuator_spn_ids` |
| 10-13 | HL/HR hip/knee | 后腿 | `actuator_leg_ids` |

### 坐标系

**世界**：+X=前, +Y=左, +Z=上。机器人初始 X=-_INIT_DIST(=0.02, path.py), 面朝 +X。

**起点按阶段**：两阶段均为圆弧接近段起点。Phase 0 由 `get_phase0_approach` 按本 episode 曲率 κ 动态确定（events.py 先置直行起点，command.py reset 时按圆弧重写）；Phase 1 为圆弧接近段起点 `(-0.0199+spacing, -0.0013)` 朝向 +11.4°（由 `_approach_rev_table` 反推，随杆间距 spacing 右移；接近段终点 = 第一根杆正上方 `(spacing, 0)`）。

**局部**：body+X ≠ 物理前向，F/H body 的 body+X 指向 world ±Y：

```python
f_body_physical = body_X_heading - π/2  # F_body: +90° → 0° (world +X)
h_body_physical = body_X_heading + π/2  # H_body: -90° → 0° (world +X)
```

已验证。**若需读取其他局部坐标系，须单独验证。**

### 身体尺寸 (`path.py`)

| 常量 | 值 | 含义 |
|------|-----|------|
| F/H_BODY_HALF_LENGTH | 0.04 m | 半长(沿身体轴) |
| F/H_BODY_HALF_WIDTH | 0.035 m | 半宽(左右方向) |
| BODY_REF_OFFSET | 0.04 m | F/H 中心距 base 偏移 |
| CORRIDOR_HALF_WIDTH | 0.04 m | 走廊半宽 |
| _RMIN | 1/CURVATURE_TARGET | 最小转弯半径 |


# SQuRo_Slalom 任务

## 文件结构

```
src/mjlab/tasks/SQuRo_Slalom/
├── SQuRo_Slalom_env_cfg.py    # 环境配置
├── config/
│   ├── __init__.py            # 任务注册
│   └── rl_cfg.py              # PPO 超参数 (网络 512→256→128, 1024 envs)
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

脚本：`scripts/SQuRo_play.py` — 策略回放，自动 Phase 0/1 识别。


## 课程学习 (mdp/curriculums.py)

整体分为两阶段课程学习框架，为：Phase0 转向基元训练 + Phase1 绕杆场景训练

每个Phase又分为数个小阶段，用于关键变量（如转向曲率、杆间距）的课程
学习，以及奖励函数的权重课程

所有阶段边界统一在 `curriculums.py` 定义，其余如 `command.py` 通过导入引用：

```
iter:   0 ─── 2000 ─── 4000 ─── 6000 ─── 8000
        PHASE1_MID  PHASE1_END  PHASE2_MID  PHASE2_END
        κ 0.5→20    基元→绕杆   (预留)     (预留)
        ├── Phase 0: 转弯基元 ──┤├── Phase 1: 绕杆 ──┤

Phase 0: 圆弧路径, κ 课程增长, 步频 1~2Hz 随机, 杆透明
Phase 1: 平滑 LUT 路径, 杆间距确定性课程(0.20→最小间距 0.10 线性缩小, iter 4000→6000), 步频 1~2Hz 随机, 杆可见(无碰撞)
```

| 常量 | 值 | 用途 |
|------|-----|------|
| CURVATURE_MIN | 0.5 | Phase 0 κ 起始值 |
| CURVATURE_TARGET_MAX | 20.0 | Phase 0 κ 上限 |
| CURVATURE_TARGET | 20.0 | Phase 1 绕杆弧曲率 (= 1/Rmin, 与 CURVATURE_TARGET_MAX 相同) |
| GAIT_FREQ_MIN/MAX | 1.0/2.0 | Phase 0 步频采样范围 |
| GAIT_FREQ_PHASE1 | 1.0 | 默认步频 (Phase 1 实际也用 1~2Hz 随机采样) |
| POLE_SPACING_START | 0.20 | Phase 1 起始杆间距 (iter 4000 前固定) |
| POLE_SPACING_MIN | 0.10 | Phase 1 最小杆间距 (= 2Rmin, iter 6000) |
| SMOOTH_TIME | 1.0 | 曲率平滑名义时间 (s), 平滑弧长 = SMOOTH_VEL×SMOOTH_TIME |
| SMOOTH_VEL | 0.025 | 名义平滑速度 (m/s, = base×gait×scale @gait=1) |

平滑无直行最小间距 = 2×x_sw_half ≈ 0.1009 m（有直行下限 2×x_sw_full ≈ 0.1260 m，见 path.py `_get_smooth_xsw`）。


## 命令系统 (mdp/command.py)

5D 命令张量 `[vel_x, height_f, height_h, gait_freq, curvature]`：

| 字段 | Phase 0 | Phase 1 | 更新 |
|------|---------|---------|------|
| vel_x | `base*gait*(1-0.75*|κ|/20)` | `base*gait*scale(CURVATURE_TARGET)`, episode 内固定 | reset |
| height_f/h | 固定 0.055 | 固定 0.055 | 周期性 |
| gait_freq | U(1.0,2.0) 随机, ep内固定 | U(1.0,2.0) 随机, ep内固定 | reset |
| curvature | 课程采样/fixed_curvature | 每步 `get_path_curvature()` ±20↔0 (平滑 LUT) | Phase0: reset; Phase1: 每步 |

Phase 1 每步动态更新：`_update_command` 中 curvature 跟随路径瞬时值（平滑 LUT 的过渡 κ）；**velocity 固定**（reset 时按 CURVATURE_TARGET 计算，episode 内不变，不再随 κ 动态缩放）。


## 期望轨迹 (mdp/path.py)

机器人初始接近段为**圆弧**（不再是直行），与主轨迹曲率一致。Phase 0 路径从世界原点 `(0,0)` 出发；Phase 1 周期起点 = 第一根杆正上方 `(spacing, 0)`（杆1 在 `(spacing, POLE_Y)`，杆2 在 `(2×spacing, POLE_Y)`）。

**Phase 0**：接近段(圆弧, κ=curvature 反推 `_INIT_DIST`) → 圆弧(弦长公式, 从原点出发, 固定 κ)

**Phase 1**：接近段(圆弧, 平台-K+过渡-K→0 反推) → **平滑 LUT** 绕杆路径。有直行模式一个周期 5 段：CW弧→CCW弧→**直行(2×straight)**→CCW弧→CW弧，其中**直行段中点 = 第二根杆的 x 坐标**（机器人从杆1 正上方出发 → 绕杆1 → 直行穿过杆2 正下方 → 绕杆2 → 到杆3 正上方）；无直行模式 4 段同向连续。弧曲率 = ±CURVATURE_TARGET，且**所有曲率跳变线性过渡**（平滑弧长 = SMOOTH_VEL×SMOOTH_TIME，与 vel 解耦，几何固定）。弧长 = vel×t − _INIT_DIST（vel 固定），跨周期 x_ref 叠加偏移保证连续。

### 平滑 LUT 的双模式

- **有直行模式**：间距 ≥ 2×x_sw_full (0.1260)，弧↔直行均有过渡；直行段合并为一段（两杆之间），长度 = 2×(spacing−2×x_sw_full)，中点对齐下一根杆
- **无直行模式**：间距 ≤ 2×x_sw_half (0.1009)，同向弧段 (S2→S4, S5→S1) **直接连续**（无 +20→0→+20 的 V 形过渡）
- **不兼容区间** (0.1009, 0.1260)：两种模式都无法周期匹配 → `get_effective_pole_spacing` 自动 clamp 到无直行最小间距（并打印警告）
- 有效间距经 `active_pole_spacing` 统一处理，保证周期位移恒 = 2×有效间距（无累积误差）

### 接近段圆弧

从 LUT 起点 `(spacing, 0)` 反推 `_INIT_DIST` 弧长生成接近段轨迹表（`_approach_rev_table`）：正向接近段 = 平台(-K) + 过渡(-K→0)，终点 `(spacing, 0)` κ=0、heading=0，与 LUT 进过渡衔接。机器人起点/姿态由 `get_approach_start(spacing)` / `get_phase1_approach(spacing)`（Phase 1）或 `get_phase0_approach`（Phase 0，随 κ 动态）确定。


## 预计算表 (mdp/reference.py)

脊柱关节角度由瞬时曲率 κ 驱动：

```
f_spine1 = -0.65 × κ/κ_max     (κ=-max→+0.65, κ=+max→-0.65)
f_body   = -0.9 × κ/κ_max
h_spine1 = -0.65 × |κ|/κ_max   (始终≤0)
h_body   = -0.7 × κ/κ_max
```

Phase 0: κ=静态命令值; Phase 1: κ=`get_path_curvature()` 动态读取 LUT 瞬时值。

腿部参考由 CSV (Trot_F/H) + 逆运动学生成，26 曲率×50 相位×14 关节预计算表，运行时按曲率插值。内侧腿 Y 轨迹按 `1-|κ|/κ_max` 缩放实现差速（κ=0 全步幅, κ=κ_max 全停）。


## 奖励函数 (mdp/rewards.py)

| 奖励项 | 功能 |
|--------|------|
| mimic_pos/vel | 关节位置/速度模仿 (腿+脊柱+颈分σ) |
| height | 身体高度跟踪 |
| track_vel | body-frame 前进速度 (含侧向/垂向 vyz) |
| track_head | 朝向跟踪 (track_omg 已停用, 见 env_cfg) |
| corridor | 身体包络走廊约束 (核心) |
| action_L1/L2 | 动作平滑性 |
| energy | 能耗惩罚 |

### 走廊一致性奖励

```
e_i = |d_lat| + |L·sin(Δθ)| + |W·cos(Δθ)|
v_i = max(0, e_i - C)
r = exp(-σ·v²)
```

同时惩罚位置和朝向偏差，死区 C 内不扣分。相比传统 track_path 自带朝向耦合。


## 杆模块 (mdp/pole.py)

MuJoCo 圆柱体，`POLE_Y = -get_smooth_x_sw()` ≈ **-0.063**（平滑路径等效圆心，不再等于 -Rmin=-0.05）。训练全程 `contype=0` (无物理碰撞)。透明度由 `update_pole_visibility()` 按 `geom_type==CYLINDER` 匹配控制。杆数量 `POLE_NUM=12`。


## 回放脚本 (scripts/SQuRo_play.py)

提取 `train_iter` → `align_iter = max(0, train_iter-10)`
- 避免运行 3999 轮的策略时误加载到 Phase1(4000) 
- 自动识别 Phase：`align_iter < 4000 → Phase 0`，否则 Phase 1
- Phase 0：设 `fixed_curvature`，Phase 1：设 `fixed_pole_spacing` 并清除 `fixed_curvature`
- Phase 1 用正确间距重建杆实体，可视化需要与期望路径一致

CSV 记录 关节角度、速度、动作空间输出等信息，并保存视频



# 踩坑记录


## 坐标系

1. **body+X ≠ 物理前向**：F/H body 的 body+X 指向 world ±Y，需分别 ±π/2 修正
2. **不要假设对称身体环节有相同局部坐标系**：F_body 和 H_body 的 XML 局部四元数不同
3. **使用模型未验证的局部坐标系需要验证朝向**：


## 奖励设计

1. **世界系速度奖励导致侧滑作弊** → 改为 body-frame 前进速度投影
2. **路径参考跟随机器人位置** → 改为固定世界原点
3. **track_path 无朝向约束** → 替换为 corridor 奖励
4. **脊柱参考静态 vs 绕杆动态** → Phase 1 用 `get_path_curvature()` 动态读取 LUT 瞬时 κ
5. **vel 动态缩放导致参考超前** → Phase 1 速度改回固定（reset 时按 CURVATURE_TARGET 计算, episode 内不变）
6. **变速时弧长需积分** → vel 固定后改用 `vel×t - _INIT_DIST`（无需积分）


## 训练

10. **Entropy 崩溃** → body-frame 速度 + corridor 组合约束
11. **Phase 1 step函数 κ 跳跃** → 改为 LUT 圆弧拼接 (CW/CCW 弧+直行)
12. **杆间距硬编码 0.3** → 统一到 `POLE_SPACING = 2/CURVATURE_TARGET`
13. **curvature ±15 硬编码** → 统一到 `CURVATURE_TARGET`
14. **平滑 LUT 几何随 vel 变化**（平滑弧长 = vel×SMOOTH_TIME）→ 同一间距下不同步频轨迹几何不同, 训练局部最优 → **平滑弧长与 vel 解耦**（固定名义 vel, 几何稳定）
15. **平滑不兼容区间** (0.1009, 0.1260) → 两种模式均周期错位 → `get_effective_pole_spacing` clamp 到无直行最小间距
16. **无直行时同向弧段仍加 +20→0→+20 过渡** → 直行段不存在时 V 形过渡多余 → 无直行模式同向弧段直接连续
17. **Phase 0 起点动态化**（随 κ 变化）→ 起始域扩大, 起步即转 → 需配合起始状态分布评估


## 可视化 / WarpBridge

1. **WarpBridge 不兼容 mj_name2id** → 改用 `geom_type==CYLINDER` 匹配
2. **model.geom_rgba 是 torch tensor** → 赋值需 `torch.tensor`
3. **entities 覆盖丢失实体** → 用 `{**orig, **pole}` merge
4. **可视化用机器人位置做起点** → 改用固定世界原点 (-_INIT_DIST, 0)
5. **可视化杆重复显示**（红球/圆柱 + 场景杆实体）→ 可视化不再绘制杆, 由场景 PoleEntity 提供（回放时按正确间距重建）


## 变量管理

1. **全局变量**：如果某一个变量在多个文件中重复出现且被硬编码，需要在一个文件中定义为全局变量并在其他文件中导入，避免后续修改时遗漏。例如 SQuRo_Slalom\mdp\curriculums.py 文件中的

```
PHASE1_MID_ITER = 2000       # iter 0-2000: 转弯曲率增大
PHASE1_END_ITER = 4000       # iter 2000-4000: 转弯基元
PHASE2_MID_ITER = 6000       # iter 4000-6000: 绕杆间距缩小阶段
PHASE2_END_ITER = 8000       # iter 6000-8000: 绕杆训练
```

2. **索引**：如关节索引、site点索引，避免**硬编码**，统一在 mdp/indices 中进行获取，并导入相关索引，需要避免**循环导入**


## 执行器

1. **动作空间维度变化**：如果涉及动作空间维度变化，如新增 Neck_yaw/Neck_pitch 需更新相关任务以及回放脚本，特别注意更新全部索引常量和预计算参考表的维度、索引
