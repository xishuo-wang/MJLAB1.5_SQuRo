# SQuRo Slalom — 脊柱型四足机器人绕杆强化学习项目

## 项目概述

基于 MuJoCo + MJLAB 框架，研究脊柱型微小四足机器人 (SQuRo) 的脊腿协同运动，实现从转弯基元训练到连续绕杆的两阶段课程学习。

**核心创新**：走廊一致性 (Corridor Conformance) 奖励框架，将机器人身体建模为两个铰接矩形 (F_body + H_body)，通过可调节宽度的走廊约束统一处理 XY 平面绕杆和 YZ 平面钻洞两种场景。

## 机器人模型

### SQuRo 身体结构

脊柱为万向节机构：F_body + H_body (扭转, 轴∥X), F_spine1 (侧摆, 初始轴∥Z), H_spine1 (俯仰, 初始轴∥Y)。头部新增 Neck_yaw + Neck_pitch。

14 个执行器，按 entity actuator 顺序：

| 序号 | 名称 | 关节 | 功能 |
|------|------|------|------|
| 0 | F_spine1 | F_spine1_joint | 侧摆(偏航) |
| 1 | F_body | F_body_joint | 扭转 |
| 2 | Neck_yaw | Neck_yaw_joint | 头部偏航 |
| 3 | Neck_pitch | Neck_pitch_joint | 头部俯仰 |
| 4-5 | FL_shoulder/elbow | — | 左前腿 |
| 6-7 | FR_shoulder/elbow | — | 右前腿 |
| 8 | H_spine1 | H_spine1_joint | 俯仰 |
| 9 | H_body | H_body_joint | 扭转 |
| 10-11 | HL_hip/knee | — | 左后腿 |
| 12-13 | HR_hip/knee | — | 右后腿 |

Action 张量中的分组索引常量 (`indices.py`)：

```python
_ACTION_SPN_IDS  = (0, 1, 8, 9)       # 身体脊柱
_ACTION_NECK_IDS = (2, 3)             # 颈部
_ACTION_LEG_IDS  = (4,5,6,7,10,11,12,13)  # 腿部
```

### 坐标系关键知识

**body+X ≠ 物理前向**，SQuRo body frame 与世界坐标系有 90° 旋转偏移：

```
世界坐标系:  前=+X, 左=+Y, 上=+Z
F_body_Link: body+X → world +Y (heading=+90°)
H_body_Link: body+X → world -Y (heading=-90°)
```

物理前向 heading 修正量符号相反：

```python
f_body_physical = body_X_heading - π/2  # +90° → 0° (world +X)
h_body_physical = body_X_heading + π/2  # -90° → 0° (world +X)
```

验证方法：F_spine1=0 时两者均应≈0°。F_spine1>0(右转)→F_body偏右(负),H_body偏左(正),呈C形。

### 身体尺寸常量 (`path.py`)

| 常量 | 值 | 含义 |
|------|-----|------|
| F_BODY_HALF_LENGTH | 0.04 m | F_body 半长(沿身体轴) |
| F/H_BODY_HALF_WIDTH | 0.035 m | 半宽(左右方向) |
| BODY_REF_OFFSET | 0.04 m | F/H 中心距 base 偏移 |
| CORRIDOR_HALF_WIDTH | 0.04 m | 走廊半宽(基元阶段) |
| _RMIN | 1/15 ≈ 0.0667 m | 最小转弯半径 (κ=±15) |

## 任务架构：SQuRo_Slalom

### 文件结构

```
src/mjlab/tasks/SQuRo_Slalom/
├── SQuRo_Slalom_env_cfg.py   # 环境配置
├── config/
│   ├── __init__.py            # 任务注册
│   └── rl_cfg.py              # PPO 超参数
├── rl/
│   └── runner.py              # ONNX 导出
├── mdp/
│   ├── command.py             # 5D 命令张量
│   ├── curriculums.py         # 两阶段课程
│   ├── events.py              # reset_model
│   ├── indices.py             # 14 执行器索引
│   ├── observations.py        # 观测空间 (~105 维)
│   ├── path.py                # 路径生成 + 走廊奖励 + 动态曲率
│   ├── pole.py                # 杆实体
│   ├── reference.py           # CSV+IK 参考轨迹表
│   ├── rewards.py             # 奖励函数
│   └── terminations.py        # 终止条件
```

关键脚本：
- `scripts/SQuRo_play.py` — 策略回放 (自动 Phase 0/1 识别)
- `scripts/slalom_path_viz.py` — 绕杆路径可视化对比

### 命令系统 (`command.py`)

5D 命令张量 `[vel_x, height_f, height_h, gait_freq, curvature]`：

| 字段 | Phase 0 | Phase 1 | 更新频率 |
|------|---------|---------|----------|
| vel_x | `base*(1-0.75*|κ|/20)` | `base*(1-0.75*15/20) ≈ 0.044` | 每ep reset |
| height_f/h | 固定 0.055 | 固定 0.055 | 周期性 |
| gait_freq | 固定 1.0 | 固定 1.0 | 周期性 |
| curvature | 课程采样/fixed_curvature | 每步动态 `get_path_curvature()` | Phase 0: reset; Phase 1: 每步 |

### 路径生成 (`path.py`)

```
compute_path_ref(env)  ← 所有观测/奖励统一入口
  ├─ slalom_mode_active=False → compute_arc_path_ref (圆弧, 固定世界原点)
  └─ slalom_mode_active=True  → compute_slalom_path_ref (LUT 查表)
```

**绕杆 LUT** 一个周期 6 段 `(0,0)→(2X,0)`：
1. CW ¼弧: (0,0)→(Rmin,-Rmin), κ=-15
2. CCW ¼弧: →(2Rmin,-2Rmin), κ=+15
3. 直行: →(X,-2Rmin), κ=0
4. CCW ¼弧: →(X+Rmin,-Rmin), κ=+15
5. CW ¼弧: →(X+2Rmin,0), κ=-15
6. 直行: →(2X,0), κ=0

路径跨周期连续通过 `x_ref += num_periods * 2*spacing` 保证。

### 脊柱参考 (`reference.py`)

脊柱关节角度由瞬时曲率 κ 驱动：

```
f_spine1 = -0.6 × κ/κ_max     (κ=-max→+0.6, κ=+max→-0.6)
f_body   = -0.9 × κ/κ_max
h_spine1 = -0.6 × |κ|/κ_max   (始终≤0)
h_body   = -0.7 × κ/κ_max
```

Phase 0: κ=静态命令值; Phase 1: κ=`get_path_curvature()` 动态读取 LUT 瞬时值。

腿部参考由 CSV (Trot_F/H) + 逆运动学生成，16 曲率×50 相位×14 关节预计算表，运行时按曲率插值。内侧腿 Y 轨迹按 `|κ|` 缩放实现差速。

### 两阶段自动课程 (`curriculums.py`)

所有阶段边界统一在 `curriculums.py` 定义，`command.py` 通过导入引用：

```
iter:   0 ─── 2000 ─── 4000 ─── 6000 ─── 8000
        PHASE1_MID  PHASE1_END  PHASE2_MID  PHASE2_END
        κ 0.5→20    基元→绕杆   间距缩窄    (预留)
        ├── Phase 0: 转弯基元 ──┤├── Phase 1: 绕杆 ──┤

Phase 0: 圆弧路径, κ 课程增长 0.5→20, 杆透明
Phase 1: LUT 路径, 杆间距 0.20→0.15m, 杆可见(无碰撞)
```

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

### 奖励函数 (`rewards.py`)

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

## 杆模块 (`pole.py`)

MuJoCo 圆柱体：直径 1cm, 高 10cm, Y 坐标 `POLE_Y = -1/15 ≈ -0.0667m (= -Rmin)`。

透明度控制：通过 `update_pole_visibility()` 按 `geom_type == CYLINDER` 匹配 WarpBridge 模型修改 alpha。Phase 0 透明，Phase 1 可见。训练全程 `contype=0` (无物理碰撞)。

## 回放脚本 (`SQuRo_play.py`)

- 提取 `train_iter` → `align_iter = max(0, train_iter-10)`
- 自动识别 Phase：`align_iter < 4000 → Phase 0`，否则 Phase 1
- Phase 0：设 `fixed_curvature`，Phase 1：设 `fixed_pole_spacing` 并清 `fixed_curvature`
- Phase 1 用正确间距重建杆实体，可视化与路径一致
- CSV 记录 F_body/H_body 的 raw+物理前向 heading

## 踩坑记录

### 坐标系

1. **body+X ≠ 物理前向**：F/H body 的 body+X 指向 world ±Y，需分别 ±π/2 修正
2. **不要假设对称身体环节有相同局部坐标系**：F_body 和 H_body 的 XML 局部四元数不同

### 奖励设计陷阱

3. **世界系速度奖励导致侧滑作弊**：改为 body-frame 前进速度投影
4. **路径参考跟随机器人位置** (动态走廊)：改为固定世界原点参考
5. **track_path 无朝向约束**：替换为 corridor 奖励 (eff_hw 项耦合朝向)
6. **脊柱参考静态 vs 绕杆动态**：Phase 1 改为 `get_path_curvature()` 动态读取
7. **TROT_FREQ 硬编码**：改为从 `gait_freq_command` 动态读取

### 训练

8. **Entropy 崩溃 (-20)**：打滑捷径 → 需要 body-frame 速度 + corridor 组合约束
9. **路径周期换行跳变**：`x_ref` 叠加周期偏移
10. **vel_command 硬编码 ×0.5**：改为 Phase 0 同款曲率缩放公式

### 可视化 / WarpBridge

11. **WarpBridge 不兼容 mj_name2id**：改用 `geom_type == CYLINDER` 匹配
12. **model.geom_rgba 是 torch tensor**：赋值需 `torch.tensor`，非 numpy
13. **entities 覆盖丢失实体**：用 `{**orig, **pole}` merge
14. **杆 Y 坐标不一致**：统一为 `POLE_Y = -1/15`

### 阶段管理

15. **STAGE2_END 分散定义**：统一到 `curriculums.py` 的 `PHASE*` 常量
16. **Phase 边界回放不一致**：`is_slalom_phase` 用 `align_iter` 而非 `train_iter`
17. **play/train 杆重复分支**：简化为默认占位 + play 脚本覆盖

### 执行器

18. **动作空间 12→14 维**：新增 Neck_yaw/Neck_pitch 需更新全部索引常量和参考表维度
19. **target_names_expr 缺少逗号**：`H_body_joint` 后缺 `,` 导致拼接两个字符串
