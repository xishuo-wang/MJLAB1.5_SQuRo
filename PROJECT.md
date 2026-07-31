# 脊柱型四足机器人强化学习项目

## 项目概述

基于 MuJoCo + MJLAB 框架，研究脊柱型微小四足机器人 (SQuRo) 的脊腿协同运动。当前聚焦 SQuRo_Slalom 子任务（连续绕杆），SQuRo_Hole（钻洞）为后续子任务。

**核心创新**：走廊一致性 (Corridor Conformance) 奖励框架 — 将机器人身体建模为两个铰接矩形 (F_body + H_body)，通过可调节宽度的走廊约束统一处理 XoY 平面绕杆和 YoZ 平面钻洞两种场景。

## 机器人模型

### SQuRo 脊柱结构

万向节机构，从前到后：F_body(扭转) → F_spine1(偏航) → H_spine1(俯仰) → H_body(扭转)。腿部 2 自由度，仅在 XoZ 平面内运动。头部新增 Neck_yaw + Neck_pitch。

14 个执行器，通过 `mdp/indices.py` 统一管理索引：

| 序号 | 名称 | 功能 | 分组 |
|------|------|------|------|
| 0-1 | F_spine1, F_body | 脊柱(侧摆+扭转) | `actuator_spn_ids` |
| 2-3 | Neck_yaw, Neck_pitch | 颈部 | `actuator_neck_ids` |
| 4-7 | FL/FR shoulder/elbow | 前腿 | `actuator_leg_ids` |
| 8-9 | H_spine1, H_body | 脊柱(俯仰+扭转) | `actuator_spn_ids` |
| 10-13 | HL/HR hip/knee | 后腿 | `actuator_leg_ids` |

### 坐标系

**世界**：+X=前, +Y=左, +Z=上。机器人初始 X=-0.05, Y=0, 面朝 +X。

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

## SQuRo_Slalom 任务

### 文件结构

```
src/mjlab/tasks/SQuRo_Slalom/
├── SQuRo_Slalom_env_cfg.py    # 环境配置
├── config/
│   ├── __init__.py            # 任务注册
│   └── rl_cfg.py              # PPO 超参数 (网络 512→256→128, 1024 envs)
├── mdp/
│   ├── command.py             # 5D 命令张量 + 两阶段采样
│   ├── curriculums.py         # 课程常量 (阶段边界/曲率/步频/杆间距/奖励权重)
│   ├── events.py              # reset_model (初始位置 X=-0.05)
│   ├── indices.py             # 14 执行器索引
│   ├── observations.py        # 观测空间 (~105维)
│   ├── path.py                # 期望路径生成 + 走廊奖励 + 碰撞检测
│   ├── pole.py                # 杆实体 (POLE_Y/POLE_RADIUS/可见性)
│   ├── reference.py           # CSV+IK 预计算表 (16曲率×50相位×14关节)
│   ├── rewards.py             # 奖励函数 (10项)
│   └── terminations.py        # 终止条件
```

脚本：`scripts/SQuRo_play.py` — 策略回放，自动 Phase 0/1 识别。

### 两阶段课程 (`curriculums.py`)

所有常量在 `curriculums.py` 唯一定义，其他模块导入引用：

```
iter:   0 ─── 2000 ─── 4000 ─── 6000 ─── 8000
        PHASE1_MID  PHASE1_END  PHASE2_MID  PHASE2_END
        κ 0.5→20    基元→绕杆   (预留)     (预留)
        ├── Phase 0: 转弯基元 ──┤├── Phase 1: 绕杆 ──┤

Phase 0: 圆弧路径, κ 课程增长, 步频 1~2Hz 随机, 杆透明
Phase 1: LUT 路径, 杆间距固定 2×Rmin, 步频固定 1Hz, 杆可见(无碰撞)
```

| 常量 | 值 | 用途 |
|------|-----|------|
| CURVATURE_MIN | 0.5 | Phase 0 κ 起始值 |
| CURVATURE_TARGET_MAX | 20.0 | Phase 0 κ 上限 |
| CURVATURE_TARGET | 16.0 | Phase 1 绕杆弧曲率 (= 1/Rmin) |
| GAIT_FREQ_MIN/MAX | 1.0/2.0 | Phase 0 步频采样范围 |
| GAIT_FREQ_PHASE1 | 1.0 | Phase 1 固定步频 |
| POLE_SPACING | 2/CURVATURE_TARGET | Phase 1 杆间距 |

### 命令系统 (`command.py`)

5D 命令张量 `[vel_x, height_f, height_h, gait_freq, curvature]`：

| 字段 | Phase 0 | Phase 1 | 更新 |
|------|---------|---------|------|
| vel_x | `base*gait*(1-0.75*|κ|/20)` | 同左, 每步动态缩放 | 每步 |
| height_f/h | 固定 0.055 | 固定 0.055 | 周期性 |
| gait_freq | U(1.0,2.0) 随机, ep内固定 | 固定 1.0 | Phase0: reset |
| curvature | 课程采样/fixed_curvature | 每步 `get_path_curvature()` ±16↔0 | Phase0: reset; Phase1: 每步 |

Phase 1 每步动态更新：`_update_command` 中 curvature 跟随路径瞬时值、velocity 按曲率比例缩放。

### 期望轨迹 (`path.py`)

机器人初始 `X=-0.05, Y=0`，面朝 `+X`。路径统一从世界原点 `(0,0)` 出发，前 0.05m 为直行接近段（`_APPROACH_DIST`）。

**Phase 0**：接近段(直行) → 圆弧(弦长公式, 从原点出发, 固定 κ)

**Phase 1**：接近段(直行) → LUT 绕杆路径。LUT 一个周期 6 段 `(0,0)→(2X,0)`：CW弧→CCW弧→直行→CCW弧→CW弧→直行。弧曲率 = ±CURVATURE_TARGET。弧长累积积分 `∫vel dt` 适配变速，跨周期 x_ref 叠加偏移保证连续。

### 脊柱参考 (`reference.py`)

预计算表：16 曲率 × 50 相位 × 14 关节。CSV (Trot_F/H) + IK → 腿部关节角。内侧腿 Y 轨迹按 |κ| 缩放实现差速。

脊柱关节由瞬时曲率驱动，Phase 0 静态、Phase 1 动态通过 `get_path_curvature()` 读取 LUT 瞬时 κ。步频 `gait_freq` 从命令动态读取。

### 奖励函数

| 奖励项 | 功能 |
|--------|------|
| mimic_pos/vel | 关节位置/速度模仿 (腿+脊柱+颈分σ) |
| height | 身体高度跟踪 |
| track_vel | body-frame 前进速度 |
| track_omg/head | 角速度/朝向跟踪 |
| corridor | 身体包络走廊约束 (核心) |
| action_L1/L2 | 动作平滑性 |
| energy | 能耗惩罚 |
| collision | 虚拟碰撞 (躯干+腿 vs 杆) |

### 走廊一致性奖励

```
e_i = |d_lat| + |L·sin(Δθ)| + |W·cos(Δθ)|
v_i = max(0, e_i - C)
r = exp(-σ·v²)
```

同时惩罚位置和朝向偏差，死区 C 内不扣分。相比传统 track_path 自带朝向耦合。

### 杆模块 (`pole.py`)

MuJoCo 圆柱体，`POLE_Y = -1/CURVATURE_TARGET`。训练全程 `contype=0` (无物理碰撞)。透明度由 `update_pole_visibility()` 按 `geom_type==CYLINDER` 匹配控制。

### 回放脚本

提取 `train_iter` → `align_iter = max(0, train_iter-10)` → Phase 0/1 自动识别。Phase 0 设 `fixed_curvature`，Phase 1 设 `fixed_pole_spacing` 并清 `fixed_curvature`。Phase 1 用正确间距重建杆实体。

## 踩坑记录

### 坐标系

1. **body+X ≠ 物理前向**：F/H body 的 body+X 指向 world ±Y，需分别 ±π/2 修正
2. **不假设对称身体环节有相同局部坐标系**：F_body 和 H_body 的 XML 局部四元数不同
3. **新模型须验证每个 body link 的朝向**

### 奖励设计

4. **世界系速度奖励导致侧滑作弊** → 改为 body-frame 前进速度投影
5. **路径参考跟随机器人位置** → 改为固定世界原点
6. **track_path 无朝向约束** → 替换为 corridor 奖励
7. **脊柱参考静态 vs 绕杆动态** → Phase 1 用 `get_path_curvature()` 动态读取 LUT 瞬时 κ
8. **vel 固定 vs 曲率动态** → Phase 1 速度每步随曲率缩放
9. **路径弧长 vel×t 不适用变速** → 改为累积积分 `∫ vel dt`

### 训练

10. **Entropy 崩溃** → body-frame 速度 + corridor 组合约束
11. **Phase 1 step函数 κ 跳跃** → 改为 LUT 圆弧拼接 (CW/CCW 弧+直行)
12. **杆间距硬编码 0.3** → 统一到 `POLE_SPACING = 2/CURVATURE_TARGET`
13. **curvature ±15 硬编码** → 统一到 `CURVATURE_TARGET`

### 可视化 / WarpBridge

14. **WarpBridge 不兼容 mj_name2id** → 改用 `geom_type==CYLINDER` 匹配
15. **model.geom_rgba 是 torch tensor** → 赋值需 `torch.tensor`
16. **entities 覆盖丢失实体** → 用 `{**orig, **pole}` merge
17. **可视化用机器人位置做起点** → 改用固定世界原点 (-0.05, 0)
18. **杆标记载体从红球改为圆柱** → 与场景 PoleEntity 外观一致

### 变量管理

19. **全局常量分散定义** → 统一到 `curriculums.py` (阶段/曲率/步频/杆间距)
20. **索引硬编码** → `indices.py` 统一管理 14 执行器索引

### 循环引用

21. **curriculums↔command 循环导入** → `CURVATURE_TARGET_MAX` 移至 curriculums 唯一定义

### 阶段管理

22. **Phase 边界回放不一致** → `is_slalom_phase` 用 `align_iter` 而非 `train_iter`
23. **play/train 杆重复分支** → 简化为默认占位 + play 脚本覆盖
