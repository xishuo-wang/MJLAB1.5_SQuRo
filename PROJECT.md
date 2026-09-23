# 脊柱型四足机器人强化学习项目

## 项目概述

基于 MuJoCo + MJLAB 框架，研究脊柱型微小四足机器人 (SQuRo) 的脊腿协同运动。

| 子任务 | 任务 ID | 状态 | 文档位置 |
| --- | --- | --- | --- |
| **SQuRo_Backup**（仰卧翻正） | `Mjlab-SQuRo-Backup` | **当前主线**：翻正已学会，站立抖动已修复并验收 | `docs/SQuRo_Backup_技术细节.md` / `_奖励对照.md` / `_修改清单.md` |
| **SQuRo_Slalom**（连续绕杆） | `Mjlab-SQuRo-Slalom` | 已实现，见本文档下半部分 | 本文档 |
| **SQuRo_Tunnel**（钻洞/越障·参考表路线） | `Mjlab-SQuRo-Tunnel` | Phase0 参考层已迁移，一阶段训练中（0~3k） | `docs/SQuRo_Tunnel_技术细节.md` |
| **SQuRo_Hole**（钻洞/越障·虚拟碰撞路线） | `Mjlab-SQuRo-Hole` | 2026-07 旧版在 mjlab 1.5 上完整恢复 + 四阶段课程，待重训验证 | `docs/SQuRo_Hole_技术细节.md` |

**核心创新**：走廊一致性 (Corridor Conformance) 奖励框架 — 将机器人身体建模为两个铰接矩形 (F_body + H_body)，通过可调节宽度的走廊约束统一处理 XoY 平面绕杆和 YoZ 平面钻洞两种场景。

---

## SQuRo 机器人模型

XML 路径：`src/mjlab/asset_zoo/robots/SQuRo/xmls/SQuRo.xml`

从前到后大致分为：头部、前肢躯干及腿部、脊柱、后肢躯干及腿部。
腿部具有两个自由度，无法实现内收/外展运动，只能在 XoZ 平面内运动。

### 脊柱结构

脊柱是一个类似万向节的机构，从前往后有四个关节：`F_body_joint`、`F_spine1_joint`、`H_spine1_joint`、`H_body_joint`。

- `F_body_joint` 与 `H_body_joint` 为**扭转**自由度（旋转轴平行于世界坐标系 X 轴）
- `F_spine1_joint` 初始为**偏航**自由度（初始旋转轴平行于世界坐标系 Z 轴）
- `H_spine1_joint` 初始为**俯仰**自由度（初始旋转轴平行于世界坐标系 Y 轴）

随着扭转，F_spine1 与 H_spine1 的旋转轴也会变化，但两个旋转轴始终保持正交。

### 执行器与索引

14 个执行器，通过 `mdp/indices.py` 统一管理索引：

| 序号 | 名称 | 功能 | 分组 |
|------|------|------|------|
| 0-1 | F_spine1, F_body | 脊柱(侧摆+扭转) | `actuator_spn_ids` |
| 2-3 | Neck_yaw, Neck_pitch | 颈部 | `actuator_neck_ids` |
| 4-7 | FL/FR shoulder/elbow | 前腿 | `actuator_leg_ids` |
| 8-9 | H_spine1, H_body | 脊柱(俯仰+扭转) | `actuator_spn_ids` |
| 10-13 | HL/HR hip/knee | 后腿 | `actuator_leg_ids` |

### 坐标系

**世界**：+X=前, +Y=左, +Z=上。

**局部：body+X ≠ 物理前向。** SQuRo (Mouse) 的 body frame 与世界系有 90° 旋转偏移：

```
世界坐标系:  前=+X, 左=+Y, 上=+Z
SQuRo body frame (base_Link & F_body_Link):
  body+X → world +Y (=物理左, heading=90°)
  body+Y → world +X (base_Link 时=物理前, F_body 时=world -Z!)
  body+Z → world -Z
```

因此 `atan2` 提取的 heading 是 body+X 在 world XY 平面的方向(=90°)，**不代表物理前向(0°)**，所有依赖"前进方向"的投影必须修正：

```python
forward_heading = body_X_heading - torch.pi / 2   # body+X→world+Y，物理前向→world+X
forward_speed = vel_x * cos(forward_heading) + vel_y * sin(forward_heading)
```

**F 段与 H 段修正方向相反**（两者 XML 局部四元数不同）：

```python
# 初始状态（面朝 world +X, 脊柱角=0）: F_body_Link body+X→+Y(+90°), H_body_Link body+X→-Y(-90°)
f_body_physical = get_body_heading(env, f_body_id) - torch.pi / 2
h_body_physical = get_body_heading(env, h_body_id) + torch.pi / 2
```

**验证方法**：F_spine1=0 时 F/H 的物理前向 heading 都应 ≈0°；F_spine1>0（右转）时 F 偏右（负）、H 偏左（正），身体呈 C 形。

**脊柱弯曲时不要用 `base_Link`**：它处于中心会受两侧拉扯而震荡，速度/角速度/朝向跟踪一律以 `F_body_Link` 为准。

**翻正任务额外要求**：姿态判据不能用四元数或重力投影（`base_up` 是万向节中心朝向，动作中会自己转过 85°~180°），必须用 `mdp/indices.py` 的 `SEGMENT_BELLY_BACK_SITES`（腹/背标记 site 的世界坐标差）。详见 `docs/SQuRo_Backup_技术细节.md` §2~§4。

### 身体尺寸（Slalom `path.py`）

| 常量 | 值 | 含义 |
|------|-----|------|
| F/H_BODY_HALF_LENGTH | 0.04 m | 半长(沿身体轴) |
| F/H_BODY_HALF_WIDTH | 0.035 m | 半宽(左右方向) |
| BODY_REF_OFFSET | 0.04 m | F/H 中心距 base 偏移 |
| CORRIDOR_HALF_WIDTH | 0.04 m | 走廊半宽 |
| _RMIN | 1/CURVATURE_TARGET | 最小转弯半径 |

---

# SQuRo_Backup 任务（仰卧翻正）

先用**手调状态机的翻正轨迹**做参考，通过模仿奖励让 RL 学会翻正，再优化行为质量。
参考实现：`src/mjlab/scripts/SQuRo_Backup_Replay.py`（手调状态机，λ=1/3 均可零重试通过）。

**该任务的完整文档不在本文件**，避免重复维护：

| 文档 | 内容 |
| --- | --- |
| `docs/SQuRo_Backup_技术细节.md` | 参考手册：当前配置、坐标系与判据陷阱、地面真值、时序、成功判据设计与标定、验收方法 |
| `docs/SQuRo_Backup_奖励对照.md` | 实验日志：每次改动的问题定位、改动内容、实测数据（按时间追加） |
| `docs/SQuRo_Backup_修改清单.md` | 状态与计划：已完成/待办、已撤回方案与教训 |

常用入口：

```powershell
uv run train Mjlab-SQuRo-Backup --agent.logger tensorboard --agent.max-iterations 3000
uv run python -B -m mjlab.scripts.Backup.verify_backup_stage_rewards      # 主回归 (21 项)
uv run python -B -m mjlab.scripts.Backup.verify_backup_config             # 配置一致性
uv run python -B -m mjlab.scripts.SQuRo_Backup_Replay --visualize none    # 手调参考回放
```

---

# SQuRo_Slalom 任务（连续绕杆）

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

整体分为两阶段：Phase 0 转向基元训练 + Phase 1 绕杆场景训练。每个 Phase 又分若干小阶段，
用于关键变量（转向曲率、杆间距）的课程学习以及奖励权重课程。所有阶段边界统一在 `curriculums.py` 定义，
其余如 `command.py` 通过导入引用：

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
| BASE_VEL | 0.1 | 基础速度 (m/s, 1Hz 时) |
| VEL_MIN | 0.15 | 最大曲率下速度缩放 (转弯低速) |
| STRAIGHT_VEL_SCALE | 0.5 | 直行段速度缩放 (= 直行基础速度 0.05 / BASE_VEL) |

平滑无直行最小间距 = 2×x_sw_half ≈ 0.1009 m（有直行下限 2×x_sw_full ≈ 0.1260 m，见 `path.py` `_get_smooth_xsw`）。

## 命令系统 (mdp/command.py)

5D 命令张量 `[vel_x, height_f, height_h, gait_freq, curvature]`：

| 字段 | Phase 0 | Phase 1 | 更新 |
|------|---------|---------|------|
| vel_x | `base*gait*(1-0.75*|κ|/20)` | **变速**: `base*gait*(STRAIGHT_VEL_SCALE−(STRAIGHT_VEL_SCALE−VEL_MIN)·|κ|/25)`, 直行 0.05×gait / 弯道 0.015×gait, 每步更新 | Phase0: reset; Phase1: 每步 |
| height_f/h | 固定 0.055 | 固定 0.055 | 周期性 |
| gait_freq | U(1.0,2.0) 随机, ep 内固定 | U(1.0,2.0) 随机, ep 内固定 | reset |
| curvature | 课程采样/fixed_curvature | 每步 `get_path_curvature()` ±20↔0 (平滑 LUT) | Phase0: reset; Phase1: 每步 |

Phase 1 每步动态更新：`_update_command` 中 curvature 跟随路径瞬时值（平滑 LUT 的过渡 κ）；
**velocity 变速**——直行段（κ=0）scale=STRAIGHT_VEL_SCALE(0.5)、弯道（κ=±25）scale=VEL_MIN(0.15)，
线性过渡，每步随 κ 更新。回放 `fixed_velocity` 作为基础速度同样变速。

## 期望轨迹 (mdp/path.py)

机器人初始接近段为**圆弧**（不再是直行），与主轨迹曲率一致。Phase 0 路径从世界原点 `(0,0)` 出发；
Phase 1 周期起点 = 第一根杆正上方 `(spacing, 0)`（杆1 在 `(spacing, POLE_Y)`，杆2 在 `(2×spacing, POLE_Y)`）。

**Phase 0**：接近段(圆弧, κ=curvature 反推 `_INIT_DIST`) → 圆弧(弦长公式, 从原点出发, 固定 κ)

**Phase 1**：接近段(圆弧, 平台-K+过渡-K→0 反推) → **平滑 LUT** 绕杆路径。有直行模式一个周期：
直行(杆1 上方, L/2) → CW弧→CCW弧(绕杆1) → 直行(杆2 下方, L) → CCW弧→CW弧(绕杆2) → 直行(杆3 上方, L/2)，
其中**每根杆均位于其直行段正中间**（直行段长度 L = spacing−2×x_sw_full，杆1/杆3 的直行段横跨周期边界）；
无直行模式 4 段同向连续。弧曲率 = ±CURVATURE_TARGET，且**所有曲率跳变线性过渡**
（平滑弧长 = SMOOTH_VEL×SMOOTH_TIME，与 vel 解耦，几何固定）。**变速弧长推进**：
期望速度 v=base×gait×scale(κ) 随曲率变化，预计算 t(s) 表（t_i += Δs_i/v_i）精确积分，
运行时 t→s 查表（直行段快、弯道慢），跨周期按周期时间 t_period 推进，x_ref 叠加偏移保证连续。

### 平滑 LUT 的双模式

- **有直行模式**：间距 ≥ 2×x_sw_full (0.1260)，弧↔直行均有过渡；**直行段以杆为中心**——每根杆位于其直行段正中间（长度 L = spacing−2×x_sw_full），杆1/杆3 的直行段横跨周期边界、杆2 的直行段完整在周期内
- **无直行模式**：间距 ≤ 2×x_sw_half (0.1009)，同向弧段 (S2→S4, S5→S1) **直接连续**（无 +20→0→+20 的 V 形过渡）
- **不兼容区间** (0.1009, 0.1260)：两种模式都无法周期匹配 → `get_effective_pole_spacing` 自动 clamp 到无直行最小间距（并打印警告）
- 有效间距经 `active_pole_spacing` 统一处理，保证周期位移恒 = 2×有效间距（无累积误差）

### 接近段圆弧

从 LUT 起点 `(spacing, 0)` 反推 `_INIT_DIST` 弧长生成接近段轨迹表（`_approach_rev_table`）：
正向接近段 = 平台(-K) + 过渡(-K→0)，终点 `(spacing, 0)` κ=0、heading=0，与 LUT 进过渡衔接。
机器人起点/姿态由 `get_approach_start(spacing)` / `get_phase1_approach(spacing)`（Phase 1）
或 `get_phase0_approach`（Phase 0，随 κ 动态）确定。

## 预计算表 (mdp/reference.py)

脊柱关节角度由瞬时曲率 κ 驱动：

```
f_spine1 = -0.65 × κ/κ_max     (κ=-max→+0.65, κ=+max→-0.65)
f_body   = -0.9 × κ/κ_max
h_spine1 = -0.65 × |κ|/κ_max   (始终≤0)
h_body   = -0.7 × κ/κ_max
```

Phase 0: κ=静态命令值; Phase 1: κ=`get_path_curvature()` 动态读取 LUT 瞬时值。

腿部参考由 CSV (Trot_F/H) + 逆运动学生成，26 曲率×50 相位×14 关节预计算表，运行时按曲率插值。
内侧腿 Y 轨迹按 `1-|κ|/κ_max` 缩放实现差速（κ=0 全步幅, κ=κ_max 全停）。

## 奖励函数 (mdp/rewards.py)

| 奖励项 | 功能 |
|--------|------|
| mimic_pos/vel | 关节位置/速度模仿 (腿+脊柱+颈分 σ) |
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

MuJoCo 圆柱体，`POLE_Y = -get_smooth_x_sw()` ≈ **-0.063**（平滑路径等效圆心，不再等于 -Rmin=-0.05）。
训练全程 `contype=0`（无物理碰撞）。透明度由 `update_pole_visibility()` 按 `geom_type==CYLINDER` 匹配控制。
杆数量 `POLE_NUM=12`。

## 回放脚本 (scripts/SQuRo_play.py)

提取 `train_iter` → `align_iter = max(0, train_iter-10)`

- 避免运行 3999 轮的策略时误加载到 Phase1(4000)
- 自动识别 Phase：`align_iter < 4000 → Phase 0`，否则 Phase 1
- Phase 0：设 `fixed_curvature`；Phase 1：设 `fixed_pole_spacing` 并清除 `fixed_curvature`
- Phase 1 用正确间距重建杆实体，可视化需要与期望路径一致

CSV 记录关节角度、速度、动作空间输出等信息，并保存视频。

## 曲率命令设计

命令格式 `[vel, h_f, h_h, freq, curvature]`, ω_cmd = curvature × vel_cmd。
曲率 κ = 1/R (signed): κ>0=左转, κ<0=右转, κ=0=直行。
每次 episode reset 时采样一次，episode 内保持不变——让 agent 针对固定转弯半径学习稳态步态。

脊柱参考角: F_spine1 = clamp(-GAIN × ω_cmd, ±0.6), GAIN = L/v = 2.0。
符号: 实测 F_body_heading = base_heading - F_spine1，左转需 F_spine1<0 → 取负号。

---

# SQuRo_Tunnel 任务（钻洞/越障）

直线走廊内连续通过多个门洞式限高障碍（板底离地 0.050 m，洞宽 0.03 m），前躯干（含头部）
与后躯干分别按沿 X 的方波期望高度压低后通过。沿用 Slalom 框架（5D 命令 + 预计算参考表 +
走廊一致性奖励），曲率恒为 0，高度变化来自方波轨迹而非转向。

**该任务的完整文档不在本文件**，避免重复维护：

| 文档 | 内容 |
| --- | --- |
| `docs/SQuRo_Tunnel_技术细节.md` | 参考手册：洞几何与偏移、走廊判据口径、奖励权重、Phase0/Phase1 速度规则、回放与诊断、踩坑与未决问题 |

常用入口：

```powershell
uv run train Mjlab-SQuRo-Tunnel --agent.logger tensorboard --agent.max-iterations 4000
uv run python -B -m mjlab.scripts.SQuRo_Tunnel_play --checkpoint_file <ckpt>   # 回放
uv run python -B -m mjlab.scripts.SQuRo_Tunnel_play --agent zero --smoke_steps 50 --no-video   # 无窗自检
uv run python -B -m mjlab.scripts.Viz_Path.Viz_Tunnel_Path                    # 期望高度轨迹图
```

---

# 踩坑记录

> 通用编码约定（常量唯一管理处、索引不硬编码、动作维度变化的联动清单）见 `AGENTS.md`。

## 坐标系

1. **body+X ≠ 物理前向**：F/H body 的 body+X 指向 world ±Y，需分别 ±π/2 修正（详见上文坐标系一节）
2. **不要假设对称身体环节有相同局部坐标系**：F_body 和 H_body 的 XML 局部四元数不同
3. **使用模型未验证的局部坐标系前必须验证朝向**：诊断代码见 `SQuRo_constants.py` 的 `__main__` 块
4. **翻正任务禁用四元数/重力投影判姿态**：`base_up` 是万向节中心朝向，动作中会自己转过 85°~180°；
   改用腹/背标记 site 的世界坐标（`docs/SQuRo_Backup_技术细节.md` §3~§4）

## 奖励设计（Slalom）

1. **世界系速度奖励导致侧滑作弊** → 改为 body-frame 前进速度投影
2. **路径参考跟随机器人位置** → 改为固定世界原点
3. **track_path 无朝向约束** → 替换为 corridor 奖励
4. **脊柱参考静态 vs 绕杆动态** → Phase 1 用 `get_path_curvature()` 动态读取 LUT 瞬时 κ
5. **vel 动态缩放导致参考超前** → Phase 1 速度改回固定（reset 时按 CURVATURE_TARGET 计算, episode 内不变）
6. **变速时弧长需积分** → 现 Phase 1 重新启用变速（直行快/弯道慢），改用**预计算 t(s) 表**精确积分（t_i += Δs_i/v_i），避免 `vel×t` 参考超前

## 训练（Slalom）

10. **Entropy 崩溃** → body-frame 速度 + corridor 组合约束
11. **Phase 1 step 函数 κ 跳跃** → 改为 LUT 圆弧拼接 (CW/CCW 弧+直行)
12. **杆间距硬编码 0.3** → 统一到 `POLE_SPACING = 2/CURVATURE_TARGET`
13. **curvature ±15 硬编码** → 统一到 `CURVATURE_TARGET`
14. **平滑 LUT 几何随 vel 变化**（平滑弧长 = vel×SMOOTH_TIME）→ 同一间距下不同步频轨迹几何不同, 训练局部最优 → **平滑弧长与 vel 解耦**（固定名义 vel, 几何稳定）
15. **平滑不兼容区间** (0.1009, 0.1260) → 两种模式均周期错位 → `get_effective_pole_spacing` clamp 到无直行最小间距
16. **无直行时同向弧段仍加 +20→0→+20 过渡** → 直行段不存在时 V 形过渡多余 → 无直行模式同向弧段直接连续
17. **Phase 0 起点动态化**（随 κ 变化）→ 起始域扩大, 起步即转 → 需配合起始状态分布评估

## 可视化 / WarpBridge（Slalom）

1. **WarpBridge 不兼容 mj_name2id** → 改用 `geom_type==CYLINDER` 匹配
2. **model.geom_rgba 是 torch tensor** → 赋值需 `torch.tensor`
3. **entities 覆盖丢失实体** → 用 `{**orig, **pole}` merge
4. **可视化用机器人位置做起点** → 改用固定世界原点 (-_INIT_DIST, 0)
5. **可视化杆重复显示**（红球/圆柱 + 场景杆实体）→ 可视化不再绘制杆, 由场景 PoleEntity 提供（回放时按正确间距重建）

## Tunnel（钻洞）

1. **起点高度与期望高度不一致**：`reset_model` 放 z=0.06, 而站立体心 z≈0.0563、期望高度 0.055。
2. **走廊判据无上界**：`e` 里先加段半高 → 死区为 0，只约束"中心 vs 期望高度 ±2 cm"，
   机体上沿与限高板下沿没有硬约束，靠 σ=1000 的高度奖励压低。
3. **限高板全程无碰撞**（`contype=0`）：穿洞成功没有判据，回放需 `--enable_collision True` 才检验。
4. **洞 0.03 m 宽装不下 0.07 m 宽的躯干**：N=1 的洞是理想化抽象，按"限高门洞"理解。
5. **`HEIGHT_LOW=0.02` 可行性未标定**：站立 0.0563 → 低高度 0.02 是 3.6 cm 下降。

详细口径、奖励权重与回放方法见 `docs/SQuRo_Tunnel_技术细节.md`。
