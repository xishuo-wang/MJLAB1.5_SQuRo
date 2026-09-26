# 脊柱型四足机器人强化学习项目

## 项目概述

基于 MuJoCo + MJLAB 框架，研究脊柱型微小四足机器人 (SQuRo) 的脊腿协同运动。

| 子任务 | 任务 ID | 状态 | 文档位置 |
| --- | --- | --- | --- |
| **SQuRo_Backup**（仰卧翻正） | `Mjlab-SQuRo-Backup` | **当前主线**：翻正已学会，站立抖动已修复并验收 | `docs/SQuRo_Backup_技术细节.md` / `_奖励对照.md` / `_修改清单.md` |
| **SQuRo_Slalom**（连续绕杆） | `Mjlab-SQuRo-Slalom` | 已实现；Phase 0 已验收，Phase 1 参考层已修复待训练验证 | 本文档 |
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
Phase 1: 平滑 LUT 路径, 杆间距确定性课程(0.20→最小间距 2Rmin=0.08 线性缩小, iter 4000→6000), 步频 1~2Hz 随机, 杆可见(无碰撞)
```

| 常量 | 值 | 用途 |
|------|-----|------|
| CURVATURE_MIN | 0.5 | Phase 0 κ 起始值 |
| CURVATURE_TARGET_MAX | 25.0 | Phase 0 κ 上限 |
| CURVATURE_TARGET | 25.0 | Phase 1 绕杆弧曲率 (= 1/Rmin, 与 CURVATURE_TARGET_MAX 相同) |
| GAIT_FREQ_MIN/MAX | 1.0/2.0 | Phase 0 步频采样范围 |
| GAIT_FREQ_PHASE1 | 1.0 | 仅作初始值 (Phase 1 实际用 1~2Hz 随机采样) |
| POLE_SPACING_START | 0.20 | Phase 1 起始杆间距 (iter 4000 前固定) |
| POLE_SPACING_MIN | 0.08 | Phase 1 课程终点 (= 2Rmin; 实际被有效间距下界 0.0812 截住, 见下) |
| SMOOTH_TIME | 1.0 | 曲率平滑名义时间 (s), 平滑弧长 = SMOOTH_VEL×SMOOTH_TIME |
| SMOOTH_VEL | 0.025 | 名义平滑速度 (m/s, = base×gait×scale @gait=1) |
| BASE_VEL | 0.1 | 基础速度 (m/s, 1Hz 时) |
| VEL_MIN | 0.15 | 最大曲率下速度缩放 (转弯低速) |
| STRAIGHT_VEL_SCALE | 0.5 | 直行段速度缩放 (= 直行基础速度 0.05 / BASE_VEL) |

Rmin=0.04、tr=SMOOTH_VEL×SMOOTH_TIME=0.025 下实测：x_sw_half=0.04059、x_sw_full=0.05312，
故**无直行最小间距 = 2×x_sw_half ≈ 0.0812 m**、**有直行下限 = 2×x_sw_full ≈ 0.1062 m**
（见 `path.py` `_get_smooth_xsw`）。区间 (0.0812, 0.1062) 两种模式都无法周期匹配，**不可用**。
课程据此分两段：iter 4000→`POLE_SPACING_RAMP_END_ITER`(5500) 在 [0.1062, 0.20] 内线性缩小，
到 5500 一次性切到 0.0812 并保持到终点；几何边界由 `get_pole_spacing_bounds()`（内部惰性引用
`path.py`）给出，**课程值恒落在可行区间**，不再依赖 `get_effective_pole_spacing` 的静默 clamp。

## 命令系统 (mdp/command.py)

5D 命令张量 `[vel_x, height_f, height_h, gait_freq, curvature]`：

| 字段 | Phase 0 | Phase 1 | 更新 |
|------|---------|---------|------|
| vel_x | `base*gait*(1-0.85*|κ|/25)` | **变速**: `base*gait*(STRAIGHT_VEL_SCALE−(STRAIGHT_VEL_SCALE−VEL_MIN)·|κ|/25)`, 直行 0.05×gait / 弯道 0.015×gait, 每步更新 | Phase0: reset; Phase1: 每步 |
| height_f/h | 固定 0.055 | 固定 0.055 | 周期性 |
| gait_freq | U(1.0,2.0) 随机, ep 内固定 | U(1.0,2.0) 随机, ep 内固定 | reset |
| curvature | 课程采样/fixed_curvature | 每步 `get_path_curvature()` ±25↔0 (平滑 LUT) | Phase0: reset; Phase1: 每步 |

Phase 1 每步动态更新：`_update_command` 中 curvature 跟随路径瞬时值（平滑 LUT 的过渡 κ）；
**velocity 变速**——直行段（κ=0）scale=STRAIGHT_VEL_SCALE(0.5)、弯道（κ=±25）scale=VEL_MIN(0.15)，
线性过渡，每步随 κ 更新。回放 `fixed_velocity` 作为基础速度同样变速。

**计时器归基类管**：`CommandTerm.compute` 已扣 `time_left` 并在到期时重采样，任务侧不得再扣
（曾经重复扣减，使 20~30 s 的重采样提前到 10~15 s，并在回合中途把别的重置批次的步频写进当前回合）。

**阶段按回合起点锁定**：路径选择与**命令更新**都用逐环境的 `phase1_mask =
get_training_phase_batch(common_step_counter - episode_length_buf)`，即"本回合开始时刻"的阶段，
回合内不翻转；`compute_path_ref` 对圆弧/绕杆两条路径逐环境取并集，`get_path_curvature` 统一读它写入的
`_path_kappa`；`_update_command` 也按该掩码逐环境写曲率/速度（未结束的旧阶段回合保持重置时写入的命令，
否则阶段切换瞬间会改掉它们的速度并让圆弧参考后跳）。`slalom_mode_active` 仍是全局口径，
只用于可视化与课程间距。

**重置边界**：框架在同一步内的顺序是"算终止步奖励 → 重置 → 算观测"，期间 `common_step_counter`
不变。因此参考缓存键含**重置代次**（`reference.invalidate_reference_cache`，由命令项 `reset` 调用），
重置后同一控制步内强制重算，首帧观测拿到的才是新回合的参考；`_ref_phase` 推进、κ 历史写入、
重置相位清零都加了"每步只做一次"的保护，重算不会二次推进。参考所用的瞬时曲率取自本步的
`compute_path_ref(t)`，与参考位置同源（重置后 t=0，故取接近段起点曲率），
Phase 1 重置时写入的初始曲率也按接近段起点取（有直行 -K / 无直行 0）。

## 期望轨迹 (mdp/path.py)

机器人初始接近段为**圆弧**（不再是直行），与主轨迹曲率一致。Phase 0 路径从世界原点 `(0,0)` 出发；
Phase 1 周期起点 = 第一根杆正上方 `(spacing, 0)`（杆1 在 `(spacing, POLE_Y)`，杆2 在 `(2×spacing, POLE_Y)`）。

**Phase 0**：接近段(圆弧, κ=curvature 反推 `_INIT_DIST`) → 圆弧(弦长公式, 从原点出发, 固定 κ)

**Phase 1**：接近段(圆弧, 平台-K+过渡-K→0 反推) → **平滑 LUT** 绕杆路径。有直行模式一个周期：
直行(杆1 上方, L/2) → CW弧→CCW弧(绕杆1) → 直行(杆2 下方, L) → CCW弧→CW弧(绕杆2) → 直行(杆3 上方, L/2)，
其中**每根杆均位于其直行段正中间**（直行段长度 L = spacing−2×x_sw_full，杆1/杆3 的直行段横跨周期边界）；
无直行模式 4 段同向连续。弧曲率 = ±CURVATURE_TARGET，且**所有曲率跳变线性过渡**
（平滑弧长 = SMOOTH_VEL×SMOOTH_TIME，与 vel 解耦，几何固定）。

接近段与主路径的衔接按模式对齐：**有直行模式**接近段 = 平台(-K)+过渡(-K→0)，终点 κ=0 接 LUT 首段直行；
**无直行模式**接近段 = 过渡(0→-K)+平台(-K)，终点 κ=-K 接 LUT 首段 -K 平台
（`_approach_rev_table(..., platform_at_end=is_no_straight_spacing(spacing))`），两种模式下 κ 与
期望速度都连续。

**名义时间 τ 推进（步频只作时间缩放）**：期望速度 v=base×gait×scale(κ)，故
`dt = dτ/gait`，其中 **τ(s) = ∫ ds/(base×scale(κ))** 与步频无关。预计算 τ(s) 表
（τ_i += Δs_i/v_nom,i，v_nom 取 gait=1 名义速度），运行时每个环境取 **τ = t×gait_自身**
查表（直行段快、弯道慢），跨周期按 τ_period 推进，x_ref 叠加偏移保证连续。
这样参考推进速度 `ds/dt = v_nom×gait` 恰好等于该环境自己的 `vel_x` 命令，
且表只随 (杆间距, 基础速度) 变化 —— 任一环境重抽步频不会再改动其他环境的参考。
gait=1 时 τ=t，与旧实现逐点等价。

### 平滑 LUT 的双模式

- **有直行模式**：间距 ≥ 2×x_sw_full (0.1062)，弧↔直行均有过渡；**直行段以杆为中心**——每根杆位于其直行段正中间（长度 L = spacing−2×x_sw_full），杆1/杆3 的直行段横跨周期边界、杆2 的直行段完整在周期内
- **无直行模式**：间距 ≤ 2×x_sw_half (0.0812)，同向弧段 (S2→S4, S5→S1) **直接连续**（无 +25→0→+25 的 V 形过渡）
- **不兼容区间** (0.0812, 0.1062)：两种模式都无法周期匹配 → `get_effective_pole_spacing` 自动落到无直行最小间距
- 有效间距经 `active_pole_spacing` 统一处理（`fixed_pole_spacing` 覆盖值同样过这一步），
  `events.reset_model` 的出生点直接读该属性，保证**间距只有一个真源**

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

腿部参考为 **2 内侧 × 31 曲率×50 相位×14 关节** 预计算表（0=左腿为内侧，1=右腿为内侧），
运行时按 `sign(κ)` 选表、按 `|κ|` 插值；内侧腿 Y 轨迹按 `1-|κ|/κ_max` 缩放实现差速
（κ=0 全步幅, κ=κ_max 全停），**每条腿保持自身相位**（曾用"交换关节列"实现内外侧，
因左右腿带半周期相位差而等价于整条腿延后半周期，κ 过零时跳变 1.20 rad，已改为选表）。
颈部同样由 κ 驱动（`neck_yaw = 0.8×κ/κ_max`、`neck_pitch = -0.3`）。
脊柱/颈参考限位取 `SQuRo.xml` 的关节 range 与执行器 ctrlrange 较小者（F/H_spine1 ±0.6、neck_yaw ±0.8），
超出部分截断，截断处速度参考为 0。
**速度参考 = 位置参考的解析时间导数**：κ̇ 由相邻步差分（回合首步置零），脊柱/颈为解析导数、
腿部补表沿曲率轴的导数 × |κ|̇ —— 位置模仿与速度模仿因此不再互相矛盾。

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

MuJoCo 圆柱体，`POLE_Y = -get_smooth_x_sw()` ≈ **-0.0531**（平滑路径等效圆心，不等于 -Rmin=-0.04）。
训练全程 `contype=0`（无物理碰撞）。透明度由 `update_pole_visibility()` 按 `geom_type==CYLINDER` 匹配控制。
杆数量 `POLE_NUM=12`。**杆实体位置在 env_cfg 里按 POLE_SPACING_START 一次性建好且从不移动**，
课程缩间距时场景里的杆与实际参考路径不一致（无碰撞，故只影响可视化；回放脚本会按正确间距重建）。

## 回放脚本 (scripts/SQuRo_Slalom_play.py)

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
（注：`reference.py` 实际按 `-0.65×κ/κ_max` 生成且未做 clamp，与本节的 ±0.6 不一致，见预计算表一节。）

## 常用入口

```powershell
uv run train Mjlab-SQuRo-Slalom --agent.logger tensorboard          # 训练 (预算 = PHASE2_END_ITER)
uv run python -B -m mjlab.scripts.Slalom.verify_slalom_ref_consistency   # 参考一致性回归 (8 项)
uv run python -B -m mjlab.scripts.SQuRo_Slalom_play --checkpoint_file <ckpt>   # 回放
```

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
6. **变速时弧长需积分** → 现 Phase 1 重新启用变速（直行快/弯道慢），改用**预计算表**精确积分（τ_i += Δs_i/v_nom,i），避免 `vel×t` 参考超前
7. **逐环境量被塞进全局槽** → 参考的 τ(s) 表原先按全局标量 `_slalom_gait_scalar` 建，而它由最后一次重置批次写入；
   1024 环境稳态下约每步就有 1 个环境重置，于是表几乎每步被别的环境的步频重建，**未重置环境的参考随之跳变**
   （桩复现 4.42 cm），参考推进速度也与该环境自己的速度命令不符 → 改成**每环境 τ = t×gait_自身**，
   表只随 (间距, 基础速度) 变化
8. **计时器被扣两次** → `CommandTerm.compute` 与任务的 `_update_command` 各扣一次 `time_left`，
   20 ms 控制步扣掉约 40 ms，配置的 20~30 s 重采样提前到 10~15 s；那次中途重采样会把别的重置批次的
   共享步频写进当前回合，破坏"回合内步频固定" → 任务侧不再碰计时器

## 训练（Slalom）

10. **Entropy 崩溃** → body-frame 速度 + corridor 组合约束
11. **Phase 1 step 函数 κ 跳跃** → 改为 LUT 圆弧拼接 (CW/CCW 弧+直行)
12. **杆间距硬编码 0.3** → 统一到 `POLE_SPACING = 2/CURVATURE_TARGET`
13. **curvature ±15 硬编码** → 统一到 `CURVATURE_TARGET`
14. **平滑 LUT 几何随 vel 变化**（平滑弧长 = vel×SMOOTH_TIME）→ 同一间距下不同步频轨迹几何不同, 训练局部最优 → **平滑弧长与 vel 解耦**（固定名义 vel, 几何稳定）
15. **平滑不兼容区间** (0.0812, 0.1062) → 两种模式均周期错位 → `get_effective_pole_spacing` 落到无直行最小间距
16. **无直行时同向弧段仍加 +25→0→+25 过渡** → 直行段不存在时 V 形过渡多余 → 无直行模式同向弧段直接连续
17. **Phase 0 起点动态化**（随 κ 变化）→ 起始域扩大, 起步即转 → 需配合起始状态分布评估
18. **课程有效间距断崖** → `POLE_SPACING_MIN=2Rmin=0.08` 低于几何下界 0.0812，且 (0.0812, 0.1062) 区间不可周期匹配，
    于是 iter 5563 有效间距从 0.1063 直接跳到 0.0812，其后到 6000 轮空转 →
    改为两段课程（斜坡到 2×x_sw_full，再在 `POLE_SPACING_RAMP_END_ITER` 一次性切到 2×x_sw_half）
19. **默认预算练不到终点** → `max_iterations` 原为 5000 而课程到 6000 才结束 → 改为直接引用 `PHASE2_END_ITER`
20. **参考位置动而速度说不动** → 脊柱/颈位置参考随 κ 变，但对应的 `ref_vel` 仍为 0（表里脊柱列全零），
    位置模仿要求转动、速度模仿同时奖励静止 → 速度参考改为位置参考的解析时间导数
21. **腿部参考在 κ 过零时整条腿延后半周期** → 用"交换左右腿关节列"实现内/外侧，而左右腿带半周期相位差，
    κ=±1e-6 时最大跳变 1.20 rad (68.6°)，绕杆每周期触发两次 → 改为按 `sign(κ)` 选左内/右内两张表
22. **脊柱参考超出模型限位** → F/H_spine1 参考峰值 0.65 rad > XML 关节 range 与 ctrlrange 的 ±0.6，
    位置模仿在该处不可达 → 按 XML 限位截断（截断处速度参考为 0）
23. **无直行模式接近段与主路径衔接处速度阶跃** → 接近段终点恒为 κ=0，而无直行 LUT 首段是 κ=-K 平台，
    衔接处 κ 硬跳 25、期望速度从 0.05 掉到 0.015 m/s → 接近段按模式选末端曲率（`platform_at_end`）
24. **阶段切换切到正在进行的回合** → 路径选择读全局计数器，iter 4000 时在跑的回合被切到绕杆路径 →
    改用按回合起点锁定的 `phase1_mask`，回合内不翻转
25. **回合首步的重置速度尖峰** → 新回路的 κ̇ 若按上一回合末尾的 κ 差分，会给出几十 rad/s 的假速度参考 →
    `episode_length_buf <= 1` 时 κ̇ 置零
26. **命令更新漏了逐环境阶段** → 阶段锁定只改了路径选择，`_update_command` 仍按全局
    `slalom_mode_active` 写全部环境：iter 4000 越过边界时未结束的 Phase 0 回合被套用 Phase 1 变速公式，
    速度 0.10 → 0.05 m/s，圆弧参考（按 `vel*t`）在回合进行到 10 s 时后跳约 0.5 m →
    `_update_command` 也按 `phase1_mask` 逐环境写
27. **重置首帧观测命中上一回合的参考缓存** → 框架在同一步内"先算奖励、再重置、再算观测"，
    `common_step_counter` 不变，而缓存只按步数判有效；曲率由 +20 重采样为 -20 时首帧前躯参考仍是
    -0.72 rad（应为 +0.72，差 1.44 rad）。相位清零并不能解决（相位清了缓存还在）→
    缓存键加入**重置代次**（`invalidate_reference_cache`，命令项 reset 调用），
    相位推进 / κ 历史 / 清零改为每步一次，且参考的 κ 取自本步 `compute_path_ref(t)` 与位置同源
    （顺带消除了模仿奖励比走廊奖励晚一步取 κ 的问题）

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
