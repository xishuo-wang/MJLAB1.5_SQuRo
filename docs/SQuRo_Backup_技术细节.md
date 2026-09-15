# SQuRo_Backup（翻正）任务技术细节

本文档记录 SQuRo 翻正任务在开发过程中**反复踩到、且不查就会重复踩**的技术细节。
主要覆盖：坐标系与姿态判据的陷阱、动作时序、状态机门控、以及已验收的地面真值测量方法。

针对的是 `src/mjlab/tasks/SQuRo_Backup/` 与 `src/mjlab/scripts/SQuRo_backup_Replay.py`。

---

## 1. 结论速查

| 事项 | 结论 |
|---|---|
| `base_up`（root 的 `projected_gravity_b[2]`） | **不能**用于判断翻正。它是万向节脊柱中心的朝向，动作过程中会自己转过 ~85°~180° |
| 判断翻正必须 | **分别判断前肢段（F_body）与后肢段（H_body）** 的背腹朝向 |
| `data.body_link_quat_w` | 实测**不是**世界系（reset 后 `root_link_quat_w` 为单位四元数），但**参考系尚未最终定论**，见 §3 |
| 可靠的姿态测量 | 用 XML 中 `*_body_belly_site` / `*_body_back_site` 的**世界坐标差**，见 §4 |
| F 段与 H 段的局部坐标 | **背腹轴正方向相反**：F 的腹面在局部 +Y，H 的腹面在局部 −Y |
| 状态机 S1 / S2 | S1 = 后段已翻正、前段未翻；S2 = 前后段都已翻正。**与 §5 的时序完全对得上** |

---

## 2. 坐标系陷阱（核心）

### 2.1 机器人构造

```
世界系: +X=前, +Y=左, +Z=上
脊柱(万向节)自前向后:  F_body_joint → F_spine1_joint → H_spine1_joint → H_body_joint
                        (扭转)          (侧摆)           (俯仰)          (扭转)
base_Link 位于脊柱正中, 是万向节中心, 不属于前三段中的任何一段
```

四个脊柱关节的行程与角色：

| 关节 | actuator idx | 行程 | 基线期角色 | 翻正期角色 |
|---|---|---|---|---|
| `F_spine1_joint` | 0 | ±0.6 | 侧摆（XoY） | 辅助 |
| `F_body_joint` | 1 | ±1.57 | 扭转 | **主导** |
| `H_spine1_joint` | 8 | ±0.6 | 俯仰（YoZ） | 辅助 |
| `H_body_joint` | 9 | ±1.57 | 扭转 | **主导** |

### 2.2 `base_up` 为什么不能用作翻正判据

`projected_gravity_b[2]` 返回的是 `base_Link` 本地 +Z 在世界 Z 上的投影。实测（λ=1 手调回放）：

| t (s) | t_nom | phase | `base_up` | 实际状态 |
|---|---|---|---|---|
| 0.01 | 0.01 | P1 | **−1.000** | 仰面躺 |
| 0.65 | 0.65 | P1 | **+0.098** | 仍仰面躺，但 base 被脊柱形变带着转了 ~85° |
| 0.80 | 0.80 | P2 | **−0.038** | 仰面躺 |
| 1.13 | 1.13 | P3 | **+1.000** | 站立 |

`base_up` 在 t=0.01→0.65 之间从 −1 冲到 +0.10，而**此时前后肢都还没有翻**。
原因：T1 段把两个扭转关节反向拧满 ±90°，这个形变把脊柱中心（base）带转了 90°，
但前后肢本身仍在原位。所以 `base_up` 与"是否翻正"完全脱钩。

### 2.3 前/后段的局部坐标相反

两个 body 的网格在各自局部系的 Y 轴跨度都是 **0.0445 m**（即背腹厚度），但**正方向相反**：

| body | 局部 Y 范围 | 腹面所在侧 |
|---|---|---|
| `F_body_Link` | [−0.0235, +0.0210] | **+Y** |
| `H_body_Link` | [−0.0210, +0.0235] | **−Y** |

这就是"对称的身体环节局部坐标系不同"这一踩坑在翻正任务中的具体表现。
用带符号的标量统一表达两段朝向时，**H 段必须补一个负号**。
`command.py` 里 `_body_up(body_id, sign)` 的 `sign` 参数就是干这个的。

---

## 3. `body_link_quat_w` 参考系问题（未完全定论，谨慎使用）

### 3.1 已确认的事实

1. reset 后 `root_link_quat_w == [1, 0, 0, 0]`（单位四元数）。而 XML 中 `base_Link` 带
   `quat="0 -0.707107 -0.707107 0"`（≈绕 XY 对角线 180°），若该量是世界系，这里应是
   `(−0.7071, −0.7071, 0, 0)`。→ **`root_link_quat_w` 不是世界朝向。**
2. 用 `p_world = p_base + R(root_link_quat_w) · offset_body` 复现 `body_link_pos_w` 时：
   H 段残差 ~4e-4 m（可认为刚性成立），F 段残差最大 ~0.027 m（F 链多段被动关节形变导致）。
   → `root_link_quat_w` 能解释 H 段的位置关系。
3. `data.py` 中 `body_link_quat_w` 的 docstring 写的是 "in world frame"，**与上述实测不符**。

### 3.2 未能解决的部分（记录以免重复排查）

曾尝试用解析式验证 `_body_up = 2(yz + wx)` 的物理含义，结论**矛盾**：

| t | 关节角 θF = F_sp1 + F_body | sin(θF) 解析预测 | `rawF` 实测 |
|---|---|---|---|
| 0.01 | +0.003 | +0.003 | **+1.000** |
| 0.41 | −1.010 | −0.847 | +0.796 |
| 0.81 | −1.568 | −1.000 | +0.993 |

`rawF` 在 θF 变化 ±1 rad 期间几乎不变（+1.000 → +0.977），**说明它不反映关节角**。
因此"`_body_up ≡ sin(θ_s + θ_b)`"这一说法**是错的**，不要在后续分析中沿用。

同时 `q_root ⊗ q_rel` 复合后的结果与实测 `rawF` 在整段运动中也对不上。
**结论：`body_link_quat_w` 的参考系未被确定，不要基于它做物理判断。**

→ 绕过方案见 §4。所有姿态判据改用标记 site 的世界坐标。

### 3.3 奖励量纲陷阱：`Episode_Reward/*` 是"每秒速率"，不是每步值

`managers/reward_manager.py` 中：

```python
value = term_cfg.func(...) * term_cfg.weight * dt          # 逐步累计时已乘 dt
extras["Episode_Reward/" + key] = episodic_sum_avg / max_episode_length_s   # 除以回合秒数
```

因此 **`Episode_Reward/x` = 该项的"每秒平均奖励"**，要还原单回合总量须再乘 `max_episode_length_s`（本任务 10 s）。

**踩坑**：训练日志里的 `Mean reward` 是 PPO 侧的统计量，与环境单回合回报量纲不同；
拿它直接和"自己逐步累加的回报"比较会得到约 10 倍的假差异。
比较不同策略时必须统一口径（建议都统计 `reset` 时刻的 `Episode_Reward/*`，或用同样的累加方式）。

---

## 4. 地面真值：标记 site（已验收）

### 4.1 新增内容

在 `src/mjlab/asset_zoo/robots/SQuRo/xmls/SQuRo.xml` 中为两个躯干段各加一对标记：

```xml
<!-- F_body_Link 内 -->
<site name="F_body_belly_site" pos="0.00  0.021 -0.023" size="0.0015" rgba="0 1 1 1"/>
<site name="F_body_back_site"  pos="0.00 -0.023 -0.023" size="0.0015" rgba="1 1 0 1"/>

<!-- H_body_Link 内 -->
<site name="H_body_belly_site" pos="0.00 -0.021 0.008" size="0.0015" rgba="0 1 1 1"/>
<site name="H_body_back_site"  pos="0.00  0.023 0.008" size="1.5e-3" rgba="1 1 0 1"/>
```

位置取在各 body 网格 Y 轴的两个极值面上（F: +0.021 / −0.023；H: −0.021 / +0.023），
两个 site 的世界坐标差即为该段的**背腹轴矢量**。

### 4.2 判据定义

```python
# d = p(belly_site) - p(back_site)   世界系矢量, 由 MuJoCo 直接给出
# 正置度 = -(d_z / |d|)              +1 = 正置(腹面朝下, 已翻正)
#                                    -1 = 倒置(腹面朝上)
```

**取负号的原因**：仰面躺时腹面朝上，`d_z/|d| = +1`；本任务语境下"正置"记 +1 更直观，
故整体取负。已由 FK 独立验证：reset 时 `belly_site` 的 z = 0.04487，
`back_site` 的 z = 0.00087，腹面确实朝上。

**此量不依赖任何四元数约定**，是当前唯一可信的姿态判据。

### 4.3 三个锚点验收结果（λ=1 手调回放）

| 锚点 | 时刻 | 预期 (F, H) | 实测 (F, H) | 状态 |
|---|---|---|---|---|
| **初始** | t=0.01 | (−1, −1) | (**−1.000**, **−1.000**) | ✓ 两段都倒置 |
| **T2 末** | t=0.80 (P1→P2) | (−1, +1) | (**−0.999**, **+0.997**) | ✓ 前段未翻 / 后段已翻 |
| **T3 末** | t=1.05 (P2→P3) | (+1, +1) | (**+0.938**, **+0.936**) | ✓ 两段都已翻 |

验收脚本：`src/mjlab/scripts/Backup/measure_segment_gravity_truth.py`

### 4.4 完整姿态时间线（λ=1，地面真值）

| t (s) | phase | F 段 | H 段 | 说明 |
|---|---|---|---|---|
| 0.01 | P1 | 倒置 | 倒置 | 仰面躺 |
| 0.25 | P1 | 倒置 | 侧立 | **后段开始翻** |
| 0.41 | P1 | 倒置 | **正置** | 后段已翻正，前段不动 |
| 0.65 | P1 | 倒置 | 正置 | T1 末（扭转拧满） |
| 0.80 | **P1→P2** | 倒置 | 正置 | **T2 末 = S1** |
| 0.97 | P2 | 正置 | 侧立 | **前段开始翻**（翻滚发生在此段） |
| 1.05 | **P2→P3** | 正置 | 正置 | **T3 末 = S2** |
| 1.13+ | P3 | 正置 | 正置 | 稳定，站起 |

**要点**：后段的翻正发生在 **T1**，前段的翻正发生在 **T3**，两段不同时。

---

## 5. 动作时序

### 5.1 手调脚本的名义时序（λ=1，`slow1_target`）

| 段 | 名义时间 | F_spine1 | F_body | H_spine1 | H_body | 动作含义 |
|---|---|---|---|---|---|---|
| **T1 起转** | 0 → 0.65 | 0 → +0.60 | 0 → **−1.57** | 0 → +0.60 | 0 → **+1.57** | 两个扭转关节反向拧满 90°，侧摆/俯仰同步加到 0.6 |
| **T2 回收** | 0.65 → 0.80 | +0.60 → **−0.30** | **−1.57 保持** | +0.60 → **−0.30** | **+1.57 保持** | 保持扭转、回收侧摆/俯仰 → 停在 S1 参考姿态 |
| **T3 解扭** | 0.80 → 0.95 | **−0.30 → 0** | −1.57 → 0 | **−0.30 → 0** | +1.57 → 0 | 解开扭转 → **实际翻滚发生在此段** |
| **T4 过渡** | 0.95 → 1.45 | **0 保持** | 0 | 0 | 0 | 腿从支撑角回站立角 |
| **T5 保持** | 1.45 → ∞ | 0 | 0 | 0 | 0 | 站立保持 |

**记忆要点**：`F_spine1` 走 **0 → 0.6 → −0.3 → 0 → 0**；`H_spine1` 走 **0 → 0.6 → −0.3 → 0 → 0**；
两个扭转关节是**反相满量程**。

### 5.2 状态机相位 ↔ 时段

```
slow1_target:  time1 ──0.65──> time2 ──0.15──> time3 ──0.15──> time4 ──0.50──> time5_end
状态机:         P1 期望 0.80s ══════════════╗  P2 期望 0.15s ╗  P3 ────────────────>
                                            ↑                ↑
                                        T2 末 = 0.80s    T3 末 = 0.95s
```

- **P1 = T1 + T2**，`_P1_EXPECT = 0.8 × λ`
- **P2 = T3**，`_P2_EXPECT = 0.15 × λ`
- **P3 = T4 + T5**，由站立判据退出（`projected_gravity_b[2] > 0.9 且 base_z > 0.05` 连续 0.4 s）

`slow1_target` 内部 `time1 = 1.0` 这个 padding 被 `current_time = 1.0 + tn·λ` 精确抵消，
**不存在 off-by-one**。

### 5.3 门控判据

**判据已改为基于 §4 的标记 site（不再依赖四元数）。**
`mdp/command.py` 与 `scripts/SQuRo_backup_Replay.py` 两侧实现保持一致：

```python
# u=(back_z-belly_z)/||p_back-p_belly||；u≥cos(45°) 为正置，u≤−cos(45°) 为倒置
# 中间姿态、退化向量、非有限向量均为未知，不参与达标或回退。
S1 = F_inverted & H_upright & (zF < 0.03) & (zH < 0.03)  # 前段倒置、后段正置
S2 = F_upright & H_upright & (zF < 0.04) & (zH < 0.04)    # 两段都正置
both_inverted = F_inverted & H_inverted                     # P2/P3 回退候选
```

用世界坐标比较腹/背 site 可**自动消除** F/H 两段局部坐标相反的问题（F 腹面在局部 +Y，H 在 −Y）。
回放脚本额外提供 `_uprightness(idx)`，返回带符号的背腹轴世界 Z 分量（`>0` = 正置），供日志打印。

| 相位切换 | 判据 | 物理含义 | 名义时刻 |
|---|---|---|---|
| P1 → P2 | **S1** | 只有后段翻正 | t_nom = 0.80 |
| P2 → P3 | **S2** | 前后段都翻正 | t_nom = 0.95 |

S1/S2 候选必须分别连续满足 `0.10 s` 才触发即时阶段切换；双倒候选连续满足 `0.15 s` 时，P2/P3 立即回退 P1。
P1/P2 末端等待分别为实际 `1.0/0.3 s`，超时未达标则当前阶段时钟归零重试；侧立、前正后倒等未知组合不被误判为双倒。

**旧版单步符号门控的历史验证**（`scripts/Backup/verify_gate_predicates.py`，两法独立重算，不一致帧数为 0）保留如下；当前训练状态机已改为上面的 45° 连续确认门控，旧时刻不再作为新实验的通过标准：

| λ | S1 首次成立 | S1 门控触发 (P1→P2) | S2 首次成立 | S2 门控触发 (P2→P3) | 最终 |
|---|---|---|---|---|---|
| 1.0 | 0.89 s | 0.90 s | 1.09 s | 1.10 s | DONE @ 1.91 s |
| 3.0 | 2.31 s | 2.41 s | 2.67 s | 2.86 s | DONE @ 4.46 s |

门控触发时刻均略晚于名义边界（0.80×λ / 0.95×λ），落在 buffer 窗内。
**S1 需要约 0.09 s 的 settle 才成立**——T2 回收段结束时两段躯干仍有残余形变，
这是 buffer 参数存在的物理原因，不要随意调小。

回放脚本 `StateMachinePolicy` 改用 site 判据后，两个 λ 仍一次通过：

| λ | S1 达成 | S2 达成 | 稳定站起 |
|---|---|---|---|
| 1.0 | 0.90 s | P2 内 0.24 s | 0.42 s |
| 3.0 | 2.40 s | P2 内 0.44 s | 1.21 s |

（λ=3.0 的 S1 精确落在 `P1_END×λ = 2.40 s`。）

---

## 6. 手调脚本与训练参考表已经统一

`mdp/timing.py` 当前值（**已与手调脚本对齐**）：

```python
P1_BUILD_DURATION   = 0.65
P1_RECOVER_DURATION = 0.15     # 手调实测的最小可行值
P2_DURATION         = 0.15
STAND_TRANSITION_DURATION = 0.50
STAND_HOLD_DURATION       = 1.05
TIME_COMPARISON_SCALE     = 3.0
```

| | T1 | T2 | T3 | P1_END | P2_END |
|---|---|---|---|---|---|
| **手调脚本** | 0→0.65 | **0.65→0.80** | 0.80→0.95 | **0.80** | **0.95** |
| **训练参考表** | 0→0.65 | **0.65→0.80** | 0.80→0.95 | **0.80** | **0.95** |

**为何取 0.15 而非 0.65**：0.15 s 是手调实测的最小可行回收时长。
原先把 T2 放大到 0.65 s 等效于**只对该段做时间缩放**，会让参考表与演示在时序上错开 0.5 s。

统一后的实测效果（λ=3.0，地面真值）：姿态轨迹明显更单调——
统一前 H 段在 t=0.49~2.17 s 长期停留在 −0.27~−0.5 的"侧立"区反复摆动；
统一后直接由倒置过渡到正置（t=1.93 s 已达 +0.551），S1 于 t=2.41 s 精确落在 `P1_END×λ`。

### 6.1 动作 scale 已统一为 0.3

| 位置 | 值 |
|---|---|
| `SQuRo_Backup_env_cfg.py` 的 `JointPositionActionCfg.scale` | **0.3** |
| `StateMachinePolicy.action_scale` | **从 env_cfg 自动读取**（可显式覆盖） |

`joint_delta = (raw_action × scale + default_offset) − default_offset = scale × raw_action`，
故 0.3 与 0.5 只差控制分辨率，不影响可达范围——但**受 `clip_actions = 6.0` 约束**。

覆盖性核算（`scripts/Backup/check_action_scale_coverage.py`，期望极值取自参考表）：

| 关节 | 期望极值 | 需要 action@0.3 | 余量 (clip=6.0) |
|---|---|---|---|
| **F_body / H_body** | ∓1.570 / ±1.570 | **5.233** | **1.15×** ← 最紧 |
| HL_hip / HR_hip | −1.500 | 4.667 | 1.29× |
| FL/FR_elbow | +0.550 | 2.833 | 2.12× |
| F_spine1 / H_spine1 | +0.600 | 2.000 | 3.00× |
| 其余 | — | ≤1.833 | ≥3.27× |

**结论：0.3 可覆盖全部期望极值，最小冗余 1.15×。** 两个扭转关节是唯一逼近 clip 的，
已由 `tests/test_squro_backup_timing.py::test_action_scale_covers_reference` 断言守护
（要求每个关节 `clip / 所需 action >= 1.1`），防止后续改动悄悄吃掉余量。

### 6.2 `_HL_HOLD` 已与手调对齐

`reference.py` 的 `_HL_HOLD` 由 `(−1.40, −0.25)` 改为 **`(−1.50, −0.25)`**，与手调
`HL_HOLD` 完全一致。后腿髋角期望极值随之变为 −1.500，需要 action 4.667（余量 1.29×，仍在安全范围）。
至此**手调脚本与训练参考表的关节角目标完全一致**。

---

## 7. 已知未解决问题

1. **`body_link_quat_w` 参考系未定论**（见 §3）。**当前所有姿态判据已完全绕开它**
   （S1/S2 改用 §4 的标记 site，见 §5.3），因此该问题不再影响翻正任务的功能。
   保留此条仅为提醒：**不要基于该量写新的物理判断代码**。若某处必须用四元数，
   请先用 `fit_body_axes.py` 一类的 site 配准法交叉验证。

2. **XML 加了 4 个 site**（nsite 23 → 27）。现有 `find_sites` 调用都按名字精确匹配，
   已回归验证通过；但任何按 site 索引顺序硬编码的代码需要复核。

3. **全量测试在本机受限环境下无法运行**：`mujoco_warp` 的 kernel cache 位于
   `%LOCALAPPDATA%\NVIDIA\warp\Cache`，只读环境下会抛大量 `PermissionError`
   （与代码无关）。验证 SQuRo 改动请只跑相关子集：
   `tests/test_squro_backup_timing.py`、`tests/test_squro_backup_replay.py`、`tests/test_asset_zoo.py`。

4. **`events.py` 硬编码了 36 个 XML joint 下标**（`[6,8,12,14,24,...]`），
   绕过了 `mdp/indices.py` 的统一管理，XML 一改就会静默错位。**建议改走 indices。**

5. **`curriculums.py` 注释写 `timestep=0.001/decimation=5`**，
   实测 `sim.mujoco.timestep=0.002`、`decimation=5` → `step_dt = 0.01 s`（注释过期）。

---

## 8. 诊断脚本与测试清单

`src/mjlab/scripts/Backup/` 下的只读诊断脚本（均不落盘，结果打到 stdout）：

| 脚本 | 用途 |
|---|---|
| `measure_segment_gravity_truth.py` | **地面真值**：标记 site 测各段正置度（本文档 §4 的验收工具） |
| `verify_gate_predicates.py` | 验收 `_check_S1`/`_check_S2` 与独立地面真值是否一致（§5.3） |
| `check_action_scale_coverage.py` | 核算 action scale 对期望极值的覆盖与 clip 冗余（§6.1） |
| `audit_handtuned_script.py` | 核对 `slow1_target` 分段与状态机相位时钟是否对齐、是否被 auto-reset |
| `trace_s1_window.py` | 逐帧观察 S1/S2 达成瞬间的关节角与几何条件 |
| `analyze_righting_attitude.py` | 从回放 CSV 反推姿态、对比手调参考与策略实际 |
| `fk_body_height.py` | 用 MuJoCo FK 精确计算 F/H body 高度（CSV 未记录该量） |
| `audit_body_frame.py` | 四元数参考系对拍（已证明未能定论，见 §3） |
| `fit_body_axes.py` | 用 site 世界坐标最小二乘拟合 body 姿态 |

单元测试（23 passed）：

| 测试 | 覆盖内容 |
|---|---|
| `tests/test_squro_backup_replay.py` | `slow1_target` 各段关键帧姿态、T4 腿角过渡、T1–T3 腿固定、`action_scale` 跟随 env |
| `tests/test_squro_backup_timing.py` | 参考表关键帧、T1/T2 速度、缓冲冻结、身体轨迹重定时、相位门限、**动作覆盖冗余** |

> **跑测试时建议加 `-p no:cacheprovider` 并设 `PYTHONDONTWRITEBYTECODE=1`**：
> 受限/只读环境下 `.pytest_cache` 与 `__pycache__` 写不进去，会读到**过期字节码**，
> 表现为测试结果与源码不符的假失败。命令：
> `$env:PYTHONDONTWRITEBYTECODE="1"; uv run python -B -m pytest tests/test_squro_backup_timing.py -q -p no:cacheprovider`

回放脚本：`src/mjlab/scripts/SQuRo_backup_Replay.py`

```powershell
# 手调状态机回放（λ=1，无头，结果打到 stdout）
uv run python -m mjlab.scripts.SQuRo_backup_Replay --time-scale 1.0 --visualize none --duration 4.0

# 地面真值姿态验收
uv run python -m mjlab.scripts.Backup.measure_segment_gravity_truth 1.0
```

> 注意：`--visualize none/video` 分支末尾会尝试写 PNG/CSV 到 `logs/rsl_rl/.../replay_videos/`，
> 在只读或受限环境下会抛 `PermissionError`（不影响前面的仿真与 stdout 结果）。

## 整体架构与设计变更：快速初筛配置（2026-09-15）

- 变更范围：仅翻正任务的仿真参数、采样长度与时间缩放课程开关；奖励、动作参考、阶段检测阈值不变。
- 旧配置：物理步长 0.002 s，decimation=5，求解迭代 100/50，120 步采样，训练固定 λ=3。
- 新配置：物理步长 0.005 s，decimation=4，求解迭代 10/20，24 步采样；PPO 仍为 5 epochs、4 mini-batches。
- 数据流：每 4 个物理步执行一次策略控制（0.02 s）；每环境采集 24 步（0.48 s）后更新策略，采样段结束不重置回合。
- 模块职责：环境配置控制仿真精度与课程开关；`curriculums.py` 的 `_STEPS_PER_ITER=24` 同时供 RL 配置和课程轮数换算使用。
- 课程恢复：0～1000 轮 λ∈[3,6]；1000～2000 轮采样下限由 3 线性降至 1；之后 λ∈[1,6]。每个环境每回合独立采样，回合内固定。
- 接口：训练 `fixed_time_scale=None`；play 仍默认固定 λ=3，手调脚本继续使用显式 `time_scale`。
- 连续确认仍按实际秒：S1/S2 的 0.10 s 对应 5 步；双倒的 0.15 s 向上量化到 8 步，即 0.16 s。冗余秒数不随 λ 缩放。
- 变更理由：降低初步试验成本。较粗接触求解和较短采样段可能影响翻正行为，初筛失败不能直接判定方案无效，确定方案后需恢复精度复验。
- 回合仍为 10 s，未自动延长；大 λ 下等待与站立过渡可能接近回合上限，分析时需区分阶段失败与整回合超时。
