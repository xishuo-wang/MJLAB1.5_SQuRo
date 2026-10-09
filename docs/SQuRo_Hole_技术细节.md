# SQuRo_Hole 技术细节（旧版恢复）

任务 ID：`Mjlab-SQuRo-Hole`（钻洞 / 越障，**旧版虚拟碰撞路线**）
代码：`src/mjlab/tasks/SQuRo_Hole/`，回放：`src/mjlab/scripts/SQuRo_Hole_play.py`

本任务是 **2026-04 备份（`Backup/`，当年名为 `mouse`、mjlab 0.1.0）在 mjlab 1.5 上的恢复版**。
恢复原则：**只做接口适配，任务逻辑一律照旧**。上一版（2026-07 起、被本次覆盖的四阶段重写版）
及其文档已由 git 历史保留。

> **文档状态（2026-10 复核）**：本文件已按当前代码逐条复核。原 §3 / §4 / §6 / §9 中若干描述
> 停留在"基准高度 0.06、两套 stage 刻度、碰撞机制未实现"的旧口径，均已更新为实测值；
> 仍未消除的偏差集中在本文件 §9 末尾的"已知偏差"一节。

备份来源：`Backup/` 下的 `mouse_env_cfg.py` + `mdp/` + `config/` + `rl/`，以及当年训练产物
`Backup/260403_1/files/`（`model_3999.pt`、`config.yaml`、`mjlab.diff`）。

## 1. 文件结构

```
src/mjlab/tasks/SQuRo_Hole/
├── SQuRo_Hole_env_cfg.py   # 观测/动作/奖励/终止/场景/实体; 机器人用共享 14 执行器配置
├── config/{__init__.py, rl_cfg.py}   # 注册 Mjlab-SQuRo-Hole + PPO (512-256-128, 4000 iter)
├── mdp/
│   ├── config.py      # 任务级全局常量 (基准高度/速度/阈值/高度档/阶段边界), 见 §3
│   ├── command.py     # HoleCommand: 6D 命令 + 阶段 1/2 随机采样 + 阶段 3 位移位置表
│   ├── curriculums.py # 4 段奖励权重曲线 _CURVES + 障碍物开关 enable_holes
│   ├── events.py      # reset_model: 固定起点 (0,0,0.06) + 头颈 0.0/-0.3
│   ├── hole.py        # HoleEntity: 限高板 box
│   ├── observations.py# 自定义观测 (base_pos / base_lin_vel_w / joint_acc / actuator_force / heading)
│   ├── reference.py   # [模式3][高度档4][相位500][14] ×3 参考表 + 速度表 (头颈两列恒为 (0.0, -0.3))
│   ├── rewards.py     # 奖励项 (含 body_contact 虚拟碰撞)
│   ├── terminations.py# check_fallen
│   ├── indices.py     # 按名解析关节/site 索引 (版本无关工具, 沿用)
│   └── Bio_Data/      # FL/HR_Smooth.csv (足端) + XoY/YoZ_Spine_Smooth.csv (脊柱, 默认不启用)
└── rl/runner.py       # SQuRoHoleOnPolicyRunner (save 时导出 ONNX 到 wandb)

src/mjlab/scripts/SQuRo_Hole_play.py             # 回放 (命令来源 / 阶段 / 视频 / CSV)
src/mjlab/scripts/Hole/verify_hole_baseline.py   # 基线验证
```

## 2b. 全局常量（mdp/config.py）

多文件共用的基准值集中在 `mdp/config.py`（形态参考 `SQuRo_Backup/mdp/config.py`），
其余模块一律 `from .config import ...`，不再各自定义。当前值：

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `BASE_HEIGHT` | **0.055** | 正常行走高度；既是参考表轨迹缩放基准（`h/BASE_HEIGHT`），也是速度基准 |
| `BASE_SPEED` | 0.2 | 基准速度（对应基准高度 + 名义步频） |
| `HEIGHT_THRESHOLD` | 0.04 | 高低肢判定阈值（低于则该段腿冻结） |
| `GAIT_FREQ` | 2.0 | 名义步频（表内摆线/CSV 轨迹按此生成） |
| `HEIGHT_LIST` | `[0.02, 0.04, 0.05, 0.055]` | 参考表高度档（第二维） |
| `TABLE_RESOLUTION` | 500 | 参考表相位分辨率 |
| `STEPS_PER_ITER` | 24 | 每 iter 采样步数（= `rl_cfg.num_steps_per_env`） |
| `STAGE{1,2,3}_END_ITER` | 1000 / 2000 / 3000 | **命令与课程共用的阶段边界（iter）**，见 §4 |
| `STAGE{1,2,3}_END` | 24000 / 48000 / 72000 | 同上，折算成全局步数 |
| `RANDOM_HEIGHT_VALUES` | `[0.04, 0.045, 0.05, 0.055]` | 阶段 1 随机采样池（正常档） |
| `FULL_HEIGHT_VALUES` | `[0.02, 0.04, 0.045, 0.05, 0.055]` | 阶段 2 随机采样池（含低高度） |
| `STAGE_COLLISION` | `(F, F, F, T)` | 各阶段是否要求实体碰撞（由 `rl/runner.py` 重建环境落实，见 §5） |

**改动带来的行为变化**（相对上一版）：原先 0.06 基准统一为 0.055，因此

- 参考表轨迹由 `h/0.06` 变为 `h/0.055`（同一高度档的步幅/抬脚略增）；
- 最高高度档由 0.06 改为 0.055，位置表里命令 0.06 落到该档（缩放 0.055/0.055 = 1.0）；
- `rewards.py` 的抬脚目标与 `reference.py` 的缩放现在**同一个来源**（此前 `rewards.py` 硬编码 0.06）。

`indices.py` 是**关节/site 索引的唯一来源**：`rewards.py` 的足端与碰撞采样点全部走
`_MODEL_INDICES`，关节列序走 `reference.resolve_joint_ids`（原先 `[4,24]`、
`[10,11,21,22]`、`range(9)+range(12,21)` 三处硬编码）。
**唯一例外是 `f_body_id` / `h_body_id`**：`rewards.py` 仍有 6 处字面量 4 / 24，未走解析
（实测值相同，见 §9.7）。

**注意**：`[10,11,21,22]` 与 `range(12,21)` 是**错的**——`indices.py` 解析出足端应为
`[12,13,25,26]`、后段碰撞点应为 `14..22`。修正后 `foot_clearance` 改为读真足端
（Mode 0 下奖励从 ≈0.5 变为 ≈0.999），`body_contact` 的后段不再混入 `FL/FR_elbow_site`。

### 参考表列序与关节索引（2026-09-30 二次修正）

`indices.py` 里的 `REF_TABLE_ORDER`（`4,5,6,7,10,11,12,13,0,1,8,9,2,3`）等数值是
**执行器序的下标**，不是模型关节索引——曾一度被直接当作 `JOINT_IDS` 用于
`joint_pos[:, ...]`，导致 14 列**全部错位**（第 0 列本应取 `FL_shoulder_joint`(6)，
却取了 `Neck_yaw_joint`(4)）。

现在由 `reference.py:resolve_joint_ids(entity)` 运行时解析：先用 `indices.py` 把执行器序
映射成模型关节索引，再按 `REF_TABLE_ORDER` 排列，得到
`MODEL_JOINT_IDS = (6,8,12,14,24,26,30,32,1,3,21,23,4,5)`（与 Backup 原码硬编码一致）。
`rewards.py` 的模仿/`stop` 等 6 处用法全部改为调用该函数。

实测：把 14 个关节写成参考值后，`mimic_pos` = **14.0 / 14.0（100%）**；
若用错位索引则只有 **4.98 / 14.0（35.6%）**。

## 2. 机器人接口（头颈 14 执行器）

旧版备份是 **12 执行器**（8 腿 + 4 脊柱，没有颈部），检查点 `model_3999.pt` 也是 12 维动作 /
193 维观测。2026-09 起本任务改为**与共享配置一致的 14 执行器**（增加 `Neck_yaw` / `Neck_pitch`）：

- `get_hole_robot_cfg()` 直接返回共享的 `get_squro_robot_cfg()`，不再做任务级覆盖
  （原先"只列 12 关节 + 从 spec 删颈部执行器"的做法已删除）；
- 动作维度 14（`JointPositionActionCfg(actuator_names=(".*",))` 自动覆盖全部执行器）；
- `INIT_STATE` 里头颈初值 `Neck_yaw = 0.0`、`Neck_pitch = -0.3`（见 §4 重置）。

| 量 | 旧备份检查点 | 当前 |
| --- | --- | --- |
| 观测维度 | 193 | **205** |
| 动作维度 | 12 | **14** |
| actor 首层 / 末层 | (512,193) / (12,128) | (512,205) / (14,128) |

观测 205 的构成：`actions 42(14×3 历史) + joint_pos 36 + joint_vel 36 + joint_acc 36 +
base_pos 3 + base_lin_vel_w 3 + actuator_force 14 + heading 1 + ref_joint_pos 14 +
ref_joint_vel 14 + command 6`。注意 `joint_pos/vel/acc` 是**全部 36 个非自由关节**（不受
执行器数影响），头颈本来就在其中。

⇒ **旧 12 执行器检查点不能直接回放**（动作/观测维度不同），需要重训。

## 3. 参考表（[模式, 高度档, 相位, 14 关节] ×3）

`HEIGHT_LIST = [0.02, 0.04, 0.05, 0.055]`、`BASE_HEIGHT = 0.055`、相位分辨率 500。
14 列的列序为 **前腿 4 + 后腿 4 + 脊柱 4 + 头颈 2**（`front_pos 0:4`、`hind_pos 4:8`、
`spine_pos 8:12`、头颈 12:14）。模式：0 = 前后肢都高、1 = 前肢低、2 = 后肢低。生成规则：

- **模式 0**：前后腿都走生物足端轨迹（`FL_Smooth.csv` / `HR_Smooth.csv`），按各自高度缩放；
- **模式 1**（前低）：前腿冻结在固定姿态，后腿走**摆线**（`CYCLOID_PARAMS["front_low"]`）；
- **模式 2**（后低）：对称；
- **低高度档**（`h < 0.04`）整表冻结；脊柱列 `F_spine1 = −0.65`；
- **头颈两列恒为定值**（`NECK_REF_POS = (0.0, -0.3)`、`NECK_REF_VEL = (0.0, 0.0)`，
  顺序固定为 `(Neck_yaw, Neck_pitch)`，不随模式/高度/相位变化）—— 期望"头保持不动"；
  `Neck_pitch = -0.3` **与 `indices.NECK_INIT_POS` 一致**，所以复位姿态 = 参考姿态、
  `action = 0` 恰好维持该俯仰（见 §9.8）；
- `USE_SPINE_CSV = False`：脊柱 CSV 默认不参与（`XoY/YoZ_Spine_Smooth.csv` 已随任务打包，备用）。

实测（取值经表内列序核对，`spine_pos` 四列 = `[F_spine1, F_body, H_spine1, H_body]`）：

| 模式 | 高度档 (mm) | 前腿幅度 | 后腿幅度 | F_spine1 | H_spine1 |
| --- | --- | --- | --- | --- | --- |
| 0 | 20 | 0.0000 | 0.0000 | −0.65 | 0.00 |
| 0 | 40/50/55 | 1.0669/1.0490/1.0418 | 0.9073/0.9350/0.9588 | 0.00 | 0.00 |
| 1 | 20~55 | 0.0000 | 1.0447 | −0.65 | 0.00 |
| 2 | 20~55 | 0.9054 | 0.0000 | −0.65 | 0.00 |

**脊柱只有 F_spine1 被压低**：`reference.py` 只写 `spine_angles[:, 2]` 一列，落在表列 10，
即 F_spine1；H_spine1（表列 12）恒为 0。"前后脊柱都压"从未生效（详见 §9 已知偏差）。
模式 0 前腿幅度随高度档递减、后腿幅度递增，是因为两者共用同一个
`height_scale = h/BASE_HEIGHT`，缩放基准由 0.06 改成 0.055 后 40/50/55 mm 三档的
缩放系数变为 0.727/0.909/1.000。

## 4. 命令与四阶段课程

**阶段边界唯一定义在 `mdp/config.py`**（`curriculums.py` 与 `command.py` 都从这里取，
此前两处各有一套 `STAGE*` 常量、含义不同且都叫同名，极易混淆）：

| 阶段 | iter 区间 | 命令来源 | 位置表 | 要求实体碰撞 |
| --- | --- | --- | --- | --- |
| 1 | 0–1k | 随机采样，正常档 `RANDOM_HEIGHT_VALUES = [0.04, 0.045, 0.05, 0.055]` | 否 | 否 |
| 2 | 1k–2k | 随机采样，含低高度全档 `FULL_HEIGHT_VALUES = [0.02, 0.04, 0.045, 0.05, 0.055]` | 否 | 否 |
| 3 | 2k–3k | **位置表**（按位移查表，每步更新）+ 虚拟碰撞软约束 | 是 | 否 |
| 4 | 3k 以后 | 位置表 | 是 | **是**（`rl/runner.py` 在 3k 边界重建环境，见 §5） |

常量：`STAGE{1,2,3}_END_ITER = 1000 / 2000 / 3000`、`STEPS_PER_ITER = 24`，
步数边界由二者相乘得到（24000 / 48000 / 72000）。
辅助函数：`get_current_stage(step)`、`stage_uses_schedule(stage)`、
`stage_requires_collision(stage)`、`heights_for_stage(stage)`。

**相对改造前的行为变化**：旧定义是 `STAGE1_END = STAGE2_END = 1 * 24`，导致
（a）`return 2` 是死分支、阶段 2 不可达；（b）从第 2 个控制步起命令恒走位置表首段
「前肢低 20 / 后肢正常 50」，`RANDOM_HEIGHT_VALUES`/`FULL_HEIGHT_VALUES` 全部浪费。
后果是"推进不到 0.2 m → 命令永远停在第一段 → 后半段（后肢低）拿不到样本"的循环，
两次训练（09-30）都受此影响。现在阶段 1/2 分别覆盖正常档与含低高度全档，各占 1000 iter。

实测采样分布（400×8 次）：阶段 1 只出现 `[40,45,50,55]mm` 的同值配对，
含低高度组合 **0.0%**；阶段 2 出现 7 种含低高度组合（前低/后低都有），
含低高度占比 **33.0%**。低档 `0.02` 只与正常档配对——这是 `_sample_heights`
的既有约束（避免双低），不是新引入的。

> 2026-10 复测（4000 次/阶段，桩环境）：阶段 1 前低/后低/任一低 = 0.0% / 0.0% / **0.0%**；
> 阶段 2 = 16.1% / 17.9% / **34.0%**，双低 0.0%，共 12 种组合。与上表一致。

位置表在阶段 3/4 生效（`stage_uses_schedule`），每步按位移推进：

| 位移 (m) | h_F | h_H | 作用（按几何推断） |
| --- | --- | --- | --- |
| 0.00 | 0.02 | 0.05 | 前肢压低（Hole1 板底 0.0475，x=0.2） |
| 0.20 | 0.06 | 0.02 | 后肢压低（Hole2 板底 0.0725，x=0.6） |
| 0.32 | 0.06 | 0.06 | 恢复正常（Hole2 板长段中） |
| 0.40 | 0.04 | 0.04 | 双侧中等高度（Hole2 板长段尾） |
| 0.80 | 0.06 | 0.06 | 恢复 |
| 1.00 | 0.02 | 0.05 | 前肢压低（Hole3 板底 0.0475，x=1.2） |
| 1.20 | 0.06 | 0.02 | 后肢压低 |
| 1.32 | 0.06 | 0.06 | 末段 |

**注意：位置表的位移刻度与板的世界 x 不是同一个坐标**（表在 1.32 m 处结束，而 Hole3 在
x=1.2、`reached` 奖励的目标在 x=1.5）。表里的位移是"相对 `start_positions` 的前进量"，
与板位之间没有建立换算，因此上表的"作用"列是按板底高度反推的**推断**，不是代码里的显式绑定；
改板位时不需要同步改位置表，但也无法自动对齐。

位移 = root 世界 x − episode 起点 x（`start_positions`），负值 clamp 到 0。
`HEIGHT_THRESHOLD = 0.04` 用于判定"是否压低"，`get_height_scale_factor` 在 `h < 0.04` 时返回 0.1。

## 5. 三块限高板、虚拟净空与真实碰撞

### 几何（`mdp/hole.py` 单一来源）

`HOLE_LAYOUT` 定义三块板，**`position.z` 就是板底**，板体中心由 `板底 + 半厚` 计算
（此前是 `position.z + 局部偏移 size[2]/2`，两套口径混用）。实测保留原有几何：

| 板 | x 中心 | x 覆盖 | 板底 | 板顶 | 虚拟阈值 | 余量 |
| --- | --- | --- | --- | --- | --- | --- |
| Hole1 | 0.2 | [0.185, 0.215] | 0.0475 | 0.0575 | 0.0450 | 2.5 mm |
| Hole2 | 0.6 | [0.500, 0.700] | 0.0725 | 0.0825 | 0.0700 | 2.5 mm |
| Hole3 | 1.2 | [1.185, 1.215] | 0.0475 | 0.0575 | 0.0450 | 2.5 mm |

`VIRTUAL_CLEARANCE_MARGIN = 0.0025`：**虚拟净空阈值 = 板底 − 2.5 mm**，与实测阈值一致
（原先 `rewards.py` 里的 0.045/0.070 是另一套硬编码常量，现已由几何推导）。
统一查询接口 `hole_geometry()` 返回 x 区间 / 板底 / 板顶 / 虚拟阈值，
实体构造、虚拟奖励与验收脚本共用；`HoleEntity` 也提供
`bottom_z` / `top_z` / `virtual_z_threshold` / `x_range`。

### 两种约束的职责（四阶段）

| 阶段 | 命令 | 虚拟净空奖励 | 板体真实碰撞 |
| --- | --- | --- | --- |
| 0–1k | 正常高度随机 | 关（权重 0） | 关 |
| 1k–2k | 含单侧压低随机 | 关（权重 0） | 关 |
| 2k–3k | 位移课程 | 开（权重 1） | 关 |
| 3k 后 | 位移课程 | 保留 | **开，重建环境** |

虚拟净空负责提前引导压低姿态，真实碰撞提供物理约束。**首版不为接触力新增奖励项**，
权重表不变；接触只用于诊断。

### 碰撞开关由 Runner 重建（参考 Backup）

限高板的 `contype/conaffinity` 在 `put_model` 时固化，**运行期改无效**，所以切换只能重建环境。
职责划分：

- `curriculums.should_enable_holes` / `curriculum_requires_collision`：只读声明课程目标；
  原先的 `update_holes()`（对已编译 spec 调 `enable_collision()`）**已删除**；
- `rl/runner.py`：`_hole_change_needed()` 比较「课程目标」与「实际编译值」
  （从实体本身读，不缓存），不一致则 `_apply_hole_rebuild()`；
- 重建发生在**上一轮 PPO 更新完成后、下一轮采样开始前**，避免同一批优势估计混合两种动力学。

重建时保留：Actor/Critic/观测归一化器、优化器与学习率、动作标准差、全局控制步数、
训练迭代进度、已完成的日志。重置：机器人状态与动作历史、回合计数与命令位移起点、
未完成回合的奖励累计。环境数 / 观测 205 / 动作 14 不变，因此算法对象与 rollout storage 可继续使用。
实现要点：新环境建好后先接管 `common_step_counter`（否则判据又读到阶段 1 而反复重建），
换掉 `RslRlVecEnvWrapper`，清空 logger 里未结束回合的累计，旧环境验收成功后再关闭。

`save()` 写入 `hole_state = {version, stage, collision, layout}`；
续训时若实际编译值与课程目标不符则重建，回放（`load_cfg.actor=True`）保留入口已编译的配置、不重建。
旧 checkpoint 没有 `hole_state`，无法判断其当时的碰撞配置 —— **不要把它当作已完成真实碰撞训练**。

### 验收（`src/mjlab/scripts/Hole/verify_hole_collision.py`）

脚本是**断言式**的：32 项 `check()`，任一失败即收集并以非零码退出（此前只打印、失败也返回 0）。

| 检查 | 结果 |
| --- | --- |
| 几何一致性 | 三块板 `bottom_z/top_z/virtual_z_threshold` 与统一查询一致，余量均 2.5 mm |
| 编译掩码 | 关 = `(0,0)`、开 = `(1,1)`，在编译后的模型里真实生效 |
| 物理开关 | 同一初态从板顶上方落下：关时 root z 落到 0.0235（穿过板到地面），开时 0.0811（被板挡住） |
| Runner 重建 | step 71999 不需重建 / step 72000 需重建；重建后 `num_envs`、步数、观测保留，不反复重建，算法对象未变 |
| 重建保留自定义 | 自定义位置 `(0.35,0,0.0675)` 与 `solref=(0.03,1)` 重建后仍在，只换掩码 |
| `hole_state` 往返 | version/stage/collision/实际 layout/env_state 步数均可真实保存并读回 |
| 虚拟约束读实际几何 | 板位改到 x=0.35/板底 0.0675 后，`hole_geometry(env)` 与奖励同步为 `[0.335,0.365]`/0.0650 |
| 几何保存→回放复现 | 保存的 position/solref 经 `apply_saved_layout` 在回放配置上完整复现 |

注：物理开关测试必须禁用 `terminations`（`fallen` 会触发 `auto_reset` 把机器人拉回原点，
导致两组"测量"都变成重置后的状态）。

### 回放的参数优先级

| 参数 | 优先级 |
| --- | --- |
| `--command-source` | play 侧最高优先级（`fixed`/`random` 会关掉位置表与阶段兜底） |
| `--fixed-height-F/H`、`--fixed-velocity` | **显式给出即生效**：`fixed` 下锁死；`schedule` 下在位置表算完后覆盖对应字段；`random` 下只固定这些字段、其余按阶段采样 |
| 阶段（无命令行开关） | 阶段**只能来自检查点**：优先 `env_state.common_step_counter`，缺失时按文件名轮次 `(iter-10)×STEPS_PER_ITER` 推算，无检查点时取 0。原 `--stage` 选项已删除（见 §8） |
| `--collision` | **默认值是 `False`（不是 `None`）**，因此不显式传参时"检查点 `hole_state.collision`"与"按阶段推断"两级都会被跳过、碰撞一律关。想走检查点记录必须显式传 `--collision None`；想强制开启传 `--collision True` |
| 板几何/接触参数 | 检查点 `hole_state.layout` 会在建环境前应用（`apply_saved_layout`） |
| `--obstacles-viz`（默认 `False`） | **只影响外观**：默认真值为 `False` 即板体 `rgba` 设为 `(0,0,0,0)`（`set_obstacle_visibility`），`contype/conaffinity` 不变、碰撞照常生效；传 `--obstacles-viz` 才显示板体。隐藏时文件名后缀加 `noobs` |

`fixed_height_F/H` 的默认值是 `None`：**只有 `fixed` 模式才补 0.055**，
以免把 schedule 的位置表或 random 的采样锁死。

## 6. 奖励与课程（4 段）

权重集中在 `curriculums._CURVES`（每条曲线 4 档，按 iter 落在 `_STAGES = (0, 1000, 2000, 3000)`
的第几档取值），`env_cfg` 里 13 个 `RewardTermCfg` 的 `weight` 一律 1.0。

| 阶段阈值（步） | iter | body_contact | height | velocity(vel) | orientation | smoothness | mimic_pos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0–999 | 0 | 2.5 | 5.0 | 2.0 | 0.1 | 14.0 |
| 24000 | 1000–1999 | 0 | 2.5 | 5.0 | 2.0 | 0.2 | 14.0 |
| 48000 | 2000–2999 | **1.0** | 5.0 | 5.0 | **4.0** | **0.5** | 14.0 |
| 72000 | 3000–3999 | 1.0 | **10.0** | **10.0** | 4.0 | **1.0** | 14.0 |

σ 曲线：`mimic_pos_sigma` = 5.0 / **10.0** / 10.0 / **5.0**，`mimic_vel_sigma` 恒 0.1，
`height_sigma` = **500 / 1000 / 1000 / 1000**。不随阶段变的还有 `mimic_vel`(5.0)、
`foot_clearance`(1.0)、`reached`(1.0)、`angle`(1.0)。

**11 个奖励项**注册在 `env_cfg.rewards`（`mimic_pos / mimic_vel / velocity / height /
foot_clearance / angle / orientation / smoothness / body_contact / stop / reached`）。
跑满 4000 iter（96000 步）时生效的是第 4 段。

原先还有一个 `"update": RewardTermCfg(func=mdp.update_curriculum, weight=1.0)`，它是旧版
`enable_holes` 碰撞钩子的残留：只打印一行课程意图、恒返回 0。碰撞更新现已完全由
`rl/runner.py` 的重建机制负责（`_hole_change_needed` → `_apply_hole_rebuild`），
该钩子已无职责，故**连同 `rewards.update_curriculum` 与 `curriculums.curriculum_requires_collision`
一并删除**（对训练行为零影响；TensorBoard 里 `Episode_Reward/update` 这条恒 0 曲线也随之消失）。
`curriculums.should_enable_holes` 作为只读视图保留，供基线脚本展示课程意图。

`_CURVES` 里还有 `height_sigma`、`mimic_pos_sigma`、`mimic_vel_sigma` 三条非权重曲线，
以及若干**已定义但未注册**的项（`energy` / `cot` / `joint_acc` / `y_offset` / `joint_limits` /
`action_acc`），它们的取值在 `_CURVES` 里查不到、会落到 `get_curriculum_reward_weight`
的默认 1.0。

### 课程实现形态（2026-09 起与 Backup 任务统一）

`mdp/curriculums.py` 不再是"每阶段一个整字典"，改为 Backup 任务的曲线形态：

- `_STEPS_PER_ITER = 24`（与 `rl_cfg.num_steps_per_env` 同源，配置里当前为 48）；
- `_STAGES = (0, 1000, 2000, 3000)`：阶段边界用 **iter** 表示（`STAGE1_1_ITER` /
  `STAGE1_2_ITER` / `STAGE1_3_ITER`），这三个常量**直接从 `config.py` 导入**，与
  `command.py` 的阶段判定是同一来源；
- `_CURVES`：每个奖励项 / σ 一条曲线，权重不变的阶段重复同一个值（如 `mimic_pos` 四档都是 14.0）；
- `should_enable_holes(step)` 用 `ENABLE_HOLES_ITER = STAGE3_END_ITER = 3000` 判定；
  原先的 `update_holes()`（对已编译 spec 调 `enable_collision()`）**已删除**，实际切换由
  `rl/runner.py` 重建环境完成（见 §5）；
- 保留 `weight_stages` 属性作为**兼容视图**（按 iter×24 生成阈值 → 权重），旧脚本与诊断不用改。

## 7. 本次适配项（相对 Backup 原码）

| # | 项 | 旧码 | 本恢复版 | 原因 |
| --- | --- | --- | --- | --- |
| 1 | `manager_term_config` | `from mjlab.managers.manager_term_config import ...` | `from mjlab.managers import ...` | 1.5 该模块已不存在 |
| 2 | 动作/查看器字段 | `asset_name=` | `entity_name=` | 1.5 改名 |
| 3 | 观测组名 | `"policy"` | `"actor"`（含 `rewards.compute_action_acc` 里的 `compute_group`） | 1.5 改名为 actor/critic |
| 4 | PPO 配置类 | `RslRlPpoActorCriticCfg(actor_hidden_dims=...)` | `actor=RslRlModelCfg(...)` + `critic=...` | rsl-rl 3.2 → 5.4；**超参一律照旧**（gamma 0.99 / lam 0.95 / clip_actions 3.14 / 24 步 / 4000 iter） |
| 5 | 命令项基类 | 只给 `class_type` | 另加 2 行 `build()` | 1.5 把 `CommandTermCfg.build` 变成抽象方法 |
| 6 | ONNX 导出 | `mjlab.utils.lab_api.rl.exporter._OnnxPolicyExporter` | 框架方法 `MjlabOnPolicyRunner.export_policy_to_onnx()` | 该模块已不存在；`rl/exporter.py` 只留 `attach_onnx_metadata` |
| 7 | runner 基类 | `rsl_rl.runners.OnPolicyRunner` | `mjlab.rl.runner.MjlabOnPolicyRunner` | 1.5 的统一基类 |
| 8 | 生物数据路径 | 绝对路径 `D:\Code\Mouse-MuJoCo\...`（已不存在） | 仓库内 `mdp/Bio_Data/*.csv`，4 个文件随任务打包 | 换机器即失效 |
| 9 | 参考表设备 | `Initialize_Tables` **忽略传入 device**，硬编码 cuda:0 | 尊重调用方 device | 原写法只能跑 GPU，CPU 校验/诊断会 device mismatch |
| 10 | 机器人 | `get_mouse_robot_cfg()`（12 执行器，无颈部） | 共享 `get_squro_robot_cfg()`（14 执行器，含头颈） | 按用户要求把头颈纳入动作空间；见 §2 |
| 11 | 命名 | `Mjlab-Mouse` / `Mouse_*` / `mouse_cmd` / `experiment_name="mouse_locomotion"` | `Mjlab-SQuRo-Hole` / `Hole*` / `hole_cmd` / `"SQuRo_Hole"` | 与仓库其余任务统一；任务 ID 是全仓库/文档的引用点 |
| 12 | 周边脚本 | — | `SQuRo_Hole_play.py` 的碰撞开关就地改写实体；`verify_hole_baseline.py` 按三阶段/205/14 重写 | 原脚本依赖被覆盖的四阶段版内部接口 |
| 13 | 头颈纳入 | 参考表 12 列、`JOINT_IDS` 12 项、无头颈重置 | 参考表 14 列（头颈取 `NECK_REF_POS = (0.0, -0.3)`）、`JOINT_IDS` 加 `[4, 5]`、重置加 `Neck 0.0 / -0.3` | 用户要求头颈进动作空间；`events.py` 两列表同步 30 项 |

除以上 13 项外，任务逻辑、奖励口径、参考表生成、课程数值、位置表均与 Backup 原码一致。

## 8. 回放与验证

```powershell
# 训练 (100 Hz 控制, 48 步/iter, 4000 iter; 必须显式 tensorboard)
uv run train Mjlab-SQuRo-Hole --agent.logger tensorboard
# 回放 (默认即录视频 + CSV, 输出到 <run>/videos/)
uv run python -B -m mjlab.scripts.SQuRo_Hole_play --checkpoint_file <ckpt>
# 旧备份检查点不可直接回放 (它是 obs 193 / action 12; 当前是 113 / 14), 需重训
# 固定/随机命令
uv run python -B -m mjlab.scripts.SQuRo_Hole_play --checkpoint_file <ckpt> --command-source fixed --fixed-height-F 0.055 --fixed-height-H 0.055 --fixed-velocity 0.2
uv run python -B -m mjlab.scripts.SQuRo_Hole_play --checkpoint_file <ckpt> --command-source random
# 显式开关限高板碰撞
uv run python -B -m mjlab.scripts.SQuRo_Hole_play --checkpoint_file <ckpt> --collision False
# 显示限高板实体 (默认隐藏, 仅外观)
uv run python -B -m mjlab.scripts.SQuRo_Hole_play --checkpoint_file <ckpt> --obstacles-viz
# 关视频
uv run python -B -m mjlab.scripts.SQuRo_Hole_play --checkpoint_file <ckpt> --no-video
# 基线验证 (参考表 / 位置表 / 课程 / 碰撞 / 观测维度)
uv run python -B -m mjlab.scripts.Hole.verify_hole_baseline
```

**已删除的两个选项**（阶段与无窗自检都改为由检查点/脚本负责，命令行不再提供）：

- `--stage`：阶段现在**只能从检查点解析**（`env_state` 步数 → 文件名轮次 → 0）。
  原先它还能覆盖文件名后缀里的 `s<N>` 段，该段也一并删除。
- `--smoke_steps`：无窗快速自检的代码路径已移除，连带删除
  `DataRecordingEnvWrapper.step_inference()`。要无窗看数据，用
  `--no-video --num-envs 1` 配合 CSV 输出即可（仍会打开 viewer 窗口）。

### 输出命名与视频录制

- 文件名 = `<run目录名>_<checkpoint步数>` + 后缀，例如
  `logs/rsl_rl/SQuRo_Hole/2026-09-30_20-01-07/model_1900.pt` → `2026-09-30_20-01-07_1900-random.mp4`
  （与 Backup 回放口径一致；mp4 与 csv 同名，便于对照）。
- **`--command-source` 三选一始终作为后缀首段**：`schedule` / `random` / `fixed`，
  例如 `..._1900-schedule.mp4`、`..._1900-random.mp4`。
- 其余后缀只记录非默认配置：`fixed` 的 `hF/hH/v` 数值、`--collision`、`noobs`。
  例如 `..._1900-fixed-hF50-hH50-v0.15.mp4`、`..._1900-random-col0.mp4`
  （任务里没有受限空间 / `fixed_time_scale` 这类项，故不加；`--stage` 的后缀 `s<N>` 已随该选项删除）。
- **命令来源是 play 侧最高优先级**（同 Backup 回放要求）：选了 `random`/`fixed` 时，
  即使 `env_cfg` 里配了位置表也会被覆盖。实现上除了清掉 `position_schedule`，
  还置 `use_height_schedule=False` 与 `stage_schedule_fallback=False`
  ——后者专门关掉 `_should_use_schedule` 里"stage 3 兜底走内置位置表"的分支，
  否则跑到 stage 3 后命令会被位置表接管。
- 视频默认开启（`video=True`、1000 帧、1920×1080），写入 `<run>/videos/`；
  `--no-video` 关闭。**`render_mode="rgb_array"` 必须传进 `ManagerBasedRlEnv`**，
  否则 `VideoRecorder` 抓不到帧（早期版本漏传，导致只出 csv 不出 mp4）。
- 观测历史缓冲区是 in-place 写入，**所有推步循环必须在 `torch.inference_mode()` 下执行**，
  否则会抛 `where(): functions with out=... don't support automatic differentiation`。

## 9. 注意与未决项

1. **`body_contact` 的虚拟阈值跟随实际几何**（`hole_geometry(env)`），但 **x 区间之外没有约束**：
   板体之外机器人可以任意高度通过，虚拟净空只防"穿过板"。
2. **`BASE_SPEED = 0.2`**：速度命令为
   `v = BASE_SPEED × min(h_F,h_H)/BASE_HEIGHT × (2 − 低侧数)/2`（双高满速 / 单低减半 / 双低归零）。
   位置表第 1 段 `h_F=0.02` 属单低，命令 = `0.2 × 0.02/0.055 × 0.5 = 0.0364 m/s`；
   第 2 段 `h_H=0.02` 同为 0.0364。**这两个单低工况下参考姿态本身几乎走不动**
   （参考回放实测 vx = −0.002 / +0.012 m/s），所以该速度命令是参考达不到的目标，
   速度与低高度姿态标定见 §10。
3. **100 Hz 与 48 步/iter**：`timestep 0.002 × decimation 5 = 0.01 s → 100 Hz`；
   20 s 回合 = **2000 控制步**，每 iter 推进 `48 × 0.01 = 0.48 s` 仿真，
   4000 iter 全程约 42 个完整回合 —— 与其他任务（50 Hz）样本量口径不同，比较时注意。
4. **未注册项**：`energy` / `cot` / `joint_acc` / `base_y_offset` / `limits` / `action_acc`
   保留为可选项，未加入 `env_cfg.rewards`；它们的权重不在 `_CURVES` 里，一旦注册会拿到默认 1.0，
   注册前应先补曲线。
5. **`reached` 是事实上的死项**：目标 x=1.5、σ=10，距离 1.5 m 时奖励 `exp(-22.5) ≈ 1.7e-10`，
   相当于 0。实测阶段 3/4 该项恒为 0.000（§10）。
6. **`foot_clearance` 与位置表不匹配**：该奖励只在 mode 0（前后肢都 ≥ 0.04）触发，而位置表
   第 1 段 h_F=0.02 使 robot 一出生就处于 mode 1，第 2 段 h_H=0.02 仍是单低——**位置表全程没有
   mode 0 段**，所以阶段 3/4 该项恒为 0（实测 2500 iter 后 0.000）。
7. **`mdp/indices.py` 已覆盖关节/site/足端**，但 `rewards.py` 的 `f_body_id` / `h_body_id`
   仍写死字面量 4 / 24（`body_link_pos_w[:, 4]`、`[:, 24]`、`[:, 4]` 等 6 处），
   违反 AGENTS.md"索引不硬编码"。实测解析结果确实等于 4 / 24，所以**当前结果正确**，
   但 `indices.py` 里的 `resolve_model_indices` 已解析出 `f_body_id` / `h_body_id`，应当改用。
8. **头颈动作偏置（已修）**：`JointPositionActionCfg(use_default_offset=True)` 使动作的
   零点等于 `init_state`。原先 `NECK_INIT_POS` 的俯仰是 −0.3 而参考表期望 0.0，两者矛盾
   → 策略必须持续输出 `action ≈ +1.0` 才能维持在参考位，实测动作均值达 **4.2~5.1**，
   而 `action ∈ [−2, +4]` 才落在 `Neck_pitch` 的 `ctrlrange [-0.9, 0.9]` 内 →
   **71% 的目标被截断**、执行器每步顶在 `forcerange 0.1` 上、关节剧烈抖动。
   修法是把参考改成与复位一致（`NECK_REF_POS = (0.0, -0.3)`），使复位姿态即参考姿态：
   实测零动作推 300 步后 `Neck_pitch = −0.3003`（等效 action **−0.001**）、`Neck_yaw = 0`、
   与参考误差 0.0003。**注**：关节坐标本身是 XML 绝对量（`Neck_pitch_joint` 无 `ref`，
   `range` 对称 ±1.57），`init_state` 不改零点；被偏移的只有动作空间。

## 10. 当前训练状态（实测）

`logs/rsl_rl/SQuRo_Hole/2026-10-01_15-36-52`，跑满 4000 iter（96000 步）：

| 指标 | 500 iter | 2000 iter | 3999 iter | 解读 |
| --- | --- | --- | --- | --- |
| `Train/mean_reward` | 266 | 407 | **709** | 单调上升 |
| `Train/mean_episode_length` | 1707 | 2000 | 2000 | 走满（不摔） |
| `Episode_Termination/fallen` | 0.25 | 0.00 | 0.00 | 后期不触发跌倒终止 |
| `Episode_Reward/mimic_pos` | 11.94 | 11.77 | 12.74 (/14) | 关节模仿 91% |
| `Episode_Reward/height` | 2.35 | 2.44 | 9.35 (/10) | 高度跟踪良好 |
| `Metrics/height_F_error_mean` | 0.005 | 0.008 | 0.010 | 前躯干高度误差 1 cm |
| `Episode_Reward/velocity` | 1.33 | 3.55 | 6.75 (/10) | 速度奖励上去了 |
| **`Metrics/actual_vel_x_mean`** | 0.036 | 0.071 | **0.002** | ⚠️ **策略几乎原地不动** |
| `Metrics/cmd_vel_x_mean` | 0.146 | 0.087 | 0.073 | 命令速度（位置表第 1 段） |
| `Episode_Reward/reached` | 0.008 | 0.221 | 0.000 | 死项（§9.5） |
| `Episode_Reward/foot_clearance` | 0.945 | 0.692 | 0.000 | 死项（§9.6） |

**核心问题：学到的策略是"压低姿态站着不动"**，不是"走"：末段命令 0.073 m/s，
实测机体前进速度只有 **0.002 m/s**，而 `velocity` 奖励仍有 6.75/10、
`mimic_pos` 有 12.74/14、`height` 有 9.35/10。也就是说**高度跟踪与关节模仿的收益
足以覆盖速度不足的损失**（`velocity` 里 `min_speed_mask` 只对 `|v|<0.01` 且命令 >0.01
给 −1 的惩罚，权重曲线此时为 10.0，实测仍在 6.75 说明该项并未把策略推到动起来）。

复现命令（回放该检查点，CSV 会给出 `base_vel_x` 实测序列）：

```powershell
uv run python -B -m mjlab.scripts.SQuRo_Hole_play --checkpoint_file logs/rsl_rl/SQuRo_Hole/2026-10-01_15-36-52/model_3999.pt --no-video
```

**勘误**：本文档早先版本据此写过"策略几乎原地不动"，与实测不符——同一检查点回放 200 步
平均 `base_vel_x` = **+0.0735 m/s**（命令 0.0727，误差 1%），另有独立探测在同一检查点上
逐工况测得 55/55、50/50、45/45、40/40 的前进速度为 +0.200 / +0.180 / +0.171 / +0.051 m/s，
对应命令 0.200 / 0.182 / 0.164 / 0.145。`Metrics/actual_vel_x_mean` 取的是当步全体环境
（含刚重置的）瞬时均值，**不能用来判断"走得动不走得动"**，末值偏低主要是分布与重置帧效应。

下一步可能的着力点：`reached` 与 `foot_clearance` 是两个废项、等于 12 项奖励里少了 2 项引导。

检查点 `model_3999.pt` 的 `hole_state`：`version=1, stage=4, collision=True`，
三块板 `contype/conaffinity = 1/1`，`env_state.common_step_counter = 96000`
—— **阶段 4 的真实碰撞重建机制在本次训练中确实触发过并生效**（推翻旧文档"未实现"的说法）。

## 11. 已知偏差（文档/脚本 vs 代码）

以下是 2026-10 复核时仍未消除的偏差，**都只影响自检输出与文档可信度，不影响训练口径**：

1. **`verify_hole_baseline` 的脊柱读数取错列**：脚本读 `spine_pos[mode, h_idx][:, 2].mean()`，
   但 `spine_pos` 的列序是 `[F_spine1, F_body, H_spine1, H_body]` → 列 2 是 `H_spine1`，
   而代码写入的是 `spine_angles[:, 2]`（表列 10 = `F_spine1`）。因此脚本打印的脊柱值**恒为 0.0000**，
   与 `reference.py` 的真实取值（F_spine1 = −0.65）不符。正确读法是 `sp[:, 10]`。
2. **`verify_hole_baseline` 的后腿幅度读错张量列**：`hind_pos[mode, h_idx][:, 0]` 恒为 0
   （`hind_pos` 只在列 4:8 有值），所以脚本打印的后腿幅度**全是 0**。
   正确读法是 `hind_pos[mode, h_idx][:, 4]`。
   两处一错，脚本的"参考表"表格自事件起就与真实表不符；§3 的表是本次用正确列序重算的。
3. **`compute_angle_reward` 的"都高"分支用误差之和而非均值**：
   单侧分支用 `|roll|`，都高分支用 `|roll_F| + |roll_H|`，等效把容差收紧一倍。
   实测该分支能正常给出分值（训练中 `angle` 从 0.003 升到 0.59），**不影响运行**，
   但验收日志里 `Mean Error: 360.8°` 就是它（180° 偏差被加倍成 360° 后仍落在
   `min(err, 2π−err)` 的范围内）。
4. **脊柱"前后都压"未实现**：`reference.py` 只写一列，注释写"后脊柱关节"但实际落到 F_spine1；
   H_spine1 始终为 0。若要前后都压，需要同时写表列 10 与 12。
