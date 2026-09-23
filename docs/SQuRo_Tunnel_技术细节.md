# SQuRo_Tunnel 技术细节

任务 ID：`Mjlab-SQuRo-Tunnel`（钻洞 / 越障，四足 + 脊柱直行场景）
代码：`src/mjlab/tasks/SQuRo_Tunnel/`，回放：`src/mjlab/scripts/SQuRo_Tunnel_play.py`

当前状态：**参考层已按旧版（`v0801` 虚拟碰撞版）迁移，Phase0 可开训**（2026-09-23）。
`logs/rsl_rl/SQuRo_Tunnel/2026-09-21_20-23-57/` 有迁移前的 3999 iter 基线 checkpoint（Phase0，
可在回放脚本里对比）。本文件是参考手册；实验数据请按时间追加，不要覆盖历史结论。

## 0. 阶段划分（2026-09-23 起）

| 阶段 | iter | 内容 | 命令来源 |
| --- | --- | --- | --- |
| Phase 0 | 0 ~ 3k | 无障碍物，随机高度命令（前后肢解耦但受配对约束） | `_resample_phase0`，episode 内固定 |
| Phase 1 | 3k ~ 6k | 受限空间，洞位置采样 + 方波高度轨迹 | `_update_phase1`，每步按实际 x |

边界常量：`curriculums.PHASE1_START_ITER = 3000`，`rl_cfg.max_iterations = 2×3000 = 6000`。

## 1. 任务定义

机器人在直线走廊内连续通过多个"门洞式"限高障碍：障碍物是一块离地 0.050 m、宽 3 cm、
厚 1 cm 的限高板，板下的通行孔从地面一直到板底，机器人必须把**前躯干（含头部）**和
**后躯干**分别压低到期望高度以下才能穿过去。

两种受限情况的物理语义（2026-09-22 与用户核对）：

| 情况 | 板底下沿 | 要求 | 对应参考 |
| --- | --- | --- | --- |
| case1 | 0.050 m | 必须用脊柱（两段躯干不同高，沿 x 依次低头） | 待补：含脊柱的参考表维 |
| case2 | 0.065 m | 不用脊柱，两段躯干同高、整体压低即可 | 匍匐/低高度档（腿折起来，脊柱静止） |

几何尺度（用户口径）：预计算表的高度 = 躯干重心高度；躯干顶部 = 表高 + 0.024 m。
正常站立表高 0.055 → 顶部 0.079 m；趴平贴地重心约 0.024 m。所以 case2 的参考高度约
`0.065 − 0.024 ≈ 0.041 m`，case1 需要沿 path 的高度轨迹（`Viz_Tunnel_Path.png`）。

任务沿用 Slalom 的框架（5D 命令 + 预计算参考表 + 走廊一致性奖励），区别在于：
曲率恒为 0（`get_path_curvature` 返回全零），**参考表的高度维替代了 Slalom 的曲率维**。

## 2. 文件结构

```
src/mjlab/tasks/SQuRo_Tunnel/
├── SQuRo_Tunnel_env_cfg.py   # 观测/动作/奖励/终止/场景 (1024 envs, 20 s, decimation=4)
├── config/{__init__.py, rl_cfg.py}   # 任务注册 + PPO (512-256-128, 6000 iter, 24 steps/env)
├── mdp/
│   ├── path.py        # 洞几何常量 + 前/后肢期望高度方波 + 双走廊超额
│   ├── command.py     # TunnelCommand: 5D 命令 [vel, h_f, h_h, gait, κ=0], Phase0/Phase1
│   ├── reference.py   # [模式3][高度档4][相位200][14关节] 预计算表 + 速度表
│   ├── rewards.py     # 模仿/高度/速度/走廊/朝向/动作平滑/能耗
│   ├── curriculums.py # _FIXED_WEIGHTS (无阶段课程), PHASE1_START_ITER=3000
│   ├── events.py      # reset_model: 固定起点 (0,0,0.06), 面朝 +X
│   ├── terminations.py# check_fallen (root 重力投影 < 0.2)
│   ├── entity.py      # HoleEntity: 限高板薄板 box (类名保留 Hole*, 泛指"洞")
│   ├── indices.py     # F/H/head body、site、关节索引统一解析
│   └── Bio_Data/      # Trot_F.csv / Trot_H.csv (足端 Y/Z 轨迹, IK 生成参考表)
└── rl/runner.py       # SQuRoOnPolicyRunner (带 ONNX 导出)

src/mjlab/scripts/Tunnel/verify_phase0_baseline.py   # Phase0 基线验证 (表/命令/速度/实测高度)
```

数据流：`sample_tunnel_positions`（每 episode 采 3 个洞）→ `get_front/rear_center_height`
（按实际 x 出方波期望高度）→ `TunnelCommand._update_phase1`（写高度命令 + 速度规则）→
`compute_height_reward` / `compute_corridor_reward`（跟踪与包络约束）+ `reference.py`（步态参考）。
Phase0 则跳过 path，直接用随机高度命令查参考表。

## 3. 坐标与洞几何

世界：+X 前、+Y 左、+Z 上。SQuRo 的 `body+X ≠ 物理前向`：F_body 需 `atan2` 后减 π/2，
H_body 需加 π/2（详见 `PROJECT.md` 坐标系一节）。本任务前进方向恒为 +X。

| 常量 (`path.py`) | 值 (m) | 含义 |
| --- | --- | --- |
| OBSTACLE_LENGTH | 0.03 | 洞宽 a |
| TUNNEL_BOTTOM | 0.050 | 洞下沿 = 限高板板底高度 |
| TUNNEL_THICKNESS | 0.01 | 板厚（实体 `size[2]=0.005`，几何体从板底向上长） |
| HEIGHT_NORMAL / HEIGHT_LOW | 0.055 / 0.02 | 正常段 / 低高度段期望高度 |
| TRANSITION_LENGTH | 0.0 | 线性过渡长度 → 当前是方波 |
| FRONT_DOWN_OFF / FRONT_UP_OFF | 0.09 / 0.065 | 前肢中心：降起点 x−9 cm，升起点 x+5+a/2 |
| REAR_DOWN_OFF / REAR_UP_OFF | 0.025 / 0.07 | 后肢中心：降起点 x−4+a/2，升起点 x+4+a |
| CORRIDOR_FRONT/REAR_HALF_HEIGHT | 0.025 | 走廊半高 |
| BODY_SEG_HALF_HEIGHT | 0.025 | 简化身体段半高（三段同高 5 cm） |

F 段窗口 `[x−0.09, x+0.065]`，H 段窗口 `[x−0.025, x+0.07]`；F–H 中心间距 0.09，因此
**后肢开始降时前肢刚好开始升**，双低区间宽度只有 0.005 m（`f1267a6` 的设计意图）。

简化模型（设计约定，不是几何真值）：身体是三个 5 cm 高的块——前段（头部+前躯干）走前肢轨迹、
后段（后躯干）走后肢轨迹，走廊判据按"段的中心 z"计算。洞宽 3 cm 描述的是限高板的宽度，
不限制躯干宽度，所以这是**理想化抽象**，不要按真实碰撞体去理解。

洞位置采样（`sample_tunnel_positions`，Phase1，episode 内固定）：
第一个洞 x∈[0.15, 1.4]，洞间距 ≥ 0.30 + U(0, 0.2)，共 `TUNNEL_NUM = 3` 个。
注意：间距抖动使最后一个洞可能超过 `TUNNEL_MAX_X=2.0`，而 20 s episode（≈2 m）也到不了
最后一个洞——这个常量目前只是软约束，尚未收紧。

## 4. 走廊判据与误差口径

```
e_i = |z_seg − z_ref(x_seg)| + BODY_SEG_HALF_HEIGHT
v_i = max(0, e_i − CORRIDOR_HALF_HEIGHT)
r   = exp(−σ_corridor · v²)
```

前肢走廊取头部（`Neck_pitch_Link`）与前躯干（`F_body_Link`）超额的最大值，后肢走廊只看
`H_body_Link`，最后 `r = (r_f + r_h)/2`。

因为 `e` 里先加了段半高 0.025，而走廊半高也是 0.025，所以**死区实际为 0**：
段中心允许范围 = 期望高度 ± (走廊半高 − 段半高) = ±0.02 m。含义是"身体中心必须贴着
期望高度 ±2 cm"，正常段对应中心 z ∈ [0.035, 0.075]，低段对应 [0.0, 0.04]。

该判据**没有直接约束机体上沿与限高板下沿的关系**：上界是由"低段走廊中心上限 0.04 m +
段半高 0.025 = 上沿 0.065 m > 板底 0.050 m"隐含允许的，真正把高度压下去的是
`compute_height_reward`（σ=1000）。若要在仿真里硬性检验，回放时用
`--enable_collision True` 让限高板参与碰撞（`HoleEntity` 支持 `set_collision`）。

## 5. 奖励与权重

奖励权重全部集中在 `mdp/curriculums.py` 的 `_FIXED_WEIGHTS`（`env_cfg` 里 weight 一律 1.0）。

| 奖励项 | 权重 | σ | 说明 |
| --- | --- | --- | --- |
| mimic_pos | 5.0 | 腿 5.0 / 脊柱 10.0 | `(r_leg + r_spn)/2 + 0.3·r_neck`，误差对默认关节角 |
| mimic_vel | 2.5 | 0.1 | 同上分组，参考速度按步频缩放 |
| height | 2.5 | 1000 | 前/后躯干高度各占一半，命令高度即期望高度 |
| track_vel | 4.0 | 50 | F_body 局部前向速度跟踪；侧向/垂向各 1.0（σ_yz=50） |
| corridor | 8.0 | 50 | 前后走廊各半，见 §4 |
| track_head | 5.0 | 20 | F_body 物理前向对齐 0°（直行） |
| action_L1 / L2 | 0.5 / 0.5 | — | 腿与脊柱分开计，neck 跟脊柱同权重 |
| energy | 0.5 | — | Σ\|q̇·τ\| 对 14 个执行器 |

日志项（`env.extras["log"]`）：`Data/vel_actual`、`Data/vel_des`、`Data/height_actual`、
`Data/head_error`、`Data/corridor_front_excess`、`Data/corridor_rear_excess`。

## 6. 训练阶段与速度规则

`_STEPS_PER_ITER = 24`（每 iter 24 步 × 0.02 s ≈ 0.48 s 仿真），边界见 §0。

Phase 0（iter < 3000，无障碍）：按**受限模式**采样高度命令，`mode = randint(0,3)` 各 1/3：

| mode | 含义 | h_f | h_h | n（低高度肢数） | 速度 |
| --- | --- | --- | --- | --- | --- |
| 0 | 都高 | U{0.04,0.045,0.05,0.055,0.06} | 同 h_f | 0 | `v_base·f·(h/0.06)·2` |
| 1 | 前低后高 | 0.02 | U{0.04…0.06} | 1 | `v_base·f·(h_h/0.06)·1` |
| 2 | 前高后低 | U{0.04…0.06} | 0.02 | 1 | `v_base·f·(h_f/0.06)·1` |

- **配对约束**：一侧低于 `HEIGHT_LOW_THRESHOLD=0.04` 时另一侧必 ≥0.04；**双低不采样**
  （低高度肢冻结不动，双腿都冻结则无法前进）。
- **速度公式**：`v = v_base × 步频 f × 高度缩放 scale × (2 − n)`，`scale = 运动侧高度/0.06`，
  `v_base = PHASE0_V_BASE = 0.125 m/s`（正常高度 1 Hz；旧版 2 Hz 基准 0.25 折合）。
  本质是"速度 = 步频 × 步幅"，步幅受高度缩放与参与运动的腿数影响。
- 高度命令在 episode reset 时采样一次（`resampling_time_range=(20,30) s` ≥ 20 s episode，
  周期重采样实际不触发）。

Phase 1（iter ≥ 3000，有洞）：`_update_command` 每步用实际 x 查方波，高度命令 = 期望高度，
速度用同一公式（`n` 由期望高度是否低于 0.04 判定；双低时 `VEL_STOP=0`）：

| 状态 | n | 速度 |
| --- | --- | --- |
| 双正常 | 0 | `0.125·f·(h/0.06)·2 = 0.25·f·scale` |
| 单低 | 1 | `0.125·f·(h/0.06)·1` |
| 双低 | 2 | `0`（步态相位短暂冻结，"停下来缩一下"） |

速度命令同时驱动 `reference.py` 的相位推进（`phase += gait_freq·dt`）。参考表按
（模式, 高度档）索引，见 §4a。

### 4a. 参考表（2026-09-23 迁移旧版）

`reference.py` 预计算表为 `[模式 3][高度档 4][相位 200][14 关节]` + 同形状速度表，
`HEIGHT_LIST = [0.02, 0.04, 0.05, 0.06]`，`BASE_HEIGHT = 0.06`。生成规则（对齐 `v0801` 旧版）：

- 每条腿的目标足端轨迹 = CSV(`Trot_F/H.csv` 的 `Y_mean/Z_mean`，200 点) 的**偏移量按
  `height_scale = h/0.06` 等比缩放**（x 与 z 同乘），`x_offset` 前 0.0 / 后 −0.01；
- `h < 0.04` 或该肢处于低模式时**冻结**：`x=0.005`(前)/`0.002`(后)、`z=−0.02`，不缩放；
- 脊柱四关节参考恒 0（低高度靠腿折起来实现，实测冻结姿态正好对应重心 21.7 mm）；
- 颈 pitch 固定 −0.3；
- 速度表 = 位置沿相位的中心差分（相位周期 1），运行时再乘步频；
- 前腿肘关节 IK 在 `a2 > 2` 时取 `+2π`（旧版口径；旧实现写成 `−2π` 会给出 −11.9 rad 的废值）。

运行时索引：`mode_from_heights(h_f, h_h)` 定模式，前/后肢各自按高度就近取档，前肢关节取前肢档、
后肢关节取后肢档。实测（`verify_phase0_baseline`，跑完整周期）：

| 模式 | 高度档 | 前重心 | 后重心 | 顶部(前) |
| --- | --- | --- | --- | --- |
| 都高 | 0.06 | 51.2 | 53.4 | 75.2 |
| 都高 | 0.05 | 43.4 | 45.2 | 67.4 |
| 前低 | 0.055 | 29.6 | 44.3 | 53.6 |
| 后低 | 0.055 | 38.9 | 25.5 | 62.9 |
| 任一 | 0.02 | 21.9 | 23.8 | 45.9 |

## 7. 回放与诊断

```powershell
# 回放 (Phase 自动识别: align_iter = train_iter − 10)
uv run python -B -m mjlab.scripts.SQuRo_Tunnel_play --checkpoint_file logs/rsl_rl/SQuRo_Tunnel/<run>/model_3999.pt
# 无 checkpoint 自检 (不启动 viewer, 跑 N 步打印高度/期望/奖励)
uv run python -B -m mjlab.scripts.SQuRo_Tunnel_play --agent zero --smoke_steps 50 --no-video
# 训练 (必须显式指定 tensorboard; 一阶段 0~3k, 二阶段 3k~6k)
uv run train Mjlab-SQuRo-Tunnel --agent.logger tensorboard
# Phase0 基线验证 (参考表结构 / 命令配对约束 / 速度公式 / 实测躯干高度)
uv run python -B -m mjlab.scripts.Tunnel.verify_phase0_baseline
# 期望高度轨迹图 (常量直接取自 path.py)
uv run python -B -m mjlab.scripts.Viz_Path.Viz_Tunnel_Path
```

回放脚本要点：`PlayConfig.fixed_tunnel_xs` 默认 `"0.30,0.60,0.90"`，会把洞位置锁死并
按该位置重建限高板实体（保证"看到的板"与"高度轨迹"一致），传 `sample` 则恢复随机采样；
`--enable-collision` 让板参与碰撞（布尔开关用 `--xxx` / `--no-xxx` 形式）。
CSV（160 列，30 步示例）含关节 pos/vel/acc/torque/ref、足端接触与位置、5D 命令、
前后肢高度与实际/期望/误差、走廊超额、F/H 物理前向 heading、三个洞位置。
受限环境下 `logs/` 不可新建目录时，输出自动回退到系统临时目录（脚本会打印 `[WARN]`）。

## 8. 踩坑与未决问题

1. **起点高度与期望高度不一致**：`events.reset_model` 把 root 放到 z=0.06，而实测站立
   体心 z=0.0563、期望高度 HEIGHT_NORMAL=0.055 —— 起始就有 5~6 mm 误差，`corridor_front_excess`
   开局即刻为 0.0055，奖励已经打折。候选修法是把 root 初始 z 对齐到站立高度（0.056 左右）。
2. **旧 checkpoint 已不兼容**：`2026-09-21_20-23-57/model_3999.pt` 是迁移前训练的
   （参考表是"曲率维"版本），迁移后它看到的 `ref_joint_pos/vel` 语义已变，回放只能当历史对照。
3. **Phase1 命令重采样失效**：`_update_command` 的 Phase1 分支不递减 `time_left`，
   `resampling_time_range` 形同虚设（当前无副作用，但别指望它做事）。
4. **限高板全程无碰撞**：`env_cfg` 里占位实体 `contype=conaffinity=0`，训练时机器人可以
   直接穿过板；穿洞成功完全靠奖励塑形，没有成功/失败判据，终止只有超时与跌倒。
5. **走廊上界缺失**：见 §4，机体上沿没有硬约束，若高度奖励权重被调低，策略可能贴板过洞。
6. **高度映射未校准（迁移后最重要的一条）**：实测"命令高度 ≠ 实际躯干重心高度"——
   命令 0.06 实测前 51.2 / 后 53.4，命令 0.05 实测 43.4 / 45.2，**前躯干系统性偏低 8~12 mm**
   （后躯干接近）。原因是参考轨迹前后不对称（腿长 0.080 vs 0.076、CSV 行程 67.6 vs 55.2 mm、
   肩髋安装高度不同），不是 `height_scale` 口径问题（两种口径只差 0.3~1.5 mm）。
   低高度档 0.02 反而是准的：冻结腿姿态实测重心 21.7 mm ≈ 目标 0.02。
7. **`HEIGHT_LOW = 0.02` 的可行性已核实**：靠腿完全折起来（冻结姿态）可达，实测重心
   21.7 mm、脚不离地；但**双腿同时压这么低做不到**（对称腿动作实测段中心最低 36~44 mm），
   所以配对约束（禁止双低）是硬性前提。
8. **N=1 的洞几何不成立**：0.03 m 宽的孔容不下 0.07 m 宽的躯干，这是简化模型；真实场景
   应把"洞"理解为限高门洞（板下空间），不要按孔洞去调参。
9. **`Viz_Tunnel_Path.py` 里的 `OBSTACLE_X_LEFT=0.185` 与 baselink 参考偏移（x−14 / x+8+a）
   只用于绘图**，代码中 baselink 已不作为控制点；改轨迹常量时只需改 `path.py`。
10. **低高度阈值存在两套口径**：参考/配对用 `HEIGHT_LOW_THRESHOLD=0.04`（旧版口径），
    速度状态判定历史上用 `(0.055+0.02)/2 = 0.0375`；Phase1 现已统一到 0.04。

## 9. 下一步计划（按优先级）

- [ ] 跑通 Phase0（0~3k）：确认三种模式都能学会（都高正常行走 / 单低侧身），
      观察 `Data/height_actual`、`Data/vel_actual`、`Data/corridor_*_excess` 曲线。
- [ ] 校准高度映射（踩坑 6）：按 `height_scale` 实测曲线反解标定系数，让前/后躯干
      都命中命令高度；这是"命令=实际"的前提，也是后续 case1/case2 的基础。
- [ ] 参考表加"受限情况"维（case1 板底 0.050 需含脊柱、case2 板底 0.065 用匍匐），
      脊柱轨迹按参数化实现（脊柱 CSV 不在本仓库）。
- [ ] 修起点高度（踩坑 1）与 Phase1 重采样（踩坑 3），一次只改一件。
- [ ] Phase1 训练前用 `--smoke_steps` + `--enable-collision` 检查低高度轨迹能否不碰板通过。
- [ ] 奖励权重若要做阶段调整，统一迁到 `mdp/curriculums.py` 的 `_CURVES` 形式，
      与 Slalom/Backup 的常量管理方式对齐。
