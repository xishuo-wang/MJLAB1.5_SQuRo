# SQuRo_Hole 单板受限空间高度课程

任务 ID：`Mjlab-SQuRo-Hole`（`Mjlab-SQuRo-Hole-Play` 为回放配置）。SQuRo 四足机器人以 6 维命令通过一块限高板，训练环境不开启板体接触，由**独立真实碰撞考核环境**推进高度课程；策略需在虚拟净空约束下学习前/后段低姿态、侧向走廊与速度跟踪。

当前代码对应 v3.1 课程协议（2026-10-18 起），并已完成旧三板布局、旧检查点迁移、旧课程推进协议的清理（2026-10-19）。本目录 `README.md`、`奖励设计.md`、`回放与工具.md` 是当前唯一有效文档；历史变更与旧实现见 `历史/`。

## 快速入口

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONDONTWRITEBYTECODE='1'; $env:UV_CACHE_DIR="$env:TEMP\uv_cache_mjlab"

uv run train Mjlab-SQuRo-Hole                                  # 训练 (默认 wandb)
uv run play Mjlab-SQuRo-Hole-Play --checkpoint_file <path>     # 回放/录视频
uv run python -B -m mjlab.scripts.Hole.evaluate_checkpoints --checkpoint <path>   # 冻结评估
uv run python -B -m mjlab.scripts.SQuRo_Hole_Replay           # 无策略参考回放
uv run python -B -m mjlab.scripts.Hole.visualize_hole_schedule  # 高度-位置表可视化
```

验收入口与诊断见 [回放与工具.md](回放与工具.md)。

## 1. 环境接口

| 项 | 内容 |
| --- | --- |
| 动作 | 14 维关节位置增量：`JointPositionAction(scale=0.3, use_default_offset=True)` |
| 观测 | 113 维：`actions`(2×14=28) + `ref_joint_pos`(14) + `ref_joint_vel`(14) + `actuator_pos`(14) + `actuator_vel`(14) + `actuator_force`(14) + `base_ang_vel`(3) + `base_lin_vel`(3) + `projected_gravity`(3) + `command`(6) |
| 命令 | 6 维：`velocity_x`、`height_f`、`height_h`、`gait_frequency`、`curvature`、`obstacle_height`（板高先验，单位 m） |
| 时序 | `timestep=0.002`、`decimation=5` → 100 Hz；回合 12 s；`num_steps_per_env=48`；训练 6000 iter；`num_envs=1024` |
| PPO | 网络 512-256-128 / ELU；lr 3e-4 adaptive、gamma 0.99、lam 0.95、epochs 5、minibatch 4、clip 0.2、entropy 0.01、seed 42 |

命令为每回合复位的目标（课程阶段内高度固定为回合级采样值）；`gait_frequency` 基准 2.0 Hz、`curvature` 基准 0。`obstacle_height` 列记录当前板底先验，无板时为 100 mm 占位。

## 2. 场景与单板几何

- 前三阶段（0～3000 iter）只创建机器人，不创建任何限高板，`obstacle_height` 为 100 mm 占位。
- 3000 iter 由 Runner 重建环境并创建单板：中心 `(0.2, 0, 0.070)`、半长 0.015 m、**半宽 0.3 m（总宽 60 cm）**、半厚 0.005 m；`position.z` 表示板底，随后按课程逐环境采样。
- 训练场景中板体 `contype=conaffinity=0`（仅用于板高采样、查表与监测）；独立考核环境才开启真实碰撞。
- 板体是 mocap 实体，每回合 reset 由 `entity.write_obstacle_bottom_z` 写入板底；虚拟净空惩罚与课程记录读取同一高度。
- 几何唯一来源：`mdp/config.py` 的 `OBSTACLE_POSITION`、`OBSTACLE_SIZE`；`mdp/entity.py` 负责构建与读写。旧三板布局已删除。

## 3. 高度查表规则

规则参数在 `mdp/config.py`，查表函数在 `mdp/curriculums.py`（`obstacle_rule_bounds`、`compute_height_commands`）。设板 X 范围为 `[x1,x2]`、板底为 `h`：

- `L=x1-0.150`、`M=(x1+x2)/2`、`R=x2+0.100`；查表位置为 `base_Link` 世界 X（`robot.data.root_link_pos_w[:,0]`），区间外前后目标均为 55 mm。
- 高板（`h>=0.065`）：板底按 `OBSTACLE_HEIGHT_EDGES` 每 5 mm 分桶映射到目标高度 `HEIGHT_LIST`，可用档位 40/45/50/55 mm。每回合按 50% 概率选择“等高”或“分段”：等高在 `[L,R]` 内前后同取该档；分段在 `[L,M)` 前端取该档、后端 55 mm，在 `[M,R]` 内交换。
- 低板（`h<0.065`）：按板底映射到低姿态标定档 20/25/30/35/40 mm；`base_Link` 在 `[L,M)` 时前端取该档、后端 55 mm，在 `[M,R]` 时交换。低于 50 mm 的板底仍钳制到 20 mm 档。
- 中点立即切换为前高后低；两段低姿态区间互斥；短板区间 `[0.035,0.315]` m，中点 0.200 m。

分桶函数 `height_bin_indices` 与命令、考核统计共用，统一按 float32 计算，避免 60 mm 等边界被统计到相邻挡位。

## 4. 速度命令与低姿态

速度由 `command.compute_cmd_vel_x(height_f, height_h)` 生成，训练、位置表回放和可视化共用：

- 前后高度先按 `h/BASE_HEIGHT` 缩放并取平均（`h<0.035` 时用标定区间线性缩放）；
- 再乘以低姿态连续减速系数 `1-(1-LOW_POSE_SPEED_RATIO)*a(min(h_F,h_H))`，`LOW_POSE_SPEED_RATIO=0.75`，`a` 由 40 mm 处的 0 线性到 20 mm 处的 1；
- 基准速度 `BASE_SPEED=0.2` m/s。20/55 mm → 0.075、30/30 mm → 0.0742、55/55 mm → 0.2 m/s（完整示例见 `历史/` 中的高度课程设计记录）。

低姿态腿部参考不使用在线逆运动学，而是按 `LOW_POSE_CALIBRATION`（20～40 mm，前后腿各两个关节角）分段线性插值，且该侧相位速度置零；脊柱俯仰按前后高度差插值到 `LOW_SPINE_PITCH=-0.65`；头颈参考固定 `NECK_REF_POS=(0,-0.3)`。

## 5. 参考轨迹

`mdp/reference.py` 预计算并缓存 [模式, 高度档, 相位, 14 关节] 参考表：

- 4 个高度模式：双高 0、前低 1、后低 2、双低 3（由命令高度 `<THRESHOLD_HEIGHT=0.040` 判定）；
- 8 个高度档（`FULL_HEIGHT_VALUES`：20/25/30/35/40/45/50/55 mm），相位 `TABLE_RESOLUTION=500`；
- 正常腿用 `Bio_Data` 的 FL/HR 光滑足端轨迹做逆解；受限期对侧腿用摆线轨迹（`CYCLOID_PARAMS`）；
- 速度表由位置表按相位差分得到，并按 `gait_frequency/BASE_FREQ` 缩放；
- 低姿态侧覆盖为标定角插值、相位速度为零；双高不同高时脊柱俯仰按实际高度差更新；
- 参考表在首次使用时按设备初始化，`initialize_tables(device, force_reload)` 可强制重建。

## 6. 课程学习

### 6.1 四阶段

| 阶段 | iter 区间 | 高度命令 | 奖励/约束 |
| --- | --- | --- | --- |
| 1 | 0～1000 | 正常档随机池 `[0.040,0.045,0.050,0.055]` | 模仿、速度、高度、姿态、足端；走廊权重 0 |
| 2 | 1000～2000 | 全档池 `[0.020…0.055]`，前后独立采样 | 同上 |
| 3 | 2000～3000 | 全档池随机 | 打开走廊惩罚（权重 5），走廊余量 30 mm→25 mm 线性收紧 |
| 4 | 3000 起 | 单板高度课程（板底 70→47 mm） | 打开虚拟碰撞惩罚；单板高度课程推进 |

阶段边界只由 `STEPS_PER_ITER=48` 与 `STAGE{1,2,3}_ITER=1000/2000/3000` 决定；奖励权重/σ 在 `_CURVES` 中按阶段取 4 个锚点（见 [奖励设计.md](奖励设计.md)）。

### 6.2 阶段 4：独立真实碰撞考核（v3.1）

- **课程序列**：70→68→66→64→62→60→58→56→54→52→50→49→48→47 mm（14 档，下限 47 mm）；46 mm 在无接触训练下不可达，已从课程移除。
- **训练无接触**：所有阶段训练环境板体碰撞关闭；3000 iter 重建只创建单板实体用于采样/监测。
- **独立考核环境**：复制训练配置，`HEIGHT_EVALUATION_ENVS=1024`、开启板体碰撞、冻结确定性策略；每次考核前后隔离随机数，轨迹不进入 PPO。
- **调度**：课程开始后 `HEIGHT_EVALUATION_PRETRAIN_ITERATIONS=500` 轮首次触发；普通模式每 `HEIGHT_EVALUATION_INTERVAL=100` 轮在 PPO 更新后执行；47 mm 通过后进入最终模式，间隔改为 `HEIGHT_EVALUATION_FINAL_INTERVAL=500` 轮，考核不停止。
- **单批口径**：每个环境只计首个完整回合；高度采样与训练一致（30% 固定当前 `Zmin`，其余在 `[Zmin,Zmax]` 均匀）；每批最低档有效回合需 ≥`HEIGHT_EVALUATION_MIN_SAMPLES=256` 且通过率 ≥60%。
- **推进/退档**：每次考核连续两批，两批全部达标且当前档驻留 ≥`HEIGHT_MIN_DWELL_ITERATIONS=100` 轮才降档（>50 mm 降 2 mm，≤50 mm 降 1 mm）；任一批不达标保持难度，跨考核不累计。普通模式连续 `RETREAT_FAILURE_LIMIT=5` 次考核失败退回课程序列上一档；退档后需连续 `RETREAT_REQUIRED_PASSES=2` 次考核达标才能重新推进。
- **最终模式采样**：47 mm 档通过一次考核后 `final_level_passed=True`，训练与考核按 8 个命令挡位等概率采样（挡内均匀）；最终验收要求两批整体通过率均 ≥`FINAL_ACCEPT_RATE=80%`，置 `final_accepted=True`（粘性，不撤销）。
- **训练侧分挡位反馈**：普通模式保留 30% 固定最低档，其余 70% 按各挡通过率缺口加权 `bin_weights[i]=1+2*max(0,0.6-rate_i)`（目标 `NORMAL_BIN_TARGET_RATE=60%`）；每次考核后更新，考核环境固定等概率采样、不继承该权重。
- **诊断**：每次考核写入 `<run>/collision_evaluations/evaluation_XXXXX.json`（含两批统计、挡位通过率、权重、耗时）；TensorBoard 记录 `Curriculum/*` 与 `Metrics/*`（指标始终取最近一次考核，降档后不清零）。

### 6.3 检查点与续训

`runner.save()` 在 `infos["obstacle_state"]` 写入：`stage`、`training_collision`、`evaluation_envs`、`layout`（单板几何/接触参数）、`high_obstacle_equal_height`、`height_course`（课程状态：`z_min/z_max`、驻留起点、批次计数、连续达标/失败、退档、`bin_weights`、`final_level_passed`、`final_accepted`、历史记录等）。

- 续训时 `runner.load()` 恢复课程状态；进入阶段 4 后首次采样前会重建单板训练场景。
- 旧检查点不保证可用：2026-10-19 清理后课程状态字段按当前版本严格读取，不再做缺失字段默认值/下限迁移；回放旧检查点可能只得到默认课程状态。
- 回放默认沿用检查点板几何与课程高度；`SQuRo_Hole_play` 的 `--collision` 可显式覆盖碰撞开关。

## 7. 奖励设计

12 个奖励/惩罚项、逐项公式、权重与 σ 曲线见 [奖励设计.md](奖励设计.md)。要点：`env_cfg` 中所有 `cfg.weight=1.0`，有效权重统一由 `mdp/curriculums.py` 的 `_CURVES` 提供；正奖励以 `exp(-σ·误差²)` 为主，惩罚为负值（接触/走廊用 `exp(-σe²)-1`）。

## 8. 终止与穿板判据

- `timeout`：12 s 超时（成功穿板后未跌倒的超时仍记成功）；`fallen`：`check_fallen`。
- `hole_passage_monitor` 终止项调用 `terminations.collect_passage_state`：每步采集 F/H 各 9 个躯干 site 的穿板状态并发布回合结果，**恒定返回 False**，不终止回合；训练环境仅作几何诊断，课程推进只由独立考核环境完成。
- 成功判据：先从前方进入（`front_max<x1`），两段躯干均完全越过板后沿（`x>x2`），两段都曾在板宽内，全程无侧绕/高度超限/跌倒。
- 失败码：1 未过板、2 侧绕、3 高度超限、4 跌倒、5 超时；侧绕判据使用实际板宽（总宽 60 cm，侧向绕行不可行）。

## 9. 验收与回归

```powershell
uv run python -B -m mjlab.scripts.Hole.verify_hole_obstacle_curriculum   # 课程/考核协议
uv run python -B -m mjlab.scripts.Hole.verify_hole_collision            # 单板几何/碰撞/状态保存回放
uv run python -B -m mjlab.scripts.Hole.verify_hole_rebuild_timing       # 3000 轮重建与首回合过滤
uv run python -B -m mjlab.scripts.Hole.verify_hole_corridor_commands    # 阶段边界/六维命令/走廊
uv run python -B -m mjlab.scripts.Hole.verify_hole_command_reset        # 命令复位与查表边界
uv run python -B -m mjlab.scripts.Hole.verify_hole_baseline             # 参考表/课程/观测维度
uv run python -B -m mjlab.scripts.Backup.verify_backup_stage_rewards    # 跨任务回归 88 项
```

诊断产物统一在 `logs/diagnostics/SQuRo_Hole/<主题>_<日期>/`（路径常量见 `scripts/_diagnostics.py`），训练产物在 `logs/rsl_rl/SQuRo_Hole/`。

## 10. 已知边界与注意事项

- 虚拟碰撞惩罚只覆盖前后躯干各 9 个 site（不含头部/腿部），实际碰撞以独立考核环境为准；训练无接触，不代表策略已学会接触支撑。
- `final_accepted` 为单向粘性标记；80% 是整体通过率，单挡位低通过率可能被整体掩盖，需看 `collision_evaluations/*.json`。
- 47/48 mm 对策略很敏感：历史实验显示 46 mm 会拖垮 47 mm 能力，故下限固定在 47 mm。
- 独立考核暂不支持分布式训练（`is_distributed` 会直接报错）；每次 1024 环境两批考核约增加 20% 墙钟（前 3000 轮不考核）。
- 任务级命名保留 `Hole`（`HoleCommand`、`hole_cmd`、`Mjlab-SQuRo-Hole`）；`PassMonitor` 状态对象与 `collect_passage_state` 钩子已统一。
- 本文档之外的日期化变更记录属历史资料，不作为当前行为依据。

## 11. 文档索引

| 文件 | 内容 |
| --- | --- |
| `README.md` | 任务设计、课程协议、检查点、验收与边界（本文件） |
| `奖励设计.md` | 12 个奖励/惩罚项公式、`_CURVES` 权重与 σ、日志键 |
| `回放与工具.md` | 训练/回放/评估/可视化入口、诊断产物、验收命令 |
| `历史/` | 按日期保留的旧实现与变更记录（内容保持原样，仅修正链接路径） |
