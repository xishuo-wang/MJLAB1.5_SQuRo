# SQuRo_Tunnel 技术细节

任务 ID：`Mjlab-SQuRo-Tunnel`（钻洞 / 越障，四足 + 脊柱直行场景）
代码：`src/mjlab/tasks/SQuRo_Tunnel/`，回放：`src/mjlab/scripts/SQuRo_Tunnel_play.py`

当前状态：**代码就绪，尚未训练**（最后一次代码改动 2026-08-06；`logs/rsl_rl/SQuRo_Tunnel/`
下无有效实验）。本文件是参考手册；实验数据请在此按时间追加，不要覆盖历史结论。

## 1. 任务定义

机器人在直线走廊内连续通过多个"门洞式"限高障碍：障碍物是一块离地 0.050 m、宽 3 cm、
厚 1 cm 的限高板，板下的通行孔从地面一直到板底，机器人必须把**前躯干（含头部）**和
**后躯干**分别压低到期望高度以下才能穿过去。

任务沿用 Slalom 的框架（5D 命令 + 预计算参考表 + 走廊一致性奖励），区别在于：
曲率恒为 0（`get_path_curvature` 返回全零），期望高度的变化来自沿 X 的方波轨迹而不是转向曲率。

## 2. 文件结构

```
src/mjlab/tasks/SQuRo_Tunnel/
├── SQuRo_Tunnel_env_cfg.py   # 观测/动作/奖励/终止/场景 (1024 envs, 20 s, decimation=4)
├── config/{__init__.py, rl_cfg.py}   # 任务注册 + PPO (512-256-128, 4000 iter, 24 steps/env)
├── mdp/
│   ├── path.py        # 洞几何常量 + 前/后肢期望高度方波 + 双走廊超额
│   ├── command.py     # TunnelCommand: 5D 命令 [vel, h_f, h_h, gait, κ=0], Phase0/Phase1
│   ├── reference.py   # 26κ×50相位×14关节预计算表 (Bio_Data 自带, κ=0 → 只用第 0 bin)
│   ├── rewards.py     # 模仿/高度/速度/走廊/朝向/动作平滑/能耗
│   ├── curriculums.py # _FIXED_WEIGHTS (无阶段课程), PHASE1_END_ITER=4000
│   ├── events.py      # reset_model: 固定起点 (0,0,0.06), 面朝 +X
│   ├── terminations.py# check_fallen (root 重力投影 < 0.2)
│   ├── entity.py      # HoleEntity: 限高板薄板 box (类名保留 Hole*, 泛指"洞")
│   ├── indices.py     # F/H/head body、site、关节索引统一解析
│   └── Bio_Data/      # Trot_F.csv / Trot_H.csv (足端 Y/Z 轨迹, IK 生成参考表)
└── rl/runner.py       # SQuRoOnPolicyRunner (带 ONNX 导出)
```

数据流：`sample_tunnel_positions`（每 episode 采 3 个洞）→ `get_front/rear_center_height`
（按实际 x 出方波期望高度）→ `TunnelCommand._update_phase1`（写高度命令 + 速度规则）→
`compute_height_reward` / `compute_corridor_reward`（跟踪与包络约束）+ `reference.py`（步态参考）。

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

`PHASE1_END_ITER = 4000`，`_STEPS_PER_ITER = 24`（每 iter 24 步 × 0.02 s ≈ 0.48 s 仿真）。

Phase 0（iter < 4000，无障碍）：`PHASE0_HEIGHTS = [0.02, 0.04, 0.045, 0.05, 0.055, 0.06]`
独立采样前/后肢高度，`vel = PHASE0_BASE_SPEED(0.25) × min(h_f,h_h)/0.06 × gait_freq`，
高度命令按 `resampling_time_range=(20,30) s` 周期重采样——而 episode 只有 20 s，
所以**训练中每集只采样一次，重采样实际不触发**。

Phase 1（iter ≥ 4000，有洞）：`_update_command` 每步用实际 x 查方波，写高度命令与速度：

| 状态 | 判定 | 速度命令 |
| --- | --- | --- |
| 双正常 | `z_ref > 0.0375` 且 `z_ref > 0.0375` | `BASE_VEL=0.1 × gait_freq` |
| 单低 | 任一肢低 | `VEL_LOW=0.05 × gait_freq` |
| 双低 | 两肢都低（窗口重叠 0.005 m） | `VEL_STOP=0.0` |

速度命令同时驱动 `reference.py` 的相位推进（`phase += gait_freq·dt`），所以双低时步态相位
会短暂冻结——这正是"停下来缩一下"的预期行为，不是 bug。曲率恒 0，因此参考表只走第 0 个
曲率 bin，脊柱四关节参考恒为 0、颈 pitch 固定 −0.3。

## 7. 回放与诊断

```powershell
# 回放 (Phase 自动识别: align_iter = train_iter − 10)
uv run python -B -m mjlab.scripts.SQuRo_Tunnel_play --checkpoint_file logs/rsl_rl/SQuRo_Tunnel/<run>/model_3999.pt
# 无 checkpoint 自检 (不启动 viewer, 跑 N 步打印高度/期望/奖励)
uv run python -B -m mjlab.scripts.SQuRo_Tunnel_play --agent zero --smoke_steps 50 --no-video
# 训练 (必须显式指定 tensorboard)
uv run train Mjlab-SQuRo-Tunnel --agent.logger tensorboard --agent.max-iterations 4000
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
2. **Phase0 速度不是常数**：`_resample_phase0` 按 `min(h_f,h_h)/0.06` 连续缩放，且 `fixed_velocity`
   一旦设置就变成 0（`_get_velocity(base_vel=0.0)`），回放时若同时给了 `--fixed_velocity`
   与 Phase0 高度会得到零速度，需要用 `--fixed_height_f/h` 固定高度再看速度。
3. **Phase1 命令重采样失效**：`_update_command` 的 Phase1 分支不递减 `time_left`，
   `resampling_time_range` 形同虚设（当前无副作用，但别指望它做事）。
4. **限高板全程无碰撞**：`env_cfg` 里占位实体 `contype=conaffinity=0`，训练时机器人可以
   直接穿过板；穿洞成功完全靠奖励塑形，没有成功/失败判据，终止只有超时与跌倒。
5. **走廊上界缺失**：见 §4，机体上沿没有硬约束，若高度奖励权重被调低，策略可能贴板过洞。
6. **`HEIGHT_LOW = 0.02` 的可行性未标定**：正常站立体心 0.0563，低高度要求 0.02，是
   3.6 cm 的下降；腿长（前 0.04+0.04、后 0.04+0.036）能否在保持步态的同时压这么低、
   有没有机体触地，需要实测（建议先跑 Phase0 高度跟踪，再决定是否抬高 `HEIGHT_LOW`）。
7. **N=1 的洞几何不成立**：0.03 m 宽的孔容不下 0.07 m 宽的躯干，这是简化模型；真实场景
   应把"洞"理解为限高门洞（板下空间），不要按孔洞去调参。
8. **`Viz_Tunnel_Path.py` 里的 `OBSTACLE_X_LEFT=0.185` 与 baselink 参考偏移（x−14 / x+8+a）
   只用于绘图**，代码中 baselink 已不作为控制点；改轨迹常量时只需改 `path.py`。

## 9. 下一步计划（按优先级）

- [ ] 跑通 Phase0：确认直行速度（命令 0.1 m/s）、步频 1 Hz、高度跟踪在 σ=1000 下可收敛，
      记录 `Data/height_actual` 与 `Data/vel_actual` 的收敛曲线。
- [ ] 标定 `HEIGHT_LOW`：从 0.02 逐步抬高（0.025 / 0.03 / 0.035）看机体是否触地，确定可行下界。
- [ ] 修起点高度（踩坑 1）与 Phase1 重采样（踩坑 3），一次只改一件。
- [ ] Phase1 训练前先用 `--smoke_steps` + `--enable_collision True` 检查低高度轨迹是否真的
      能不碰板通过，再开 4000 iter 之后的课程。
- [ ] 奖励权重若要做阶段调整，统一迁到 `mdp/curriculums.py` 的 `_CURVES` 形式，
      与 Slalom/Backup 的常量管理方式对齐。
