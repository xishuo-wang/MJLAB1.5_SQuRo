# HANDOFF — SQuRo 翻正任务模仿学习

日期：2026-09-11 ｜ 任务：`Mjlab-SQuRo-Backup`

## 1. 当前目标（Current Goal）

先用**手调状态机的翻正轨迹**做参考，通过基本的**模仿奖励**让 RL 学会翻正行为；
行为优化与受限空间（窄缝/低顶棚）留到后续阶段。

参考实现：`src/mjlab/scripts/SQuRo_backup_Replay.py`（手调状态机，λ=1/3 均可零重试通过）。

## 2. 已完成工作（Completed Work）

### 2.1 手调脚本与训练参考表已完全对齐

| 项目 | 改动 |
|---|---|
| `mdp/timing.py` | `P1_RECOVER_DURATION` 0.65 → **0.15**（手调实测最小可行值）→ `P1_END=0.80`、`P2_END=0.95` |
| `mdp/reference.py` | `_HL_HOLD` `(-1.40,-0.25)` → **`(-1.50,-0.25)`**，与手调 `HL_HOLD` 一致 |
| `SQuRo_Backup_env_cfg.py` | `JointPositionActionCfg.scale` 0.5 → **0.3** |
| `scripts/SQuRo_backup_Replay.py` | `action_scale` 改为**从 env_cfg 自动读取**（可显式覆盖） |

**动作覆盖性核算**（`scripts/Backup/check_action_scale_coverage.py`，`clip_actions=6.0`）：
最紧的是两个扭转关节，期望 ±1.57 需 action **5.233**，余量 **1.15×**；HL/HR_hip 需 4.667，余量 1.29×。全部在限内。

### 2.2 姿态判据改为标记 site 方案

`data.body_link_quat_w` 的参考系**至今未定论**（见 `docs/SQuRo_Backup_技术细节.md` §3），
旧判据 `2(yz+wx)` 在翻正过程中会失效。已在 XML 为两个躯干段各加一对腹/背标记 site：

```xml
<site name="F_body_belly_site" pos="0.00  0.021 -0.023"/>  <site name="F_body_back_site" pos="0.00 -0.023 -0.023"/>
<site name="H_body_belly_site" pos="0.00 -0.021  0.008"/>  <site name="H_body_back_site" pos="0.00  0.023  0.008"/>
```

`mdp/command.py` 的 `_check_S1` / `_check_S2` 改为 `belly_z < back_z ⇔ 该段已翻正`（世界坐标，零四元数依赖）。
验收：与独立重算的地面真值**不一致帧数为 0**（`scripts/Backup/verify_gate_predicates.py`）。

### 2.3 关键发现（已归档到文档）

- **`base_up` 不能判断翻正**：它是万向节中心的朝向，动作中会自己转过 ~85°~180°。
- **必须分段判断**：F 段与 H 段局部坐标背腹轴相反。
- **时序对应**：P1 = T1+T2，P2 = T3，P3 = T4+T5；S1 = 仅后段翻正，S2 = 两段都翻正。
- **翻滚发生在 T3**（P2 段），T1/T2 只做脊柱形变。

### 2.4 训练管线已验证可跑

```powershell
uv run train Mjlab-SQuRo-Backup --env.scene.num-envs 96 --agent.max-iterations 120 \
  --agent.logger tensorboard --log-root "$env:TEMP\sq_learn1"
```

`_STEPS_PER_ITER` 60 → **300**（rollout 从 0.6 s 提到 3.0 s，原先一个 rollout 装不下完整翻正）。

### 2.5 训练结果：收敛到平凡解（**未完成目标**）

96 envs × 120 iters（约 2 分钟/20 iter，3300 steps/s）：

| iter | Mean reward | mimic_pos | height | episode len | milestone_s1/s2/success |
|---|---|---|---|---|---|
| 0 | 5.25 | 0.53 | 0.84 | 176 | 0 / 0 / 0 |
| 22 | 48.28 | 3.29 | 4.85 | 1000（撞上限）| 0 / 0 / 0 |
| 95 | 102.84 | 6.17 | 4.97 | 1000 | 0 / 0 / 0 |

**结论：策略学会"仰面躺 + 部分跟随参考"，从未触发任何里程碑。**
`Data/height_actual` 全程 0.0311~0.0320（仰面高度），`backup_s1_ok` 恒为 0。

## 3. 下一步计划（Next Steps）

- [ ] **先跑完 `baseline_trivial_policy.py` 的基线对照**（脚本已修好两个 bug：action_dim 硬编码为 14、
      `env.step` 返回 5 元组，但最后一次运行仍 exit=1 未看到输出，需再查）。
      目的：确认"零动作仰面躺"的累计回报，与策略的 102.84 对比，量化平凡解的高度。
- [ ] **诊断为何走不出平凡解**，候选方向（按优先级）：
  - 奖励构成：`height` 已吃满 4.97/5.0，说明径向高度参考在仰面时就接近满分；
    `mimic_pos` 仅 6.17/11。考虑提高 `mimic_pos` 权重 或 降低初始 `sigma` 让梯度更宽。
  - 课程：`TIME_SCALE_MIN_START=3.0` 使 λ∈[3,6]，rollout（3.0 s）只覆盖 `P2_END×λ` 的一部分，
    策略在单个 rollout 内看不到"站立"阶段。考虑把 `TIME_SCALE_MIN` 起点降到 1.0~2.0，
    让早期 rollout 覆盖完整动作。
  - `weight_mimic_vel=5.0` 但 `sigma_*_vel=0.5`：速度项可能过松，靠"不动"就能拿分。
- [ ] 每轮训练后检查 `Data/milestone_s1` / `milestone_s2` / `backup_s1_ok` 是否开始非零。
- [ ] 目标达成后再进入行为优化与受限空间设计。

## 4. 踩过的坑 & 失败尝试（Pitfalls）

1. **`body_link_quat_w` 参考系陷阱**：docstring 写 "in world frame"，但 reset 后
   `root_link_quat_w` 为单位四元数，实测不是世界系。曾误判 `_body_up ≡ sin(θ_s+θ_b)`，
   被 λ=1 早期几个点的巧合骗过，后被完整数据推翻。**结论：不要基于该量写物理判断。**
2. **`env.action_space.shape` 是 `(num_envs, dim)`**，取 `shape[0]` 会得到 64 而不是 14。
3. **`env.step()` 返回 5 元组** `(obs, rew, dones, timeouts, extras)`，不是 4 元组。
4. **受限环境下必须关掉 wandb**：`--agent.logger tensorboard`。
   wandb-core 需要命名管道，沙箱会拒绝，报 `ServicePollForTokenError`。
5. **测试假失败**：`.pytest_cache` / `__pycache__` 写不进去时会读到**过期字节码**，
   表现为结果与源码不符。必须加 `-p no:cacheprovider` 且设 `PYTHONDONTWRITEBYTECODE=1`。
6. **edit 工具曾吞掉换行**：一次替换把 `def test_...()` 并进了注释行，函数未定义，
   导致测试行为诡异。改动后应复核文件实际内容。
7. **全量测试在本机跑不了**：`mujoco_warp` kernel cache 只读，产生数千处 `PermissionError`。
   验证 SQuRo 改动只需跑：
   `tests/test_squro_backup_timing.py`、`tests/test_squro_backup_replay.py`、`tests/test_asset_zoo.py`（25 passed）。

## 5. 整体架构与设计变更

### 变更范围

环境层（动作空间 / 奖励门控 / 参考表时序 / 机器人模型）、数据流（rollout 长度）。

### 旧架构

```
actions: scale=0.5  ──┐
reference: T2=0.65s ──┼──► 策略学到比演示慢 0.5s 的回收段, 且动作口径与手调(0.3)不一致
S1/S2 判据: 2(yz+wx) ─┘    在 base 翻转 180° 时失效, 门控靠"瞬态蒙对"
rollout: 60 步 (0.6s)      一个 rollout 装不下完整翻正
```

### 新架构设计

**数据流**：
```
obs(114维: actions×2 + ref_joint_pos + ref_joint_vel + actuator_pos/vel/force
              + base_ang_vel + base_lin_vel + projected_gravity + command(7))
  → MLP(512,256,128) → action(14)
  → JointPositionAction: q_target = default_pos + 0.3 × action
  → MuJoCo (96 envs) → reward:
       mimic_pos(10,σ_leg=10,σ_spn=20) + mimic_vel(5,σ=0.5) + height(5,σ=500)
       + milestone_s1(2) + milestone_s2(3) + milestone_success(10)
       - 平滑/能耗惩罚(各 0.1)
  → 状态机(BackupCommand): phase P1/P2/P3, t_phase 按 λ 推进
       门控 S1/S2 基于 belly/back site 世界坐标 + 高度
  → PPO 更新
```

**模块职责**：

| 模块 | 职责 |
|---|---|
| `mdp/indices.py` | 新增 `SEGMENT_BELLY_BACK_SITES` 常量与 `segment_belly_back_ids` 字段，统一 site 索引 |
| `mdp/command.py` | `_segment_upright(idx)` 用 site 世界坐标判正置；`_check_S1/_check_S2` 据此重写 |
| `mdp/reference.py` | 参考关节角表；`_HL_HOLD` 与手调对齐 |
| `mdp/timing.py` | 单一时序真相源（P1/P2/站立过渡/保持 + λ） |
| `mdp/curriculums.py` | `_STEPS_PER_ITER=300`；奖励权重与 λ 采样课程 |
| `scripts/Backup/` | 9 个只读诊断脚本（地面真值、门控验收、动作覆盖、基线对照） |

**接口变化**：
- `env_cfg.actions["joint_pos"].scale`: `0.5` → `0.3`
- `StateMachinePolicy.__init__` 新增可选参数 `action_scale: float | None = None`（默认跟随 env_cfg）
- 新增 XML site：`F_body_belly_site`、`F_body_back_site`、`H_body_belly_site`、`H_body_back_site`（nsite 23 → 27）
- `_check_S1/_check_S2` 的**返回值语义不变**，但极性由"F 倒置/H 正置"改为"F 未翻/H 已翻"（等价）

**变更理由**：手调脚本已实测修正（T2=0.15），训练参考表若不同步，策略学的就是另一条轨迹；
S1/S2 旧判据在 base 翻转时物理失效，导致门控只能靠瞬态蒙对，必须换成不依赖四元数约定的量。

## 6. 关键文件

| 文件 | 说明 |
|---|---|
| `docs/SQuRo_Backup_技术细节.md` | **翻正任务完整技术文档**（坐标系陷阱、地面真值、时序、门控、验收数据） |
| `src/mjlab/tasks/SQuRo_Backup/mdp/timing.py` | 时序单一真相源 |
| `src/mjlab/tasks/SQuRo_Backup/mdp/command.py` | 状态机 + site 判据 |
| `src/mjlab/scripts/SQuRo_backup_Replay.py` | 手调参考实现（回放/可视化） |
| `src/mjlab/scripts/Backup/verify_gate_predicates.py` | 门控验收 |
| `src/mjlab/scripts/Backup/check_action_scale_coverage.py` | 动作覆盖核算 |
| `src/mjlab/scripts/Backup/baseline_trivial_policy.py` | 平凡解基线（**待跑通**） |
| `tests/test_squro_backup_timing.py` | 时序/门限/动作覆盖测试 |
| `tests/test_squro_backup_replay.py` | `slow1_target` 关键帧测试 |

## 7. 环境注意事项

- 训练日志目录：`$env:TEMP\sq_learn1`（临时目录，重启可能丢失；正式训练请用仓库内 `logs/`）
- 训练必须加 `--agent.logger tensorboard`（沙箱下 wandb 不可用）
- 跑测试：`$env:PYTHONDONTWRITEBYTECODE="1"; uv run python -B -m pytest <files> -q -p no:cacheprovider`
- 尚未 commit：本轮所有改动都在工作区，未执行 `git commit`
