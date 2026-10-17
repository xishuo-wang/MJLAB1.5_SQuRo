# SQuRo_Hole 无碰撞训练与独立考核审查清单（2026-10-17）

供其他审查 agent 使用。目标是一句话：**把 Hole 第二阶段改成"训练无接触 + 独立真实碰撞考核推进课程"，同时保持既有高度序列/速度/参考/走廊/预算完全不变。**

## 一、审查对象

| 提交 | 内容 |
| --- | --- |
| `01d8356` | 新增板下虚拟碰撞惩罚（`virtual_contact`） |
| `5c053b7` | 训练关闭板碰撞 + 独立真实碰撞考核课程 + 验收脚本同步 |
| 分支 | `codex/slalom-collision-course`（Hole 代码与 main@271f594 一致，`hole-v2` tag 为改造前备份） |
| 文档 | `docs/SQuRo_Hole_高度课程设计.md` 末尾新增"无碰撞训练与独立真实碰撞考核课程" |

改动文件：
`src/mjlab/tasks/SQuRo_Hole/{SQuRo_Hole_env_cfg.py, mdp/curriculums.py, mdp/terminations.py, mdp/rewards.py, rl/runner.py}`
`src/mjlab/scripts/Hole/{verify_hole_obstacle_curriculum.py, verify_hole_collision.py, verify_hole_rebuild_timing.py, verify_hole_corridor_commands.py, verify_hole_command_reset.py}`

## 二、期望行为（逐条核对）

**E1 训练全程无接触。** `curriculums.STAGE_COLLISION_ENABLED` 必须四阶段全为 False；3000 轮 runner 重建仍必须创建单板实体（用于板高采样/高度规则/监测），但 `contype=conaffinity=0`；`_obstacle_change_needed()` 在重建后不得反复触发。

**E2 虚拟碰撞惩罚口径。** 仅当某段躯干中心 X 落在板体 `[x1,x2]` 内才约束该段；阈值 = `实际板底 − VIRTUAL_CLEARANCE_MARGIN(2.5mm)`；超限量 `e = max(前/后段各 9 个躯干 site 最高 Z − 阈值, 0)`；惩罚 `-w·mean(1−exp(−σe²))`，`w=5`、`σ=2000`；曲线 `virtual_contact=(0,0,0,5)`（仅第四阶段），`sigma_virtual_contact` 四阶段 2000；无板/权重 0 时返回零；记录 `Metrics/虚拟碰撞/前段超限均值`、`后段超限均值`。**不应**使用命令高度、**不应**使用板外位置、**不应**包含头部/腿部（后续再加）。

**E3 考核环境独立且开启接触。** 复制训练配置、`HEIGHT_EVALUATION_ENVS=1024`、`configure_obstacle_entity(enable_collision=True)`、`enable_height_course=True`、`enable_course_progress=False`、`debug_vis=False`、清空 fixed_*；创建后必须校验 `obstacle_collision_enabled` 为真，否则抛错。

**E4 考核调度。** 课程开始后 `HEIGHT_EVALUATION_PRETRAIN_ITERATIONS=500` 轮才首次触发；此后每 `HEIGHT_EVALUATION_INTERVAL=100` 轮在 PPO 更新之后执行；`final_level_passed=True` 后不再触发；非分布式、非 fixed/random 回放条件下才启用。

**E5 单批考核口径。** 每批每个环境只计首个完整回合；高度采样与训练一致（30% 固定当前 `Zmin`、其余在 `[Zmin, Zmax]` 均匀）；课程状态 z_min/z_max 与训练同步；高板等高/分段策略沿用训练配置；episode 长度、参考、动作维度与训练一致。

**E6 课程推进规则。** 每次考核连续两批；每批要求"最低档有效回合 ≥ `HEIGHT_EVALUATION_MIN_SAMPLES=256` 且最低档通过率 ≥ 60%"；两批都达标且驻留 ≥100 轮才把 `Zmin` 降一档（>50mm 降 2mm，≤50mm 降 1mm，下限 46mm）；任一批不达标保持当前难度；`Zmin` 到达 46mm 后两批达标置 `final_level_passed=True` 并停止后续考核。

**E7 考核不污染训练。** 考核轨迹不进入 PPO rollout；考核用冻结推理策略；考核前后恢复 Python/NumPy 随机数状态并用 `torch.random.fork_rng` 隔离 Torch 随机数；考核后恢复 `alg.train_mode()`；训练策略参数、观测归一化、`common_step_counter`、episode 计数不被修改。

**E8 训练侧只诊断。** 训练 command `enable_course_progress=False`；`HolePassageMonitor` 在无碰撞训练环境仍按几何判据输出结果（`Metrics/训练无碰撞几何通过率`、`Metrics/训练无碰撞通过高度均值`），但**不得**推进课程；`Metrics/Zmin_m/success_rate/failure_rate/level_progress` 只反映最近一次考核批次。

**E9 检查点。** `hole_state` 新增 `training_collision=False`、`evaluation_envs=1024`；`collision` 记录训练侧实际编译值（新检查点为 False）；`height_course` 新增 `final_level_passed` 并可往返；旧检查点缺少这些字段时必须给默认值（False / 沿用旧行为），不得报错；`load()` 续训时课程状态可恢复。

**E10 回放默认行为。** 主回放默认沿用检查点碰撞配置，因此新检查点回放**无板体接触**；需要物理考核时显式 `--collision True`。文档已说明，检查脚本不得再假定"stage 4 训练碰撞开"。

**E11 验收脚本同步。** 五个 Hole 验收脚本按新协议更新：碰撞验收改为"训练环境碰撞关 / 独立考核环境碰撞开 / 检查点记录 False"；重建计时验收改为"建板但无碰撞、训练侧课程计数为 0"；课程验收新增考核调度/样本门槛/两批降档/最终档停止检查；修复 10-17 规则变更后过期的硬编码期望（低板挡位 0.02→离散目标，改用生产函数计算）。

## 三、必须保持的不变量

- I1 高度查表规则、65mm 分界、5mm 离散挡位、xmax 固定/Zmin 递减；
- I2 速度公式、低姿态固定角标定、脊柱俯仰参考、头颈参考；
- I3 corridor 奖励公式、权重曲线、2000～3000 轮余量线性收紧；
- I4 命令 6 维、观测 113 维、动作 14 维、48 步/轮、PPO 参数、训练预算 6000 轮；
- I5 第四阶段前不创建板、不触发考核、不产生虚拟碰撞惩罚；
- I6 `HolePassageMonitor` 的几何成功/失败判据、失败码 1~5 语义不变。

## 四、高风险点（优先审查）

1. **随机数隔离**：`_maybe_evaluate_curriculum` 是否在异常路径也恢复 numpy/python 随机数状态并恢复 train_mode；`torch.manual_seed` 是否只产生在 fork 内。
2. **课程状态对象同一性**：runner 的 `_height_curriculum_state` 与训练 env 的 command 状态是否同一对象；3000 轮重建后是否重新绑定；考核推进后训练采样是否立即使用新 `Zmin`。
3. **批次判定**：`record_evaluation` 对 `valid`、`height≈z_min`（1e-6 容差）、重复环境、样本不足的处理是否正确；`consecutive_success_batches` 与 dwell 的组合是否正确；到达下限时 `final_level_passed` 与"不再考核"是否一致。
4. **重建判据**：`_obstacle_change_needed` 在新协议下第一次进入第四阶段必须重建、之后不得反复重建；续训加载第四阶段检查点时是否在首次采样前建板（且无碰撞）。
5. **考核环境泄漏**：考核 env 的 `enable_course_progress` 是否为 False、`fixed_*` 是否清空、`common_step_counter` 是否同步、episode 是否完整 12s、每环境是否只计首回合。
6. **训练污染**：考核是否意外写入训练 env 的 monitor completed/课程批次；`_consume_obstacle_results` 是否误用训练结果推进课程；logger 是否重复计回合。
7. **异常安全**：考核缺回合时抛错、eval env 创建/关闭、`learn()` 的 finally 是否关闭 eval env 并保存最终检查点。
8. **旧检查点兼容**：缺 `final_level_passed`/`training_collision`/`evaluation_envs` 时 load/save/回放是否正常。

## 五、验证入口与预期

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONDONTWRITEBYTECODE='1'; $env:UV_CACHE_DIR="$env:TEMP\uv_cache_mjlab"
uv run --no-sync python -B -m mjlab.scripts.Hole.verify_hole_obstacle_curriculum   # 全部通过, 含新增 8 项考核检查
uv run --no-sync python -B -m mjlab.scripts.Hole.verify_hole_collision            # 训练碰撞关/考核碰撞开
uv run --no-sync python -B -m mjlab.scripts.Hole.verify_hole_rebuild_timing       # 只重建一次且无碰撞
uv run --no-sync python -B -m mjlab.scripts.Hole.verify_hole_corridor_commands    # 通过
uv run --no-sync python -B -m mjlab.scripts.Hole.verify_hole_command_reset        # 通过
uv run --no-sync python -B -m mjlab.scripts.Hole.verify_hole_baseline             # 通过
uv run --no-sync python -B -m mjlab.scripts.Backup.verify_backup_stage_rewards    # 88 项 OK
```

证据：`logs/diagnostics/SQuRo_Hole/evaluation_course_20261017/`（含各验收日志与 `smoke_result.json`）。其中 64 环境 102 轮短训从 3499 轮触发 3500/3600 两次考核（未达标保持 70mm、训练碰撞 False），1024 环境两批考核实测约 106 秒、最低档样本 289/317。

## 六、明确不在本次范围

- 训练侧评估反馈（Level-1 课程/奖励调制、Level-2 可行性预测器与 potential-based shaping）尚未实现，只留了 `collision_evaluations/*.json` 诊断数据；
- 考核失败诊断字段（失败部位/相位/穿透深度/接触力）尚未扩展；
- 头部/腿部采样点、整机五盒包络未接入虚拟碰撞；
- 主回放与可视化脚本的碰撞默认值只做了文档说明，未改其参数面；
- attitude/head/low-pose 等其他 Hole 验收脚本未重跑（本次改动不涉及）。
