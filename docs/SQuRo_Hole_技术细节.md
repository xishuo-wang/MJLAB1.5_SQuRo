# SQuRo_Hole 技术细节（旧版恢复）

任务 ID：`Mjlab-SQuRo-Hole`（钻洞 / 越障，**旧版虚拟碰撞路线**）
代码：`src/mjlab/tasks/SQuRo_Hole/`，回放：`src/mjlab/scripts/SQuRo_Hole_play.py`

本任务是 **2026-04 备份（`Backup/`，当年名为 `mouse`、mjlab 0.1.0）在 mjlab 1.5 上的恢复版**。
恢复原则：**只做接口适配，任务逻辑一律照旧**。上一版（2026-07 起、被本次覆盖的四阶段重写版）
及其文档已由 git 历史保留。

备份来源：`Backup/` 下的 `mouse_env_cfg.py` + `mdp/` + `config/` + `rl/`，以及当年训练产物
`Backup/260403_1/files/`（`model_3999.pt`、`config.yaml`、`mjlab.diff`）。

## 1. 文件结构

```
src/mjlab/tasks/SQuRo_Hole/
├── SQuRo_Hole_env_cfg.py   # 观测/动作/奖励/终止/场景/实体; 含 12 执行器机器人配置
├── config/{__init__.py, rl_cfg.py}   # 注册 Mjlab-SQuRo-Hole + PPO (512-256-128, 4000 iter)
├── mdp/
│   ├── command.py     # HoleCommand: 6D 命令 + 阶段 1/2 随机采样 + 阶段 3 位移位置表
│   ├── curriculums.py # 3 段奖励权重课程 + 障碍物开关 enable_holes
│   ├── events.py      # reset_model: 固定起点 (0,0,0.06)
│   ├── hole.py        # HoleEntity: 限高板 box
│   ├── observations.py# 自定义观测 (base_pos / base_lin_vel_w / joint_acc / actuator_force / heading)
│   ├── reference.py   # [模式3][高度档4][相位500][4] ×3 参考表 + 速度表
│   ├── rewards.py     # 奖励项 (含 body_contact 虚拟碰撞)
│   ├── terminations.py# check_fallen
│   ├── indices.py     # 按名解析关节/site 索引 (版本无关工具, 沿用)
│   └── Bio_Data/      # FL/HR_Smooth.csv (足端) + XoY/YoZ_Spine_Smooth.csv (脊柱, 默认不启用)
└── rl/runner.py       # SQuRoHoleOnPolicyRunner (save 时导出 ONNX 到 wandb)

src/mjlab/scripts/SQuRo_Hole_play.py             # 回放 (命令来源 / 阶段 / 视频 / CSV)
src/mjlab/scripts/Hole/verify_hole_baseline.py   # 基线验证
```

## 2. 机器人接口（本次恢复的关键项）

旧版机器人是**同一台 SQuRo 的 12 执行器版本**（8 腿 + 4 脊柱，**没有颈部**）。
当前共享的 `SQuRo_constants.SQURO_ARTICULATION` 是 14 执行器（含 `Neck_yaw/Neck_pitch`），
其他任务（Slalom/Tunnel/Backup）都在用，**不能改**，所以本任务在 env_cfg 里任务级覆盖：

- `HOLE_ARTICULATION`：只列 12 个关节；
- `get_hole_spec()`：从 spec 里**删除** `Neck_yaw` / `Neck_pitch` 两个执行器
  —— 只改 articulation 是不够的，`data.actuator_force` 读的是**模型里全部执行器**，
  不删就还是 14 维；
- `get_hole_robot_cfg()`：`dataclasses.replace` 把上面两项盖到共享配置上。

| 量 | 旧备份检查点实测 | 本恢复版实测 |
| --- | --- | --- |
| 观测维度 | 193 | **193** |
| 动作维度 | 12 | **12** |
| actor 首层 / 末层 | (512,193) / (12,128) | 同 |

⇒ **旧 `model_3999.pt` 可直接加载回放**（实测加载成功并跑 200 步，累计平均回报 22.01）。

## 3. 参考表（[模式, 高度档, 相位, 4 关节] ×3）

`HEIGHT_LIST = [0.02, 0.04, 0.05, 0.06]`、`BASE_HEIGHT = 0.06`、相位分辨率 500。
模式：0 = 前后肢都高、1 = 前肢低、2 = 后肢低。生成规则：

- **模式 0**：前后腿都走生物足端轨迹（`FL_Smooth.csv` / `HR_Smooth.csv`），按各自高度缩放；
- **模式 1**（前低）：前腿冻结在固定姿态，后腿走**摆线**（`CYCLOID_PARAMS["front_low"]`）；
- **模式 2**（后低）：对称；
- **低高度档**（`h < 0.04`）整表冻结；脊柱列 `H_spine1 = −0.65`；
- `USE_SPINE_CSV = False`：脊柱 CSV 默认不参与（`XoY/YoZ_Spine_Smooth.csv` 已随任务打包，备用）。

实测（`verify_hole_baseline`）：

| 模式 | 高度档 (mm) | 前腿幅度 | 后腿幅度 | H_spine1 |
| --- | --- | --- | --- | --- |
| 0 | 20 | 0.0000 | 0.0000 | −0.65 |
| 0 | 40/50/60 | 1.07/1.06/1.04 | 0.90/0.92/0.96 | 0.00 |
| 1 | 20~60 | 0.0000 | 1.0447 | −0.65 |
| 2 | 20~60 | 0.9054 | 0.0000 | −0.65 |

## 4. 命令与三阶段课程

命令来源随阶段切换（全局步数刻度，`common_step_counter` 每次 `env.step()` +1，iter = 步数/24）：

| 阶段 | 步数边界 | 命令来源 | 限高板碰撞 | `enable_holes` |
| --- | --- | --- | --- | --- |
| 1 | `< 24` | 随机中等高度 `[0.04, 0.045, 0.05, 0.055, 0.06]` | 关 | False |
| 2 | `24`（**与阶段 1 同界**） | 随机全高度 `[0.02, 0.04, ..., 0.06]` | 关 | False |
| 3 | `≥ 24`，至 `96000` | **位置表**（按位移查表，每步更新） | 关 | True（第 72000 步起） |

`STAGE1_END = STAGE2_END = 1 * 24`、`STAGE3_END = 4000 * 24`。
**注意 `return 2` 是死分支**：第 2 个控制步起恒为 stage 3，随机全高度档实际不会被采样到
——这是旧版的原始行为，本次未改。

位置表只在 stage 3 生效（`_should_use_schedule`），每步按位移推进：

| 位移 (m) | h_F | h_H | 对应障碍 |
| --- | --- | --- | --- |
| 0.00 | 0.02 | 0.05 | Hole1 (x=0.2) 前肢低 |
| 0.20 | 0.06 | 0.02 | Hole2 (x=0.6) 后肢低 |
| 0.32 | 0.06 | 0.06 | 段间恢复正常 |
| 0.40 | 0.04 | 0.04 | Hole2 板顶通过 |
| 0.80 | 0.06 | 0.06 | 恢复 |
| 1.00 | 0.02 | 0.05 | Hole3 (x=1.2) 前肢低 |
| 1.20 | 0.06 | 0.02 | 后肢低 |
| 1.32 | 0.06 | 0.06 | 末段 |

位移 = root 世界 x − episode 起点 x（`start_positions`），负值 clamp 到 0。
`HEIGHT_THRESHOLD = 0.04` 用于判定"是否压低"，`get_height_scale_factor` 在 `h < 0.04` 时返回 0.1。

## 5. 三块限高板与虚拟碰撞

三块板：x = 0.2 / 0.6 / 1.2，板底高 0.05 / 0.075 / 0.05，半长 0.015 / 0.1 / 0.015。

- **训练**：`contype=conaffinity=0`（只有 `body_contact` 软约束生效）；
- **回放**：`contype=conaffinity=1`（真实碰撞）。
  两者都编译期固化，回放脚本的 `--enable-collision` 在建环境前改写实体 cfg。

旧版还有一条 `enable_holes` 课程开关（stage 3 起调 `HoleEntity.enable_collision()` 改 spec）。
**该开关挂在 `weight=0.0` 的奖励项 `update` 上，且运行期改 contype 在 mjwarp 下无效** ——
旧版从未真正生效，训练全程没有限高板碰撞。本次按原样保留代码，不改其语义。

## 6. 奖励与课程（3 段）

权重集中在 `curriculums.RewardWeightCurriculum.weight_stages`，`env_cfg` 里一律 1.0。

| 阶段阈值（步） | iter | body_contact | height | mimic_pos | height_sigma | enable_holes |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0–999 | 0 | 2.5 | 10 | 500 | False |
| 24000 | 1000–1999 | 0 | 2.5 | 10 | 1000 | False |
| 48000 | 2000–2999 | **1** | 5.0 | 10 | 1000 | False |
| 72000 | 3000–3999 | 1 | **10.0** | 10 | 1000 | **True** |

12 个奖励项注册在 `env_cfg.rewards`（`update` 一项 `weight=0.0`，是课程钩子）。
跑满 4000 iter（96000 步）时生效的是第 4 段。

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
| 10 | 机器人 | `get_mouse_robot_cfg()`（12 执行器，无颈部） | 任务级 12 执行器配置 + 从 spec 删颈部执行器 | 共享配置已是 14 执行器；见 §2 |
| 11 | 命名 | `Mjlab-Mouse` / `Mouse_*` / `mouse_cmd` / `experiment_name="mouse_locomotion"` | `Mjlab-SQuRo-Hole` / `Hole*` / `hole_cmd` / `"SQuRo_Hole"` | 与仓库其余任务统一；任务 ID 是全仓库/文档的引用点 |
| 12 | 周边脚本 | — | `SQuRo_Hole_play.py` 的碰撞开关就地改写实体；`verify_hole_baseline.py` 按三阶段/193/12 重写 | 原脚本依赖被覆盖的四阶段版内部接口 |

除以上 12 项外，任务逻辑、奖励口径、参考表生成、课程数值、位置表均与 Backup 原码一致。

## 8. 回放与验证

```powershell
# 训练 (200 Hz, 4000 iter; 必须显式 tensorboard)
uv run train Mjlab-SQuRo-Hole --agent.logger tensorboard
# 回放 (默认: 命令走 8 段位置表)
uv run python -B -m mjlab.scripts.SQuRo_Hole_play --checkpoint_file <ckpt>
# 旧备份检查点可直接回放 (obs 193 / action 12 已对齐)
uv run python -B -m mjlab.scripts.SQuRo_Hole_play --checkpoint_file Backup/260403_1/files/model_3999.pt
# 固定/随机命令
uv run python -B -m mjlab.scripts.SQuRo_Hole_play --checkpoint_file <ckpt> --command-source fixed --fixed-height-F 0.055 --fixed-height-H 0.055 --fixed-velocity 0.2
uv run python -B -m mjlab.scripts.SQuRo_Hole_play --checkpoint_file <ckpt> --command-source random --stage 1
# 显式开关限高板碰撞
uv run python -B -m mjlab.scripts.SQuRo_Hole_play --checkpoint_file <ckpt> --enable-collision False
# 无 checkpoint 自检
uv run python -B -m mjlab.scripts.SQuRo_Hole_play --agent zero --smoke_steps 60 --no-video
# 基线验证 (参考表 / 位置表 / 课程 / 碰撞 / 观测维度)
uv run python -B -m mjlab.scripts.Hole.verify_hole_baseline
```

## 9. 注意与未决项

1. **训练时限高板实际不碰撞**：`contype=0` 编译期固化，`enable_holes` 的运行期开关无效。
   要真正启用实体碰撞，需要像其他任务那样"按阶段重建环境"；旧版没有这套机制，本次未加。
2. **stage 2 不可达**：`STAGE1_END = STAGE2_END = 24`，第 2 个控制步起即为 stage 3，
   `STAGE2_HEIGHT_VALUES`（含 0.02 档）不会被采样到。
3. **`body_contact` 的 x 区间是硬编码的**，与三块板位置（0.2/0.6/1.2）绑定；
   改板位置必须同步改 `rewards.py` 里的阈值。
4. **`BASE_SPEED = 0.25`**：旧版未按步态能力标定，实测命令速度可能高于参考步态的行走能力。
5. **200 Hz 与 24 步/iter**：每 iter 只推进 0.12 s 仿真，20 s 回合需 167 iter，
   4000 iter 全程约 24 个完整回合 —— 与其他任务（50 Hz）样本量口径不同，比较时注意。
6. **未注册项**：旧版还有若干未注册的奖励函数（`energy`/`cot`/`joint_acc`/`base_y_offset`/
   `limits`/`action_acc`），本次按原样保留为可选项，未加入 `env_cfg.rewards`。
7. **`mdp/indices.py` 是沿用项**（非旧码）：旧版把关节/site 索引硬编码在 `reference.py` /
   `rewards.py` / `events.py` 里，本次保持原样；`indices.py` 只被回放脚本用于按名取
   F/H 躯干 id，不影响训练口径。
