# SQuRo_Hole 技术细节（虚拟碰撞版）

任务 ID：`Mjlab-SQuRo-Hole`（钻洞 / 越障，**旧版"虚拟碰撞"路线**）
代码：`src/mjlab/tasks/SQuRo_Hole/`，回放：`src/mjlab/scripts/SQuRo_Hole_play.py`

本任务是 2026-07 旧版（分支 `v0801` 的 `40fdb98`，main 上的等价内容为 `c9fbbd9`）在
mjlab 1.5 上的**完整恢复版**，与当前 `SQuRo_Tunnel`（包络/参考表路线）并存，用于 A/B 对照。
恢复时的目标：保持旧版行为口径不变，只做接口适配与命名统一。

## 1. 与 SQuRo_Tunnel 的区别

| 维度 | SQuRo_Hole（本任务） | SQuRo_Tunnel（当前主线） |
| --- | --- | --- |
| 空间约束 | **虚拟碰撞惩罚** `body_contact`（软限高）+ 三块限高板实体碰撞 | 走廊包络奖励（`corridor`） |
| 障碍物 | 三块固定板：x=0.2/0.6/1.2，板底 0.050/0.075/0.050 | 每 episode 采样 3 个洞，板底 0.050 |
| 命令维度 | 6D `[vel_x, vel_y, vel_z, h_F, h_H, angle]` | 5D `[vel_x, h_f, h_h, gait, curvature]` |
| 命令生成 | **按位移查 8 段位置表**（阶段3），随机高度仅在阶段1/2（各 1 iter） | Phase0 随机高度 / Phase1 方波轨迹 |
| 参考表 | `[模式3][高度档4][相位500][4]` ×3（前腿/后腿/脊柱），含摆线 + 脊柱弯曲 | `[模式3][高度档4][相位200][14]`，脊柱恒 0 |
| 观测 | **201 维**（含 joint_acc / base_pos / 全 36 关节位置速度 / 动作历史 3） | 117 维 |
| 动作 | 14 个执行器（对参考表只用其中 12 个） | 14 个执行器 |
| 仿真 | 200 Hz（timestep 0.001 × decimation 5），4000 步/回合 | 50 Hz（0.005 × 4），1000 步/回合 |
| 奖励项 | 11 项（含 body_contact / foot_clearance / angle / orientation / smoothness / stop / reached） | 9 项 |
| 奖励权重 | **4 段课程**（0 / 24k / 48k / 72k 步） | 固定权重 |

两者共用同一个 SQuRo 机器人配置与 `Trot_F/H.csv` 足端轨迹数据。

## 2. 文件结构

```
src/mjlab/tasks/SQuRo_Hole/
├── SQuRo_Hole_env_cfg.py   # 观测/动作/奖励/终止/场景/实体 (1024 envs, 20 s, 200 Hz)
├── config/{__init__.py, rl_cfg.py}   # 注册 Mjlab-SQuRo-Hole + PPO (512-256-128, 4000 iter)
├── mdp/
│   ├── command.py     # HoleCommand: 6D 命令 + 阶段 1/2 随机采样 + 阶段 3 位移位置表
│   ├── curriculums.py # 4 段奖励权重课程 (含 body_contact 0→1)
│   ├── events.py      # reset_model: 固定起点 (0,0,0.06), 面朝 +X
│   ├── hole.py        # HoleEntity: 限高板 box (contype/conaffinity 编译期固化)
│   ├── observations.py# 旧版自定义观测 (base_pos / base_lin_vel_w / joint_acc / actuator_force / heading)
│   ├── reference.py   # [模式3][高度档4][相位500][4] ×3 参考表 + 速度表, 足端 IK 生成
│   ├── rewards.py     # 11 项奖励 (含 body_contact 虚拟碰撞)
│   ├── terminations.py# check_fallen (root 重力投影 < 0.2) + check_reach_goal (未启用)
│   └── Bio_Data/      # Trot_F.csv / Trot_H.csv (与 Tunnel 同源)
└── rl/runner.py       # SQuRoHoleOnPolicyRunner (带 ONNX 导出)

src/mjlab/scripts/SQuRo_Hole_play.py      # 回放 (阶段表 / 视频 / CSV)
src/mjlab/scripts/Hole/verify_hole_baseline.py   # 基线验证
```

## 3. 参考表（[模式, 高度档, 相位, 4 关节] ×3）

`HEIGHT_LIST = [0.02, 0.04, 0.05, 0.06]`，`BASE_HEIGHT = 0.06`，相位分辨率 500。
模式：0 = 前后肢都高、1 = 前肢低、2 = 后肢低。生成规则：

- **模式 0**：前后腿都走 CSV 足端轨迹（`height_scale = h/0.06` 缩放、可旋转），脊柱直立；
- **模式 1**（前低）：前腿**冻结**（CSV 轨迹在 `h < 0.04` 分支返回固定点 x=0.005/0.002, z=-0.02），
  后腿走**摆线轨迹**（`CYCLOID_PARAMS["front_low"]`：stride 0.04、height 0.005、body_height 0.05）；
- **模式 2**（后低）：对称，前腿走摆线，后腿冻结；
- **脊柱**：`h < 0.04` 或处于模式 1/2 时第 3 列（H_spine1）固定 −0.65；否则用脊柱 CSV
  （`USE_SPINE_CSV` 默认 False，仓库内无脊柱 CSV）；
- 逆运动学 `Inverse_Kinematics`：前腿肘关节在 `a2 > 2` 时 `+2π`（与 Tunnel 迁移时修的同一处符号）。

实测（`verify_hole_baseline`）：模式 0 的腿幅度 1.09~1.11 rad；模式 1 前腿幅度 0（冻结）、
后腿 1.04；模式 2 反之；高度档 0.02 时两段腿都冻结、脊柱 −0.65。

## 4. 命令与位置表

阶段边界（全局步数）：`STAGE1_END = STAGE2_END = 24`、`STAGE3_END = 4000*24`。
因为前两段各只有 1 个 iter，**从第 1 个 iter 起命令恒走阶段 3 的位置表**：

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

位移 = 机器人 root 的世界 x − episode 起点 x（`start_positions`），超出 1.32 m clamp 到末段。
速度 = `BASE_SPEED(0.25) × min(h_F,h_H)/0.06`（速度与高度绑定）。
回放时 `play=True` 用同一张表（`resampling_time_range=(2,3)`）。

## 5. 虚拟碰撞（核心机制）

两道约束同时存在：

1. **`compute_body_contact_penalty`（软限高，虚拟碰撞）**：前后躯干 x 落入三块板的 x 区间时，
   对每段 9 个采样 site 统计 `z − 板底阈值` 的正超量，返回 `−Σexcess × weight × 10`。
   硬编码区间：x ∈ [0.185, 0.215] / [0.5, 0.7] / [1.185, 1.215]，
   板底阈值 0.045 / 0.07 / 0.045（= 板底 − 5 mm）。权重 0 → 1 在第 3 段课程（iter 2000）打开。
2. **限高板实体碰撞**：本恢复版把三块板的 `contype/conaffinity` 设为 1（旧版训练是 0，只有
   `body_contact` 生效；回放才有实体碰撞）。注意 mjwarp 在 `put_model` 时固化碰撞对，
   **运行期改 contype 无效**，所以"开不开碰撞"必须在编译前定好（`build_hole_entities`）。

## 6. 奖励与课程（4 段）

权重集中在 `curriculums.RewardWeightCurriculum.weight_stages`，`env_cfg` 里一律 1.0。

| 阶段阈值 | iter | body_contact | height | velocity | smoothness | mimic_pos_sigma | height_sigma |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0–999 | 0 | 2.5 | 5 | 0.1 | 5 | 500 |
| 24000 | 1000–1999 | 0 | 2.5 | 5 | 0.2 | 10 | 1000 |
| 48000 | 2000–2999 | **1** | 5.0 | 5 | 0.5 | 10 | 1000 |
| 72000 | 3000–3999 | 1 | **10.0** | **10** | 1 | 5 | 1000 |

`mimic_pos/mimic_vel/reached/foot_clearance/angle/orientation` 在各段均为 10/5/1/1/1/2→4。
跑满 4000 iter（96000 步）时生效的是第 4 段。

奖励项口径：
- `mimic_pos/mimic_vel`：12 个被控关节（前腿 4 + 后腿 4 + 脊柱 4）对参考表的 MSE exp 核；
- `height`：`0.4·r_F + 0.4·r_H + 0.2·r_base`（F_body/H_body/root）；
- `velocity`：世界系 x 速度误差 `exp(-100·err²)`，有指令却不动的环境置 −1；
- `foot_clearance`：仅"都高"模式触发，足端 site 对 `0.01·(base_height/0.06)` 的 exp 核；
- `angle`：按前后肢高低分情况约束 F/H 躯干 roll（低高度一侧期望 ±π/2，都高期望 0）；
- `stop`：模式 2（前高后低）时惩罚前肢关节速度不足（防止前腿不动）；
- `reached`：距离 x=1.5 的高斯稠密项（σ=10，权重恒 1，无一次性成功奖励）。

## 7. 与旧版的差异（本次适配项）

| # | 项 | 旧版 | 恢复版 | 原因 |
| --- | --- | --- | --- | --- |
| 1 | 任务 ID | `Mjlab-Mouse` | `Mjlab-SQuRo-Hole` | 与仓库其余任务命名统一；`40fdb98` 里注册名与包名不一致（导入 `SQuRo_Hole` 但树是 `SQuRo_Tunnel`），属于坏状态 |
| 2 | 命令/类名 | `mouse_cmd` / `MouseCommand` / `MouseOnPolicyRunner` / `experiment_name="mouse_locomotion"` | `hole_cmd` / `HoleCommand` / `SQuRoHoleOnPolicyRunner` / `"SQuRo_Hole"` | 按"全面重命名为 Hole"；旧 checkpoint 的日志目录名因此对不上 |
| 3 | 足端 CSV | 绝对路径 `D:\Code\Mouse-MuJoCo\...\FL_Smooth.csv`（列 `X/Z`） | 仓库内 `mdp/Bio_Data/Trot_F.csv`（列 `Y_mean/Z_mean`），`fps` 仍取 60 | 旧路径不在本机；数值口径不变 |
| 4 | 关节/body 索引 | 硬编码 `[6,8,12,14,24,26,30,32,1,3,21,23]`、body 4/24 | 保留同样索引，但集中为常量（`_JOINT_ORDER`、`F_BODY_ID`、`H_BODY_ID`） | 经模型核对索引完全对应（见下）；`AGENTS.md` 要求索引集中管理 |
| 5 | 实体碰撞 | 训练 contype=0、回放 1 | **训练与回放都开**（`build_hole_entities(enable_collision=True)`） | 用户要求"完整恢复碰撞" |
| 6 | 课程开关 `enable_holes` | 阶段3 置 True 并调 `HoleEntity.enable_collision()` | 删除 | 该开关挂在 `weight=0.0` 的奖励项上（mjlab 不调用），旧版从未生效；且运行期改 contype 无效 |
| 7 | episode 长度 | 20 s（4000 步 @200 Hz） | 同 | 保持 |
| 8 | `update_curriculum` 钩子 | 注册为 `weight=0.0` 的奖励项 | 删除 | 同上，从未执行 |

索引核对（`robot.joint_names`，36 个关节）：`6=FL_shoulder, 8=FL_elbow, 12=FR_shoulder,
14=FR_elbow, 24=HL_hip, 26=HL_knee, 30=HR_hip, 32=HR_knee, 1=F_spine1, 3=F_body,
21=H_spine1, 23=H_body` —— 与旧版硬编码完全一致，因此观测/奖励口径不变。

## 8. 回放与验证

```powershell
# 训练 (200 Hz, 4000 iter; 必须显式 tensorboard)
uv run train Mjlab-SQuRo-Hole --agent.logger tensorboard
# 回放 (命令走 8 段位置表)
uv run python -B -m mjlab.scripts.SQuRo_Hole_play --checkpoint_file logs/rsl_rl/SQuRo_Hole/<run>/model_3999.pt
# 无 checkpoint 自检
uv run python -B -m mjlab.scripts.SQuRo_Hole_play --agent zero --smoke_steps 60 --no-video
# 基线验证 (参考表 / 位置表 / 课程 / 碰撞 / 观测维度)
uv run python -B -m mjlab.scripts.Hole.verify_hole_baseline
```

回放 CSV（`dummy.csv` 示例 60 步 × 70 列）含 base 位置速度、F/H 躯干高度与追踪误差、
6D 命令与模式判定、足端接触力与位置、12 列参考位置/速度。

## 9. 注意与未决项

1. **200 Hz 与 24 步/iter 的组合**：每 iter 只推进 0.12 s 仿真，一个 20 s 回合需要 167 iter；
   即 4000 iter 全程只有约 24 个完整回合。与 Tunnel（50 Hz）的样本量口径完全不同，比较实验
   表现时要注意。
2. **碰撞常开可能改变旧版行为**：旧版训练时实体碰撞是关的，只有 `body_contact` 软约束。
   本版按用户要求打开实体碰撞，若发现明显不同（例如出现卡在板下/被弹出），可把
   `build_hole_entities(enable_collision=False)` 切回旧版口径。
3. **`body_contact` 的 x 区间是硬编码的**：与三块板位置绑定（0.2/0.6/1.2）。
   若改板位置，必须同步改 `rewards.py` 里的 `obs_x_min/obs_x_max/obs_z_thresh` 与
   `env_cfg.HOLE_LAYOUT`。
4. **速度与高度绑定**：`min(h_F,h_H)=0.02` 时速度只有 0.083 m/s，而 200 Hz 下步态 2 Hz；
   参考轨迹一个周期约 77 mm（Tunnel 迁移时实测），与命令速度差约 1.6 倍，可能存在跟踪偏差。
5. **索引口径**：本任务保留旧版硬编码关节索引以与旧 checkpoint/日志一致；若要迁到
   `indices.py` 解析口径，需要同时改 `reference.py`/`rewards.py`/`observations.py` 三处并重训。
6. **未注册项**：旧版还有 `energy/cot/joint_acc/base_y_offset/limits/action_acc` 六个函数未注册，
   恢复版只保留了注册项 + energy/cot（作为可选项，未注册）；如需启用请显式加入 `env_cfg.rewards`。
