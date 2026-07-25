# Role
你是专注于机器人强化学习开发的 AI 助手，当前代码库基于 **MJLAB** 框架。当用户发出“结束会话”、“总结一下”、“生成 handoff”、等明确表示需要交接/总结的指令时，你**必须**生成一份专业的 `HANDOFF.md` 交接文档。

## 核心约束：架构变更强制说明
**重要规则**：如果本次会话涉及以下任意一项“重大变更”，你必须在文档中新增 `## 整体架构与设计变更` 章节，并详细描述变更前后的结构：
- 修改或扩展了环境封装（Env Wrapper / 观测空间 / 动作空间 / 奖励函数）。
- 更换或深度定制了 RL 算法（如从 PPO 改为 SAC，或修改了 Loss 计算逻辑）。
- 重构了训练循环（Training Loop）、数据采集管道（Rollout Buffer）或分布式通信机制。
- 调整了仿真与物理参数（MuJoCo 模型、时间步长、领域随机化策略）。
- 新增了自定义的神经网络结构（Policy/Value Network 的 Backbone 变化）。

## 文档必须包含的固定章节

### 1. 当前目标（Current Goal）
- 用一句话说明本次 RL 实验的期望方法与最终目标（例如：利用模仿奖励训练双足机器人实现抗扰动站立行走）。
- 如果目标在会话中发生偏移，注明“目标调整”，并解释调整原因（例如：原策略发散，降级为优先实现稳定站立）。

### 2. 已完成工作（Completed Work）
- 列出新增或修改的关键代码文件（例如：`mjlab/envs/my_robot_env.py`、`configs/exp_dynamic_walk.yaml`）。
- 如果有，描述当前策略的表现（例如：在仿真中能坚持 5 秒不倒，或完成 10 次抓取尝试）。

### 3. 下一步计划（Next Steps）
- 按优先级列出待办事项（- [ ]）。
- 如果计划涉及超参数调整，给出建议的调整范围（例如：将学习率从 1e-4 调低至 5e-5）。

### 4. 踩过的坑 & 失败尝试（Pitfalls & Failed Attempts）
- 记录**奖励函数陷阱**：哪些设计导致 Agent 走了捷径（cheating）或陷入局部最优。
- 记录**训练崩溃**：如梯度爆炸、NaN 损失，及尝试过的补救措施（如梯度裁剪、调整熵系数）。
- 记录**仿真物理问题**：例如接触参数设置不当导致机器人“滑步”或“穿模”。
- 记录已尝试但失败的网络结构或优化器方案。

### 5. 整体架构与设计变更（仅在涉及重大变更时强制出现）
- **变更范围**：说明影响的是环境层、算法层还是数据流层。
- **旧架构简述**（若存在）：修改前的逻辑流程。
- **新架构设计**：
  - **数据流图描述**：观测 -> 网络 -> 动作 -> 环境交互 -> 存储 -> 更新的路径。
  - **模块职责**：明确新增/修改的类或函数的具体职责（例如：`CustomRewardCalculator` 负责计算密集型的接触力奖惩）。
  - **接口变化**：标注是否改变了配置文件（YAML）的字段，或改变了 Env 的 `step()` 返回值。
- **变更理由**：解释为什么必须这样改（例如：原有框架不支持多模态观测输入）。

## 输出格式与行为
- 文档使用标准 Markdown，总行数控制在 **80~120 行**（若架构变动大可放宽，要求优先保障描述清晰完整，再考虑字数）。
- 如果当前环境支持文件写入（如 Claude Code），将内容写入项目根目录下的 `HANDOFF.md`；否则将完整文档打印在聊天框中，并提醒用户保存。

## 触发词
当用户说出“结束会话”、“生成 handoff”、“总结一下”时，立即执行上述流程。



# CLAUDE.md

## 项目概述

MuJoCoLab — 基于 MuJoCo 的四足机器人强化学习研究框架，研究关于脊柱型四足机器人的脊腿协同运动。

### 语言
- 注释、文档字符串、日志输出、图表标签一律使用**中文**
- 类型注解、变量名、函数名使用**英文**

### 注释风格
- 使用 `#` 行注释，不使用多行文档字符串 `"""`
- 函数上方用单行注释说明用途，如 `# 神经元频谱扫描`
- 函数内部逻辑块用简洁注释，如 `# 1倍频调谐窗口`
- 不使用 `----` 等装饰性分隔符
- 如果出现类型检查报错，不影响运行的情况下可以使用# type: ignore

### 文件结构

src/mjlab/
├── envs/           # Base RL environment (ManagerBasedRlEnv)
│   └── mdp/        # Base MDP terms (actions, observations, rewards, terminations)
├── managers/       # Manager pattern: CommandManager, RewardManager, EventManager, etc.
├── rl/             # RSL-RL integration layer
│   ├── config.py   # RslRlModelCfg, RslRlPpoAlgorithmCfg, RslRlOnPolicyRunnerCfg (+ csc_cfg field)
│   ├── runner.py   # MjlabOnPolicyRunner (checkpoint + ONNX export)
│   ├── csc_config.py  # ContrastiveConfig dataclass for CSC auxiliary loss
│   └── vecenv_wrapper.py  # Env → VecEnv bridge
├── tasks/           # Concrete tasks, each in its own directory
│   ├── registry.py  # Global task registry (register_mjlab_task → _REGISTRY dict)
│   ├── velocity/    # Velocity-tracking locomotion (Go1, G1)
│   └── tracking/    # Motion-tracking locomotion
├── asset_zoo/       # Robot model definitions (URDF/MJCF + constants)
├── entity/          # Entity abstraction over MuJoCo bodies
├── sensor/          # Contact sensors, etc.
├── sim/             # MuJoCo simulation config
├── scene/           # Scene layout (entities + sensors + terrain)
├── viewer/          # Playback visualization
├── scripts/         # CLI entry points (train, play, demo, etc.)
└── utils/           # GPU selection, W&B helpers, buffers

## 开发工作流

**始终使用 `uv run`，不使用 `python`。**

## Git / GitHub

- **每次代码修改完成后自动 commit 并 push**，无需等待用户确认，但需要告诉用户提交 GitHub 的内容，以及涉及到的文件。
- **Commit message 规范**（基于 Conventional Commits）：
  - 格式：`<type>(<scope>): <中文简述>`，例如：`feat(env): 新增抗扰动奖励项`
  - type 可选值：
    - `feat`: 新功能或实验性改动（如添加新的 reward 项、新的网络结构）
    - `fix`: 修复 bug 或仿真异常（如修复坐标旋转错误、接触力 NaN）
    - `refactor`: 重构代码结构，不改变外部行为
    - `docs`: 文档或注释更新
    - `config`: 仅修改配置文件（如 YAML / Hydra 配置）
    - `exp`: 实验记录、结果日志、checkpoint 相关提交
    - `chore`: 杂项（依赖更新、格式化、gitignore 等）
  - scope 使用主要模块名：`env`, `rl`, `mdp`, `managers`, `sensor`, `sim`, `viewer`, `scripts`, `config` 等
  - 示例：
    - `feat(mdp): 实现曲率跟踪命令生成器`
    - `fix(env): 修正 F_body 坐标系朝向提取`
    - `refactor(rl): 抽取 CSC 辅助损失为独立模块`
    - `config(exp): 更新抗扰动实验参数`
- **网络超时处理**：如果 `git push` 因网络超时或连接失败而无法完成，**不要反复重试**。直接跳过本次 push，并告知用户“⚠️ 网络超时，已跳过 git push，请手动推送”。本地的 commit 保留，待下次网络恢复时一并推送。

## SQuRo 踩坑记录

### 坐标系陷阱：body+X ≠ 物理前向

SQuRo (Mouse) 模型的 body frame 与世界坐标系有 90° 旋转偏移：

```
世界坐标系:  前=+X, 左=+Y, 上=+Z
SQuRo body frame (base_Link & F_body_Link):
  body+X → world +Y (=物理左, heading=90°)
  body+Y → world +X (base_Link 时=物理前, F_body 时=world -Z!)
  body+Z → world -Z
```

**关键后果**: `atan2` 提取的 heading 是 body+X 在世界 XY 平面的方向(=90°)，不代表物理前向(0°)。

**修复**: 所有依赖"前进方向"的投影必须减去 π/2:
```python
forward_heading = body_X_heading - torch.pi / 2  # body+X→world+Y → forward→world+X
forward_speed = vel_x * cos(forward_heading) + vel_y * sin(forward_heading)
```

**教训**: 使用新机器人模型前，务必用诊断脚本验证 body frame 各轴在世界系的实际指向。诊断代码见 `SQuRo_constants.py` 的 `__main__` 块。

### F_body vs base_Link 选择

SQuRo 脊柱是万向节结构(F_spine1=侧摆, H_spine1=俯仰, F_body/H_body=扭转)。
脊柱弯曲时 base_Link 处于中心会受两侧拉扯而震荡，因此**所有跟踪应以 F_body_Link 为准**:
- 速度: `body_link_lin_vel_w[:, f_body_id]`
- 角速度: `body_link_ang_vel_w[:, f_body_id, 2]`
- 朝向: F_body 四元数提取 heading

### 曲率命令设计

命令格式 `[vel, h_f, h_h, freq, curvature]`, ω_cmd = curvature × vel_cmd。
曲率 κ = 1/R (signed): κ>0=左转, κ<0=右转, κ=0=直行。
每次 episode reset 时采样一次，episode 内保持不变——让 agent 针对固定转弯半径学习稳态步态。

脊柱参考角: F_spine1 = clamp(-GAIN × ω_cmd, ±0.6), GAIN = L/v = 2.0。
符号: 实测 F_body_heading = base_heading - F_spine1，左转需 F_spine1<0 → 取负号。