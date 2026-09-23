# AGENTS.md — AI 协作规则

本文件是 AI 助手在本仓库工作时的规则；**项目背景**见 `PROJECT.md`，**任务技术细节**见 `docs/`。

## 角色与工作流

- 角色：机器人强化学习开发助手，框架为 **MJLAB**（MuJoCo + mujoco_warp）。
- **始终用 `uv run`，不要直接用 `python`。**
- **需求模糊时必须先问**：任务目标、实现范围、超参数、期望行为不清楚时向用户追问，不要自行假设或猜测。
- **一次只改一件事**：同时改多个变量会让实验无法归因。
- 每次代码修改完成后**自动 commit 并 push**，无需等待确认，但要告诉用户提交内容与涉及文件。

## 代码风格

- 注释、日志输出、图表标签一律**中文**；类型注解、变量名、函数名用**英文**。
- 用 `#` 行注释，**不使用多行文档字符串 `"""`**。
- 函数上方一行说明用途（如 `# 神经元频谱扫描`），函数内逻辑块用简短注释。
- **不使用 `====` / `----` 等装饰性分隔符。**
- **代码里不要写大段注释**：设计依据、标定过程、踩坑一律写进 `docs/`，代码里只留一句
  说明和指向文档的指针。经验阈值：**一个代码块内的注释不超过 3 行**；超出的内容搬到
  `docs/`。代码注释只回答"这段在做什么"，不回答"为什么这么做、怎么标定出来的"。
- 类型检查报错但不影响运行时，可用 `# type: ignore`。

## Git 提交规范

格式 `<type>(<scope>): <中文简述>`（Conventional Commits）：

- type：`feat` 新功能/实验性改动 · `fix` 修 bug/仿真异常 · `refactor` 不改外部行为的重构 ·
  `docs` 文档或注释 · `config` 仅改配置 · `exp` 实验记录/日志/checkpoint · `chore` 杂项
- scope 用模块名：`env` `rl` `mdp` `managers` `sensor` `sim` `viewer` `scripts` `config`
- 例：`feat(mdp): 实现曲率跟踪命令生成器`、`fix(env): 修正 F_body 坐标系朝向提取`

**网络超时**：`git push` 因网络失败时**不要反复重试**，跳过并提示
"⚠️ 网络超时，已跳过 git push，请手动推送"；本地 commit 保留，等网络恢复一并推送。

## 通用约定

- **常量唯一管理处**：同一常量在多处出现时，只在一个文件定义并导入。
  奖励权重 / σ / 配比统一在 `mdp/curriculums.py` 的 `_CURVES`（env_cfg 里 `cfg.weight` 一律 1.0）。
- **索引不硬编码**：关节 / site / 执行器索引统一在 `mdp/indices.py` 解析后导入；注意不要造成循环导入。
- **动作空间维度变化时**：必须同步更新全部索引常量、预计算参考表维度、回放脚本与诊断脚本。
- **动手改姿态或前向判据前，先看 `PROJECT.md` 的坐标系一节**——body+X ≠ 物理前向，F/H 两段修正方向相反。
- **API 陷阱**：`env.action_space.shape` 是 `(num_envs, dim)`，要维度用 `shape[-1]`（用 `shape[0]` 会拿到环境数）；
  裸环境 `env.step()` 返回 5 元组 `(obs, rew, dones, timeouts, extras)`，经 `RslRlVecEnvWrapper` 后是 4 元组。
- 分层：`src/mjlab/{envs,managers,rl,sim,scene,viewer,entity,sensor,utils}` 是框架层，
  `src/mjlab/tasks/<任务>/` 是任务层，`src/mjlab/scripts/` 是入口与诊断脚本。

### 框架目录

```
src/mjlab/
├── envs/         # Base RL environment (ManagerBasedRlEnv) 与基础 MDP terms
├── managers/     # Manager pattern: Command / Reward / Event / Termination ...
├── rl/           # RSL-RL 集成层 (config, runner, CSC 辅助损失, VecEnv 桥)
├── tasks/        # 具体任务，每个任务一个目录 (registry.py 全局注册)
├── asset_zoo/    # 机器人模型定义 (URDF/MJCF + 常量)
├── entity/  sensor/  sim/  scene/  viewer/
├── scripts/      # CLI 入口与诊断脚本
└── utils/
```

## 环境注意事项

- **训练必须显式指定 tensorboard**：`--agent.logger tensorboard`（沙箱下 wandb 需要命名管道，会被拒绝）。
- `logs/` 已在 `.gitignore` 中，训练产物不会误提交。
- 跑 pytest 时加 `-p no:cacheprovider` 并设 `PYTHONDONTWRITEBYTECODE=1`，
  否则受限环境会读到过期字节码，造成"结果与源码不符"的假失败。
- 全量 `pytest` 在本机跑不了（`mujoco_warp` 的 kernel cache 只读）。SQuRo 相关验证请跑
  `uv run python -B -m mjlab.scripts.Backup.verify_backup_stage_rewards`。

## 会话交接

用户说"结束会话""总结一下""生成 handoff"时，在项目根生成 `HANDOFF.md`，包含：
当前目标（一句话，目标漂移要注明原因）/ 已完成工作（关键文件 + 当前策略表现）/
下一步计划（`- [ ]` 按优先级，含建议超参范围）/ 踩坑与失败尝试（奖励陷阱、训练崩溃、仿真物理问题）。

若本次会话涉及**重大变更**（环境封装、观测/动作/奖励空间、RL 算法或 Loss、训练循环或数据管道、
仿真与物理参数、网络结构），另加 `## 整体架构与设计变更` 章节：
变更范围（环境/算法/数据流层）、旧架构、新架构（数据流路径、模块职责、接口变化）、变更理由。
总长约 80~120 行。

## 文档索引

| 文件 | 内容 |
| --- | --- |
| `PROJECT.md` | 项目背景：机器人模型、坐标系、各子任务、踩坑记录 |
| `docs/SQuRo_Backup_技术细节.md` | 翻正任务**参考手册**：当前配置、原理与标定、踩坑、验收方法 |
| `docs/SQuRo_Backup_奖励对照.md` | 翻正任务**实验日志**（按时间追加） |
| `docs/SQuRo_Backup_修改清单.md` | 翻正任务**状态与计划** |
| `docs/SQuRo_Tunnel_技术细节.md` | 钻洞任务**参考手册**：洞几何、走廊口径、奖励权重、Phase0/1 速度规则、回放与未决问题 |
| `docs/SQuRo_Hole_技术细节.md` | 钻洞**虚拟碰撞版**：6D 命令与位移位置表、4 段奖励课程、body_contact 软限高、与 Tunnel 的差异对照 |
| `docs/source/*.rst` | 上游 MJLAB 框架的 Sphinx 文档（与本研究无关，一般不用改） |
