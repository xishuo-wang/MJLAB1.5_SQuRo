# MuJoCoLab — SQuRo 脊腿协同运动

基于 **MuJoCo + MJLAB** 的脊柱型微小四足机器人 (SQuRo) 强化学习研究。
机器人脊柱是万向节结构，研究目标是让脊、腿协同完成翻正、绕杆、钻洞等动作。

## 子任务

| 子任务 | 任务 ID | 状态 |
| --- | --- | --- |
| **SQuRo_Backup** 仰卧翻正 | `Mjlab-SQuRo-Backup` | 当前主线；翻正已学会，站立抖动已修复并验收 |
| **SQuRo_Slalom** 连续绕杆 | `Mjlab-SQuRo-Slalom` | 已实现 |
| **SQuRo_Hole** 钻洞 | — | 后续 |

## 快速开始

```powershell
uv run train Mjlab-SQuRo-Backup --agent.logger tensorboard   # 训练（沙箱下 wandb 不可用）
uv run python -B -m mjlab.scripts.Backup.verify_backup_stage_rewards   # 主回归（21 项）
uv run python -B -m mjlab.scripts.SQuRo_Backup_Replay --visualize none # 手调参考回放
```

始终用 `uv run`，不要直接用 `python`。

## 文档

| 文件 | 内容 |
| --- | --- |
| `AGENTS.md` | AI 协作规则：工作流、代码风格、提交规范、环境注意 |
| `PROJECT.md` | 项目背景：机器人模型与坐标系、各子任务、踩坑记录 |
| `docs/SQuRo_Backup_技术细节.md` | 翻正任务参考手册（当前配置、原理与标定、验收方法） |
| `docs/SQuRo_Backup_奖励对照.md` | 翻正任务实验日志 |
| `docs/SQuRo_Backup_修改清单.md` | 翻正任务状态与计划 |
| `docs/source/*.rst` | 上游 MJLAB 框架文档（Sphinx，与本研究无关） |
