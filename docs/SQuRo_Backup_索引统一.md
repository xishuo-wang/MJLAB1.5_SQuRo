# SQuRo_Backup 索引统一（2026-10-22）

本次整理以 `SQuRo_Backup-v0.1`（`d22dca9`）为源码基线。最近一次正式 Backup 训练为 `2026-09-27_14-33-27_v1/model_8999.pt`，其 Git 快照为 `backup-v1`（`e1343b4`）。两者不能当作同一训练版本。

## 统一入口

`mdp/indices.py` 统一维护执行器顺序、关节分组、被动连杆关节名称、躯干与足端名称。模型内部关节、刚体、site 索引由名称解析；动作列仍使用原来的 14 维执行器顺序。

复位、参考表、奖励、手调回放和相关诊断脚本改为读取统一索引。原来的 `_ACTUATED_JOINT_NAMES`、`_ACTION_*` 与脊柱坐标系导出继续保留，避免破坏调用方。模型索引只有在全部名称解析成功后才写入缓存，缺失被动关节不会留下部分有效的缓存。

CSV 足端力按传感器的 `primary_names` 重排到既有的 FL、FR、HL、HR 列顺序；此前传感器配置顺序为 FR、FL、HR、HL。这只修正记录列与足端的对应关系，不改变训练观测、奖励和动作。

## 等价性证据

- 复位时 30 个关节的名称与目标角度映射与基线一致；打乱模型关节顺序后仍得到相同映射。
- 参考表时间、关节位置和 P2 曲线逐点完全相等，最大差为 `0.0`。
- 手调回放在多个时间缩放、全部分段边界及相邻浮点数处与基线完全相等，最大差为 `0.0`。
- PPO 配置、奖励 `_CURVES` 与基线一致。
- 索引与阶段奖励检查共 93 项通过；环境重建检查 31 项通过；配置校验通过。

诊断快照与数值对比保存在 `logs/diagnostics/SQuRo_Backup/curriculum_refactor_20261022/`。本次不修改课程推进、奖励系数、参考动作、墙体接触参数或网络结构；独立考核课程另行提交。

## 复现

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
uv run python -B -m unittest mjlab.scripts.Backup.verify_backup_indices mjlab.scripts.Backup.verify_backup_stage_rewards
uv run python -B -m mjlab.scripts.Backup.verify_backup_corridor_rebuild
uv run python -B -m mjlab.scripts.Backup.verify_backup_config
```
