# Hole 固定高度参考回放

入口：`src/mjlab/scripts/SQuRo_Hole_Replay.py`。参考 Backup Replay 的位置控制回放方式，不需要策略或 checkpoint。

## 使用

当前入口读取脚本顶部的直接配置常量，不解析命令行参数。运行前修改 `HEIGHT_F`、`HEIGHT_H`（单位米）、`VISUALIZE`、`DURATION`、`DEVICE`、`ENABLE_COLLISION` 和 `OUTPUT_DIR`：

```python
HEIGHT_F = 0.055
HEIGHT_H = 0.020
VISUALIZE = "none"  # 无窗；交互窗口用 viewer，录制视频用 video
DURATION = 10.0
```

在项目根目录执行：

```powershell
uv run python -B -m mjlab.scripts.SQuRo_Hole_Replay
```

默认单环境、10秒，不创建限高板，障碍物先验为100 mm占位。地面、机器人接触与模型动力学保留。设置 `ENABLE_COLLISION=True` 创建真实单板并开启碰撞；固定高度不会随位置切换，因此该配置不是完整钻洞课程验收。

通过 `OUTPUT_DIR` 修改输出根目录，默认 `logs/rsl_rl/SQuRo_Hole/reference_replay/`，每次生成带时间、前后高度与碰撞状态的子目录。

## 与训练参考的一致性

- 直接调用 `mdp/reference.py` 的 `get_reference_joint_pos` 和 `get_reference_joint_vel`，沿用模式判定、最近高度档、相位离散采样及头颈参考。
- 不复制 IK 或轨迹公式，不改参考表、关节范围、执行器增益或力矩上限。
- 用动作项运行时的关节顺序、缩放和偏置，将参考角反解为位置控制动作。参考出口已按执行器顺序排列，模型关节编号由 `mdp/indices.py` 的 `resolve_model_indices` 解析。
- 绕过 PPO 与 RSL-RL 的动作裁剪，原始关节参考直接交给现有位置执行器；模型自身控制范围和物理限位仍然生效。
- 固定前后高度并关闭高度课程；关闭奖励及自动终止重置，摔倒后的行为保留到回放结束。手动重置查看器后相位跟随环境归零。
- 使用任务原始初态，不直接写入参考 qpos，也不添加起步插值。最初几秒包含初态到参考的过渡，评估稳定步态时应与后续周期分开。

当前高度档只有 20、40、50、55 mm。输入 45 或 60 mm 时仍按训练规则取最近档，控制台显示选档信息。Mode 2 的前腿轨迹本身不随前高度变化，脚本如实复现这一行为。

## 输出和判读

启动时扫描完整步态周期，报告原始参考超出关节范围或执行器控制范围的关节，不替用户修正参考。

| 文件 | 内容 |
| --- | --- |
| `reference.csv` | 同一仿真时刻的参考角／速度、实际角／速度、力矩、控制范围限幅后的目标，及前后高度、位移、速度、身体表面采样最高点 |
| `joint_tracking.png` | 全部受控关节的原始参考、限幅控制目标与实际角；红线为关节限位 |
| `height_tracking.png` | 前后躯干目标／实际高度和推进速度 |
| `reference.mp4` | 仅 `--visualize video` 时生成 |

CSV 中 `time` 是累计仿真时间，`reference_time` 是当前回合的参考时间；查看器手动重置后后者归零。`*_control_target` 是原始参考按模型控制范围计算的目标，不是原始参考，也不是关节实际状态。

若启动即报告限位冲突，可直接确认参考与模型不一致。若参考均在范围内但实际关节明显偏离，则还需检查执行器力矩、闭链机构约束和地面接触，不能仅凭物理回放失败判定轨迹公式错误。躯干高度跟踪也不等同于身体表面净空。
