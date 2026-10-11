# SQuRo_Hole 受限空间随机化技术细节

任务：`Mjlab-SQuRo-Hole`（钻洞 / 越障 · 虚拟碰撞路线）
本文件记录**限高板板底高度随机化 + 课程下降**的设计与实测依据。

## 1. 目标

把三块固定的限高板改为**每回合随机板底高度**：

- x 位置固定，几何统一（见 §5 未决项）；
- 板底高度在 `[下界, 上界]` 内均匀采样，**上界固定 0.09 m**；
- 下界从 0.07 m 起，按成功率课程逐档下降，步长 **0.002 m**，地板 **0.0475 m**（= 几何表默认板底）；
- 同一回合内板高不变（回合级随机化）。

常量唯一来源：`mdp/config.py` 的 `OBSTACLE_Z_LOW_START / OBSTACLE_Z_HIGH / OBSTACLE_Z_STEP / OBSTACLE_Z_FLOOR`。
其中 `OBSTACLE_Z_FLOOR` 由 `mdp/entity.py` 的 `HOLE_LAYOUT` 派生，避免两处硬编码。

## 2. 为什么必须走 mocap（而不是域随机化模型字段）

mjlab 有"逐环境模型字段随机化"设施（`mdp.dr.geom_pos` / `dr.body_pos`，配 `@requires_model_fields` 与
`sim.expand_model_fields()`），看起来是改板位的天然入口。**但在这套 mujoco_warp 后端下它对物理完全无效**：

1. 前向运动学对"焊在世界上"的几何体直接早退：
   `mujoco_warp/_src/smooth.py:195-198` —— `body_weldid[bodyid] == 0 and body_mocapid[body_rootid] == -1`
   时 `return`，注释写明"static geoms 的 geom_xpos/geom_xquat 只在 make_data 时算一次"。
2. `geom_xpos` 只在 `put_data` 由 CPU 模型 tile 一次（`mujoco_warp/_src/io.py:1583`），此后无 kernel 重算。
3. 实测：写 `model.body_pos[:, 板, 2] += 0.10` 后 `geom_xpos.z` 不变（0.0525），只有 `xpos` 变成 0.1475；
   写 `model.geom_pos` 同样无效。碰撞用的是 `geom_xpos` ⇒ 板纹丝不动。

板体自身建成 **mocap body** 后，`body_mocapid >= 0`，上面的早退条件不再成立，`geom_xpos` 逐帧更新。
仓库内先例：`src/mjlab/tasks/SQuRo_Backup/mdp/entity.py`（`RestrictedSpaceEntity` 的两面墙同为 mocap body，
逐环境写 `mocap_pos`），本任务的写法与验收脚本都照它对齐。

注意：`contype/conaffinity` 仍是**编译期固化**（warp 在 `put_model` 时烘焙碰撞候选对），
所以"阶段 3→4 切真实碰撞"仍然只能重建环境（`rl/runner.py`）。**移动板不需要重建，两者互不影响。**

## 3. 当前实现

| 文件 | 内容 |
| --- | --- |
| `mdp/entity.py` | `HoleEntity._build_geometry` 用 `add_body(..., mocap=True)` 建板体；新增 `board_body_id` / `board_mocap_id` / `write_bottom_z(env, z, env_ids=None)` / `read_bottom_z(env)` |
| `mdp/config.py` | 板底高度采样范围常量（§1） |
| `mdp/events.py` | `reset_obstacle_height(env, env_ids)`：每回合在 `[下界, 上界]` 采样一个值，**三块板共用**，写进各自 mocap |
| `scripts/Hole/verify_hole_obstacle_z.py` | 验收脚本（17 项，断言式，失败非零退出） |

几何口径：

```
板底(运行期) = mocap_pos.z
板底(默认/复位值) = HOLE_LAYOUT 的 position.z      (0.0475 / 0.0725 / 0.0475)
x 仍取 position.x, 写板只改 z
```

`reset_obstacle_height` **默认未注册到 `env_cfg.events`** —— 训练口径保持现状，
需要时显式挂上（本阶段只在验收脚本里挂）。原因：高度随机化必须与高度命令（位置表）配套，
否则同一命令会对应不同净空，策略目标自相矛盾（见 §5）。

## 4. 复位顺序（必须遵守）

`sim.reset` → `reset_data` → `reset_mocap` 会把 `mocap_pos` **打回编译期默认值**（`body_pos`）。
因此写板高必须挂在 `mode="reset"` 上；`_reset_idx` 的顺序天然满足：

```
manager_based_rl_env.py:555   sim.reset(env_ids)
manager_based_rl_env.py:560   event_manager.apply(mode="reset")   ← 板高在这里写
```

验收脚本 [4] 专门覆盖了这一点（连续 4 次复位后板底仍是采样值、不是 0.0475）。

## 5. 实测验收（2026-10-11）

`uv run python -B -m mjlab.scripts.Hole.verify_hole_obstacle_z` → **17/17 通过**

| 项 | 实测 |
| --- | --- |
| 三块板是三个独立 mocap body，不贡献自由度 | `nq/nv` 不变，`body_dofnum == 0` |
| 复位后板底 == `HOLE_LAYOUT` 默认 | 0.0475 / 0.0725 / 0.0475 |
| **geom 世界板底跟随写入**（物理链路） | 最大误差 **3.73e-09** |
| 只改 z，x 不受影响 | 误差 < 1e-6 |
| **碰撞跟着板走** | 板底 0.0475 → 落点 z 0.0811；板底 0.0700 → 0.1036；**Δz = +0.0225 = 板底差** |
| 复位事件生效、不被 `sim.reset` 打回 | 连续 4 次复位均在 `[0.07, 0.09]` 且非默认值 |
| 三块板共用同一采样值 | 一致 |

回归：`Hole/verify_hole_collision` **39 PASS / 0 FAIL**（物理落点与改 mocap 前一致，仍是 0.0811）、
`Hole/verify_hole_baseline` 退出码 0。

## 6. 未决项（下一步）

1. **三块板是否共享一个板高**：现按"共用"实现（对应"一个回合只有一个受限空间高度"）。
2. **几何是否统一为 Hole1 规格**（`size=(0.01, 0.1, 0.005)`）：现在仍是"短低 30 mm + 长高 200 mm"两种。
3. **成功率课程**：`成功率达标 → 下界 −0.002`，需定阈值、结算窗口、地板与"难样本比例"。
   参考实现是 Backup 的 `_maybe_lower_dmin` 支路（批结算 + 连续达标 + 地板）；
   **门控量必须有区分度**（Backup 用饱和到 0.98 的量，课程曾退化成固定时钟）。
4. **奖励读运行期板底**：`hole_geometry(env)` 目前读 `cfg.position`，随机化后必须改读实际值；
   `body_contact` 的三块板阈值现在是"全局标量广播"，要改成逐环境张量。
5. **策略观测板高**：观测里目前没有任何障碍物信息，需新增（obs 维度会变，须重训）。
6. **高度命令（位置表）与板高的耦合**：待定方案；在此之前随机化只能用于机制验证。
