# SQuRo 虚实碰撞示意素材

## 用途与范围

对应论文训练框图中关闭障碍物接触的训练素材，以及开启接触的仿真评价素材。
原生机器人网格保留全部腿部结构，机器人检测几何复用五盒工具。
素材用于解释机制，不报告任务成功率，也不表示当前 Hole 奖励已经切换为五盒计算。

## 同源几何

障碍物的位置与半尺寸直接解析 `tasks/SQuRo_Hole/mdp/config.py` 中的唯一常量。
按现有 `ObstacleEntity` 的约定，body 的 Z 是板底，geom 中心在板底上方一个半厚。
默认示意板底为 75 mm，完整尺寸由任务常量决定，不额外添加侧柱。
两份模型只改变板体碰撞掩码和显示颜色，地面接触均保留。
蓝色线框和灰板都使用同一组八个世界角点，正交相机和输出画布完全相同。

渲染程序通过 AST 读取几何常量，避免任务自动注册依赖其他正在编辑的奖励模块。
白色背景、有限浅色地面和材质调整只用于独立可视化模型。

## 姿态与检测

训练图从项目中性参考角开始，先拟合被动闭链，再保留地面接触运行 0.36 秒。
该姿态为方法机制示意，用于展示五盒局部进入虚拟板；并非收敛策略的能力数据。
评价图从已有检查点的接触开启回放中提取完整 qpos，并按关节名称映射至原生模型。
候选帧要求前躯干处于限高板附近，原生接触检查不得出现超过 0.8 mm 的板体穿透。

取帧使用独立进程，固定板高，关闭课程推进，不进行 PPO 更新。
只读视觉回放不计算训练奖励。若工作区奖励接口正在编辑，取帧进程读取 Git HEAD
中的奖励模块快照以完成环境注册，原工作区文件保持不变。

五盒覆盖前躯干与前腿、前脊柱、后脊柱、后躯干与后腿、头颈的全部真实网格顶点。
五盒与实际有限板体的交集通过半空间平面求交后计算凸包，橙红色区域来自真实交集。
局部 δ 为选中交集沿世界 Z 方向的重叠厚度，不是通用最小平移距离。
若将来奖励增加安全裕度，应另画阈值，不改变蓝框与灰板本身的尺寸。

## 运行

先准备接触开启回放的 `evaluation_candidates.npz`，其 `qpos` 为二维完整姿态数组。
2026-10-19 的取帧脚本及元数据保存在对应诊断目录的 `source/`。

```powershell
$env:UV_CACHE_DIR = 'D:/MuJoCoLab_1.5/logs/diagnostics/_cache/uv'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:TEMP = 'D:/MuJoCoLab_1.5/logs/diagnostics/_cache/contact_pair'
$env:TMP = $env:TEMP
uv run python -B -m mjlab.scripts.Figure.render_squro_contact_pair
```

可用 `--evaluation-candidates` 指定候选文件，用 `--azimuth`、`--elevation` 同时调整两图视角。
`--height` 必须与候选回放板高一致，几何一致性检查写入 `geometry_checks.json`。

## 输出

默认目录为 `logs/diagnostics/SQuRo_Envelope/contact_pair_20261019/`。

- `01_training_virtual.png`：透明底机器人、五盒、蓝色虚拟板和局部越界。
- `02_evaluation_contact.png`：同机位机器人和灰色实体板。
- `03_violation_detail_clean.png`：不带文字的局部放大。
- `04_violation_detail_delta.png`：白底 δ 标注放大图。
- `05_pair_preview.png`：英文并排预览，供检查相同场景与比例。
- `layer_*.png`：同画布独立图层，便于 PPT 叠加与调整显示。
- `training_qpos.npy`、`evaluation_qpos.npy`、`geometry_checks.json`：复现与检查记录。
