# SQuRo 整机五盒几何素材

本脚本用于生成论文框图的局部素材，不接入训练奖励，不修改机器人 XML、策略或总图。
包围盒由当前 MuJoCo 模型网格和完整关节姿态直接计算；默认几何外扩为 0 mm。

## 五组划分

| 组 | 覆盖部件 | 姿态锚点 |
| --- | --- | --- |
| 前躯干与前腿 | F_body_Link 及 FL_/FR_ 全部闭链零件和足端 | F_body_Link |
| 前脊柱 | base_Link、FU_spine_Link、FD_spine_Link、F_spine1_Link、F_spine2_Link | F_spine1_Link |
| 后脊柱 | HL_spine_Link、HR_spine_Link、H_spine1_Link、H_spine2_Link | H_spine1_Link |
| 后躯干与后腿 | H_body_Link 及 HL_/HR_ 全部腿部闭链零件和足端 | H_body_Link |
| 头部与颈部 | Neck_yaw_Link、Neck_pitch_Link | Neck_pitch_Link |

名称精确匹配优先于腿部前缀，避免把 HL_spine_Link、HR_spine_Link 误归为腿。
中心连接件 base_Link 归入前脊柱，五组无遗漏、无重复地覆盖全部机器人网格。
环境地面不属于整机包络。遇到未归属的部件或非网格机器人几何时直接报错。

## 几何定义

1. 从编译后的 mesh_vert 读取网格顶点，以 geom_xmat、geom_xpos 转到当前世界坐标。
2. 各组朝向为当前锚点旋转乘以其 qpos0 参考旋转的转置，使中性姿态的盒轴对应世界 x/y/z。
3. 将该组全部顶点转到上述朝向中，逐轴取最小值和最大值，得到一个有向长方体。
4. 整机近似占用是五盒的并集，盒间允许重叠；不额外求覆盖所有部件的单个大盒。
5. 每次输入姿态均重算尺寸，前后腿的运动不是由固定躯干盒近似。

包围盒保证覆盖当前姿态下的离散网格及其三角面，不代表整个步态周期的扫掠包络。
对含多个可动零件的分组，一个盒会保守地填充部件之间的空隙。
不能将五盒尺寸、并集体积或当前截图直接解释为最小可通过空间或任务能力边界。
连接件归属和锚点定义属于论文几何模型的显式约定，后续可单独评估其保守性。

## 默认姿态与坐标

默认使用 SQuRo_constants.py 的 INIT_STATE.joint_pos，保留 XML 中的根位置和根四元数。
INIT_STATE 的根四元数不能直接覆盖独立 XML 的 qpos 根旋转，否则会改变模型物理朝向。
也可通过 --qpos-file 输入同一编译模型的完整单帧 qpos.npy。
模型未执行仿真松弛；这是初始关节配置的运动学展示，不是训练策略的动作或成功样例。

中性姿态世界 +x 向头、+y 向左、+z 向上；所有视图使用同一正交投影。
先用 MuJoCo 原生离屏渲染真实机器人，再按同一相机绘制盒面和边线。
通过分割通道生成透明底，保留实际闭链腿部和足端外形。
每个视图检查原生渲染的机器人轮廓是否落在绘制的五盒投影内，以可见盒区域外扩 2 px 作为栅格对齐容差。
该容差仅用于栅格对齐检查，不改变几何盒或其外扩量。
素材及预览页的英文标签沿用本次论文框图的明确要求。

## 复现

```powershell
$env:UV_CACHE_DIR='D:/MuJoCoLab_1.5/logs/diagnostics/_cache/uv'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
uv run --no-sync python -B -m mjlab.scripts.Figure.render_squro_five_box
```

默认输出：logs/diagnostics/SQuRo_Envelope/five_box_geometry/。
使用 --output-dir 可将一轮素材保存到单独目录，避免覆盖已选定的版本。

```powershell
uv run --no-sync python -B -m mjlab.scripts.Figure.render_squro_five_box --qpos-file <完整姿态.npy> --output-dir <新输出目录>
```

## 输出

- five_box_preview.png：带英文分组图例的总览，供检查本局部素材。
- five_box_overlay_*.png：机器人叠加五盒，透明底。
- five_box_geometry_*.png：五盒几何，透明底。
- five_box_robot_*.png：相同相机下的机器人原始渲染，透明底。
- four views：oblique、top、side、front；不赋予任务成功含义。
- five_box_geometry.json：XML 与网格指纹、分组、盒尺寸、旋转、角点、顶点及轮廓覆盖结果。
- pose_qpos.npy：本次使用的完整姿态。

显示用边线粗细和盒面透明度不会修改几何边界。真实外扩只通过 --margin-mm 指定并记入元数据。

## 2026-10-16 首版验证

- MuJoCo 3.10.0；完整覆盖 37 个机器人网格、175340 个顶点；几何外扩 0 mm。
- 斜视、俯视、侧视、正视的机器人轮廓均通过上述栅格覆盖检查，越界像素为 0。
- 完整姿态快照保存和读回一致；整体旋转、平移后盒角点最大误差 1.67e-16 m。
- 新脚本通过 Ruff 检查与格式检查。
- 图示为项目初始关节配置，不代表三个任务的典型动作或实测能力边界。
