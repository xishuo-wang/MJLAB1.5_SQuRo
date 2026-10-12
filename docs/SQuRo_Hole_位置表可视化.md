# SQuRo_Hole 高度—位置表可视化

脚本：`src/mjlab/scripts/Hole/visualize_hole_schedule.py`。

## 展示内容

- 上图展示当前完整位置表与三块板，保留原始场景作为参照。
- 下图分别展开第一块短板和长板对应的区间。
- 蓝色、橙色曲线为前后躯干中心的目标高度；灰色矩形为板实体。
- 红色虚线为实际板底减去 `VIRTUAL_CLEARANCE_MARGIN` 的虚拟净空阈值。
- 位置滑块显示高度命令、速度命令、高低模式和参考表实际选用的高度档位。

布局读取 `entity.py` 的 `HOLE_LAYOUT`；位置表读取 `command.py` 的
`STAGE3_POSITION_SCHEDULE`，脚本不另外维护位置表数值。
当前短板位于 X=0.2 m，长度 30 mm，板底 50 mm；长板位于 X=0.6 m，
长度 200 mm，板底 70 mm。两块板均厚 10 mm。

当前只有这两种固定板高下的原始配置。本脚本尚不定义任意板高到新位置表的映射，
后续确定联动公式后再增加板高控件。目标中心高度不能直接与板底当作同一几何量。

## 使用方式

打开交互窗口，同时保存图片：

```powershell
uv run python -B -m mjlab.scripts.Hole.visualize_hole_schedule
```

只导出图片：

```powershell
uv run python -B -m mjlab.scripts.Hole.visualize_hole_schedule --no-show
```

默认输出目录为项目根下的 `logs/rsl_rl/SQuRo_Hole/schedule_visualization/`，
包含 PNG、SVG 和记录源码参数的 JSON。可用 `--output-dir` 指定其他目录。

横轴是机器人相对复位起点的 X 位移。当前任务复位 X=0，所以默认同时对应世界 X；
检查其他出生位置时使用 `--start-x`，板的位置会换算到同一横轴，位置表节点保持源码值。

## 查表口径

采样通过轻量状态调用生产函数 `HoleCommand._update_command_from_schedule`，
不创建 MuJoCo 环境，不加载检查点。节点处采用当前代码的右侧区间，倒退位移截断为零。
参考表档位按当前 `HEIGHT_LIST` 的最近邻规则计算，没有高度插值。

短板与长板展示窗口按相邻板中心的中点划分，因此边界处的读数可能已切入下一段；
上方完整位置表提供上下文。脚本不验证实际机器人运动、接触或穿板成功率。
