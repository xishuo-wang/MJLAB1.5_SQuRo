# SQuRo 实际脊柱形变角接口

本接口用于按实际位姿分析侧摆、矢状面弯曲和前后相对扭转。它不接入奖励、观测、状态机或训练日志，不修改仿真状态。三个输出是形变描述量，不是四个执行器的角度映射，也不无损描述机构的全部内部自由度。

## 几何定义

使用三个参考点 F、C、H。默认分别为 `F_body_Link`、`base_Link`、`H_body_Link` 的局部坐标原点，**不是质量质心或自动推断的几何中心**。可通过 `front_offset`、`center_offset`、`hind_offset` 指定各自刚体坐标中的固定偏置。改变参考点会改变测量定义，实验之间必须保持一致。

两条连线均统一朝向头部：`u_F = normalize(p_F - p_C)`、`u_H = normalize(p_C - p_H)`。中间纵轴取 `x_M = normalize(u_F + u_H)`，因此是角平分方向，不按前后段长度加权。虚拟参考系原点位于 C；模型中的 `base_Link` 姿态不会被改写。

前后段实际解剖坐标的三轴分别为头向、左向、背向，必须先校正原模型局部轴：

| 解剖方向 | F_body_Link 局部方向 | H_body_Link 局部方向 |
|---|---|---|
| 头向 | −Z | −Z |
| 左向 | +X | −X |
| 背向 | −Y | +Y |

校正常量和具名索引解析位于 `tasks/SQuRo_Backup/mdp/indices.py`；当前模型的腹背 site 和头颈位置已用回归核对。

先以最短旋转分别将两端解剖纵轴对齐到 `x_M`，同步搬运各自背向。绕 `x_M` 从后背向到前背向的有符号夹角为 `axial`。这是通过共同纵轴去弯曲后的相对扭转，不是绕世界轴的 roll 差。

将这个相对扭转连续展开成 `axial_unwrapped`，再将搬运后的后段背向绕 `x_M` 旋转其一半，得到平均背向 `z_M`；`y_M = z_M × x_M`。这实现了平均扭转身体参考平面，同时避免两端背向相反时直接求和得到零向量。相对扭转的正号约定为前段相对后段绕共同头向轴的右手旋转。

把 `u_F`、`u_H` 转入 `[x_M, y_M, z_M]`，分别记为 f、h：

- `lateral = atan2(h_x*f_y - h_y*f_x, h_x*f_x + h_y*f_y)`。
- `sagittal = atan2(h_x*f_z - h_z*f_x, h_x*f_x + h_z*f_z)`。

左弯为正，前段向背侧弯为正。它们是三点折线的投影夹角，**不是 Euler yaw/pitch，也不是前后躯干姿态差的投影角**。整体刚体平移、转向或翻滚不改变三个形变角。

## 调用方法

已有 MJLAB 环境时，创建一次读取器并按时间顺序调用：

```python
from mjlab.tasks.SQuRo_Backup.api.spine_reader import SQuRoSpineDeformation

reader = SQuRoSpineDeformation(env.scene.entities["robot"])
result = reader.compute()
angles_rad = result.angles       # (num_envs, 3)，顺序为侧摆、俯仰、扭转主值
usable = result.valid
frame_w = result.frame_w         # (num_envs, 3, 3)，列为纵轴、左向、背向

reader.reset(env_ids)            # 在这些环境物理复位后、下一次 compute 前调用
reader.reset()                  # 全部清理
```

接口按实体名称解析索引，不共享训练侧的全局索引缓存。适用于使用同一模型、同名刚体的其他 SQuRo 任务；它不依赖 Backup 的 command 或 phase 状态。

分层：`tasks/SQuRo_Backup/api/spine_reader.py` 是**任务侧只读接口**（按名取实体刚体、装配解剖轴），
放在 `api/` 而不是 `mdp/`，表示它**不参与 MDP**（不接奖励、观测、命令、终止）；
底层算法在 `utils/spine_deformation.py`，与具体机器人无关。解剖常量仍在 `mdp/indices.py`。

离线分析、其他机器人或自定义参考点可以直接使用纯张量接口：

```python
from mjlab.utils.spine_deformation import compute_spine_deformation

state = None
for sample in trajectory:
    result = compute_spine_deformation(
        sample.front_pos_w, sample.center_pos_w, sample.hind_pos_w,
        sample.front_frame_w, sample.hind_frame_w, state,
    )
    state = result.state
```

纯接口的位置形状是 `(..., 3)`，姿态是 `(..., 3, 3)`，后者必须已校正为解剖坐标系到世界的旋转矩阵。支持 CPU/CUDA、float32/float64、单样本与批量样本，输入的批量维度、设备和精度必须相同。独立时间序列应各自维护历史，不能把时间轴当作相互独立的环境批量后期望自动展开。

## 连续性和复位

`axial` 是约在 `[−π, π]` 内的瞬时主值；`axial_unwrapped` 是从本段历史连续展开的角。用后者的半角定义平均平面，因此越过 ±π 不会突然将平均平面翻转 180°。首次样本默认选取最短相对扭转分支；如果从中途开始且已知历史绕转，可向纯接口传入显式 `SpineDeformationState`。

展开要求相邻有效采样的真实相对扭转变化小于 π。超过 π 的漏采样无法仅由姿态识别；恰好差 π 会标记 `branch_ambiguous`。首次样本背向恰好相反时也无法唯一确定平均平面，不能伪造零值。此时扭转主值仍可有效，但弯曲角和平均平面返回 NaN。应从较早的非歧义构型开始记录，或提供已知分支。

完整回合复位、成功后的循环物理复位、主动改写姿态及不连续的离线片段切换，都需要调用 `reset`。接口不会自动检测这些事件，**不能仅凭 episode_length_buf 推断循环复位**。部分复位只清指定环境，其他环境历史保留；返回的历史状态已脱离计算图。

## 无效样本和统计

零长度连线、前后连线完全折返导致角平分方向消失、纵轴最短对齐接近反向、非法旋转矩阵、非有限输入和退化投影均有显式有效性标记。

| 字段 | 含义 |
|---|---|
| `bending_valid` | 两个投影弯曲角及其参考平面可用 |
| `axial_valid` | 扭转主值可用 |
| `branch_ambiguous` | 有几何数据，但历史不足以唯一选择平均平面分支 |
| `valid` | 三个输出均可用 |

对应无效输出为 NaN，不能直接交给训练日志的普通 mean。调用方应先按掩码筛选；无样本时省略均值键并另记样本数。无效几何或分支歧义会使该样本的历史失效；恢复后重新初始化最短分支，不能把缺口两侧当作已保证连续。

接口不做低通滤波、绝对值包络、零位减法或最大幅值归一化。真实模型的三个原点可能略有安装偏置，因此零关节角不必严格得到零折线弯曲。若要零位标定，应保存固定参考点与标定规则，不能逐帧重新归零。论文基元指标应在有效时段中另行计算，并明确扭转采用主值幅值还是累计绕转；实际弯曲上限不能直接沿用单个关节的 0.6 rad。

## 验证入口

`uv run python -B -m mjlab.scripts.Spine_Kinematics.verify_spine_deformation`

覆盖独立形变、共同扭转带来的平面转换、整体刚体运动不变性、跨两周展开、相反背向分支歧义、部分复位、无效样本隔离、CPU/CUDA 一致性以及生产模型解剖轴和实体读取接线。
