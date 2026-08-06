from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from .indices import _MODEL_INDICES, resolve_model_indices
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# ==================== 障碍物 (门洞式洞) 配置 ====================
# 对齐旧版 hole1 尺寸: 洞宽 3cm, 洞下沿 0.050m
OBSTACLE_LENGTH = 0.03       # 洞宽 a (m) = 3cm
HOLE_BOTTOM = 0.050          # 洞下沿高度 (m) = hole position z = 5cm
HOLE_THICKNESS = 0.01        # 限高板厚度 (m) = 2*size[2]

# ==================== 期望高度规则 ====================
HEIGHT_NORMAL = 0.055        # 正常段期望高度 (m) — 对齐 Slalom FIXED_HEIGHT
HEIGHT_HOLE = 0.02           # 低高度期望 (m)
TRANSITION_LENGTH = 0.0      # 线性过渡长度 (m): 正常↔低高度 — 方波

# ==================== 洞动态采样 (Phase1, 单 episode 内固定) ====================
HOLE_NUM = 3                 # 每 episode 洞数量 (多个洞)
HOLE_MIN_X = 0.15            # 第一个洞距机器人起始原点最小距离 (m)
HOLE_MIN_SPACING = 0.30      # 洞与洞之间最小间距 (m)
HOLE_MAX_X = 2.0             # 采样 x 上限 (m, 覆盖 episode 距离)

# ==================== 轨迹公式偏移 (cm 换算为 m) ====================
# 前肢中心: 降 x-9cm, 升 x+5+(a/2)
# 后肢中心: 降 x-4+(a/2), 升 x+4+(a)  ← 降起点 = x-0.04+a/2 = x-0.025
# 衔接: 前肢中心-后肢中心距离 0.09 → 后肢开始降时前肢刚好开始升 (永无双低)
# 其中 x 为洞左侧位置
FRONT_DOWN_OFF = 0.09                    # 前肢 降起点相对洞左侧偏移
FRONT_UP_OFF = 0.05 + 0.5 * OBSTACLE_LENGTH   # 前肢 升起点: +5cm+a/2
REAR_DOWN_OFF = 0.04 - 0.5 * OBSTACLE_LENGTH # 后肢 降起点: -4cm+a/2 = 0.025
REAR_UP_OFF = 0.04 + OBSTACLE_LENGTH          # 后肢 升起点: +4cm+a

# ==================== 走廊 (高度上下限) ====================
# 前肢走廊: 用于简化模型中的头部 + 前躯干(前肢)
CORRIDOR_FRONT_HALF_HEIGHT = 0.025   # 前肢走廊半高 (m)
# 后肢走廊: 用于简化模型中的后躯干(后肢)
CORRIDOR_REAR_HALF_HEIGHT = 0.025    # 后肢走廊半高 (m)
# 身体段半高 (简化模型三段高度均为 50mm)
BODY_SEG_HALF_HEIGHT = 0.025         # 身体段半高 (m)


# 采样洞位置 (洞左侧 x): 第一个洞 >= HOLE_MIN_X, 间距 >= HOLE_MIN_SPACING
def sample_hole_positions(env: "ManagerBasedRlEnv", n: int) -> torch.Tensor:
    device = env.device
    x_max = HOLE_MAX_X - (HOLE_NUM - 1) * HOLE_MIN_SPACING
    x0 = torch.rand(n, device=device) * (x_max - HOLE_MIN_X) + HOLE_MIN_X
    gaps = HOLE_MIN_SPACING + torch.rand(n, HOLE_NUM - 1, device=device) * 0.2
    offsets = torch.cat([torch.zeros(n, 1, device=device), gaps.cumsum(dim=1)], dim=1)
    return x0.unsqueeze(1) + offsets   # [N, HOLE_NUM] 洞左侧


# 获取当前 episode 的洞位置 (env 缓存, 单 episode 内固定); 未采样时用默认单洞 0.185
def get_holes(env: "ManagerBasedRlEnv") -> torch.Tensor:
    holes = getattr(env, "_tunnel_hole_xs", None)
    if holes is None:
        holes = torch.full((env.num_envs, 1), 0.185, device=env.device)
    return holes


# 获取指定 body_link 的偏航角
def get_body_heading(env: "ManagerBasedRlEnv", body_id: int | None = None) -> torch.Tensor:
    asset = env.scene["robot"]
    if body_id is None:
        quat = asset.data.root_link_quat_w
    else:
        quat = asset.data.body_link_quat_w[:, body_id]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sin_h = 2.0 * (w * z + x * y)
    cos_h = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(sin_h, cos_h)


# F_body_Link 物理前向 heading (body+X ≠ 物理前向, 需减 π/2)
def get_f_body_physical_heading(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return get_body_heading(env, _MODEL_INDICES.f_body_id) - (torch.pi / 2)


# H_body_Link 物理前向 heading (body+X → world -Y, 需加 π/2)
def get_h_body_physical_heading(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return get_body_heading(env, _MODEL_INDICES.h_body_id) + (torch.pi / 2)


# 多洞期望高度: xs [N], holes [N, M] (洞左侧), 返回 [N] 期望高度
def _height_from_holes(xs: torch.Tensor, holes: torch.Tensor,
                       down_off: float, up_off: float) -> torch.Tensor:
    down = holes - down_off          # 低区间起点 (洞左侧 - 偏移)
    up = holes + up_off              # 低区间终点
    low_mask = (xs.unsqueeze(1) >= down) & (xs.unsqueeze(1) <= up)  # [N, M]
    any_low = low_mask.any(dim=1)
    return torch.where(any_low, torch.full_like(xs, HEIGHT_HOLE), torch.full_like(xs, HEIGHT_NORMAL))


# 前肢中心期望高度 (按实际 x 查询, 多洞)
def get_front_center_height(env: "ManagerBasedRlEnv", x: torch.Tensor) -> torch.Tensor:
    return _height_from_holes(x, get_holes(env), FRONT_DOWN_OFF, FRONT_UP_OFF)


# 后肢中心期望高度 (按实际 x 查询, 多洞)
def get_rear_center_height(env: "ManagerBasedRlEnv", x: torch.Tensor) -> torch.Tensor:
    return _height_from_holes(x, get_holes(env), REAR_DOWN_OFF, REAR_UP_OFF)


# 获取路径瞬时曲率 — Tunnel 直行, 恒为 0 (对齐 Slalom reference 接口)
def get_path_curvature(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


# 路径参考: 前/后肢中心期望高度 + 高度误差 (供策略观测)
def compute_path_ref(env: "ManagerBasedRlEnv"):
    asset = env.scene["robot"]
    resolve_model_indices(asset)
    body_pos = asset.data.body_link_pos_w
    # 前肢组 (头部 + 前躯干) 用前肢轨迹; 后肢组 (后躯干) 用后肢轨迹
    x_head = body_pos[:, _MODEL_INDICES.head_body_id, 0]
    z_head = body_pos[:, _MODEL_INDICES.head_body_id, 2]
    x_f = body_pos[:, _MODEL_INDICES.f_body_id, 0]
    z_f = body_pos[:, _MODEL_INDICES.f_body_id, 2]
    x_h = body_pos[:, _MODEL_INDICES.h_body_id, 0]
    z_h = body_pos[:, _MODEL_INDICES.h_body_id, 2]

    front_z_ref = get_front_center_height(env, x_f)
    rear_z_ref = get_rear_center_height(env, x_h)
    # 前肢组误差取头部/前躯干中较大者 (保证两者都贴走廊)
    front_err = torch.maximum((z_head - get_front_center_height(env, x_head)).abs(),
                              (z_f - front_z_ref).abs())
    rear_err = (z_h - rear_z_ref).abs()
    return front_z_ref, rear_z_ref, front_err, rear_err


# 单段走廊超额: v = max(0, |Δz| + 段半高 - 走廊半高)
def _segment_excess(env: "ManagerBasedRlEnv", z_actual: torch.Tensor, x_actual: torch.Tensor,
                    z_ref_fn, corridor_half: float, seg_half: float) -> torch.Tensor:
    z_ref = z_ref_fn(env, x_actual)
    e = (z_actual - z_ref).abs() + seg_half
    return (e - corridor_half).clamp(min=0.0)


# 前肢走廊超额: 头部 + 前躯干 (取最大)
def compute_corridor_front_excess(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset = env.scene["robot"]
    resolve_model_indices(asset)
    body_pos = asset.data.body_link_pos_w
    x_head = body_pos[:, _MODEL_INDICES.head_body_id, 0]
    z_head = body_pos[:, _MODEL_INDICES.head_body_id, 2]
    x_f = body_pos[:, _MODEL_INDICES.f_body_id, 0]
    z_f = body_pos[:, _MODEL_INDICES.f_body_id, 2]
    v_head = _segment_excess(env, z_head, x_head, get_front_center_height,
                             CORRIDOR_FRONT_HALF_HEIGHT, BODY_SEG_HALF_HEIGHT)
    v_f = _segment_excess(env, z_f, x_f, get_front_center_height,
                          CORRIDOR_FRONT_HALF_HEIGHT, BODY_SEG_HALF_HEIGHT)
    return torch.maximum(v_head, v_f)


# 后肢走廊超额: 后躯干
def compute_corridor_rear_excess(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset = env.scene["robot"]
    resolve_model_indices(asset)
    body_pos = asset.data.body_link_pos_w
    x_h = body_pos[:, _MODEL_INDICES.h_body_id, 0]
    z_h = body_pos[:, _MODEL_INDICES.h_body_id, 2]
    return _segment_excess(env, z_h, x_h, get_rear_center_height,
                           CORRIDOR_REAR_HALF_HEIGHT, BODY_SEG_HALF_HEIGHT)
