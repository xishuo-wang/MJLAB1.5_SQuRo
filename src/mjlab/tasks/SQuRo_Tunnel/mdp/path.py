from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from .indices import _MODEL_INDICES, resolve_model_indices
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# ==================== 障碍物 (门洞式洞) 配置 ====================
# 参考 hole.py HoleEntityCfg: 左侧 = x, 长度 = a, 洞下沿 = position z
OBSTACLE_X_LEFT = 0.185      # 障碍物左侧 x (m)
OBSTACLE_LENGTH = 0.03       # 障碍物长度 a (m) → 右侧 x+a = 0.215
HOLE_BOTTOM = 0.055          # 洞下沿高度 (m) = hole position z
HOLE_THICKNESS = 0.01        # 限高板厚度 (m) = 2*size[2]

# ==================== 期望高度规则 ====================
HEIGHT_NORMAL = 0.06         # 正常段期望高度 (m)
HEIGHT_CLEARANCE = 0.025     # 低高度相对洞下沿下降量 (m) = 走廊半高
HEIGHT_HOLE = HOLE_BOTTOM - HEIGHT_CLEARANCE   # 低高度期望 = 0.055 - 0.025 = 0.03
TRANSITION_LENGTH = 0.0      # 线性过渡长度 (m): 正常↔低高度 — 先设 0 (方波)

# ==================== 三点轨迹起点 (m, 由 cm 公式换算) ====================
# 前肢中心: 降 x-9cm, 升 x+5+(a/2)
FRONT_DOWN = OBSTACLE_X_LEFT - 0.09
FRONT_UP = OBSTACLE_X_LEFT + 0.05 + OBSTACLE_LENGTH / 2
# 后肢中心: 降 x-4-(a/2), 升 x+4+(a)
REAR_DOWN = OBSTACLE_X_LEFT - 0.04 - OBSTACLE_LENGTH / 2
REAR_UP = OBSTACLE_X_LEFT + 0.04 + OBSTACLE_LENGTH

# ==================== 走廊 (高度上下限) ====================
# 前肢走廊: 用于简化模型中的头部 + 前躯干(前肢)
CORRIDOR_FRONT_HALF_HEIGHT = 0.025   # 前肢走廊半高 (m)
# 后肢走廊: 用于简化模型中的后躯干(后肢)
CORRIDOR_REAR_HALF_HEIGHT = 0.025    # 后肢走廊半高 (m)
# 身体段半高 (简化模型三段高度均为 50mm)
BODY_SEG_HALF_HEIGHT = 0.025         # 身体段半高 (m)


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


# 计算单条期望高度轨迹 (梯形波): [down, down+trans] 线性降, [down+trans, up] 低, [up, up+trans] 升
def compute_point_height(xs: torch.Tensor, down: float, up: float,
                         trans: float = TRANSITION_LENGTH) -> torch.Tensor:
    if trans <= 1e-9:
        # 方波: 过渡长度 0
        low_mask = (xs >= down) & (xs <= up)
        return torch.where(low_mask, torch.full_like(xs, HEIGHT_HOLE), torch.full_like(xs, HEIGHT_NORMAL))
    z = torch.full_like(xs, HEIGHT_NORMAL)
    in_down = (xs >= down) & (xs < down + trans)
    t_down = (xs - down) / trans
    z = torch.where(in_down, HEIGHT_NORMAL + (HEIGHT_HOLE - HEIGHT_NORMAL) * t_down, z)
    low_mask = (xs >= down + trans) & (xs <= up)
    z = torch.where(low_mask, torch.full_like(xs, HEIGHT_HOLE), z)
    in_up = (xs > up) & (xs <= up + trans)
    t_up = (xs - up) / trans
    z = torch.where(in_up, HEIGHT_HOLE + (HEIGHT_NORMAL - HEIGHT_HOLE) * t_up, z)
    return z


# 前肢中心期望高度 (按实际 x 查询)
def get_front_center_height(env: "ManagerBasedRlEnv", x: torch.Tensor) -> torch.Tensor:
    return compute_point_height(x, FRONT_DOWN, FRONT_UP, TRANSITION_LENGTH)


# 后肢中心期望高度 (按实际 x 查询)
def get_rear_center_height(env: "ManagerBasedRlEnv", x: torch.Tensor) -> torch.Tensor:
    return compute_point_height(x, REAR_DOWN, REAR_UP, TRANSITION_LENGTH)


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
def _segment_excess(z_actual: torch.Tensor, x_actual: torch.Tensor,
                    z_ref_fn, corridor_half: float, seg_half: float) -> torch.Tensor:
    z_ref = z_ref_fn(None, x_actual)  # 类型忽略: z_ref_fn(env, x)
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
    v_head = _segment_excess(z_head, x_head, get_front_center_height,
                             CORRIDOR_FRONT_HALF_HEIGHT, BODY_SEG_HALF_HEIGHT)
    v_f = _segment_excess(z_f, x_f, get_front_center_height,
                          CORRIDOR_FRONT_HALF_HEIGHT, BODY_SEG_HALF_HEIGHT)
    return torch.maximum(v_head, v_f)


# 后肢走廊超额: 后躯干
def compute_corridor_rear_excess(env: "ManagerBasedRlEnv") -> torch.Tensor:
    asset = env.scene["robot"]
    resolve_model_indices(asset)
    body_pos = asset.data.body_link_pos_w
    x_h = body_pos[:, _MODEL_INDICES.h_body_id, 0]
    z_h = body_pos[:, _MODEL_INDICES.h_body_id, 2]
    return _segment_excess(z_h, x_h, get_rear_center_height,
                           CORRIDOR_REAR_HALF_HEIGHT, BODY_SEG_HALF_HEIGHT)
