from __future__ import annotations
from typing import Any
from .config import FULL_HEIGHT_VALUES, RANDOM_HEIGHT_VALUES


# 课程学习边界
STAGE1_END_ITER = 1000      # 1) 0-1k     正常高度范围内的随机采样
STAGE2_END_ITER = 2000      # 2) 1k-2k    含低高度状态的全高度随机采样
STAGE3_END_ITER = 3000      # 3) 2k-3k    位置表 (位移时刻表) + 虚拟碰撞软约束
NUM_STAGES = 4              # 4) 3k 以后  位置表 + 真实碰撞

STEPS_PER_ITER = 48
STAGE1_END = STAGE1_END_ITER * STEPS_PER_ITER
STAGE2_END = STAGE2_END_ITER * STEPS_PER_ITER
STAGE3_END = STAGE3_END_ITER * STEPS_PER_ITER

# 各阶段是否要求限高板实体碰撞 (编译期固化; 切换由 runner 重建环境完成)
STAGE_COLLISION = (False, False, False, True)



# 按全局步数取当前阶段 (1~4); 阶段边界是命令/课程/碰撞的唯一来源
def get_current_stage(step_counter: int) -> int:
    if step_counter < STAGE1_END:
        return 1
    if step_counter < STAGE2_END:
        return 2
    if step_counter < STAGE3_END:
        return 3
    return 4


# 该阶段是否使用位置表 (阶段 3/4)
def stage_uses_schedule(stage: int) -> bool:
    return stage >= 3


# 该阶段是否要求实体碰撞
def stage_requires_collision(stage: int) -> bool:
    idx = min(max(int(stage), 1), NUM_STAGES) - 1
    return STAGE_COLLISION[idx]


# 按阶段的随机采样高度池; 阶段 3/4 返回 None (用位置表)
def heights_for_stage(stage: int):
    if stage == 1:
        return RANDOM_HEIGHT_VALUES
    if stage == 2:
        return FULL_HEIGHT_VALUES
    return None



# 四段课程对应的 iter 边界 (供 _CURVES 的 4 段取值)
STAGE1_1_ITER = STAGE1_END_ITER      # 0-1k:  正常高度随机采样 (纯模仿)
STAGE1_2_ITER = STAGE2_END_ITER      # 1k-2k: 含低高度全档随机采样 (高度课程收紧)
STAGE1_3_ITER = STAGE3_END_ITER      # 2k-3k: 位置表 + 虚拟碰撞 (body_contact)
_STAGES = (0, STAGE1_1_ITER, STAGE1_2_ITER, STAGE1_3_ITER)




# 奖励权重课程曲线
_CURVES: dict[str, tuple[float, ...]] = {
    # 奖励项
    "mimic_pos":                (10.0, 10.0, 10.0, 10.0),
    "mimic_vel":                (5.0, 5.0, 5.0, 5.0),
    "vel":                      (5.0, 5.0, 5.0, 10.0),
    "height":                   (2.5, 2.5, 5.0, 5.0),
    "orientation":              (2.0, 2.0, 4.0, 4.0),
    "angle":                    (1.0, 1.0, 1.0, 1.0),
    "smoothness":               (0.1, 0.2, 0.5, 1.0),
    "body_contact":             (0.0, 0.0, 5.0, 5.0),
    # 虚拟净空 σ: 超出量的指数衰减系数 (超出 5mm 时按 exp(-sigma*0.005^2) 计)
    "sigma_body_contact":       (2000.0, 2000.0, 2000.0, 2000.0),
    # 位置/速度模仿的腿部 σ 与颈部权重
    "sigma_leg_pos":            (5.0, 10.0, 10.0, 5.0),
    "sigma_leg_vel":            (0.1, 0.1, 0.1, 0.1),
    "alpha_neck_pos":           (0.30, 0.30, 0.30, 0.30),
    "alpha_neck_vel":           (0.30, 0.30, 0.30, 0.30),
    # 位置/速度模仿的脊柱 σ 与脊柱项权重
    "sigma_spn_pos":            (5.0, 10.0, 10.0, 5.0),
    "sigma_spn_vel":            (0.1, 0.1, 0.1, 0.1),
    "alpha_spn_pos":            (1.0, 1.0, 1.0, 1.0),
    "alpha_spn_vel":            (1.0, 1.0, 1.0, 1.0),
    # 头颈 σ: 曲线与脊柱一致, 保证替换模仿奖励前后行为不变
    "sigma_neck_pos":           (5.0, 10.0, 10.0, 5.0),
    "sigma_neck_vel":           (0.1, 0.1, 0.1, 0.1),
    # 高度 σ
    "height_sigma":             (500.0, 1000.0, 1000.0, 1000.0),
}



# 脊柱误差的逐轴缩放, 顺序 = 参考表脊柱 4 列 (F_spine1, F_body, H_spine1, H_body)
# 即 (侧摆, 扭转, 俯仰, 扭转)
SPN_AXIS_SCALE: tuple[float, float, float, float] = (1.5, 0.7, 1.5, 0.7)



# 奖励权重课程: 按当前训练 iter 返回各奖励项权重/σ
class RewardWeightCurriculum:
    def get_reward_weights(self, current_step: int) -> dict[str, float]:
        current_iter = current_step // STEPS_PER_ITER
        result: dict[str, float] = {}
        for name, values in _CURVES.items():
            idx = 0
            for i, t in enumerate(_STAGES):
                if current_iter >= t:
                    idx = i
            safe_idx = idx if idx < len(values) else len(values) - 1
            result[name] = values[safe_idx]
        return result


    def get_current_stage_info(self, current_step: int) -> dict[str, Any]:
        current_iter = current_step // STEPS_PER_ITER
        stage = 0
        for t in _STAGES:
            if current_iter >= t:
                stage = t
        return {
            "current_stage": stage,
            "current_iter": current_iter,
            "current_step": current_step,
            "reward_weights": self.get_reward_weights(current_step),
        }


    # 关卡是否要求开启限高板碰撞 (只读视图, 供基线脚本展示课程意图)
    def should_enable_holes(self, current_step: int) -> bool:
        return stage_requires_collision(get_current_stage(current_step))


    def __init__(self):
        pass


    # 兼容旧接口: 按阶段阈值 (步数) 展开的权重表视图
    @property
    def weight_stages(self) -> dict[int, dict[str, float]]:
        return {t * STEPS_PER_ITER: self.get_reward_weights(t * STEPS_PER_ITER)for t in _STAGES}


reward_weight_curriculum = RewardWeightCurriculum()



# 获取课程奖励权重
def get_curriculum_reward_weight(env, reward_name: str) -> float:
    return reward_weight_curriculum.get_reward_weights(env.common_step_counter).get(reward_name, 1.0)
