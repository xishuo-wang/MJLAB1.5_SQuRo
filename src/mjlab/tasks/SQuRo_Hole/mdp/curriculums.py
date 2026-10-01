from __future__ import annotations
from typing import Any

from .config import (
    STAGE1_END_ITER,
    STAGE2_END_ITER,
    STAGE3_END_ITER,
    STEPS_PER_ITER as _STEPS_PER_ITER,
)



# 阶段边界 (iter): 与 command.py 共用 config 里的同一来源, 四段一一对应
STAGE1_1_ITER = STAGE1_END_ITER      # 0-1k:  正常高度随机采样 (纯模仿)
STAGE1_2_ITER = STAGE2_END_ITER      # 1k-2k: 含低高度全档随机采样 (高度课程收紧)
STAGE1_3_ITER = STAGE3_END_ITER      # 2k-3k: 位置表 + 虚拟碰撞 (body_contact)
_STAGES = (0, STAGE1_1_ITER, STAGE1_2_ITER, STAGE1_3_ITER)



# 障碍物碰撞开关 (编译期固化, 运行期改 contype 无效)
ENABLE_HOLES_ITER = STAGE3_END_ITER  # 3k 起进入真实碰撞阶段
HOLE_ENTITY_NAMES = ("hole1", "hole2", "hole3")



# 奖励权重课程曲线: 每个奖励项/σ 一条曲线, 按 iter 落在 _STAGES 的第几档取值
_CURVES: dict[str, tuple[float, ...]] = {
    # 奖励项
    "mimic_pos":                (14.0, 14.0, 14.0, 14.0),
    "mimic_vel":                (5.0, 5.0, 5.0, 5.0),
    "vel":                      (5.0, 5.0, 5.0, 10.0),
    "height":                   (2.5, 2.5, 5.0, 10.0),
    "foot_clearance":           (1.0, 1.0, 1.0, 1.0),
    "reached":                  (1.0, 1.0, 1.0, 1.0),
    "orientation":              (2.0, 2.0, 4.0, 4.0),
    "angle":                    (1.0, 1.0, 1.0, 1.0),
    "smoothness":               (0.1, 0.2, 0.5, 1.0),
    "body_contact":             (0.0, 0.0, 1.0, 1.0),
    # 非权重项
    "mimic_pos_sigma":          (5.0, 10.0, 10.0, 5.0),
    "mimic_vel_sigma":          (0.1, 0.1, 0.1, 0.1),
    "height_sigma":             (500.0, 1000.0, 1000.0, 1000.0),
}



# 奖励权重课程: 按当前训练 iter 返回各奖励项权重/σ
class RewardWeightCurriculum:
    def get_reward_weights(self, current_step: int) -> dict[str, float]:
        current_iter = current_step // _STEPS_PER_ITER
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
        current_iter = current_step // _STEPS_PER_ITER
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


    # 关卡是否要求开启限高板碰撞 (只读元数据; 实际切换由 runner 重建环境完成)
    def should_enable_holes(self, current_step: int) -> bool:
        return current_step // _STEPS_PER_ITER >= ENABLE_HOLES_ITER


    def __init__(self):
        pass


    # 兼容旧接口: 按阶段阈值 (步数) 展开的权重表视图
    @property
    def weight_stages(self) -> dict[int, dict[str, float]]:
        return {t * _STEPS_PER_ITER: self.get_reward_weights(t * _STEPS_PER_ITER)
                for t in _STAGES}


reward_weight_curriculum = RewardWeightCurriculum()



# 获取课程奖励权重
def get_curriculum_reward_weight(env, reward_name: str) -> float:
    return reward_weight_curriculum.get_reward_weights(env.common_step_counter).get(reward_name, 1.0)


# 课程是否要求开启限高板碰撞 (只读; 切换由 runner 重建环境完成)
def curriculum_requires_collision(env) -> bool:
    return reward_weight_curriculum.should_enable_holes(env.common_step_counter)
