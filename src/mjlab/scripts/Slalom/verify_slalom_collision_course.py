from types import SimpleNamespace as NS
import unittest

import torch

from mjlab.tasks.SQuRo_Slalom.mdp.passage import SlalomPassageMonitor
from mjlab.tasks.SQuRo_Slalom.mdp.terminations import check_episode_timeout
from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import (
    SlalomSpacingCurriculum, SPACING_SUCCESS_BATCH,
)


# 创建实际位置通过判据的最小输入，两个环境允许检查局部重置。
def make_monitor():
    command = NS(pole_spacing=torch.full((2,), 0.2), pole_y=torch.zeros(2),
                 phase1_mask=torch.ones(2, dtype=torch.bool))
    env = NS(num_envs=2, device="cpu", scene=NS(env_origins=torch.zeros(2, 3)),
             command_manager=NS(_terms={"slalom_cmd": command}))
    return SlalomPassageMonitor(env)


# 生成按交替侧别绕杆的躯干中心，侧别取决于实际位置。
def feed_path(monitor, wrong_side=False, rear_lag=0.1):
    for x in torch.linspace(0.1, 0.75, 150):
        longitudinal = x + torch.tensor([0.0, 0.04, -0.04])
        lateral = 0.08 * torch.cos(torch.pi * (longitudinal - 0.2) / 0.2)
        if wrong_side:
            lateral = lateral.abs()
        centers = torch.stack((longitudinal, lateral), dim=-1)[None].repeat(2, 1, 1)
        monitor.update(centers, torch.full((2,), float(x - rear_lag)))


class TestSlalomPassage(unittest.TestCase):
    # 正确交替绕行且全身越过三杆才通过。
    def test_actual_passage(self):
        monitor = make_monitor()
        feed_path(monitor)
        self.assertTrue(bool(monitor.passed.all()))
        self.assertFalse(bool(monitor.wrong_side.any()))

    # 只经过三个纵向杆位，但未左右交替，必须失败。
    def test_straight_cut_is_not_success(self):
        monitor = make_monitor()
        feed_path(monitor, wrong_side=True)
        self.assertTrue(bool(monitor.wrong_side.all()))
        self.assertFalse(bool(monitor.passed.all()))

    # 基座过杆而足端尚未过杆时，不记最后一杆通过。
    def test_rear_extent_must_clear(self):
        monitor = make_monitor()
        feed_path(monitor, rear_lag=0.2)
        self.assertTrue(bool(monitor.crossed.all()))
        self.assertTrue(bool(monitor.passed[:, :2].all()))
        self.assertFalse(bool(monitor.passed[:, 2].any()))

    # 局部重置不影响其他环境已经取得的进度。
    def test_partial_reset(self):
        monitor = make_monitor()
        feed_path(monitor)
        monitor.reset(torch.tensor([0]))
        self.assertFalse(bool(monitor.passed[0].any()))
        self.assertTrue(bool(monitor.passed[1].all()))

    # 混合阶段分别遵守二十秒和三十秒回合上限。
    def test_phase_specific_timeout(self):
        command = NS(phase1_mask=torch.tensor([False, True]))
        env = NS(command_manager=NS(_terms={"slalom_cmd": command}),
                 step_dt=0.01, max_episode_length=3000,
                 episode_length_buf=torch.tensor([2000, 2000]))
        self.assertEqual(check_episode_timeout(env).tolist(), [True, False])
        env.episode_length_buf[:] = 3000
        self.assertTrue(bool(check_episode_timeout(env).all()))


# 用完整考核批次检查推进边界，避免用无碰撞训练结果冒充成绩。
class TestSpacingCurriculum(unittest.TestCase):
    # 制造指定档位的一批完整碰撞考核结果。
    def batch(self, curvature, successes=SPACING_SUCCESS_BATCH):
        return [{"curvature": curvature, "success": index < successes,
                 "failure": "" if index < successes else "跌倒"}
                for index in range(SPACING_SUCCESS_BATCH)]

    # 无碰撞预训练至少五百轮，之后才开始考核。
    def test_pretraining_and_two_batches(self):
        course = SlalomSpacingCurriculum()
        self.assertFalse(course.should_evaluate(3499))
        self.assertTrue(course.should_evaluate(3500))
        self.assertFalse(course.record_evaluation(self.batch(10, 154), 3500, True))
        self.assertFalse(course.should_evaluate(3599))
        self.assertTrue(course.record_evaluation(self.batch(10, 154), 3600, True))
        self.assertEqual(course.max_curvature, 11)
        self.assertFalse(course.should_evaluate(3699))

    # 失败批次清空连续达标数，不会跳档。
    def test_failure_breaks_streak(self):
        course = SlalomSpacingCurriculum()
        course.record_evaluation(self.batch(10), 3500, True)
        course.record_evaluation(self.batch(10, 153), 3600, True)
        self.assertEqual(course.consecutive_success_batches, 0)
        self.assertEqual(course.max_curvature, 10)

    # 无碰撞结果、旧档位与不完整批次均被拒绝。
    def test_invalid_results_rejected(self):
        course = SlalomSpacingCurriculum()
        for results, collision in ((self.batch(10), False), (self.batch(9), True),
                                   (self.batch(10)[:-1], True)):
            with self.assertRaises(ValueError):
                course.record_evaluation(results, 3500, collision)
        self.assertEqual(course.history, [])

    # 同一轮不能重复计分，保存恢复后保留连续批次。
    def test_resume_and_duplicate(self):
        course = SlalomSpacingCurriculum()
        course.record_evaluation(self.batch(10), 3500, True)
        course.record_evaluation(self.batch(10), 3500, True)
        self.assertEqual(len(course.history), 1)
        restored = SlalomSpacingCurriculum()
        restored.load_state_dict(course.state_dict())
        self.assertTrue(restored.record_evaluation(self.batch(10), 3600, True))

    # 曲率十五仍须通过考核才标记最终档达标，不能继续增长。
    def test_final_level(self):
        course = SlalomSpacingCurriculum()
        course.max_curvature = 15
        course.record_evaluation(self.batch(15), 3500, True)
        course.record_evaluation(self.batch(15), 3600, True)
        self.assertTrue(course.final_level_passed)
        self.assertEqual(course.max_curvature, 15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
