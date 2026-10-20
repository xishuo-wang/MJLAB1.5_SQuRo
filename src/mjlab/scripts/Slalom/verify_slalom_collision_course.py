from types import SimpleNamespace as NS
import unittest

import torch

from mjlab.tasks.SQuRo_Slalom.mdp.terminations import SlalomPassageMonitor
from mjlab.tasks.SQuRo_Slalom.mdp.terminations import check_episode_timeout
from mjlab.tasks.SQuRo_Slalom.mdp.curriculums import (
    SLALOM_BIN_COUNT, SlalomSpacingCurriculum,
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
        self.assertEqual(check_episode_timeout(env).tolist(), [True, False])  # type: ignore
        env.episode_length_buf[:] = 3000
        self.assertTrue(bool(check_episode_timeout(env).all()))  # type: ignore


# 制造各档位指定成功数的完整考核结果。
def make_results(plan: dict, count: int = 1024) -> list[dict]:
    results = []
    for curvature, successes in plan.items():
        results.extend({"curvature": curvature, "success": index < successes,
                        "failure": "" if index < successes else "跌倒"}
                       for index in range(count))
    return results


# 用完整考核批次检查推进、最终验收与分档权重。
class TestSpacingCurriculum(unittest.TestCase):
    # 无碰撞预训练至少五百轮，两批最低档都达标才推进一档。
    def test_pretraining_and_advance(self):
        course = SlalomSpacingCurriculum()
        self.assertFalse(course.should_evaluate(3499))
        self.assertTrue(course.should_evaluate(3500))
        batches = [course.measure_batch(make_results({10: 700})) for _ in range(2)]
        result = course.record_evaluation(batches, 3500, True)
        self.assertTrue(result["passed"])
        self.assertTrue(result["advanced"])
        self.assertEqual(course.max_curvature, 11)
        self.assertFalse(course.should_evaluate(3599))

    # 任一批未达最低档 60% 就不推进，连续达标数清零。
    def test_failure_keeps_level(self):
        course = SlalomSpacingCurriculum()
        batches = [course.measure_batch(make_results({10: 700})),
                   course.measure_batch(make_results({10: 500}))]
        result = course.record_evaluation(batches, 3500, True)
        self.assertFalse(result["passed"])
        self.assertEqual(course.max_curvature, 10)
        self.assertEqual(course.consecutive_success_batches, 0)

    # 无碰撞结果、批数不足被拒绝，未到考核间隔的重复提交不计分。
    def test_invalid_results_rejected(self):
        course = SlalomSpacingCurriculum()
        with self.assertRaises(ValueError):
            course.record_evaluation([], 3500, False)
        with self.assertRaises(ValueError):
            course.record_evaluation([course.measure_batch(make_results({10: 700}))], 3500, True)
        course.record_evaluation(
            [course.measure_batch(make_results({10: 700})) for _ in range(2)], 3500, True)
        duplicate = course.record_evaluation(
            [course.measure_batch(make_results({11: 700})) for _ in range(2)], 3500, True)
        self.assertFalse(duplicate["counted"])

    # 保存恢复后保留档位与分档权重，续训不会重新起课。
    def test_resume(self):
        course = SlalomSpacingCurriculum()
        course.record_evaluation(
            [course.measure_batch(make_results({10: 700})) for _ in range(2)], 3500, True)
        restored = SlalomSpacingCurriculum()
        restored.load_state_dict(course.state_dict())
        self.assertEqual(restored.max_curvature, 11)
        restored.record_evaluation(
            [restored.measure_batch(make_results({11: 700})) for _ in range(2)], 3600, True)
        self.assertEqual(restored.max_curvature, 12)

    # 分档权重按普通模式目标通过率补齐缺口，无样本档位保持 1.0。
    def test_bin_weights_from_gap(self):
        course = SlalomSpacingCurriculum()
        plan = {10: 1024, 9: 400, 8: 300}
        batches = [course.measure_batch(make_results(plan)) for _ in range(2)]
        course.record_evaluation(batches, 3500, True)
        weights = course.bin_weights
        self.assertAlmostEqual(weights[5], 1.0)
        self.assertAlmostEqual(weights[4], 1.0 + 2.0 * (0.60 - 400 / 1024), places=4)
        self.assertAlmostEqual(weights[3], 1.0 + 2.0 * (0.60 - 300 / 1024), places=4)
        self.assertEqual(weights[0], 1.0)

    # 曲率十五档通过后进入最终档，档位权重重置为等概率。
    def test_final_level_entry(self):
        course = SlalomSpacingCurriculum()
        course.max_curvature = 15
        batches = [course.measure_batch(make_results({15: 700})) for _ in range(2)]
        result = course.record_evaluation(batches, 3500, True)
        self.assertTrue(result["entered_final_mode"])
        self.assertTrue(course.final_level_passed)
        self.assertEqual(course.bin_weights, [1.0] * SLALOM_BIN_COUNT)

    # 最终档两批整体通过率都达到 80% 才算最终验收，并按缺口更新训练权重。
    def test_final_acceptance(self):
        course = SlalomSpacingCurriculum()
        course.max_curvature = 15
        course.final_level_passed = True
        failing = {curvature: 700 for curvature in range(5, 16)}
        passing = {curvature: 900 for curvature in range(5, 16)}
        result = course.record_evaluation(
            [course.measure_final_batch(make_results(failing)) for _ in range(2)], 3500, True)
        self.assertFalse(result["passed"])
        self.assertFalse(course.final_accepted)
        self.assertGreater(course.bin_weights[0], 1.0)
        result = course.record_evaluation(
            [course.measure_final_batch(make_results(passing)) for _ in range(2)], 3600, True)
        self.assertTrue(result["passed"])
        self.assertTrue(course.final_accepted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
