# 独立真实碰撞墙位课程验收: 状态边界、翻正快照与训练隔离
import copy
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from mjlab.scripts.Backup import verify_backup_corridor_rebuild as rebuild
from mjlab.scripts.Backup.verify_backup_stage_rewards import make_env
from mjlab.tasks.SQuRo_Backup.mdp import curriculums as C
from mjlab.tasks.SQuRo_Backup.mdp.indices import _MODEL_INDICES
from mjlab.tasks.SQuRo_Backup.mdp.terminations import righting_time_out


# 一批独立环境首回合: 最低档 256 个, 其余使用宽墙位
def evaluation_results(lower: float, passed: bool = True, minimum_count: int = 256) -> list[dict]:
    return [
        {"env_id": i, "valid": True, "wall_d": lower if i < minimum_count else C.WALL_D_MAX,
         "success": passed}
        for i in range(C.WALL_EVALUATION_ENVS)
    ]


# 使用生产统计器构造两批同档成绩
def batches(course, flags=(True, True)):
    return [course.measure_batch(evaluation_results(course.d_min, flag)) for flag in flags]


class WallStateTests(unittest.TestCase):
    # 使用中间墙位初始化课程测试
    def setUp(self):
        self.course = C.WallCurriculumState()
        self.course.d_min = 0.10

    # 核对净宽下限与旧离散表终点
    def test_clear_width_floor_and_legacy_endpoint(self):
        from mjlab.tasks.SQuRo_Backup.mdp.entity import WALL_HALF_THICKNESS
        self.assertAlmostEqual(C.WALL_D_MIN_END + C.WALL_X_POS - 2 * WALL_HALF_THICKNESS, 0.09)
        self.assertAlmostEqual(C.WALL_X_NEG_LEVELS[-1], -0.06)
        self.assertEqual(len(C.WALL_D_LEVELS), 15)
        self.assertEqual(C.WALL_BIN_COUNT, 14)

    # 核对考核间隔与续训去重
    def test_schedule_and_resume_skip_completed_event(self):
        for iteration in (3000, 3400, 3499, 3501):
            self.assertFalse(self.course.should_evaluate(iteration))
        self.assertTrue(self.course.should_evaluate(3500))
        self.course.record_evaluation(batches(self.course, (False, False)), 3500)
        self.assertFalse(self.course.should_evaluate(3500))
        self.course.final_level_passed = True
        self.assertFalse(self.course.should_evaluate(3600))
        self.assertTrue(self.course.should_evaluate(4000))

    # 核对最低档计分与样本门槛
    def test_minimum_only_success_rate_and_sample_floor(self):
        results = evaluation_results(0.10)
        for item in results[:102]:
            item["success"] = False
        stats = self.course.measure_batch(results)
        self.assertEqual(stats["count"], 256)
        self.assertEqual(stats["successes"], 154)
        self.assertTrue(stats["passed"])
        results[102]["success"] = False
        self.assertFalse(self.course.measure_batch(results)["passed"])
        self.assertFalse(self.course.measure_batch(evaluation_results(0.10, minimum_count=255))["passed"])

    # 核对同档两批只推进一次并保留成绩
    def test_two_batches_commit_one_step_and_keep_last_metrics(self):
        stats = batches(self.course)
        self.assertEqual(self.course.d_min, 0.10)
        result = self.course.record_evaluation(stats, 3500)
        self.assertTrue(result["advanced"])
        self.assertAlmostEqual(self.course.d_min, 0.09)
        self.assertEqual(self.course.metrics()["success_rate"], 1.0)
        self.assertEqual(self.course.metrics()["sample_count"], 512)

    # 核对不完整或混档结果不修改课程
    def test_incomplete_or_mixed_batches_leave_state_untouched(self):
        for mode in ("single", "missing", "mixed"):
            stats = batches(self.course)
            if mode == "single":
                stats = stats[:1]
            elif mode == "missing":
                stats[1]["eligible"] -= 1
            else:
                stats[1]["wall_d_min"] = 0.09
            before = self.course.state_dict()
            with self.assertRaises(ValueError):
                self.course.record_evaluation(stats, 3500)
            self.assertEqual(self.course.state_dict(), before)

    # 核对跨轮交错成绩不能拼接达标
    def test_interleaved_rounds_cannot_borrow_a_successful_batch(self):
        self.course.record_evaluation(batches(self.course, (False, True)), 3500)
        self.course.record_evaluation(batches(self.course, (True, False)), 3600)
        self.assertEqual(self.course.d_min, 0.10)
        self.assertEqual(self.course.consecutive_success_evaluations, 0)
        self.assertEqual(self.course.consecutive_failures, 2)

    # 核对每档最短驻留
    def test_dwell_blocks_then_allows_one_step(self):
        self.course.level_start_iteration = 3480
        self.assertFalse(self.course.record_evaluation(batches(self.course), 3500)["advanced"])
        self.assertTrue(self.course.record_evaluation(batches(self.course), 3600)["advanced"])

    # 核对连续失败退档与两次通过恢复
    def test_five_failed_events_retreat_and_two_events_recover(self):
        for iteration in range(3500, 4000, 100):
            result = self.course.record_evaluation(batches(self.course, (False, True)), iteration)
        self.assertTrue(result["retreated"])
        self.assertAlmostEqual(self.course.d_min, 0.11)
        self.assertEqual(self.course.retreats, 1)
        self.assertFalse(self.course.record_evaluation(batches(self.course), 4000)["advanced"])
        self.assertTrue(self.course.record_evaluation(batches(self.course), 4100)["advanced"])
        self.assertAlmostEqual(self.course.d_min, 0.10)

    # 核对最终模式与两批八成翻正率门槛
    def test_floor_enters_final_then_requires_two_eighty_percent_batches(self):
        self.course.d_min = C.WALL_D_MIN_END
        result = self.course.record_evaluation(batches(self.course), 3500)
        self.assertTrue(result["entered_final_mode"])
        self.assertFalse(self.course.final_accepted)
        self.assertEqual(self.course.bin_weights, [1.0] * C.WALL_BIN_COUNT)
        mixed = batches(self.course)
        for item in mixed:
            self.assertEqual(item["count"], C.WALL_EVALUATION_ENVS)
        bad_results = evaluation_results(self.course.d_min)
        for item in bad_results[:205]:
            item["success"] = False
        mixed[1] = self.course.measure_batch(bad_results)
        self.assertFalse(self.course.record_evaluation(mixed, 4000)["passed"])
        self.assertFalse(self.course.final_accepted)
        self.assertTrue(self.course.record_evaluation(batches(self.course), 4500)["passed"])
        self.assertTrue(self.course.final_accepted)
        self.course.record_evaluation(batches(self.course, (False, False)), 5000)
        self.assertTrue(self.course.final_accepted)

    # 核对难例权重与无样本区间的默认值
    def test_bin_feedback_and_no_sample_default(self):
        self.course.record_evaluation(batches(self.course, (False, False)), 3500)
        self.assertAlmostEqual(self.course.bin_weights[4], 2.2)
        self.assertEqual(self.course.bin_weights[0], 1.0)

    # 核对检查点恢复及无效状态不改写当前状态
    def test_checkpoint_roundtrip_and_invalid_state_atomicity(self):
        self.course.record_evaluation(batches(self.course, (False, True)), 3500)
        saved = self.course.state_dict()
        restored = C.WallCurriculumState()
        restored.load_state_dict(saved)
        self.assertEqual(restored.state_dict(), saved)
        for key, value in (("d_min", 0.05), ("bin_weights", [1.0]), ("retreats", -1)):
            invalid = copy.deepcopy(saved)
            invalid[key] = value
            with self.assertRaises(ValueError):
                restored.load_state_dict(invalid)
            self.assertEqual(restored.state_dict(), saved)

    # 核对考核轮次与环境编号去重
    def test_duplicate_event_or_environment_rejected(self):
        self.course.record_evaluation(batches(self.course, (False, False)), 3500)
        with self.assertRaises(ValueError):
            self.course.record_evaluation(batches(self.course), 3500)
        results = evaluation_results(self.course.d_min)
        results[0]["env_id"] = results[1]["env_id"]
        with self.assertRaises(ValueError):
            self.course.measure_batch(results)

    # 核对普通模式下界占比与最终模式均匀分档
    def test_sampling_normal_minimum_atom_and_final_equal_bins(self):
        torch.manual_seed(14)
        distances = C.sample_wall_distances(40000, 0.06, 0.20, 0.30, [1.0] * C.WALL_BIN_COUNT, "cpu")
        self.assertAlmostEqual(float((distances == 0.06).float().mean()), 0.30, delta=0.01)
        final = C.sample_wall_distances(40000, 0.06, 0.20, 0.30, [1.0] * C.WALL_BIN_COUNT, "cpu", final_mode=True)
        counts = torch.bincount(C.wall_bin_indices(final), minlength=C.WALL_BIN_COUNT)
        self.assertLess(float((counts.float() / len(final) - 1 / C.WALL_BIN_COUNT).abs().max()), 0.007)
        self.assertTrue(bool(((final >= 0.06) & (final <= 0.20)).all()))

    # 核对区间权重只改变有效范围内的采样密度
    def test_sampling_weights_shift_density_within_current_range(self):
        torch.manual_seed(15)
        weights = [1.0] * C.WALL_BIN_COUNT
        weights[4] = 3.0
        distances = C.sample_wall_distances(30000, 0.10, 0.20, 0.0, weights, "cpu")
        counts = torch.bincount(C.wall_bin_indices(distances), minlength=C.WALL_BIN_COUNT)
        self.assertTrue(bool((counts[:4] == 0).all()))
        self.assertGreater(float(counts[4] / counts[5]), 2.7)


class RightingSnapshotTests(unittest.TestCase):
    # 保存共享索引并建立姿态判据的测试环境
    def setUp(self):
        self.saved_indices = {name: copy.deepcopy(getattr(_MODEL_INDICES, name)) for name in _MODEL_INDICES.__slots__}
        _MODEL_INDICES.f_body_id, _MODEL_INDICES.h_body_id = 0, 1
        self.env, self.cmd = make_env([0, 1])
        self.cmd._ensure_buffers()

    # 恢复其他测试使用的共享模型索引
    def tearDown(self):
        for name, value in self.saved_indices.items():
            setattr(_MODEL_INDICES, name, value)

    # 核对翻正确认不依赖相位时钟与关节速度
    def test_confirmation_ignores_phase_clock_and_joint_velocity(self):
        self.cmd.test_heights[:] = 0.024
        self.cmd.test_vel[:] = 1000
        for _ in range(9):
            self.cmd.record_righting_step()
        self.assertFalse(bool(self.cmd._ep_righted.any()))
        self.cmd.record_righting_step()
        self.assertTrue(bool(self.cmd._ep_righted.all()))
        self.assertTrue(bool((self.cmd.phase == torch.tensor([0, 1])).all()))

    # 核对中断确认和无效姿态不计成功
    def test_interrupted_confirmation_and_invalid_pose_do_not_pass(self):
        self.cmd.test_heights[:] = 0.024
        for _ in range(9):
            self.cmd.record_righting_step()
        self.cmd.test_u[:] = float("nan")
        self.cmd.record_righting_step()
        self.cmd.test_u[:] = 1
        for _ in range(9):
            self.cmd.record_righting_step()
        self.assertFalse(bool(self.cmd._ep_righted.any()))

    # 核对超时末步在完整复位前发布结果
    def test_terminal_step_is_recorded_before_full_reset(self):
        self.cmd.test_heights[:] = 0.024
        for _ in range(9):
            self.cmd.record_righting_step()
        self.env.episode_length_buf[:] = self.env.max_episode_length
        self.assertTrue(bool(righting_time_out(self.env).all()))
        self.cmd._wall_d[:] = torch.tensor([0.06, 0.20])
        self.cmd.reset(torch.arange(2))
        self.assertTrue(bool(self.cmd._last_ep_righted.all()))
        self.assertFalse(bool(self.cmd._ep_righted.any()))
        self.assertTrue(torch.equal(self.cmd._last_ep_wall_d, torch.tensor([0.06, 0.20])))
        self.assertFalse(bool(self.cmd._last_ep_valid.any()))

    # 核对循环重采保留回合翻正结果
    def test_cycle_resampling_preserves_episode_righting_result(self):
        self.cmd._ep_righted[:] = True
        self.cmd._resample_command(torch.arange(2))
        self.assertTrue(bool(self.cmd._ep_righted.all()))

    # 核对旁路统计不改变既有姿态缓存
    def test_recording_preserves_existing_pose_caches(self):
        cached = torch.ones(2, 2)
        self.cmd._pose_cos_cache = cached
        self.cmd.record_righting_step()
        self.assertIs(self.cmd._pose_cos_cache, cached)
        self.assertIsNone(self.cmd._pose_cache)


class EvaluationIsolationTests(unittest.TestCase):
    # 建立带真实课程状态的轻量训练替身
    def setUp(self):
        self.fixture = rebuild.CorridorRebuildTest()
        self.fixture.setUp()
        self.runner = self.fixture._build(3500, self.fixture._level0(), True)
        self.runner._wall_random = True
        self.runner._ensure_wall_state()
        self.runner._wall_course.d_min = 0.10
        self.runner._wall_d_min = 0.10
        self.runner.logger.log_dir = None

    # 核对两批使用同一档位再推送新范围
    def test_both_sweeps_use_same_level_and_push_next_course(self):
        seen = []
        # 返回当前档位的一批考核结果
        def sweep(seed):
            seen.append(self.runner._wall_course.d_min)
            return evaluation_results(seen[-1])
        self.runner._run_evaluation_sweep = sweep
        self.runner._maybe_evaluate_curriculum(3500)
        self.assertEqual(seen, [0.10, 0.10])
        self.assertAlmostEqual(self.runner._wall_course.d_min, 0.09)
        self.assertAlmostEqual(self.runner._command_term().wall_updates[-1][0], 0.09)

    # 核对考核不改变训练随机状态与步数
    def test_rng_and_training_clock_preserved(self):
        random.seed(31)
        np.random.seed(31)
        torch.manual_seed(31)
        py, np_state, cpu = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
        clock = self.runner.env.unwrapped.common_step_counter
        # 模拟考核对三个随机数生成器的消耗
        def sweep(seed):
            random.random()
            np.random.rand()
            torch.rand(20)
            return evaluation_results(0.10)
        self.runner._run_evaluation_sweep = sweep
        self.runner._maybe_evaluate_curriculum(3500)
        self.assertEqual(random.getstate(), py)
        self.assertTrue(np.array_equal(np.random.get_state()[1], np_state[1]))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), cpu))
        self.assertEqual(self.runner.env.unwrapped.common_step_counter, clock)

    # 核对第二批异常时不结算并恢复训练模式
    def test_second_sweep_exception_keeps_course_and_restores_mode(self):
        before = self.runner._wall_course.state_dict()
        modes = []
        self.runner.alg.train_mode = lambda: modes.append("train")
        self.runner._run_evaluation_sweep = lambda seed: (
            evaluation_results(0.10) if seed == 3542 else (_ for _ in ()).throw(RuntimeError("考核中断"))
        )
        with self.assertRaisesRegex(RuntimeError, "考核中断"):
            self.runner._maybe_evaluate_curriculum(3500)
        self.assertEqual(self.runner._wall_course.state_dict(), before)
        self.assertEqual(modes, ["train"])

    # 核对固定命令与固定墙位跳过自动考核
    def test_fixed_command_and_fixed_walls_skip_evaluation(self):
        self.runner._run_evaluation_sweep = lambda seed: self.fail("固定配置不应考核")
        self.runner._command_term().cfg.fixed_time_scale = 3.0
        self.runner._maybe_evaluate_curriculum(3500)
        self.runner._command_term().cfg.fixed_time_scale = None
        self.runner._corridor_fixed = True
        self.runner._maybe_evaluate_curriculum(3500)
        self.assertEqual(self.runner._wall_course.history, [])

    # 核对训练成功率不推进课程
    def test_training_success_cannot_change_course(self):
        command = self.runner._command_term()
        for _ in range(5):
            command.publish([0, 1], [True, True], stood=True)
            self.runner._ingest_episode_results()
        before = self.runner._wall_course.state_dict()
        self.assertFalse(self.runner._maybe_promote(3500))
        self.assertFalse(self.runner._maybe_lower_dmin(3500))
        self.assertEqual(self.runner._wall_course.state_dict(), before)

    # 核对关闭异常时仍保存检查点和关闭日志
    def test_evaluation_close_failure_does_not_skip_checkpoint_and_logger(self):
        calls = []
        self.runner.logger.writer = SimpleNamespace(add_scalar=lambda *args: None)
        self.runner.logger.log_dir = "unused"
        self.runner.logger.stop_logging_writer = lambda: calls.append("stop")
        self.runner.save = lambda *args, **kwargs: calls.append("save")
        self.runner._evaluation_env = SimpleNamespace(
            close=lambda: (_ for _ in ()).throw(RuntimeError("关闭失败")),
        )
        self.runner._maybe_evaluate_curriculum = lambda it: (_ for _ in ()).throw(RuntimeError("考核失败"))
        p1, p2 = self.fixture._patches()
        with p1, p2, self.assertRaisesRegex(RuntimeError, "考核失败"):
            self.runner.learn(1)
        self.assertEqual(calls, ["save", "stop"])
        self.assertIsNone(self.runner._evaluation_env)

    # 核对旧检查点迁移几何下限且不继承训练批次
    def test_old_checkpoint_migrates_floor_without_inheriting_training_batches(self):
        infos = {"corridor_state": {"wall_d_min": 0.05, "curriculum_iter": 3400}}
        with patch("mjlab.rl.runner.MjlabOnPolicyRunner.load", return_value=infos):
            self.runner.load("unused")
        self.assertAlmostEqual(self.runner._wall_course.d_min, 0.06)
        self.assertEqual(self.runner._wall_course.history, [])
        self.assertEqual(self.runner._wall_course.consecutive_success_evaluations, 0)

    # 核对新检查点完整恢复课程
    def test_new_checkpoint_restores_full_state(self):
        course = self.runner._wall_course
        course.record_evaluation(batches(course, (False, True)), 3400)
        saved = course.state_dict()
        infos = {"corridor_state": {"wall_course": saved}}
        self.runner._wall_course = C.WallCurriculumState()
        with patch("mjlab.rl.runner.MjlabOnPolicyRunner.load", return_value=infos):
            self.runner.load("unused")
        self.assertEqual(self.runner._wall_course.state_dict(), saved)

    # 核对每环境只统计首个完整回合并拒绝缺回合
    def test_first_full_episode_per_environment_only(self):
        command = SimpleNamespace(
            _ep_seq=torch.zeros(3, dtype=torch.long),
            _last_ep_righted=torch.zeros(3, dtype=torch.bool),
            _last_ep_wall_d=torch.tensor([0.10, 0.12, 0.20]),
            set_wall_curriculum=lambda *args, **kwargs: None,
        )
        raw = SimpleNamespace(
            common_step_counter=0, num_envs=3, max_episode_length=4, device="cpu",
            episode_length_buf=torch.zeros(3),
            command_manager=SimpleNamespace(get_term=lambda name: command),
        )
        steps = [0]
        # 在不同物理步结束环境回合并发布结果
        def step(actions):
            steps[0] += 1
            dones = torch.zeros(3, dtype=torch.bool)
            ids = [0] if steps[0] == 2 else ([1] if steps[0] == 3 else [0, 2] if steps[0] == 4 else [])
            for index in ids:
                command._ep_seq[index] += 1
                command._last_ep_righted[index] = steps[0] == 2
                dones[index] = True
            return torch.zeros(3, 2), None, dones, {}
        wrapper = SimpleNamespace(
            unwrapped=raw, device="cpu", reset=lambda: (torch.zeros(3, 2), {}),
            step=step,
        )
        self.runner._get_evaluation_env = lambda: wrapper
        self.runner.get_inference_policy = lambda device: (lambda obs: torch.zeros(3, 14))
        results = self.runner._run_evaluation_sweep(12)
        self.assertEqual(len(results), 3)
        self.assertTrue(results[0]["success"])
        self.assertEqual([item["env_id"] for item in results], [0, 1, 2])
        command._ep_seq.zero_()
        steps[0] = -10
        with self.assertRaisesRegex(RuntimeError, "缺少完整回合"):
            self.runner._run_evaluation_sweep(12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
