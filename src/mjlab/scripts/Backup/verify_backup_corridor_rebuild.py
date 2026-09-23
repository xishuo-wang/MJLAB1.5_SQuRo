# 训练循环内受限空间重建回归：不建真环境，用假 env/wrapper 跑一次真实的 learn()。
# 覆盖三个曾在审查中发现失效的环节：重建时机、新环境接管 common_step_counter、观测刷新与统计清零。
import unittest
from unittest.mock import patch

import torch

from mjlab.tasks.SQuRo_Backup.mdp import curriculums as C
from mjlab.tasks.SQuRo_Backup.rl import runner as runner_mod
from mjlab.tasks.SQuRo_Backup.rl.runner import SQuRoBackupOnPolicyRunner


# 观测/重置事件序列，用来断言"重建必须发生在重新取观测之前"
EVENTS: list[str] = []


class FakeEnv:
    # 裸环境：只提供 runner 与包装器真正读到的接口
    def __init__(self, step_count: int = 0, tag: str = "initial"):
        self.device = "cpu"
        self.num_envs = 2
        self.max_episode_length = 1000
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.render_mode = None
        self.tag = tag
        self._state = type("U", (), {
            "common_step_counter": step_count,
            "scene": type("S", (), {"num_envs": 2})(),
            "render_mode": None,
            "cfg": "CFG",
        })()

    @property
    def unwrapped(self):
        return self._state

    @property
    def common_step_counter(self):
        return self._state.common_step_counter

    @common_step_counter.setter
    def common_step_counter(self, value):
        self._state.common_step_counter = value

    def reset(self):
        EVENTS.append(f"reset({self.tag})")
        return torch.zeros(2, 3)

    def get_observations(self):
        EVENTS.append(f"get_obs({self.tag})")
        return torch.full((2, 3), float(len(EVENTS)))

    def step(self, actions):
        self._state.common_step_counter += 1
        return (torch.zeros(2, 3), torch.zeros(2), torch.zeros(2, dtype=torch.bool), {})

    def close(self):
        EVENTS.append(f"close({self.tag})")


class FakeWrapper:
    def __init__(self, env, clip_actions=None):
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = env.num_envs
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self.episode_length_buf = env.episode_length_buf
        env.reset()

    @property
    def unwrapped(self):
        return self.env

    @property
    def common_step_counter(self):
        return self.env.common_step_counter

    def get_observations(self):
        return self.env.get_observations()

    def step(self, actions):
        return self.env.step(actions)

    def close(self):
        return self.env.close()


class FakeAlg:
    learning_rate = 0.001
    intrinsic_rewards = None

    def __init__(self):
        self.obs_seen: list[float] = []

    def train_mode(self):
        pass

    def act(self, obs):
        self.obs_seen.append(float(obs[0, 0].item()))
        return torch.zeros(2, 14)

    def process_env_step(self, *args, **kwargs):
        pass

    def compute_returns(self, obs):
        pass

    def update(self):
        return {}

    def get_policy(self):
        return type("P", (), {"output_std": torch.tensor(0.5)})()

    # 返回加载到的轮次；父类据此设置 current_learning_iteration
    def load(self, loaded_dict, load_cfg=None, strict=True):
        return loaded_dict.get("iter")


class FakeLogger:
    # 字段名必须与 rsl_rl Logger 一致，否则清零逻辑会静默失效
    def __init__(self):
        self.writer = None
        self.log_dir = "."
        self.ep_extras = ["stale"]
        self.cur_reward_sum = torch.full((2,), 13.0)
        self.cur_episode_length = torch.full((2,), 7.0)
        self.cur_ereward_sum = torch.full((2,), 1.0)
        self.cur_ireward_sum = torch.full((2,), 2.0)

    def init_logging_writer(self):
        pass

    def process_env_step(self, *args, **kwargs):
        pass

    def log(self, **kwargs):
        pass

    def stop_logging_writer(self):
        pass


# 重建模板需要 scene.num_envs 与 events 两个接口
def fake_env_cfg():
    return type("Cfg", (), {
        "scene": type("S", (), {"num_envs": 2})(),
        "events": {"init_restricted_space": None, "reset_all": None},
    })()


class CorridorRebuildTest(unittest.TestCase):
    def setUp(self):
        EVENTS.clear()
        self.built: list[FakeEnv] = []

    def _ctor(self, cfg=None, device=None, render_mode=None):
        env = FakeEnv(tag=f"new{len(self.built)}")
        self.built.append(env)
        return env

    def _patches(self):
        return (
            patch.object(runner_mod, "RslRlVecEnvWrapper", FakeWrapper),
            patch.object(runner_mod, "ManagerBasedRlEnv", self._ctor),
            patch.object(runner_mod.mdp_entity, "configure_restricted_space",
                         lambda *a, **k: None),
        )

    # 直接给实例灌属性，绕过真实父类 __init__（不建环境）
    def _build(self, start_iter: int, width: float, collision: bool,
               fixed: bool = False) -> SQuRoBackupOnPolicyRunner:
        r = SQuRoBackupOnPolicyRunner.__new__(SQuRoBackupOnPolicyRunner)
        r._corridor_device = "cpu"
        r._corridor_env_cfg = fake_env_cfg()
        r._corridor_num_envs = 2
        r._corridor_clip_actions = None
        r._corridor_render_mode = None
        r._corridor_width = width
        r._corridor_collision = collision
        r._corridor_fixed = fixed
        r.env = FakeWrapper(FakeEnv(step_count=start_iter * C._STEPS_PER_ITER, tag="old"))
        r.alg = FakeAlg()
        r.logger = FakeLogger()
        r.cfg = {
            "num_steps_per_env": C._STEPS_PER_ITER,
            "save_interval": 10 ** 9,
            "algorithm": {"rnd_cfg": None},
            "check_for_nan": False,
        }
        r.device = "cpu"
        r.is_distributed = False
        r.gpu_global_rank = 0
        r.current_learning_iteration = start_iter
        r.save = lambda *a, **k: None
        return r

    def test_no_rebuild_before_stage_boundary(self):
        r = self._build(2990, 0.40, False)
        p1, p2, p3 = self._patches()
        with p1, p2, p3:
            r.learn(6)
        self.assertEqual(len(self.built), 0)
        self.assertEqual((r._corridor_width, r._corridor_collision), (0.40, False))

    def test_rebuild_opens_collision_across_stage_boundary(self):
        # 训练只调用一次 learn(6000)，阶段切换发生在循环内部 —— 只在入口判断会漏掉
        r = self._build(2990, 0.40, False)
        p1, p2, p3 = self._patches()
        with p1, p2, p3:
            r.learn(12)
        self.assertEqual(len(self.built), 1)
        self.assertIs(r._corridor_collision, True)
        close_at = next(i for i, e in enumerate(EVENTS) if e.startswith("close("))
        self.assertTrue(any(e.startswith("get_obs(new") for e in EVENTS[close_at:]),
                        "重建后必须重新取观测，旧环境的 obs 不能继续用")

    def test_counter_survives_rebuild(self):
        # 新环境从 0 起算；不接管计数器会在下一轮读到阶段一，把刚开的碰撞又关回去
        r = self._build(2990, 0.40, False)
        p1, p2, p3 = self._patches()
        with p1, p2, p3:
            r.learn(12)
            counter = int(r.env.unwrapped.common_step_counter)
        self.assertEqual(len(self.built), 1, "不得反复重建")
        self.assertGreaterEqual(counter, C.STAGE1_3_ITER * C._STEPS_PER_ITER)
        self.assertEqual(C.get_training_phase(counter), 1)

    def test_rebuild_follows_width_ladder(self):
        # 0.35 -> 0.30 的切档点实测在 iter 3750 (相邻档中点)
        r = self._build(3749, 0.35, True)
        p1, p2, p3 = self._patches()
        with p1, p2, p3:
            r.learn(2)
        self.assertEqual(len(self.built), 1)
        self.assertAlmostEqual(r._corridor_width,
                               C.get_corridor_width_for_iter(3750), places=12)
        self.assertAlmostEqual(r._corridor_width, C.CORRIDOR_WIDTH_LADDER[2], places=12)

    def test_logger_accumulators_cleared_on_rebuild(self):
        # 重建会中断所有在跑的回合，不清零就会跨重建拼接统计
        r = self._build(2999, 0.40, False)
        p1, p2, p3 = self._patches()
        with p1, p2, p3:
            r.learn(2)
        self.assertEqual(r.logger.cur_reward_sum[0].item(), 0.0)
        self.assertEqual(r.logger.cur_episode_length[0].item(), 0.0)
        self.assertEqual(r.logger.ep_extras, [])

    def test_fixed_width_still_switches_collision(self):
        # 锁死宽度时宽度判断走不到，碰撞判断必须放在 fixed 分支之外
        r = self._build(2999, 0.30, False, fixed=True)
        p1, p2, p3 = self._patches()
        with p1, p2, p3:
            r.learn(2)
        self.assertEqual(len(self.built), 1)
        self.assertIs(r._corridor_collision, True)
        self.assertAlmostEqual(r._corridor_width, 0.30, places=12)

    def test_fixed_width_single_rebuild_per_boundary(self):
        # 宽度锁死时重建后不应再因宽度差反复重建
        r = self._build(2999, 0.30, False, fixed=True)
        p1, p2, p3 = self._patches()
        with p1, p2, p3:
            r.learn(24)
        self.assertEqual(len(self.built), 1)

    # 续训：检查点记录的编译期状态要能盖过启动配置，并保证首采前环境与记录一致
    def test_load_restores_recorded_state(self):
        r = self._build(0, 0.40, False)
        saved_iter = 4000
        saved_width = C.get_corridor_width_for_iter(saved_iter)
        p1, p2, p3 = self._patches()
        with p1, p2, p3, self._fake_checkpoint(saved_iter, saved_width, True):
            infos = r.load("model_x.pt")
        # 记录与按轮次推算一致 ⇒ 无需重建，但状态必须已被记录覆盖
        self.assertEqual(len(self.built), 0)
        self.assertAlmostEqual(r._corridor_width, saved_width, places=12)
        self.assertIs(r._corridor_collision, True)
        self.assertEqual(infos["corridor_state"]["corridor_collision"], True)
        self.assertGreaterEqual(int(r.env.unwrapped.common_step_counter),
                                C.STAGE1_3_ITER * C._STEPS_PER_ITER)

    def test_load_rebuilds_when_record_disagrees_with_iteration(self):
        # 记录自相矛盾时（轮次在阶段一却记着开碰撞，例如手动改过 tag 的检查点），
        # 以**轮次推算**为准，并在首次采样前重建过来。
        r = self._build(0, 0.40, True)
        p1, p2, p3 = self._patches()
        with p1, p2, p3, self._fake_checkpoint(2999, 0.40, True):
            r.load("model_x.pt")
        self.assertEqual(len(self.built), 1)
        self.assertIs(r._corridor_collision, False)
        self.assertEqual(C.get_training_phase(int(r.env.unwrapped.common_step_counter)), 0)

    def test_load_without_record_falls_back_to_iteration(self):
        # 老检查点没有 corridor_state：按恢复后的轮次推算，而不是留在启动配置上
        r = self._build(0, 0.40, False)
        p1, p2, p3 = self._patches()
        with p1, p2, p3, self._fake_checkpoint(4000, None, None):
            r.load("model_x.pt")
        self.assertEqual(len(self.built), 1)
        self.assertIs(r._corridor_collision, True)
        self.assertAlmostEqual(r._corridor_width,
                               C.get_corridor_width_for_iter(4000), places=12)

    def test_load_matching_state_does_not_rebuild(self):
        # 检查点状态与当前编译配置一致时不要白重建一次
        width = C.get_corridor_width_for_iter(4000)
        r = self._build(0, width, True)
        p1, p2, p3 = self._patches()
        with p1, p2, p3, self._fake_checkpoint(4000, width, True):
            r.load("model_x.pt")
        self.assertEqual(len(self.built), 0)

    # 伪造检查点：torch.load 返回的最小可用结构（父类只读这几个键）
    def _fake_checkpoint(self, it: int, width, collision):
        infos = {
            "env_state": {"common_step_counter": it * C._STEPS_PER_ITER},
            "corridor_state": {
                "corridor_width": width,
                "corridor_collision": collision,
                "corridor_fixed": False,
            },
        }
        payload = {"iter": it, "infos": infos, "model_state_dict": {}}
        return patch.object(runner_mod.torch, "load", lambda *a, **k: payload)


if __name__ == '__main__':
    unittest.main(verbosity=2)
