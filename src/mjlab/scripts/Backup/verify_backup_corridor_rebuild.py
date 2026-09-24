# 训练循环内受限空间重建回归：不建真环境，用假 env/wrapper 跑一次真实的 learn()。
# **关键**：假环境里放的是真的 RestrictedSpaceEntityCfg，所以断言读的是"实际编译进仿真的
# 配置"，而不是 runner 自己缓存的变量 —— 只查 runner 变量会把"变量对、环境错"测成通过。
import unittest
from unittest.mock import patch

import torch

from mjlab.tasks.SQuRo_Backup.mdp import curriculums as C
from mjlab.tasks.SQuRo_Backup.mdp import entity as E
from mjlab.tasks.SQuRo_Backup.rl import runner as runner_mod
from mjlab.tasks.SQuRo_Backup.rl.runner import SQuRoBackupOnPolicyRunner


# 观测/重置事件序列，用来断言"重建必须发生在重新取观测之前"
EVENTS: list[str] = []


def compiled_state(env) -> tuple[float, bool]:
    # 读"实际编译值"的唯一合法途径：场景实体本身 (宽度取自 cfg, 碰撞取自 contype)
    ent = env.unwrapped.scene.entities["restricted_space"]
    return float(ent.cfg.corridor_width), bool(ent.collision_enabled)


class FakeEnv:
    # 裸环境：只提供 runner 与包装器真正读到的接口
    def __init__(self, step_count: int = 0, tag: str = "initial",
                 width: float = E.DEFAULT_CORRIDOR_WIDTH, collision: bool = False,
                 fixed: bool = False):
        self.device = "cpu"
        self.num_envs = 2
        self.max_episode_length = 1000
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.render_mode = None
        self.tag = tag
        # 真的实体：用生产代码建, 所以断言读到的就是"编译进仿真的配置"
        cfg = E.build_restricted_space_cfg(enable_collision=collision,
                                           corridor_width=width, fixed_width=fixed)
        entity = cfg.build()
        self._state = type("U", (), {
            "common_step_counter": step_count,
            "scene": type("S", (), {
                "num_envs": 2,
                "entities": {"restricted_space": entity},
            })(),
            "render_mode": None,
            "cfg": "CFG",
        })()

    @property
    def unwrapped(self):
        return self._state

    @property
    def scene(self):
        return self._state.scene

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
        env.reset()

    # 必须像真包装器一样用 property 转发到裸环境, 否则写入只落在假包装器上,
    # "重建后回合计时是否被打散"这类断言会读到与生产代码不同的对象。
    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    @property
    def unwrapped(self):
        return self.env

    @property
    def scene(self):
        return self.env.scene

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


# 重建模板：需要 scene.num_envs、scene.entities 与 events 三个接口
def fake_env_cfg():
    return type("Cfg", (), {
        "scene": type("S", (), {
            "num_envs": 2,
            "entities": {"restricted_space": E.build_restricted_space_cfg(True, 0.40)},
        })(),
        "events": {"init_restricted_space": None, "reset_all": None},
        "seed": 42,
    })()


class CorridorRebuildTest(unittest.TestCase):
    def setUp(self):
        EVENTS.clear()
        self.built: list[FakeEnv] = []
        self.seen_seed: list = []

    def _ctor(self, cfg=None, device=None, render_mode=None):
        # 模拟真实编译：按 configure_restricted_space 写进 env_cfg 的值建实体
        self.seen_seed.append(getattr(cfg, "seed", "MISSING"))
        ent_cfg = cfg.scene.entities["restricted_space"]
        env = FakeEnv(tag=f"new{len(self.built)}",
                      width=float(ent_cfg.corridor_width),
                      collision=bool(ent_cfg.contype > 0),
                      fixed=bool(ent_cfg.fixed_width))
        self.built.append(env)
        return env

    def _patches(self):
        # 真的 configure_restricted_space：让假重建路径与生产代码写的是同一份 env_cfg
        return (
            patch.object(runner_mod, "RslRlVecEnvWrapper", FakeWrapper),
            patch.object(runner_mod, "ManagerBasedRlEnv", self._ctor),
        )

    # 直接给实例灌属性，绕过真实父类 __init__（不建环境）
    def _build(self, start_iter: int, width: float, collision: bool,
               fixed: bool = False, fixed_by_cli: bool | None = None
               ) -> SQuRoBackupOnPolicyRunner:
        r = SQuRoBackupOnPolicyRunner.__new__(SQuRoBackupOnPolicyRunner)
        r._corridor_device = "cpu"
        r._corridor_env_cfg = fake_env_cfg()
        r._corridor_num_envs = 2
        r._corridor_clip_actions = None
        r._corridor_render_mode = None
        r._corridor_start_width = width
        r._corridor_fixed = fixed
        r._corridor_fixed_by_cli = fixed if fixed_by_cli is None else fixed_by_cli
        r._randomize_ep_len = False
        # 启动环境按 width/collision 编译 (与 env_cfg 初始配置同源)
        r.env = FakeWrapper(FakeEnv(step_count=start_iter * C._STEPS_PER_ITER,
                                    tag="old", width=width, collision=collision,
                                    fixed=fixed))
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
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(6)
        self.assertEqual(len(self.built), 0)
        self.assertEqual(compiled_state(r.env), (0.40, False))

    def test_rebuild_opens_collision_across_stage_boundary(self):
        # 训练只调用一次 learn(6000)，阶段切换发生在循环内部 —— 只在入口判断会漏掉
        r = self._build(2990, 0.40, False)
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(12)
        self.assertEqual(len(self.built), 1)
        self.assertEqual(compiled_state(r.env), (0.40, True))
        close_at = next(i for i, e in enumerate(EVENTS) if e.startswith("close("))
        self.assertTrue(any(e.startswith("get_obs(new") for e in EVENTS[close_at:]),
                        "重建后必须重新取观测，旧环境的 obs 不能继续用")

    def test_counter_survives_rebuild(self):
        # 新环境从 0 起算；不接管计数器会在下一轮读到阶段一，把刚开的碰撞又关回去
        r = self._build(2990, 0.40, False)
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(12)
            counter = int(r.env.unwrapped.common_step_counter)
        self.assertEqual(len(self.built), 1, "不得反复重建")
        self.assertGreaterEqual(counter, C.STAGE1_3_ITER * C._STEPS_PER_ITER)
        self.assertEqual(C.get_training_phase(counter), 1)

    def test_rebuild_follows_width_ladder(self):
        # 0.35 -> 0.30 的切档点实测在 iter 3750 (相邻档中点)
        r = self._build(3749, 0.35, True)
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(2)
        self.assertEqual(len(self.built), 1)
        expected = C.get_corridor_width_for_iter(3750)
        self.assertAlmostEqual(expected, C.CORRIDOR_WIDTH_LADDER[2], places=12)
        self.assertAlmostEqual(compiled_state(r.env)[0], expected, places=12)

    def test_logger_accumulators_cleared_on_rebuild(self):
        # 重建会中断所有在跑的回合，不清零就会跨重建拼接统计
        r = self._build(2999, 0.40, False)
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(2)
        self.assertEqual(r.logger.cur_reward_sum[0].item(), 0.0)
        self.assertEqual(r.logger.cur_episode_length[0].item(), 0.0)
        self.assertEqual(r.logger.ep_extras, [])

    # 新环境的 episode_length_buf 全为 0；本任务只有 timeout 一种终止，不打散回合计时就会让
    # 全部环境永久同进同出（每轮 rollout 只覆盖一个任务相位，全环境瞬时平均指标随之失真）。
    def test_rebuild_randomizes_episode_phase(self):
        torch.manual_seed(7)                     # 固定 RNG，避免 2 个环境随机撞成同值
        r = self._build(2999, 0.40, False)
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(2, init_at_random_ep_len=True)
        self.assertEqual(len(self.built), 1)
        buf = r.env.unwrapped.episode_length_buf
        self.assertGreater(int(buf.sum()), 0, "重建后必须重新随机化回合计时, 不能停在全体 0")
        self.assertNotEqual(int(buf[0]), int(buf[1]), "各环境的回合相位必须被打散")

    # 开关关闭时必须保持调用方要求的固定回合长度：重建不得擅自改变回合计时。
    # 标准训练入口传 True, 因此这一条只影响固定时长的诊断/消融入口。
    def test_rebuild_respects_disabled_episode_randomization(self):
        torch.manual_seed(7)
        r = self._build(2999, 0.40, False)
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(2)                           # init_at_random_ep_len 默认 False
        self.assertEqual(len(self.built), 1)
        buf = r.env.unwrapped.episode_length_buf
        self.assertEqual(int(buf.sum()), 0, "开关关闭时重建不得随机化回合计时")
        self.assertFalse(r._randomize_ep_len)

    # 重建模板只能改课程控制的两个量；直接新建默认 cfg 会把自定义墙体尺寸悄悄改回默认值
    # （实测墙高 0.25/半长 1.0/半厚 0.03 被恢复成 0.10/0.30/0.01）。
    def test_rebuild_preserves_custom_wall_dimensions(self):
        custom = E.RestrictedSpaceEntityCfg(
            name="restricted_space", corridor_width=0.40,
            wall_height=0.25, wall_half_length=1.0, wall_half_thickness=0.03)
        cfg = type("Cfg", (), {
            "scene": type("S", (), {"entities": {"restricted_space": custom}})(),
            "events": {},
        })()
        out = E.configure_restricted_space(cfg, 0.30, enable_collision=True)
        self.assertAlmostEqual(out.corridor_width, 0.30, places=12)
        self.assertEqual(out.contype, 1, "课程仍必须能改碰撞开关")
        self.assertEqual(out.conaffinity, 1)
        self.assertEqual(
            (out.wall_height, out.wall_half_length, out.wall_half_thickness),
            (0.25, 1.0, 0.03), "自定义墙体尺寸不得被重置为默认值")

    # 重建不是新实验的开始: 不能按 env_cfg.seed 重新播种 —— ManagerBasedRlEnv.__init__ 会调
    # seed_rng -> torch.manual_seed (全设备) + random/np/wp, 把"换墙距"和"复位所有随机流"绑在一起。
    def test_rebuild_does_not_reseed_rng(self):
        r = self._build(2999, 0.40, False)
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(2, init_at_random_ep_len=True)
        self.assertEqual(len(self.built), 1)
        self.assertEqual(self.seen_seed, [None],
                         "重建模板必须清掉 seed, 否则新环境会重新播种全局随机流")

    def test_fixed_width_still_switches_collision(self):
        # 锁死宽度时宽度判断走不到，碰撞判断必须放在 fixed 分支之外
        r = self._build(2999, 0.30, False, fixed=True)
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(2)
        self.assertEqual(len(self.built), 1)
        self.assertEqual(compiled_state(r.env), (0.30, True))

    def test_fixed_width_single_rebuild_per_boundary(self):
        # 宽度锁死时重建后不应再因宽度差反复重建
        r = self._build(2999, 0.30, False, fixed=True)
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(24)
        self.assertEqual(len(self.built), 1)

    # 审查场景：记录与课程目标一致、但环境里编译的不是这个值 ⇒ 必须按**实际编译值**判定重建
    def test_load_rebuilds_even_when_record_matches_target(self):
        saved_iter = 3750
        width = C.get_corridor_width_for_iter(saved_iter)
        self.assertAlmostEqual(width, 0.30, places=12)
        # 启动环境编译成 0.40；检查点记录 0.30 ⇒ 记录与目标一致，但环境是 0.40
        r = self._build(0, 0.40, True)
        p1, p2 = self._patches()
        with p1, p2, self._fake_checkpoint(saved_iter, width, True):
            infos = r.load("model_x.pt")
        cw, cc = compiled_state(r.env)
        self.assertAlmostEqual(cw, width, places=12,
                               msg="续训后实际编译的墙宽必须是记录值, 不是启动时的 0.40")
        self.assertIs(cc, True)
        self.assertAlmostEqual(float(infos["corridor_state"]["corridor_width"]),
                               width, places=12)

    def test_load_without_record_falls_back_to_iteration(self):
        # 老检查点没有 corridor_state：按恢复后的轮次推算，而不是留在启动配置上
        r = self._build(0, 0.40, False)
        p1, p2 = self._patches()
        with p1, p2, self._fake_checkpoint(4000, None, None):
            r.load("model_x.pt")
        self.assertEqual(compiled_state(r.env),
                         (C.get_corridor_width_for_iter(4000), True))

    def test_load_matching_state_does_not_rebuild(self):
        # 环境本来就编译成了目标值 ⇒ 不要白重建一次
        width = C.get_corridor_width_for_iter(4000)
        r = self._build(0, width, True)
        p1, p2 = self._patches()
        with p1, p2, self._fake_checkpoint(4000, width, True):
            r.load("model_x.pt")
        self.assertEqual(len(self.built), 0)
        self.assertEqual(compiled_state(r.env), (width, True))

    def test_load_actor_only_keeps_compiled_env(self):
        # 回放加载 (load_cfg={"actor": True}): 回放入口已按记录把墙编译好了,
        # 这里重建会把查看器手里的 env 换掉 (查看器不接管 runner.env), 必须不重建。
        r = self._build(2999, 0.40, False)
        p1, p2 = self._patches()
        with p1, p2, self._fake_checkpoint(4000, 0.30, True):
            r.load("model_x.pt", load_cfg={"actor": True})
        self.assertEqual(len(self.built), 0, "回放加载不得重建环境")
        cw, cc = compiled_state(r.env)
        self.assertAlmostEqual(cw, 0.40, places=12)
        self.assertIs(cc, False, "回放必须保留入口已编译好的无碰撞配置")
        self.assertEqual(EVENTS.count("close(old)"), 0, "回放加载不得关闭入口的环境")

    # 锁死宽度续训：记录里的标记要生效, 且模板宽度同步成记录值
    def test_load_restores_fixed_width_mode(self):
        saved_iter, saved_width = 4000, 0.30
        r = self._build(0, 0.40, True)          # 启动: 非锁死, 0.40
        p1, p2 = self._patches()
        with p1, p2, self._fake_checkpoint(saved_iter, saved_width, True, fixed=True):
            r.load("model_x.pt")
        cw, cc = compiled_state(r.env)
        self.assertAlmostEqual(cw, saved_width, places=12,
                               msg="锁死模式下应按记录的宽度重建, 而不是启动宽度")
        self.assertIs(cc, True)

    # 命令行显式给了宽度就不该被记录里的锁死标记带跑
    def test_cli_fixed_width_wins_over_record(self):
        r = self._build(0, 0.25, False, fixed=True, fixed_by_cli=True)
        p1, p2 = self._patches()
        # 记录说"非锁死、4000 轮的档位", 但本次命令行锁死 0.25 ⇒ 目标仍是 0.25
        with p1, p2, self._fake_checkpoint(4000, 0.35, True, fixed=False):
            r.load("model_x.pt")
        cw, _ = compiled_state(r.env)
        self.assertAlmostEqual(cw, 0.25, places=12,
                               msg="命令行锁死宽度优先于检查点记录的锁死标记")
        self.assertTrue(r._corridor_fixed)

    # 伪造检查点：torch.load 返回的最小可用结构（父类只读这几个键）
    def _fake_checkpoint(self, it: int, width, collision, fixed: bool = False):
        state = {"corridor_width": width, "corridor_collision": collision,
                 "corridor_fixed": fixed}
        infos = {
            "env_state": {"common_step_counter": it * C._STEPS_PER_ITER},
            "corridor_state": state,
        }
        payload = {"iter": it, "infos": infos, "model_state_dict": {}}
        return patch.object(runner_mod.torch, "load", lambda *a, **k: payload)


if __name__ == '__main__':
    unittest.main(verbosity=2)
