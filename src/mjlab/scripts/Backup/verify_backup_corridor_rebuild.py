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


def compiled_state(env) -> tuple[tuple[float, float], bool]:
    # 读"实际编译值"的唯一合法途径：场景实体本身 (墙位取自 cfg, 碰撞取自 contype)
    ent = env.unwrapped.scene.entities["restricted_space"]
    return (float(ent.cfg.wall_x_neg), float(ent.cfg.wall_x_pos)), bool(ent.collision_enabled)


class FakeCommand:
    # 只提供 runner 读的回合发布接口 (课程门控的输入); 真 command 的其余部分与本回归无关
    def __init__(self, n: int = 2):
        self._ep_seq = torch.zeros(n, dtype=torch.long)
        self._last_ep_valid = torch.zeros(n, dtype=torch.bool)
        self._last_ep_success = torch.zeros(n, dtype=torch.bool)
        self._last_ep_stood_pose = torch.zeros(n, dtype=torch.bool)
        self._last_ep_stood_onset = torch.zeros(n, dtype=torch.bool)

    # 模拟"这些环境又结束了一个回合"; valid=False 表示被重建截短/初始化等无效回合。
    # stood/onset 默认跟随 success, 需要区分 (门控读 stood) 时显式传。
    def publish(self, env_ids, success, valid: bool = True, stood=None, onset=None) -> None:
        for i, ok in zip(env_ids, success):
            self._last_ep_valid[i] = bool(valid)
            self._last_ep_success[i] = bool(ok) and bool(valid)
            self._last_ep_stood_pose[i] = (bool(ok) if stood is None else bool(stood)) and bool(valid)
            self._last_ep_stood_onset[i] = (bool(ok) if onset is None else bool(onset)) and bool(valid)
            self._ep_seq[i] += 1


class FakeEnv:
    # 裸环境：只提供 runner 与包装器真正读到的接口
    def __init__(self, step_count: int = 0, tag: str = "initial",
                 walls: tuple[float, float] = (-0.20, 0.08), collision: bool = False,
                 fixed: bool = False):
        self.device = "cpu"
        self.num_envs = 2
        self.max_episode_length = 1000
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.render_mode = None
        self.tag = tag
        self.command = FakeCommand(2)
        # 真的实体：用生产代码建, 所以断言读到的就是"编译进仿真的配置"
        cfg = E.build_restricted_space_cfg(enable_collision=collision,
                                           wall_x_neg=walls[0], wall_x_pos=walls[1],
                                           fixed_width=fixed)
        entity = cfg.build()
        self._state = type("U", (), {
            "common_step_counter": step_count,
            "scene": type("S", (), {
                "num_envs": 2,
                "entities": {"restricted_space": entity},
            })(),
            "command_manager": type("CM", (), {
                "get_term": lambda _self, _name: self.command,
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
    def command_manager(self):
        return self._state.command_manager

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
                      walls=(float(ent_cfg.wall_x_neg), float(ent_cfg.wall_x_pos)),
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
    def _build(self, start_iter: int, walls: tuple[float, float], collision: bool,
               fixed: bool = False, fixed_by_cli: bool | None = None, level: int = 0,
               ) -> SQuRoBackupOnPolicyRunner:
        r = SQuRoBackupOnPolicyRunner.__new__(SQuRoBackupOnPolicyRunner)
        r._corridor_device = "cpu"
        r._corridor_env_cfg = fake_env_cfg()
        r._corridor_num_envs = 2
        r._corridor_clip_actions = None
        r._corridor_render_mode = None
        r._corridor_start_walls = walls
        r._corridor_fixed = fixed
        r._corridor_fixed_by_cli = fixed if fixed_by_cli is None else fixed_by_cli
        r._randomize_ep_len = False
        r._cur_level = level
        r._cur_level_iter = start_iter
        r._cur_level_history = []
        r._reset_curriculum_window()
        # 启动环境按 walls/collision 编译 (与 env_cfg 初始配置同源)
        r.env = FakeWrapper(FakeEnv(step_count=start_iter * C._STEPS_PER_ITER,
                                    tag="old", walls=walls, collision=collision,
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

    # 课程第 0 档的墙位 (启动配置), 用它建"课程 run"的初始环境
    def _level0(self) -> tuple[float, float]:
        return C.get_wall_positions_for_level(0)

    def test_no_rebuild_before_stage_boundary(self):
        r = self._build(2990, self._level0(), False)
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(6)
        self.assertEqual(len(self.built), 0)
        self.assertEqual(compiled_state(r.env), (self._level0(), False))

    def test_rebuild_opens_collision_across_stage_boundary(self):
        # 训练只调用一次 learn(6000)，阶段切换发生在循环内部 —— 只在入口判断会漏掉
        r = self._build(2990, self._level0(), False)
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(12)
        self.assertEqual(len(self.built), 1)
        self.assertEqual(compiled_state(r.env), (self._level0(), True))
        close_at = next(i for i, e in enumerate(EVENTS) if e.startswith("close("))
        self.assertTrue(any(e.startswith("get_obs(new") for e in EVENTS[close_at:]),
                        "重建后必须重新取观测，旧环境的 obs 不能继续用")

    def test_counter_survives_rebuild(self):
        # 新环境从 0 起算；不接管计数器会在下一轮读到阶段一，把刚开的碰撞又关回去
        r = self._build(2990, self._level0(), False)
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(12)
            counter = int(r.env.unwrapped.common_step_counter)
        self.assertEqual(len(self.built), 1, "不得反复重建")
        self.assertGreaterEqual(counter, C.STAGE1_3_ITER * C._STEPS_PER_ITER)
        self.assertEqual(C.get_training_phase(counter), 1)

    # 课程档位 → 编译墙位: 档位变了就必须重建，且编译值等于该档位的墙位
    def test_rebuild_follows_curriculum_level(self):
        r = self._build(3749, self._level0(), True)
        r._cur_level = 2
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(2)
        self.assertEqual(len(self.built), 1)
        self.assertEqual(compiled_state(r.env), (C.get_wall_positions_for_level(2), True))
        self.assertAlmostEqual(C.get_wall_positions_for_level(2)[1], C.WALL_X_POS, places=12)

    # 门控三要素: 最短驻留、窗口填满、p_stood 达标 —— 缺一不可 (窗口只装有效回合)
    def test_promotion_requires_dwell_window_and_gate(self):
        r = self._build(3000, self._level0(), True)
        n_slot = 2 * C.CURRICULUM_WINDOW_EPISODES      # 2 个环境 × 窗口长度

        def fill(n_stood_slots, level_iter):
            r._cur_level_iter = level_iter
            r._w_fill[:] = C.CURRICULUM_WINDOW_EPISODES
            r._w_stood[:] = False
            if n_stood_slots:
                r._w_stood[:, :n_stood_slots] = True

        # 驻留不足
        fill(n_slot, 3000)
        self.assertFalse(r._maybe_promote(3000 + C.CURRICULUM_MIN_DWELL_ITER - 1))
        self.assertEqual(r._cur_level, 0)
        it = 3000 + C.CURRICULUM_MIN_DWELL_ITER
        # 窗口未填满
        fill(n_slot, 3000)
        r._w_fill[:] = C.CURRICULUM_WINDOW_EPISODES - 1
        self.assertFalse(r._maybe_promote(it))
        # p_stood 不达标: 2/6 = 0.33 < 0.60
        fill(1, 3000)
        self.assertFalse(r._maybe_promote(it))
        # 三要素齐备 -> 推进一档, 且窗口被清空 (旧墙位的成绩不得用于新墙位)
        # 4/6 = 0.67 >= 0.60
        fill(2, 3000)
        self.assertTrue(r._maybe_promote(it))
        self.assertEqual(r._cur_level, 1)
        self.assertEqual(r._cur_level_iter, it)
        self.assertEqual(int(r._w_fill.sum()), 0)
        self.assertEqual(r._cur_level_history[-1]["level"], 0)
        self.assertAlmostEqual(r._cur_level_history[-1]["p_stood"], 4 / 6, places=6)

    # 本批的核心语义: 门控读的是**站起来率** (不含速度), 而不是稳定站立成功率。
    # 实测依据: 速度项在开墙前后完全相同 (5.638 vs 5.641 rad/s), 与墙位无关 —— 拿它当门控
    # 会让课程等一个自己影响不了的条件 (2026-09-25 run 的 p_done 全程 0、5100 轮没动一档)。
    def test_gate_uses_stood_rate_not_success_rate(self):
        r = self._build(3000, self._level0(), True)
        cmd = r.env.unwrapped.command_manager.get_term("backup_cmd")
        r._cur_level_iter = 3000
        # 每个回合都"站起来了"但从未"稳定站立成功" -> 必须推进
        for _ in range(C.CURRICULUM_WINDOW_EPISODES):
            cmd.publish([0, 1], [False, False], stood=True)
            r._ingest_episode_results()
        p_stood, p_onset, p_done, ready, n_valid = r._curriculum_metrics()
        self.assertTrue(ready)
        self.assertAlmostEqual(p_stood, 1.0, places=6)
        self.assertAlmostEqual(p_done, 0.0, places=6)
        self.assertTrue(r._maybe_promote(3100), "站起来率达标就必须推进, 不受成功率为 0 阻挡")
        self.assertEqual(r._cur_level, 1)

    def test_gate_blocked_when_never_stood_up(self):
        r = self._build(3000, self._level0(), True)
        cmd = r.env.unwrapped.command_manager.get_term("backup_cmd")
        r._cur_level_iter = 3000
        for _ in range(C.CURRICULUM_WINDOW_EPISODES):
            cmd.publish([0, 1], [False, False], stood=False, onset=True)   # 只有窗口起始
            r._ingest_episode_results()
        p_stood, p_onset, p_done, ready, n_valid = r._curriculum_metrics()
        self.assertTrue(ready)
        self.assertAlmostEqual(p_stood, 0.0, places=6)
        self.assertAlmostEqual(p_onset, 1.0, places=6)
        self.assertFalse(r._maybe_promote(3100),
                         "只有'窗口起始'不足以推进 —— 单帧条件可能被翻滚瞬时满足")

    def test_promotion_blocked_before_start_iter_and_in_fixed_mode(self):
        # 计数器与轮次必须一致: 物理对齐守卫会比 common_step_counter 推出的阶段
        r = self._build(C.CURRICULUM_START_ITER, self._level0(), True)
        r._cur_level_iter = 0
        r._w_stood[:] = True
        r._w_fill[:] = C.CURRICULUM_WINDOW_EPISODES
        self.assertFalse(r._maybe_promote(C.CURRICULUM_START_ITER - 1), "起始轮数之前不得推进")
        self.assertTrue(r._maybe_promote(C.CURRICULUM_START_ITER))
        # 锁死模式 (诊断/消融) 永不推进
        r2 = self._build(5000, (-0.125, 0.125), True, fixed=True)
        r2._cur_level_iter = 0
        r2._w_stood[:] = True
        r2._w_fill[:] = C.CURRICULUM_WINDOW_EPISODES
        self.assertFalse(r2._maybe_promote(5000))

    # 新环境的回合发布序号从 0 重新开始; 不清 _ep_seq_seen 就再也收不到事件, 门控永久饿死
    def test_rebuild_resets_curriculum_window(self):
        r = self._build(2999, self._level0(), False)
        r._ep_seq_seen[:] = 99
        r._w_fill[:] = 3
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(2)
        self.assertEqual(len(self.built), 1)
        self.assertEqual(int(r._ep_seq_seen.sum()), 0, "重建后必须重置发布序号基线")
        self.assertEqual(int(r._w_fill.sum()), 0, "重建后必须清空门控窗口")

    # 课程标量写 tensorboard: 无样本时不写三个率 (NaN 会污染曲线), 其余键恒写
    def test_log_curriculum_writes_scalars(self):
        class RecWriter:
            def __init__(self):
                self.calls: dict[str, float] = {}

            def add_scalar(self, tag, value, step):
                self.calls[tag] = value

        r = self._build(3000, self._level0(), True)
        w = RecWriter()
        r.logger.writer = w
        r._log_curriculum(3000)
        self.assertEqual(w.calls["Curriculum/level"], 0)
        self.assertAlmostEqual(w.calls["Curriculum/wall_x_neg"], self._level0()[0], places=12)
        self.assertAlmostEqual(w.calls["Curriculum/wall_x_pos"], self._level0()[1], places=12)
        self.assertEqual(w.calls["Curriculum/window_ready"], 0.0)
        for key in ("Curriculum/p_stood", "Curriculum/p_onset", "Curriculum/p_done"):
            self.assertNotIn(key, w.calls, f"无样本时不得写 {key} (NaN 会污染曲线)")
        # 有样本后三个率才出现; 三者共用窗口, 所以分母相同
        cmd = r.env.unwrapped.command_manager.get_term("backup_cmd")
        cmd.publish([0, 1], [True, False], stood=True, onset=True)
        r._ingest_episode_results()
        r._log_curriculum(3001)
        self.assertAlmostEqual(w.calls["Curriculum/p_done"], 0.5, places=6)
        self.assertAlmostEqual(w.calls["Curriculum/p_stood"], 1.0, places=6)
        self.assertAlmostEqual(w.calls["Curriculum/p_onset"], 1.0, places=6)
        self.assertEqual(w.calls["Curriculum/n_valid_episodes"], 2)
        # 没有 writer 时不得报错 (回放/诊断入口)
        r.logger.writer = None
        r._log_curriculum(3002)

    # 开墙那一刻不得凭"无碰撞成绩"缩档 (审查 P1): 必须先以第 0 档 + 开碰撞重建, 清空旧成绩,
    # 并把驻留起点重置到开墙那一刻, 之后才允许按能力推进。
    def test_collision_switch_does_not_promote_on_collision_free_record(self):
        r = self._build(2999, self._level0(), False)
        # 无碰撞期间窗口已攒满且完成率 100%
        r._w_fill[:] = C.CURRICULUM_WINDOW_EPISODES
        r._w_stood[:] = True
        r._cur_level_iter = 0
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(4)
        self.assertEqual(r._cur_level, 0, "开墙那一刻不得升级 (旧成绩来自无碰撞物理)")
        self.assertEqual(compiled_state(r.env), (self._level0(), True),
                         "必须先以第 0 档 + 开碰撞重建")
        self.assertEqual(len(self.built), 1)
        self.assertGreaterEqual(r._cur_level_iter, C.CURRICULUM_START_ITER,
                                "驻留起点必须重置到开墙那一刻")
        self.assertEqual(int(r._w_fill.sum()), 0, "开墙必须清空旧物理下的成绩")

    # 无效回合不得占用"有效回合"窗口槽位 (审查 P2): 否则 1 无效 + 2 有效就会被判为窗口已满
    def test_invalid_episode_does_not_fill_window(self):
        r = self._build(3000, self._level0(), True)
        cmd = r.env.unwrapped.command_manager.get_term("backup_cmd")
        cmd.publish([0, 1], [True, True], valid=False)
        r._ingest_episode_results()
        self.assertEqual(int(r._w_fill.sum()), 0, "无效回合不得占用窗口槽位")
        self.assertEqual(int(r._ep_seq_seen[0]), 1, "无效回合仍必须推进已消费序号")
        for _ in range(C.CURRICULUM_WINDOW_EPISODES):
            cmd.publish([0, 1], [True, False])
            r._ingest_episode_results()
        self.assertEqual(int(r._w_fill[0]), C.CURRICULUM_WINDOW_EPISODES)
        p_stood, p_onset, p_done, ready, n_valid = r._curriculum_metrics()
        self.assertTrue(ready)
        self.assertEqual(n_valid, 2 * C.CURRICULUM_WINDOW_EPISODES,
                         "分母只能是有效回合数")

    # 物理环境尚未与当前档位一致时不得推进 (不变量: 只在"已编译配置 == 目标配置"时推进)
    def test_promotion_requires_physics_to_match_level(self):
        r = self._build(4000, self._level0(), False)     # 目标要开碰撞, 实际关着
        r._w_fill[:] = C.CURRICULUM_WINDOW_EPISODES
        r._w_stood[:] = True
        r._cur_level_iter = 0
        self.assertFalse(r._maybe_promote(4000), "物理未对齐时不得推进")

    # 升级必须**当轮**生效: 判门控之后立刻重建, 否则本轮采样仍跑在旧墙位上, 而
    # Curriculum/level 已经记成新档位 —— 档位与实际物理错开一轮 (审查: "升级晚一轮生效")。
    def test_promotion_takes_effect_in_the_same_iteration(self):
        r = self._build(3100, self._level0(), True)
        r._cur_level_iter = 3000                       # 驻留已满
        r._w_fill[:] = C.CURRICULUM_WINDOW_EPISODES
        r._w_stood[:] = True
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(1)                                 # 只跑 3100 这一轮
        self.assertEqual(r._cur_level, 1, "本轮应已升级")
        self.assertEqual(len(self.built), 1, "升级当轮就必须重建到新档位")
        self.assertEqual(compiled_state(r.env),
                         (C.get_wall_positions_for_level(1), True),
                         "升级当轮编译值必须已是新档位的墙位")

    # 更强的一条: 升级当轮**第一次 act() 之前**实际墙位就必须已是新档位 ——
    # "当轮晚些时候才重建"同样会被这条抓住。
    def test_promotion_rebuilds_before_the_first_act(self):
        r = self._build(3100, self._level0(), True)
        r._cur_level_iter = 3000
        r._w_fill[:] = C.CURRICULUM_WINDOW_EPISODES
        r._w_stood[:] = True
        seen: list = []
        orig_act = r.alg.act

        def act(obs):
            seen.append(compiled_state(r.env))
            return orig_act(obs)

        r.alg.act = act
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(1)
        self.assertTrue(seen, "本轮必须至少采样一次")
        self.assertEqual(seen[0], (C.get_wall_positions_for_level(1), True),
                         "升级当轮第一次 act() 之前, 实际墙位必须已是新档位")

    def test_ingest_fills_window_from_published_episodes(self):
        r = self._build(3000, self._level0(), True)
        cmd = r.env.unwrapped.command_manager.get_term("backup_cmd")
        for _ in range(C.CURRICULUM_WINDOW_EPISODES):
            cmd.publish([0, 1], [True, False])
            r._ingest_episode_results()
        self.assertEqual(int(r._w_fill[0]), C.CURRICULUM_WINDOW_EPISODES)
        p_stood, p_onset, p_done, ready, n_valid = r._curriculum_metrics()
        self.assertTrue(ready)
        self.assertEqual(n_valid, 2 * C.CURRICULUM_WINDOW_EPISODES)
        self.assertAlmostEqual(p_done, 0.5, places=6)
        self.assertAlmostEqual(p_stood, 0.5, places=6, msg="三个率共用同一窗口与分母")
        # 再发布一次不重复计入 (序号增量)
        r._ingest_episode_results()
        self.assertEqual(int(r._w_fill[0]), C.CURRICULUM_WINDOW_EPISODES)

    def test_logger_accumulators_cleared_on_rebuild(self):
        # 重建会中断所有在跑的回合，不清零就会跨重建拼接统计
        r = self._build(2999, self._level0(), False)
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
        r = self._build(2999, self._level0(), False)
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
        r = self._build(2999, self._level0(), False)
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
            name="restricted_space", wall_x_neg=-0.20, wall_x_pos=0.20,
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

    # 墙位是原语、间距是派生量：不对称配置必须能被正确表达，且两种写法不能混用
    # （历史上"a=0.40"与"负侧墙位 −0.40"混用会得到完全不同的几何）。
    def test_wall_pair_is_primitive_and_width_is_derived(self):
        cfg = E.build_restricted_space_cfg(True, wall_x_neg=-0.20, wall_x_pos=0.08)
        self.assertAlmostEqual(cfg.corridor_width, 0.28, places=12)
        self.assertAlmostEqual(cfg.clear_width, 0.26, places=12)
        sym = E.build_restricted_space_cfg(True, corridor_width=0.40)
        self.assertAlmostEqual(sym.wall_x_neg, -0.20, places=12)
        self.assertAlmostEqual(sym.wall_x_pos, 0.20, places=12)

    def test_mixed_wall_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            E.build_restricted_space_cfg(True, corridor_width=0.40,
                                         wall_x_neg=-0.20, wall_x_pos=0.08)
        with self.assertRaises(ValueError):
            E.build_restricted_space_cfg(True, wall_x_neg=-0.20)   # 必须成对
        with self.assertRaises(ValueError):
            E.build_restricted_space_cfg(True, wall_x_neg=0.10, wall_x_pos=-0.10)

    # 重建不是新实验的开始: 不能按 env_cfg.seed 重新播种 —— ManagerBasedRlEnv.__init__ 会调
    # seed_rng -> torch.manual_seed (全设备) + random/np/wp, 把"换墙距"和"复位所有随机流"绑在一起。
    def test_rebuild_does_not_reseed_rng(self):
        r = self._build(2999, self._level0(), False)
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(2, init_at_random_ep_len=True)
        self.assertEqual(len(self.built), 1)
        self.assertEqual(self.seen_seed, [None],
                         "重建模板必须清掉 seed, 否则新环境会重新播种全局随机流")

    def test_fixed_width_still_switches_collision(self):
        # 锁死宽度时宽度判断走不到，碰撞判断必须放在 fixed 分支之外
        r = self._build(2999, (-0.15, 0.15), False, fixed=True)
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(2)
        self.assertEqual(len(self.built), 1)
        self.assertEqual(compiled_state(r.env), ((-0.15, 0.15), True))

    def test_fixed_width_single_rebuild_per_boundary(self):
        # 宽度锁死时重建后不应再因宽度差反复重建
        r = self._build(2999, (-0.15, 0.15), False, fixed=True)
        p1, p2 = self._patches()
        with p1, p2:
            r.learn(24)
        self.assertEqual(len(self.built), 1)

    # 审查场景：记录与课程目标一致、但环境里编译的不是这个值 ⇒ 必须按**实际编译值**判定重建
    def test_load_rebuilds_even_when_record_matches_target(self):
        saved_iter, saved_level = 3750, 2
        walls = C.get_wall_positions_for_level(saved_level)
        # 启动环境编译成第 0 档；检查点记录第 2 档 ⇒ 记录与目标一致，但环境是第 0 档
        r = self._build(0, self._level0(), True)
        p1, p2 = self._patches()
        with p1, p2, self._fake_checkpoint(saved_iter, walls, True, level=saved_level):
            infos = r.load("model_x.pt")
        cw, cc = compiled_state(r.env)
        self.assertEqual(cw, walls, "续训后实际编译的墙位必须是记录值, 不是启动时的第 0 档")
        self.assertIs(cc, True)
        self.assertAlmostEqual(float(infos["corridor_state"]["wall_x_neg"]),
                               walls[0], places=12)
        self.assertEqual(int(infos["corridor_state"]["curriculum_level"]), saved_level)

    def test_load_without_record_defaults_to_level_zero(self):
        # 新检查点缺墙位记录时不能按轮次反推 (课程按能力推进, 轮次与墙位没有函数关系):
        # 档位按 0 处理并打警告, 而不是静默给一个猜出来的墙位。
        r = self._build(0, self._level0(), False)
        p1, p2 = self._patches()
        with p1, p2, self._fake_checkpoint(4000, None, None):
            r.load("model_x.pt")
        self.assertEqual(r._cur_level, 0)
        # 阶段二 (iter 4000) 应为开碰撞 ⇒ 与启动的无碰撞不一致, 会重建一次
        self.assertEqual(compiled_state(r.env), (self._level0(), True))

    def test_load_matching_state_does_not_rebuild(self):
        # 环境本来就编译成了目标值 ⇒ 不要白重建一次
        walls = C.get_wall_positions_for_level(0)
        r = self._build(0, walls, True)
        p1, p2 = self._patches()
        with p1, p2, self._fake_checkpoint(4000, walls, True, level=0):
            r.load("model_x.pt")
        self.assertEqual(len(self.built), 0)
        self.assertEqual(compiled_state(r.env), (walls, True))

    def test_load_actor_only_keeps_compiled_env(self):
        # 回放加载 (load_cfg={"actor": True}): 回放入口已按记录把墙编译好了,
        # 这里重建会把查看器手里的 env 换掉 (查看器不接管 runner.env), 必须不重建。
        r = self._build(2999, self._level0(), False)
        p1, p2 = self._patches()
        with p1, p2, self._fake_checkpoint(4000, ((-0.15, 0.15)), True):
            r.load("model_x.pt", load_cfg={"actor": True})
        self.assertEqual(len(self.built), 0, "回放加载不得重建环境")
        cw, cc = compiled_state(r.env)
        self.assertEqual(cw, self._level0())
        self.assertIs(cc, False, "回放必须保留入口已编译好的无碰撞配置")
        self.assertEqual(EVENTS.count("close(old)"), 0, "回放加载不得关闭入口的环境")

    # 锁死模式续训：记录里的标记要生效, 且模板墙位同步成记录值
    def test_load_restores_fixed_width_mode(self):
        saved_iter, saved_walls = 4000, (-0.15, 0.15)
        r = self._build(0, self._level0(), True)          # 启动: 非锁死, 第 0 档
        p1, p2 = self._patches()
        with p1, p2, self._fake_checkpoint(saved_iter, saved_walls, True, fixed=True):
            r.load("model_x.pt")
        cw, cc = compiled_state(r.env)
        self.assertEqual(cw, saved_walls,
                         "锁死模式下应按记录的墙位重建, 而不是启动墙位")
        self.assertIs(cc, True)
        self.assertEqual(r._corridor_start_walls, saved_walls)

    # 命令行显式锁死墙位就不该被记录里的锁死标记带跑
    def test_cli_fixed_width_wins_over_record(self):
        cli_walls = (-0.125, 0.125)
        r = self._build(0, cli_walls, False, fixed=True, fixed_by_cli=True)
        p1, p2 = self._patches()
        # 记录说"非锁死、第 4 档", 但本次命令行锁死 (-0.125, 0.125) ⇒ 目标仍是它
        with p1, p2, self._fake_checkpoint(4000, C.get_wall_positions_for_level(4), True,
                                           fixed=False, level=4):
            r.load("model_x.pt")
        cw, _ = compiled_state(r.env)
        self.assertEqual(cw, cli_walls, "命令行锁死墙位优先于检查点记录的锁死标记")
        self.assertTrue(r._corridor_fixed)
        # 锁死模式下目标墙位恒为命令行墙位, 与课程档位无关
        self.assertEqual(r._corridor_target()[0], cli_walls)

    # 伪造检查点：torch.load 返回的最小可用结构（父类只读这几个键）
    def _fake_checkpoint(self, it: int, walls, collision, fixed: bool = False, level: int = 0):
        state = {"corridor_collision": collision, "corridor_fixed": fixed,
                 "curriculum_level": level, "curriculum_iter": it}
        if walls is not None:
            state["wall_x_neg"] = walls[0]
            state["wall_x_pos"] = walls[1]
            state["corridor_width"] = walls[1] - walls[0]
        infos = {
            "env_state": {"common_step_counter": it * C._STEPS_PER_ITER},
            "corridor_state": state,
        }
        payload = {"iter": it, "infos": infos, "model_state_dict": {}}
        return patch.object(runner_mod.torch, "load", lambda *a, **k: payload)


if __name__ == '__main__':
    unittest.main(verbosity=2)
