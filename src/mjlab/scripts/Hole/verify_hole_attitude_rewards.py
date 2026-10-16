import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mjlab.asset_zoo.robots.SQuRo.SQuRo_constants import SPINE_FRONT_FRAME, SPINE_HIND_FRAME
from mjlab.tasks.SQuRo_Backup.mdp import indices as backup_indices
from mjlab.tasks.SQuRo_Hole.mdp import rewards
from mjlab.tasks.SQuRo_Hole.mdp.curriculums import STAGE3_STEP
from mjlab.tasks.SQuRo_Hole.mdp.config import CMD_HEIGHT_F_IDS, CMD_HEIGHT_H_IDS, COMMAND_DIM


# 由独立旋转库构造已知物理姿态，再转回模型原生坐标。
def make_env(front, hind=None, heights=None, device="cpu"):
    front = np.asarray(front, dtype=float).reshape(-1, 3)
    hind = front if hind is None else np.asarray(hind, dtype=float).reshape(-1, 3)
    physical = np.stack((Rotation.from_euler("xyz", front, degrees=True).as_matrix(),
                         Rotation.from_euler("xyz", hind, degrees=True).as_matrix()), axis=1)
    frames = np.asarray((SPINE_FRONT_FRAME, SPINE_HIND_FRAME))
    raw = physical @ frames.transpose(0, 2, 1)
    xyzw = Rotation.from_matrix(raw.reshape(-1, 3, 3)).as_quat().reshape(len(front), 2, 4)
    quat = torch.tensor(np.concatenate((xyzw[..., -1:], xyzw[..., :3]), axis=-1), device=device)
    command = torch.zeros((len(front), COMMAND_DIM), dtype=torch.float64, device=device)
    command[:, [CMD_HEIGHT_F_IDS, CMD_HEIGHT_H_IDS]] = command.new_tensor(
        heights if heights is not None else [[.055, .055]] * len(front)
    )
    return NS(scene={"robot": NS(data=NS(body_link_quat_w=quat))}, num_envs=len(front),
              device=device, common_step_counter=0,
              command_manager=NS(_terms={"hole_cmd": NS(command=command)}))


class AttitudeRewardsTests(unittest.TestCase):
    # 为合成实体临时替换索引，测试结束后恢复生产缓存。
    def setUp(self):
        self.front_patch = patch.object(rewards._MODEL_INDICES, "f_body_id", 0)
        self.hind_patch = patch.object(rewards._MODEL_INDICES, "h_body_id", 1)
        self.front_patch.start()
        self.hind_patch.start()
        self.addCleanup(self.front_patch.stop)
        self.addCleanup(self.hind_patch.stop)

    # 两个任务必须引用同一份物理标定，避免复制常量。
    def test_shared_frames(self):
        self.assertIs(SPINE_FRONT_FRAME, backup_indices.SPINE_FRONT_FRAME)
        self.assertIs(SPINE_HIND_FRAME, backup_indices.SPINE_HIND_FRAME)

    # 纯轴与复合姿态分别读回物理滚转和偏航，俯仰不串入滚转。
    def test_calibrated_angles(self):
        values = [[0, 0, 0], [20, 0, 0], [-20, 0, 0], [0, 25, 0], [0, -25, 0],
                  [0, 0, 30], [10, 20, 30], [-25, -30, -45], [179, 0, 0]]
        env = make_env(values)
        yaw, roll = rewards._get_body_orientation(env)
        expected = torch.deg2rad(torch.tensor(values, dtype=torch.float64))
        torch.testing.assert_close(yaw, expected[:, 2:3].expand(-1, 2), atol=1e-12, rtol=0)
        torch.testing.assert_close(roll, expected[:, :1].expand(-1, 2), atol=1e-12, rtol=0)

    # 只改变俯仰时，两个姿态奖励均保持满分。
    def test_pitch_is_not_roll(self):
        env = make_env([[0, 25, 0], [0, -25, 0], [0, 0, 0]])
        torch.testing.assert_close(rewards.compute_angle_reward(env), torch.ones(3, dtype=torch.float64))
        torch.testing.assert_close(rewards.compute_orientation_reward(env), torch.full((3,), 2., dtype=torch.float64))

    # 偏航使用圆周平均，跨越正负 180 度时不能被均成零。
    def test_circular_mean(self):
        front = [[0, 0, 30], [0, 0, 179], [0, 0, 20], [0, 0, 10]]
        hind = [[0, 0, 10], [0, 0, -179], [0, 0, -20], [0, 0, 30]]
        env = make_env(front, hind)
        expected = 2 * torch.exp(-5 * torch.deg2rad(torch.tensor([20., 180., 0., 20.], dtype=torch.float64)).square())
        torch.testing.assert_close(rewards.compute_orientation_reward(env), expected, atol=1e-12, rtol=1e-12)

    # 四种高度模式保留单低选段和双高相加，双低两段都参与且不能符号抵消。
    def test_mode_selection_and_scale(self):
        env = make_env([[2, 0, 0]] * 4, [[-3, 0, 0]] * 4,
                       [[.055, .055], [.02, .055], [.055, .02], [.02, .02]])
        error = torch.deg2rad(torch.tensor([5., 2., 3., 5.], dtype=torch.float64))
        sigma = error.new_tensor([100., 50., 50., 50.])
        torch.testing.assert_close(rewards.compute_angle_reward(env), torch.exp(-sigma * error.square()))

    # 改动其他环境的姿态或模式不改变当前环境的奖励。
    def test_environment_isolation(self):
        env = make_env([[5, 10, 15], [1, 2, 3]], heights=[[.02, .055], [.055, .055]])
        first = [fn(env)[0].clone() for fn in (rewards.compute_angle_reward, rewards.compute_orientation_reward)]
        env.scene["robot"].data.body_link_quat_w[1] = make_env([[65, -35, -150]]).scene["robot"].data.body_link_quat_w[0]
        env.command_manager._terms["hole_cmd"].command[1, CMD_HEIGHT_H_IDS] = .02
        for value, fn in zip(first, (rewards.compute_angle_reward, rewards.compute_orientation_reward)):
            torch.testing.assert_close(value, fn(env)[0], atol=0, rtol=0)

    # 阶段切换只改变既有朝向权重，角度权重和指数核保持现有数值。
    def test_stage_weights(self):
        env = make_env([[0, 0, 20]])
        first = rewards.compute_orientation_reward(env)
        env.common_step_counter = STAGE3_STEP
        torch.testing.assert_close(rewards.compute_orientation_reward(env), first * 2)

    # CUDA 与 CPU 使用同一坐标约定并返回有限的逐环境奖励。
    @unittest.skipUnless(torch.cuda.is_available(), "没有可用 CUDA")
    def test_cpu_cuda(self):
        front = [[10, 20, 30], [-20, 25, -179], [0, -15, 179]]
        hind = [[-5, 10, -20], [15, -10, 179], [3, 20, -179]]
        for fn in (rewards.compute_angle_reward, rewards.compute_orientation_reward):
            cpu = fn(make_env(front, hind))
            gpu = fn(make_env(front, hind, device="cuda:0")).cpu()
            self.assertTrue(torch.isfinite(gpu).all())
            torch.testing.assert_close(cpu, gpu, atol=1e-10, rtol=1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
