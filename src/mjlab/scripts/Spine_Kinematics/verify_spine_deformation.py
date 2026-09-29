import math
import unittest
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mjlab.asset_zoo.robots.SQuRo.SQuRo_constants import get_spec
from mjlab.tasks.SQuRo_Backup.mdp.spine_deformation import SQuRoSpineDeformation
from mjlab.utils.spine_deformation import (
  SpineDeformationState,
  SpineDeformationTracker,
  compute_spine_deformation,
)


# 构造具有已知弯曲平面和前后扭转的几何姿态。
def pose(lateral=0.0, sagittal=0.0, mean=0.0, axial=0.0, count=1):
  front_bend = Rotation.from_rotvec([0, -sagittal / 2, lateral / 2]).as_matrix()
  hind_bend = Rotation.from_rotvec([0, sagittal / 2, -lateral / 2]).as_matrix()
  front = front_bend @ Rotation.from_euler("x", mean + axial / 2).as_matrix()
  hind = hind_bend @ Rotation.from_euler("x", mean - axial / 2).as_matrix()
  values = (front_bend[:, 0], np.zeros(3), -hind_bend[:, 0], front, hind)
  return tuple(
    torch.tensor(v, dtype=torch.float64).unsqueeze(0).repeat(count, *([1] * v.ndim))
    for v in values
  )


class GeometryTests(unittest.TestCase):
  # 纯侧摆、纯俯仰和纯扭转分别只激活对应通道。
  def test_independent_deformations(self):
    for kwargs, expected in (
      ({}, [0, 0, 0]),
      ({"lateral": 0.7}, [0.7, 0, 0]),
      ({"sagittal": -0.5}, [0, -0.5, 0]),
      ({"axial": 1.2}, [0, 0, 1.2]),
    ):
      with self.subTest(kwargs=kwargs):
        result = compute_spine_deformation(*pose(**kwargs))
        torch.testing.assert_close(
          result.angles[0],
          torch.tensor(expected, dtype=torch.float64),
          atol=1e-12,
          rtol=0,
        )
        self.assertTrue(result.valid.all())

  # 同向扭转不产生相对扭转，但会改变几何弯曲所属的身体平面。
  def test_common_roll_rotates_bending_plane(self):
    result = compute_spine_deformation(*pose(lateral=0.6, mean=math.pi / 2))
    torch.testing.assert_close(
      result.angles[0],
      torch.tensor([0, -0.6, 0], dtype=torch.float64),
      atol=1e-12,
      rtol=0,
    )

  # 前后扭转分别为二十度和六十度时，中间平面取四十度。
  def test_mean_twist_is_midpoint(self):
    result = compute_spine_deformation(
      *pose(mean=math.radians(40), axial=math.radians(-40))
    )
    expected = torch.tensor(Rotation.from_euler("x", 40, degrees=True).as_matrix())
    torch.testing.assert_close(result.frame_w[0], expected, atol=1e-12, rtol=0)

  # 整体旋转和平移只搬运中间坐标系，不改变内部形变角。
  def test_rigid_motion_invariance(self):
    original = pose(lateral=0.5, sagittal=0.3, mean=0.4, axial=1.1)
    result = compute_spine_deformation(*original)
    for rotation in (
      Rotation.from_euler("x", math.pi),
      Rotation.from_rotvec([1.2, -0.7, 2.0]),
    ):
      q = torch.tensor(rotation.as_matrix())
      translation = torch.tensor([7.0, -4.0, 8.0], dtype=torch.float64)
      moved = tuple(v @ q.T + translation for v in original[:3]) + tuple(
        q @ v for v in original[3:]
      )
      actual = compute_spine_deformation(*moved)
      torch.testing.assert_close(actual.angles, result.angles, atol=1e-12, rtol=0)
      torch.testing.assert_close(actual.frame_w, q @ result.frame_w, atol=1e-12, rtol=0)

  # 扭转跨越正负一百八十度和整周时，连续分支及弯曲符号保持正确。
  def test_unwrap_through_two_turns(self):
    tracker = SpineDeformationTracker()
    for axial in np.linspace(0, 4 * math.pi, 65):
      result = tracker.compute(*pose(lateral=0.4, axial=float(axial)))
      self.assertTrue(result.valid.all())
      self.assertAlmostEqual(result.axial_unwrapped.item(), axial, places=10)
      self.assertAlmostEqual(result.lateral.item(), 0.4, places=10)
      self.assertAlmostEqual(result.sagittal.item(), 0, places=10)

  # 初帧背向相反时不伪造平均方向，有明确历史分支时允许计算。
  def test_opposite_backs_require_branch(self):
    values = pose(lateral=0.4, axial=math.pi)
    result = compute_spine_deformation(*values)
    self.assertTrue(result.axial_valid.all())
    self.assertTrue(result.branch_ambiguous.all())
    self.assertFalse(result.bending_valid.any())
    self.assertTrue(torch.isnan(result.lateral).all())
    state = SpineDeformationState(
      torch.tensor([math.pi], dtype=torch.float64), torch.tensor([True])
    )
    resolved = compute_spine_deformation(*values, state=state)
    self.assertTrue(resolved.valid.all())
    self.assertAlmostEqual(resolved.lateral.item(), 0.4, places=10)

  # 两帧恰差半周无法确定展开方向，必须暴露歧义。
  def test_half_turn_sample_gap_is_ambiguous(self):
    tracker = SpineDeformationTracker()
    tracker.compute(*pose())
    result = tracker.compute(*pose(axial=math.pi))
    self.assertTrue(result.branch_ambiguous.all())

  # 部分复位不清理其他环境，也不改写先前交付的结果。
  def test_partial_reset_is_isolated(self):
    tracker = SpineDeformationTracker()
    for axial in (0.0, 1.0, 2.0, 3.0, 3.5):
      last = tracker.compute(*pose(axial=axial, count=2))
    tracker.reset([0])
    self.assertTrue(last.state.valid.all())
    result = tracker.compute(*pose(axial=3.6, count=2))
    self.assertAlmostEqual(result.axial_unwrapped[0].item(), 3.6 - 2 * math.pi)
    self.assertAlmostEqual(result.axial_unwrapped[1].item(), 3.6)
    tracker.reset()
    self.assertIsNone(tracker.state)

  # 连线零长、角平分线退化、纵轴反向、非有限值均不得冒充零形变。
  def test_invalid_geometry(self):
    cases = []
    values = list(pose())
    values[0] = values[1].clone()
    cases.append(values)
    values = list(pose())
    values[2] = values[0].clone()
    cases.append(values)
    values = list(pose())
    values[3] = torch.tensor(Rotation.from_euler("y", math.pi).as_matrix()).unsqueeze(0)
    cases.append(values)
    values = list(pose())
    values[0][0, 1] = float("nan")
    cases.append(values)
    values = list(pose())
    values[3] *= 2.0
    cases.append(values)
    for values in cases:
      result = compute_spine_deformation(*values)
      self.assertFalse(result.valid.any())
      self.assertTrue(torch.isnan(result.angles).all())
      self.assertFalse(result.state.valid.any())

  # 一批中的退化样本不污染其他环境。
  def test_invalid_sample_does_not_poison_batch(self):
    values = list(pose(lateral=0.4, count=3))
    values[0][1] = values[1][1]
    result = compute_spine_deformation(*values)
    self.assertEqual(result.valid.tolist(), [True, False, True])
    self.assertTrue(torch.isfinite(result.angles[[0, 2]]).all())

  # 平面只由连线方向决定，不因前后段长度比例变化而偏向长段。
  def test_segment_lengths_do_not_bias_frame(self):
    values = list(pose(lateral=0.6, sagittal=0.2, axial=0.9))
    reference = compute_spine_deformation(*values)
    values[0] *= 0.03
    values[2] *= 0.07
    result = compute_spine_deformation(*values)
    torch.testing.assert_close(result.angles, reference.angles, atol=1e-12, rtol=0)

  # 支持无批量维度，错误维度和精度必须明确报错。
  def test_scalar_and_shape_validation(self):
    result = compute_spine_deformation(*(v[0] for v in pose(axial=0.5)))
    self.assertEqual(result.angles.shape, (3,))
    values = list(pose())
    values[0] = values[0].float()
    with self.assertRaises(ValueError):
      compute_spine_deformation(*values)

  # GPU 与 CPU 的批量计算采用相同定义。
  @unittest.skipUnless(torch.cuda.is_available(), "未检测到 CUDA")
  def test_cuda_matches_cpu(self):
    values = tuple(
      v.float() for v in pose(lateral=0.6, sagittal=0.2, mean=0.5, axial=2.5, count=16)
    )
    expected = compute_spine_deformation(*values)
    result = compute_spine_deformation(*(v.cuda() for v in values))
    torch.testing.assert_close(result.angles.cpu(), expected.angles, atol=2e-6, rtol=0)


class ModelTests(unittest.TestCase):
  # 直接编译生产模型，用具名刚体与腹背标记检验解剖轴映射。
  @classmethod
  def setUpClass(cls):
    cls.model = get_spec().compile()

  # 为实体接口提供生产模型的实际前向位姿，保留名称解析过程。
  def make_reader(self, **kwargs):
    model = self.model
    data = mujoco.MjData(model)
    data.qpos[:7] = [0, 0, 0.5, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    robot = SimpleNamespace(data=SimpleNamespace())
    robot.find_bodies = lambda names, preserve_order: (
      [model.body(n).id for n in names],
      names,
    )

    # 每次更新都复制当前仿真位姿，避免测试读取过时缓存。
    def refresh():
      robot.data.body_link_pos_w = torch.tensor(data.xpos.copy()).unsqueeze(0)
      robot.data.body_link_quat_w = torch.tensor(data.xquat.copy()).unsqueeze(0)

    refresh()
    return SQuRoSpineDeformation(robot, **kwargs), data, refresh

  # 零弯曲反向扭转时读出两端真实相对角，而不是角度和。
  def test_model_twist_and_body_mapping(self):
    reader, data, refresh = self.make_reader()
    for f, h in ((0.0, 0.0), (0.3, 0.3), (-0.3, 0.3), (-1.57, 1.57)):
      for name, value in (("F_body_joint", f), ("H_body_joint", h)):
        data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]] = value
      mujoco.mj_forward(self.model, data)
      refresh()
      reader.reset()
      result = reader.compute()
      self.assertTrue(result.valid.all())
      self.assertAlmostEqual(result.axial.item(), h - f, places=5)

  # 解剖背向必须与生产模型腹背 site 连线一致。
  def test_anatomical_back_matches_sites(self):
    reader, data, _ = self.make_reader()
    for prefix, body_id, correction in (
      ("F_body", reader.body_ids[0], reader.front_frame),
      ("H_body", reader.body_ids[2], reader.hind_frame),
    ):
      back = (
        data.site_xpos[self.model.site(prefix + "_back_site").id]
        - data.site_xpos[self.model.site(prefix + "_belly_site").id]
      )
      back /= np.linalg.norm(back)
      matrix = data.xmat[body_id].reshape(3, 3) @ correction.numpy()
      np.testing.assert_allclose(matrix[:, 2], back, atol=1e-12)

  # 自定义参考点偏置随刚体旋转，且不改变仿真状态。
  def test_offsets_are_local_and_read_only(self):
    reader, data, refresh = self.make_reader(front_offset=(0.001, 0.002, -0.003))
    original = data.qpos.copy()
    result = reader.compute()
    np.testing.assert_array_equal(data.qpos, original)
    rotations = torch.tensor(data.xmat.reshape(-1, 3, 3))
    front, center, hind = reader.body_ids
    points = torch.tensor(data.xpos[[front, center, hind]].copy())
    points[0] += rotations[front] @ reader.offsets[0]
    direct = compute_spine_deformation(
      points[0:1],
      points[1:2],
      points[2:3],
      (rotations[front] @ reader.front_frame).unsqueeze(0),
      (rotations[hind] @ reader.hind_frame).unsqueeze(0),
    )
    torch.testing.assert_close(result.angles, direct.angles)


if __name__ == "__main__":
  unittest.main(verbosity=2)
