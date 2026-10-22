# 验证索引解析、乱序模型复位与足端力名称映射，不创建仿真环境。
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from mjlab.asset_zoo.robots.SQuRo import SQuRo_constants
from mjlab.tasks.SQuRo_Backup.mdp import events, indices


# 提供按名称查询的乱序模型及状态写入记录。
class NamedModel:
    def __init__(self, joint_names):
        self.joint_names = list(joint_names)
        self.num_joints = len(joint_names)
        self.site_names = list(indices.FOOT_SITE_NAMES) + [
            name for pair in indices.SEGMENT_BELLY_BACK_SITES for name in pair
        ]
        self.positions = None

    # 按请求顺序返回身体编号。
    def find_bodies(self, names, **kwargs):
        return [indices.BODY_NAMES.index(name) for name in names], list(names)

    # 按请求顺序返回标记编号。
    def find_sites(self, names, **kwargs):
        return [self.site_names.index(name) for name in names], list(names)

    # 按请求顺序返回模型关节编号。
    def find_joints(self, names, **kwargs):
        return [self.joint_names.index(name) for name in names], list(names)

    # 保存本次写入的关节状态。
    def write_joint_state_to_sim(self, positions, velocities, env_ids):
        self.positions = positions.clone()
        self.velocities = velocities.clone()

    # 保存本次写入的基座状态。
    def write_root_state_to_sim(self, state, env_ids):
        self.root = state.clone()


class BackupIndicesTests(unittest.TestCase):
    # 隔离全局索引缓存。
    def setUp(self):
        self.saved = {name: getattr(indices._MODEL_INDICES, name) for name in indices.ModelIndices.__slots__}
        fresh = indices.ModelIndices()
        for name in indices.ModelIndices.__slots__:
            setattr(indices._MODEL_INDICES, name, getattr(fresh, name))
        xml_path = Path(SQuRo_constants.__file__).parent / "xmls" / "SQuRo.xml"
        self.xml_names = [node.get("name") for node in ET.parse(xml_path).findall(".//joint") if node.get("name")]

    # 恢复其他验收脚本使用的缓存。
    def tearDown(self):
        for name, value in self.saved.items():
            setattr(indices._MODEL_INDICES, name, value)

    # 旧版复位姿态在模型关节乱序后仍必须逐名称相等。
    def test_reset_matches_v01_with_permuted_model(self):
        legacy_ids = [6, 8, 12, 14, 24, 26, 30, 32, 1, 3, 21, 23, 4, 5,
                      7, 9, 10, 11, 13, 15, 16, 17, 25, 27, 28, 29, 31, 33, 34, 35]
        expected = dict(zip((self.xml_names[i] for i in legacy_ids), events.JOINT_POSITIONS, strict=True))
        robot = NamedModel(list(reversed(self.xml_names)))
        env = NS(device="cpu", scene=NS(entities={"robot": robot}))
        events.apply_fallen_state(env, torch.tensor([0]))
        for name in self.xml_names:
            self.assertAlmostEqual(float(robot.positions[0, robot.joint_names.index(name)]), expected.get(name, 0.0), places=6)
        self.assertTrue(bool((robot.velocities == 0).all()))
        self.assertAlmostEqual(float(robot.root[0, 2]), 0.024, places=6)

    # 解析失败不得留下已完成的缓存标记。
    def test_missing_linkage_does_not_commit_partial_indices(self):
        robot = NamedModel(self.xml_names)
        original = robot.find_joints
        robot.find_joints = lambda names, **kw: ([], []) if tuple(names) == indices.LINKAGE_JOINT_NAMES else original(names, **kw)
        with self.assertRaisesRegex(ValueError, "解析不完整"):
            indices.resolve_model_indices(robot)
        self.assertEqual(indices._MODEL_INDICES.f_body_id, -1)
        self.assertEqual(indices._MODEL_INDICES.joint_ids, ())

    # 不完整的闭链缓存必须重新解析。
    def test_incomplete_cache_is_resolved(self):
        indices._MODEL_INDICES.f_body_id = 0
        indices._MODEL_INDICES.joint_ids = tuple(range(indices._ACT_NUM))
        robot = NamedModel(self.xml_names)
        indices.resolve_model_indices(robot)
        self.assertEqual(indices._MODEL_INDICES.linkage_joint_ids,
                         tuple(self.xml_names.index(name) for name in indices.LINKAGE_JOINT_NAMES))

    # 动作和参考分组必须保持旧策略的列顺序。
    def test_actuator_groups_preserve_layout(self):
        self.assertEqual(indices._ACT_NUM, 14)
        self.assertEqual(indices._ACT_LEG_IDS, (4, 5, 6, 7, 10, 11, 12, 13))
        self.assertEqual(indices._ACT_SPN_IDS, (0, 1, 8, 9))
        self.assertEqual(indices._ACT_NECK_IDS, (2, 3))
        self.assertIs(indices._ACTUATED_JOINT_NAMES, indices._ACT_JOINT_NAMES)

    # 接触传感器的原始顺序变化不改变 CSV 的足端顺序。
    def test_force_order_uses_sensor_names(self):
        names = ["HR_foot_collision", "FL_foot_collision", "FR_foot_collision", "HL_foot_collision"]
        self.assertEqual(indices.resolve_foot_force_ids(NS(primary_names=names)), (1, 2, 3, 0))


if __name__ == "__main__":
    unittest.main()
