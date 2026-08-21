"""共享视觉接口中姿态/遮挡适配器的单元测试。"""

import unittest

import numpy as np

from cross_event_verifier.participant_b.adapters import occlusion_scores, pose_feature


class AdapterTests(unittest.TestCase):
    def test_occlusion_is_per_box_coverage(self) -> None:
        scores = occlusion_scores([[0, 0, 100, 200], [25, 50, 75, 150]])
        self.assertAlmostEqual(float(scores[0]), 0.25, places=5)
        self.assertAlmostEqual(float(scores[1]), 1.0, places=5)

    def test_pose_adapter_keeps_41_dimensions(self) -> None:
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[:, 2] = 0.9
        keypoints[:, 0] = np.linspace(20, 80, 17)
        keypoints[:, 1] = np.linspace(10, 170, 17)
        feature = pose_feature([0, 0, 100, 200], keypoints)
        self.assertIsNotNone(feature)
        self.assertEqual(feature.shape, (41,))
        self.assertAlmostEqual(float(np.linalg.norm(feature)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
