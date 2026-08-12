"""校准、可靠性和开放集指标的小型确定性测试。"""

import unittest

from cross_event_verifier.participant_a.evaluation import equal_error_rate, threshold_metrics
from cross_event_verifier.participant_a.reliability import fuse_similarity


class NumericTests(unittest.TestCase):
    def test_legacy_reliability_wrapper_keeps_gait_bounded(self) -> None:
        score, weight = fuse_similarity(0.7, 0.99, 0.8, 1.0)
        self.assertLessEqual(weight, 0.35)
        self.assertGreater(score, 0.7)

    def test_open_set_metrics(self) -> None:
        report = threshold_metrics([0.9, 0.8], [0.1, 0.4], 0.75)
        self.assertEqual(report.tar, 1.0)
        self.assertEqual(report.far, 0.0)
        eer, threshold = equal_error_rate([0.9, 0.8], [0.1, 0.4])
        self.assertGreaterEqual(eer, 0.0)
        self.assertLessEqual(eer, 1.0)
        self.assertIn(threshold, (0.1, 0.4, 0.8, 0.9))


if __name__ == "__main__":
    unittest.main()
