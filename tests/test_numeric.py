"""校准、可靠性和开放集指标的小型确定性测试。"""

import unittest

from cross_event_verifier.evaluation import (
    compare_encoder_embeddings,
    d_prime,
    equal_error_rate,
    fnir_at_fpir,
    max_formal_similarity,
    threshold_at_fpir,
    threshold_metrics,
)
from cross_event_verifier.reliability import fuse_similarity


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

    def test_open_set_reports_separation_and_max_impostor(self) -> None:
        self.assertGreater(d_prime([0.9, 0.8], [0.1, 0.4]), 0.0)
        fnir, threshold = fnir_at_fpir([0.9, 0.8], [0.1, 0.4], 0.01)
        self.assertEqual(fnir, 0.0)
        self.assertGreaterEqual(threshold, 0.4)
        self.assertGreaterEqual(threshold_at_fpir([0.1, 0.4], 0.01), 0.4)
        values = max_formal_similarity(
            [[0.9, 0.435889894], [0.435889894, 0.9]],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        self.assertEqual(len(values), 2)
        self.assertAlmostEqual(values[0], 0.9, places=6)
        self.assertAlmostEqual(values[1], 0.9, places=6)

    def test_encoder_ab_report_contains_genuine_impostor_and_operating_points(self) -> None:
        report = compare_encoder_embeddings(
            {
                "rtmpose": {
                    "P1": [[1.0, 0.0], [0.99, 0.14]],
                    "P2": [[0.0, 1.0], [0.14, 0.99]],
                },
                "hrnet": {
                    "P1": [[1.0, 0.0], [0.99, 0.14]],
                    "P2": [[0.0, 1.0], [0.14, 0.99]],
                },
            },
        )

        self.assertEqual({item.encoder for item in report}, {"rtmpose", "hrnet"})
        self.assertTrue(all(item.genuine_similarity for item in report))
        self.assertTrue(all(item.impostor_similarity for item in report))
        self.assertTrue(all(len(item.fnir_at_fpir) == 2 for item in report))
        self.assertTrue(all(item.max_impostor_similarity is not None for item in report))


if __name__ == "__main__":
    unittest.main()
