import unittest

from analysis.build_imo_error_boundary_atlas import classify_boundary_bucket, infer_error_causes
from analysis.direct_imo_utils import bootstrap_delta_summary, slice_rows


class TestDirectImoRobustness(unittest.TestCase):
    def test_slice_rows_respects_bounds(self) -> None:
        rows = [{"id": str(i)} for i in range(10)]
        shard = slice_rows(rows, 2, 5)
        self.assertEqual([row["id"] for row in shard], ["2", "3", "4"])

    def test_bootstrap_delta_summary_is_reproducible(self) -> None:
        baseline = [
            {"Grading ID": "a", "prediction": "correct", "Reward": "partial"},
            {"Grading ID": "b", "prediction": "incorrect", "Reward": "incorrect"},
            {"Grading ID": "c", "prediction": "correct", "Reward": "correct"},
        ]
        candidate = [
            {"Grading ID": "a", "prediction": "partial", "Reward": "partial"},
            {"Grading ID": "b", "prediction": "incorrect", "Reward": "incorrect"},
            {"Grading ID": "c", "prediction": "correct", "Reward": "correct"},
        ]
        first = bootstrap_delta_summary(baseline, candidate, iterations=200, seed=7)
        second = bootstrap_delta_summary(baseline, candidate, iterations=200, seed=7)
        self.assertEqual(first, second)
        self.assertGreater(first["accuracy_delta_mean"], 0.0)
        self.assertLess(first["mae_delta_mean"], 0.0)

    def test_boundary_bucket_detects_overcredit(self) -> None:
        bucket = classify_boundary_bucket("partial", "correct", "correct")
        self.assertEqual(bucket, "overcredit_correct")
        causes = infer_error_causes(bucket, "partial", "correct", "correct")
        self.assertIn("prompt_policy_calibration", causes)

    def test_boundary_bucket_detects_almost_cases(self) -> None:
        bucket = classify_boundary_bucket("almost", "correct", "partial")
        self.assertEqual(bucket, "almost_boundary")
        causes = infer_error_causes(bucket, "almost", "correct", "partial")
        self.assertIn("class_imbalance_rare_case", causes)
        self.assertIn("rubric_boundary", causes)

    def test_boundary_bucket_detects_stubborn_same_wrong(self) -> None:
        bucket = classify_boundary_bucket("incorrect", "partial", "partial")
        self.assertEqual(bucket, "partial_boundary")
        causes = infer_error_causes("stubborn_same_wrong", "incorrect", "partial", "partial")
        self.assertIn("model_reasoning_limit_candidate", causes)


if __name__ == "__main__":
    unittest.main()
