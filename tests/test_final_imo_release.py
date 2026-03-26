from __future__ import annotations

import unittest
from pathlib import Path

from analysis.direct_imo_utils import load_yaml
from analysis.run_fresh_generalization_eval import (
    build_fresh_generalization_set,
    classify_fresh_error_bucket,
    classify_generalization_scale,
)


class FinalImoReleaseTests(unittest.TestCase):
    def test_fresh_generalization_set_matches_expected_remainder(self) -> None:
        config = load_yaml(Path("configs/baseline_freeze/fresh_generalization_eval.yaml"))
        rows, metadata = build_fresh_generalization_set(config)
        self.assertEqual(len(rows), 512)
        self.assertEqual(metadata["remaining_total"], 512)
        self.assertEqual(metadata["used_total"], 300)
        self.assertEqual(metadata["used_not_in_base"], 0)
        self.assertEqual(metadata["problem_id_overlap"], 30)
        self.assertEqual(metadata["problem_id_only_remaining"], 0)

    def test_fresh_error_bucket_detects_overcredit(self) -> None:
        bucket = classify_fresh_error_bucket(
            gold="partial",
            baseline="correct",
            winner="correct",
        )
        self.assertEqual(bucket, "overgenerous_full_credit")

    def test_generalization_scale_reports_absent_for_non_positive_gain(self) -> None:
        self.assertEqual(
            classify_generalization_scale(
                lockbox_accuracy_delta=0.13,
                lockbox_mae_delta=-0.086,
                fresh_accuracy_delta=0.0,
                fresh_mae_delta=-0.01,
            ),
            "absent",
        )
        self.assertEqual(
            classify_generalization_scale(
                lockbox_accuracy_delta=0.13,
                lockbox_mae_delta=-0.086,
                fresh_accuracy_delta=0.09,
                fresh_mae_delta=-0.07,
            ),
            "similar",
        )


if __name__ == "__main__":
    unittest.main()
