import os
import unittest
from unittest.mock import patch

from analysis.build_final_imo_remaining_error_atlas import classify_remaining_error_bucket
from proofgrade.policy import build_instruction


IMO_INPUTS = {
    "domain": "imo_grading",
    "problem": "P",
    "solution": "S",
    "grading_guidelines": "(Partial) key step",
    "student_answer": "A",
}


class TestFinalImoLock(unittest.TestCase):
    def test_no_top_end_guard_variant_removes_only_top_end_lines(self) -> None:
        prompt = build_instruction(
            problem="P",
            solution="S",
            grading_guidelines="(Partial) key step",
            student_answer="A",
            prompt_variant="guideline_gate_no_top_end_guard_v1",
        )
        self.assertIn("If more than one serious gap remains, or a required case is still missing, do not use `almost`.", prompt)
        self.assertNotIn("Choose `correct` only when nothing important is missing.", prompt)
        self.assertNotIn("If the answer uses examples, hand-waving, or says a proof is omitted, it cannot be `correct`.", prompt)

    def test_remaining_error_bucket_detects_overgenerous_full_credit(self) -> None:
        bucket = classify_remaining_error_bucket(
            gold="partial",
            baseline="correct",
            gate="correct",
            final_winner="correct",
            ablation="partial",
        )
        self.assertEqual(bucket, "overgenerous_full_credit")

    def test_remaining_error_bucket_detects_almost_boundary(self) -> None:
        bucket = classify_remaining_error_bucket(
            gold="almost",
            baseline="correct",
            gate="correct",
            final_winner="partial",
            ablation="partial",
        )
        self.assertEqual(bucket, "almost_vs_partial_boundary")

    def test_remaining_error_bucket_detects_reasoning_failure(self) -> None:
        bucket = classify_remaining_error_bucket(
            gold="incorrect",
            baseline="correct",
            gate="correct",
            final_winner="correct",
            ablation="correct",
        )
        self.assertEqual(bucket, "overgenerous_full_credit")
        bucket = classify_remaining_error_bucket(
            gold="correct",
            baseline="partial",
            gate="partial",
            final_winner="partial",
            ablation="partial",
        )
        self.assertEqual(bucket, "reasoning_or_comprehension_failure")


if __name__ == "__main__":
    unittest.main()
