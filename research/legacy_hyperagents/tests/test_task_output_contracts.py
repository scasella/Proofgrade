from __future__ import annotations

import unittest

from utils.prediction_contracts import build_task_instruction, parse_prediction_output


class TaskOutputContractTests(unittest.TestCase):
    def test_paper_review_json_label(self) -> None:
        result = parse_prediction_output("paper_review", '<json>{"label":"accept"}</json>')
        self.assertEqual(result.label, "accept")

    def test_paper_review_backward_compatible_response(self) -> None:
        result = parse_prediction_output("paper_review", '<json>{"response":"reject"}</json>')
        self.assertEqual(result.label, "reject")

    def test_paper_review_overall_score_fallback(self) -> None:
        result = parse_prediction_output("paper_review", '<json>{"overall_score":7}</json>')
        self.assertEqual(result.label, "accept")

    def test_paper_review_verbose_phrase(self) -> None:
        result = parse_prediction_output("paper_review", "I would reject this paper.")
        self.assertEqual(result.label, "reject")

    def test_paper_review_ambiguous_text_is_invalid(self) -> None:
        result = parse_prediction_output(
            "paper_review",
            "The review discusses reasons to accept and reasons to reject without choosing one.",
        )
        self.assertIsNone(result.label)

    def test_imo_json_label(self) -> None:
        result = parse_prediction_output("imo_grading", '<json>{"label":"almost"}</json>')
        self.assertEqual(result.label, "almost")

    def test_imo_backward_compatible_response(self) -> None:
        result = parse_prediction_output("imo_grading", '<json>{"response":"partial progress"}</json>')
        self.assertEqual(result.label, "partial")

    def test_imo_points_tag(self) -> None:
        result = parse_prediction_output(
            "imo_grading",
            "Reasoning here.\n<points>6 out of 7</points>",
        )
        self.assertEqual(result.label, "almost")

    def test_imo_mixed_case_text(self) -> None:
        result = parse_prediction_output("imo_grading", "Almost Correct")
        self.assertEqual(result.label, "almost")

    def test_imo_final_answer_phrase(self) -> None:
        result = parse_prediction_output(
            "imo_grading",
            "The solution is complete. The final answer is \\boxed{correct}.",
        )
        self.assertEqual(result.label, "correct")

    def test_imo_boxed_text_final_answer_phrase(self) -> None:
        result = parse_prediction_output(
            "imo_grading",
            "The final answer is \\boxed{\\text{correct}}.",
        )
        self.assertEqual(result.label, "correct")

    def test_imo_invalid_outcome_phrase_stays_invalid(self) -> None:
        result = parse_prediction_output("imo_grading", "Outcome 1: COMPLETE PROOF")
        self.assertIsNone(result.label)

    def test_generic_domain_keeps_old_behavior(self) -> None:
        result = parse_prediction_output("search_arena", '<json>{"response":"A"}</json>')
        self.assertEqual(result.label, "A")

    def test_paper_review_instruction_mentions_review_scoring(self) -> None:
        instruction = build_task_instruction({"domain": "paper_review", "paper_text": "Example paper"})
        self.assertIn("overall_score", instruction)
        self.assertIn("originality", instruction.lower())
        self.assertIn("selective machine learning venue", instruction)


if __name__ == "__main__":
    unittest.main()
