import os
import unittest
from unittest.mock import patch

from agent.llm import _infer_gemini_response_schema
from utils.prediction_contracts import build_task_instruction, parse_prediction_output


IMO_INPUTS = {
    "domain": "imo_grading",
    "problem": "P",
    "solution": "S",
    "grading_guidelines": "(Partial) key step",
    "student_answer": "A",
}


class TestDirectImoImprovement(unittest.TestCase):
    def test_imo_almost_boundary_gate_is_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            prompt = build_task_instruction(IMO_INPUTS)
        self.assertIn("Apply this hidden decision gate before labeling:", prompt)
        self.assertIn('"matched_guideline"', prompt)
        self.assertIn("If more than one serious gap remains, or a required case is still missing, do not use `almost`.", prompt)

    def test_imo_baseline_prompt_is_available_explicitly(self) -> None:
        with patch.dict(os.environ, {"HYPERAGENTS_IMO_GRADING_PROMPT_VARIANT": "baseline"}, clear=False):
            prompt = build_task_instruction(IMO_INPUTS)
        self.assertIn('Allowed labels: "incorrect", "partial", "almost", "correct".', prompt)
        self.assertNotIn("decision_basis", prompt)
        self.assertNotIn("missing_piece", prompt)

    def test_strict_rubric_variant_changes_prompt(self) -> None:
        with patch.dict(os.environ, {"HYPERAGENTS_IMO_GRADING_PROMPT_VARIANT": "strict_rubric_v1"}, clear=False):
            prompt = build_task_instruction(IMO_INPUTS)
        self.assertIn("Use the official solution and grading guidelines as the source of truth.", prompt)
        self.assertIn("If you are torn between two labels, choose the lower one.", prompt)
        self.assertIn('"decision_basis"', prompt)

    def test_missing_piece_variant_changes_prompt(self) -> None:
        with patch.dict(os.environ, {"HYPERAGENTS_IMO_GRADING_PROMPT_VARIANT": "missing_piece_v1"}, clear=False):
            prompt = build_task_instruction(IMO_INPUTS)
        self.assertIn('"missing_piece"', prompt)
        self.assertIn("For `correct`, set `missing_piece` to `none`.", prompt)

    def test_almost_boundary_variant_changes_prompt(self) -> None:
        with patch.dict(os.environ, {"HYPERAGENTS_IMO_GRADING_PROMPT_VARIANT": "guideline_gate_almost_boundary_v1"}, clear=False):
            prompt = build_task_instruction(IMO_INPUTS)
        self.assertIn("If more than one serious gap remains, or a required case is still missing, do not use `almost`.", prompt)

    def test_fatal_flaw_variant_changes_prompt(self) -> None:
        with patch.dict(os.environ, {"HYPERAGENTS_IMO_GRADING_PROMPT_VARIANT": "guideline_gate_fatal_flaw_v1"}, clear=False):
            prompt = build_task_instruction(IMO_INPUTS)
        self.assertIn("Before choosing `correct` or `almost`, check for a fatal flaw", prompt)

    def test_no_top_end_guard_ablation_variant_changes_prompt(self) -> None:
        with patch.dict(os.environ, {"HYPERAGENTS_IMO_GRADING_PROMPT_VARIANT": "guideline_gate_no_top_end_guard_v1"}, clear=False):
            prompt = build_task_instruction(IMO_INPUTS)
        self.assertIn("If more than one serious gap remains, or a required case is still missing, do not use `almost`.", prompt)
        self.assertNotIn("Choose `correct` only when nothing important is missing.", prompt)
        self.assertNotIn("it cannot be `correct`.", prompt)

    def test_imo_parser_ignores_extra_variant_fields(self) -> None:
        raw = """
        <json>
        {
          "label": "partial",
          "decision_basis": "partial key step",
          "missing_piece": "final case",
          "matched_guideline": "partial",
          "rationale": "Found one key lemma."
        }
        </json>
        """
        result = parse_prediction_output("imo_grading", raw)
        self.assertEqual(result.label, "partial")

    def test_gemini_schema_allows_variant_fields(self) -> None:
        schema = _infer_gemini_response_schema('"domain": "imo_grading"')
        self.assertIsNotNone(schema)
        props = schema["properties"]
        self.assertIn("decision_basis", props)
        self.assertIn("missing_piece", props)
        self.assertIn("matched_guideline", props)


if __name__ == "__main__":
    unittest.main()
