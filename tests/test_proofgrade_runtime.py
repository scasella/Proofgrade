from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from proofgrade.config import load_settings
from proofgrade.exceptions import ConfigurationError
from proofgrade.grader import grade_submission
from proofgrade.schemas import GradeRequest


class ProofgradeRuntimeTests(unittest.TestCase):
    def test_load_settings_respects_cli_override_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "model: gemini-3-flash-preview\nprompt_variant: baseline\napi_port: 9000\n"
            )
            with patch.dict(
                os.environ,
                {
                    "PROOFGRADE_PROMPT_VARIANT": "guideline_gate_v1",
                    "PROOFGRADE_API_PORT": "9001",
                },
                clear=False,
            ):
                settings = load_settings(
                    str(config_path),
                    {"prompt_variant": "guideline_gate_almost_boundary_v1"},
                )
        self.assertEqual(settings.prompt_variant, "guideline_gate_almost_boundary_v1")
        self.assertEqual(settings.api_port, 9001)

    def test_load_settings_rejects_unknown_prompt_variant(self) -> None:
        with self.assertRaises(ConfigurationError):
            load_settings(overrides={"prompt_variant": "not_a_real_variant"})

    def test_grade_submission_returns_frozen_metadata(self) -> None:
        request = GradeRequest(
            problem="Problem",
            solution="Solution",
            grading_guidelines="Guidelines",
            student_answer="Answer",
        )
        settings = load_settings(overrides={"prompt_variant": "guideline_gate_almost_boundary_v1"})
        fake_text = """<json>
{
  "label": "almost",
  "rationale": "One case is still missing",
  "matched_guideline": "almost"
}
</json>"""
        with patch("proofgrade.grader.complete") as mocked_complete:
            mocked_complete.return_value = type(
                "FakeCompletion",
                (),
                {
                    "text": fake_text,
                    "provider": "gemini_rest",
                    "model": "gemini/gemini-3-flash-preview",
                    "latency_ms": 42,
                },
            )()
            result = grade_submission(request, settings)

        payload = result.response.model_dump()
        self.assertEqual(payload["label"], "almost")
        self.assertEqual(payload["matched_guideline"], "almost")
        self.assertTrue(payload["review_recommended"])
        self.assertEqual(payload["prompt_variant"], "guideline_gate_almost_boundary_v1")
        self.assertEqual(payload["model_provider"], "gemini")


if __name__ == "__main__":
    unittest.main()
