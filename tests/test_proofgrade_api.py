from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from proofgrade.api import create_app
from proofgrade.config import load_settings
from proofgrade.schemas import GradeResponse


class ProofgradeApiTests(unittest.TestCase):
    def setUp(self) -> None:
        settings = load_settings(overrides={"prompt_variant": "guideline_gate_almost_boundary_v1"})
        self.client = TestClient(create_app(settings, validate_credentials_on_startup=False))

    def test_health_and_version_endpoints(self) -> None:
        health = self.client.get("/health")
        version = self.client.get("/version")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(version.status_code, 200)
        self.assertEqual(health.json()["default_prompt_variant"], "guideline_gate_almost_boundary_v1")
        self.assertEqual(version.json()["package"], "proofgrade")

    def test_grade_endpoint_returns_structured_payload(self) -> None:
        fake_response = GradeResponse(
            label="correct",
            rationale="Complete proof",
            matched_guideline="complete",
            confidence=None,
            review_recommended=False,
            prompt_variant="guideline_gate_almost_boundary_v1",
            model_provider="gemini",
            model_name="gemini-3-flash-preview",
            parse_source="json:label",
            latency_ms=11,
            version="0.1.0",
            git_sha="abc123",
            request_id="req-1",
        )
        with patch("proofgrade.api.grade_submission") as mocked_grade:
            mocked_grade.return_value = type(
                "FakeResult",
                (),
                {
                    "response": fake_response,
                    "raw_text": "{}",
                },
            )()
            response = self.client.post(
                "/grade",
                json={
                    "problem": "P",
                    "solution": "S",
                    "grading_guidelines": "G",
                    "student_answer": "A",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["label"], "correct")
        self.assertEqual(response.json()["prompt_variant"], "guideline_gate_almost_boundary_v1")


if __name__ == "__main__":
    unittest.main()
