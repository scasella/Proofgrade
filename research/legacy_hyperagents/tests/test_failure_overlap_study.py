from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.build_failure_overlap_atlas import _classify_output_symptoms, _score_overlap
from analysis.run_overlap_guided_patch_benchmark import _apply_shared_common_wrapper_minimal_json
from analysis.select_source_by_overlap import select_source


class FailureOverlapStudyTests(unittest.TestCase):
    def test_generic_schema_mismatch_is_detected(self) -> None:
        symptoms = _classify_output_symptoms(
            "imo_proof",
            '{"label":"candidate proof","rationale":"Starts correctly"}',
            None,
        )
        self.assertIn("missing_response_field", symptoms)
        self.assertIn("generic_schema_key_mismatch_label_instead_of_response", symptoms)

    def test_overlap_score_zero_when_target_has_no_shared_symptoms(self) -> None:
        result = _score_overlap(
            source_summary={
                "shared_surface_symptom_counts": {"truncated_json_with_visible_label": 7},
                "not_shared_surface_symptom_counts": {},
            },
            target_summary={
                "shared_surface_symptom_counts": {},
                "not_shared_surface_symptom_counts": {},
            },
            weights={"truncated_json_with_visible_label": 2.5},
        )
        self.assertEqual(result["raw_overlap_score"], 0.0)
        self.assertEqual(result["weighted_overlap_score"], 0.0)

    def test_select_source_returns_none_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            atlas_path = root / "atlas.json"
            config_path = root / "config.yaml"
            atlas_path.write_text(
                json.dumps(
                    {
                        "target": {"domain": "imo_grading"},
                        "candidate_sources": [
                            {
                                "domain": "paper_review",
                                "allowlist_path": "/tmp/paper.yaml",
                                "overlap": {
                                    "weighted_overlap_score": 0.0,
                                    "raw_overlap_score": 0.0,
                                    "source_only_penalty": 0.0,
                                    "contributions": [],
                                },
                            },
                            {
                                "domain": "imo_proof",
                                "allowlist_path": "/tmp/proof.yaml",
                                "overlap": {
                                    "weighted_overlap_score": 0.5,
                                    "raw_overlap_score": 1.0,
                                    "source_only_penalty": 0.5,
                                    "contributions": [],
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config_path.write_text(
                "\n".join(
                    [
                        f"atlas_json: {atlas_path}",
                        f"selection_json: {root / 'selection.json'}",
                        "overlap:",
                        "  min_meaningful_score: 1.5",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            selection = select_source(config_path)
            self.assertIsNone(selection["selected_source"])

    def test_common_wrapper_variant_changes_both_branches(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            path = root / "utils/prediction_contracts.py"
            path.parent.mkdir(parents=True)
            path.write_text(
                """def build_task_instruction(inputs: dict[str, Any]) -> str:
    contract = get_prediction_contract(inputs["domain"])
    if contract is None:
        return f\"\"\"You are an agent.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": ...
}}
</json>\"\"\"
    return contract.build_instruction(inputs)
""",
                encoding="utf-8",
            )
            _apply_shared_common_wrapper_minimal_json(root)
            updated = path.read_text(encoding="utf-8")
            self.assertIn("Shared response rules", updated)
            self.assertIn('return shared_wrapper + "\\n\\n" + contract.build_instruction(inputs)', updated)


if __name__ == "__main__":
    unittest.main()
