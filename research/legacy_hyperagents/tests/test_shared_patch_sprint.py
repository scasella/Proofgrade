from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from analysis.build_shared_failure_atlas import _classify_invalid_output
from analysis.run_shared_patch_benchmark import (
    _apply_shared_instruction_wrapper_tightening,
    _apply_shared_truncated_json_label_salvage,
)


class SharedPatchSprintTests(unittest.TestCase):
    def test_classify_invalid_output_detects_truncated_json_label(self) -> None:
        result = _classify_invalid_output(
            '{"label": "accept", "confidence": 1, "rationale": "This is cut off'
        )
        self.assertIn("truncated_json_with_visible_label", result["symptoms"])
        self.assertEqual(result["visible_partial_label_value"], "accept")
        self.assertEqual(result["fixability"], "shared_fixable")

    def test_apply_salvage_variant_rewrites_shared_parser(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            path = root / "utils/prediction_contracts.py"
            path.parent.mkdir(parents=True)
            path.write_text(
                """def _extract_json_label_candidate(raw_text: str) -> tuple[str | int | float | None, str]:
    objects = _extract_json_objects(raw_text)
    for obj in reversed(objects):
        for key in JSON_LABEL_KEYS:
            if key in obj:
                return obj[key], f"json:{key}"
    return None, "none"
""",
                encoding="utf-8",
            )
            _apply_shared_truncated_json_label_salvage(root)
            updated = path.read_text(encoding="utf-8")
            self.assertIn('partial_json:', updated)
            self.assertIn('partial_match = re.search', updated)

    def test_apply_instruction_wrapper_variant_inserts_shared_wrapper(self) -> None:
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
            _apply_shared_instruction_wrapper_tightening(root)
            updated = path.read_text(encoding="utf-8")
            self.assertIn("Shared output rules for every classification task", updated)
            self.assertIn("Put `label` first", updated)


if __name__ == "__main__":
    unittest.main()
