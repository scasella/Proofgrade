from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from analysis.validate_transfer_eligibility import validate_candidate_snapshot


class SharedPathTransferGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.base_root = self.root / "base"
        self.base_root.mkdir(parents=True)
        (self.base_root / "task_agent.py").write_text(
            "class TaskAgent:\n"
            "    def forward(self, inputs):\n"
            "        return inputs['domain']\n",
            encoding="utf-8",
        )
        (self.base_root / "utils").mkdir()
        (self.base_root / "utils/prediction_contracts.py").write_text(
            "def build_task_instruction(inputs):\n"
            "    return 'shared'\n\n"
            "def _build_paper_review_instruction(inputs):\n"
            "    return 'paper'\n",
            encoding="utf-8",
        )
        (self.base_root / "agent").mkdir()
        (self.base_root / "agent/llm.py").write_text(
            "def _infer_gemini_response_schema(msg):\n"
            "    if 'paper_review' in msg:\n"
            "        return {'label': 'accept'}\n"
            "    if 'imo_grading' in msg:\n"
            "        return {'label': 'correct'}\n"
            "    if '<json>' in msg:\n"
            "        return {'label': 'fallback'}\n"
            "    return None\n",
            encoding="utf-8",
        )

        self.allowlist_path = self.root / "allowlist.yaml"
        self.allowlist_path.write_text(
            yaml.safe_dump(
                {
                    "noise_excludes": {"path_prefixes": []},
                    "forbidden_path_prefixes": {
                        "source_only": [],
                        "target_only": [],
                        "reporting_only": [],
                    },
                    "shared_editable_symbols": [
                        {
                            "symbol_id": "task_agent.py::TaskAgent.forward",
                            "path": "task_agent.py",
                            "symbol": "TaskAgent.forward",
                            "lines": [1, 3],
                            "reason": "shared",
                        }
                    ],
                    "forbidden_symbols": {
                        "source_only": [
                            {
                                "symbol_id": "utils/prediction_contracts.py::_build_paper_review_instruction",
                                "path": "utils/prediction_contracts.py",
                                "symbol": "_build_paper_review_instruction",
                                "lines": [4, 5],
                                "reason": "source only",
                            }
                        ],
                        "target_only": [],
                    },
                    "mixed_symbols": [
                        {
                            "symbol_id": "agent/llm.py::_infer_gemini_response_schema",
                            "path": "agent/llm.py",
                            "policy": "ambiguous",
                            "blocks": [
                                {"block_id": "paper_review_branch", "category": "source_only", "lines": [2, 3]},
                                {"block_id": "imo_grading_branch", "category": "target_only", "lines": [4, 5]},
                                {"block_id": "generic_json_branch", "category": "unexecuted", "lines": [6, 7]},
                            ],
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _copy_base(self, name: str) -> Path:
        candidate = self.root / name
        candidate.mkdir()
        for path in self.base_root.rglob("*"):
            if path.is_dir():
                (candidate / path.relative_to(self.base_root)).mkdir(parents=True, exist_ok=True)
            else:
                target = candidate / path.relative_to(self.base_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        return candidate

    def test_shared_change_is_eligible(self) -> None:
        candidate = self._copy_base("candidate_shared")
        (candidate / "task_agent.py").write_text(
            "class TaskAgent:\n"
            "    def forward(self, inputs):\n"
            "        return inputs['domain'].strip()\n",
            encoding="utf-8",
        )
        result = validate_candidate_snapshot(
            base_root=self.base_root,
            candidate_root=candidate,
            allowlist_path=self.allowlist_path,
        )
        self.assertEqual(result["verdict"], "eligible")

    def test_paper_review_local_change_is_ineligible(self) -> None:
        candidate = self._copy_base("candidate_source_only")
        (candidate / "utils/prediction_contracts.py").write_text(
            "def build_task_instruction(inputs):\n"
            "    return 'shared'\n\n"
            "def _build_paper_review_instruction(inputs):\n"
            "    return 'paper changed'\n",
            encoding="utf-8",
        )
        result = validate_candidate_snapshot(
            base_root=self.base_root,
            candidate_root=candidate,
            allowlist_path=self.allowlist_path,
        )
        self.assertEqual(result["verdict"], "ineligible")

    def test_mixed_generic_branch_is_ambiguous(self) -> None:
        candidate = self._copy_base("candidate_mixed")
        (candidate / "agent/llm.py").write_text(
            "def _infer_gemini_response_schema(msg):\n"
            "    if 'paper_review' in msg:\n"
            "        return {'label': 'accept'}\n"
            "    if 'imo_grading' in msg:\n"
            "        return {'label': 'correct'}\n"
            "    if '<json>' in msg:\n"
            "        return {'label': 'fallback changed'}\n"
            "    return None\n",
            encoding="utf-8",
        )
        result = validate_candidate_snapshot(
            base_root=self.base_root,
            candidate_root=candidate,
            allowlist_path=self.allowlist_path,
        )
        self.assertEqual(result["verdict"], "ambiguous")


if __name__ == "__main__":
    unittest.main()
