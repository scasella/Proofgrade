from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from analysis.build_lineage_dataset import build_lineage_rows
from analysis.run_transfer_ablation import build_manifest
from analysis.select_transfer_agents import load_candidate_dataframe, rank_candidates
from analysis.smoke_data import create_meta_transfer_fixture
from utils.meta_memory import build_meta_memory


class MetaTransferSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.manifest = create_meta_transfer_fixture(self.root / "fixture")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_lineage_rows_include_expected_categories(self) -> None:
        run_dirs = list(self.manifest["source_runs"].values()) + list(self.manifest["target_runs"].values())
        rows = build_lineage_rows(run_dirs)
        self.assertTrue(rows)
        self.assertTrue(any(row["task_level"] for row in rows))
        self.assertTrue(any(row["meta_level"] for row in rows))
        self.assertTrue(any(row["search_level"] for row in rows))

    def test_meta_memory_contains_trends_and_hypotheses(self) -> None:
        memory = build_meta_memory(
            self.manifest["source_runs"]["source_a"],
            window=3,
            include_patch_labels=True,
        )
        self.assertIn("recent_score_trends", memory)
        self.assertIn("candidate_hypotheses", memory)
        self.assertTrue(memory["candidate_hypotheses"])
        self.assertTrue(memory["best_generations_by_domain"]["paper_review"])

    def test_selector_and_manifest_preparation(self) -> None:
        candidates = load_candidate_dataframe(list(self.manifest["source_runs"].values()))
        ranked = rank_candidates(candidates, "descendant_growth")
        self.assertFalse(ranked.empty)
        self.assertEqual(ranked.iloc[0]["run_id"], "smoke_source_a")

        manifest = build_manifest(
            source_run_dirs=list(self.manifest["source_runs"].values()),
            selector="meta_patch_density",
            transfer_modes=["meta_only_transfer", "task_only_transfer", "memory_only_transfer"],
            output_dir=self.root / "prepared",
            target_domain="imo_grading",
            max_generation=5,
            continue_self_improve=False,
            use_meta_memory=True,
        )
        ready_modes = {row["mode"] for row in manifest["experiments"] if row["status"] == "ready"}
        self.assertIn("meta_only_transfer", ready_modes)
        self.assertIn("task_only_transfer", ready_modes)
        self.assertIn("memory_only_transfer", ready_modes)

    def test_reproduction_script_writes_report(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        output_dir = self.root / "repro"
        report_path = self.root / "repro.md"
        subprocess.run(
            [
                sys.executable,
                "analysis/reproduce_transfer_figures.py",
                "--auto_discover",
                str(self.root / "fixture" / "target_runs"),
                str(self.root / "fixture" / "selector_runs"),
                "--output_dir",
                str(output_dir),
                "--report_path",
                str(report_path),
                "--workspace_root",
                str(repo_root),
            ],
            cwd=repo_root,
            check=True,
        )
        self.assertTrue(report_path.exists())
        self.assertIn("Closest Match", report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
