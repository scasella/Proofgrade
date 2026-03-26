"""Synthetic fixtures for smoke-testing the meta-transfer pipeline."""

from __future__ import annotations

import json
from pathlib import Path


def _report_payload(domain: str, score: float, mae: float | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "total_correct": int(round(score * 100)),
        "total": 100,
    }
    if domain in {"paper_review", "imo_grading", "search_arena"}:
        payload.update(
            {
                "overall_accuracy": score,
                "accuracy_by_ground_truth": {},
                "label_distribution": {
                    "ground_truth": {"accept": 0.5, "reject": 0.5},
                    "prediction": {"accept": 0.5, "reject": 0.5},
                },
                "random_guess_accuracy": 0.5,
                "question_ids_failed": [],
                "question_ids_passed": [],
            }
        )
    elif "genesis" in domain:
        payload["average_fitness"] = score
    else:
        payload["score"] = score
    if mae is not None:
        payload["normalized_mean_absolute_error"] = mae
    return payload


def _patch_text(kind: str) -> str:
    if kind == "task":
        return """diff --git a/task_agent.py b/task_agent.py
index 1111111..2222222 100644
--- a/task_agent.py
+++ b/task_agent.py
@@ -1,3 +1,9 @@
-instruction = "Respond in JSON"
+instruction = \"\"\"You are a rigorous reviewer.
+Follow a decision checklist.
+Respond in JSON.
+\"\"\"
"""
    if kind == "meta_memory":
        return """diff --git a/meta_agent.py b/meta_agent.py
index 1111111..2222222 100644
--- a/meta_agent.py
+++ b/meta_agent.py
@@ -1,3 +1,8 @@
+memory_summary = build_meta_memory(eval_path)
+instruction += f"\\nStructured memory:\\n{memory_summary}"
diff --git a/utils/thread_logger.py b/utils/thread_logger.py
index 1111111..2222222 100644
--- a/utils/thread_logger.py
+++ b/utils/thread_logger.py
@@ -1,3 +1,8 @@
+class PerformanceTracker:
+    def record_generation(self, generation_id, domain, score):
+        self.history.append((generation_id, domain, score))
"""
    if kind == "search":
        return """diff --git a/select_next_parent.py b/select_next_parent.py
index 1111111..2222222 100644
--- a/select_next_parent.py
+++ b/select_next_parent.py
@@ -1,3 +1,10 @@
+exploration_bonus = exploration_weight * math.sqrt(
+    math.log(total_children + 1) / (children + 1)
+)
+score = normalized_score + exploration_bonus
"""
    if kind == "eval":
        return """diff --git a/utils/common.py b/utils/common.py
index 1111111..2222222 100644
--- a/utils/common.py
+++ b/utils/common.py
@@ -1,3 +1,9 @@
+def extract_points_block(response):
+    match = re.search(r"<points>(.*?)</points>", response)
+    return match.group(1) if match else None
"""
    return """diff --git a/README.md b/README.md
index 1111111..2222222 100644
--- a/README.md
+++ b/README.md
@@ -1,3 +1,4 @@
+Updated documentation.
"""


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_patch(path: Path, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_patch_text(kind), encoding="utf-8")


def _create_generation(
    run_dir: Path,
    genid: str | int,
    parent_genid: str | int | None,
    domain_scores: dict[str, dict[str, float]],
    patch_kind: str | None = None,
    metadata_extra: dict[str, object] | None = None,
) -> None:
    gen_dir = run_dir / f"gen_{genid}"
    metadata_extra = metadata_extra or {}
    curr_patch_files: list[str] = []
    if patch_kind:
        patch_path = gen_dir / "agent_output" / "model_patch.diff"
        _write_patch(patch_path, patch_kind)
        curr_patch_files = [str(patch_path.resolve())]

    metadata = {
        "gen_output_dir": str(gen_dir.resolve()),
        "current_genid": genid,
        "parent_genid": parent_genid,
        "prev_patch_files": [],
        "curr_patch_files": curr_patch_files,
        "run_eval": True,
        "run_full_eval": True,
        "valid_parent": True,
        "can_select_next_parent": True,
        "parent_agent_success": True,
    }
    metadata.update(metadata_extra)
    _write_json(gen_dir / "metadata.json", metadata)

    for domain, split_scores in domain_scores.items():
        for split, score in split_scores.items():
            eval_dir = gen_dir / (f"{domain}_eval" if split == "train" else f"{domain}_eval_{split}")
            mae = 0.25 - (score * 0.1) if domain == "imo_grading" else None
            _write_json(eval_dir / "report.json", _report_payload(domain, score, mae=mae))


def _create_run(run_dir: Path, archive: list[str | int], parent_map: dict[str | int, str | int | None], scores: dict[str | int, dict[str, dict[str, float]]], patch_kinds: dict[str | int, str | None]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for genid in archive:
        _create_generation(
            run_dir=run_dir,
            genid=genid,
            parent_genid=parent_map.get(genid),
            domain_scores=scores.get(genid, {}),
            patch_kind=patch_kinds.get(genid),
        )
    archive_entry = {
        "current_genid": archive[-1],
        "archive": archive,
    }
    (run_dir / "archive.jsonl").write_text(json.dumps(archive_entry) + "\n", encoding="utf-8")


def create_meta_transfer_fixture(output_dir: str | Path) -> dict[str, str]:
    """Create a small, fully local fixture for smoke tests."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_root = output_dir / "source_runs"
    target_root = output_dir / "target_runs"
    selector_root = output_dir / "selector_runs"

    source_archive = ["initial", 1, 2, 3, 4]
    source_parent_map = {"initial": None, 1: "initial", 2: 1, 3: 1, 4: 2}
    source_scores_a = {
        "initial": {"paper_review": {"train": 0.0, "val": 0.0, "test": 0.0}, "genesis_go2walking": {"train": 0.06, "test": 0.06}},
        1: {"paper_review": {"train": 0.44, "val": 0.42, "test": 0.45}, "genesis_go2walking": {"train": 0.10, "test": 0.12}},
        2: {"paper_review": {"train": 0.61, "val": 0.59, "test": 0.60}, "genesis_go2walking": {"train": 0.39, "test": 0.41}},
        3: {"paper_review": {"train": 0.47, "val": 0.45, "test": 0.46}, "genesis_go2walking": {"train": 0.21, "test": 0.25}},
        4: {"paper_review": {"train": 0.68, "val": 0.65, "test": 0.66}, "genesis_go2walking": {"train": 0.52, "test": 0.54}},
    }
    source_scores_b = {
        "initial": {"paper_review": {"train": 0.0, "val": 0.0, "test": 0.0}, "genesis_go2walking": {"train": 0.06, "test": 0.06}},
        1: {"paper_review": {"train": 0.30, "val": 0.29, "test": 0.31}, "genesis_go2walking": {"train": 0.11, "test": 0.12}},
        2: {"paper_review": {"train": 0.34, "val": 0.33, "test": 0.35}, "genesis_go2walking": {"train": 0.18, "test": 0.19}},
        3: {"paper_review": {"train": 0.36, "val": 0.34, "test": 0.37}, "genesis_go2walking": {"train": 0.24, "test": 0.26}},
        4: {"paper_review": {"train": 0.39, "val": 0.37, "test": 0.40}, "genesis_go2walking": {"train": 0.28, "test": 0.30}},
    }
    patch_kinds_source = {"initial": None, 1: "task", 2: "meta_memory", 3: "search", 4: "eval"}

    run_source_a = source_root / "generate_smoke_source_a"
    run_source_b = source_root / "generate_smoke_source_b"
    _create_run(run_source_a, source_archive, source_parent_map, source_scores_a, patch_kinds_source)
    _create_run(run_source_b, source_archive, source_parent_map, source_scores_b, patch_kinds_source)

    mode_scores = {
        "initial_baseline": [0.00, 0.04, 0.08],
        "full_transfer": [0.18, 0.41, 0.62],
        "meta_only_transfer": [0.15, 0.33, 0.55],
        "task_only_transfer": [0.03, 0.11, 0.18],
        "search_only_transfer": [0.02, 0.07, 0.12],
        "memory_only_transfer": [0.10, 0.26, 0.42],
        "random_source_transfer": [0.06, 0.17, 0.25],
    }
    mode_patch_map = {
        "initial_baseline": {1: "task", 2: "task"},
        "full_transfer": {1: "meta_memory", 2: "task"},
        "meta_only_transfer": {1: "meta_memory", 2: "eval"},
        "task_only_transfer": {1: "task", 2: "task"},
        "search_only_transfer": {1: "search", 2: "search"},
        "memory_only_transfer": {1: "meta_memory", 2: "meta_memory"},
        "random_source_transfer": {1: "task", 2: "search"},
    }
    target_parent_map = {"initial": None, 1: "initial", 2: 1}
    target_archive = ["initial", 1, 2]

    target_runs: dict[str, str] = {}
    for mode, series in mode_scores.items():
        run_dir = target_root / f"generate_smoke_{mode}"
        score_map = {
            "initial": {"imo_grading": {"train": 0.0, "val": 0.0, "test": series[0]}},
            1: {"imo_grading": {"train": series[1], "val": max(series[1] - 0.02, 0.0), "test": series[1]}},
            2: {"imo_grading": {"train": series[2], "val": max(series[2] - 0.02, 0.0), "test": series[2]}},
        }
        patch_map = {"initial": None, **mode_patch_map[mode]}
        _create_run(run_dir, target_archive, target_parent_map, score_map, patch_map)
        target_runs[mode] = str(run_dir.resolve())

    selector_scores = {
        "best_score": 0.41,
        "descendant_growth": 0.62,
        "random_valid": 0.25,
        "meta_patch_density": 0.55,
        "hybrid_growth_meta": 0.64,
        "diversity_aware": 0.57,
    }
    selector_runs: dict[str, str] = {}
    for selector, best_score in selector_scores.items():
        run_dir = selector_root / f"generate_smoke_selector_{selector}"
        score_map = {
            "initial": {"imo_grading": {"train": 0.0, "val": 0.0, "test": 0.0}},
            1: {"imo_grading": {"train": best_score * 0.55, "val": best_score * 0.5, "test": best_score * 0.55}},
            2: {"imo_grading": {"train": best_score, "val": max(best_score - 0.03, 0.0), "test": best_score}},
        }
        patch_kind = "meta_memory" if selector in {"descendant_growth", "meta_patch_density", "hybrid_growth_meta"} else "task"
        if selector == "random_valid":
            patch_kind = "search"
        if selector == "diversity_aware":
            patch_kind = "eval"
        patch_map = {"initial": None, 1: patch_kind, 2: patch_kind}
        _create_run(run_dir, target_archive, target_parent_map, score_map, patch_map)
        selector_runs[selector] = str(run_dir.resolve())

    manifest = {
        "root": str(output_dir),
        "source_runs": {
            "source_a": str(run_source_a.resolve()),
            "source_b": str(run_source_b.resolve()),
        },
        "target_runs": target_runs,
        "selector_runs": selector_runs,
    }
    _write_json(output_dir / "fixture_manifest.json", manifest)
    return manifest
