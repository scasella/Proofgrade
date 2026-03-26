"""Build a normalized lineage dataset from archived HyperAgents runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.lineage import collect_run_records, discover_run_dirs, serialize_score_map
from utils.patch_taxonomy import classify_patch_file


def _merge_patch_labels(patch_files: list[str]) -> dict[str, Any]:
    file_categories: dict[str, set[str]] = {}
    patch_categories: set[str] = set()
    category_line_totals: dict[str, int] = {}
    dominant_categories: list[str] = []
    task_level = False
    meta_level = False
    search_level = False
    utility_only = True

    for patch_file in patch_files:
        path = Path(patch_file)
        if not path.exists():
            continue
        labels = classify_patch_file(path)
        for file_path, categories in labels["file_categories"].items():
            file_categories.setdefault(file_path, set()).update(categories)
        patch_categories.update(labels["patch_categories"])
        for category, value in labels["category_line_totals"].items():
            category_line_totals[category] = category_line_totals.get(category, 0) + value
        if labels["dominant_category"]:
            dominant_categories.append(labels["dominant_category"])
        task_level = task_level or bool(labels["task_level"])
        meta_level = meta_level or bool(labels["meta_level"])
        search_level = search_level or bool(labels["search_level"])
        utility_only = utility_only and bool(labels["utility_only"])

    dominant_category = None
    if category_line_totals:
        dominant_category = max(
            category_line_totals,
            key=lambda category: category_line_totals[category],
        )

    return {
        "file_categories": {
            path: sorted(categories)
            for path, categories in sorted(file_categories.items())
        },
        "patch_categories": sorted(patch_categories),
        "dominant_patch_category": dominant_category,
        "task_level": task_level,
        "meta_level": meta_level,
        "search_level": search_level,
        "utility_only": utility_only,
    }


def _flatten_scores(row: dict[str, Any], score_map: dict[str, dict[str, dict[str, float | None]]]) -> None:
    for domain, domain_scores in score_map.items():
        for split, split_scores in domain_scores.items():
            for score_type, score in split_scores.items():
                row[f"{domain}__{split}__{score_type}_score"] = score


def build_lineage_rows(run_dirs: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        run_data = collect_run_records(run_dir)
        for record in run_data["records"]:
            scores = serialize_score_map(record["scores"])
            patch_labels = _merge_patch_labels(record["curr_patch_files"])
            row = {
                "run_id": record["run_id"],
                "run_dir": record["run_dir"],
                "generation_id": record["generation_id"],
                "parent_id": record["parent_id"],
                "source_domains": json.dumps(record["source_domains"]),
                "depth_in_archive": record["depth"],
                "descendant_count": record["descendant_count"],
                "descendant_growth_score": record["descendant_growth_score"],
                "selection_score": record["selection_score"],
                "compile_success": record["compile_success"],
                "eval_success": record["eval_success"],
                "valid_parent": record["valid_parent"],
                "can_select_next_parent": record["can_select_next_parent"],
                "run_full_eval": record["run_full_eval"],
                "current_patch_files": json.dumps(record["curr_patch_files"]),
                "previous_patch_files": json.dumps(record["prev_patch_files"]),
                "touched_files": json.dumps(record["touched_files"]),
                "patch_size_files": record["patch_stats"]["files"],
                "patch_size_added_lines": record["patch_stats"]["added_lines"],
                "patch_size_removed_lines": record["patch_stats"]["removed_lines"],
                "patch_size_changed_lines": record["patch_stats"]["changed_lines"],
                "train_scores_by_domain": json.dumps({domain: scores.get(domain, {}).get("train", {}) for domain in scores}),
                "val_scores_by_domain": json.dumps({domain: scores.get(domain, {}).get("val", {}) for domain in scores}),
                "test_scores_by_domain": json.dumps({domain: scores.get(domain, {}).get("test", {}) for domain in scores}),
                "patch_categories": json.dumps(patch_labels["patch_categories"]),
                "dominant_patch_category": patch_labels["dominant_patch_category"],
                "task_level": patch_labels["task_level"],
                "meta_level": patch_labels["meta_level"],
                "search_level": patch_labels["search_level"],
                "utility_only": patch_labels["utility_only"],
                "file_categories": json.dumps(patch_labels["file_categories"]),
            }
            _flatten_scores(row, scores)
            rows.append(row)

    return rows


def resolve_run_dirs(args: argparse.Namespace) -> list[str]:
    run_dirs = [str(Path(path).resolve()) for path in args.run_dirs]
    if args.auto_discover:
        for base_dir in args.auto_discover:
            run_dirs.extend(discover_run_dirs(base_dir))
    return sorted(dict.fromkeys(run_dirs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lineage dataset from one or more HyperAgents runs.")
    parser.add_argument("--run_dirs", nargs="*", default=[], help="Explicit run directories to include")
    parser.add_argument(
        "--auto_discover",
        nargs="*",
        default=[],
        help="Directories under which generate_* runs should be discovered",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis/outputs/meta_transfer",
        help="Directory for CSV/JSON outputs",
    )
    args = parser.parse_args()

    run_dirs = resolve_run_dirs(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_lineage_rows(run_dirs)
    dataframe = pd.DataFrame(rows)

    csv_path = output_dir / "lineage_dataset.csv"
    json_path = output_dir / "lineage_dataset.jsonl"
    dataframe.to_csv(csv_path, index=False)
    dataframe.to_json(json_path, orient="records", lines=True)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
