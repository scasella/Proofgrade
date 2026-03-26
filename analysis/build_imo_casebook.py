from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analysis.direct_imo_utils import ensure_parent_dir, label_distance, load_yaml, normalize_label


DEFAULT_CONFIG = Path("configs/baseline_freeze/final_imo_release.yaml")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _prediction_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["Grading ID"]: row for row in rows}


def _change_entry(source: str, baseline_row: dict[str, Any], winner_row: dict[str, Any]) -> dict[str, Any]:
    gold = normalize_label(winner_row.get("Reward"))
    baseline_prediction = normalize_label(baseline_row.get("prediction"))
    winner_prediction = normalize_label(winner_row.get("prediction"))
    baseline_distance = label_distance(baseline_prediction, gold)
    winner_distance = label_distance(winner_prediction, gold)
    return {
        "source": source,
        "grading_id": winner_row["Grading ID"],
        "problem_id": winner_row.get("Problem ID", ""),
        "problem_source": winner_row.get("Problem Source", ""),
        "ground_truth": gold,
        "baseline_prediction": baseline_prediction,
        "winner_prediction": winner_prediction,
        "baseline_distance": baseline_distance,
        "winner_distance": winner_distance,
        "distance_improvement": baseline_distance - winner_distance,
        "guidelines_snippet": winner_row.get("Grading guidelines", "")[:220].replace("\n", " "),
        "student_snippet": winner_row.get("Response", "")[:260].replace("\n", " "),
    }


def _sorted_true_improvements(
    source: str,
    baseline_predictions: list[dict[str, Any]],
    winner_predictions: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    baseline_by_id = _prediction_map(baseline_predictions)
    improvements: list[dict[str, Any]] = []
    for winner_row in winner_predictions:
        baseline_row = baseline_by_id[winner_row["Grading ID"]]
        if normalize_label(baseline_row.get("prediction")) == normalize_label(winner_row.get("prediction")):
            continue
        entry = _change_entry(source, baseline_row, winner_row)
        if entry["winner_distance"] < entry["baseline_distance"]:
            improvements.append(entry)
    improvements.sort(
        key=lambda item: (
            -int(item["distance_improvement"]),
            item["winner_distance"],
            item["grading_id"],
        )
    )
    return improvements[:limit]


def _sorted_regressions(
    source: str,
    baseline_predictions: list[dict[str, Any]],
    winner_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_id = _prediction_map(baseline_predictions)
    regressions: list[dict[str, Any]] = []
    for winner_row in winner_predictions:
        baseline_row = baseline_by_id[winner_row["Grading ID"]]
        if normalize_label(baseline_row.get("prediction")) == normalize_label(winner_row.get("prediction")):
            continue
        entry = _change_entry(source, baseline_row, winner_row)
        if entry["winner_distance"] > entry["baseline_distance"]:
            regressions.append(entry)
    regressions.sort(
        key=lambda item: (
            item["distance_improvement"],
            item["winner_distance"],
            item["grading_id"],
        )
    )
    return regressions


def _remaining_failures(remaining_summary: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    ordered_buckets = sorted(
        remaining_summary["bucket_counts"].items(),
        key=lambda item: (-int(item[1]), item[0]),
    )
    failures: list[dict[str, Any]] = []
    for bucket, _ in ordered_buckets:
        for item in remaining_summary["examples_by_bucket"][bucket]:
            failures.append(
                {
                    "bucket": bucket,
                    "grading_id": item["grading_id"],
                    "problem_id": item.get("problem_id", ""),
                    "ground_truth": item["ground_truth"],
                    "baseline_prediction": item["baseline_prediction"],
                    "winner_prediction": item["final_prediction"],
                    "guideline_gate_prediction": item["guideline_gate_prediction"],
                    "guidelines_snippet": item["guidelines_snippet"],
                    "student_snippet": item["student_snippet"],
                }
            )
            if len(failures) >= limit:
                return failures
    return failures


def build_casebook(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = config["artifacts"]
    lockbox_summary = _load_json(Path(artifacts["lockbox_summary"]))
    remaining_summary = _load_json(Path(artifacts["remaining_error_summary"]))
    fresh_path = Path(artifacts["fresh_generalization_summary"])
    fresh_summary = _load_json(fresh_path) if fresh_path.exists() else None

    lockbox_baseline = lockbox_summary["variants"][config["baseline_variant"]]["predictions"]
    lockbox_winner = lockbox_summary["variants"][config["winner_variant"]]["predictions"]
    lockbox_improvements = _sorted_true_improvements("lockbox_test", lockbox_baseline, lockbox_winner, limit=5)

    fresh_improvements: list[dict[str, Any]] = []
    fresh_regressions: list[dict[str, Any]] = []
    if fresh_summary is not None:
        fresh_baseline = fresh_summary["variants"][config["baseline_variant"]]["predictions"]
        fresh_winner = fresh_summary["variants"][config["winner_variant"]]["predictions"]
        fresh_improvements = _sorted_true_improvements(
            "fresh_generalization",
            fresh_baseline,
            fresh_winner,
            limit=5,
        )
        fresh_regressions = _sorted_regressions("fresh_generalization", fresh_baseline, fresh_winner)

    lockbox_regressions = _sorted_regressions("lockbox_test", lockbox_baseline, lockbox_winner)
    regressions = (lockbox_regressions + fresh_regressions)[:3]
    casebook = {
        "lockbox_true_improvements": lockbox_improvements,
        "fresh_true_improvements": fresh_improvements,
        "remaining_failures": _remaining_failures(remaining_summary, limit=5),
        "regressions": regressions,
    }

    json_path = Path(config["casebook_json_path"])
    ensure_parent_dir(json_path)
    json_path.write_text(json.dumps(casebook, indent=2))

    lines = [
        "# IMO Casebook",
        "",
        "## Lockbox test improvements",
        "",
    ]
    for item in casebook["lockbox_true_improvements"]:
        lines.extend(
            [
                f"- `{item['grading_id']}` `{item['baseline_prediction']}` -> `{item['winner_prediction']}` (gold `{item['ground_truth']}`; source `{item['problem_source']}`)",
                f"- Guideline snippet: `{item['guidelines_snippet']}`",
                f"- Student snippet: `{item['student_snippet']}`",
            ]
        )

    lines.extend(["", "## Fresh generalization improvements", ""])
    if casebook["fresh_true_improvements"]:
        for item in casebook["fresh_true_improvements"]:
            lines.extend(
                [
                    f"- `{item['grading_id']}` `{item['baseline_prediction']}` -> `{item['winner_prediction']}` (gold `{item['ground_truth']}`; source `{item['problem_source']}`)",
                    f"- Guideline snippet: `{item['guidelines_snippet']}`",
                    f"- Student snippet: `{item['student_snippet']}`",
                ]
            )
    else:
        lines.append("- No fresh generalization improvements recorded yet.")

    lines.extend(["", "## Remaining frozen-winner failures", ""])
    for item in casebook["remaining_failures"]:
        lines.extend(
            [
                f"- `{item['grading_id']}` bucket `{item['bucket']}`; gold `{item['ground_truth']}`; baseline `{item['baseline_prediction']}`; winner `{item['winner_prediction']}`",
                f"- Guideline snippet: `{item['guidelines_snippet']}`",
                f"- Student snippet: `{item['student_snippet']}`",
            ]
        )

    lines.extend(["", "## Regressions", ""])
    if casebook["regressions"]:
        for item in casebook["regressions"]:
            lines.extend(
                [
                    f"- `{item['grading_id']}` on `{item['source']}`: `{item['baseline_prediction']}` -> `{item['winner_prediction']}` (gold `{item['ground_truth']}`)",
                    f"- Guideline snippet: `{item['guidelines_snippet']}`",
                    f"- Student snippet: `{item['student_snippet']}`",
                ]
            )
    else:
        lines.append("- No regressions.")

    md_path = Path(config["casebook_md_path"])
    ensure_parent_dir(md_path)
    md_path.write_text("\n".join(lines) + "\n")
    return casebook


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact casebook for the locked IMO result.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to the final IMO release config.")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    build_casebook(config)


if __name__ == "__main__":
    main()
