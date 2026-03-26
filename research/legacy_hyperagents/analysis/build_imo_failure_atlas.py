from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path("configs/baseline_freeze/direct_imo_improvement.yaml")
DEFAULT_REPORT = Path("reports/direct_imo_failure_atlas.md")

RISK_MARKERS = {
    "explicit_missing_proof": [
        "it is generally believed",
        "formal proof might be complicated",
        "assuming this holds",
        "we checked",
    ],
    "example_or_empirical_language": [
        "example",
        "examples",
        "we checked",
        "pattern",
        "numerically",
    ],
    "hedging_or_sketch": [
        "we can show",
        "it follows that",
        "without loss of generality",
        "suppose",
        "assume",
    ],
}


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _read_predictions(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "predictions.csv").open() as f:
        return list(csv.DictReader(f))


def _confusion_counts(rows: list[dict[str, str]]) -> Counter[tuple[str, str]]:
    return Counter(
        (
            row["Reward"].strip().lower(),
            row["prediction"].strip().lower(),
        )
        for row in rows
    )


def _marker_hits(text: str) -> list[str]:
    lowered = text.lower()
    hits = []
    for marker_group, markers in RISK_MARKERS.items():
        if any(marker in lowered for marker in markers):
            hits.append(marker_group)
    return hits


def build_failure_atlas(run_dir: Path) -> dict[str, Any]:
    rows = _read_predictions(run_dir)
    confusion = _confusion_counts(rows)
    failures = []
    marker_counter: Counter[str] = Counter()
    for row in rows:
        reward = row["Reward"].strip().lower()
        prediction = row["prediction"].strip().lower()
        if reward == prediction:
            continue
        student_answer = row.get("Response", "")
        markers = _marker_hits(student_answer)
        marker_counter.update(markers)
        failures.append(
            {
                "grading_id": row["Grading ID"],
                "ground_truth": reward,
                "prediction": prediction,
                "problem_id": row.get("Problem ID", ""),
                "markers": markers,
                "guidelines_snippet": row.get("Grading guidelines", "")[:220].replace("\n", " "),
                "student_snippet": student_answer[:260].replace("\n", " "),
            }
        )

    label_dist = Counter(row["prediction"].strip().lower() for row in rows)
    gt_dist = Counter(row["Reward"].strip().lower() for row in rows)
    summary = {
        "run_dir": str(run_dir),
        "total_examples": len(rows),
        "total_failures": len(failures),
        "prediction_distribution": dict(label_dist),
        "ground_truth_distribution": dict(gt_dist),
        "confusion_counts": {
            f"{gold}->{pred}": count for (gold, pred), count in sorted(confusion.items())
        },
        "marker_counts_on_failures": dict(marker_counter),
        "dominant_failure_modes": [
            {
                "mode": "overcredit_incorrect_as_correct",
                "count": confusion.get(("incorrect", "correct"), 0),
            },
            {
                "mode": "overcredit_partial_as_correct",
                "count": confusion.get(("partial", "correct"), 0),
            },
            {
                "mode": "miss_almost_as_correct",
                "count": confusion.get(("almost", "correct"), 0),
            },
            {
                "mode": "overcredit_incorrect_as_partial",
                "count": confusion.get(("incorrect", "partial"), 0),
            },
        ],
        "representative_failures": failures[:8],
    }
    return summary


def _write_report(summary: dict[str, Any], report_path: Path) -> None:
    lines = [
        "# Direct IMO Failure Atlas",
        "",
        "## Summary",
        "",
        f"- Run analyzed: `{summary['run_dir']}`",
        f"- Total examples: `{summary['total_examples']}`",
        f"- Total failures: `{summary['total_failures']}`",
        "",
        "## Dominant failure modes",
        "",
    ]
    for item in summary["dominant_failure_modes"]:
        lines.append(f"- `{item['mode']}`: `{item['count']}`")

    lines.extend(
        [
            "",
            "## Distribution snapshot",
            "",
            f"- Ground-truth labels: `{summary['ground_truth_distribution']}`",
            f"- Predicted labels: `{summary['prediction_distribution']}`",
            "",
            "## Observed failure pattern",
            "",
            "- The dominant problem is over-crediting. The grader gives full credit too often to work that is either partial or outright incorrect.",
            "- The current target path almost never finds the middle labels. On this slice it misses the only `almost` example entirely.",
            "- The most useful direct lever is stricter rubric calibration: only award `correct` when the proof is complete and justified, and downgrade missing cases or unsupported claims.",
            "",
            "## Marker counts on failures",
            "",
        ]
    )
    if summary["marker_counts_on_failures"]:
        for marker, count in sorted(summary["marker_counts_on_failures"].items()):
            lines.append(f"- `{marker}`: `{count}`")
    else:
        lines.append("- No heuristic markers fired.")

    lines.extend(["", "## Representative failures", ""])
    for item in summary["representative_failures"]:
        lines.extend(
            [
                f"### {item['grading_id']}",
                f"- Ground truth: `{item['ground_truth']}`",
                f"- Prediction: `{item['prediction']}`",
                f"- Markers: `{item['markers'] or ['none']}`",
                f"- Guidelines snippet: `{item['guidelines_snippet']}`",
                f"- Student snippet: `{item['student_snippet']}`",
                "",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a failure atlas for direct IMO grading improvement.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to YAML config.")
    parser.add_argument("--run-dir", default=None, help="Optional run directory override.")
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT), help="Markdown report output path.")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    run_dir = Path(args.run_dir) if args.run_dir else Path(config["frozen_baseline_run"])
    summary = build_failure_atlas(run_dir)

    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "failure_atlas.json").write_text(json.dumps(summary, indent=2))
    _write_report(summary, Path(args.report_path))


if __name__ == "__main__":
    main()
