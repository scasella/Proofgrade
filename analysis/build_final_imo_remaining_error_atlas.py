from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analysis.direct_imo_utils import ensure_parent_dir, label_distance, load_yaml, normalize_label


DEFAULT_CONFIG = Path("configs/baseline_freeze/final_imo_lock.yaml")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def classify_remaining_error_bucket(
    *,
    gold: str,
    baseline: str,
    gate: str,
    final_winner: str,
    ablation: str,
) -> str:
    gold = normalize_label(gold)
    baseline = normalize_label(baseline)
    gate = normalize_label(gate)
    final_winner = normalize_label(final_winner)
    ablation = normalize_label(ablation)
    final_distance = label_distance(final_winner, gold)
    stable_wrong = baseline == gate == final_winner == ablation and final_distance > 0

    if final_winner == "correct" and gold != "correct":
        return "overgenerous_full_credit"
    if gold == "almost" or final_winner == "almost" or {gold, final_winner} == {"almost", "partial"}:
        return "almost_vs_partial_boundary"
    if stable_wrong and final_distance >= 2:
        return "reasoning_or_comprehension_failure"
    if final_distance == 1:
        return "rubric_ambiguity"
    if stable_wrong or final_distance >= 2:
        return "unlikely_prompt_fix"
    return "rubric_ambiguity"


def build_remaining_error_summary(summary: dict[str, Any]) -> dict[str, Any]:
    baseline_rows = {row["Grading ID"]: row for row in summary["variants"][summary["baseline_variant"]]["pooled_predictions"]}
    gate_rows = {row["Grading ID"]: row for row in summary["variants"][summary["original_gate_variant"]]["pooled_predictions"]}
    final_rows = {row["Grading ID"]: row for row in summary["variants"][summary["final_winner_variant"]]["pooled_predictions"]}
    ablation_rows = {row["Grading ID"]: row for row in summary["variants"][summary["ablation_variant"]]["pooled_predictions"]}

    bucket_counts: Counter[str] = Counter()
    examples_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompt_fixable = 0
    model_limited = 0

    for grading_id, final_row in final_rows.items():
        gold = normalize_label(final_row.get("Reward"))
        final_prediction = normalize_label(final_row.get("prediction"))
        if final_prediction == gold:
            continue
        baseline = normalize_label(baseline_rows[grading_id].get("prediction"))
        gate = normalize_label(gate_rows[grading_id].get("prediction"))
        ablation = normalize_label(ablation_rows[grading_id].get("prediction"))
        bucket = classify_remaining_error_bucket(
            gold=gold,
            baseline=baseline,
            gate=gate,
            final_winner=final_prediction,
            ablation=ablation,
        )
        bucket_counts[bucket] += 1
        if bucket in {"overgenerous_full_credit", "almost_vs_partial_boundary", "rubric_ambiguity"}:
            prompt_fixable += 1
        else:
            model_limited += 1
        examples_by_bucket[bucket].append(
            {
                "grading_id": grading_id,
                "problem_id": final_row.get("Problem ID", ""),
                "ground_truth": gold,
                "baseline_prediction": baseline,
                "guideline_gate_prediction": gate,
                "final_prediction": final_prediction,
                "ablation_prediction": ablation,
                "final_distance": label_distance(final_prediction, gold),
                "guidelines_snippet": final_row.get("Grading guidelines", "")[:220].replace("\n", " "),
                "student_snippet": final_row.get("Response", "")[:260].replace("\n", " "),
            }
        )

    total_errors = sum(bucket_counts.values())
    main_bottleneck = bucket_counts.most_common(1)[0][0] if bucket_counts else "none"
    likely_diminishing_returns = model_limited * 3 >= total_errors * 2 if total_errors else True
    return {
        "total_remaining_errors": total_errors,
        "bucket_counts": dict(bucket_counts),
        "main_bottleneck": main_bottleneck,
        "prompt_fixable_count": prompt_fixable,
        "model_limited_count": model_limited,
        "likely_diminishing_returns": likely_diminishing_returns,
        "examples_by_bucket": dict(examples_by_bucket),
    }


def _write_report(summary: dict[str, Any], report_path: Path) -> None:
    bucket_counts = Counter(summary["bucket_counts"])
    lines = [
        "# Final IMO Ceiling And Remaining Errors",
        "",
        "## Remaining error buckets",
        "",
    ]
    for bucket, count in bucket_counts.most_common():
        lines.append(f"- `{bucket}`: `{count}`")

    lines.extend(
        [
            "",
            "## Direct answers",
            "",
            f"1. What is the main remaining bottleneck? `{summary['main_bottleneck']}`",
            (
                "2. Are the remaining errors mostly prompt-fixable or model-capability-limited? "
                f"`Prompt-fixable: {summary['prompt_fixable_count']}; model-limited or unlikely prompt-fix: {summary['model_limited_count']}`"
            ),
            (
                "3. Is there still obvious room for another prompt-policy iteration, or have we likely hit diminishing returns? "
                + (
                    "`Likely diminishing returns on prompt policy.`"
                    if summary["likely_diminishing_returns"]
                    else "`Some prompt-policy room remains, but the easy gains are mostly gone.`"
                )
            ),
            "",
            "## Representative remaining errors",
            "",
        ]
    )
    for bucket, items in bucket_counts.most_common():
        lines.append(f"### {bucket}")
        for item in summary["examples_by_bucket"][bucket][:3]:
            lines.extend(
                [
                    f"- `{item['grading_id']}` gold `{item['ground_truth']}`; baseline `{item['baseline_prediction']}`, gate `{item['guideline_gate_prediction']}`, final `{item['final_prediction']}`, no-top-end ablation `{item['ablation_prediction']}`",
                    f"- Guidelines snippet: `{item['guidelines_snippet']}`",
                    f"- Student snippet: `{item['student_snippet']}`",
                ]
            )
        lines.append("")

    ensure_parent_dir(report_path)
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final remaining-error atlas for the frozen IMO winner.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to final lock YAML config.")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    ablation_summary = _load_json(Path(config["summary_path"]))
    summary = build_remaining_error_summary(ablation_summary)

    summary_path = Path(config["remaining_error_summary_path"])
    ensure_parent_dir(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2))
    _write_report(summary, Path(config["remaining_error_report_path"]))


if __name__ == "__main__":
    main()
