from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

from analysis.direct_imo_utils import changed_examples, label_distance, load_yaml, normalize_label, prediction_correct, read_csv_rows
from utils.common import extract_jsons


DEFAULT_CONFIG = Path("configs/baseline_freeze/direct_imo_robustness.yaml")
DEFAULT_REPORT = Path("reports/direct_imo_error_boundary.md")
DEFAULT_SUMMARY = Path("analysis/outputs/direct_imo_robustness/error_boundary_summary.json")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _parse_last_output_text(chat_history_path: Path) -> str | None:
    if not chat_history_path.exists():
        return None
    outputs: list[str] = []
    for line in chat_history_path.read_text().splitlines():
        if not line.startswith("Output: "):
            continue
        payload = line[len("Output: ") :].strip()
        try:
            outputs.append(ast.literal_eval(payload))
        except Exception:
            continue
    return outputs[-1] if outputs else None


def _extract_json_object(raw_text: str | None) -> dict[str, Any] | None:
    if not raw_text:
        return None
    extracted = extract_jsons(raw_text) or []
    if extracted:
        last = extracted[-1]
        if isinstance(last, dict):
            return last
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw_text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw_text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _winner_metadata(run_dir: Path, grading_id: str) -> dict[str, Any]:
    raw = _parse_last_output_text(run_dir / "agent_evals" / f"chat_history_{grading_id}.md")
    obj = _extract_json_object(raw)
    if not obj:
        return {}
    return {
        "matched_guideline": obj.get("matched_guideline"),
        "rationale": obj.get("rationale"),
        "decision_basis": obj.get("decision_basis"),
        "missing_piece": obj.get("missing_piece"),
    }


def classify_boundary_bucket(
    ground_truth: str,
    baseline_prediction: str,
    winner_prediction: str,
) -> str:
    gold = normalize_label(ground_truth)
    baseline = normalize_label(baseline_prediction)
    winner = normalize_label(winner_prediction)
    base_dist = label_distance(baseline, gold)
    winner_dist = label_distance(winner, gold)

    if winner_dist > base_dist:
        return "winner_regression"
    if winner == "correct" and gold != "correct":
        return "overcredit_correct"
    if gold == "almost" or baseline == "almost" or winner == "almost":
        return "almost_boundary"
    if winner_dist < base_dist and winner_dist > 0:
        return "right_direction_not_far_enough"
    if gold == "partial" or baseline == "partial" or winner == "partial":
        return "partial_boundary"
    if winner_dist == 0:
        return "winner_fix"
    return "stubborn_same_wrong"


def infer_error_causes(bucket: str, ground_truth: str, baseline_prediction: str, winner_prediction: str) -> list[str]:
    causes: list[str] = []
    gold = normalize_label(ground_truth)
    baseline = normalize_label(baseline_prediction)
    winner = normalize_label(winner_prediction)
    if gold == "almost":
        causes.append("class_imbalance_rare_case")
    if bucket in {"almost_boundary", "partial_boundary"}:
        causes.append("rubric_boundary")
    if bucket in {"overcredit_correct", "right_direction_not_far_enough"}:
        causes.append("prompt_policy_calibration")
    if bucket == "stubborn_same_wrong" and baseline == winner:
        causes.append("model_reasoning_limit_candidate")
    if not causes:
        causes.append("prompt_policy_calibration")
    return causes


def build_error_boundary_summary(robustness_summary: dict[str, Any]) -> dict[str, Any]:
    baseline_variant = robustness_summary["baseline_variant"]
    winner_variant = robustness_summary["reference_variant"]
    contrast_variant = robustness_summary.get("contrast_variant")

    baseline_rows = robustness_summary["variants"][baseline_variant]["pooled_predictions"]
    winner_rows = robustness_summary["variants"][winner_variant]["pooled_predictions"]
    contrast_rows = (
        robustness_summary["variants"][contrast_variant]["pooled_predictions"]
        if contrast_variant and contrast_variant in robustness_summary["variants"]
        else []
    )
    winner_shard_dirs = {
        item["name"]: Path(item["run_dir"])
        for item in robustness_summary["variants"][winner_variant]["shards"]
    }
    contrast_by_id = {row["Grading ID"]: row for row in contrast_rows}
    winner_by_id = {row["Grading ID"]: row for row in winner_rows}

    examples: list[dict[str, Any]] = []
    bucket_counts: Counter[str] = Counter()
    cause_counts: Counter[str] = Counter()
    changed_vs_baseline = changed_examples(baseline_rows, winner_rows)
    remaining_errors = 0
    almost_related_errors = 0
    overcredit_errors = 0
    for shard in robustness_summary["variants"][winner_variant]["shards"]:
        shard_rows = read_csv_rows(Path(shard["run_dir"]) / "predictions.csv")
        for row in shard_rows:
            qid = row["Grading ID"]
            gold = normalize_label(row["Reward"])
            winner = normalize_label(row["prediction"])
            baseline = normalize_label(next(item for item in baseline_rows if item["Grading ID"] == qid)["prediction"])
            contrast = normalize_label(contrast_by_id.get(qid, {}).get("prediction"))
            base_correct = prediction_correct(baseline, gold)
            win_correct = prediction_correct(winner, gold)
            bucket = classify_boundary_bucket(gold, baseline, winner)
            causes = infer_error_causes(bucket, gold, baseline, winner)
            metadata = _winner_metadata(winner_shard_dirs[shard["name"]], qid)
            example = {
                "grading_id": qid,
                "problem_id": row.get("Problem ID", ""),
                "ground_truth": gold,
                "baseline_prediction": baseline,
                "winner_prediction": winner,
                "contrast_prediction": contrast,
                "baseline_distance": label_distance(baseline, gold),
                "winner_distance": label_distance(winner, gold),
                "changed": baseline != winner,
                "moved_in_right_direction": label_distance(winner, gold) < label_distance(baseline, gold),
                "regressed": label_distance(winner, gold) > label_distance(baseline, gold),
                "bucket": bucket,
                "causes": causes,
                "matched_guideline": metadata.get("matched_guideline"),
                "winner_rationale": metadata.get("rationale"),
                "guidelines_snippet": row.get("Grading guidelines", "")[:220].replace("\n", " "),
                "student_snippet": row.get("Response", "")[:260].replace("\n", " "),
            }
            examples.append(example)
            if not win_correct:
                remaining_errors += 1
                bucket_counts[bucket] += 1
                cause_counts.update(causes)
                if bucket == "almost_boundary" or gold == "almost":
                    almost_related_errors += 1
                if bucket == "overcredit_correct":
                    overcredit_errors += 1

    primary_boundary = bucket_counts.most_common(1)[0][0] if bucket_counts else "none"
    robustness_passed = robustness_summary.get("robustness_gate", {}).get("passes", False)
    prompt_fixable = sum(bucket_counts[b] for b in ("overcredit_correct", "almost_boundary", "partial_boundary", "right_direction_not_far_enough"))
    stubborn = bucket_counts.get("stubborn_same_wrong", 0)
    recommended_followup = robustness_passed and prompt_fixable >= stubborn and prompt_fixable > 0

    recommended_variants: list[str] = []
    if recommended_followup:
        if bucket_counts.get("overcredit_correct", 0) >= bucket_counts.get("almost_boundary", 0):
            recommended_variants.append("guideline_gate_fatal_flaw_v1")
        if bucket_counts.get("almost_boundary", 0) > 0 or bucket_counts.get("partial_boundary", 0) > 0:
            recommended_variants.append("guideline_gate_almost_boundary_v1")
        recommended_variants = recommended_variants[:2]

    return {
        "robustness_passed": robustness_passed,
        "changed_vs_baseline_count": changed_vs_baseline["changed_count"],
        "remaining_error_count": remaining_errors,
        "bucket_counts": dict(bucket_counts),
        "cause_counts": dict(cause_counts),
        "primary_boundary": primary_boundary,
        "almost_related_error_count": almost_related_errors,
        "overcredit_error_count": overcredit_errors,
        "recommended_followup": recommended_followup,
        "recommended_followup_variants": recommended_variants,
        "followup_reason": (
            "Remaining errors still cluster on prompt-sensitive rubric boundaries."
            if recommended_followup
            else "Remaining errors are mostly stubborn enough that more prompt policy work is unlikely to pay off."
        ),
        "examples": examples,
    }


def _write_report(summary: dict[str, Any], report_path: Path, robustness_summary: dict[str, Any]) -> None:
    bucket_counts = Counter(summary["bucket_counts"])
    cause_counts = Counter(summary["cause_counts"])
    lines = [
        "# Direct IMO Error Boundary",
        "",
        "## Robustness status",
        "",
        f"- Robustness gate passed: `{summary['robustness_passed']}`",
        f"- Changed predictions vs baseline across pooled validation: `{summary['changed_vs_baseline_count']}`",
        f"- Remaining winner errors: `{summary['remaining_error_count']}`",
        "",
        "## Remaining error concentration",
        "",
    ]
    for bucket, count in bucket_counts.most_common():
        lines.append(f"- `{bucket}`: `{count}`")

    lines.extend(["", "## Likely causes", ""])
    for cause, count in cause_counts.most_common():
        lines.append(f"- `{cause}`: `{count}`")

    lines.extend(
        [
            "",
            "## Direct answers",
            "",
            f"1. Is the current gain robust? `{'Yes' if summary['robustness_passed'] else 'Not strongly enough'}`",
            f"2. Where are the remaining errors concentrated? `Primary bucket: {summary['primary_boundary']}`",
            (
                "3. Are `almost` cases truly the next bottleneck, or is that just anecdotal? "
                f"`Almost-related errors account for {summary['almost_related_error_count']} of {summary['remaining_error_count']} remaining winner errors.`"
            ),
            (
                "4. Are the remaining errors likely fixable by prompt/rubric policy, or mostly model-capability errors? "
                f"`Prompt-sensitive buckets: {sum(bucket_counts[b] for b in ('overcredit_correct','almost_boundary','partial_boundary','right_direction_not_far_enough'))}; stubborn same-wrong: {bucket_counts.get('stubborn_same_wrong', 0)}.`"
            ),
            "",
            "## Representative changed or still-wrong examples",
            "",
        ]
    )
    selected = [
        item for item in summary["examples"]
        if item["changed"] or item["winner_distance"] > 0
    ][:12]
    for item in selected:
        lines.extend(
            [
                f"### {item['grading_id']}",
                f"- Ground truth: `{item['ground_truth']}`",
                f"- Baseline -> winner: `{item['baseline_prediction']}` -> `{item['winner_prediction']}`",
                f"- Contrast: `{item['contrast_prediction']}`",
                f"- Bucket: `{item['bucket']}`",
                f"- Causes: `{item['causes']}`",
                f"- Winner matched guideline: `{item.get('matched_guideline')}`",
                f"- Winner rationale: `{item.get('winner_rationale')}`",
                f"- Guidelines snippet: `{item['guidelines_snippet']}`",
                f"- Student snippet: `{item['student_snippet']}`",
                "",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an IMO grading error-boundary atlas.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to robustness config.")
    parser.add_argument("--robustness-summary", default=None, help="Optional robustness summary path override.")
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT), help="Markdown report path.")
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY), help="JSON summary path.")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    robustness_summary_path = Path(args.robustness_summary or config["summary_path"])
    robustness_summary = _load_json(robustness_summary_path)
    summary = build_error_boundary_summary(robustness_summary)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(summary, report_path, robustness_summary)

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
