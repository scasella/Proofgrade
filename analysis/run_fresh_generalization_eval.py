from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from analysis.direct_imo_utils import (
    IMO_LABEL_ORDER,
    bootstrap_delta_summary,
    changed_examples,
    ensure_parent_dir,
    evaluate_imo_rows,
    label_distance,
    load_yaml,
    normalize_label,
    pooled_metrics,
    read_csv_rows,
    write_csv_rows,
    QUESTION_ID,
    build_imo_report,
    grade_imo_row,
)
from proofgrade.config import RuntimeSettings


DEFAULT_CONFIG = Path("configs/baseline_freeze/fresh_generalization_eval.yaml")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def build_fresh_generalization_set(config: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    base_rows = read_csv_rows(Path(config["base_dataset_path"]))
    used_rows: list[dict[str, str]] = []
    used_ids: set[str] = set()
    used_problem_ids: set[str] = set()
    for path_str in config["exclude_dataset_paths"]:
        rows = read_csv_rows(Path(path_str))
        used_rows.extend(rows)
        for row in rows:
            used_ids.add(row["Grading ID"])
            used_problem_ids.add(row["Problem ID"])

    remaining = [dict(row) for row in base_rows if row["Grading ID"] not in used_ids]
    remaining_ids = {row["Grading ID"] for row in remaining}
    remaining_problem_ids = {row["Problem ID"] for row in remaining}
    expected_fresh_size = int(config["expected_fresh_size"])
    if len(remaining) != expected_fresh_size:
        raise ValueError(f"Expected {expected_fresh_size} fresh rows, found {len(remaining)}.")
    if remaining_ids & used_ids:
        raise ValueError("Fresh generalization set overlaps with used benchmark IDs.")

    metadata = {
        "base_total": len(base_rows),
        "used_total": len(used_ids),
        "remaining_total": len(remaining),
        "used_overlap_with_base": len({row['Grading ID'] for row in base_rows} & used_ids),
        "used_not_in_base": len(used_ids - {row['Grading ID'] for row in base_rows}),
        "used_problem_ids": len(used_problem_ids),
        "remaining_problem_ids": len(remaining_problem_ids),
        "problem_id_overlap": len(remaining_problem_ids & used_problem_ids),
        "problem_id_only_remaining": len(remaining_problem_ids - used_problem_ids),
        "problem_id_only_used": len(used_problem_ids - remaining_problem_ids),
        "remaining_label_counts": dict(Counter(normalize_label(row["Reward"]) for row in remaining)),
        "remaining_problem_source_counts": dict(Counter(row["Problem Source"] for row in remaining)),
        "used_label_counts": dict(Counter(normalize_label(row["Reward"]) for row in used_rows)),
        "used_problem_source_counts": dict(Counter(row["Problem Source"] for row in used_rows)),
    }
    return remaining, metadata


def _run_variant(config: dict[str, Any], variant: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    output_root = Path(config["output_root"]) / variant["id"]
    run_id = f"{variant['id']}_fresh_filtered_remaining_{len(rows)}"
    num_workers = int(config.get("num_workers", 1))
    save_interval = int(config.get("save_interval", 10))
    if num_workers <= 1:
        run = evaluate_imo_rows(
            rows=rows,
            output_root=output_root,
            run_id=run_id,
            model=config["model"],
            variant_env=config["variant_env"],
            variant_id=variant["env_value"],
            save_interval=save_interval,
        )
    else:
        run = _evaluate_imo_rows_parallel(
            rows=rows,
            output_root=output_root,
            run_id=run_id,
            model=config["model"],
            variant_env=config["variant_env"],
            variant_id=variant["env_value"],
            num_workers=num_workers,
            save_interval=save_interval,
            max_attempts=int(config.get("max_attempts_per_example", 5)),
            retry_sleep_seconds=float(config.get("retry_sleep_seconds", 2)),
        )
    return {
        "id": variant["id"],
        "role": variant["role"],
        "hypothesis": variant.get("hypothesis", ""),
        "run_dir": run["run_dir"],
        "pooled": pooled_metrics(run["predictions"]),
        "predictions": run["predictions"],
        "report": run["report"],
    }


def _evaluate_imo_rows_parallel(
    *,
    rows: list[dict[str, str]],
    output_root: Path,
    run_id: str,
    model: str,
    variant_env: str,
    variant_id: str,
    num_workers: int,
    save_interval: int,
    max_attempts: int,
    retry_sleep_seconds: float,
) -> dict[str, Any]:
    run_dir = output_root / run_id
    predictions_path = run_dir / "predictions.csv"
    report_path = run_dir / "report.json"
    if predictions_path.exists() and report_path.exists():
        with report_path.open() as f:
            metrics = json.load(f)
        return {
            "run_dir": str(run_dir),
            "report": metrics,
            "predictions": read_csv_rows(predictions_path),
        }

    fieldnames = list(rows[0].keys())
    extra_fields = [
        "prediction",
        "rationale",
        "matched_guideline",
        "parse_source",
        "prompt_variant",
        "model_provider",
        "model_name",
        "version",
        "request_id",
    ]
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    row_index_by_id = {row[QUESTION_ID]: index for index, row in enumerate(rows)}
    settings = RuntimeSettings(model=model, prompt_variant=variant_id)

    def _evaluate_one(index: int, row: dict[str, str]) -> tuple[int, dict[str, Any]]:
        return index, grade_imo_row(row, settings=settings)

    results_by_index: dict[int, dict[str, Any]] = {}
    if predictions_path.exists():
        for existing_row in read_csv_rows(predictions_path):
            qid = existing_row.get(QUESTION_ID)
            if qid not in row_index_by_id:
                continue
            prediction = existing_row.get("prediction")
            if prediction is None or str(prediction).strip() == "":
                continue
            results_by_index[row_index_by_id[qid]] = existing_row

    completed = len(results_by_index)
    pending_items = [
        (index, row)
        for index, row in enumerate(rows)
        if index not in results_by_index
    ]
    attempts_by_index = {index: 0 for index, _ in pending_items}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_item: dict[Any, tuple[int, dict[str, str]]] = {}

        def _submit(index: int, row: dict[str, str]) -> None:
            attempts_by_index[index] += 1
            future = executor.submit(_evaluate_one, index, row)
            future_to_item[future] = (index, row)

        for index, row in pending_items:
            _submit(index, row)

        while future_to_item:
            future = next(as_completed(list(future_to_item.keys())))
            index, row = future_to_item.pop(future)
            try:
                index, result_row = future.result()
            except Exception:
                if attempts_by_index[index] < max_attempts:
                    time.sleep(retry_sleep_seconds)
                    _submit(index, row)
                    continue
                raise
            results_by_index[index] = result_row
            completed += 1
            if completed % save_interval == 0:
                ordered = [results_by_index[i] for i in sorted(results_by_index)]
                write_csv_rows(predictions_path, ordered, fieldnames)

    ordered_results = [results_by_index[i] for i in range(len(rows))]
    write_csv_rows(predictions_path, ordered_results, fieldnames)
    report_dict = build_imo_report(ordered_results)
    report_path.write_text(json.dumps(report_dict, indent=2))
    return {
        "run_dir": str(run_dir),
        "report": report_dict,
        "predictions": ordered_results,
    }


def _group_metric_breakdown(rows: list[dict[str, str]], field_name: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[field_name]].append(row)
    ordered_keys = list(grouped.keys())
    if field_name == "Reward":
        ordered_keys = [label for label in IMO_LABEL_ORDER if label in grouped]
    else:
        ordered_keys = sorted(ordered_keys, key=lambda key: (-len(grouped[key]), key))
    return {key: pooled_metrics(grouped[key]) for key in ordered_keys}


def classify_fresh_error_bucket(*, gold: str, baseline: str, winner: str) -> str:
    gold = normalize_label(gold)
    baseline = normalize_label(baseline)
    winner = normalize_label(winner)
    winner_distance = label_distance(winner, gold)
    stable_wrong = baseline == winner and winner_distance > 0
    if winner == "correct" and gold != "correct":
        return "overgenerous_full_credit"
    if gold == "almost" or winner == "almost" or {gold, winner} == {"almost", "partial"}:
        return "almost_vs_partial_boundary"
    if stable_wrong and winner_distance >= 2:
        return "reasoning_or_comprehension_failure"
    if winner_distance == 1:
        return "rubric_ambiguity"
    return "unlikely_prompt_fix"


def _winner_error_bucket_summary(
    baseline_predictions: list[dict[str, str]],
    winner_predictions: list[dict[str, str]],
) -> dict[str, Any]:
    baseline_by_id = {row["Grading ID"]: row for row in baseline_predictions}
    bucket_counts: Counter[str] = Counter()
    examples_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for winner_row in winner_predictions:
        gold = normalize_label(winner_row.get("Reward"))
        winner_prediction = normalize_label(winner_row.get("prediction"))
        if winner_prediction == gold:
            continue
        baseline_row = baseline_by_id[winner_row["Grading ID"]]
        baseline_prediction = normalize_label(baseline_row.get("prediction"))
        bucket = classify_fresh_error_bucket(
            gold=gold,
            baseline=baseline_prediction,
            winner=winner_prediction,
        )
        bucket_counts[bucket] += 1
        examples_by_bucket[bucket].append(
            {
                "grading_id": winner_row["Grading ID"],
                "problem_id": winner_row.get("Problem ID", ""),
                "problem_source": winner_row.get("Problem Source", ""),
                "ground_truth": gold,
                "baseline_prediction": baseline_prediction,
                "winner_prediction": winner_prediction,
                "guidelines_snippet": winner_row.get("Grading guidelines", "")[:220].replace("\n", " "),
                "student_snippet": winner_row.get("Response", "")[:260].replace("\n", " "),
            }
        )
    return {
        "bucket_counts": dict(bucket_counts),
        "examples_by_bucket": dict(examples_by_bucket),
        "main_bottleneck": bucket_counts.most_common(1)[0][0] if bucket_counts else "none",
    }


def classify_generalization_scale(
    *,
    lockbox_accuracy_delta: float,
    lockbox_mae_delta: float,
    fresh_accuracy_delta: float,
    fresh_mae_delta: float,
) -> str:
    if fresh_accuracy_delta <= 0.0 or fresh_mae_delta >= 0.0:
        return "absent"
    if lockbox_accuracy_delta <= 0.0 or lockbox_mae_delta >= 0.0:
        return "smaller"
    accuracy_ratio = fresh_accuracy_delta / lockbox_accuracy_delta if lockbox_accuracy_delta else 0.0
    mae_ratio = abs(fresh_mae_delta) / abs(lockbox_mae_delta) if lockbox_mae_delta else 0.0
    if accuracy_ratio >= 0.65 and mae_ratio >= 0.65:
        return "similar"
    return "smaller"


def _same_error_type_as_lockbox(
    fresh_buckets: dict[str, int],
    remaining_summary: dict[str, Any],
) -> bool:
    if not fresh_buckets:
        return False
    fresh_top = max(fresh_buckets.items(), key=lambda item: (item[1], item[0]))[0]
    previous_top = remaining_summary["main_bottleneck"]
    return fresh_top == previous_top


def _write_plan_report(config: dict[str, Any], metadata: dict[str, Any]) -> None:
    source_counts = sorted(
        metadata["remaining_problem_source_counts"].items(),
        key=lambda item: (-int(item[1]), item[0]),
    )
    lines = [
        "# Fresh Generalization Plan",
        "",
        "## Chosen route",
        "",
        "- Route: `A`",
        "- Dataset: `domains/imo/gradingbench_filtered.csv`",
        "- Fresh-set construction: remove every `Grading ID` already used in the 100-example train, validation, and test splits.",
        f"- Resulting fresh set size: `{metadata['remaining_total']}`",
        f"- Note: `{config['fresh_set_note']}`",
        "",
        "## Why this is the best available next check",
        "",
        "- It uses real labeled grading examples already present in the repo.",
        "- It is stronger than an alternate-evaluator check because it adds new judged responses, not just a new scorer.",
        "- It avoids new challenge-set curation decisions.",
        "- It stays inside the same filtered benchmark regime instead of expanding to excluded rows with different curation status.",
        "",
        "## Important caveat",
        "",
        f"- The fresh set uses unseen responses, but not unseen problem IDs: problem-ID overlap with the development line is `{metadata['problem_id_overlap']}` out of `{metadata['remaining_problem_ids']}` fresh problem IDs.",
        "",
        "## Fresh-set makeup",
        "",
        f"- Remaining label counts: `{metadata['remaining_label_counts']}`",
        f"- Remaining problem-source counts: `{dict(source_counts[:8])}`",
        "",
        "## Rejected alternatives",
        "",
        "- Curated challenge set: would add new selection choices and another tuning temptation.",
        "- Alternate evaluator/model robustness only: weaker than using new labeled responses already available here.",
        "- Other IMO datasets in this repo: not the same direct grading task with comparable labels.",
        "",
    ]
    report_path = Path(config["plan_report_path"])
    ensure_parent_dir(report_path)
    report_path.write_text("\n".join(lines) + "\n")


def _write_result_report(
    config: dict[str, Any],
    summary: dict[str, Any],
    lockbox_summary: dict[str, Any],
    previous_remaining_summary: dict[str, Any],
) -> None:
    baseline = summary["variants"][config["baseline_variant"]]
    winner = summary["variants"][config["winner_variant"]]
    delta = summary["winner_vs_baseline"]
    same_error_type = _same_error_type_as_lockbox(
        summary["winner_error_buckets"]["bucket_counts"],
        previous_remaining_summary,
    )
    lines = [
        "# Fresh Generalization Result",
        "",
        "## Exact command",
        "",
        "- `GEMINI_API_KEY=... GOOGLE_API_KEY=... PYTHONPATH=. .venv/bin/python analysis/run_fresh_generalization_eval.py --config configs/baseline_freeze/fresh_generalization_eval.yaml`",
        "",
        "## Fresh-set summary",
        "",
        f"- Fresh examples: `{summary['fresh_set_metadata']['remaining_total']}`",
        f"- Fresh label counts: `{summary['fresh_set_metadata']['remaining_label_counts']}`",
        f"- Fresh problem-source counts: `{summary['fresh_set_metadata']['remaining_problem_source_counts']}`",
        f"- Problem-ID overlap with the benchmark line: `{summary['fresh_set_metadata']['problem_id_overlap']}`",
        "",
        "## Baseline vs frozen winner",
        "",
        f"- Baseline accuracy: `{baseline['pooled']['overall_accuracy']:.3f}`",
        f"- Winner accuracy: `{winner['pooled']['overall_accuracy']:.3f}`",
        f"- Baseline normalized grading error: `{baseline['pooled']['normalized_mean_absolute_error']:.3f}`",
        f"- Winner normalized grading error: `{winner['pooled']['normalized_mean_absolute_error']:.3f}`",
        f"- Baseline valid-label rate: `{baseline['pooled']['valid_label_rate']:.3f}`",
        f"- Winner valid-label rate: `{winner['pooled']['valid_label_rate']:.3f}`",
        f"- Accuracy delta: `{delta['bootstrap']['accuracy_delta_mean']:+.3f}`",
        f"- Accuracy delta 95% bootstrap CI: `{delta['bootstrap']['accuracy_delta_ci']}`",
        f"- Normalized grading error delta: `{delta['bootstrap']['mae_delta_mean']:+.3f}`",
        f"- Normalized grading error delta 95% bootstrap CI: `{delta['bootstrap']['mae_delta_ci']}`",
        f"- Changed predictions vs baseline: `{delta['changed']['changed_count']}`",
        f"- Better `{delta['changed']['better']}`, worse `{delta['changed']['worse']}`, same-score changes `{delta['changed']['same_score']}`",
        "",
        "## Comparison to the lockbox test",
        "",
        f"- Lockbox accuracy delta: `{lockbox_summary['winner_vs_baseline']['bootstrap']['accuracy_delta_mean']:+.3f}`",
        f"- Fresh accuracy delta: `{delta['bootstrap']['accuracy_delta_mean']:+.3f}`",
        f"- Lockbox error delta: `{lockbox_summary['winner_vs_baseline']['bootstrap']['mae_delta_mean']:+.3f}`",
        f"- Fresh error delta: `{delta['bootstrap']['mae_delta_mean']:+.3f}`",
        f"- Generalization scale: `{summary['generalization_scale']}`",
        "",
        "## Remaining winner error types on the fresh set",
        "",
        f"- Main fresh bottleneck: `{summary['winner_error_buckets']['main_bottleneck']}`",
        f"- Fresh winner error buckets: `{summary['winner_error_buckets']['bucket_counts']}`",
        (
            f"- Same broad error type as the locked benchmark line: `{'Yes' if same_error_type else 'No'}`"
        ),
        "",
        "## Interpretation",
        "",
        f"- Did the frozen winner generalize? `{summary['generalized']}`",
        f"- Was the gain similar, smaller, or absent? `{summary['generalization_scale']}`",
        f"- Were the remaining errors the same type as before? `{'Mostly yes' if same_error_type else 'Not clearly'}`",
        "",
    ]
    report_path = Path(config["report_path"])
    ensure_parent_dir(report_path)
    report_path.write_text("\n".join(lines) + "\n")


def _write_next_step_report(
    config: dict[str, Any],
    summary: dict[str, Any],
    previous_remaining_summary: dict[str, Any],
) -> None:
    fresh_buckets = Counter(summary["winner_error_buckets"]["bucket_counts"])
    fresh_prompt_like = (
        int(fresh_buckets.get("overgenerous_full_credit", 0))
        + int(fresh_buckets.get("rubric_ambiguity", 0))
        + int(fresh_buckets.get("almost_vs_partial_boundary", 0))
    )
    fresh_model_like = (
        int(fresh_buckets.get("reasoning_or_comprehension_failure", 0))
        + int(fresh_buckets.get("unlikely_prompt_fix", 0))
    )
    fresh_delta = summary["winner_vs_baseline"]["bootstrap"]

    if summary["generalized"] == "yes" and fresh_prompt_like >= fresh_model_like * 2:
        policy_room = "Some narrow policy-only room remains."
        diminishing = "Prompt-policy gains are probably entering diminishing returns, not exhausted."
        next_jump = "A stronger model is the more likely next source of a larger jump."
    elif fresh_prompt_like > fresh_model_like:
        policy_room = "A small policy-only refinement is still plausible, but not obviously high leverage."
        diminishing = "Diminishing returns from prompt-policy work are becoming more likely."
        next_jump = "A stronger model is slightly more likely than another prompt rewrite to deliver the next meaningful gain."
    else:
        policy_room = "There is not much obvious room for another policy-only change."
        diminishing = "Prompt-policy work likely faces diminishing returns now."
        next_jump = "A stronger model is the most likely next lever."

    lines = [
        "# Model vs Policy Next Step",
        "",
        "## Inputs considered",
        "",
        f"- Lockbox winner vs baseline: accuracy delta `{_load_json(Path(config['comparison_artifacts']['lockbox_summary']))['winner_vs_baseline']['bootstrap']['accuracy_delta_mean']:+.3f}`, error delta `{_load_json(Path(config['comparison_artifacts']['lockbox_summary']))['winner_vs_baseline']['bootstrap']['mae_delta_mean']:+.3f}`",
        f"- Fresh winner vs baseline: accuracy delta `{fresh_delta['accuracy_delta_mean']:+.3f}`, error delta `{fresh_delta['mae_delta_mean']:+.3f}`",
        f"- Locked remaining-error bottleneck: `{previous_remaining_summary['main_bottleneck']}`",
        f"- Fresh remaining-error buckets: `{dict(fresh_buckets)}`",
        "",
        "## Direct answers",
        "",
        f"1. Is there still obvious room for one more policy-only refinement? `{policy_room}`",
        f"2. Have prompt-policy gains likely reached diminishing returns? `{diminishing}`",
        f"3. Is the next likely jump more likely from a stronger model than from another prompt change? `{next_jump}`",
        (
            "4. If a stronger model is next, how should it be tested cleanly with the frozen winner prompt? "
            "`Keep guideline_gate_almost_boundary_v1 frozen, keep the parser and harness fixed, compare the current model against one stronger model on a new untouched evaluation pack, and do not retune the prompt after seeing that result.`"
        ),
        "",
    ]

    report_path = Path(config["next_step_report_path"])
    ensure_parent_dir(report_path)
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fresh IMO generalization evaluation on the unused filtered remainder.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to the fresh generalization YAML config.")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    rows, metadata = build_fresh_generalization_set(config)
    _write_plan_report(config, metadata)

    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    variants = {item["id"]: _run_variant(config, item, rows) for item in config["variants"]}
    baseline = variants[config["baseline_variant"]]
    winner = variants[config["winner_variant"]]
    lockbox_summary = _load_json(Path(config["comparison_artifacts"]["lockbox_summary"]))
    previous_remaining_summary = _load_json(Path(config["comparison_artifacts"]["remaining_error_summary"]))

    changed = changed_examples(baseline["predictions"], winner["predictions"])
    bootstrap = bootstrap_delta_summary(
        baseline["predictions"],
        winner["predictions"],
        iterations=int(config["bootstrap_iterations"]),
        seed=int(config["bootstrap_seed"]),
    )
    generalization_scale = classify_generalization_scale(
        lockbox_accuracy_delta=float(lockbox_summary["winner_vs_baseline"]["bootstrap"]["accuracy_delta_mean"]),
        lockbox_mae_delta=float(lockbox_summary["winner_vs_baseline"]["bootstrap"]["mae_delta_mean"]),
        fresh_accuracy_delta=float(bootstrap["accuracy_delta_mean"]),
        fresh_mae_delta=float(bootstrap["mae_delta_mean"]),
    )
    generalized = (
        "yes"
        if float(bootstrap["accuracy_delta_mean"]) > 0.0
        and float(bootstrap["mae_delta_mean"]) < 0.0
        and float(winner["pooled"]["valid_label_rate"]) >= float(baseline["pooled"]["valid_label_rate"])
        else "no"
    )
    summary = {
        "study_type": config["study_type"],
        "model": config["model"],
        "parser_version": config["parser_version"],
        "subset_policy": config["subset_policy"],
        "fresh_set_note": config["fresh_set_note"],
        "baseline_variant": config["baseline_variant"],
        "winner_variant": config["winner_variant"],
        "fresh_set_metadata": metadata,
        "variants": variants,
        "winner_vs_baseline": {
            "changed": changed,
            "bootstrap": bootstrap,
            "by_gold_label": {
                config["baseline_variant"]: _group_metric_breakdown(baseline["predictions"], "Reward"),
                config["winner_variant"]: _group_metric_breakdown(winner["predictions"], "Reward"),
            },
            "by_problem_source": {
                config["baseline_variant"]: _group_metric_breakdown(baseline["predictions"], "Problem Source"),
                config["winner_variant"]: _group_metric_breakdown(winner["predictions"], "Problem Source"),
            },
        },
        "winner_error_buckets": _winner_error_bucket_summary(
            baseline["predictions"],
            winner["predictions"],
        ),
        "generalization_scale": generalization_scale,
        "generalized": generalized,
    }

    summary_path = Path(config["summary_path"])
    ensure_parent_dir(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2))
    _write_result_report(config, summary, lockbox_summary, previous_remaining_summary)
    _write_next_step_report(config, summary, previous_remaining_summary)


if __name__ == "__main__":
    main()
