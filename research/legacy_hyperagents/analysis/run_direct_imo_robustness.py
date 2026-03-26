from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analysis.direct_imo_utils import (
    bootstrap_delta_summary,
    build_shards,
    changed_examples,
    ensure_parent_dir,
    evaluate_imo_rows,
    label_distance,
    load_yaml,
    pooled_metrics,
    pooled_prediction_rows,
    read_csv_rows,
)
from utils.prediction_contracts import build_task_instruction


DEFAULT_CONFIG = Path("configs/baseline_freeze/direct_imo_robustness.yaml")


def _dummy_inputs() -> dict[str, str]:
    return {
        "domain": "imo_grading",
        "problem": "P",
        "solution": "S",
        "grading_guidelines": "(Partial) key step",
        "student_answer": "A",
    }


def _prompt_header(variant_env: str, variant_id: str) -> str:
    import os

    original = os.environ.get(variant_env)
    os.environ[variant_env] = variant_id
    try:
        prompt = build_task_instruction(_dummy_inputs())
    finally:
        if original is None:
            os.environ.pop(variant_env, None)
        else:
            os.environ[variant_env] = original
    return prompt.split("\n\nTask input:\n```json\n", 1)[0].strip()


def _variant_sort_key(item: dict[str, Any]) -> tuple[float, float]:
    pooled = item["pooled"]
    return (
        float(pooled["overall_accuracy"]),
        -float(pooled["normalized_mean_absolute_error"]),
    )


def _non_worse_shard_counts(
    baseline_shards: list[dict[str, Any]],
    candidate_shards: list[dict[str, Any]],
) -> tuple[int, int]:
    by_name = {item["name"]: item for item in baseline_shards}
    acc = 0
    mae = 0
    for item in candidate_shards:
        base = by_name[item["name"]]
        if float(item["overall_accuracy"]) >= float(base["overall_accuracy"]):
            acc += 1
        if float(item["normalized_mean_absolute_error"]) <= float(base["normalized_mean_absolute_error"]):
            mae += 1
    return acc, mae


def _robustness_gate(summary: dict[str, Any]) -> dict[str, Any]:
    baseline = summary["variants"][summary["baseline_variant"]]
    reference = summary["variants"][summary["reference_variant"]]
    acc_non_worse, mae_non_worse = _non_worse_shard_counts(
        baseline["shards"],
        reference["shards"],
    )
    no_valid_regression = all(
        float(item["valid_label_rate"]) >= 1.0 for item in reference["shards"]
    )
    pooled_acc_delta = float(reference["bootstrap_vs_baseline"]["accuracy_delta_mean"])
    pooled_mae_delta = float(reference["bootstrap_vs_baseline"]["mae_delta_mean"])
    passes = (
        no_valid_regression
        and pooled_acc_delta > 0.0
        and pooled_mae_delta < 0.0
        and acc_non_worse >= 3
        and mae_non_worse >= 3
    )
    return {
        "passes": passes,
        "no_valid_regression": no_valid_regression,
        "non_worse_accuracy_shards": acc_non_worse,
        "non_worse_mae_shards": mae_non_worse,
        "pooled_accuracy_delta": pooled_acc_delta,
        "pooled_mae_delta": pooled_mae_delta,
        "accuracy_delta_ci": reference["bootstrap_vs_baseline"]["accuracy_delta_ci"],
        "mae_delta_ci": reference["bootstrap_vs_baseline"]["mae_delta_ci"],
    }


def _run_study(config: dict[str, Any], precomputed_variants: dict[str, Any] | None = None) -> dict[str, Any]:
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_rows = read_csv_rows(Path(config["dataset_path"]))
    shards = build_shards(dataset_rows, config["shards"])
    variant_env = config["variant_env"]
    model = config["model"]
    save_interval = int(config.get("save_interval", 5))

    variant_results: dict[str, Any] = {}
    for variant in config["variants"]:
        if precomputed_variants and variant["id"] in precomputed_variants:
            variant_results[variant["id"]] = precomputed_variants[variant["id"]]
            continue
        shard_runs: list[dict[str, Any]] = []
        for shard in shards:
            run = evaluate_imo_rows(
                rows=shard["rows"],
                output_root=output_root / variant["id"],
                run_id=f"{variant['id']}_{shard['name']}",
                model=model,
                variant_env=variant_env,
                variant_id=variant["id"],
                save_interval=save_interval,
            )
            shard_runs.append(
                {
                    "shard": {k: shard[k] for k in ("name", "start", "end")},
                    **run,
                }
            )

        pooled_predictions = pooled_prediction_rows(shard_runs)
        variant_results[variant["id"]] = {
            "id": variant["id"],
            "role": variant.get("role", ""),
            "hypothesis": variant.get("hypothesis", ""),
            "prompt_header": _prompt_header(variant_env, variant["id"]),
            "shards": [
                {
                    "name": item["shard"]["name"],
                    "start": item["shard"]["start"],
                    "end": item["shard"]["end"],
                    "run_dir": item["run_dir"],
                    **item["report"],
                }
                for item in shard_runs
            ],
            "pooled": pooled_metrics(pooled_predictions),
            "pooled_predictions": pooled_predictions,
        }

    baseline_variant = config["baseline_variant"]
    for variant_id, item in variant_results.items():
        if variant_id == baseline_variant:
            continue
        item["changed_vs_baseline"] = changed_examples(
            variant_results[baseline_variant]["pooled_predictions"],
            item["pooled_predictions"],
        )
        item["bootstrap_vs_baseline"] = bootstrap_delta_summary(
            variant_results[baseline_variant]["pooled_predictions"],
            item["pooled_predictions"],
            iterations=int(config["bootstrap_iterations"]),
            seed=int(config["bootstrap_seed"]),
        )

    summary = {
        "study_type": config["study_type"],
        "model": config["model"],
        "variant_env": variant_env,
        "parser_version": config["parser_version"],
        "subset_policy": config["subset_policy"],
        "dataset_path": config["dataset_path"],
        "baseline_variant": baseline_variant,
        "reference_variant": config.get("reference_variant"),
        "contrast_variant": config.get("contrast_variant"),
        "variants": variant_results,
    }
    if config["study_type"] == "robustness":
        summary["robustness_gate"] = _robustness_gate(summary)
    return summary


def _candidate_ids_for_followup(config: dict[str, Any], error_summary: dict[str, Any] | None) -> list[str]:
    candidate_ids = [item["id"] for item in config["variants"] if item.get("role") == "candidate"]
    if not error_summary:
        return candidate_ids[:2]
    ranked = error_summary.get("recommended_followup_variants") or []
    if ranked:
        seen = []
        for item in ranked:
            if item in candidate_ids and item not in seen:
                seen.append(item)
        return seen[:2]
    return candidate_ids[:2]


def _write_robustness_report(summary: dict[str, Any], report_path: Path) -> None:
    gate = summary["robustness_gate"]
    baseline = summary["variants"][summary["baseline_variant"]]
    reference = summary["variants"][summary["reference_variant"]]
    contrast = summary["variants"][summary["contrast_variant"]]
    lines = [
        "# Direct IMO Robustness",
        "",
        "## Frozen comparators",
        "",
        f"- Model/provider: `{summary['model']}`",
        f"- Parser version: `{summary['parser_version']}`",
        f"- Subset policy: `{summary['subset_policy']}`",
        f"- Baseline: `{summary['baseline_variant']}`",
        f"- Winner under test: `{summary['reference_variant']}`",
        f"- Contrast: `{summary['contrast_variant']}`",
        "",
        "## Pooled comparison over 100 held-out validation examples",
        "",
        f"- Baseline accuracy: `{baseline['pooled']['overall_accuracy']:.3f}`",
        f"- Winner accuracy: `{reference['pooled']['overall_accuracy']:.3f}`",
        f"- Contrast accuracy: `{contrast['pooled']['overall_accuracy']:.3f}`",
        f"- Baseline normalized MAE: `{baseline['pooled']['normalized_mean_absolute_error']:.3f}`",
        f"- Winner normalized MAE: `{reference['pooled']['normalized_mean_absolute_error']:.3f}`",
        f"- Contrast normalized MAE: `{contrast['pooled']['normalized_mean_absolute_error']:.3f}`",
        f"- Winner valid-label rate: `{reference['pooled']['valid_label_rate']:.3f}`",
        f"- Winner accuracy delta vs baseline: `{reference['bootstrap_vs_baseline']['accuracy_delta_mean']:+.3f}`",
        f"- Winner accuracy delta 95% bootstrap CI: `{reference['bootstrap_vs_baseline']['accuracy_delta_ci']}`",
        f"- Winner normalized MAE delta vs baseline: `{reference['bootstrap_vs_baseline']['mae_delta_mean']:+.3f}`",
        f"- Winner normalized MAE delta 95% bootstrap CI: `{reference['bootstrap_vs_baseline']['mae_delta_ci']}`",
        "",
        "## Shard results",
        "",
    ]
    for shard_name in [item["name"] for item in baseline["shards"]]:
        base_shard = next(item for item in baseline["shards"] if item["name"] == shard_name)
        ref_shard = next(item for item in reference["shards"] if item["name"] == shard_name)
        contrast_shard = next(item for item in contrast["shards"] if item["name"] == shard_name)
        lines.extend(
            [
                f"### {shard_name}",
                f"- Baseline: accuracy `{base_shard['overall_accuracy']:.3f}`, normalized MAE `{base_shard['normalized_mean_absolute_error']:.3f}`, valid-label rate `{base_shard['valid_label_rate']:.3f}`",
                f"- Winner: accuracy `{ref_shard['overall_accuracy']:.3f}`, normalized MAE `{ref_shard['normalized_mean_absolute_error']:.3f}`, valid-label rate `{ref_shard['valid_label_rate']:.3f}`",
                f"- Contrast: accuracy `{contrast_shard['overall_accuracy']:.3f}`, normalized MAE `{contrast_shard['normalized_mean_absolute_error']:.3f}`, valid-label rate `{contrast_shard['valid_label_rate']:.3f}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Robustness decision",
            "",
            f"- No valid-label regression on any shard: `{gate['no_valid_regression']}`",
            f"- Non-worse accuracy shards: `{gate['non_worse_accuracy_shards']}` / 4",
            f"- Non-worse MAE shards: `{gate['non_worse_mae_shards']}` / 4",
            f"- Robustness gate passed: `{gate['passes']}`",
            "",
            "## Direct answer",
            "",
            (
                "The guideline-gate win replicated strongly enough to justify one narrow follow-up."
                if gate["passes"]
                else "The guideline-gate win did not replicate strongly enough to justify more prompt iteration yet."
            ),
        ]
    )
    ensure_parent_dir(report_path)
    report_path.write_text("\n".join(lines) + "\n")


def _write_followup_report(
    summary: dict[str, Any],
    report_path: Path,
    robustness_summary: dict[str, Any] | None,
    error_summary: dict[str, Any] | None,
) -> None:
    ensure_parent_dir(report_path)
    if robustness_summary and not robustness_summary.get("robustness_gate", {}).get("passes", False):
        report_path.write_text(
            "# Direct IMO Follow-Up Variant\n\n"
            "## Outcome\n\n"
            "Follow-up benchmarking was skipped because the robustness check did not support the current winner strongly enough.\n"
        )
        return
    if error_summary and not error_summary.get("recommended_followup", False):
        report_path.write_text(
            "# Direct IMO Follow-Up Variant\n\n"
            "## Outcome\n\n"
            "Follow-up benchmarking was skipped because the error-boundary atlas suggested the remaining errors are not mainly prompt-policy leverage.\n"
        )
        return

    baseline = summary["variants"][summary["baseline_variant"]]
    reference = summary["variants"][summary["reference_variant"]]
    candidates = [
        item for key, item in summary["variants"].items()
        if key not in {summary["baseline_variant"], summary["reference_variant"]}
    ]
    candidates.sort(key=_variant_sort_key, reverse=True)
    best = candidates[0] if candidates else None
    winner_replacement = (
        best is not None
        and best["pooled"]["valid_label_rate"] >= reference["pooled"]["valid_label_rate"]
        and best["pooled"]["overall_accuracy"] > reference["pooled"]["overall_accuracy"]
        and best["pooled"]["normalized_mean_absolute_error"] < reference["pooled"]["normalized_mean_absolute_error"]
    )
    reference_by_id = {row["Grading ID"]: row for row in reference["pooled_predictions"]}
    error_examples = error_summary.get("examples", []) if error_summary else []
    primary_bucket = error_summary.get("primary_boundary") if error_summary else None
    primary_ids = [item["grading_id"] for item in error_examples if item.get("bucket") == primary_bucket]
    almost_ids = sorted(
        {
            item["grading_id"]
            for item in error_examples
            if item.get("bucket") == "almost_boundary" or item.get("ground_truth") == "almost"
        }
    )

    def _bucket_delta(candidate_predictions: list[dict[str, Any]], grading_ids: list[str]) -> dict[str, int]:
        candidate_by_id = {row["Grading ID"]: row for row in candidate_predictions}
        improved = 0
        worse = 0
        same = 0
        for grading_id in grading_ids:
            ref_row = reference_by_id.get(grading_id)
            cand_row = candidate_by_id.get(grading_id)
            if not ref_row or not cand_row:
                continue
            gold = ref_row.get("Reward")
            ref_distance = label_distance(ref_row.get("prediction"), gold)
            cand_distance = label_distance(cand_row.get("prediction"), gold)
            if cand_distance < ref_distance:
                improved += 1
            elif cand_distance > ref_distance:
                worse += 1
            else:
                same += 1
        return {"improved": improved, "worse": worse, "same": same}

    lines = [
        "# Direct IMO Follow-Up Variant",
        "",
        "## Setup",
        "",
        f"- Model/provider: `{summary['model']}`",
        f"- Parser version: `{summary['parser_version']}`",
        f"- Baseline: `{summary['baseline_variant']}`",
        f"- Current winner: `{summary['reference_variant']}`",
        "",
    ]
    if error_summary:
        lines.extend(
            [
                "## Why these variants were chosen",
                "",
                f"- Primary remaining boundary: `{error_summary['primary_boundary']}`",
                f"- Follow-up recommendation: `{error_summary['followup_reason']}`",
                "",
            ]
        )
    lines.append("## Variant texts")
    lines.append("")
    for item in [reference] + candidates:
        lines.extend(
            [
                f"### {item['id']}",
                f"- Hypothesis: `{item['hypothesis']}`",
                "```text",
                item["prompt_header"],
                "```",
                "",
            ]
        )

    lines.extend(["## Pooled validation comparison", ""])
    lines.append(
        f"- Baseline: accuracy `{baseline['pooled']['overall_accuracy']:.3f}`, normalized MAE `{baseline['pooled']['normalized_mean_absolute_error']:.3f}`, valid-label rate `{baseline['pooled']['valid_label_rate']:.3f}`"
    )
    lines.append(
        f"- Current winner: accuracy `{reference['pooled']['overall_accuracy']:.3f}`, normalized MAE `{reference['pooled']['normalized_mean_absolute_error']:.3f}`, valid-label rate `{reference['pooled']['valid_label_rate']:.3f}`"
    )
    for item in candidates:
        item["changed_vs_reference"] = changed_examples(reference["pooled_predictions"], item["pooled_predictions"])
        item["bootstrap_vs_reference"] = bootstrap_delta_summary(
            reference["pooled_predictions"],
            item["pooled_predictions"],
            iterations=10000,
            seed=0,
        )
        primary_delta = _bucket_delta(item["pooled_predictions"], primary_ids) if primary_ids else None
        almost_delta = _bucket_delta(item["pooled_predictions"], almost_ids) if almost_ids else None
        lines.extend(
            [
                f"### {item['id']}",
                f"- Accuracy: `{item['pooled']['overall_accuracy']:.3f}`",
                f"- Normalized MAE: `{item['pooled']['normalized_mean_absolute_error']:.3f}`",
                f"- Valid-label rate: `{item['pooled']['valid_label_rate']:.3f}`",
                f"- Accuracy delta vs current winner: `{item['bootstrap_vs_reference']['accuracy_delta_mean']:+.3f}`",
                f"- Normalized MAE delta vs current winner: `{item['bootstrap_vs_reference']['mae_delta_mean']:+.3f}`",
                f"- Changed predictions vs current winner: `{item['changed_vs_reference']['changed_count']}`",
                f"- Better: `{item['changed_vs_reference']['better']}`, worse: `{item['changed_vs_reference']['worse']}`, same-score changes: `{item['changed_vs_reference']['same_score']}`",
            ]
        )
        if primary_delta and primary_bucket:
            lines.append(
                f"- Primary boundary `{primary_bucket}`: improved `{primary_delta['improved']}`, worsened `{primary_delta['worse']}`, unchanged `{primary_delta['same']}`"
            )
        if almost_delta:
            lines.append(
                f"- Almost-related cases: improved `{almost_delta['improved']}`, worsened `{almost_delta['worse']}`, unchanged `{almost_delta['same']}`"
            )
        if item["changed_vs_reference"]["examples"]:
            lines.append("- Changed example IDs:")
            for example in item["changed_vs_reference"]["examples"][:12]:
                lines.append(
                    f"  - `{example['grading_id']}`: `{example['baseline_prediction']}` -> `{example['candidate_prediction']}` (gold `{example['ground_truth']}`)"
                )
        lines.append("")

    lines.extend(
        [
            "## Judgment",
            "",
            (
                f"`{best['id']}` should replace `guideline_gate_v1` as default."
                if winner_replacement and best is not None
                else "No follow-up variant beat `guideline_gate_v1` cleanly enough to replace it as default."
            ),
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shard-based robustness or follow-up studies for IMO grading.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to YAML config.")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    if config["study_type"] == "robustness":
        summary = _run_study(config)
        summary_path = Path(config["summary_path"])
        ensure_parent_dir(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2))
        report_path = Path(config["report_path"])
        _write_robustness_report(summary, report_path)
        return

    robustness_summary = None
    error_summary = None
    if config.get("robustness_summary_path"):
        robustness_path = Path(config["robustness_summary_path"])
        if robustness_path.exists():
            robustness_summary = json.loads(robustness_path.read_text())
    if config.get("error_boundary_summary_path"):
        error_path = Path(config["error_boundary_summary_path"])
        if error_path.exists():
            error_summary = json.loads(error_path.read_text())

    report_path = Path(config["report_path"])
    if robustness_summary and not robustness_summary.get("robustness_gate", {}).get("passes", False):
        _write_followup_report({}, report_path, robustness_summary, error_summary)
        return
    if error_summary and not error_summary.get("recommended_followup", False):
        _write_followup_report({}, report_path, robustness_summary, error_summary)
        return

    if error_summary:
        allowed = set(_candidate_ids_for_followup(config, error_summary))
        config["variants"] = [
            item
            for item in config["variants"]
            if item["id"] in {config["baseline_variant"], config["reference_variant"]} or item["id"] in allowed
        ]
    precomputed_variants = {}
    if robustness_summary:
        for variant_id in {config["baseline_variant"], config["reference_variant"]}:
            existing = robustness_summary.get("variants", {}).get(variant_id)
            if existing is not None:
                precomputed_variants[variant_id] = existing
    summary = _run_study(config, precomputed_variants or None)
    summary_path = Path(config["summary_path"])
    ensure_parent_dir(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2))
    _write_followup_report(summary, report_path, robustness_summary, error_summary)


if __name__ == "__main__":
    main()
