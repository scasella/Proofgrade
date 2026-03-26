from __future__ import annotations

import argparse
import json
from collections import Counter
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
    normalize_label,
    pooled_metrics,
    pooled_prediction_rows,
    read_csv_rows,
)
from proofgrade.policy import build_instruction


DEFAULT_CONFIG = Path("configs/baseline_freeze/final_imo_lock.yaml")
SELECTED_TRANSITIONS = (
    ("correct", "almost"),
    ("correct", "partial"),
    ("partial", "incorrect"),
    ("incorrect", "partial"),
    ("partial", "almost"),
    ("almost", "correct"),
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _dummy_inputs() -> dict[str, str]:
    return {
        "domain": "imo_grading",
        "problem": "P",
        "solution": "S",
        "grading_guidelines": "(Partial) key step",
        "student_answer": "A",
    }


def _prompt_header(variant_env: str, variant_id: str) -> str:
    prompt = build_instruction(
        problem="P",
        solution="S",
        grading_guidelines="(Partial) key step",
        student_answer="A",
        prompt_variant=variant_id,
    )
    return prompt.split("\n\nTask input:\n```json\n", 1)[0].strip()


def _variant_from_summary(summary: dict[str, Any], variant_id: str) -> dict[str, Any]:
    return dict(summary["variants"][variant_id])


def _run_fresh_variant(config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    dataset_rows = read_csv_rows(Path(config["validation_dataset_path"]))
    shards = build_shards(dataset_rows, config["shards"])
    shard_runs: list[dict[str, Any]] = []
    for shard in shards:
        run = evaluate_imo_rows(
            rows=shard["rows"],
            output_root=Path(config["ablation_output_root"]) / variant["id"],
            run_id=f"{variant['id']}_{shard['name']}",
            model=config["model"],
            variant_env=config["variant_env"],
            variant_id=variant["env_value"],
            save_interval=int(config.get("save_interval", 5)),
        )
        shard_runs.append(
            {
                "shard": {k: shard[k] for k in ("name", "start", "end")},
                **run,
            }
        )
    pooled_predictions = pooled_prediction_rows(shard_runs)
    return {
        "id": variant["id"],
        "role": variant["role"],
        "hypothesis": variant.get("hypothesis", ""),
        "prompt_header": _prompt_header(config["variant_env"], variant["env_value"]),
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


def _prediction_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["Grading ID"]: row for row in rows}


def _transition_summary(from_rows: list[dict[str, Any]], to_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from_by_id = _prediction_map(from_rows)
    selected_counts = {f"{a}->{b}": 0 for a, b in SELECTED_TRANSITIONS}
    all_counts: Counter[str] = Counter()
    corrected_overcredit = 0
    for row in to_rows:
        grading_id = row["Grading ID"]
        source_row = from_by_id[grading_id]
        source_pred = normalize_label(source_row.get("prediction"))
        target_pred = normalize_label(row.get("prediction"))
        gold = normalize_label(row.get("Reward"))
        if source_pred == target_pred:
            continue
        key = f"{source_pred}->{target_pred}"
        all_counts[key] += 1
        if key in selected_counts:
            selected_counts[key] += 1
        if (
            source_pred == "correct"
            and gold != "correct"
            and target_pred != "correct"
            and label_distance(target_pred, gold) < label_distance(source_pred, gold)
        ):
            corrected_overcredit += 1
    return {
        "selected_transitions": selected_counts,
        "all_transitions": dict(all_counts),
        "corrected_overcredit_count": corrected_overcredit,
    }


def _pairwise_summary(
    config: dict[str, Any],
    from_variant: dict[str, Any],
    to_variant: dict[str, Any],
) -> dict[str, Any]:
    changed = changed_examples(from_variant["pooled_predictions"], to_variant["pooled_predictions"])
    bootstrap = bootstrap_delta_summary(
        from_variant["pooled_predictions"],
        to_variant["pooled_predictions"],
        iterations=int(config["bootstrap_iterations"]),
        seed=int(config["bootstrap_seed"]),
    )
    transitions = _transition_summary(from_variant["pooled_predictions"], to_variant["pooled_predictions"])
    return {
        "from": from_variant["id"],
        "to": to_variant["id"],
        "from_metrics": from_variant["pooled"],
        "to_metrics": to_variant["pooled"],
        "accuracy_delta": float(to_variant["pooled"]["overall_accuracy"]) - float(from_variant["pooled"]["overall_accuracy"]),
        "mae_delta": float(to_variant["pooled"]["normalized_mean_absolute_error"]) - float(from_variant["pooled"]["normalized_mean_absolute_error"]),
        "valid_label_delta": float(to_variant["pooled"]["valid_label_rate"]) - float(from_variant["pooled"]["valid_label_rate"]),
        "bootstrap": bootstrap,
        "changed": changed,
        "transitions": transitions,
    }


def _mechanism_summary(summary: dict[str, Any]) -> list[str]:
    comparisons = summary["pairwise_comparisons"]
    gate = comparisons["baseline_to_guideline_gate_v1"]
    almost = comparisons["guideline_gate_v1_to_guideline_gate_almost_boundary_v1"]
    top_end = comparisons["guideline_gate_almost_boundary_v1_to_guideline_gate_no_top_end_guard_v1"]
    lines = [
        f"- Original guideline gate: accuracy `{gate['accuracy_delta']:+.3f}`, normalized MAE `{gate['mae_delta']:+.3f}`, corrected overcredit `{gate['transitions']['corrected_overcredit_count']}`.",
        f"- Almost-boundary follow-up: accuracy `{almost['accuracy_delta']:+.3f}`, normalized MAE `{almost['mae_delta']:+.3f}`, corrected overcredit `{almost['transitions']['corrected_overcredit_count']}`.",
    ]
    if top_end["accuracy_delta"] < 0.0 or top_end["mae_delta"] > 0.0:
        lines.append(
            f"- Removing the top-end guard hurts the frozen winner: accuracy `{top_end['accuracy_delta']:+.3f}`, normalized MAE `{top_end['mae_delta']:+.3f}`."
        )
    else:
        lines.append(
            f"- Removing the top-end guard does not hurt overall metrics: accuracy `{top_end['accuracy_delta']:+.3f}`, normalized MAE `{top_end['mae_delta']:+.3f}`."
        )
    if almost["transitions"]["selected_transitions"]["correct->almost"] > 0 or almost["transitions"]["selected_transitions"]["correct->partial"] > 0:
        lines.append("- The follow-up gain is partly driven by pulling borderline full-credit calls downward.")
    else:
        lines.append("- The follow-up gain is broader than a simple full-credit clamp.")
    return lines


def _write_report(config: dict[str, Any], summary: dict[str, Any], report_path: Path) -> None:
    commands = [
        "PYTHONPATH=. .venv/bin/python analysis/run_final_imo_ablation.py --config configs/baseline_freeze/final_imo_lock.yaml",
        "PYTHONPATH=. .venv/bin/python analysis/build_final_imo_remaining_error_atlas.py --config configs/baseline_freeze/final_imo_lock.yaml",
        "GEMINI_API_KEY=... GOOGLE_API_KEY=... PYTHONPATH=. .venv/bin/python analysis/run_final_imo_lockbox_test.py --config configs/baseline_freeze/final_imo_lockbox_test.yaml",
    ]
    lines = [
        "# Final IMO Lock And Ablation",
        "",
        "## Frozen variants",
        "",
        f"- Model/provider: `{config['model']}`",
        f"- Parser version: `{config['parser_version']}`",
        f"- Held-out validation protocol: `{config['validation_protocol']}`",
        f"- Test protocol: `{config['test_protocol']}`",
        "",
    ]
    for variant in config["variants"]:
        if variant["role"] == "ablation_only":
            continue
        lines.extend(
            [
                f"### {variant['id']}",
                f"- Role: `{variant['role']}`",
                f"- Recover via env switch: `{config['variant_env']}={variant['env_value']}`",
                f"- Hypothesis: `{variant['hypothesis']}`",
                "",
            ]
        )

    lines.extend(["## Evaluation commands", ""])
    for command in commands:
        lines.append(f"- `{command}`")

    lines.extend(["", "## Validation ablation", ""])
    for comparison in config["pairwise_comparisons"]:
        result = summary["pairwise_comparisons"][comparison["name"]]
        lines.extend(
            [
                f"### {comparison['name']}",
                f"- From `{result['from']}` to `{result['to']}`",
                f"- Accuracy: `{result['from_metrics']['overall_accuracy']:.3f}` -> `{result['to_metrics']['overall_accuracy']:.3f}` (`{result['accuracy_delta']:+.3f}`)",
                f"- Normalized grading error: `{result['from_metrics']['normalized_mean_absolute_error']:.3f}` -> `{result['to_metrics']['normalized_mean_absolute_error']:.3f}` (`{result['mae_delta']:+.3f}`)",
                f"- Valid-label rate: `{result['from_metrics']['valid_label_rate']:.3f}` -> `{result['to_metrics']['valid_label_rate']:.3f}` (`{result['valid_label_delta']:+.3f}`)",
                f"- Paired bootstrap accuracy delta 95% CI: `{result['bootstrap']['accuracy_delta_ci']}`",
                f"- Paired bootstrap grading-error delta 95% CI: `{result['bootstrap']['mae_delta_ci']}`",
                f"- Changed predictions: `{result['changed']['changed_count']}`",
                f"- Better `{result['changed']['better']}`, worse `{result['changed']['worse']}`, same-score changes `{result['changed']['same_score']}`",
                f"- Corrected over-generous full-credit cases: `{result['transitions']['corrected_overcredit_count']}`",
                "- Key transitions:",
            ]
        )
        for transition in ("correct->almost", "correct->partial", "partial->incorrect", "incorrect->partial", "partial->almost", "almost->correct"):
            lines.append(f"  - `{transition}`: `{result['transitions']['selected_transitions'][transition]}`")
        lines.append("")

    lines.extend(["## Mechanistic readout", ""])
    lines.extend(_mechanism_summary(summary))
    lines.append("")
    lines.append(
        "The frozen validation winner for the lockbox test is `guideline_gate_almost_boundary_v1`, and the only new validation run in this lock phase was the ablation-only `guideline_gate_no_top_end_guard_v1`."
    )
    lines.append("")

    ensure_parent_dir(report_path)
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze final IMO variants and run the validation ablation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to YAML config.")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    robustness_summary = _load_json(Path(config["reuse_summaries"]["robustness"]))
    followup_summary = _load_json(Path(config["reuse_summaries"]["followup"]))

    variants: dict[str, dict[str, Any]] = {
        config["baseline_variant"]: _variant_from_summary(robustness_summary, config["baseline_variant"]),
        config["original_gate_variant"]: _variant_from_summary(robustness_summary, config["original_gate_variant"]),
        config["final_winner_variant"]: _variant_from_summary(followup_summary, config["final_winner_variant"]),
    }

    ablation_variant = next(item for item in config["variants"] if item["id"] == config["ablation_variant"])
    variants[config["ablation_variant"]] = _run_fresh_variant(config, ablation_variant)

    pairwise = {}
    for item in config["pairwise_comparisons"]:
        pairwise[item["name"]] = _pairwise_summary(
            config,
            variants[item["from"]],
            variants[item["to"]],
        )

    summary = {
        "model": config["model"],
        "variant_env": config["variant_env"],
        "parser_version": config["parser_version"],
        "validation_protocol": config["validation_protocol"],
        "test_protocol": config["test_protocol"],
        "baseline_variant": config["baseline_variant"],
        "original_gate_variant": config["original_gate_variant"],
        "final_winner_variant": config["final_winner_variant"],
        "ablation_variant": config["ablation_variant"],
        "variants": variants,
        "pairwise_comparisons": pairwise,
    }

    summary_path = Path(config["summary_path"])
    ensure_parent_dir(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2))
    _write_report(config, summary, Path(config["report_path"]))


if __name__ == "__main__":
    main()
