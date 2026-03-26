from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analysis.direct_imo_utils import (
    bootstrap_delta_summary,
    changed_examples,
    ensure_parent_dir,
    evaluate_imo_rows,
    load_yaml,
    pooled_metrics,
    read_csv_rows,
)


DEFAULT_CONFIG = Path("configs/baseline_freeze/final_imo_lockbox_test.yaml")


def _run_variant(config: dict[str, Any], variant: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    run = evaluate_imo_rows(
        rows=rows,
        output_root=Path(config["output_root"]) / variant["id"],
        run_id=f"{variant['id']}_test_00_99",
        model=config["model"],
        variant_env=config["variant_env"],
        variant_id=variant["env_value"],
        save_interval=int(config.get("save_interval", 5)),
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


def _judgment(baseline: dict[str, Any], winner: dict[str, Any]) -> tuple[str, str]:
    acc_delta = float(winner["pooled"]["overall_accuracy"]) - float(baseline["pooled"]["overall_accuracy"])
    mae_delta = float(winner["pooled"]["normalized_mean_absolute_error"]) - float(baseline["pooled"]["normalized_mean_absolute_error"])
    if acc_delta > 0.0 and mae_delta < 0.0 and float(winner["pooled"]["valid_label_rate"]) >= float(baseline["pooled"]["valid_label_rate"]):
        return (
            "strong_positive",
            "The prompt-policy improvements held on test and should remain the new default.",
        )
    if acc_delta >= 0.0 or mae_delta < 0.0:
        return (
            "partial_positive",
            "The validation gains were real, but the test gain was smaller or mixed.",
        )
    return (
        "negative_surprise",
        "The validation win did not hold on test, so the default should be reconsidered.",
    )


def _write_report(config: dict[str, Any], summary: dict[str, Any], report_path: Path) -> None:
    baseline = summary["variants"][config["baseline_variant"]]
    winner = summary["variants"][config["winner_variant"]]
    delta = summary["winner_vs_baseline"]
    lines = [
        "# Final IMO Lockbox Test",
        "",
        "## Exact commands run",
        "",
        f"- `GEMINI_API_KEY=... GOOGLE_API_KEY=... PYTHONPATH=. .venv/bin/python analysis/run_final_imo_lockbox_test.py --config {DEFAULT_CONFIG}`",
        "",
        "## Test results",
        "",
        f"- Baseline accuracy: `{baseline['pooled']['overall_accuracy']:.3f}`",
        f"- Final-winner accuracy: `{winner['pooled']['overall_accuracy']:.3f}`",
        f"- Baseline normalized grading error: `{baseline['pooled']['normalized_mean_absolute_error']:.3f}`",
        f"- Final-winner normalized grading error: `{winner['pooled']['normalized_mean_absolute_error']:.3f}`",
        f"- Baseline valid-label rate: `{baseline['pooled']['valid_label_rate']:.3f}`",
        f"- Final-winner valid-label rate: `{winner['pooled']['valid_label_rate']:.3f}`",
        f"- Accuracy delta: `{delta['bootstrap']['accuracy_delta_mean']:+.3f}`",
        f"- Accuracy delta 95% bootstrap CI: `{delta['bootstrap']['accuracy_delta_ci']}`",
        f"- Normalized grading error delta: `{delta['bootstrap']['mae_delta_mean']:+.3f}`",
        f"- Normalized grading error delta 95% bootstrap CI: `{delta['bootstrap']['mae_delta_ci']}`",
        f"- Changed predictions vs baseline: `{delta['changed']['changed_count']}`",
        f"- Better `{delta['changed']['better']}`, worse `{delta['changed']['worse']}`, same-score changes `{delta['changed']['same_score']}`",
        "",
        "## Interpretation",
        "",
        f"- Judgment: `{summary['judgment_label']}`",
        f"- Conclusion: {summary['judgment_text']}",
        "",
    ]
    ensure_parent_dir(report_path)
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one-time final IMO lockbox test.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to lockbox test YAML config.")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    rows = read_csv_rows(Path(config["dataset_path"]))
    variants = {item["id"]: _run_variant(config, item, rows) for item in config["variants"]}
    baseline = variants[config["baseline_variant"]]
    winner = variants[config["winner_variant"]]
    changed = changed_examples(baseline["predictions"], winner["predictions"])
    bootstrap = bootstrap_delta_summary(
        baseline["predictions"],
        winner["predictions"],
        iterations=int(config["bootstrap_iterations"]),
        seed=int(config["bootstrap_seed"]),
    )
    judgment_label, judgment_text = _judgment(baseline, winner)
    summary = {
        "model": config["model"],
        "parser_version": config["parser_version"],
        "subset_policy": config["subset_policy"],
        "baseline_variant": config["baseline_variant"],
        "winner_variant": config["winner_variant"],
        "variants": variants,
        "winner_vs_baseline": {
            "changed": changed,
            "bootstrap": bootstrap,
        },
        "judgment_label": judgment_label,
        "judgment_text": judgment_text,
    }

    summary_path = Path(config["summary_path"])
    ensure_parent_dir(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2))
    _write_report(config, summary, Path(config["report_path"]))


if __name__ == "__main__":
    main()
