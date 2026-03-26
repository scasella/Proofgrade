from __future__ import annotations

import argparse
import csv
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from domains.harness import harness
from domains.report import report


DEFAULT_CONFIG = Path("configs/baseline_freeze/direct_imo_improvement.yaml")
DEFAULT_REPORT = Path("reports/direct_imo_improvement.md")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


@contextmanager
def _temporary_env(key: str, value: str):
    original = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _changed_examples(
    baseline_predictions: list[dict[str, str]],
    candidate_predictions: list[dict[str, str]],
) -> dict[str, Any]:
    baseline_by_id = {row["Grading ID"]: row for row in baseline_predictions}
    changed: list[dict[str, str]] = []
    better = 0
    worse = 0
    same_score = 0
    for row in candidate_predictions:
        qid = row["Grading ID"]
        base = baseline_by_id[qid]
        base_pred = base["prediction"].strip().lower()
        cand_pred = row["prediction"].strip().lower()
        gold = row["Reward"].strip().lower()
        if base_pred == cand_pred:
            continue
        changed.append(
            {
                "grading_id": qid,
                "ground_truth": gold,
                "baseline_prediction": base_pred,
                "candidate_prediction": cand_pred,
            }
        )
        base_correct = int(base_pred == gold)
        cand_correct = int(cand_pred == gold)
        if cand_correct > base_correct:
            better += 1
        elif cand_correct < base_correct:
            worse += 1
        else:
            same_score += 1
    return {
        "changed_count": len(changed),
        "better": better,
        "worse": worse,
        "same_score": same_score,
        "examples": changed[:10],
    }


def _run_eval(
    *,
    output_root: Path,
    variant_env: str,
    variant_id: str,
    model: str,
    subset: str,
    num_samples: int,
    num_workers: int,
    save_interval: int,
) -> dict[str, Any]:
    run_id = f"{variant_id}{subset}_n{num_samples}".replace("__", "_")
    run_dir = output_root / run_id
    predictions_path = run_dir / "predictions.csv"
    report_path = run_dir / "report.json"

    if not predictions_path.exists() or not report_path.exists():
        with _temporary_env(variant_env, variant_id):
            output_folder = harness(
                agent_path="./task_agent.py",
                output_dir=str(output_root),
                run_id=run_id,
                domain="imo_grading",
                model=model,
                num_samples=num_samples,
                save_interval=save_interval,
                num_workers=num_workers,
                subset=subset,
            )
            report(dname=output_folder, domain="imo_grading")

    with report_path.open() as f:
        metrics = json.load(f)
    predictions = _read_predictions(predictions_path)
    return {"run_dir": str(run_dir), "report": metrics, "predictions": predictions}


def _sort_key(result: dict[str, Any]) -> tuple[float, float]:
    report_data = result["report"]
    return (
        float(report_data["overall_accuracy"]),
        -float(report_data["normalized_mean_absolute_error"]),
    )


def _write_report(
    *,
    report_path: Path,
    config: dict[str, Any],
    atlas_path: Path,
    train_baseline: dict[str, Any],
    train_results: list[dict[str, Any]],
    winner: dict[str, Any] | None,
    val_baseline: dict[str, Any] | None,
    val_winner: dict[str, Any] | None,
) -> None:
    lines = [
        "# Direct IMO Improvement",
        "",
        "## Setup",
        "",
        f"- Model: `{config['model']}`",
        f"- Train slice: `{config['train']['subset']}` first `{config['train']['num_samples']}` examples",
        f"- Validation slice: `{config['val']['subset']}` first `{config['val']['num_samples']}` examples",
        f"- Variant switch: `{config['variant_env']}`",
        f"- Failure atlas: `{atlas_path}`",
        "",
        "## Train benchmark",
        "",
        f"- Baseline accuracy: `{train_baseline['report']['overall_accuracy']:.3f}`",
        f"- Baseline normalized MAE: `{train_baseline['report']['normalized_mean_absolute_error']:.3f}`",
        f"- Baseline valid-label rate: `{train_baseline['report']['valid_label_rate']:.3f}`",
        "",
    ]

    for result in train_results:
        report_data = result["report"]
        changed = result["changed_vs_baseline"]
        lines.extend(
            [
                f"### {result['variant_id']}",
                f"- Hypothesis: `{result['hypothesis']}`",
                f"- Accuracy: `{report_data['overall_accuracy']:.3f}`",
                f"- Normalized MAE: `{report_data['normalized_mean_absolute_error']:.3f}`",
                f"- Valid-label rate: `{report_data['valid_label_rate']:.3f}`",
                f"- Changed predictions vs train baseline: `{changed['changed_count']}`",
                f"- Better: `{changed['better']}`, worse: `{changed['worse']}`, same-score changes: `{changed['same_score']}`",
                "",
            ]
        )

    if winner is None:
        lines.extend(
            [
                "## Outcome",
                "",
                "No train-side variant beat the baseline without hurting validity, so validation promotion was skipped.",
            ]
        )
        report_path.write_text("\n".join(lines) + "\n")
        return

    lines.extend(
        [
            "## Promoted winner",
            "",
            f"- Winner: `{winner['variant_id']}`",
            f"- Reason: best train accuracy with no valid-label regression",
            "",
        ]
    )

    if val_baseline is not None and val_winner is not None:
        val_changed = val_winner["changed_vs_baseline"]
        lines.extend(
            [
                "## Validation result",
                "",
                f"- Baseline accuracy: `{val_baseline['report']['overall_accuracy']:.3f}`",
                f"- Baseline normalized MAE: `{val_baseline['report']['normalized_mean_absolute_error']:.3f}`",
                f"- Baseline valid-label rate: `{val_baseline['report']['valid_label_rate']:.3f}`",
                f"- Winner accuracy: `{val_winner['report']['overall_accuracy']:.3f}`",
                f"- Winner normalized MAE: `{val_winner['report']['normalized_mean_absolute_error']:.3f}`",
                f"- Winner valid-label rate: `{val_winner['report']['valid_label_rate']:.3f}`",
                f"- Accuracy delta: `{val_winner['report']['overall_accuracy'] - val_baseline['report']['overall_accuracy']:+.3f}`",
                f"- Normalized MAE delta: `{val_winner['report']['normalized_mean_absolute_error'] - val_baseline['report']['normalized_mean_absolute_error']:+.3f}`",
                f"- Changed predictions vs validation baseline: `{val_changed['changed_count']}`",
                f"- Better: `{val_changed['better']}`, worse: `{val_changed['worse']}`, same-score changes: `{val_changed['same_score']}`",
                "",
                "## Representative changed validation examples",
                "",
            ]
        )
        if val_changed["examples"]:
            for item in val_changed["examples"]:
                lines.append(
                    f"- `{item['grading_id']}`: `{item['baseline_prediction']}` -> `{item['candidate_prediction']}` "
                    f"(ground truth `{item['ground_truth']}`)"
                )
        else:
            lines.append("- No validation predictions changed.")

        lines.extend(
            [
                "",
                "## Judgment",
                "",
                (
                    "The direct target-improvement patch improved the held-out validation slice."
                    if val_winner["report"]["overall_accuracy"] > val_baseline["report"]["overall_accuracy"]
                    else "The direct target-improvement patch did not improve the held-out validation slice."
                ),
            ]
        )

    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark direct IMO grading prompt variants.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to YAML config.")
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT), help="Markdown report output path.")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    atlas_path = output_root / "failure_atlas.json"

    train_cfg = config["train"]
    val_cfg = config["val"]
    variant_env = config["variant_env"]

    train_baseline = _run_eval(
        output_root=output_root / "train_runs",
        variant_env=variant_env,
        variant_id=config["baseline_variant"],
        model=config["model"],
        subset=train_cfg["subset"],
        num_samples=int(train_cfg["num_samples"]),
        num_workers=int(train_cfg["num_workers"]),
        save_interval=int(train_cfg["save_interval"]),
    )

    train_results: list[dict[str, Any]] = []
    for variant in config.get("variants", []):
        result = _run_eval(
            output_root=output_root / "train_runs",
            variant_env=variant_env,
            variant_id=variant["id"],
            model=config["model"],
            subset=train_cfg["subset"],
            num_samples=int(train_cfg["num_samples"]),
            num_workers=int(train_cfg["num_workers"]),
            save_interval=int(train_cfg["save_interval"]),
        )
        result["variant_id"] = variant["id"]
        result["hypothesis"] = variant["hypothesis"]
        result["changed_vs_baseline"] = _changed_examples(train_baseline["predictions"], result["predictions"])
        train_results.append(result)

    eligible = [
        item
        for item in train_results
        if item["report"]["valid_label_rate"] >= train_baseline["report"]["valid_label_rate"]
        and item["report"]["overall_accuracy"] > train_baseline["report"]["overall_accuracy"]
    ]
    eligible.sort(key=_sort_key, reverse=True)
    winner = eligible[0] if eligible else None

    val_baseline = None
    val_winner = None
    if winner is not None:
        val_baseline = _run_eval(
            output_root=output_root / "val_runs",
            variant_env=variant_env,
            variant_id=config["baseline_variant"],
            model=config["model"],
            subset=val_cfg["subset"],
            num_samples=int(val_cfg["num_samples"]),
            num_workers=int(val_cfg["num_workers"]),
            save_interval=int(val_cfg["save_interval"]),
        )
        val_winner = _run_eval(
            output_root=output_root / "val_runs",
            variant_env=variant_env,
            variant_id=winner["variant_id"],
            model=config["model"],
            subset=val_cfg["subset"],
            num_samples=int(val_cfg["num_samples"]),
            num_workers=int(val_cfg["num_workers"]),
            save_interval=int(val_cfg["save_interval"]),
        )
        val_winner["changed_vs_baseline"] = _changed_examples(val_baseline["predictions"], val_winner["predictions"])

    summary = {
        "winner_variant": winner["variant_id"] if winner else None,
        "train_baseline": train_baseline["report"],
        "train_variants": [
            {
                "variant_id": item["variant_id"],
                "report": item["report"],
                "changed_vs_baseline": item["changed_vs_baseline"],
            }
            for item in train_results
        ],
        "val_baseline": val_baseline["report"] if val_baseline else None,
        "val_winner": val_winner["report"] if val_winner else None,
    }
    (output_root / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))

    _write_report(
        report_path=Path(args.report_path),
        config=config,
        atlas_path=atlas_path,
        train_baseline=train_baseline,
        train_results=train_results,
        winner=winner,
        val_baseline=val_baseline,
        val_winner=val_winner,
    )


if __name__ == "__main__":
    main()
