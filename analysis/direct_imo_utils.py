from __future__ import annotations

import csv
import json
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import yaml

from proofgrade.config import RuntimeSettings
from proofgrade.grader import grade_submission
from proofgrade.schemas import GradeRequest


IMO_LABEL_ORDER = {
    "incorrect": 0,
    "partial": 1,
    "almost": 2,
    "correct": 3,
}
QUESTION_ID = "Grading ID"
GROUND_TRUTH_KEY = "Reward"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


@contextmanager
def temporary_env(key: str, value: str):
    original = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def slice_rows(rows: list[dict[str, str]], start: int, end: int) -> list[dict[str, str]]:
    return [dict(row) for row in rows[start:end]]


def build_shards(rows: list[dict[str, str]], shards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    built = []
    for shard in shards:
        start = int(shard["start"])
        end = int(shard["end"])
        built.append(
            {
                **shard,
                "start": start,
                "end": end,
                "rows": slice_rows(rows, start, end),
            }
        )
    return built


def evaluate_imo_rows(
    *,
    rows: list[dict[str, str]],
    output_root: Path,
    run_id: str,
    model: str,
    variant_env: str,
    variant_id: str,
    save_interval: int = 5,
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
    results: list[dict[str, Any]] = []

    settings = RuntimeSettings(model=model, prompt_variant=variant_id)
    for idx, row in enumerate(rows, start=1):
        results.append(grade_imo_row(row, settings=settings))
        if idx % save_interval == 0:
            write_csv_rows(predictions_path, results, fieldnames)

    write_csv_rows(predictions_path, results, fieldnames)
    report_dict = build_imo_report(results)
    ensure_parent_dir(report_path)
    report_path.write_text(json.dumps(report_dict, indent=2))
    return {
        "run_dir": str(run_dir),
        "report": report_dict,
        "predictions": results,
    }


def format_input_dict(row: dict[str, str]) -> dict[str, str]:
    return {
        "domain": "imo_grading",
        "problem": row["Problem"],
        "solution": row["Solution"],
        "grading_guidelines": row["Grading guidelines"],
        "student_answer": row["Response"],
    }


def grade_imo_row(row: dict[str, str], *, settings: RuntimeSettings) -> dict[str, Any]:
    request = GradeRequest(
        problem=row["Problem"],
        solution=row["Solution"],
        grading_guidelines=row["Grading guidelines"],
        student_answer=row["Response"],
        prompt_variant=settings.prompt_variant,
        model=settings.model,
    )
    result = grade_submission(request, settings)
    response = result.response
    graded_row = dict(row)
    graded_row["prediction"] = response.label
    graded_row["rationale"] = response.rationale
    graded_row["matched_guideline"] = response.matched_guideline
    graded_row["parse_source"] = response.parse_source
    graded_row["prompt_variant"] = response.prompt_variant
    graded_row["model_provider"] = response.model_provider
    graded_row["model_name"] = response.model_name
    graded_row["version"] = response.version
    graded_row["request_id"] = response.request_id
    return graded_row


def normalize_label(label: str | None) -> str:
    return (label or "").strip().lower()


def label_distance(prediction: str | None, ground_truth: str | None) -> int:
    pred = normalize_label(prediction)
    gold = normalize_label(ground_truth)
    if pred not in IMO_LABEL_ORDER or gold not in IMO_LABEL_ORDER:
        return 4
    return abs(IMO_LABEL_ORDER[pred] - IMO_LABEL_ORDER[gold])


def prediction_correct(prediction: str | None, ground_truth: str | None) -> int:
    return int(normalize_label(prediction) == normalize_label(ground_truth))


def normalized_error(prediction: str | None, ground_truth: str | None) -> float:
    pred = normalize_label(prediction)
    gold = normalize_label(ground_truth)
    points = {"incorrect": 0, "partial": 1, "almost": 6, "correct": 7}
    if pred not in points or gold not in points:
        return 1.0
    return abs(points[pred] - points[gold]) / 7.0


def pooled_prediction_rows(runs_by_shard: list[dict[str, Any]]) -> list[dict[str, str]]:
    pooled: list[dict[str, str]] = []
    for item in runs_by_shard:
        pooled.extend(item["predictions"])
    return pooled


def changed_examples(
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
        base_pred = normalize_label(base.get("prediction"))
        cand_pred = normalize_label(row.get("prediction"))
        gold = normalize_label(row.get("Reward"))
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
        base_correct = prediction_correct(base_pred, gold)
        cand_correct = prediction_correct(cand_pred, gold)
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
        "examples": changed,
    }


def pooled_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(prediction_correct(row.get("prediction"), row.get("Reward")) for row in rows)
    valid = sum(int(normalize_label(row.get("prediction")) in IMO_LABEL_ORDER) for row in rows)
    mae = sum(normalized_error(row.get("prediction"), row.get("Reward")) for row in rows) / total if total else 0.0
    return {
        "total": total,
        "overall_accuracy": correct / total if total else 0.0,
        "normalized_mean_absolute_error": mae,
        "valid_label_rate": valid / total if total else 0.0,
        "total_correct": correct,
    }


def build_imo_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = pooled_metrics(rows)
    label_counts: dict[str, int] = {}
    correct_by_label: dict[str, int] = {}
    pred_counts: dict[str, int] = {}
    question_ids_failed: list[str] = []
    question_ids_passed: list[str] = []
    invalid_question_ids: list[str] = []
    allowed = set(IMO_LABEL_ORDER)

    for row in rows:
        gold = normalize_label(row.get(GROUND_TRUTH_KEY))
        pred = normalize_label(row.get("prediction"))
        qid = row.get(QUESTION_ID, "")
        label_counts[gold] = label_counts.get(gold, 0) + 1
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
        if pred not in allowed:
            invalid_question_ids.append(qid)
        if pred == gold:
            correct_by_label[gold] = correct_by_label.get(gold, 0) + 1
            question_ids_passed.append(qid)
        else:
            question_ids_failed.append(qid)

    accuracy_by_ground_truth: dict[str, dict[str, Any]] = {}
    for label, total in label_counts.items():
        tp = sum(
            1
            for row in rows
            if normalize_label(row.get("prediction")) == label and normalize_label(row.get(GROUND_TRUTH_KEY)) == label
        )
        fp = sum(
            1
            for row in rows
            if normalize_label(row.get("prediction")) == label and normalize_label(row.get(GROUND_TRUTH_KEY)) != label
        )
        fn = sum(
            1
            for row in rows
            if normalize_label(row.get("prediction")) != label and normalize_label(row.get(GROUND_TRUTH_KEY)) == label
        )
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy_by_ground_truth[label] = {
            "precision": precision,
            "recall": recall,
            "correct": correct_by_label.get(label, 0),
            "total": total,
        }

    total = metrics["total"] or 1
    winner_distribution = {label: count / total for label, count in label_counts.items()}
    prediction_distribution = {label: count / total for label, count in pred_counts.items()}
    random_guess_accuracy = sum(value * value for value in winner_distribution.values())
    return {
        **metrics,
        "invalid_prediction_count": len(invalid_question_ids),
        "invalid_question_ids": invalid_question_ids,
        "accuracy_by_ground_truth": accuracy_by_ground_truth,
        "label_distribution": {
            "ground_truth": winner_distribution,
            "prediction": prediction_distribution,
        },
        "random_guess_accuracy": random_guess_accuracy,
        "question_ids_failed": question_ids_failed,
        "question_ids_passed": question_ids_passed,
    }


def bootstrap_delta_summary(
    baseline_predictions: list[dict[str, str]],
    candidate_predictions: list[dict[str, str]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    baseline_by_id = {row["Grading ID"]: row for row in baseline_predictions}
    paired: list[tuple[float, float]] = []
    for row in candidate_predictions:
        base = baseline_by_id[row["Grading ID"]]
        gold = row.get("Reward")
        paired.append(
            (
                prediction_correct(row.get("prediction"), gold) - prediction_correct(base.get("prediction"), gold),
                normalized_error(row.get("prediction"), gold) - normalized_error(base.get("prediction"), gold),
            )
        )
    if not paired:
        return {
            "accuracy_delta_mean": 0.0,
            "mae_delta_mean": 0.0,
            "accuracy_delta_ci": [0.0, 0.0],
            "mae_delta_ci": [0.0, 0.0],
        }

    rng = random.Random(seed)
    acc_deltas: list[float] = []
    mae_deltas: list[float] = []
    n = len(paired)
    for _ in range(iterations):
        sample = [paired[rng.randrange(n)] for _ in range(n)]
        acc_deltas.append(sum(item[0] for item in sample) / n)
        mae_deltas.append(sum(item[1] for item in sample) / n)
    acc_sorted = sorted(acc_deltas)
    mae_sorted = sorted(mae_deltas)
    low_idx = int(iterations * 0.025)
    high_idx = min(iterations - 1, int(iterations * 0.975))
    return {
        "accuracy_delta_mean": sum(item[0] for item in paired) / n,
        "mae_delta_mean": sum(item[1] for item in paired) / n,
        "accuracy_delta_ci": [acc_sorted[low_idx], acc_sorted[high_idx]],
        "mae_delta_ci": [mae_sorted[low_idx], mae_sorted[high_idx]],
    }


def shard_metric_summary(shard_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for item in shard_runs:
        summary.append(
            {
                "name": item["shard"]["name"],
                "start": item["shard"]["start"],
                "end": item["shard"]["end"],
                **item["report"],
            }
        )
    return summary


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
