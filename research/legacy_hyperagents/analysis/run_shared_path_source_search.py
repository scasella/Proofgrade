"""Run a shared-path-constrained source search and optional transfer pilot."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.validate_transfer_eligibility import validate_candidate_snapshot


DEFAULT_CONFIG_PATH = Path(
    "/Users/scasella/Downloads/hyperagents/HyperAgents/configs/baseline_freeze/shared_path_source_search.yaml"
)
DEFAULT_TRANSFER_CONFIG_PATH = Path(
    "/Users/scasella/Downloads/hyperagents/HyperAgents/configs/baseline_freeze/shared_path_transfer_pilot.yaml"
)

MODULE_PREFIXES = (
    "domains",
    "utils",
    "agent",
    "task_agent",
    "meta_agent",
    "run_meta_agent",
    "generate_loop",
    "select_next_parent",
)

PYTHON_BIN = Path(__file__).resolve().parents[1] / ".venv312/bin/python"
HOST_REPO_ROOT = Path(__file__).resolve().parents[1]
IMO_POINT_MAP = {
    "incorrect": 0,
    "partial": 1,
    "almost": 6,
    "correct": 7,
}


@dataclass(frozen=True)
class EvalResult:
    run_id: str
    output_dir: Path
    predictions_path: Path
    report_path: Path
    report: dict[str, Any]


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _clear_snapshot_modules() -> None:
    for name in list(sys.modules):
        if name == "analysis.run_shared_path_source_search":
            continue
        if name in MODULE_PREFIXES or any(name.startswith(prefix + ".") for prefix in MODULE_PREFIXES):
            del sys.modules[name]


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _run_subprocess(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed in {cwd}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )


def _domain_utils(domain: str):
    utils_prefix = domain.split("_", 1)[1] + "_" if domain.startswith("imo_") else ""
    domain_folder = domain.split("_")[0] if "imo_" in domain else domain
    return importlib.import_module(f"domains.{domain_folder}.{utils_prefix}utils")


def _dataset_path(domain: str, subset: str) -> Path:
    if domain.startswith("imo_"):
        return HOST_REPO_ROOT / f"domains/imo/{domain.split('_')[-1]}bench{subset}.csv"
    return HOST_REPO_ROOT / f"domains/{domain}/dataset{subset}.csv"


def _load_dataset(domain: str, subset: str, num_samples: int) -> pd.DataFrame:
    df = pd.read_csv(_dataset_path(domain, subset), dtype=str)
    if num_samples > 0:
        df = df.iloc[:num_samples].copy()
    return df


def _predictions_complete(
    *,
    domain: str,
    subset: str,
    num_samples: int,
    predictions_path: Path,
) -> bool:
    if not predictions_path.exists():
        return False
    try:
        predictions_df = pd.read_csv(predictions_path, dtype=str)
    except Exception:
        return False
    if "prediction" not in predictions_df.columns:
        return False
    expected = _load_dataset(domain, subset, num_samples)
    utils_module = _domain_utils(domain)
    question_id_col = utils_module.QUESTION_ID
    if len(predictions_df) != len(expected):
        return False
    if question_id_col not in predictions_df.columns:
        return False
    expected_ids = expected[question_id_col].fillna("").tolist()
    actual_ids = predictions_df[question_id_col].fillna("").tolist()
    if actual_ids != expected_ids:
        return False
    normalized_predictions = (
        predictions_df["prediction"].fillna("").astype(str).str.strip().str.lower()
    )
    return bool((normalized_predictions != "").all())


def _run_harness_with_completion_rescue(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    domain: str,
    subset: str,
    num_samples: int,
    predictions_path: Path,
    total_timeout_seconds: int = 1800,
    completion_grace_seconds: int = 15,
) -> None:
    if _predictions_complete(
        domain=domain,
        subset=subset,
        num_samples=num_samples,
        predictions_path=predictions_path,
    ):
        return

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    start_time = time.monotonic()
    completed_since: float | None = None

    while True:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(
                    f"Command failed in {cwd}.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"STDOUT:\n{stdout}\n"
                    f"STDERR:\n{stderr}"
                )
            return

        if _predictions_complete(
            domain=domain,
            subset=subset,
            num_samples=num_samples,
            predictions_path=predictions_path,
        ):
            if completed_since is None:
                completed_since = time.monotonic()
            elif time.monotonic() - completed_since >= completion_grace_seconds:
                process.terminate()
                try:
                    stdout, stderr = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate(timeout=10)
                if process.returncode not in (0, -15):
                    raise RuntimeError(
                        f"Command stalled after writing complete predictions in {cwd}.\n"
                        f"Command: {' '.join(cmd)}\n"
                        f"STDOUT:\n{stdout}\n"
                        f"STDERR:\n{stderr}"
                    )
                return
        else:
            completed_since = None

        if time.monotonic() - start_time >= total_timeout_seconds:
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate(timeout=10)
            raise RuntimeError(
                f"Command timed out in {cwd}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{stdout}\n"
                f"STDERR:\n{stderr}"
            )

        time.sleep(2)


def _run_eval(
    *,
    repo_root: Path,
    domain: str,
    run_id: str,
    output_dir: Path,
    subset: str,
    num_samples: int,
    model: str,
    num_workers: int,
    save_interval: int,
    env: dict[str, str],
) -> EvalResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir = output_dir / run_id
    predictions_path = run_output_dir / "predictions.csv"
    harness_cmd = [
        str(PYTHON_BIN),
        "-m",
        "domains.harness",
        "--domain",
        domain,
        "--run_id",
        run_id,
        "--output_dir",
        str(output_dir),
        "--subset",
        subset,
        "--num_samples",
        str(num_samples),
        "--num_workers",
        str(num_workers),
        "--save_interval",
        str(save_interval),
        "--model",
        model,
    ]
    _run_harness_with_completion_rescue(
        cmd=harness_cmd,
        cwd=repo_root,
        env=env,
        domain=domain,
        subset=subset,
        num_samples=num_samples,
        predictions_path=predictions_path,
    )

    _run_subprocess(
        [
            str(PYTHON_BIN),
            "-m",
            "domains.report",
            "--domain",
            domain,
            "--dname",
            str(run_output_dir),
        ],
        cwd=repo_root,
        env=env,
    )
    report_path = run_output_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return EvalResult(
        run_id=run_id,
        output_dir=run_output_dir,
        predictions_path=predictions_path,
        report_path=report_path,
        report=report,
    )


def _score_label_error(domain: str, prediction: str, truth: str) -> float:
    prediction = (prediction or "").strip().lower()
    truth = (truth or "").strip().lower()
    if domain == "imo_grading":
        return abs(IMO_POINT_MAP.get(prediction, 0) - IMO_POINT_MAP.get(truth, 0)) / 7.0
    return 0.0 if prediction == truth else 1.0


def _prediction_change_summary(
    *,
    domain: str,
    subset: str,
    num_samples: int,
    baseline_predictions_path: Path,
    candidate_predictions_path: Path,
) -> dict[str, Any]:
    utils_module = _domain_utils(domain)
    question_id_col = utils_module.QUESTION_ID
    ground_truth_key = utils_module.GROUND_TRUTH_KEY

    dataset = _load_dataset(domain, subset, num_samples)
    baseline_df = pd.read_csv(baseline_predictions_path, dtype=str)
    candidate_df = pd.read_csv(candidate_predictions_path, dtype=str)

    columns = [question_id_col, "prediction"]
    merged = dataset.merge(
        baseline_df[columns].rename(columns={"prediction": "baseline_prediction"}),
        on=question_id_col,
        how="left",
    ).merge(
        candidate_df[columns].rename(columns={"prediction": "candidate_prediction"}),
        on=question_id_col,
        how="left",
    )
    merged["baseline_prediction"] = merged["baseline_prediction"].fillna("").str.strip().str.lower()
    merged["candidate_prediction"] = merged["candidate_prediction"].fillna("").str.strip().str.lower()
    merged[ground_truth_key] = merged[ground_truth_key].fillna("").str.strip().str.lower()

    changed_examples: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        if row["baseline_prediction"] == row["candidate_prediction"]:
            continue
        baseline_error = _score_label_error(domain, row["baseline_prediction"], row[ground_truth_key])
        candidate_error = _score_label_error(domain, row["candidate_prediction"], row[ground_truth_key])
        if candidate_error < baseline_error:
            delta_type = "better"
        elif candidate_error > baseline_error:
            delta_type = "worse"
        else:
            delta_type = "different_same_score"
        changed_examples.append(
            {
                "question_id": row[question_id_col],
                "ground_truth": row[ground_truth_key],
                "baseline_prediction": row["baseline_prediction"],
                "candidate_prediction": row["candidate_prediction"],
                "baseline_error": baseline_error,
                "candidate_error": candidate_error,
                "delta_type": delta_type,
            }
        )

    return {
        "changed_prediction_count": len(changed_examples),
        "changed_examples": changed_examples,
        "better_count": sum(1 for row in changed_examples if row["delta_type"] == "better"),
        "worse_count": sum(1 for row in changed_examples if row["delta_type"] == "worse"),
        "same_score_count": sum(1 for row in changed_examples if row["delta_type"] == "different_same_score"),
    }


def _copy_snapshot(source_root: Path, destination_root: Path) -> None:
    if destination_root.exists():
        shutil.rmtree(destination_root)
    shutil.copytree(
        source_root,
        destination_root,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".venv",
            ".venv312",
            "venv",
            "outputs",
            "analysis/outputs",
        ),
    )


def _render_directory_diff(base_root: Path, candidate_root: Path) -> str:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--no-index",
            "--",
            str(base_root),
            str(candidate_root),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"git diff failed: {result.stderr}")
    return result.stdout


def _write_attempt_log(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def logger(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    return logger


def _build_edit_policy_text(allowlist: dict[str, Any]) -> str:
    shared_lines = []
    for entry in allowlist.get("shared_editable_symbols", []):
        shared_lines.append(
            f"- {entry['symbol_id']} at {entry['path']} lines {entry['lines'][0]}-{entry['lines'][1]}"
        )
    forbidden_lines = []
    for bucket_name in ("source_only", "target_only"):
        for entry in allowlist.get("forbidden_symbols", {}).get(bucket_name, []):
            forbidden_lines.append(f"- {entry['symbol_id']}")
    for mixed in allowlist.get("mixed_symbols", []):
        forbidden_lines.append(f"- {mixed['symbol_id']} (mixed domain-local branches)")

    return (
        "Shared-path transfer gate for this run.\n"
        "You must stay inside the shared executed surface below. Any edit outside it will be discarded and not evaluated.\n\n"
        "Allowed shared edit surface:\n"
        + ("\n".join(shared_lines) if shared_lines else "- none\n")
        + "\n\nForbidden edits:\n"
        + ("\n".join(forbidden_lines) if forbidden_lines else "- none\n")
        + "\n\nAdditional hard rules:\n"
        "- Do not edit paper_review-only prompt or parser code.\n"
        "- Do not edit imo_grading-only prompt or parser code.\n"
        "- Do not edit docs, reports, configs, outputs, datasets, or caches.\n"
        "- Make one small code change and stop.\n"
    )


def _run_meta_edit_attempt(
    *,
    candidate_root: Path,
    eval_path: Path,
    model: str,
    instruction: str,
    chat_log_path: Path,
) -> None:
    with _pushd(candidate_root):
        sys.path.insert(0, str(candidate_root))
        try:
            _clear_snapshot_modules()
            llm_withtools = importlib.import_module("agent.llm_withtools")
            logger = _write_attempt_log(chat_log_path)
            full_instruction = (
                "You are improving a self-improving evaluation system.\n"
                f"Repository: `{candidate_root}`\n"
                f"Previous paper_review evaluation outputs: `{eval_path}`\n\n"
                "Goal: make one small code change that improves paper_review validation behavior while staying transfer-eligible for imo_grading.\n"
                "Start from the evaluation evidence, inspect only the most relevant files, make a focused change, and stop.\n"
                "Prefer shared task/output handling, shared provider request formatting, or shared task-agent scaffolding.\n"
                "Do not waste turns on directory listings.\n\n"
                f"{instruction}"
            )
            llm_withtools.chat_with_agent(
                full_instruction,
                model=model,
                msg_history=[],
                logging=logger,
                tools_available="all",
            )
        finally:
            try:
                sys.path.remove(str(candidate_root))
            except ValueError:
                pass


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_improved(baseline: EvalResult, candidate: EvalResult) -> bool:
    baseline_valid = baseline.report.get("valid_label_rate")
    candidate_valid = candidate.report.get("valid_label_rate")
    if baseline_valid is not None and candidate_valid is not None and candidate_valid < baseline_valid:
        return False
    return float(candidate.report.get("overall_accuracy", 0.0)) > float(baseline.report.get("overall_accuracy", 0.0))


def _write_source_report(
    *,
    report_path: Path,
    config: dict[str, Any],
    baseline_screen: EvalResult,
    baseline_confirm: EvalResult,
    attempts: list[dict[str, Any]],
    selected: dict[str, Any] | None,
) -> None:
    lines = [
        "# Shared-Path Source Results",
        "",
        "## Setup",
        "",
        f"- base snapshot: `{config['base_root']}`",
        f"- allowlist: `{config['allowlist_path']}`",
        f"- source domain: `{config['source_domain']}`",
        f"- source subset: `{config['source_subset']}`",
        f"- eval model: `{config['eval_model']}`",
        f"- meta-search model: `{config['meta_search_model']}`",
        f"- max attempts: `{config['max_attempts']}`",
        "",
        "## Frozen source baseline",
        "",
        f"- val-10 accuracy: `{baseline_screen.report.get('overall_accuracy')}`",
        f"- val-10 valid-label rate: `{baseline_screen.report.get('valid_label_rate')}`",
        f"- val-25 accuracy: `{baseline_confirm.report.get('overall_accuracy')}`",
        f"- val-25 valid-label rate: `{baseline_confirm.report.get('valid_label_rate')}`",
        "",
        "## Attempt summary",
        "",
        f"- total attempts: `{len(attempts)}`",
        f"- rejected ineligible: `{sum(1 for item in attempts if item['gate_verdict'] == 'ineligible')}`",
        f"- rejected ambiguous: `{sum(1 for item in attempts if item['gate_verdict'] == 'ambiguous')}`",
        f"- eligible candidates evaluated: `{sum(1 for item in attempts if item['gate_verdict'] == 'eligible')}`",
        "",
    ]
    for attempt in attempts:
        lines.extend(
            [
                f"### Attempt {attempt['attempt_index']}",
                "",
                f"- gate verdict: `{attempt['gate_verdict']}`",
                f"- gate summary: `{attempt['gate_reason']}`",
                f"- changed files: `{', '.join(attempt['changed_files']) if attempt['changed_files'] else 'none'}`",
            ]
        )
        if attempt.get("screen_report"):
            screen_report = attempt["screen_report"]
            lines.append(
                f"- source val-10: accuracy `{screen_report['overall_accuracy']}`, valid-label rate `{screen_report.get('valid_label_rate')}`"
            )
        if attempt.get("confirm_report"):
            confirm_report = attempt["confirm_report"]
            lines.append(
                f"- source val-25: accuracy `{confirm_report['overall_accuracy']}`, valid-label rate `{confirm_report.get('valid_label_rate')}`"
            )
        lines.append("")

    lines.extend(
        [
            "## Selected patch",
            "",
        ]
    )
    if selected is None:
        lines.extend(
            [
                "No transfer-eligible patch improved `paper_review` validation under the shared-path gate.",
                "",
                "This cycle therefore ends as a structural blocker result:",
                "",
                "> We could not generate any source-improving patch while restricting edits to the shared exercised surface.",
            ]
        )
    else:
        lines.extend(
            [
                f"- attempt: `{selected['attempt_index']}`",
                f"- selected repo: `{selected['candidate_root']}`",
                f"- selected diff: `{selected['diff_path']}`",
                f"- changed files/functions: `{', '.join(selected['changed_files'])}`",
                f"- confirmed val-25 accuracy: `{selected['confirm_report']['overall_accuracy']}`",
                f"- confirmed val-25 valid-label rate: `{selected['confirm_report'].get('valid_label_rate')}`",
            ]
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_transfer_report(
    *,
    report_path: Path,
    config: dict[str, Any],
    source_selected: dict[str, Any] | None,
    baseline_smoke: EvalResult | None,
    candidate_smoke: EvalResult | None,
    baseline_real: EvalResult | None,
    candidate_real: EvalResult | None,
    baseline_real_changes: dict[str, Any] | None,
    negative_control: dict[str, Any] | None,
) -> None:
    lines = [
        "# Shared-Path Transfer Pilot",
        "",
        "## Frozen setup",
        "",
        f"- baseline snapshot: `{config['baseline_root']}`",
        f"- target domain: `{config['target_domain']}`",
        f"- target subset: `{config['target_subset']}`",
        f"- eval model: `{config['eval_model']}`",
        "",
    ]
    if source_selected is None:
        lines.extend(
            [
                "## Outcome",
                "",
                "Transfer was not run.",
                "",
                "Reason: no transfer-eligible source patch improved `paper_review`, so the source stage failed the required gate.",
            ]
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.extend(
        [
            "## Source patch",
            "",
            f"- selected attempt: `{source_selected['attempt_index']}`",
            f"- selected candidate root: `{source_selected['candidate_root']}`",
            f"- selected diff: `{source_selected['diff_path']}`",
            f"- selected changed files/functions: `{', '.join(source_selected['changed_files'])}`",
            "",
            "## Metrics",
            "",
            "| Arm | Valid-label rate | Accuracy | Normalized MAE | Changed predictions vs baseline |",
            "| --- | --- | --- | --- | --- |",
            f"| baseline smoke | `{baseline_smoke.report.get('valid_label_rate')}` | `{baseline_smoke.report.get('overall_accuracy')}` | `{baseline_smoke.report.get('normalized_mean_absolute_error')}` | `n/a` |",
            f"| shared-path transfer smoke | `{candidate_smoke.report.get('valid_label_rate')}` | `{candidate_smoke.report.get('overall_accuracy')}` | `{candidate_smoke.report.get('normalized_mean_absolute_error')}` | `n/a` |",
            f"| baseline real | `{baseline_real.report.get('valid_label_rate')}` | `{baseline_real.report.get('overall_accuracy')}` | `{baseline_real.report.get('normalized_mean_absolute_error')}` | `0` |",
            f"| shared-path transfer real | `{candidate_real.report.get('valid_label_rate')}` | `{candidate_real.report.get('overall_accuracy')}` | `{candidate_real.report.get('normalized_mean_absolute_error')}` | `{baseline_real_changes['changed_prediction_count']}` |",
        ]
    )
    if negative_control is not None:
        lines.append(
            f"| negative control reused ({negative_control['run_id']}) | `{negative_control['report'].get('valid_label_rate')}` | `{negative_control['report'].get('overall_accuracy')}` | `{negative_control['report'].get('normalized_mean_absolute_error')}` | `{negative_control['changes'].get('changed_prediction_count')}` |"
        )

    lines.extend(
        [
            "",
            "## Changed target examples",
            "",
        ]
    )
    if baseline_real_changes["changed_prediction_count"] == 0:
        lines.append("No target predictions changed relative to the fresh repaired baseline.")
    else:
        for item in baseline_real_changes["changed_examples"]:
            lines.append(
                f"- `{item['question_id']}`: `{item['baseline_prediction']}` -> `{item['candidate_prediction']}` against ground truth `{item['ground_truth']}` ({item['delta_type']})"
            )

    full_helped = float(candidate_real.report.get("overall_accuracy", 0.0)) > float(
        baseline_real.report.get("overall_accuracy", 0.0)
    ) and float(candidate_real.report.get("valid_label_rate", 0.0)) >= float(
        baseline_real.report.get("valid_label_rate", 0.0)
    )
    lines.extend(
        [
            "",
            "## Judgment",
            "",
            f"- transfer helped: `{'yes' if full_helped else 'no'}`",
            f"- accuracy delta vs baseline: `{float(candidate_real.report.get('overall_accuracy', 0.0)) - float(baseline_real.report.get('overall_accuracy', 0.0)):.3f}`",
            f"- normalized MAE delta vs baseline: `{float(candidate_real.report.get('normalized_mean_absolute_error', 0.0)) - float(baseline_real.report.get('normalized_mean_absolute_error', 0.0)):.3f}`",
        ]
    )
    if full_helped:
        lines.extend(
            [
                "",
                "> We generated a transfer-eligible shared-path source patch, it improved `paper_review`, and it improved `imo_grading`.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "> We generated a transfer-eligible shared-path source patch, it improved `paper_review`, but it did not help `imo_grading`.",
            ]
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_source_search(
    *,
    config_path: Path,
    transfer_config_path: Path | None = None,
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    allowlist = _load_yaml(Path(config["allowlist_path"]))
    env = os.environ.copy()

    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_output_dir = output_root / "source_baseline"
    attempts_output_dir = output_root / "attempts"
    selected_output_dir = output_root / "selected"
    receipts_dir = output_root / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)

    base_root = Path(config["base_root"]).resolve()
    policy_text = _build_edit_policy_text(allowlist)

    baseline_screen = _run_eval(
        repo_root=base_root,
        domain=config["source_domain"],
        run_id="shared_path_source_baseline_val10",
        output_dir=baseline_output_dir,
        subset=config["source_subset"],
        num_samples=int(config["source_screen_samples"]),
        model=config["eval_model"],
        num_workers=int(config["num_workers"]),
        save_interval=int(config["save_interval"]),
        env=env,
    )
    baseline_confirm = _run_eval(
        repo_root=base_root,
        domain=config["source_domain"],
        run_id="shared_path_source_baseline_val25",
        output_dir=baseline_output_dir,
        subset=config["source_subset"],
        num_samples=int(config["source_confirm_samples"]),
        model=config["eval_model"],
        num_workers=int(config["num_workers"]),
        save_interval=int(config["save_interval"]),
        env=env,
    )

    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    eligible_evaluations = 0

    for attempt_index in range(1, int(config["max_attempts"]) + 1):
        attempt_root = attempts_output_dir / f"attempt_{attempt_index:02d}"
        candidate_root = attempt_root / "repo"
        _copy_snapshot(base_root, candidate_root)

        meta_log_path = attempt_root / "meta_agent.log"
        _run_meta_edit_attempt(
            candidate_root=candidate_root,
            eval_path=baseline_confirm.output_dir,
            model=config["meta_search_model"],
            instruction=policy_text,
            chat_log_path=meta_log_path,
        )

        diff_text = _render_directory_diff(base_root, candidate_root)
        diff_path = attempt_root / "candidate.diff"
        diff_path.write_text(diff_text, encoding="utf-8")

        gate_summary = validate_candidate_snapshot(
            base_root=base_root,
            candidate_root=candidate_root,
            allowlist_path=Path(config["allowlist_path"]),
        )
        gate_path = attempt_root / "gate_validation.json"
        _write_json(gate_path, gate_summary)

        changed_files = sorted({change["file_path"] for change in gate_summary.get("changes", [])})
        gate_reason = gate_summary.get("reason") or ", ".join(
            sorted({change["reason"] for change in gate_summary.get("changes", [])})
        )

        attempt_record: dict[str, Any] = {
            "attempt_index": attempt_index,
            "candidate_root": str(candidate_root),
            "diff_path": str(diff_path),
            "gate_verdict": gate_summary["verdict"],
            "gate_reason": gate_reason,
            "changed_files": changed_files,
        }

        if gate_summary["verdict"] != "eligible":
            attempts.append(attempt_record)
            continue

        eligible_evaluations += 1
        screen_eval = _run_eval(
            repo_root=candidate_root,
            domain=config["source_domain"],
            run_id=f"shared_path_candidate_{attempt_index:02d}_val10",
            output_dir=attempt_root / "eval_outputs",
            subset=config["source_subset"],
            num_samples=int(config["source_screen_samples"]),
            model=config["eval_model"],
            num_workers=int(config["num_workers"]),
            save_interval=int(config["save_interval"]),
            env=env,
        )
        attempt_record["screen_report"] = screen_eval.report
        attempt_record["screen_predictions"] = str(screen_eval.predictions_path)

        if not _source_improved(baseline_screen, screen_eval):
            attempts.append(attempt_record)
            if eligible_evaluations >= int(config["max_eligible_evaluations"]):
                break
            continue

        confirm_eval = _run_eval(
            repo_root=candidate_root,
            domain=config["source_domain"],
            run_id=f"shared_path_candidate_{attempt_index:02d}_val25",
            output_dir=attempt_root / "eval_outputs",
            subset=config["source_subset"],
            num_samples=int(config["source_confirm_samples"]),
            model=config["eval_model"],
            num_workers=int(config["num_workers"]),
            save_interval=int(config["save_interval"]),
            env=env,
        )
        attempt_record["confirm_report"] = confirm_eval.report
        attempt_record["confirm_predictions"] = str(confirm_eval.predictions_path)

        if _source_improved(baseline_confirm, confirm_eval):
            selected_candidate_root = selected_output_dir / "candidate_repo"
            _copy_snapshot(candidate_root, selected_candidate_root)
            selected_diff_path = selected_output_dir / "selected_candidate.diff"
            selected_diff_path.write_text(diff_text, encoding="utf-8")
            selected = {
                **attempt_record,
                "candidate_root": str(selected_candidate_root),
                "diff_path": str(selected_diff_path),
                "screen_report": screen_eval.report,
                "confirm_report": confirm_eval.report,
            }
            attempts.append(attempt_record)
            break

        attempts.append(attempt_record)
        if eligible_evaluations >= int(config["max_eligible_evaluations"]):
            break

    source_summary = {
        "config_path": str(config_path),
        "allowlist_path": config["allowlist_path"],
        "base_root": config["base_root"],
        "baseline_screen_report": baseline_screen.report,
        "baseline_confirm_report": baseline_confirm.report,
        "attempts": attempts,
        "selected": selected,
    }
    _write_json(receipts_dir / "shared_path_source_search_summary.json", source_summary)

    source_report_path = Path(config["source_report_path"]).resolve()
    _write_source_report(
        report_path=source_report_path,
        config=config,
        baseline_screen=baseline_screen,
        baseline_confirm=baseline_confirm,
        attempts=attempts,
        selected=selected,
    )

    transfer_summary: dict[str, Any] | None = None
    transfer_report_path = None

    if transfer_config_path is not None:
        transfer_config = _load_yaml(transfer_config_path)
        transfer_report_path = Path(transfer_config["report_path"]).resolve()
        if selected is None:
            _write_transfer_report(
                report_path=transfer_report_path,
                config=transfer_config,
                source_selected=None,
                baseline_smoke=None,
                candidate_smoke=None,
                baseline_real=None,
                candidate_real=None,
                baseline_real_changes=None,
                negative_control=None,
            )
        else:
            selected_root = Path(selected["candidate_root"]).resolve()
            target_output_dir = Path(transfer_config["output_root"]).resolve()
            baseline_smoke = _run_eval(
                repo_root=Path(transfer_config["baseline_root"]).resolve(),
                domain=transfer_config["target_domain"],
                run_id="shared_path_transfer_baseline_val10_smoke",
                output_dir=target_output_dir,
                subset=transfer_config["target_subset"],
                num_samples=int(transfer_config["smoke_num_samples"]),
                model=transfer_config["eval_model"],
                num_workers=int(transfer_config["num_workers"]),
                save_interval=int(transfer_config["save_interval"]),
                env=env,
            )
            candidate_smoke = _run_eval(
                repo_root=selected_root,
                domain=transfer_config["target_domain"],
                run_id="shared_path_transfer_candidate_val10_smoke",
                output_dir=target_output_dir,
                subset=transfer_config["target_subset"],
                num_samples=int(transfer_config["smoke_num_samples"]),
                model=transfer_config["eval_model"],
                num_workers=int(transfer_config["num_workers"]),
                save_interval=int(transfer_config["save_interval"]),
                env=env,
            )
            baseline_real = _run_eval(
                repo_root=Path(transfer_config["baseline_root"]).resolve(),
                domain=transfer_config["target_domain"],
                run_id="shared_path_transfer_baseline_val25",
                output_dir=target_output_dir,
                subset=transfer_config["target_subset"],
                num_samples=int(transfer_config["real_num_samples"]),
                model=transfer_config["eval_model"],
                num_workers=int(transfer_config["num_workers"]),
                save_interval=int(transfer_config["save_interval"]),
                env=env,
            )
            candidate_real = _run_eval(
                repo_root=selected_root,
                domain=transfer_config["target_domain"],
                run_id="shared_path_transfer_candidate_val25",
                output_dir=target_output_dir,
                subset=transfer_config["target_subset"],
                num_samples=int(transfer_config["real_num_samples"]),
                model=transfer_config["eval_model"],
                num_workers=int(transfer_config["num_workers"]),
                save_interval=int(transfer_config["save_interval"]),
                env=env,
            )
            baseline_real_changes = _prediction_change_summary(
                domain=transfer_config["target_domain"],
                subset=transfer_config["target_subset"],
                num_samples=int(transfer_config["real_num_samples"]),
                baseline_predictions_path=baseline_real.predictions_path,
                candidate_predictions_path=candidate_real.predictions_path,
            )

            negative_control = None
            negative_cfg = transfer_config.get("negative_control")
            if negative_cfg and negative_cfg.get("enabled", False):
                negative_report = json.loads(Path(negative_cfg["report_json"]).read_text(encoding="utf-8"))
                negative_changes = _prediction_change_summary(
                    domain=transfer_config["target_domain"],
                    subset=transfer_config["target_subset"],
                    num_samples=int(transfer_config["real_num_samples"]),
                    baseline_predictions_path=baseline_real.predictions_path,
                    candidate_predictions_path=Path(negative_cfg["predictions_csv"]),
                )
                negative_control = {
                    "run_id": negative_cfg["run_id"],
                    "report": negative_report,
                    "changes": negative_changes,
                }

            transfer_summary = {
                "baseline_smoke": baseline_smoke.report,
                "candidate_smoke": candidate_smoke.report,
                "baseline_real": baseline_real.report,
                "candidate_real": candidate_real.report,
                "real_prediction_changes": baseline_real_changes,
                "negative_control": negative_control,
            }
            _write_json(receipts_dir / "shared_path_transfer_summary.json", transfer_summary)
            _write_transfer_report(
                report_path=transfer_report_path,
                config=transfer_config,
                source_selected=selected,
                baseline_smoke=baseline_smoke,
                candidate_smoke=candidate_smoke,
                baseline_real=baseline_real,
                candidate_real=candidate_real,
                baseline_real_changes=baseline_real_changes,
                negative_control=negative_control,
            )

    return {
        "source_summary": source_summary,
        "transfer_summary": transfer_summary,
        "source_report_path": str(source_report_path),
        "transfer_report_path": str(transfer_report_path) if transfer_report_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a shared-path-constrained source search and optional transfer pilot.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--transfer-config", type=Path, default=DEFAULT_TRANSFER_CONFIG_PATH)
    args = parser.parse_args()

    result = run_source_search(
        config_path=args.config,
        transfer_config_path=args.transfer_config if args.transfer_config.exists() else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
