"""Benchmark overlap-guided shared patches and optionally transfer the best winner."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.run_shared_path_source_search import EvalResult, _prediction_change_summary, _run_eval  # noqa: E402
from analysis.run_shared_patch_benchmark import (  # noqa: E402
    _apply_shared_instruction_wrapper_tightening,
    _apply_shared_truncated_json_label_salvage,
)
from analysis.validate_transfer_eligibility import validate_candidate_snapshot  # noqa: E402


DEFAULT_CONFIG_PATH = Path(
    "/Users/scasella/Downloads/hyperagents/HyperAgents/configs/baseline_freeze/overlap_guided_transfer.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        ),
    )


def _replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise ValueError(f"Expected snippet not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _apply_shared_common_wrapper_minimal_json(candidate_root: Path) -> list[dict[str, Any]]:
    path = candidate_root / "utils/prediction_contracts.py"
    old = """def build_task_instruction(inputs: dict[str, Any]) -> str:
    contract = get_prediction_contract(inputs["domain"])
    if contract is None:
        return f\"\"\"You are an agent.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": ...
}}
</json>\"\"\"
    return contract.build_instruction(inputs)
"""
    new = """def build_task_instruction(inputs: dict[str, Any]) -> str:
    shared_wrapper = (
        "Shared response rules:\\n"
        "- Reply with the smallest valid JSON object that solves the task.\\n"
        "- Do not add any prose before the JSON object.\\n"
        "- Keep optional explanation text minimal.\\n"
        "- Stop immediately after the JSON object."
    )
    contract = get_prediction_contract(inputs["domain"])
    if contract is None:
        return shared_wrapper + "\\n\\n" + f\"\"\"You are an agent.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": ...
}}
</json>\"\"\"
    return shared_wrapper + "\\n\\n" + contract.build_instruction(inputs)
"""
    _replace_once(path, old, new)
    return [
        {
            "path": "utils/prediction_contracts.py",
            "symbol": "build_task_instruction",
            "mechanism": "Add a compact shared wrapper before both generic and contract task instructions.",
        }
    ]


def _apply_shared_moderate_json_output_budget(candidate_root: Path) -> list[dict[str, Any]]:
    path = candidate_root / "agent/llm.py"
    old = """    response_schema = _infer_gemini_response_schema(msg)
    if response_schema is not None:
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = response_schema
"""
    new = """    response_schema = _infer_gemini_response_schema(msg)
    if response_schema is not None:
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = response_schema
        generation_config["maxOutputTokens"] = min(generation_config["maxOutputTokens"], 1024)
"""
    _replace_once(path, old, new)
    return [
        {
            "path": "agent/llm.py",
            "symbol": "_get_response_from_gemini_rest",
            "mechanism": "Use a moderate shared output cap for JSON-style task responses.",
        }
    ]


PATCH_APPLIERS = {
    "shared_common_wrapper_minimal_json": _apply_shared_common_wrapper_minimal_json,
    "shared_truncated_json_label_salvage": _apply_shared_truncated_json_label_salvage,
    "shared_instruction_wrapper_tightening": _apply_shared_instruction_wrapper_tightening,
    "shared_moderate_json_output_budget": _apply_shared_moderate_json_output_budget,
}


def _apply_variant(candidate_root: Path, variant_id: str) -> list[dict[str, Any]]:
    if variant_id not in PATCH_APPLIERS:
        raise KeyError(f"Unknown variant_id: {variant_id}")
    return PATCH_APPLIERS[variant_id](candidate_root)


def _source_metric_key(domain: str) -> str:
    if domain == "paper_review":
        return "overall_accuracy"
    return "structured_prediction_rate"


def _structured_prediction_rate(predictions_path: Path) -> float:
    df = pd.read_csv(predictions_path, dtype=str)
    if "prediction" not in df.columns or len(df) == 0:
        return 0.0
    preds = df["prediction"].fillna("").astype(str).str.strip()
    return float((preds != "").mean())


def _collect_source_metrics(domain: str, eval_result: EvalResult) -> dict[str, Any]:
    metrics = {
        "structured_prediction_rate": _structured_prediction_rate(eval_result.predictions_path),
        "report": eval_result.report,
    }
    if domain == "paper_review":
        metrics["overall_accuracy"] = float(eval_result.report.get("overall_accuracy") or 0.0)
        metrics["valid_label_rate"] = float(eval_result.report.get("valid_label_rate") or 0.0)
    return metrics


def _pick_variant_ids(selected_source: dict[str, Any], config: dict[str, Any]) -> list[str]:
    top_symptoms = [
        item["symptom"] for item in selected_source.get("top_shared_symptoms", [])[:2]
    ]
    variant_ids: list[str] = []
    for variant in config.get("patch_library", []):
        if any(symptom in variant.get("symptom_tags", []) for symptom in top_symptoms):
            variant_ids.append(variant["id"])
    for fallback in ("shared_common_wrapper_minimal_json", "shared_moderate_json_output_budget"):
        if fallback not in variant_ids and fallback in PATCH_APPLIERS:
            variant_ids.append(fallback)
    return variant_ids[:3]


def _render_source_report(
    *,
    selected_source: dict[str, Any] | None,
    baseline_metrics: dict[str, Any] | None,
    variant_results: list[dict[str, Any]],
    report_path: Path,
) -> None:
    lines = [
        "# Overlap-Guided Source Patch",
        "",
    ]
    if selected_source is None:
        lines.extend(
            [
                "No overlap-guided source patch benchmark was run.",
                "",
                "Reason: no source domain showed enough shared, patchable failure overlap with `imo_grading` to justify a source-side patch sprint.",
            ]
        )
    else:
        lines.extend(
            [
                f"- Chosen source: `{selected_source['domain']}`",
                f"- Targeted shared symptom cluster: `{', '.join(item['symptom'] for item in selected_source.get('top_shared_symptoms', [])) or 'none'}`",
                f"- Baseline source metric (`{selected_source['metric_key']}`): `{baseline_metrics.get(selected_source['metric_key'])}`",
                "",
                "## Candidate patch results",
                "",
            ]
        )
        for item in variant_results:
            lines.extend(
                [
                    f"### {item['variant_id']}",
                    "",
                    f"- Gate verdict: `{item['validation']['verdict']}`",
                    f"- Source metric: `{item.get('metric_value')}`",
                    f"- Structured prediction rate: `{item.get('structured_prediction_rate')}`",
                    f"- Changed predictions vs baseline: `{item.get('changes', {}).get('changed_prediction_count', 0)}`",
                    "",
                ]
            )
        winner = next((item for item in variant_results if item.get("selected")), None)
        if winner is None:
            lines.extend(
                [
                    "## Decision",
                    "",
                    "No eligible shared patch improved the chosen source domain enough to justify transfer.",
                ]
            )
        else:
            lines.extend(
                [
                    "## Decision",
                    "",
                    f"Selected winner: `{winner['variant_id']}`",
                    f"- Mechanism: `{winner['mechanism']}`",
                    f"- Source metric delta: `{winner['metric_delta']:+.3f}`",
                ]
            )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_transfer_report(
    *,
    selected_source: dict[str, Any] | None,
    winner: dict[str, Any] | None,
    baseline_target: EvalResult | None,
    target_eval: EvalResult | None,
    changes: dict[str, Any] | None,
    report_path: Path,
) -> None:
    lines = ["# Overlap-Guided Transfer Result", ""]
    if selected_source is None:
        lines.extend(
            [
                "No transfer run was executed.",
                "",
                "Reason: the overlap study found no source domain with enough shared, patchable failure overlap with `imo_grading`.",
                "",
                "> No available source domain in the current repo shows enough overlap with `imo_grading` to make this transfer claim plausible.",
            ]
        )
    elif winner is None:
        lines.extend(
            [
                "No transfer run was executed.",
                "",
                f"Reason: `{selected_source['domain']}` was selected by overlap, but no transfer-eligible shared patch produced a real source-side win.",
            ]
        )
    else:
        lines.extend(
            [
                f"- Chosen source: `{selected_source['domain']}`",
                f"- Winning patch: `{winner['variant_id']}`",
                f"- Baseline target accuracy: `{baseline_target.report.get('overall_accuracy')}`",
                f"- Baseline target normalized MAE: `{baseline_target.report.get('normalized_mean_absolute_error')}`",
                f"- Transferred accuracy: `{target_eval.report.get('overall_accuracy')}`",
                f"- Transferred normalized MAE: `{target_eval.report.get('normalized_mean_absolute_error')}`",
                f"- Changed predictions vs baseline: `{changes.get('changed_prediction_count')}`",
                "",
                "## Direct answers",
                "",
                f"- Was `paper_review` the wrong source? `{'Yes' if selected_source['domain'] != 'paper_review' else 'No.'}`",
                f"- Did another source show better failure-mode overlap? `{'Yes' if selected_source['domain'] != 'paper_review' else 'No.'}`",
                f"- Did the overlap-guided patch improve the source? `Yes.`",
                f"- Did it transfer to `imo_grading`? `{'Yes' if (float(target_eval.report.get('overall_accuracy') or 0.0) > float(baseline_target.report.get('overall_accuracy') or 0.0) or float(target_eval.report.get('normalized_mean_absolute_error') or 1.0) < float(baseline_target.report.get('normalized_mean_absolute_error') or 1.0)) else 'No.'}`",
            ]
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    config = _load_yaml(args.config)
    selection = _read_json(Path(config["selection_json"]))
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    selected_source = None
    if selection.get("selected_source") is not None:
        ranked = selection["ranked_candidates"]
        selected_source = next(
            item for item in ranked if item["domain"] == selection["selected_source"]["domain"]
        )
        top_contribs = selected_source["overlap"].get("contributions", [])
        selected_source["top_shared_symptoms"] = top_contribs
        selected_source["metric_key"] = _source_metric_key(selected_source["domain"])

    if selected_source is None:
        _render_source_report(
            selected_source=None,
            baseline_metrics=None,
            variant_results=[],
            report_path=Path(config["source_patch_report_path"]),
        )
        _render_transfer_report(
            selected_source=None,
            winner=None,
            baseline_target=None,
            target_eval=None,
            changes=None,
            report_path=Path(config["transfer_report_path"]),
        )
        return

    base_root = Path(config["base_root"])
    allowlist_path = Path(selected_source["allowlist_path"])
    source_domain = selected_source["domain"]
    source_subset = selected_source["subset"]
    source_num_samples = int(selected_source["num_samples"])

    env = os.environ.copy()
    baseline_source = _run_eval(
        repo_root=base_root,
        domain=source_domain,
        run_id=f"overlap_guided_{source_domain}_baseline_val{source_num_samples}",
        output_dir=output_root / "source_baseline",
        subset=source_subset,
        num_samples=source_num_samples,
        model=config["eval_model"],
        num_workers=int(config["num_workers"]),
        save_interval=int(config["save_interval"]),
        env=env,
    )
    baseline_metrics = _collect_source_metrics(source_domain, baseline_source)

    variant_results: list[dict[str, Any]] = []
    for variant_id in _pick_variant_ids(selected_source, config):
        candidate_root = output_root / "candidates" / variant_id / "repo"
        _copy_snapshot(base_root, candidate_root)
        changes = _apply_variant(candidate_root, variant_id)
        validation = validate_candidate_snapshot(
            base_root=base_root,
            candidate_root=candidate_root,
            allowlist_path=allowlist_path,
        )
        entry: dict[str, Any] = {
            "variant_id": variant_id,
            "mechanism": next(
                item["mechanism"] for item in config["patch_library"] if item["id"] == variant_id
            ),
            "changes": changes,
            "validation": validation,
        }
        if validation["verdict"] != "eligible":
            variant_results.append(entry)
            continue
        eval_result = _run_eval(
            repo_root=candidate_root,
            domain=source_domain,
            run_id=f"overlap_guided_{source_domain}_{variant_id}_val{source_num_samples}",
            output_dir=output_root / "candidate_evals" / variant_id,
            subset=source_subset,
            num_samples=source_num_samples,
            model=config["eval_model"],
            num_workers=int(config["num_workers"]),
            save_interval=int(config["save_interval"]),
            env=env,
        )
        source_metrics = _collect_source_metrics(source_domain, eval_result)
        metric_key = selected_source["metric_key"]
        baseline_value = float(baseline_metrics.get(metric_key) or 0.0)
        metric_value = float(source_metrics.get(metric_key) or 0.0)
        change_summary = _prediction_change_summary(
            baseline_predictions_path=baseline_source.predictions_path,
            candidate_predictions_path=eval_result.predictions_path,
            domain=source_domain,
        )
        entry.update(
            {
                "eval_result": {
                    "run_id": eval_result.run_id,
                    "report": eval_result.report,
                    "predictions_path": str(eval_result.predictions_path),
                },
                "metric_value": metric_value,
                "metric_delta": metric_value - baseline_value,
                "structured_prediction_rate": source_metrics["structured_prediction_rate"],
                "changes": change_summary,
            }
        )
        variant_results.append(entry)

    winners = [
        item
        for item in variant_results
        if item["validation"]["verdict"] == "eligible"
        and item.get("metric_delta", 0.0) > 0
    ]
    winners.sort(key=lambda item: (item["metric_delta"], item.get("structured_prediction_rate", 0.0)), reverse=True)
    winner = winners[0] if winners else None
    if winner is not None:
        winner["selected"] = True

    _render_source_report(
        selected_source=selected_source,
        baseline_metrics=baseline_metrics,
        variant_results=variant_results,
        report_path=Path(config["source_patch_report_path"]),
    )

    if winner is None:
        _render_transfer_report(
            selected_source=selected_source,
            winner=None,
            baseline_target=None,
            target_eval=None,
            changes=None,
            report_path=Path(config["transfer_report_path"]),
        )
        _write_json(output_root / "benchmark_summary.json", {"variant_results": variant_results})
        return

    baseline_target = _run_eval(
        repo_root=base_root,
        domain="imo_grading",
        run_id="overlap_guided_target_baseline_val25",
        output_dir=output_root / "target_baseline",
        subset="_filtered_100_val",
        num_samples=int(config["target_confirm_samples"]),
        model=config["eval_model"],
        num_workers=int(config["num_workers"]),
        save_interval=int(config["save_interval"]),
        env=env,
    )
    transferred_repo = output_root / "candidates" / winner["variant_id"] / "repo"
    target_eval = _run_eval(
        repo_root=transferred_repo,
        domain="imo_grading",
        run_id=f"overlap_guided_target_{winner['variant_id']}_val25",
        output_dir=output_root / "target_candidate",
        subset="_filtered_100_val",
        num_samples=int(config["target_confirm_samples"]),
        model=config["eval_model"],
        num_workers=int(config["num_workers"]),
        save_interval=int(config["save_interval"]),
        env=env,
    )
    changes = _prediction_change_summary(
        baseline_predictions_path=baseline_target.predictions_path,
        candidate_predictions_path=target_eval.predictions_path,
        domain="imo_grading",
    )
    _render_transfer_report(
        selected_source=selected_source,
        winner=winner,
        baseline_target=baseline_target,
        target_eval=target_eval,
        changes=changes,
        report_path=Path(config["transfer_report_path"]),
    )
    _write_json(
        output_root / "benchmark_summary.json",
        {
            "selected_source": selected_source["domain"],
            "baseline_source_metrics": baseline_metrics,
            "variant_results": variant_results,
            "winner": winner["variant_id"],
            "target_changes": changes,
            "target_report": target_eval.report,
        },
    )


if __name__ == "__main__":
    main()
