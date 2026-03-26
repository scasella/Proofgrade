"""Benchmark manual shared-path patch candidates and transfer the best winner."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.run_shared_path_source_search import (  # noqa: E402
    EvalResult,
    _prediction_change_summary,
    _run_eval,
)
from analysis.validate_transfer_eligibility import validate_candidate_snapshot  # noqa: E402
from analysis.build_shared_failure_atlas import _load_yaml  # noqa: E402


DEFAULT_CONFIG_PATH = Path(
    "/Users/scasella/Downloads/hyperagents/HyperAgents/configs/baseline_freeze/shared_patch_variants.yaml"
)


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


def _render_targeted_diff(base_root: Path, candidate_root: Path, changed_paths: list[str]) -> str:
    import subprocess

    chunks: list[str] = []
    for rel_path in changed_paths:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--no-index",
                "--",
                str(base_root / rel_path),
                str(candidate_root / rel_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(f"git diff failed for {rel_path}: {result.stderr}")
        if result.stdout:
            chunks.append(result.stdout)
    return "\n".join(chunks)


def _load_existing_eval(report_json: Path, predictions_csv: Path, run_id: str) -> EvalResult:
    return EvalResult(
        run_id=run_id,
        output_dir=report_json.parent,
        predictions_path=predictions_csv,
        report_path=report_json,
        report=_read_json(report_json),
    )


def _replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise ValueError(f"Expected snippet not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _serialize_eval_result(result: EvalResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "run_id": result.run_id,
        "output_dir": str(result.output_dir),
        "predictions_path": str(result.predictions_path),
        "report_path": str(result.report_path),
        "report": result.report,
    }


def _validation_reason(validation: dict[str, Any]) -> str:
    if validation.get("reason"):
        return str(validation["reason"])
    changes = validation.get("changes") or []
    if changes:
        return str(changes[0].get("reason", validation.get("verdict", "unknown")))
    return str(validation.get("verdict", "unknown"))


def _apply_shared_truncated_json_label_salvage(candidate_root: Path) -> list[dict[str, Any]]:
    path = candidate_root / "utils/prediction_contracts.py"
    old = """def _extract_json_label_candidate(raw_text: str) -> tuple[str | int | float | None, str]:
    objects = _extract_json_objects(raw_text)
    for obj in reversed(objects):
        for key in JSON_LABEL_KEYS:
            if key in obj:
                return obj[key], f"json:{key}"
    return None, "none"
"""
    new = """def _extract_json_label_candidate(raw_text: str) -> tuple[str | int | float | None, str]:
    objects = _extract_json_objects(raw_text)
    for obj in reversed(objects):
        for key in JSON_LABEL_KEYS:
            if key in obj:
                return obj[key], f"json:{key}"

    partial_match = re.search(
        r'\"(?P<key>label|decision|prediction|response)\"\\s*:\\s*\"(?P<value>[^\"\\r\\n]{1,32})\"',
        raw_text,
        re.IGNORECASE,
    )
    if partial_match:
        return partial_match.group("value"), f"partial_json:{partial_match.group('key').lower()}"
    return None, "none"
"""
    _replace_once(path, old, new)
    return [
        {
            "path": "utils/prediction_contracts.py",
            "symbol": "_extract_json_label_candidate",
            "mechanism": "Recover a visible label from truncated JSON.",
        }
    ]


def _apply_shared_instruction_wrapper_tightening(candidate_root: Path) -> list[dict[str, Any]]:
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

    shared_wrapper = (
        "Shared output rules for every classification task:\\n"
        "- Return the smallest valid JSON object.\\n"
        "- Put `label` first.\\n"
        "- Omit optional fields unless they are necessary.\\n"
        "- Do not restate the task input.\\n"
        "- Stop immediately after the JSON object."
    )
    return shared_wrapper + "\\n\\n" + contract.build_instruction(inputs)
"""
    _replace_once(path, old, new)
    return [
        {
            "path": "utils/prediction_contracts.py",
            "symbol": "build_task_instruction",
            "mechanism": "Add a compact shared wrapper around every task instruction.",
        }
    ]


def _apply_shared_compact_input_format(candidate_root: Path) -> list[dict[str, Any]]:
    path = candidate_root / "utils/prediction_contracts.py"
    old = """def _format_inputs(inputs: dict[str, Any]) -> str:
    return json.dumps(inputs, indent=2, ensure_ascii=True, sort_keys=True)
"""
    new = """def _format_inputs(inputs: dict[str, Any]) -> str:
    return json.dumps(inputs, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
"""
    _replace_once(path, old, new)
    return [
        {
            "path": "utils/prediction_contracts.py",
            "symbol": "_format_inputs",
            "mechanism": "Use a more compact shared task-input serialization.",
        }
    ]


def _apply_shared_short_label_budget(candidate_root: Path) -> list[dict[str, Any]]:
    path = candidate_root / "agent/llm.py"
    old = """    generation_config = {
        "temperature": temperature,
        # The preview model can spend a long time thinking unless we cap
        # the output budget. These task agents only need a short JSON label.
        "maxOutputTokens": min(max_tokens, GEMINI_PREVIEW_MAX_OUTPUT_TOKENS),
    }
    response_schema = _infer_gemini_response_schema(msg)
"""
    new = """    generation_config = {
        "temperature": temperature,
        # The preview model can spend a long time thinking unless we cap
        # the output budget. These task agents only need a short JSON label.
        "maxOutputTokens": min(max_tokens, GEMINI_PREVIEW_MAX_OUTPUT_TOKENS),
    }
    if "Allowed labels:" in msg and "JSON object" in msg:
        generation_config["maxOutputTokens"] = min(generation_config["maxOutputTokens"], 384)
    response_schema = _infer_gemini_response_schema(msg)
"""
    _replace_once(path, old, new)
    return [
        {
            "path": "agent/llm.py",
            "symbol": "_get_response_from_gemini_rest",
            "mechanism": "Clamp label-task output budget on the shared Gemini path.",
        }
    ]


PATCH_APPLIERS = {
    "shared_truncated_json_label_salvage": _apply_shared_truncated_json_label_salvage,
    "shared_instruction_wrapper_tightening": _apply_shared_instruction_wrapper_tightening,
    "shared_compact_input_format": _apply_shared_compact_input_format,
    "shared_short_label_budget": _apply_shared_short_label_budget,
}


def _apply_variant(candidate_root: Path, variant_id: str) -> list[dict[str, Any]]:
    if variant_id not in PATCH_APPLIERS:
        raise KeyError(f"Unknown variant_id: {variant_id}")
    return PATCH_APPLIERS[variant_id](candidate_root)


def _passes_source_screen(
    *,
    baseline: EvalResult,
    candidate: EvalResult,
    tolerance: float,
) -> bool:
    baseline_valid = float(baseline.report.get("valid_label_rate") or 0.0)
    candidate_valid = float(candidate.report.get("valid_label_rate") or 0.0)
    if candidate_valid + tolerance < baseline_valid:
        return False

    baseline_accuracy = float(baseline.report.get("overall_accuracy") or 0.0)
    candidate_accuracy = float(candidate.report.get("overall_accuracy") or 0.0)
    if candidate_accuracy > baseline_accuracy:
        return True

    baseline_invalid = int(baseline.report.get("invalid_prediction_count") or 0)
    candidate_invalid = int(candidate.report.get("invalid_prediction_count") or 0)
    return candidate_accuracy == baseline_accuracy and candidate_invalid < baseline_invalid


def _is_source_winner(
    *,
    baseline: EvalResult,
    candidate: EvalResult,
    tolerance: float,
) -> bool:
    baseline_valid = float(baseline.report.get("valid_label_rate") or 0.0)
    candidate_valid = float(candidate.report.get("valid_label_rate") or 0.0)
    if candidate_valid + tolerance < baseline_valid:
        return False
    return float(candidate.report.get("overall_accuracy") or 0.0) > float(
        baseline.report.get("overall_accuracy") or 0.0
    )


def _variant_sort_key(item: dict[str, Any]) -> tuple[float, float, int]:
    confirm = item["confirm_eval"].report
    changes = item["confirm_changes"]
    return (
        float(confirm.get("overall_accuracy") or 0.0),
        float(confirm.get("valid_label_rate") or 0.0),
        int(changes.get("better_count") or 0) - int(changes.get("worse_count") or 0),
    )


def _markdown_changes(changes: dict[str, Any], limit: int = 6) -> list[str]:
    lines = [
        f"- Changed predictions vs baseline: `{changes['changed_prediction_count']}`",
        f"- Better: `{changes['better_count']}`, worse: `{changes['worse_count']}`, same-score changes: `{changes['same_score_count']}`",
    ]
    for item in changes["changed_examples"][:limit]:
        lines.append(
            f"- `{item['question_id']}`: `{item['baseline_prediction']}` -> `{item['candidate_prediction']}` "
            f"(truth `{item['ground_truth']}`, outcome `{item['delta_type']}`)"
        )
    return lines


def _render_benchmark_report(
    *,
    config: dict[str, Any],
    atlas: dict[str, Any],
    baseline_screen: EvalResult,
    baseline_confirm: EvalResult,
    variant_results: list[dict[str, Any]],
    winner: dict[str, Any] | None,
) -> str:
    lines = [
        "# Shared Patch Benchmark",
        "",
        "## Frozen source baseline",
        "",
        f"- Source domain: `{config['source_domain']}`",
        f"- Model: `{config['eval_model']}`",
        f"- Val-10 accuracy: `{baseline_screen.report.get('overall_accuracy')}`",
        f"- Val-10 valid-label rate: `{baseline_screen.report.get('valid_label_rate')}`",
        f"- Val-25 accuracy: `{baseline_confirm.report.get('overall_accuracy')}`",
        f"- Val-25 valid-label rate: `{baseline_confirm.report.get('valid_label_rate')}`",
        "",
        "## Failure-atlas takeaway",
        "",
        f"- Shared-fixable invalid examples: `{atlas['shared_fixable_summary']['shared_fixable_invalid_examples']}`",
        f"- Wrong-but-valid examples outside the shared policy surface: `{atlas['shared_fixable_summary']['not_shared_fixable_valid_wrong_examples']}`",
        "",
        "## Variant results",
        "",
    ]
    for result in variant_results:
        variant = result["variant"]
        lines.extend(
            [
                f"### {variant['id']}",
                "",
                f"- Mechanism: `{variant['mechanism']}`",
                f"- Gate verdict: `{result['validation']['verdict']}`",
                f"- Changed files: `{', '.join(result['changed_files']) if result['changed_files'] else 'none'}`",
                f"- Diff: `{result['diff_path']}`",
            ]
        )
        if result["validation"]["verdict"] != "eligible":
            lines.append(f"- Skipped benchmarking because `{_validation_reason(result['validation'])}`")
            lines.append("")
            continue

        screen_eval = result["screen_eval"]
        lines.extend(
            [
                f"- Val-10 accuracy: `{screen_eval.report.get('overall_accuracy')}`",
                f"- Val-10 valid-label rate: `{screen_eval.report.get('valid_label_rate')}`",
            ]
        )
        lines.extend(_markdown_changes(result["screen_changes"], limit=3))

        if result["confirm_eval"] is not None:
            confirm_eval = result["confirm_eval"]
            lines.extend(
                [
                    f"- Val-25 accuracy: `{confirm_eval.report.get('overall_accuracy')}`",
                    f"- Val-25 valid-label rate: `{confirm_eval.report.get('valid_label_rate')}`",
                ]
            )
            lines.extend(_markdown_changes(result["confirm_changes"], limit=5))
        else:
            lines.append("- Did not advance to val-25.")
        lines.append("")

    lines.extend(["## Selected winner", ""])
    if winner is None:
        lines.extend(
            [
                "No shared patch improved `paper_review` validation enough to justify transfer.",
                "",
                "> Outcome: we manually designed transfer-eligible patches, but none produced a real source-side win.",
            ]
        )
    else:
        lines.extend(
            [
                f"- Winner: `{winner['variant']['id']}`",
                f"- Mechanism: `{winner['variant']['mechanism']}`",
                f"- Val-25 accuracy: `{winner['confirm_eval'].report.get('overall_accuracy')}`",
                f"- Val-25 valid-label rate: `{winner['confirm_eval'].report.get('valid_label_rate')}`",
                f"- Changed files/functions: `{', '.join(winner['changed_files'])}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _render_transfer_report(
    *,
    config: dict[str, Any],
    winner: dict[str, Any] | None,
    baseline_target: EvalResult,
    target_smoke: EvalResult | None,
    target_real: EvalResult | None,
    target_changes: dict[str, Any] | None,
) -> str:
    lines = [
        "# Shared Patch Transfer Result",
        "",
        "## Matched target setup",
        "",
        f"- Target domain: `{config['target_domain']}`",
        f"- Target subset: `{config['target_subset']}`",
        f"- Model: `{config['eval_model']}`",
        f"- Frozen target baseline accuracy: `{baseline_target.report.get('overall_accuracy')}`",
        f"- Frozen target baseline normalized MAE: `{baseline_target.report.get('normalized_mean_absolute_error')}`",
        f"- Frozen target baseline valid-label rate: `{baseline_target.report.get('valid_label_rate')}`",
        "",
    ]
    if winner is None:
        lines.extend(
            [
                "## Outcome",
                "",
                "Transfer was intentionally skipped.",
                "",
                "Reason: no shared patch cleared the source-side win gate on `paper_review`.",
                "",
                "## Direct answer",
                "",
                "A strong programmer could design transfer-eligible shared patches, but none improved the source domain enough to justify a transfer test.",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## Winner transferred",
            "",
            f"- Variant: `{winner['variant']['id']}`",
            f"- Intended mechanism: `{winner['variant']['mechanism']}`",
            f"- Source val-25 accuracy: `{winner['confirm_eval'].report.get('overall_accuracy')}`",
            f"- Source val-25 valid-label rate: `{winner['confirm_eval'].report.get('valid_label_rate')}`",
            "",
        ]
    )
    if target_smoke is not None:
        lines.extend(
            [
                "## Target smoke",
                "",
                f"- Val-10 accuracy: `{target_smoke.report.get('overall_accuracy')}`",
                f"- Val-10 normalized MAE: `{target_smoke.report.get('normalized_mean_absolute_error')}`",
                f"- Val-10 valid-label rate: `{target_smoke.report.get('valid_label_rate')}`",
                "",
            ]
        )
    if target_real is None or target_changes is None:
        lines.extend(
            [
                "## Outcome",
                "",
                "The transferred patch failed the target smoke gate, so the full target comparison was not run.",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## Target result",
            "",
            f"- Transferred accuracy: `{target_real.report.get('overall_accuracy')}`",
            f"- Transferred normalized MAE: `{target_real.report.get('normalized_mean_absolute_error')}`",
            f"- Transferred valid-label rate: `{target_real.report.get('valid_label_rate')}`",
            f"- Accuracy delta vs frozen baseline: `{float(target_real.report.get('overall_accuracy') or 0.0) - float(baseline_target.report.get('overall_accuracy') or 0.0):+.2f}`",
            f"- Normalized MAE delta vs frozen baseline: `{float(target_real.report.get('normalized_mean_absolute_error') or 0.0) - float(baseline_target.report.get('normalized_mean_absolute_error') or 0.0):+.3f}`",
            "",
        ]
    )
    lines.extend(_markdown_changes(target_changes, limit=10))
    lines.extend(
        [
            "",
            "## Direct answers",
            "",
            f"- Could a manual shared patch be created? `Yes.`",
            f"- Did a manual shared patch improve paper_review? `Yes, {winner['variant']['id']}.`",
            f"- Did it help imo_grading? `{'Yes' if float(target_real.report.get('overall_accuracy') or 0.0) > float(baseline_target.report.get('overall_accuracy') or 0.0) or float(target_real.report.get('normalized_mean_absolute_error') or 0.0) < float(baseline_target.report.get('normalized_mean_absolute_error') or 0.0) else 'No.'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_benchmark(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    base_root = Path(config["base_root"])
    allowlist_path = Path(config["allowlist_path"])

    atlas = _read_json(Path(config["failure_atlas_json"]))

    baseline_screen = _load_existing_eval(
        Path(config["baseline"]["source_screen_report_json"]),
        Path(config["baseline"]["source_screen_predictions_csv"]),
        run_id="shared_path_source_baseline_val10",
    )
    baseline_confirm = _load_existing_eval(
        Path(config["baseline"]["source_confirm_report_json"]),
        Path(config["baseline"]["source_confirm_predictions_csv"]),
        run_id="shared_path_source_baseline_val25",
    )
    baseline_target = _load_existing_eval(
        Path(config["baseline"]["target_confirm_report_json"]),
        Path(config["baseline"]["target_confirm_predictions_csv"]),
        run_id="first_transfer_baseline_val25",
    )

    variant_results: list[dict[str, Any]] = []
    env = dict()
    import os

    env.update(os.environ)

    for variant in config.get("variants", []):
        variant_id = variant["id"]
        variant_root = output_root / "candidates" / variant_id
        candidate_root = variant_root / "repo"
        _copy_snapshot(base_root, candidate_root)
        receipts = _apply_variant(candidate_root, variant_id)

        diff_text = _render_targeted_diff(base_root, candidate_root, [item["path"] for item in receipts])
        diff_path = variant_root / "candidate.diff"
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_path.write_text(diff_text, encoding="utf-8")

        validation = validate_candidate_snapshot(
            base_root=base_root,
            candidate_root=candidate_root,
            allowlist_path=allowlist_path,
        )
        validation_path = variant_root / "validation.json"
        _write_json(validation_path, validation)

        result: dict[str, Any] = {
            "variant": variant,
            "receipts": receipts,
            "changed_files": [item["path"] for item in receipts],
            "diff_path": str(diff_path),
            "validation": validation,
            "screen_eval": None,
            "screen_changes": {
                "changed_prediction_count": 0,
                "changed_examples": [],
                "better_count": 0,
                "worse_count": 0,
                "same_score_count": 0,
            },
            "confirm_eval": None,
            "confirm_changes": {
                "changed_prediction_count": 0,
                "changed_examples": [],
                "better_count": 0,
                "worse_count": 0,
                "same_score_count": 0,
            },
        }

        if validation["verdict"] != "eligible":
            variant_results.append(result)
            continue

        screen_eval = _run_eval(
            repo_root=candidate_root,
            domain=config["source_domain"],
            run_id=f"{variant_id}_source_val10",
            output_dir=variant_root / "source_screen",
            subset=config["source_subset"],
            num_samples=int(config["source_screen_samples"]),
            model=config["eval_model"],
            num_workers=int(config["num_workers"]),
            save_interval=int(config["save_interval"]),
            env=env,
        )
        screen_changes = _prediction_change_summary(
            domain=config["source_domain"],
            subset=config["source_subset"],
            num_samples=int(config["source_screen_samples"]),
            baseline_predictions_path=baseline_screen.predictions_path,
            candidate_predictions_path=screen_eval.predictions_path,
        )
        result["screen_eval"] = screen_eval
        result["screen_changes"] = screen_changes

        if not _passes_source_screen(
            baseline=baseline_screen,
            candidate=screen_eval,
            tolerance=float(config["selection"]["source_valid_label_drop_tolerance"]),
        ):
            variant_results.append(result)
            continue

        confirm_eval = _run_eval(
            repo_root=candidate_root,
            domain=config["source_domain"],
            run_id=f"{variant_id}_source_val25",
            output_dir=variant_root / "source_confirm",
            subset=config["source_subset"],
            num_samples=int(config["source_confirm_samples"]),
            model=config["eval_model"],
            num_workers=int(config["num_workers"]),
            save_interval=int(config["save_interval"]),
            env=env,
        )
        confirm_changes = _prediction_change_summary(
            domain=config["source_domain"],
            subset=config["source_subset"],
            num_samples=int(config["source_confirm_samples"]),
            baseline_predictions_path=baseline_confirm.predictions_path,
            candidate_predictions_path=confirm_eval.predictions_path,
        )
        result["confirm_eval"] = confirm_eval
        result["confirm_changes"] = confirm_changes
        variant_results.append(result)

    eligible_winners = [
        item
        for item in variant_results
        if item["confirm_eval"] is not None
        and _is_source_winner(
            baseline=baseline_confirm,
            candidate=item["confirm_eval"],
            tolerance=float(config["selection"]["source_valid_label_drop_tolerance"]),
        )
    ]
    eligible_winners.sort(key=_variant_sort_key, reverse=True)
    winner = eligible_winners[0] if eligible_winners else None

    benchmark_report_path = Path(config["benchmark_report_path"])
    benchmark_report_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_report_path.write_text(
        _render_benchmark_report(
            config=config,
            atlas=atlas,
            baseline_screen=baseline_screen,
            baseline_confirm=baseline_confirm,
            variant_results=variant_results,
            winner=winner,
        ),
        encoding="utf-8",
    )

    target_smoke = None
    target_real = None
    target_changes = None
    if winner is not None:
        winner_root = output_root / "candidates" / winner["variant"]["id"] / "repo"
        target_smoke = _run_eval(
            repo_root=winner_root,
            domain=config["target_domain"],
            run_id=f"{winner['variant']['id']}_target_val10",
            output_dir=output_root / "transfer" / "target_smoke",
            subset=config["target_subset"],
            num_samples=int(config["target_smoke_samples"]),
            model=config["eval_model"],
            num_workers=int(config["num_workers"]),
            save_interval=int(config["save_interval"]),
            env=env,
        )
        if float(target_smoke.report.get("valid_label_rate") or 0.0) >= 0.95:
            target_real = _run_eval(
                repo_root=winner_root,
                domain=config["target_domain"],
                run_id=f"{winner['variant']['id']}_target_val25",
                output_dir=output_root / "transfer" / "target_confirm",
                subset=config["target_subset"],
                num_samples=int(config["target_confirm_samples"]),
                model=config["eval_model"],
                num_workers=int(config["num_workers"]),
                save_interval=int(config["save_interval"]),
                env=env,
            )
            target_changes = _prediction_change_summary(
                domain=config["target_domain"],
                subset=config["target_subset"],
                num_samples=int(config["target_confirm_samples"]),
                baseline_predictions_path=baseline_target.predictions_path,
                candidate_predictions_path=target_real.predictions_path,
            )

    transfer_report_path = Path(config["transfer_report_path"])
    transfer_report_path.parent.mkdir(parents=True, exist_ok=True)
    transfer_report_path.write_text(
        _render_transfer_report(
            config=config,
            winner=winner,
            baseline_target=baseline_target,
            target_smoke=target_smoke,
            target_real=target_real,
            target_changes=target_changes,
        ),
        encoding="utf-8",
    )

    summary = {
        "winner_variant_id": winner["variant"]["id"] if winner is not None else None,
        "variant_results": [
            {
                "variant_id": item["variant"]["id"],
                "validation_verdict": item["validation"]["verdict"],
                "screen_eval": _serialize_eval_result(item["screen_eval"]),
                "confirm_eval": _serialize_eval_result(item["confirm_eval"]),
                "screen_changes": item["screen_changes"],
                "confirm_changes": item["confirm_changes"],
            }
            for item in variant_results
        ],
    }
    _write_json(output_root / "benchmark_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the shared patch sprint config YAML.",
    )
    args = parser.parse_args()
    run_benchmark(args.config)


if __name__ == "__main__":
    main()
