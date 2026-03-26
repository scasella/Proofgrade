"""Build a paper_review failure atlas constrained to the traced shared surface."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.common import extract_jsons
from utils.prediction_contracts import JSON_LABEL_KEYS


DEFAULT_CONFIG_PATH = Path(
    "/Users/scasella/Downloads/hyperagents/HyperAgents/configs/baseline_freeze/shared_patch_variants.yaml"
)

PARTIAL_JSON_LABEL_PATTERN = re.compile(
    r'"(?P<key>label|decision|prediction|response)"\s*:\s*"(?P<value>[^"\r\n]{1,32})"',
    re.IGNORECASE,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_path(domain: str, subset: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    if domain.startswith("imo_"):
        return root / f"domains/imo/{domain.split('_')[-1]}bench{subset}.csv"
    return root / f"domains/{domain}/dataset{subset}.csv"


def _extract_last_output_text(markdown_text: str) -> str | None:
    matches = list(re.finditer(r"^Output:\s*(.+)$", markdown_text, re.MULTILINE))
    if not matches:
        return None
    raw_value = matches[-1].group(1).strip()
    try:
        parsed = ast.literal_eval(raw_value)
    except Exception:
        return raw_value
    return parsed if isinstance(parsed, str) else str(parsed)


def _has_complete_json_label(raw_text: str) -> bool:
    extracted = extract_jsons(raw_text) or []
    for obj in extracted:
        if not isinstance(obj, dict):
            continue
        if any(key in obj for key in JSON_LABEL_KEYS):
            return True
    return False


def _classify_invalid_output(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text or ""
    partial_match = PARTIAL_JSON_LABEL_PATTERN.search(raw_text)
    brace_balance = raw_text.count("{") - raw_text.count("}")
    has_complete_json = _has_complete_json_label(raw_text)
    symptoms: list[str] = []
    fixability = "not_shared_fixable"

    if partial_match and not has_complete_json:
        symptoms.append("truncated_json_with_visible_label")
        fixability = "shared_fixable"
    elif len(raw_text) > 300 and not has_complete_json:
        symptoms.append("overlong_or_truncated_without_full_json")
        fixability = "probably_shared_fixable"
    else:
        symptoms.append("invalid_without_clear_shared_signal")

    if brace_balance > 0:
        symptoms.append("unclosed_json_object")
    if len(raw_text) > 400:
        symptoms.append("overlong_output")
    if partial_match:
        symptoms.append("label_visible_before_failure")

    return {
        "symptoms": symptoms,
        "fixability": fixability,
        "visible_partial_label_key": partial_match.group("key").lower() if partial_match else None,
        "visible_partial_label_value": partial_match.group("value").lower() if partial_match else None,
        "has_complete_json_label": has_complete_json,
        "brace_balance": brace_balance,
        "output_length": len(raw_text),
    }


def _build_allowed_surface_summary(allowlist: dict[str, Any]) -> dict[str, list[str]]:
    editable = [entry["symbol_id"] for entry in allowlist.get("shared_editable_symbols", [])]
    source_only = [entry["symbol_id"] for entry in allowlist.get("forbidden_symbols", {}).get("source_only", [])]
    target_only = [entry["symbol_id"] for entry in allowlist.get("forbidden_symbols", {}).get("target_only", [])]
    mixed = [entry["symbol_id"] for entry in allowlist.get("mixed_symbols", [])]
    return {
        "shared_editable_symbols": editable,
        "forbidden_source_only_symbols": source_only,
        "forbidden_target_only_symbols": target_only,
        "mixed_forbidden_symbols": mixed,
    }


def build_failure_atlas(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    allowlist = _load_yaml(Path(config["allowlist_path"]))

    baseline_report = _read_json(Path(config["baseline"]["source_confirm_report_json"]))
    predictions = pd.read_csv(Path(config["baseline"]["source_confirm_predictions_csv"]), dtype=str)
    dataset = pd.read_csv(_dataset_path(config["source_domain"], config["source_subset"]), dtype=str).iloc[
        : config["source_confirm_samples"]
    ].copy()
    merged = dataset.merge(predictions[["question_id", "prediction"]], on="question_id", how="left")
    merged["prediction"] = merged["prediction"].fillna("").astype(str).str.strip().str.lower()
    merged["outcome"] = merged["outcome"].fillna("").astype(str).str.strip().str.lower()

    invalid_examples: list[dict[str, Any]] = []
    invalid_dir = Path(config["baseline"]["source_confirm_report_json"]).parent / "agent_evals"
    for question_id in baseline_report.get("invalid_question_ids", []):
        chat_path = invalid_dir / f"chat_history_{question_id}.md"
        raw_text = _extract_last_output_text(chat_path.read_text(encoding="utf-8", errors="ignore"))
        example = merged.loc[merged["question_id"] == question_id].iloc[0]
        invalid_examples.append(
            {
                "question_id": question_id,
                "ground_truth": example["outcome"],
                "predicted_label": None,
                "raw_output_excerpt": (raw_text or "")[:500],
                **_classify_invalid_output(raw_text or ""),
            }
        )

    valid_wrong_examples: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        prediction = row["prediction"]
        truth = row["outcome"]
        if not prediction or prediction == truth:
            continue
        valid_wrong_examples.append(
            {
                "question_id": row["question_id"],
                "ground_truth": truth,
                "predicted_label": prediction,
                "symptom": f"valid_{prediction}_overcall" if prediction in {"accept", "reject"} else "valid_wrong_label",
                "fixability": "not_shared_fixable",
            }
        )

    invalid_symptoms = Counter(
        symptom
        for item in invalid_examples
        for symptom in item["symptoms"]
    )
    visible_partial_labels = Counter(
        item["visible_partial_label_value"]
        for item in invalid_examples
        if item["visible_partial_label_value"]
    )
    wrong_valid_symptoms = Counter(item["symptom"] for item in valid_wrong_examples)

    atlas = {
        "config_path": str(config_path),
        "source_domain": config["source_domain"],
        "baseline_metrics": {
            "overall_accuracy": baseline_report.get("overall_accuracy"),
            "valid_label_rate": baseline_report.get("valid_label_rate"),
            "invalid_prediction_count": baseline_report.get("invalid_prediction_count"),
            "prediction_distribution": baseline_report.get("prediction_distribution", {}),
        },
        "allowed_surface": _build_allowed_surface_summary(allowlist),
        "invalid_examples": invalid_examples,
        "valid_wrong_examples": valid_wrong_examples,
        "symptom_counts": {
            "invalid_output_symptoms": dict(invalid_symptoms),
            "visible_partial_labels": dict(visible_partial_labels),
            "valid_wrong_symptoms": dict(wrong_valid_symptoms),
        },
        "shared_fixable_summary": {
            "shared_fixable_invalid_examples": sum(1 for item in invalid_examples if item["fixability"] == "shared_fixable"),
            "probably_shared_fixable_invalid_examples": sum(
                1 for item in invalid_examples if item["fixability"] == "probably_shared_fixable"
            ),
            "not_shared_fixable_valid_wrong_examples": len(valid_wrong_examples),
        },
        "candidate_patch_ids": [variant["id"] for variant in config.get("variants", [])],
    }
    return atlas


def _render_markdown(atlas: dict[str, Any], config: dict[str, Any]) -> str:
    allowed = atlas["allowed_surface"]
    invalid_examples = atlas["invalid_examples"]
    valid_wrong_examples = atlas["valid_wrong_examples"]
    partial_label_examples = [item for item in invalid_examples if item["visible_partial_label_value"]]
    lines = [
        "# Shared Patch Design Sprint",
        "",
        "## Shared surface we are allowed to edit",
        "",
        "The sprint stays inside the traced shared task surface only.",
        "",
        f"- Shared editable symbols: `{len(allowed['shared_editable_symbols'])}`",
        f"- Source-only forbidden symbols: `{len(allowed['forbidden_source_only_symbols'])}`",
        f"- Target-only forbidden symbols: `{len(allowed['forbidden_target_only_symbols'])}`",
        f"- Mixed forbidden symbols: `{len(allowed['mixed_forbidden_symbols'])}`",
        "",
        "The practical shared levers are:",
        "",
        "- shared task-instruction scaffolding",
        "- shared generic JSON extraction and cleanup",
        "- shared input formatting",
        "- shared Gemini request behavior",
        "",
        "Tempting ideas that are explicitly out of bounds:",
        "",
        "- paper_review-only reviewer prompt edits",
        "- paper_review-only parser edits",
        "- imo_grading-specific logic",
        "- mixed schema-branch edits in `agent/llm.py::_infer_gemini_response_schema`",
        "",
        "## Failure atlas from paper_review validation",
        "",
        f"- Frozen source baseline accuracy: `{atlas['baseline_metrics']['overall_accuracy']}`",
        f"- Frozen source baseline valid-label rate: `{atlas['baseline_metrics']['valid_label_rate']}`",
        f"- Invalid outputs on val-25: `{atlas['baseline_metrics']['invalid_prediction_count']}`",
        f"- Wrong-but-valid outputs on val-25: `{len(valid_wrong_examples)}`",
        "",
        "Observed shared-fixable failure pattern:",
        "",
        f"- `{atlas['shared_fixable_summary']['shared_fixable_invalid_examples']}` invalid outputs already contain a visible label before the JSON is cut off.",
        f"- Visible partial labels among those failures: `{atlas['symptom_counts']['visible_partial_labels']}`",
        "",
        "Observed but not safely shared-fixable from this surface:",
        "",
        f"- `{len(valid_wrong_examples)}` valid paper_review mistakes are plain decision errors, mostly accept-overcalls. The underlying reviewer policy lives in paper_review-local prompt logic, so this sprint should not try to fix those with domain-local edits.",
        "",
        "Representative invalid examples:",
        "",
    ]
    for item in partial_label_examples[:4]:
        lines.append(
            f"- `{item['question_id']}`: visible partial label `{item['visible_partial_label_value']}`, truth `{item['ground_truth']}`, symptoms `{', '.join(item['symptoms'])}`"
        )
    lines.extend(
        [
            "",
            "## Manual patch slate",
            "",
        ]
    )
    for variant in config.get("variants", []):
        lines.extend(
            [
                f"### {variant['id']}",
                "",
                f"- Mechanism: `{variant['mechanism']}`",
                f"- Shared symbols: `{', '.join(variant['changed_symbols'])}`",
                f"- Why it might help paper_review: {variant['source_hypothesis']}",
                f"- Why it is transfer-eligible: {variant['transfer_hypothesis']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Design decision",
            "",
            "The sprint should focus first on shared fixes for truncated structured output. Those are the only source failures that are both common and clearly reachable from the shared exercised surface.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the shared patch sprint config YAML.",
    )
    args = parser.parse_args()

    config = _load_yaml(args.config)
    atlas = build_failure_atlas(args.config)

    output_json_path = Path(config["failure_atlas_json"])
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(atlas, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_path = Path(config["design_report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_markdown(atlas, config), encoding="utf-8")


if __name__ == "__main__":
    main()
