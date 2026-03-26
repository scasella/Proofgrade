from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analysis.direct_imo_utils import ensure_parent_dir, load_yaml


DEFAULT_CONFIG = Path("configs/baseline_freeze/final_imo_release.yaml")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _fmt(value: float) -> str:
    return f"{float(value):.3f}"


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, divider, *body])


def _main_result_rows(
    validation_summary: dict[str, Any],
    lockbox_summary: dict[str, Any],
    fresh_summary: dict[str, Any] | None,
) -> list[dict[str, str]]:
    rows = []
    validation_variants = validation_summary["variants"]
    rows.extend(
        [
            {
                "split": "Held-out validation (100)",
                "variant": "baseline",
                "accuracy": _fmt(validation_variants["baseline"]["pooled"]["overall_accuracy"]),
                "normalized_grading_error": _fmt(validation_variants["baseline"]["pooled"]["normalized_mean_absolute_error"]),
                "valid_label_rate": _fmt(validation_variants["baseline"]["pooled"]["valid_label_rate"]),
            },
            {
                "split": "Held-out validation (100)",
                "variant": "guideline_gate_almost_boundary_v1",
                "accuracy": _fmt(validation_variants["guideline_gate_almost_boundary_v1"]["pooled"]["overall_accuracy"]),
                "normalized_grading_error": _fmt(validation_variants["guideline_gate_almost_boundary_v1"]["pooled"]["normalized_mean_absolute_error"]),
                "valid_label_rate": _fmt(validation_variants["guideline_gate_almost_boundary_v1"]["pooled"]["valid_label_rate"]),
            },
        ]
    )

    lockbox_variants = lockbox_summary["variants"]
    rows.extend(
        [
            {
                "split": "Untouched test (100)",
                "variant": "baseline",
                "accuracy": _fmt(lockbox_variants["baseline"]["pooled"]["overall_accuracy"]),
                "normalized_grading_error": _fmt(lockbox_variants["baseline"]["pooled"]["normalized_mean_absolute_error"]),
                "valid_label_rate": _fmt(lockbox_variants["baseline"]["pooled"]["valid_label_rate"]),
            },
            {
                "split": "Untouched test (100)",
                "variant": "guideline_gate_almost_boundary_v1",
                "accuracy": _fmt(lockbox_variants["guideline_gate_almost_boundary_v1"]["pooled"]["overall_accuracy"]),
                "normalized_grading_error": _fmt(lockbox_variants["guideline_gate_almost_boundary_v1"]["pooled"]["normalized_mean_absolute_error"]),
                "valid_label_rate": _fmt(lockbox_variants["guideline_gate_almost_boundary_v1"]["pooled"]["valid_label_rate"]),
            },
        ]
    )

    if fresh_summary is not None:
        fresh_variants = fresh_summary["variants"]
        rows.extend(
            [
                {
                    "split": f"Fresh filtered remainder ({fresh_summary['fresh_set_metadata']['remaining_total']})",
                    "variant": "baseline",
                    "accuracy": _fmt(fresh_variants["baseline"]["pooled"]["overall_accuracy"]),
                    "normalized_grading_error": _fmt(fresh_variants["baseline"]["pooled"]["normalized_mean_absolute_error"]),
                    "valid_label_rate": _fmt(fresh_variants["baseline"]["pooled"]["valid_label_rate"]),
                },
                {
                    "split": f"Fresh filtered remainder ({fresh_summary['fresh_set_metadata']['remaining_total']})",
                    "variant": "guideline_gate_almost_boundary_v1",
                    "accuracy": _fmt(fresh_variants["guideline_gate_almost_boundary_v1"]["pooled"]["overall_accuracy"]),
                    "normalized_grading_error": _fmt(fresh_variants["guideline_gate_almost_boundary_v1"]["pooled"]["normalized_mean_absolute_error"]),
                    "valid_label_rate": _fmt(fresh_variants["guideline_gate_almost_boundary_v1"]["pooled"]["valid_label_rate"]),
                },
            ]
        )

    return rows


def _mechanism_rows(validation_summary: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    display = {
        "baseline_to_guideline_gate_v1": "baseline -> guideline_gate_v1",
        "guideline_gate_v1_to_guideline_gate_almost_boundary_v1": "guideline_gate_v1 -> guideline_gate_almost_boundary_v1",
        "guideline_gate_almost_boundary_v1_to_guideline_gate_no_top_end_guard_v1": "guideline_gate_almost_boundary_v1 -> guideline_gate_no_top_end_guard_v1",
    }
    for key, item in validation_summary["pairwise_comparisons"].items():
        rows.append(
            {
                "comparison": display.get(key, key),
                "accuracy_delta": _fmt(item["accuracy_delta"]),
                "normalized_grading_error_delta": _fmt(item["mae_delta"]),
                "changed_predictions": str(item["changed"]["changed_count"]),
                "better": str(item["changed"]["better"]),
                "worse": str(item["changed"]["worse"]),
                "corrected_overcredit": str(item["transitions"]["corrected_overcredit_count"]),
            }
        )
    return rows


def _error_rows(remaining_summary: dict[str, Any]) -> list[dict[str, str]]:
    total = int(remaining_summary["total_remaining_errors"] or 0)
    rows = []
    for bucket, count in sorted(
        remaining_summary["bucket_counts"].items(),
        key=lambda item: (-int(item[1]), item[0]),
    ):
        share = (int(count) / total) if total else 0.0
        rows.append(
            {
                "bucket": bucket,
                "count": str(count),
                "share": _fmt(share),
            }
        )
    return rows


def build_tables(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = config["artifacts"]
    validation_summary = _load_json(Path(artifacts["validation_ablation_summary"]))
    remaining_summary = _load_json(Path(artifacts["remaining_error_summary"]))
    lockbox_summary = _load_json(Path(artifacts["lockbox_summary"]))
    fresh_path = Path(artifacts["fresh_generalization_summary"])
    fresh_summary = _load_json(fresh_path) if fresh_path.exists() else None

    tables = {
        "main_result_rows": _main_result_rows(validation_summary, lockbox_summary, fresh_summary),
        "mechanism_rows": _mechanism_rows(validation_summary),
        "remaining_error_rows": _error_rows(remaining_summary),
        "fresh_available": fresh_summary is not None,
    }

    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    Path(config["tables_json_path"]).write_text(json.dumps(tables, indent=2))

    main_table = _render_table(
        ["Split", "Variant", "Accuracy", "Normalized grading error", "Valid-label rate"],
        [
            [
                row["split"],
                row["variant"],
                row["accuracy"],
                row["normalized_grading_error"],
                row["valid_label_rate"],
            ]
            for row in tables["main_result_rows"]
        ],
    )
    mechanism_table = _render_table(
        ["Comparison", "Acc delta", "Error delta", "Changed", "Better", "Worse", "Corrected overcredit"],
        [
            [
                row["comparison"],
                row["accuracy_delta"],
                row["normalized_grading_error_delta"],
                row["changed_predictions"],
                row["better"],
                row["worse"],
                row["corrected_overcredit"],
            ]
            for row in tables["mechanism_rows"]
        ],
    )
    error_table = _render_table(
        ["Remaining-error bucket", "Count", "Share of remaining errors"],
        [
            [row["bucket"], row["count"], row["share"]]
            for row in tables["remaining_error_rows"]
        ],
    )

    Path(config["main_table_md_path"]).write_text(main_table + "\n")
    Path(config["mechanism_table_md_path"]).write_text(mechanism_table + "\n")
    Path(config["error_bucket_md_path"]).write_text(error_table + "\n")

    fresh_line = ""
    if fresh_summary is not None:
        fresh_line = (
            f"- Fresh filtered remainder: baseline `{_fmt(fresh_summary['variants']['baseline']['pooled']['overall_accuracy'])}` "
            f"-> winner `{_fmt(fresh_summary['variants']['guideline_gate_almost_boundary_v1']['pooled']['overall_accuracy'])}` accuracy; "
            f"error `{_fmt(fresh_summary['variants']['baseline']['pooled']['normalized_mean_absolute_error'])}` "
            f"-> `{_fmt(fresh_summary['variants']['guideline_gate_almost_boundary_v1']['pooled']['normalized_mean_absolute_error'])}`."
        )
    else:
        fresh_line = "- Fresh generalization result: pending."

    report_lines = [
        "# Final IMO Result Package",
        "",
        "## Locked release",
        "",
        f"- Model/provider: `{config['model']}`",
        f"- Parser version: `{config['parser_version']}`",
        f"- Baseline env switch: `{config['variant_env']}=baseline`",
        f"- Final winner env switch: `{config['variant_env']}=guideline_gate_almost_boundary_v1`",
        f"- Validation ablation summary: `{artifacts['validation_ablation_summary']}`",
        f"- Remaining-error summary: `{artifacts['remaining_error_summary']}`",
        f"- Lockbox test summary: `{artifacts['lockbox_summary']}`",
        "",
        "## Reproduction commands",
        "",
    ]
    report_lines.extend(f"- `{command}`" for command in config["commands"])
    report_lines.extend(
        [
            "",
            "## Main result table",
            "",
            main_table,
            "",
            "## Mechanism and ablation table",
            "",
            mechanism_table,
            "",
            "## Remaining-error summary",
            "",
            error_table,
            "",
            "## Fresh generalization status",
            "",
            fresh_line,
            "",
            "## Saved artifacts",
            "",
            f"- Markdown report: `{config['report_path']}`",
            f"- Result tables JSON: `{config['tables_json_path']}`",
            f"- Main table snippet: `{config['main_table_md_path']}`",
            f"- Mechanism table snippet: `{config['mechanism_table_md_path']}`",
            f"- Error-bucket table snippet: `{config['error_bucket_md_path']}`",
            f"- Casebook JSON: `{config['casebook_json_path']}`",
            f"- Casebook markdown: `{config['casebook_md_path']}`",
            "",
        ]
    )

    report_path = Path(config["report_path"])
    ensure_parent_dir(report_path)
    report_path.write_text("\n".join(report_lines) + "\n")
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Build concise tables for the locked IMO result package.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to the final IMO release config.")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    build_tables(config)


if __name__ == "__main__":
    main()
