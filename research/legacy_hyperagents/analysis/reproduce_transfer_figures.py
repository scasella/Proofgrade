"""Reproduce transfer-oriented figures when archived runs are available."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.lineage import collect_run_records, detect_lfs_pointer, discover_run_dirs


TARGET_DOMAIN = "imo_grading"


def _mode_from_run_name(name: str) -> str:
    for mode in [
        "initial_baseline",
        "full_transfer",
        "meta_only_transfer",
        "task_only_transfer",
        "search_only_transfer",
        "memory_only_transfer",
        "random_source_transfer",
    ]:
        if mode in name:
            return mode
    if "selector_" in name:
        return name.split("selector_")[-1]
    return "unlabeled"


def _extract_progress(run_dir: str) -> dict[str, object]:
    run_data = collect_run_records(run_dir)
    series = []
    for record in run_data["records"]:
        domain_scores = record["scores"].get(TARGET_DOMAIN, {})
        score = None
        for split in ("test", "val", "train"):
            if split in domain_scores and domain_scores[split].get("max") is not None:
                score = float(domain_scores[split]["max"])
                break
        series.append(
            {
                "generation_id": record["generation_id"],
                "score": score,
            }
        )

    valid_scores = [item["score"] for item in series if item["score"] is not None]
    initial_score = series[0]["score"] if series else None
    best_score = max(valid_scores) if valid_scores else None
    return {
        "run_id": run_data["run_id"],
        "run_dir": run_dir,
        "mode": _mode_from_run_name(Path(run_dir).name),
        "series": series,
        "initial_score": initial_score,
        "best_score": best_score,
        "imp_at_50_proxy": (best_score - initial_score) if best_score is not None and initial_score is not None else None,
    }


def _plot_transfer_bars(rows: list[dict[str, object]], output_dir: Path) -> Path | None:
    plot_rows = [row for row in rows if row["mode"] != "unlabeled" and row["imp_at_50_proxy"] is not None]
    if not plot_rows:
        return None
    dataframe = pd.DataFrame(plot_rows).sort_values("imp_at_50_proxy", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(dataframe["mode"], dataframe["imp_at_50_proxy"], color="#0F766E")
    plt.ylabel("imp@50-style gain")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    output_path = output_dir / "transfer_imp_at_k.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def _plot_progress_curves(rows: list[dict[str, object]], output_dir: Path) -> Path | None:
    curve_rows = [row for row in rows if row["mode"] in {"initial_baseline", "full_transfer", "meta_only_transfer", "task_only_transfer"}]
    if not curve_rows:
        return None
    plt.figure(figsize=(10, 5))
    for row in curve_rows:
        xs = list(range(len(row["series"])))
        ys = [item["score"] if item["score"] is not None else 0.0 for item in row["series"]]
        plt.plot(xs, ys, marker="o", label=row["mode"])
    plt.xlabel("Generation")
    plt.ylabel("Best available target score")
    plt.legend()
    plt.tight_layout()
    output_path = output_dir / "transfer_curves.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def build_report(workspace_root: Path, rows: list[dict[str, object]], lfs_pointer_paths: list[str]) -> str:
    real_runs = [row for row in rows if "smoke" not in row["run_id"]]
    available_modes = sorted({row["mode"] for row in rows if row["mode"] != "unlabeled"})
    lines = [
        "# Transfer Reproduction Results",
        "",
        "## Status",
        "",
    ]
    if real_runs:
        lines.append(f"- Detected {len(real_runs)} non-smoke run(s) with `imo_grading`-style target scores.")
    else:
        lines.append("- Exact paper-style reproduction from archived real runs is not possible in this workspace as-is.")

    if lfs_pointer_paths:
        lines.append("- The bundled archived outputs are Git LFS pointers rather than the underlying log archive:")
        for path in lfs_pointer_paths:
            lines.append(f"  - `{path}`")

    imo_csvs = sorted(workspace_root.glob("domains/imo/*.csv"))
    if imo_csvs:
        lines.append(f"- Detected local IMO grading CSVs: {', '.join(path.name for path in imo_csvs)}")
    else:
        lines.append("- Local IMO grading CSVs are absent; `domains/imo/setup.sh` would be required before real target runs.")

    lines.extend(["", "## Closest Match"])
    if available_modes:
        lines.append("")
        lines.append(f"- Available labeled modes: {', '.join(available_modes)}")
        summary = pd.DataFrame(rows)
        if not summary.empty:
            summary = summary[summary["imp_at_50_proxy"].notna()].sort_values("imp_at_50_proxy", ascending=False)
            if not summary.empty:
                lines.append("- Highest observed imp@50-style gain among available runs:")
                top = summary.iloc[0]
                lines.append(
                    f"  - `{top['mode']}` in `{top['run_id']}`: initial={top['initial_score']:.3f}, best={top['best_score']:.3f}, gain={top['imp_at_50_proxy']:.3f}"
                )
    else:
        lines.append("")
        lines.append("- No labeled transfer runs were discovered locally, so the closest match is the audit itself: the public code exposes transfer utilities, but the raw paper runs are not present.")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This script uses available run artifacts only. It does not infer missing results from hard-coded paper figures.",
            "- Smoke fixtures can still be used to validate that the reproduction pipeline, plots, and summaries execute end to end.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce transfer-oriented summaries from available runs.")
    parser.add_argument("--run_dirs", nargs="*", default=[], help="Explicit run directories to analyze")
    parser.add_argument("--auto_discover", nargs="*", default=["outputs"], help="Directories to scan for generate_* runs")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis/outputs/meta_transfer",
        help="Directory for generated plots",
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="reports/reproduce_transfer_results.md",
        help="Markdown report path",
    )
    parser.add_argument(
        "--workspace_root",
        type=str,
        default=".",
        help="Repo root for gap checks",
    )
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    run_dirs = [str(Path(path).resolve()) for path in args.run_dirs]
    for base_dir in args.auto_discover:
        run_dirs.extend(discover_run_dirs(workspace_root / base_dir))
    run_dirs = sorted(dict.fromkeys(run_dirs))

    rows = [_extract_progress(run_dir) for run_dir in run_dirs]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_transfer_bars(rows, output_dir)
    _plot_progress_curves(rows, output_dir)

    lfs_pointer_paths = []
    for relative_path in [
        "outputs_os_parts.zip",
        "outputs_os_parts.z01",
        "outputs_os_parts.z08",
    ]:
        path = workspace_root / relative_path
        if detect_lfs_pointer(path):
            lfs_pointer_paths.append(relative_path)

    report_text = build_report(workspace_root, rows, lfs_pointer_paths)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
