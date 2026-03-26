"""Prepare and summarize causal transfer ablations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.select_transfer_agents import SELECTORS, load_candidate_dataframe, rank_candidates
from analysis.smoke_data import create_meta_transfer_fixture
from utils.lineage import collect_run_records, discover_run_dirs
from utils.patch_filtering import filter_patch_by_categories
from utils.patch_taxonomy import (
    MEMORY_CATEGORIES,
    SEARCH_CATEGORIES,
    TASK_CATEGORIES,
    TRANSFERABLE_META_CATEGORIES,
    classify_patch_file,
)


TRANSFER_MODES = [
    "initial_baseline",
    "full_transfer",
    "meta_only_transfer",
    "task_only_transfer",
    "search_only_transfer",
    "memory_only_transfer",
    "random_source_transfer",
]

MODE_TO_CATEGORIES = {
    "meta_only_transfer": set(TRANSFERABLE_META_CATEGORIES),
    "task_only_transfer": set(TASK_CATEGORIES),
    "search_only_transfer": set(SEARCH_CATEGORIES),
    "memory_only_transfer": set(MEMORY_CATEGORIES),
}


def _resolve_source_record(run_dir: str, generation_id: Any) -> dict[str, Any]:
    run_data = collect_run_records(run_dir)
    for record in run_data["records"]:
        if str(record["generation_id"]) == str(generation_id):
            return record
    raise ValueError(f"Generation {generation_id} not found in {run_dir}")


def _filter_patch_for_mode(patch_path: str, mode: str) -> str:
    patch_path = Path(patch_path)
    patch_text = patch_path.read_text(encoding="utf-8")
    if mode == "full_transfer":
        return patch_text
    if mode == "initial_baseline":
        return ""

    classification = classify_patch_file(patch_path)
    file_categories = {
        path: set(categories)
        for path, categories in classification["file_categories"].items()
    }
    include_categories = MODE_TO_CATEGORIES.get(mode)
    if not include_categories:
        return patch_text
    return filter_patch_by_categories(
        patch_text,
        file_categories=file_categories,
        include_categories=include_categories,
    )


def _prepared_patch_paths(record: dict[str, Any], mode: str, output_dir: Path) -> list[str]:
    patch_paths = []
    lineage_patch_files = list(record["prev_patch_files"]) + list(record["curr_patch_files"])
    if mode == "initial_baseline":
        return patch_paths
    for index, patch_file in enumerate(lineage_patch_files):
        source_path = Path(patch_file)
        if not source_path.exists():
            continue
        filtered_patch = _filter_patch_for_mode(str(source_path), mode)
        if not filtered_patch.strip():
            continue
        destination = output_dir / f"{mode}_patch_{index}.diff"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(filtered_patch, encoding="utf-8")
        patch_paths.append(str(destination.resolve()))
    return patch_paths


def build_manifest(
    source_run_dirs: list[str],
    selector: str,
    transfer_modes: list[str],
    output_dir: Path,
    target_domain: str,
    max_generation: int,
    continue_self_improve: bool,
    use_meta_memory: bool,
    meta_model: str | None = None,
    docker_image_name: str | None = None,
    dockerfile_name: str | None = None,
) -> dict[str, Any]:
    candidates = load_candidate_dataframe(source_run_dirs)
    manifest_rows = []

    for mode in transfer_modes:
        effective_selector = "random_valid" if mode == "random_source_transfer" else selector
        ranked = rank_candidates(candidates, effective_selector)
        if ranked.empty and mode != "initial_baseline":
            manifest_rows.append(
                {
                    "mode": mode,
                    "selector": effective_selector,
                    "status": "unavailable",
                    "reason": "no ranked source candidates",
                }
            )
            continue

        if mode == "initial_baseline":
            prepared_patches = []
            source_run_id = "initial"
            generation_id = "initial"
        else:
            prepared_patches = []
            source_run_id = None
            generation_id = None
            for _, candidate in ranked.iterrows():
                source_run_id = candidate["run_id"]
                generation_id = candidate["generation_id"]
                record = _resolve_source_record(candidate["run_dir"], generation_id)
                prepared_patches = _prepared_patch_paths(
                    record,
                    mode="full_transfer" if mode == "random_source_transfer" else mode,
                    output_dir=output_dir / "filtered_patches" / effective_selector / f"{source_run_id}_gen_{generation_id}",
                )
                if prepared_patches:
                    break
            if not prepared_patches:
                manifest_rows.append(
                    {
                        "mode": mode,
                        "selector": effective_selector,
                        "source_run_id": source_run_id,
                        "source_generation_id": generation_id,
                        "status": "unavailable",
                        "reason": "selected source agent has no patches matching this transfer mode",
                    }
                )
                continue

        command = [
            "python",
            "generate_loop.py",
            "--domains",
            target_domain,
            "--max_generation",
            str(max_generation),
        ]
        if prepared_patches:
            command.extend(["--meta_patch_files", *prepared_patches])
        if not continue_self_improve:
            command.extend(["--run_baseline", "no_selfimprove"])
        if use_meta_memory:
            command.extend(
                [
                    "--use_meta_memory",
                    "--meta_memory_format",
                    "both",
                    "--meta_memory_window",
                    "5",
                    "--meta_memory_include_patch_labels",
                ]
            )
        if meta_model:
            command.extend(["--meta_model", meta_model])
        if docker_image_name:
            command.extend(["--docker_image_name", docker_image_name])
        if dockerfile_name:
            command.extend(["--dockerfile_name", dockerfile_name])

        manifest_rows.append(
            {
                "mode": mode,
                "selector": effective_selector,
                "source_run_id": source_run_id,
                "source_generation_id": generation_id,
                "prepared_patch_files": prepared_patches,
                "command": command,
                "status": "ready",
            }
        )

    manifest = {
        "target_domain": target_domain,
        "selector": selector,
        "continue_self_improve": continue_self_improve,
        "use_meta_memory": use_meta_memory,
        "meta_model": meta_model,
        "docker_image_name": docker_image_name,
        "dockerfile_name": dockerfile_name,
        "experiments": manifest_rows,
    }
    return manifest


def _progress_summary(run_dir: str, domain: str) -> dict[str, Any]:
    run_data = collect_run_records(run_dir)
    scores = []
    for record in run_data["records"]:
        domain_scores = record["scores"].get(domain, {})
        value = None
        for split in ("test", "val", "train"):
            if split in domain_scores and domain_scores[split].get("max") is not None:
                value = float(domain_scores[split]["max"])
                break
        scores.append(value)

    initial_score = next((score for score in scores if score is not None), None)
    best_score = max((score for score in scores if score is not None), default=None)
    return {
        "run_id": run_data["run_id"],
        "mode": _mode_from_run_name(Path(run_dir).name),
        "scores": scores,
        "initial_score": initial_score,
        "best_score": best_score,
        "gain": (best_score - initial_score) if best_score is not None and initial_score is not None else None,
    }


def _mode_from_run_name(name: str) -> str:
    for mode in TRANSFER_MODES:
        if mode in name:
            return mode
    if "selector_" in name:
        return name.split("selector_")[-1]
    return "unlabeled"


def _plot_mode_gains(dataframe: pd.DataFrame, output_dir: Path) -> Path | None:
    rows = dataframe[dataframe["mode"].isin(TRANSFER_MODES) & dataframe["gain"].notna()]
    if rows.empty:
        return None
    order = [mode for mode in TRANSFER_MODES if mode in set(rows["mode"])]
    rows = rows.set_index("mode").loc[order].reset_index()
    plt.figure(figsize=(10, 5))
    plt.bar(rows["mode"], rows["gain"], color="#2563EB")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Transfer gain")
    plt.tight_layout()
    path = output_dir / "transfer_ablation_modes.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _plot_selector_comparison(dataframe: pd.DataFrame, output_dir: Path) -> Path | None:
    rows = dataframe[~dataframe["mode"].isin(TRANSFER_MODES) & dataframe["gain"].notna()]
    if rows.empty:
        return None
    rows = rows.sort_values("gain", ascending=False)
    plt.figure(figsize=(9, 5))
    plt.bar(rows["mode"], rows["gain"], color="#9333EA")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Transfer gain")
    plt.tight_layout()
    path = output_dir / "selector_comparison.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _plot_patch_attribution(dataframe: pd.DataFrame, output_dir: Path) -> Path | None:
    rows = dataframe[dataframe["mode"].isin(TRANSFER_MODES) & dataframe["gain"].notna()]
    if rows.empty:
        return None
    label_map = {
        "meta_only_transfer": "meta-level",
        "task_only_transfer": "task-level",
        "search_only_transfer": "search-level",
        "memory_only_transfer": "memory-level",
        "full_transfer": "full",
        "initial_baseline": "baseline",
        "random_source_transfer": "random source",
    }
    rows = rows.copy()
    rows["label"] = rows["mode"].map(label_map).fillna(rows["mode"])
    plt.figure(figsize=(9, 5))
    plt.bar(rows["label"], rows["gain"], color="#0F766E")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Gain vs initial")
    plt.tight_layout()
    path = output_dir / "patch_category_attribution.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _plot_summary_figure(dataframe: pd.DataFrame, output_dir: Path) -> Path | None:
    rows = dataframe[dataframe["mode"].isin(["full_transfer", "meta_only_transfer", "task_only_transfer", "random_source_transfer"]) & dataframe["gain"].notna()]
    if rows.empty:
        return None
    rows = rows.set_index("mode")
    plt.figure(figsize=(8, 4))
    plt.axvline(0.0, color="black", linewidth=1)
    ordered_labels = [label for label in ["full_transfer", "meta_only_transfer", "task_only_transfer", "random_source_transfer"] if label in rows.index]
    values = [rows.loc[label, "gain"] for label in ordered_labels]
    plt.barh(ordered_labels, values, color=["#2563EB", "#0F766E", "#D97706", "#7C3AED"][: len(values)])
    plt.xlabel("Target-domain gain")
    plt.tight_layout()
    path = output_dir / "what_actually_transfers_summary.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def summarize_runs(run_dirs: list[str], domain: str, output_dir: Path) -> pd.DataFrame:
    rows = [_progress_summary(run_dir, domain) for run_dir in run_dirs]
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return dataframe
    dataframe.to_csv(output_dir / "transfer_ablation_summary.csv", index=False)
    _plot_mode_gains(dataframe, output_dir)
    _plot_selector_comparison(dataframe, output_dir)
    _plot_patch_attribution(dataframe, output_dir)
    _plot_summary_figure(dataframe, output_dir)
    return dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and summarize causal transfer ablations.")
    parser.add_argument("--source_run_dirs", nargs="*", default=[], help="Source run directories")
    parser.add_argument("--summary_run_dirs", nargs="*", default=[], help="Completed target run directories to summarize")
    parser.add_argument("--auto_discover_sources", nargs="*", default=[], help="Directories to scan for source runs")
    parser.add_argument("--auto_discover_summary", nargs="*", default=[], help="Directories to scan for completed target runs")
    parser.add_argument("--selector", choices=SELECTORS, default="descendant_growth")
    parser.add_argument("--transfer_modes", nargs="*", default=TRANSFER_MODES, choices=TRANSFER_MODES)
    parser.add_argument("--target_domain", default="imo_grading")
    parser.add_argument("--max_generation", type=int, default=50)
    parser.add_argument("--continue_self_improve", action="store_true", default=False)
    parser.add_argument("--use_meta_memory", action="store_true", default=False)
    parser.add_argument("--meta_model", type=str, default=None, help="Optional override for the meta-agent model passed to generate_loop.")
    parser.add_argument("--docker_image_name", type=str, default=None, help="Optional Docker image override passed to generate_loop.")
    parser.add_argument("--dockerfile_name", type=str, default=None, help="Optional Dockerfile name passed to generate_loop if the image must be built.")
    parser.add_argument("--execute", action="store_true", default=False, help="Execute prepared generate_loop commands.")
    parser.add_argument("--create_smoke_fixture", type=str, default=None, help="Optional fixture root to create and use")
    parser.add_argument("--output_dir", type=str, default="analysis/outputs/meta_transfer")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.create_smoke_fixture:
        manifest = create_meta_transfer_fixture(args.create_smoke_fixture)
        args.source_run_dirs.extend(manifest["source_runs"].values())
        args.summary_run_dirs.extend(manifest["target_runs"].values())
        args.summary_run_dirs.extend(manifest["selector_runs"].values())

    source_run_dirs = [str(Path(path).resolve()) for path in args.source_run_dirs]
    for base_dir in args.auto_discover_sources:
        source_run_dirs.extend(discover_run_dirs(base_dir))
    source_run_dirs = sorted(dict.fromkeys(source_run_dirs))

    summary_run_dirs = [str(Path(path).resolve()) for path in args.summary_run_dirs]
    for base_dir in args.auto_discover_summary:
        summary_run_dirs.extend(discover_run_dirs(base_dir))
    summary_run_dirs = sorted(dict.fromkeys(summary_run_dirs))

    manifest = build_manifest(
        source_run_dirs=source_run_dirs,
        selector=args.selector,
        transfer_modes=args.transfer_modes,
        output_dir=output_dir,
        target_domain=args.target_domain,
        max_generation=args.max_generation,
        continue_self_improve=args.continue_self_improve,
        use_meta_memory=args.use_meta_memory,
        meta_model=args.meta_model,
        docker_image_name=args.docker_image_name,
        dockerfile_name=args.dockerfile_name,
    )
    manifest_path = output_dir / "transfer_ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)

    if args.execute:
        repo_root = Path(__file__).resolve().parent.parent
        execution_rows = []
        for experiment in manifest["experiments"]:
            if experiment["status"] != "ready":
                continue
            result = subprocess.run(
                experiment["command"],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            execution_rows.append(
                {
                    "mode": experiment["mode"],
                    "selector": experiment["selector"],
                    "returncode": result.returncode,
                    "stdout_tail": result.stdout[-500:],
                    "stderr_tail": result.stderr[-500:],
                }
            )
        execution_path = output_dir / "transfer_ablation_execution.json"
        execution_path.write_text(json.dumps(execution_rows, indent=2) + "\n", encoding="utf-8")
        print(execution_path)

    if summary_run_dirs:
        summary = summarize_runs(summary_run_dirs, args.target_domain, output_dir)
        summary_path = output_dir / "transfer_ablation_summary.csv"
        print(summary_path)
        if summary.empty:
            print("No completed target runs were available for summary.")


if __name__ == "__main__":
    main()
