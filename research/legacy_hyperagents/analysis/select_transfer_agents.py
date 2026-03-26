"""Rank transferable source agents using multiple selector rules."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.lineage import collect_run_records, discover_run_dirs
from utils.patch_taxonomy import (
    TRANSFERABLE_META_CATEGORIES,
    classify_patch_file,
)


SELECTORS = [
    "best_score",
    "descendant_growth",
    "random_valid",
    "meta_patch_density",
    "hybrid_growth_meta",
    "diversity_aware",
]


def _record_meta_density(record: dict[str, Any]) -> float:
    total_lines = 0
    meta_lines = 0
    for patch_file in record["curr_patch_files"]:
        path = Path(patch_file)
        if not path.exists():
            continue
        labels = classify_patch_file(path)
        line_totals = labels["category_line_totals"]
        file_total = sum(line_totals.values())
        if file_total <= 0:
            continue
        total_lines += file_total
        meta_lines += sum(
            value
            for category, value in line_totals.items()
            if category in TRANSFERABLE_META_CATEGORIES
        )
    if total_lines <= 0:
        return 0.0
    return meta_lines / total_lines


def _normalize(series: pd.Series) -> pd.Series:
    minimum = series.min()
    maximum = series.max()
    if pd.isna(minimum) or pd.isna(maximum) or minimum == maximum:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - minimum) / (maximum - minimum)


def load_candidate_dataframe(run_dirs: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_data = collect_run_records(run_dir)
        for record in run_data["records"]:
            if str(record["generation_id"]) == "initial":
                continue
            if not record["valid_parent"]:
                continue
            if record["selection_score"] is None:
                continue
            rows.append(
                {
                    "run_id": record["run_id"],
                    "run_dir": record["run_dir"],
                    "generation_id": record["generation_id"],
                    "source_domains": record["source_domains"],
                    "selection_score": record["selection_score"],
                    "descendant_growth_score": record["descendant_growth_score"],
                    "descendant_count": record["descendant_count"],
                    "depth_in_archive": record["depth"],
                    "touched_files": record["touched_files"],
                    "meta_patch_density": _record_meta_density(record),
                }
            )
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return dataframe
    dataframe["descendant_growth_score"] = dataframe["descendant_growth_score"].fillna(float("-inf"))
    dataframe["unique_file_count"] = dataframe["touched_files"].apply(lambda value: len(set(value)))
    dataframe["best_score_norm"] = _normalize(dataframe["selection_score"])
    valid_growth = dataframe["descendant_growth_score"].replace(float("-inf"), float("nan")).astype(float)
    growth_floor = valid_growth.min(skipna=True) if valid_growth.notna().any() else 0.0
    dataframe["growth_norm"] = _normalize(valid_growth.fillna(growth_floor).astype(float))
    dataframe["meta_density_norm"] = _normalize(dataframe["meta_patch_density"])
    dataframe["file_diversity_norm"] = _normalize(dataframe["unique_file_count"])
    return dataframe


def rank_candidates(dataframe: pd.DataFrame, selector: str, seed: int = 42) -> pd.DataFrame:
    ranked = dataframe.copy()
    if ranked.empty:
        return ranked

    if selector == "best_score":
        ranked["selector_score"] = ranked["selection_score"]
    elif selector == "descendant_growth":
        ranked["selector_score"] = ranked["descendant_growth_score"]
    elif selector == "random_valid":
        rng = random.Random(seed)
        ranked["selector_score"] = [rng.random() for _ in range(len(ranked))]
    elif selector == "meta_patch_density":
        ranked["selector_score"] = ranked["meta_patch_density"]
    elif selector == "hybrid_growth_meta":
        ranked["selector_score"] = (0.7 * ranked["growth_norm"]) + (0.3 * ranked["meta_density_norm"])
    elif selector == "diversity_aware":
        ranked["selector_score"] = (
            0.5 * ranked["best_score_norm"]
            + 0.25 * ranked["meta_density_norm"]
            + 0.25 * ranked["file_diversity_norm"]
        )
    else:
        raise ValueError(f"Unknown selector: {selector}")

    return ranked.sort_values(
        by=["selector_score", "selection_score", "descendant_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select transfer agents using multiple ranking rules.")
    parser.add_argument("--run_dirs", nargs="*", default=[], help="Source run directories")
    parser.add_argument(
        "--auto_discover",
        nargs="*",
        default=[],
        help="Directories under which generate_* runs should be discovered",
    )
    parser.add_argument(
        "--selector",
        choices=SELECTORS,
        default="descendant_growth",
        help="Selector to apply",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of ranked candidates to keep")
    parser.add_argument("--seed", type=int, default=42, help="Seed for stochastic selectors")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis/outputs/meta_transfer",
        help="Directory for selector outputs",
    )
    args = parser.parse_args()

    run_dirs = [str(Path(path).resolve()) for path in args.run_dirs]
    for base_dir in args.auto_discover:
        run_dirs.extend(discover_run_dirs(base_dir))
    run_dirs = sorted(dict.fromkeys(run_dirs))

    candidates = load_candidate_dataframe(run_dirs)
    ranked = rank_candidates(candidates, args.selector, seed=args.seed)
    top_k = ranked.head(args.top_k)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"transfer_selector_{args.selector}.csv"
    json_path = output_dir / f"transfer_selector_{args.selector}.json"
    top_k.to_csv(csv_path, index=False)
    json_path.write_text(top_k.to_json(orient="records", indent=2), encoding="utf-8")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
