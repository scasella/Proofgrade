"""CLI wrapper around the shared deterministic patch taxonomy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.patch_taxonomy import (
    CATEGORY_INFRA,
    CATEGORY_MEMORY,
    CATEGORY_META,
    CATEGORY_ORDER,
    CATEGORY_SEARCH,
    CATEGORY_TASK,
    CATEGORY_EVAL,
    CATEGORY_ENSEMBLE,
    MEMORY_CATEGORIES,
    SEARCH_CATEGORIES,
    TASK_CATEGORIES,
    TRANSFERABLE_META_CATEGORIES,
    classify_delta,
    classify_patch_file,
    classify_patch_files,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify patch files into deterministic meta-transfer categories.")
    parser.add_argument("patch_files", nargs="+", help="Patch files to classify")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    results = classify_patch_files(args.patch_files)
    output_text = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(output_text + "\n", encoding="utf-8")
        print(args.output)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
