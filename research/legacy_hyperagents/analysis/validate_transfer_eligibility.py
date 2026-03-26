"""Validate whether a candidate patch or snapshot is transfer-eligible."""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.patch_filtering import parse_patch_file


DEFAULT_BASE_ROOT = Path(
    "/Users/scasella/Downloads/hyperagents/HyperAgents/analysis/outputs/first_transfer_pilot/frozen_roots/shared_repaired_baseline"
)
DEFAULT_ALLOWLIST_PATH = Path(
    "/Users/scasella/Downloads/hyperagents/HyperAgents/configs/baseline_freeze/shared_path_allowlist.yaml"
)
DEFAULT_OUTPUT_PATH = Path(
    "/Users/scasella/Downloads/hyperagents/HyperAgents/analysis/outputs/shared_path_transfer/transfer_eligibility_validation.json"
)

SCAN_EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    ".venv312",
    "venv",
    "outputs",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
SCAN_EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".zip",
    ".gz",
    ".bz2",
    ".xz",
    ".tar",
    ".parquet",
    ".feather",
    ".pickle",
    ".pkl",
    ".pt",
    ".bin",
    ".dylib",
    ".so",
    ".a",
    ".class",
    ".jar",
    ".sqlite",
    ".db",
    ".csv",
    ".tsv",
}
SCAN_EXCLUDED_NAMES = {".DS_Store"}
SCAN_ALLOWED_TEXT_SUFFIXES = {
    "",
    ".py",
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
    ".zsh",
    ".bash",
    ".fish",
    ".js",
    ".mjs",
    ".cjs",
}


@dataclass(frozen=True)
class SymbolSpan:
    name: str
    lineno: int
    end_lineno: int


@dataclass(frozen=True)
class ChangeDecision:
    file_path: str
    symbol: str
    before_lines: tuple[int, int]
    after_lines: tuple[int, int]
    added_lines: int
    removed_lines: int
    verdict: str
    reason: str
    matched_symbols: list[str]
    matched_blocks: list[str]


def _iter_source_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(root)
        if any(part in SCAN_EXCLUDED_DIRS for part in rel_path.parts):
            continue
        if path.suffix in SCAN_EXCLUDED_SUFFIXES or path.name in SCAN_EXCLUDED_NAMES:
            continue
        if path.suffix not in SCAN_ALLOWED_TEXT_SUFFIXES:
            continue
        if path.suffix == "" and "." in path.name:
            continue
        files[str(rel_path)] = path
    return files


def _safe_read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _python_symbol_spans(text: str) -> list[SymbolSpan]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    spans: list[SymbolSpan] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> Any:
            self.stack.append(node.name)
            spans.append(
                SymbolSpan(
                    name=".".join(self.stack),
                    lineno=node.lineno,
                    end_lineno=getattr(node, "end_lineno", node.lineno),
                )
            )
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            self.stack.append(node.name)
            spans.append(
                SymbolSpan(
                    name=".".join(self.stack),
                    lineno=node.lineno,
                    end_lineno=getattr(node, "end_lineno", node.lineno),
                )
            )
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
            self.visit_FunctionDef(node)  # type: ignore[arg-type]

    Visitor().visit(tree)
    spans.sort(key=lambda item: (item.lineno, item.end_lineno, item.name))
    return spans


def _find_symbol(spans: list[SymbolSpan], start: int, end: int) -> str | None:
    if not spans:
        return None
    end = max(end, start)
    candidates = [
        span
        for span in spans
        if not (end < span.lineno or start > span.end_lineno)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item.end_lineno - item.lineno, item.lineno))
    return candidates[0].name


def _normalize_change_window(i1: int, i2: int, total_lines: int) -> tuple[int, int]:
    if i2 > i1:
        return (i1 + 1, i2)
    anchor = max(1, min(total_lines, i1 + 1 if total_lines else 1))
    return (anchor, anchor)


def _load_allowlist(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _overlaps(window: tuple[int, int], lines: list[int] | tuple[int, int]) -> bool:
    start, end = window
    line_start, line_end = int(lines[0]), int(lines[1])
    return not (end < line_start or start > line_end)


def _prefix_match(path: str, prefixes: list[str]) -> str | None:
    for prefix in prefixes:
        if path == prefix.rstrip("/") or path.startswith(prefix):
            return prefix
    return None


def _classify_changed_window(
    path: str,
    window: tuple[int, int],
    allowlist: dict[str, Any],
) -> tuple[str, str, list[str], list[str]]:
    matched_symbols: list[str] = []
    matched_blocks: list[str] = []

    noise_prefix = _prefix_match(path, allowlist.get("noise_excludes", {}).get("path_prefixes", []))
    if noise_prefix:
        return ("ineligible", f"path_is_noise:{noise_prefix}", matched_symbols, matched_blocks)

    forbidden_prefixes = allowlist.get("forbidden_path_prefixes", {})
    source_prefix = _prefix_match(path, forbidden_prefixes.get("source_only", []))
    if source_prefix:
        return ("ineligible", f"path_is_source_only:{source_prefix}", matched_symbols, matched_blocks)
    target_prefix = _prefix_match(path, forbidden_prefixes.get("target_only", []))
    if target_prefix:
        return ("ineligible", f"path_is_target_only:{target_prefix}", matched_symbols, matched_blocks)
    reporting_prefix = _prefix_match(path, forbidden_prefixes.get("reporting_only", []))
    if reporting_prefix:
        return ("ineligible", f"path_is_reporting_only:{reporting_prefix}", matched_symbols, matched_blocks)

    for bucket_name in ("source_only", "target_only"):
        for symbol_entry in allowlist.get("forbidden_symbols", {}).get(bucket_name, []):
            if symbol_entry["path"] == path and _overlaps(window, symbol_entry["lines"]):
                matched_symbols.append(symbol_entry["symbol_id"])
                return ("ineligible", f"overlaps_forbidden_symbol:{symbol_entry['symbol_id']}", matched_symbols, matched_blocks)

    for mixed_symbol in allowlist.get("mixed_symbols", []):
        if mixed_symbol["path"] != path:
            continue
        mixed_symbol_id = mixed_symbol["symbol_id"]
        touched_any_mixed = False
        for block in mixed_symbol.get("blocks", []):
            if _overlaps(window, block["lines"]):
                touched_any_mixed = True
                matched_blocks.append(f"{mixed_symbol_id}:{block['block_id']}")
                if block["category"] in {"source_only", "target_only"}:
                    return (
                        "ineligible",
                        f"overlaps_mixed_domain_local_block:{mixed_symbol_id}:{block['block_id']}",
                        matched_symbols,
                        matched_blocks,
                    )
                return (
                    "ambiguous",
                    f"overlaps_mixed_unshared_block:{mixed_symbol_id}:{block['block_id']}",
                    matched_symbols,
                    matched_blocks,
                )
        if touched_any_mixed:
            return ("ambiguous", f"touches_mixed_symbol:{mixed_symbol_id}", matched_symbols, matched_blocks)

    for symbol_entry in allowlist.get("shared_editable_symbols", []):
        if symbol_entry["path"] == path and _overlaps(window, symbol_entry["lines"]):
            matched_symbols.append(symbol_entry["symbol_id"])
            return ("eligible", f"overlaps_shared_symbol:{symbol_entry['symbol_id']}", matched_symbols, matched_blocks)

    shared_paths = {entry["path"] for entry in allowlist.get("shared_editable_symbols", [])}
    mixed_paths = {entry["path"] for entry in allowlist.get("mixed_symbols", [])}
    forbidden_paths = {
        entry["path"]
        for entries in allowlist.get("forbidden_symbols", {}).values()
        for entry in entries
    }
    if path in shared_paths or path in mixed_paths or path in forbidden_paths:
        return ("ambiguous", f"touches_unclassified_region_in_shared_file:{path}", matched_symbols, matched_blocks)

    return ("ambiguous", f"path_outside_traced_surface:{path}", matched_symbols, matched_blocks)


def validate_candidate_snapshot(
    *,
    base_root: Path,
    candidate_root: Path,
    allowlist_path: Path,
) -> dict[str, Any]:
    allowlist = _load_allowlist(allowlist_path)
    base_files = _iter_source_files(base_root)
    candidate_files = _iter_source_files(candidate_root)
    all_paths = sorted(set(base_files) | set(candidate_files))

    decisions: list[ChangeDecision] = []

    for rel_path in all_paths:
        before = base_files.get(rel_path)
        after = candidate_files.get(rel_path)

        if before is None or after is None:
            verdict, reason, matched_symbols, matched_blocks = _classify_changed_window(
                rel_path,
                (1, 1),
                allowlist,
            )
            if before is None:
                if verdict == "eligible":
                    verdict = "ambiguous"
                    reason = f"new_file_requires_symbol_mapping:{rel_path}"
                decisions.append(
                    ChangeDecision(
                        file_path=rel_path,
                        symbol="<new_file>",
                        before_lines=(0, 0),
                        after_lines=(1, 1),
                        added_lines=1,
                        removed_lines=0,
                        verdict=verdict,
                        reason=reason,
                        matched_symbols=matched_symbols,
                        matched_blocks=matched_blocks,
                    )
                )
            else:
                if verdict == "eligible":
                    verdict = "ambiguous"
                    reason = f"deleted_file_requires_symbol_mapping:{rel_path}"
                decisions.append(
                    ChangeDecision(
                        file_path=rel_path,
                        symbol="<deleted_file>",
                        before_lines=(1, 1),
                        after_lines=(0, 0),
                        added_lines=0,
                        removed_lines=1,
                        verdict=verdict,
                        reason=reason,
                        matched_symbols=matched_symbols,
                        matched_blocks=matched_blocks,
                    )
                )
            continue

        before_text = _safe_read_text(before)
        after_text = _safe_read_text(after)
        if before_text is None or after_text is None:
            decisions.append(
                ChangeDecision(
                    file_path=rel_path,
                    symbol="<unreadable_file>",
                    before_lines=(0, 0),
                    after_lines=(0, 0),
                    added_lines=0,
                    removed_lines=0,
                    verdict="ineligible",
                    reason="unreadable_or_binary_file_change",
                    matched_symbols=[],
                    matched_blocks=[],
                )
            )
            continue
        if before_text == after_text:
            continue

        before_lines = before_text.splitlines()
        after_lines = after_text.splitlines()
        matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines)
        before_spans = _python_symbol_spans(before_text) if rel_path.endswith(".py") else []
        after_spans = _python_symbol_spans(after_text) if rel_path.endswith(".py") else []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            base_window = _normalize_change_window(i1, i2, len(before_lines))
            symbol = _find_symbol(after_spans, j1 + 1, max(j2, j1 + 1)) or _find_symbol(before_spans, *base_window) or "<module>"
            verdict, reason, matched_symbols, matched_blocks = _classify_changed_window(
                rel_path,
                base_window,
                allowlist,
            )
            decisions.append(
                ChangeDecision(
                    file_path=rel_path,
                    symbol=symbol,
                    before_lines=(i1 + 1, i2),
                    after_lines=(j1 + 1, j2),
                    added_lines=max(j2 - j1, 0),
                    removed_lines=max(i2 - i1, 0),
                    verdict=verdict,
                    reason=reason,
                    matched_symbols=matched_symbols,
                    matched_blocks=matched_blocks,
                )
            )

    overall_verdict = "eligible"
    if not decisions:
        overall_verdict = "ineligible"
    elif any(item.verdict == "ineligible" for item in decisions):
        overall_verdict = "ineligible"
    elif any(item.verdict == "ambiguous" for item in decisions):
        overall_verdict = "ambiguous"
    elif not any(item.verdict == "eligible" for item in decisions):
        overall_verdict = "ineligible"

    summary = {
        "base_root": str(base_root),
        "candidate_root": str(candidate_root),
        "allowlist_path": str(allowlist_path),
        "verdict": overall_verdict,
        "eligible_change_count": sum(1 for item in decisions if item.verdict == "eligible"),
        "ambiguous_change_count": sum(1 for item in decisions if item.verdict == "ambiguous"),
        "ineligible_change_count": sum(1 for item in decisions if item.verdict == "ineligible"),
        "changes": [asdict(item) for item in decisions],
    }
    if not decisions:
        summary["reason"] = "no_code_changes"
    return summary


def validate_patch_file(
    *,
    patch_path: Path,
    allowlist_path: Path,
    base_root: Path,
) -> dict[str, Any]:
    allowlist = _load_allowlist(allowlist_path)
    patch_text = patch_path.read_text(encoding="utf-8")
    deltas = parse_patch_file(patch_path)
    if patch_text.strip() and not deltas:
        return {
            "base_root": str(base_root),
            "patch_path": str(patch_path),
            "allowlist_path": str(allowlist_path),
            "verdict": "ambiguous",
            "reason": "patch_format_unparseable_without_snapshot",
            "changes": [],
        }

    change_records: list[dict[str, Any]] = []
    overall_verdict = "ambiguous"
    touched_shared_path = False
    for delta in deltas:
        matched_symbols: list[str] = []
        matched_blocks: list[str] = []
        verdict, reason, _, _ = _classify_changed_window(delta.path, (1, 1), allowlist)
        shared_paths = {entry["path"] for entry in allowlist.get("shared_editable_symbols", [])}
        if delta.path in shared_paths:
            touched_shared_path = True
            if verdict == "eligible":
                verdict = "ambiguous"
                reason = f"patch_path_only_requires_snapshot_for_symbol_level_gate:{delta.path}"
        change_records.append(
            {
                "file_path": delta.path,
                "symbol": "<patch_block>",
                "before_lines": [0, 0],
                "after_lines": [0, 0],
                "added_lines": delta.added_lines,
                "removed_lines": delta.removed_lines,
                "verdict": verdict,
                "reason": reason,
                "matched_symbols": matched_symbols,
                "matched_blocks": matched_blocks,
            }
        )
        if verdict == "ineligible":
            overall_verdict = "ineligible"
    if not deltas:
        overall_verdict = "ineligible"
    elif overall_verdict != "ineligible" and not touched_shared_path:
        overall_verdict = "ineligible"
    return {
        "base_root": str(base_root),
        "patch_path": str(patch_path),
        "allowlist_path": str(allowlist_path),
        "verdict": overall_verdict,
        "changes": change_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate transfer eligibility for a candidate snapshot or patch.")
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST_PATH)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--patch", type=Path)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    if bool(args.candidate_root) == bool(args.patch):
        parser.error("Provide exactly one of --candidate-root or --patch.")

    if args.candidate_root:
        result = validate_candidate_snapshot(
            base_root=args.base_root,
            candidate_root=args.candidate_root,
            allowlist_path=args.allowlist,
        )
    else:
        result = validate_patch_file(
            patch_path=args.patch,
            allowlist_path=args.allowlist,
            base_root=args.base_root,
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
