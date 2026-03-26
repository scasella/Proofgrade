"""Classify the exact transfer delta between frozen pilot roots."""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))


DEFAULT_BASE_ROOT = Path(
    "/Users/scasella/Downloads/hyperagents/HyperAgents/analysis/outputs/first_transfer_pilot/frozen_roots/shared_repaired_baseline"
)
DEFAULT_TARGET_ROOT = Path(
    "/Users/scasella/Downloads/hyperagents/HyperAgents/analysis/outputs/first_transfer_pilot/frozen_roots/transferred_full_agent"
)
DEFAULT_OUTPUT_PATH = Path(
    "/Users/scasella/Downloads/hyperagents/HyperAgents/analysis/outputs/first_transfer_pilot/transfer_delta_manifest.json"
)

EXCLUDED_DIRS = {
    "__pycache__",
    "outputs",
    ".git",
    ".venv312",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
EXCLUDED_NAMES = {".DS_Store"}

CATEGORY_PAPER_REVIEW = "domain_local_paper_review_only"
CATEGORY_IMO = "domain_local_imo_grading_only"
CATEGORY_SHARED_CONTRACT = "shared_task_output_contract"
CATEGORY_SHARED_PARSER = "shared_parsing_normalization"
CATEGORY_SHARED_LLM = "shared_llm_provider_behavior"
CATEGORY_META = "meta_agent_self_improvement"
CATEGORY_REPORT = "evaluation_reporting_only"
CATEGORY_DEAD = "dead_inert_cosmetic"


@dataclass(frozen=True)
class SymbolSpan:
    name: str
    lineno: int
    end_lineno: int


@dataclass(frozen=True)
class ChangeRecord:
    change_id: str
    file_path: str
    symbol: str
    before_lines: tuple[int, int]
    after_lines: tuple[int, int]
    added_lines: int
    removed_lines: int
    category: str
    transfer_eligible: bool
    expected_imo_reachability: str
    bundle_membership: list[str]
    classification_reason: str


def _iter_source_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in rel_path.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES or path.name in EXCLUDED_NAMES:
            continue
        files[str(rel_path)] = path
    return files


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _change_category(file_path: str, symbol: str) -> tuple[str, bool, str, list[str], str]:
    symbol_lower = symbol.lower()
    if "paper_review" in symbol_lower:
        bundles = ["domain_local_only_transfer", "task_only_transfer"]
        return (
            CATEGORY_PAPER_REVIEW,
            False,
            "not_reached_for_imo_domain_dispatch",
            bundles,
            "Changed symbol is only selected when the domain is `paper_review`.",
        )
    if "imo_grading" in symbol_lower:
        return (
            CATEGORY_IMO,
            True,
            "reached_for_imo_domain_dispatch",
            ["shared_only_transfer"],
            "Changed symbol is only selected when the domain is `imo_grading`.",
        )
    if file_path == "agent/llm.py":
        return (
            CATEGORY_SHARED_LLM,
            True,
            "reached_for_all_llm_calls",
            ["shared_only_transfer"],
            "Changed symbol alters shared provider behavior for the target run.",
        )
    if file_path in {"meta_agent.py", "run_meta_agent.py", "generate_loop.py"}:
        return (
            CATEGORY_META,
            True,
            "not_used_in_direct_eval_target_run",
            ["meta_only_transfer", "shared_only_transfer"],
            "Changed symbol belongs to self-improvement logic, not the direct eval harness.",
        )
    if file_path in {"domains/report.py"}:
        return (
            CATEGORY_REPORT,
            False,
            "not_used_for_prediction_generation",
            ["shared_only_transfer"],
            "Changed symbol is reporting-only and does not affect harness predictions.",
        )
    if file_path == "utils/prediction_contracts.py":
        if symbol in {"build_task_instruction", "parse_prediction_output", "get_prediction_contract"}:
            return (
                CATEGORY_SHARED_CONTRACT,
                True,
                "reached_for_imo_domain_dispatch",
                ["shared_only_transfer"],
                "Changed symbol is shared task/output contract dispatch used on every target example.",
            )
        return (
            CATEGORY_SHARED_PARSER,
            True,
            "potentially_reached_if_symbol_selected_by_domain_dispatch",
            ["shared_only_transfer"],
            "Changed symbol is in shared parsing/normalization code.",
        )
    return (
        CATEGORY_DEAD,
        False,
        "not_used",
        [],
        "Changed lines are outside the current target prediction path.",
    )


def _diff_changes(file_path: str, before_path: Path, after_path: Path) -> tuple[list[ChangeRecord], dict[str, Any]]:
    before_text = before_path.read_text(encoding="utf-8")
    after_text = after_path.read_text(encoding="utf-8")
    before_lines = before_text.splitlines()
    after_lines = after_text.splitlines()
    matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines)

    before_spans = _python_symbol_spans(before_text) if file_path.endswith(".py") else []
    after_spans = _python_symbol_spans(after_text) if file_path.endswith(".py") else []

    records: list[ChangeRecord] = []
    function_counts: dict[str, int] = {}

    for index, (tag, i1, i2, j1, j2) in enumerate(matcher.get_opcodes(), start=1):
        if tag == "equal":
            continue
        symbol = _find_symbol(after_spans, j1 + 1, j2) or _find_symbol(before_spans, i1 + 1, i2) or "<module>"
        function_counts[symbol] = function_counts.get(symbol, 0) + 1
        change_id = f"{file_path}:{symbol}:{function_counts[symbol]}"
        category, eligible, reachability, bundles, reason = _change_category(file_path, symbol)
        records.append(
            ChangeRecord(
                change_id=change_id,
                file_path=file_path,
                symbol=symbol,
                before_lines=(i1 + 1, i2),
                after_lines=(j1 + 1, j2),
                added_lines=max(j2 - j1, 0),
                removed_lines=max(i2 - i1, 0),
                category=category,
                transfer_eligible=eligible,
                expected_imo_reachability=reachability,
                bundle_membership=bundles,
                classification_reason=reason,
            )
        )

    file_summary = {
        "file_path": file_path,
        "before_sha256": _sha256(before_path),
        "after_sha256": _sha256(after_path),
        "added_lines": sum(record.added_lines for record in records),
        "removed_lines": sum(record.removed_lines for record in records),
        "changed_line_total": sum(record.added_lines + record.removed_lines for record in records),
        "changed_symbols": [record.symbol for record in records],
        "categories": sorted({record.category for record in records}),
    }
    return records, file_summary


def build_manifest(base_root: Path, target_root: Path) -> dict[str, Any]:
    base_files = _iter_source_files(base_root)
    target_files = _iter_source_files(target_root)
    all_paths = sorted(set(base_files) | set(target_files))

    changed_files: list[dict[str, Any]] = []
    changes: list[ChangeRecord] = []
    added_only: list[str] = []
    removed_only: list[str] = []

    for rel_path in all_paths:
        before = base_files.get(rel_path)
        after = target_files.get(rel_path)
        if before is None:
            added_only.append(rel_path)
            continue
        if after is None:
            removed_only.append(rel_path)
            continue
        if _sha256(before) == _sha256(after):
            continue
        file_changes, file_summary = _diff_changes(rel_path, before, after)
        changes.extend(file_changes)
        changed_files.append(file_summary)

    changed_ids = [change.change_id for change in changes]
    shared_only_ids = [change.change_id for change in changes if "shared_only_transfer" in change.bundle_membership and change.transfer_eligible]
    domain_local_ids = [change.change_id for change in changes if "domain_local_only_transfer" in change.bundle_membership]
    task_only_ids = [change.change_id for change in changes if "task_only_transfer" in change.bundle_membership]
    meta_only_ids = [change.change_id for change in changes if "meta_only_transfer" in change.bundle_membership]

    bundles = {
        "shared_only_transfer": {
            "included_change_ids": shared_only_ids,
            "excluded_change_ids": [change_id for change_id in changed_ids if change_id not in shared_only_ids],
            "equivalent_to_existing_arm": "baseline" if not shared_only_ids else "none",
            "reason": "No target-reachable shared changes exist in the frozen delta." if not shared_only_ids else "Contains only target-reachable shared changes.",
        },
        "domain_local_only_transfer": {
            "included_change_ids": domain_local_ids,
            "excluded_change_ids": [change_id for change_id in changed_ids if change_id not in domain_local_ids],
            "equivalent_to_existing_arm": "transferred_full" if sorted(domain_local_ids) == sorted(changed_ids) else "none",
            "reason": "All meaningful changes are domain-local paper_review edits." if sorted(domain_local_ids) == sorted(changed_ids) else "Contains only paper_review-local changes.",
        },
    }
    if task_only_ids and sorted(task_only_ids) == sorted(domain_local_ids):
        bundles["task_only_transfer"] = {
            "included_change_ids": task_only_ids,
            "excluded_change_ids": [change_id for change_id in changed_ids if change_id not in task_only_ids],
            "equivalent_to_existing_arm": "transferred_full" if sorted(task_only_ids) == sorted(changed_ids) else "none",
            "reason": "Every domain-local change is task behavior / prompt logic.",
        }
    if meta_only_ids:
        bundles["meta_only_transfer"] = {
            "included_change_ids": meta_only_ids,
            "excluded_change_ids": [change_id for change_id in changed_ids if change_id not in meta_only_ids],
            "equivalent_to_existing_arm": "none",
            "reason": "Contains only cleanly separable meta-level changes.",
        }

    manifest = {
        "base_root": str(base_root),
        "target_root": str(target_root),
        "summary": {
            "changed_file_count": len(changed_files),
            "added_only_files": added_only,
            "removed_only_files": removed_only,
            "changed_line_total": sum(file_summary["changed_line_total"] for file_summary in changed_files),
            "transfer_eligible_change_count": sum(1 for change in changes if change.transfer_eligible),
            "domain_local_paper_review_change_count": sum(1 for change in changes if change.category == CATEGORY_PAPER_REVIEW),
        },
        "changed_files": changed_files,
        "changes": [asdict(change) for change in changes],
        "bundles": bundles,
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify the exact frozen transfer delta.")
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    manifest = build_manifest(args.base_root, args.target_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
