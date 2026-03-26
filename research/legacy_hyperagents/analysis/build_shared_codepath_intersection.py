"""Trace the shared executed task surface for paper_review and imo_grading."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib
import json
import os
import re
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml


DEFAULT_BASE_ROOT = Path(
    "/Users/scasella/Downloads/hyperagents/HyperAgents/analysis/outputs/first_transfer_pilot/frozen_roots/shared_repaired_baseline"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/scasella/Downloads/hyperagents/HyperAgents/analysis/outputs/shared_path_transfer"
)
DEFAULT_ALLOWLIST_PATH = Path(
    "/Users/scasella/Downloads/hyperagents/HyperAgents/configs/baseline_freeze/shared_path_allowlist.yaml"
)
DEFAULT_REPORT_PATH = Path(
    "/Users/scasella/Downloads/hyperagents/HyperAgents/reports/shared_path_transfer_plan.md"
)

TRACE_MODEL = "gemini-3-flash-preview"
TRACE_SUBSET = "_filtered_100_val"
TRACE_NUM_SAMPLES = 25
TRACE_NUM_WORKERS = 1
TRACE_SAVE_INTERVAL = 5

MODULE_PREFIXES = (
    "domains",
    "utils",
    "agent",
    "task_agent",
    "meta_agent",
    "run_meta_agent",
    "generate_loop",
    "select_next_parent",
)

NOISE_PATH_PREFIXES = [
    "outputs/",
    "analysis/outputs/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".venv/",
    "venv/",
    "reports/",
    "docs/",
    "configs/",
]

FORBIDDEN_SOURCE_ONLY_PATH_PREFIXES = [
    "domains/paper_review/",
]
FORBIDDEN_TARGET_ONLY_PATH_PREFIXES = [
    "domains/imo/",
]
FORBIDDEN_REPORTING_PATH_PREFIXES = [
    "domains/report.py",
]

MIXED_SYMBOL_ID = "agent/llm.py::_infer_gemini_response_schema"
PAPER_REVIEW_BRANCH_MARKER = '\'"domain": "paper_review"\''
IMO_BRANCH_MARKER = '\'"domain": "imo_grading"\''
PAPER_REVIEW_ALT_BRANCH_MARKER = "'domain': 'paper_review'"
IMO_ALT_BRANCH_MARKER = "'domain': 'imo_grading'"


@dataclass(frozen=True)
class SymbolSpan:
    name: str
    lineno: int
    end_lineno: int


@dataclass(frozen=True)
class TraceEntry:
    symbol_id: str
    file_path: str
    symbol: str
    call_count: int
    domains: tuple[str, ...]
    branches: tuple[str, ...]
    return_labels: tuple[str, ...]


def _append_unique(values: list[str], item: str | None) -> None:
    if item is None:
        return
    if item not in values:
        values.append(item)


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _clear_snapshot_modules() -> None:
    for name in list(sys.modules):
        if name == "analysis.build_shared_codepath_intersection":
            continue
        if name in MODULE_PREFIXES or any(name.startswith(prefix + ".") for prefix in MODULE_PREFIXES):
            del sys.modules[name]


def _new_entry(symbol_id: str, file_path: str, symbol: str) -> dict[str, Any]:
    return {
        "symbol_id": symbol_id,
        "file_path": file_path,
        "symbol": symbol,
        "call_count": 0,
        "domains": [],
        "branches": [],
        "return_labels": [],
        "executed": False,
    }


def _extract_domain_from_inputs(args: tuple[Any, ...], _: dict[str, Any]) -> str | None:
    if not args:
        return None
    payload = args[0]
    if isinstance(payload, dict):
        return payload.get("domain")
    return None


def _extract_domain_from_task_forward(args: tuple[Any, ...], _: dict[str, Any]) -> str | None:
    if len(args) < 2:
        return None
    payload = args[1]
    if isinstance(payload, dict):
        return payload.get("domain")
    return None


def _extract_domain_from_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    if args:
        return str(args[0])
    value = kwargs.get("domain")
    return str(value) if value is not None else None


def _extract_domain_from_run_agent(args: tuple[Any, ...], _: dict[str, Any]) -> str | None:
    if len(args) < 5:
        return None
    row = args[2]
    format_input_dict = args[4]
    try:
        payload = format_input_dict(row)
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload.get("domain")
    return None


def _extract_domain_from_text_message(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    msg = None
    if args:
        msg = args[0]
    if msg is None:
        msg = kwargs.get("msg")
    if not isinstance(msg, str):
        return None
    if PAPER_REVIEW_BRANCH_MARKER in msg or PAPER_REVIEW_ALT_BRANCH_MARKER in msg:
        return "paper_review"
    if IMO_BRANCH_MARKER in msg or IMO_ALT_BRANCH_MARKER in msg:
        return "imo_grading"
    return None


def _extract_parse_label(result: Any) -> str | None:
    label = getattr(result, "label", None)
    return str(label) if label is not None else None


def _extract_schema_branch(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    msg = None
    if args:
        msg = args[0]
    if msg is None:
        msg = kwargs.get("msg")
    if not isinstance(msg, str):
        return None
    if PAPER_REVIEW_BRANCH_MARKER in msg or PAPER_REVIEW_ALT_BRANCH_MARKER in msg:
        return "paper_review_branch"
    if IMO_BRANCH_MARKER in msg or IMO_ALT_BRANCH_MARKER in msg:
        return "imo_grading_branch"
    if "<json>" in msg or "Return exactly one JSON object" in msg:
        return "generic_json_branch"
    return "no_schema_branch"


def _wrap_attribute(
    target: Any,
    attr_name: str,
    symbol_id: str,
    file_path: str,
    symbol: str,
    trace_data: dict[str, dict[str, Any]],
    *,
    domain_extractor: Callable[[tuple[Any, ...], dict[str, Any]], str | None] | None = None,
    branch_extractor: Callable[[tuple[Any, ...], dict[str, Any]], str | None] | None = None,
    return_extractor: Callable[[Any], str | None] | None = None,
) -> None:
    original = getattr(target, attr_name)
    trace_data.setdefault(symbol_id, _new_entry(symbol_id, file_path, symbol))

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        entry = trace_data[symbol_id]
        entry["executed"] = True
        entry["call_count"] += 1
        if domain_extractor is not None:
            _append_unique(entry["domains"], domain_extractor(args, kwargs))
        if branch_extractor is not None:
            _append_unique(entry["branches"], branch_extractor(args, kwargs))
        result = original(*args, **kwargs)
        if return_extractor is not None:
            _append_unique(entry["return_labels"], return_extractor(result))
        return result

    setattr(target, attr_name, wrapped)


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
    return spans


def _symbol_line_span(root: Path, file_path: str, symbol: str) -> tuple[int, int]:
    if symbol == "<module>":
        text = (root / file_path).read_text(encoding="utf-8")
        return (1, max(1, len(text.splitlines())))
    spans = _python_symbol_spans((root / file_path).read_text(encoding="utf-8"))
    for item in spans:
        if item.name == symbol:
            return (item.lineno, item.end_lineno)
    raise KeyError(f"Could not locate symbol span for {file_path}::{symbol}")


def _find_line(lines: list[str], needle: str) -> int:
    for index, line in enumerate(lines, start=1):
        if needle in line:
            return index
    raise KeyError(f"Could not find marker {needle!r}")


def _mixed_symbol_blocks(root: Path) -> list[dict[str, Any]]:
    llm_path = root / "agent/llm.py"
    lines = llm_path.read_text(encoding="utf-8").splitlines()
    function_start, function_end = _symbol_line_span(root, "agent/llm.py", "_infer_gemini_response_schema")
    paper_start = _find_line(lines, PAPER_REVIEW_BRANCH_MARKER)
    imo_start = _find_line(lines, IMO_BRANCH_MARKER)
    generic_start = _find_line(lines, 'if "<json>" in msg or "Return exactly one JSON object" in msg:')
    ordered_branches = sorted(
        [
            ("imo_grading_branch", "target_only", imo_start),
            ("paper_review_branch", "source_only", paper_start),
            ("generic_json_branch", "ambiguous", generic_start),
        ],
        key=lambda item: item[2],
    )
    blocks: list[dict[str, Any]] = []
    for index, (block_id, category, start_line) in enumerate(ordered_branches):
        next_start = ordered_branches[index + 1][2] if index + 1 < len(ordered_branches) else function_end + 1
        blocks.append(
            {
                "block_id": block_id,
                "category": category,
                "lines": [start_line, next_start - 1],
            }
        )
    return blocks


def _trace_snapshot(
    snapshot_root: Path,
    *,
    domain: str,
    run_id: str,
    subset: str,
    num_samples: int,
    num_workers: int,
    save_interval: int,
    model: str,
    output_dir: Path,
) -> dict[str, Any]:
    trace_data: dict[str, dict[str, Any]] = {}
    snapshot_root = snapshot_root.resolve()
    tracked_files = {
        "task_agent.py",
        "agent/llm_withtools.py",
        "agent/llm.py",
        "utils/prediction_contracts.py",
        "domains/harness.py",
    }

    def register_call(file_path: str, symbol: str) -> None:
        symbol_id = f"{file_path}::{symbol}"
        entry = trace_data.setdefault(symbol_id, _new_entry(symbol_id, file_path, symbol))
        entry["executed"] = True
        entry["call_count"] += 1
        _append_unique(entry["domains"], domain)

    def profiler(frame, event, arg):  # type: ignore[no-untyped-def]
        if event != "call":
            return profiler
        try:
            filename = Path(frame.f_code.co_filename).resolve()
            relative = str(filename.relative_to(snapshot_root))
        except Exception:
            return profiler
        if relative not in tracked_files:
            return profiler
        register_call(relative, frame.f_code.co_qualname)
        return profiler

    with _pushd(snapshot_root):
        sys.path.insert(0, str(snapshot_root))
        try:
            _clear_snapshot_modules()
            llm_mod = importlib.import_module("agent.llm")
            harness_mod = importlib.import_module("domains.harness")
            original_schema_fn = llm_mod._infer_gemini_response_schema

            def traced_schema_fn(msg: str):
                entry = trace_data.setdefault(
                    MIXED_SYMBOL_ID,
                    _new_entry(MIXED_SYMBOL_ID, "agent/llm.py", "_infer_gemini_response_schema"),
                )
                entry["executed"] = True
                _append_unique(entry["domains"], domain)
                _append_unique(entry["branches"], _extract_schema_branch((msg,), {}))
                return original_schema_fn(msg)

            llm_mod._infer_gemini_response_schema = traced_schema_fn

            sys.setprofile(profiler)
            threading.setprofile(profiler)
            try:
                harness_output = harness_mod.harness(
                    agent_path="./task_agent.py",
                    output_dir=str(output_dir),
                    run_id=run_id,
                    domain=domain,
                    model=model,
                    num_samples=num_samples,
                    save_interval=save_interval,
                    num_workers=num_workers,
                    subset=subset,
                )
            finally:
                sys.setprofile(None)
                threading.setprofile(None)
        finally:
            try:
                sys.path.remove(str(snapshot_root))
            except ValueError:
                pass

    entries = [
        TraceEntry(
            symbol_id=symbol_id,
            file_path=entry["file_path"],
            symbol=entry["symbol"],
            call_count=int(entry["call_count"]),
            domains=tuple(entry["domains"]),
            branches=tuple(entry["branches"]),
            return_labels=tuple(entry["return_labels"]),
        )
        for symbol_id, entry in sorted(trace_data.items())
    ]
    summary = {
        "snapshot_root": str(snapshot_root),
        "domain": domain,
        "run_id": run_id,
        "subset": subset,
        "num_samples": num_samples,
        "num_workers": num_workers,
        "save_interval": save_interval,
        "model": model,
        "harness_output": str(Path(harness_output).resolve()),
        "trace": [dataclasses.asdict(entry) for entry in entries],
    }
    return summary


def _status_for_symbol(source_calls: int, target_calls: int) -> str:
    if source_calls > 0 and target_calls > 0:
        return "shared_executed"
    if source_calls > 0:
        return "source_only"
    if target_calls > 0:
        return "target_only"
    return "unexecuted"


def _build_allowlist(
    base_root: Path,
    symbols: list[dict[str, Any]],
    *,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    def surface_relevant(symbol_name: str) -> bool:
        if "<locals>" in symbol_name or "<genexpr>" in symbol_name:
            return False
        if symbol_name in {"<module>", "ParseResult", "PredictionContract", "TaskAgent"}:
            return False
        return True

    shared_editable_symbols: list[dict[str, Any]] = []
    source_only_symbols: list[dict[str, Any]] = []
    target_only_symbols: list[dict[str, Any]] = []

    for symbol in symbols:
        if not surface_relevant(symbol["symbol"]):
            continue
        symbol_id = symbol["symbol_id"]
        file_path = symbol["file_path"]
        line_start, line_end = _symbol_line_span(base_root, file_path, symbol["symbol"])
        entry = {
            "symbol_id": symbol_id,
            "path": file_path,
            "symbol": symbol["symbol"],
            "lines": [line_start, line_end],
            "reason": symbol["reason"],
        }
        if symbol["status"] == "shared_executed" and symbol_id != MIXED_SYMBOL_ID:
            shared_editable_symbols.append(entry)
        elif symbol["status"] == "source_only":
            source_only_symbols.append(entry)
        elif symbol["status"] == "target_only":
            target_only_symbols.append(entry)

    mixed_blocks = _mixed_symbol_blocks(base_root)

    return {
        "version": 1,
        "base_root": str(base_root),
        "domains": {
            "source": "paper_review",
            "target": "imo_grading",
        },
        "evaluation": {
            "model": evaluation["model"],
            "subset": evaluation["subset"],
            "num_samples": evaluation["num_samples"],
            "num_workers": evaluation["num_workers"],
            "save_interval": evaluation["save_interval"],
        },
        "policy": {
            "allow_new_files": False,
            "eligible_requires_shared_change": True,
            "ambiguous_is_rejected": True,
        },
        "noise_excludes": {
            "path_prefixes": NOISE_PATH_PREFIXES,
        },
        "forbidden_path_prefixes": {
            "source_only": FORBIDDEN_SOURCE_ONLY_PATH_PREFIXES,
            "target_only": FORBIDDEN_TARGET_ONLY_PATH_PREFIXES,
            "reporting_only": FORBIDDEN_REPORTING_PATH_PREFIXES,
        },
        "shared_editable_symbols": shared_editable_symbols,
        "forbidden_symbols": {
            "source_only": source_only_symbols,
            "target_only": target_only_symbols,
        },
        "mixed_symbols": [
            {
                "symbol_id": MIXED_SYMBOL_ID,
                "path": "agent/llm.py",
                "symbol": "_infer_gemini_response_schema",
                "policy": "ambiguous",
                "reason": "The function executes for both domains, but the exercised branches are domain-local and there is no safe shared branch inside the traced surface.",
                "blocks": mixed_blocks,
            }
        ],
    }


def _combine_trace_payloads(
    *,
    base_root: Path,
    output_dir: Path,
    allowlist_path: Path,
    report_path: Path,
    source_trace: dict[str, Any],
    target_trace: dict[str, Any],
) -> dict[str, Any]:
    source_trace_path = output_dir / "traces" / "paper_review_trace.json"
    target_trace_path = output_dir / "traces" / "imo_grading_trace.json"
    source_entries = {entry["symbol_id"]: entry for entry in source_trace["trace"]}
    target_entries = {entry["symbol_id"]: entry for entry in target_trace["trace"]}

    symbol_ids = sorted(set(source_entries) | set(target_entries))
    symbols: list[dict[str, Any]] = []
    for symbol_id in symbol_ids:
        source_entry = source_entries.get(symbol_id, {})
        target_entry = target_entries.get(symbol_id, {})
        file_path = source_entry.get("file_path") or target_entry.get("file_path")
        symbol = source_entry.get("symbol") or target_entry.get("symbol")
        source_calls = int(source_entry.get("call_count", 0))
        target_calls = int(target_entry.get("call_count", 0))
        status = _status_for_symbol(source_calls, target_calls)
        reason = {
            "shared_executed": "Executed on both paper_review and imo_grading.",
            "source_only": "Executed only on paper_review.",
            "target_only": "Executed only on imo_grading.",
            "unexecuted": "Not exercised in the traced validation runs.",
        }[status]
        symbol_record = {
            "symbol_id": symbol_id,
            "file_path": file_path,
            "symbol": symbol,
            "status": status,
            "reason": reason,
            "source_call_count": source_calls,
            "target_call_count": target_calls,
            "source_domains": source_entry.get("domains", []),
            "target_domains": target_entry.get("domains", []),
            "source_branches": source_entry.get("branches", []),
            "target_branches": target_entry.get("branches", []),
            "source_return_labels": source_entry.get("return_labels", []),
            "target_return_labels": target_entry.get("return_labels", []),
        }
        symbols.append(symbol_record)

    allowlist = _build_allowlist(
        base_root,
        symbols,
        evaluation={
            "model": source_trace["model"],
            "subset": source_trace["subset"],
            "num_samples": source_trace["num_samples"],
            "num_workers": source_trace["num_workers"],
            "save_interval": source_trace["save_interval"],
        },
    )
    allowlist_path.parent.mkdir(parents=True, exist_ok=True)
    allowlist_path.write_text(yaml.safe_dump(allowlist, sort_keys=False), encoding="utf-8")

    mixed_symbols = [
        {
            "symbol_id": MIXED_SYMBOL_ID,
            "path": "agent/llm.py",
            "source_branches": source_entries.get(MIXED_SYMBOL_ID, {}).get("branches", []),
            "target_branches": target_entries.get(MIXED_SYMBOL_ID, {}).get("branches", []),
            "safe_shared_edit_surface": [],
            "reason": "The function is shared at file level, but paper_review and imo_grading hit different domain-specific branches inside it.",
        },
        {
            "symbol_id": "utils/prediction_contracts.py::dispatch_surface",
            "path": "utils/prediction_contracts.py",
            "source_only_symbols": [
                "utils/prediction_contracts.py::_build_paper_review_instruction",
                "utils/prediction_contracts.py::_parse_paper_review_output",
            ],
            "target_only_symbols": [
                "utils/prediction_contracts.py::_build_imo_grading_instruction",
                "utils/prediction_contracts.py::_parse_imo_grading_output",
            ],
            "shared_symbols": [
                "utils/prediction_contracts.py::get_prediction_contract",
                "utils/prediction_contracts.py::build_task_instruction",
                "utils/prediction_contracts.py::parse_prediction_output",
            ],
            "reason": "The file is shared, but only the dispatch helpers are safe shared edit surfaces. Domain-specific builders and parsers remain forbidden.",
        },
    ]

    def include_in_report(symbol: dict[str, Any]) -> bool:
        if "<locals>" in symbol["symbol"] or "<genexpr>" in symbol["symbol"]:
            return False
        if symbol["symbol"] in {"<module>", "ParseResult", "PredictionContract", "TaskAgent"}:
            return False
        return True

    summary = {
        "base_root": str(base_root),
        "source_trace_path": str(source_trace_path),
        "target_trace_path": str(target_trace_path),
        "allowlist_path": str(allowlist_path),
        "evaluation": allowlist["evaluation"],
        "shared_executed_symbols": [
            symbol
            for symbol in symbols
            if symbol["status"] == "shared_executed"
            and symbol["symbol_id"] != MIXED_SYMBOL_ID
            and include_in_report(symbol)
        ],
        "source_only_symbols": [
            symbol
            for symbol in symbols
            if symbol["status"] == "source_only"
            and include_in_report(symbol)
        ],
        "target_only_symbols": [
            symbol
            for symbol in symbols
            if symbol["status"] == "target_only"
            and include_in_report(symbol)
        ],
        "mixed_symbols": mixed_symbols,
    }

    summary_path = output_dir / "shared_codepath_intersection.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_lines = [
        "# Shared-Path Transfer Plan",
        "",
        "## Goal",
        "",
        "Trace the repaired frozen baseline on both `paper_review` and `imo_grading`, then treat only the shared executed task surface as transfer-eligible for the next source search.",
        "",
        "## Traced setup",
        "",
        f"- base snapshot: `{base_root}`",
        f"- model: `{allowlist['evaluation']['model']}`",
        f"- subset: `{allowlist['evaluation']['subset']}`",
        f"- examples per domain trace: `{allowlist['evaluation']['num_samples']}`",
        "",
        "## Shared exercised codepaths",
        "",
    ]
    for symbol in summary["shared_executed_symbols"]:
        report_lines.append(
            f"- `{symbol['symbol_id']}`: executed in both domains (`paper_review` calls: {symbol['source_call_count']}, `imo_grading` calls: {symbol['target_call_count']})"
        )
    report_lines.extend(
        [
            "",
            "## Tempting Shared Files That Are Not Fully Transfer-Eligible",
            "",
            "- `utils/prediction_contracts.py` is shared at file level, but only the dispatch helpers are shared. The actual `paper_review` builder/parser and `imo_grading` builder/parser stay domain-local.",
            "- `agent/llm.py::_infer_gemini_response_schema` executes in both domains, but the exercised branch is different in each domain. The paper-review schema branch and IMO schema branch are both forbidden for the shared-only search.",
            "",
            "## Source-only exercised codepaths",
            "",
        ]
    )
    for symbol in summary["source_only_symbols"]:
        report_lines.append(f"- `{symbol['symbol_id']}`")
    report_lines.extend(
        [
            "",
            "## Target-only exercised codepaths",
            "",
        ]
    )
    for symbol in summary["target_only_symbols"]:
        report_lines.append(f"- `{symbol['symbol_id']}`")
    report_lines.extend(
        [
            "",
            "## Safe transfer-eligible edit surface",
            "",
            "The shared-path gate should allow edits only to these traced shared symbols:",
            "",
        ]
    )
    for symbol in allowlist["shared_editable_symbols"]:
        report_lines.append(
            f"- `{symbol['symbol_id']}` lines `{symbol['lines'][0]}-{symbol['lines'][1]}`"
        )
    report_lines.extend(
        [
            "",
            "The gate should reject:",
            "",
            "- any change under `domains/paper_review/` or `domains/imo/`",
            "- any change to `utils/prediction_contracts.py::_build_paper_review_instruction` or `_parse_paper_review_output`",
            "- any change to `utils/prediction_contracts.py::_build_imo_grading_instruction` or `_parse_imo_grading_output`",
            "- any change inside `agent/llm.py::_infer_gemini_response_schema`, because the traced exercised branches are domain-local",
            "- docs, reports, outputs, caches, and other non-predictive noise",
            "",
            "## Output artifacts",
            "",
            f"- shared trace summary: `{summary_path}`",
            f"- source trace: `{source_trace_path}`",
            f"- target trace: `{target_trace_path}`",
            f"- allowlist: `{allowlist_path}`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return summary


def build_shared_intersection(
    base_root: Path,
    *,
    output_dir: Path,
    allowlist_path: Path,
    report_path: Path,
    model: str = TRACE_MODEL,
    subset: str = TRACE_SUBSET,
    num_samples: int = TRACE_NUM_SAMPLES,
    num_workers: int = TRACE_NUM_WORKERS,
    save_interval: int = TRACE_SAVE_INTERVAL,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    source_trace_path = traces_dir / "paper_review_trace.json"
    target_trace_path = traces_dir / "imo_grading_trace.json"

    subprocess_env = os.environ.copy()
    common_args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--base-root",
        str(base_root),
        "--output-dir",
        str(output_dir),
        "--model",
        model,
        "--subset",
        subset,
        "--num-samples",
        str(num_samples),
        "--num-workers",
        str(num_workers),
        "--save-interval",
        str(save_interval),
    ]
    for domain_name, trace_output in [
        ("paper_review", source_trace_path),
        ("imo_grading", target_trace_path),
    ]:
        completed = subprocess.run(
            common_args
            + [
                "--single-domain",
                domain_name,
                "--trace-output",
                str(trace_output),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=subprocess_env,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Single-domain trace failed for {domain_name}.\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )

    source_trace = json.loads(source_trace_path.read_text(encoding="utf-8"))
    target_trace = json.loads(target_trace_path.read_text(encoding="utf-8"))
    return _combine_trace_payloads(
        base_root=base_root,
        output_dir=output_dir,
        allowlist_path=allowlist_path,
        report_path=report_path,
        source_trace=source_trace,
        target_trace=target_trace,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace the shared executed codepath intersection for paper_review and imo_grading.")
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allowlist-path", type=Path, default=DEFAULT_ALLOWLIST_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--model", type=str, default=TRACE_MODEL)
    parser.add_argument("--subset", type=str, default=TRACE_SUBSET)
    parser.add_argument("--num-samples", type=int, default=TRACE_NUM_SAMPLES)
    parser.add_argument("--num-workers", type=int, default=TRACE_NUM_WORKERS)
    parser.add_argument("--save-interval", type=int, default=TRACE_SAVE_INTERVAL)
    parser.add_argument("--single-domain", type=str, choices=["paper_review", "imo_grading"])
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--combine-only", action="store_true", default=False)
    args = parser.parse_args()

    if args.single_domain:
        if args.trace_output is None:
            parser.error("--trace-output is required with --single-domain.")
        summary = _trace_snapshot(
            args.base_root,
            domain=args.single_domain,
            run_id=f"shared_path_trace_{args.single_domain}_val25",
            subset=args.subset,
            num_samples=args.num_samples,
            num_workers=args.num_workers,
            save_interval=args.save_interval,
            model=args.model,
            output_dir=args.output_dir / "eval_outputs",
        )
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.trace_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.combine_only:
        source_trace = json.loads((args.output_dir / "traces" / "paper_review_trace.json").read_text(encoding="utf-8"))
        target_trace = json.loads((args.output_dir / "traces" / "imo_grading_trace.json").read_text(encoding="utf-8"))
        summary = _combine_trace_payloads(
            base_root=args.base_root,
            output_dir=args.output_dir,
            allowlist_path=args.allowlist_path,
            report_path=args.report_path,
            source_trace=source_trace,
            target_trace=target_trace,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    summary = build_shared_intersection(
        args.base_root,
        output_dir=args.output_dir,
        allowlist_path=args.allowlist_path,
        report_path=args.report_path,
        model=args.model,
        subset=args.subset,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        save_interval=args.save_interval,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
