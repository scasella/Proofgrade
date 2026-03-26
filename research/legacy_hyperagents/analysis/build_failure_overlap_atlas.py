"""Build a label-safe failure-overlap atlas for candidate source domains."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.run_shared_path_source_search import _load_dataset  # noqa: E402
from utils.common import extract_jsons  # noqa: E402
from utils.prediction_contracts import JSON_LABEL_KEYS  # noqa: E402


DEFAULT_CONFIG_PATH = Path(
    "/Users/scasella/Downloads/hyperagents/HyperAgents/configs/baseline_freeze/failure_overlap_selection.yaml"
)

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

HOST_REPO_ROOT = Path(__file__).resolve().parents[1]
PARTIAL_KEY_VALUE_PATTERN = re.compile(
    r'"(?P<key>label|decision|prediction|response)"\s*:\s*"(?P<value>[^"\r\n]{1,80})"',
    re.IGNORECASE,
)

SHARED_SYMBOL_SPECS: tuple[tuple[str, str, str], ...] = (
    ("agent.llm", "normalize_model_name", "agent/llm.py::normalize_model_name"),
    ("agent.llm", "_get_gemini_api_key", "agent/llm.py::_get_gemini_api_key"),
    ("agent.llm", "_convert_messages_for_gemini", "agent/llm.py::_convert_messages_for_gemini"),
    ("agent.llm", "_get_response_from_gemini_rest", "agent/llm.py::_get_response_from_gemini_rest"),
    ("agent.llm", "get_response_from_llm", "agent/llm.py::get_response_from_llm"),
    ("agent.llm_withtools", "chat_with_agent", "agent/llm_withtools.py::chat_with_agent"),
    ("agent.llm_withtools", "check_for_tool_uses", "agent/llm_withtools.py::check_for_tool_uses"),
    ("agent.llm_withtools", "get_tooluse_prompt", "agent/llm_withtools.py::get_tooluse_prompt"),
    ("agent.llm_withtools", "should_retry_tool_use", "agent/llm_withtools.py::should_retry_tool_use"),
    ("domains.harness", "get_dataset", "domains/harness.py::get_dataset"),
    ("domains.harness", "load_task_agent", "domains/harness.py::load_task_agent"),
    ("domains.harness", "run_agent", "domains/harness.py::run_agent"),
    ("domains.harness", "harness", "domains/harness.py::harness"),
    ("task_agent", "TaskAgent.forward", "task_agent.py::TaskAgent.forward"),
    ("utils.prediction_contracts", "get_prediction_contract", "utils/prediction_contracts.py::get_prediction_contract"),
    ("utils.prediction_contracts", "build_task_instruction", "utils/prediction_contracts.py::build_task_instruction"),
    ("utils.prediction_contracts", "parse_prediction_output", "utils/prediction_contracts.py::parse_prediction_output"),
    ("utils.prediction_contracts", "_format_inputs", "utils/prediction_contracts.py::_format_inputs"),
    ("utils.prediction_contracts", "_extract_json_objects", "utils/prediction_contracts.py::_extract_json_objects"),
    ("utils.prediction_contracts", "_extract_json_label_candidate", "utils/prediction_contracts.py::_extract_json_label_candidate"),
    ("utils.prediction_contracts", "_normalize_whitespace", "utils/prediction_contracts.py::_normalize_whitespace"),
    ("utils.prediction_contracts", "_canonical_label_text", "utils/prediction_contracts.py::_canonical_label_text"),
    ("utils.prediction_contracts", "_build_paper_review_instruction", "utils/prediction_contracts.py::_build_paper_review_instruction"),
    ("utils.prediction_contracts", "_parse_paper_review_output", "utils/prediction_contracts.py::_parse_paper_review_output"),
    ("utils.prediction_contracts", "_build_imo_grading_instruction", "utils/prediction_contracts.py::_build_imo_grading_instruction"),
    ("utils.prediction_contracts", "_parse_imo_grading_output", "utils/prediction_contracts.py::_parse_imo_grading_output"),
)

SYMPtom_SURFACES: dict[str, tuple[str, ...]] = {
    "truncated_json_with_visible_label": (
        "utils/prediction_contracts.py::_extract_json_label_candidate",
        "utils/prediction_contracts.py::_extract_json_objects",
        "agent/llm.py::_get_response_from_gemini_rest",
    ),
    "truncated_json_with_visible_response": (
        "utils/prediction_contracts.py::parse_prediction_output",
        "agent/llm.py::_get_response_from_gemini_rest",
    ),
    "unclosed_json_object": (
        "utils/prediction_contracts.py::build_task_instruction",
        "agent/llm.py::_get_response_from_gemini_rest",
    ),
    "extra_text_before_json": (
        "utils/prediction_contracts.py::build_task_instruction",
        "agent/llm_withtools.py::chat_with_agent",
    ),
    "plain_text_without_json": (
        "utils/prediction_contracts.py::build_task_instruction",
        "agent/llm_withtools.py::chat_with_agent",
    ),
    "missing_label_field": (
        "utils/prediction_contracts.py::parse_prediction_output",
        "utils/prediction_contracts.py::build_task_instruction",
    ),
    "missing_response_field": (
        "utils/prediction_contracts.py::parse_prediction_output",
        "utils/prediction_contracts.py::build_task_instruction",
    ),
    "generic_schema_key_mismatch_label_instead_of_response": (
        "utils/prediction_contracts.py::parse_prediction_output",
        "utils/prediction_contracts.py::build_task_instruction",
    ),
    "tool_scaffold_echo": (
        "agent/llm_withtools.py::chat_with_agent",
        "agent/llm_withtools.py::get_tooluse_prompt",
        "agent/llm_withtools.py::should_retry_tool_use",
    ),
    "overlong_output": (
        "utils/prediction_contracts.py::build_task_instruction",
        "agent/llm.py::_get_response_from_gemini_rest",
        "agent/llm_withtools.py::chat_with_agent",
    ),
}

LOW_PRIORITY_SYMPTOMS = {"overlong_output"}


@dataclass(frozen=True)
class DomainRun:
    domain: str
    subset: str
    num_samples: int
    run_id: str
    snapshot_root: Path
    output_dir: Path
    predictions_path: Path
    trace_path: Path
    trace: dict[str, Any]
    question_id_col: str


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        if name == "analysis.build_failure_overlap_atlas":
            continue
        if name in MODULE_PREFIXES or any(name.startswith(prefix + ".") for prefix in MODULE_PREFIXES):
            del sys.modules[name]


def _append_unique(values: list[str], item: str | None) -> None:
    if item is None:
        return
    if item not in values:
        values.append(item)


def _new_trace_entry(symbol_id: str, symbol: str, file_path: str) -> dict[str, Any]:
    return {
        "symbol_id": symbol_id,
        "symbol": symbol,
        "file_path": file_path,
        "call_count": 0,
        "executed": False,
        "domains": [],
        "branches": [],
    }


def _extract_inputs_domain(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    payload = args[0] if args else kwargs.get("inputs")
    if isinstance(payload, dict):
        return str(payload.get("domain")) if payload.get("domain") is not None else None
    return None


def _extract_domain_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    if args:
        return str(args[0])
    domain = kwargs.get("domain")
    return str(domain) if domain is not None else None


def _extract_domain_from_run_agent(args: tuple[Any, ...], _: dict[str, Any]) -> str | None:
    if len(args) < 5:
        return None
    row = args[2]
    formatter = args[4]
    try:
        payload = formatter(row)
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload.get("domain")
    return None


def _extract_build_branch(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    payload = args[0] if args else kwargs.get("inputs")
    if not isinstance(payload, dict):
        return None
    domain = payload.get("domain")
    return "contract_branch" if domain in {"paper_review", "imo_grading"} else "generic_branch"


def _extract_parse_branch(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    domain = _extract_domain_arg(args, kwargs)
    return "contract_branch" if domain in {"paper_review", "imo_grading"} else "generic_branch"


def _extract_schema_branch(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    msg = args[0] if args else kwargs.get("msg")
    if not isinstance(msg, str):
        return None
    if '"domain": "paper_review"' in msg or "'domain': 'paper_review'" in msg:
        return "paper_review_schema"
    if '"domain": "imo_grading"' in msg or "'domain': 'imo_grading'" in msg:
        return "imo_grading_schema"
    if "<json>" in msg or '"response":' in msg or "'response':" in msg:
        return "generic_json_schema"
    return "no_schema"


def _wrap_function(
    target: Any,
    attr_name: str,
    trace_data: dict[str, dict[str, Any]],
    symbol_id: str,
    file_path: str,
    symbol: str,
    *,
    domain_extractor: Callable[[tuple[Any, ...], dict[str, Any]], str | None] | None = None,
    branch_extractor: Callable[[tuple[Any, ...], dict[str, Any]], str | None] | None = None,
) -> None:
    if "." in attr_name:
        owner_name, method_name = attr_name.split(".", 1)
        owner = getattr(target, owner_name)
        original = getattr(owner, method_name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            entry = trace_data.setdefault(symbol_id, _new_trace_entry(symbol_id, symbol, file_path))
            entry["executed"] = True
            entry["call_count"] += 1
            if domain_extractor:
                _append_unique(entry["domains"], domain_extractor(args, kwargs))
            if branch_extractor:
                _append_unique(entry["branches"], branch_extractor(args, kwargs))
            return original(*args, **kwargs)

        setattr(owner, method_name, wrapped)
        return

    original = getattr(target, attr_name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        entry = trace_data.setdefault(symbol_id, _new_trace_entry(symbol_id, symbol, file_path))
        entry["executed"] = True
        entry["call_count"] += 1
        if domain_extractor:
            _append_unique(entry["domains"], domain_extractor(args, kwargs))
        if branch_extractor:
            _append_unique(entry["branches"], branch_extractor(args, kwargs))
        return original(*args, **kwargs)

    setattr(target, attr_name, wrapped)


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


def _read_last_json_object(raw_text: str) -> dict[str, Any] | None:
    extracted = extract_jsons(raw_text) or []
    for obj in reversed(extracted):
        if isinstance(obj, dict):
            return obj
    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r"\{", raw_text))):
        try:
            candidate, _ = decoder.raw_decode(raw_text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    return None


def _classify_output_symptoms(domain: str, raw_text: str, prediction_value: str | None) -> list[str]:
    raw_text = raw_text or ""
    stripped = raw_text.strip()
    obj = _read_last_json_object(raw_text)
    symptoms: list[str] = []
    partial_match = PARTIAL_KEY_VALUE_PATTERN.search(raw_text)
    brace_balance = raw_text.count("{") - raw_text.count("}")

    if not stripped:
        return ["empty_output"]

    first_brace = stripped.find("{")
    if first_brace > 0:
        prefix = stripped[:first_brace].strip()
        if prefix:
            symptoms.append("extra_text_before_json")
    elif first_brace < 0 and not stripped.startswith("<json>"):
        symptoms.append("plain_text_without_json")

    if brace_balance > 0:
        symptoms.append("unclosed_json_object")

    if obj is None and partial_match:
        key = partial_match.group("key").lower()
        if key == "response":
            symptoms.append("truncated_json_with_visible_response")
        else:
            symptoms.append("truncated_json_with_visible_label")

    if obj is not None:
        if domain in {"paper_review", "imo_grading"}:
            if not any(key in obj for key in JSON_LABEL_KEYS):
                symptoms.append("missing_label_field")
        else:
            if "response" not in obj:
                symptoms.append("missing_response_field")
            if "label" in obj and "response" not in obj:
                symptoms.append("generic_schema_key_mismatch_label_instead_of_response")

    if "tool_name" in raw_text and "tool_input" in raw_text:
        symptoms.append("tool_scaffold_echo")

    if len(stripped) > 320:
        symptoms.append("overlong_output")

    if not symptoms and not (prediction_value or "").strip():
        if obj is not None and domain not in {"paper_review", "imo_grading"} and "response" not in obj:
            symptoms.append("generic_schema_key_mismatch_label_instead_of_response")
        else:
            symptoms.append("plain_text_without_json")

    return sorted(set(symptoms))


def _domain_info(domain: str) -> tuple[str, str]:
    if domain in {"paper_review", "search_arena"}:
        module = importlib.import_module(f"domains.{domain}.utils")
    elif domain.startswith("imo_"):
        module = importlib.import_module(f"domains.imo.{domain.split('_', 1)[1]}_utils")
    else:
        raise ValueError(f"Unsupported domain: {domain}")
    return str(module.QUESTION_ID), str(module.GROUND_TRUTH_KEY)


def _build_pair_allowlist(
    *,
    config: dict[str, Any],
    source_domain: str,
    source_trace: dict[str, Any],
    target_trace: dict[str, Any],
    output_path: Path,
) -> Path:
    shared_symbols: list[dict[str, Any]] = []
    source_entries = source_trace["trace"]
    target_entries = target_trace["trace"]
    for symbol_id, source_entry in source_entries.items():
        target_entry = target_entries.get(symbol_id)
        if not target_entry:
            continue
        if not source_entry.get("executed") or not target_entry.get("executed"):
            continue
        branches_source = set(source_entry.get("branches") or [])
        branches_target = set(target_entry.get("branches") or [])
        if symbol_id in {
            "utils/prediction_contracts.py::build_task_instruction",
            "utils/prediction_contracts.py::parse_prediction_output",
            "agent/llm.py::_get_response_from_gemini_rest",
        } and branches_source != branches_target:
            continue
        path = source_entry["file_path"]
        if path.startswith("domains/") or path == "domains/report.py":
            continue
        if symbol_id in {
            "utils/prediction_contracts.py::_build_paper_review_instruction",
            "utils/prediction_contracts.py::_parse_paper_review_output",
            "utils/prediction_contracts.py::_build_imo_grading_instruction",
            "utils/prediction_contracts.py::_parse_imo_grading_output",
        }:
            continue
        shared_symbols.append(
            {
                "symbol_id": symbol_id,
                "path": path,
                "symbol": source_entry["symbol"],
                "reason": f"Executed on both {source_domain} and {config['target']['domain']}.",
            }
        )

    allowlist = {
        "version": 1,
        "base_root": str(Path(config["base_root"]).resolve()),
        "domains": {"source": source_domain, "target": config["target"]["domain"]},
        "evaluation": {
            "model": config["eval_model"],
            "subset": config["target"]["subset"],
            "num_samples": config["target"]["num_samples"],
            "num_workers": config["num_workers"],
            "save_interval": config["save_interval"],
        },
        "policy": {
            "allow_new_files": False,
            "eligible_requires_shared_change": True,
            "ambiguous_is_rejected": True,
        },
        "noise_excludes": {
            "path_prefixes": [
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
        },
        "forbidden_path_prefixes": {
            "source_only": [f"domains/{source_domain}/"] if not source_domain.startswith("imo_") else ["domains/imo/"],
            "target_only": ["domains/imo/"],
            "reporting_only": ["domains/report.py"],
        },
        "shared_editable_symbols": shared_symbols,
        "forbidden_symbols": {
            "source_only": [
                {
                    "symbol_id": "utils/prediction_contracts.py::_build_paper_review_instruction",
                    "path": "utils/prediction_contracts.py",
                    "symbol": "_build_paper_review_instruction",
                    "lines": [382, 417],
                    "reason": "Paper-review-only instruction path.",
                },
                {
                    "symbol_id": "utils/prediction_contracts.py::_parse_paper_review_output",
                    "path": "utils/prediction_contracts.py",
                    "symbol": "_parse_paper_review_output",
                    "lines": [347, 362],
                    "reason": "Paper-review-only parser path.",
                },
            ],
            "target_only": [
                {
                    "symbol_id": "utils/prediction_contracts.py::_build_imo_grading_instruction",
                    "path": "utils/prediction_contracts.py",
                    "symbol": "_build_imo_grading_instruction",
                    "lines": [390, 417],
                    "reason": "IMO-grading-only instruction path.",
                },
                {
                    "symbol_id": "utils/prediction_contracts.py::_parse_imo_grading_output",
                    "path": "utils/prediction_contracts.py",
                    "symbol": "_parse_imo_grading_output",
                    "lines": [360, 374],
                    "reason": "IMO-grading-only parser path.",
                },
            ],
        },
        "mixed_symbols": [
            {
                "symbol_id": "agent/llm.py::_infer_gemini_response_schema",
                "path": "agent/llm.py",
                "symbol": "_infer_gemini_response_schema",
                "policy": "ambiguous",
                "blocks": [
                    {"block_id": "paper_review_branch", "category": "source_only", "lines": [71, 98]},
                    {"block_id": "imo_grading_branch", "category": "target_only", "lines": [60, 69]},
                    {"block_id": "generic_json_branch", "category": "unexecuted", "lines": [100, 112]},
                ],
            }
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(allowlist, sort_keys=False), encoding="utf-8")
    return output_path


def _trace_domain_run(
    *,
    snapshot_root: Path,
    domain: str,
    subset: str,
    num_samples: int,
    model: str,
    num_workers: int,
    save_interval: int,
    output_root: Path,
) -> DomainRun:
    run_id = f"failure_overlap_{domain}_{num_samples}"
    run_output_dir = output_root / "runs" / domain / run_id
    trace_path = output_root / "traces" / f"{domain}_trace.json"
    predictions_path = run_output_dir / "predictions.csv"

    if trace_path.exists() and predictions_path.exists():
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        question_id_col, _ = _domain_info(domain)
        return DomainRun(
            domain=domain,
            subset=subset,
            num_samples=num_samples,
            run_id=run_id,
            snapshot_root=snapshot_root,
            output_dir=run_output_dir,
            predictions_path=predictions_path,
            trace_path=trace_path,
            trace=trace,
            question_id_col=question_id_col,
        )

    trace_data: dict[str, dict[str, Any]] = {}
    with _pushd(snapshot_root):
        sys.path.insert(0, str(snapshot_root))
        try:
            _clear_snapshot_modules()
            llm = importlib.import_module("agent.llm")
            llm_tools = importlib.import_module("agent.llm_withtools")
            harness_mod = importlib.import_module("domains.harness")
            task_agent_mod = importlib.import_module("task_agent")
            contracts = importlib.import_module("utils.prediction_contracts")

            symbol_map = {
                "agent.llm": llm,
                "agent.llm_withtools": llm_tools,
                "domains.harness": harness_mod,
                "task_agent": task_agent_mod,
                "utils.prediction_contracts": contracts,
            }

            for module_name, attr_name, symbol_id in SHARED_SYMBOL_SPECS:
                module = symbol_map[module_name]
                file_path, symbol = symbol_id.split("::", 1)
                kwargs: dict[str, Any] = {}
                if symbol_id == "task_agent.py::TaskAgent.forward":
                    kwargs["domain_extractor"] = _extract_inputs_domain
                elif module_name in {"domains.harness", "utils.prediction_contracts"}:
                    if attr_name == "run_agent":
                        kwargs["domain_extractor"] = _extract_domain_from_run_agent
                    else:
                        kwargs["domain_extractor"] = _extract_domain_arg if attr_name in {"get_prediction_contract", "parse_prediction_output"} else _extract_inputs_domain
                if symbol_id == "utils/prediction_contracts.py::build_task_instruction":
                    kwargs["branch_extractor"] = _extract_build_branch
                elif symbol_id == "utils/prediction_contracts.py::parse_prediction_output":
                    kwargs["branch_extractor"] = _extract_parse_branch
                elif symbol_id == "agent/llm.py::_get_response_from_gemini_rest":
                    kwargs["branch_extractor"] = _extract_schema_branch
                _wrap_function(
                    target=module,
                    attr_name=attr_name,
                    trace_data=trace_data,
                    symbol_id=symbol_id,
                    file_path=file_path,
                    symbol=symbol,
                    **kwargs,
                )

            harness_path = harness_mod.harness(
                agent_path="./task_agent.py",
                output_dir=str(output_root / "runs" / domain),
                run_id=run_id,
                domain=domain,
                model=model,
                num_samples=num_samples,
                save_interval=save_interval,
                num_workers=num_workers,
                subset=subset,
            )
        finally:
            try:
                sys.path.remove(str(snapshot_root))
            except ValueError:
                pass

    question_id_col, _ = _domain_info(domain)
    summary = {
        "domain": domain,
        "run_id": run_id,
        "subset": subset,
        "num_samples": num_samples,
        "model": model,
        "output_dir": str(Path(harness_path).resolve()),
        "trace": trace_data,
    }
    _write_json(trace_path, summary)
    return DomainRun(
        domain=domain,
        subset=subset,
        num_samples=num_samples,
        run_id=run_id,
        snapshot_root=snapshot_root,
        output_dir=Path(harness_path).resolve(),
        predictions_path=Path(harness_path).resolve() / "predictions.csv",
        trace_path=trace_path,
        trace=summary,
        question_id_col=question_id_col,
    )


def _load_domain_examples(run: DomainRun) -> list[dict[str, Any]]:
    sys.path.insert(0, str(run.snapshot_root))
    try:
        _clear_snapshot_modules()
        contracts = importlib.import_module("utils.prediction_contracts")
        parse_prediction_output = contracts.parse_prediction_output
    finally:
        try:
            sys.path.remove(str(run.snapshot_root))
        except ValueError:
            pass

    dataset = _load_dataset(run.domain, run.subset, run.num_samples).copy()
    predictions = pd.read_csv(run.predictions_path, dtype=str)
    merged = dataset.merge(predictions[[run.question_id_col, "prediction"]], on=run.question_id_col, how="left")
    examples: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        question_id = str(row[run.question_id_col])
        chat_path = run.output_dir / "agent_evals" / f"chat_history_{question_id}.md"
        raw_text = ""
        if chat_path.exists():
            raw_text = _extract_last_output_text(chat_path.read_text(encoding="utf-8", errors="ignore")) or ""
        prediction = row.get("prediction")
        prediction_text = None if pd.isna(prediction) else str(prediction).strip().lower()
        symptoms = _classify_output_symptoms(run.domain, raw_text, prediction_text)
        pair_key = PARTIAL_KEY_VALUE_PATTERN.search(raw_text)
        parse_result = parse_prediction_output(run.domain, raw_text)
        examples.append(
            {
                "question_id": question_id,
                "prediction": prediction_text,
                "parse_label": parse_result.label,
                "parse_source": parse_result.source,
                "parse_reason": parse_result.reason,
                "raw_output_excerpt": raw_text[:500],
                "output_length": len(raw_text),
                "visible_partial_key": pair_key.group("key").lower() if pair_key else None,
                "visible_partial_value": pair_key.group("value").lower() if pair_key else None,
                "symptoms": symptoms,
            }
        )
    return examples


def _pair_shared_symbols(source_trace: dict[str, Any], target_trace: dict[str, Any]) -> set[str]:
    shared: set[str] = set()
    source_entries = source_trace["trace"]
    target_entries = target_trace["trace"]
    for symbol_id, source_entry in source_entries.items():
        target_entry = target_entries.get(symbol_id)
        if not target_entry:
            continue
        if not source_entry.get("executed") or not target_entry.get("executed"):
            continue
        source_branches = set(source_entry.get("branches") or [])
        target_branches = set(target_entry.get("branches") or [])
        if symbol_id in {
            "utils/prediction_contracts.py::build_task_instruction",
            "utils/prediction_contracts.py::parse_prediction_output",
            "agent/llm.py::_get_response_from_gemini_rest",
        } and source_branches != target_branches:
            continue
        if symbol_id in {
            "utils/prediction_contracts.py::_build_paper_review_instruction",
            "utils/prediction_contracts.py::_parse_paper_review_output",
            "utils/prediction_contracts.py::_build_imo_grading_instruction",
            "utils/prediction_contracts.py::_parse_imo_grading_output",
        }:
            continue
        shared.add(symbol_id)
    return shared


def _symptom_surface_status(symptom: str, shared_symbols: set[str]) -> str:
    surfaces = SYMPtom_SURFACES.get(symptom, ())
    if not surfaces:
        return "unmapped"
    return "shared" if any(surface in shared_symbols for surface in surfaces) else "not_shared"


def _summarize_examples(examples: list[dict[str, Any]], shared_symbols: set[str]) -> dict[str, Any]:
    symptom_counts = Counter()
    shared_symptom_counts = Counter()
    not_shared_counts = Counter()
    for item in examples:
        for symptom in item["symptoms"]:
            symptom_counts[symptom] += 1
            status = _symptom_surface_status(symptom, shared_symbols)
            if status == "shared":
                shared_symptom_counts[symptom] += 1
            elif status == "not_shared":
                not_shared_counts[symptom] += 1
    return {
        "symptom_counts": dict(symptom_counts),
        "shared_surface_symptom_counts": dict(shared_symptom_counts),
        "not_shared_surface_symptom_counts": dict(not_shared_counts),
        "structured_prediction_rate": float(
            sum(1 for item in examples if (item.get("prediction") or "").strip()) / len(examples)
        )
        if examples
        else 0.0,
        "sample_examples": examples[:5],
    }


def _score_overlap(
    *,
    source_summary: dict[str, Any],
    target_summary: dict[str, Any],
    weights: dict[str, float],
) -> dict[str, Any]:
    source_shared = Counter(source_summary["shared_surface_symptom_counts"])
    target_shared = Counter(target_summary["shared_surface_symptom_counts"])
    shared_symptoms = sorted(set(source_shared) | set(target_shared))
    contributions: list[dict[str, Any]] = []
    total = 0.0
    for symptom in shared_symptoms:
        overlap = min(source_shared.get(symptom, 0), target_shared.get(symptom, 0))
        if overlap <= 0:
            continue
        weight = float(weights.get(symptom, 1.0))
        contribution = overlap * weight
        total += contribution
        contributions.append(
            {
                "symptom": symptom,
                "source_count": int(source_shared.get(symptom, 0)),
                "target_count": int(target_shared.get(symptom, 0)),
                "overlap_count": int(overlap),
                "weight": weight,
                "contribution": contribution,
            }
        )

    penalty = 0.0
    source_only_counts = Counter(source_summary["not_shared_surface_symptom_counts"])
    for symptom, count in source_only_counts.items():
        if symptom in LOW_PRIORITY_SYMPTOMS:
            penalty += 0.1 * count
        else:
            penalty += 0.25 * count
    final_score = total - penalty
    return {
        "weighted_overlap_score": final_score,
        "raw_overlap_score": total,
        "source_only_penalty": penalty,
        "contributions": contributions,
    }


def build_failure_overlap_atlas(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    base_root = Path(config["base_root"])
    output_root = Path(config["output_root"])
    if output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)
    target_cfg = config["target"]

    target_output_dir = Path(target_cfg["reuse_output_dir"])
    target_trace = None
    target_trace_path = output_root / "traces" / f"{target_cfg['domain']}_trace.json"
    if target_trace_path.exists():
        target_trace = json.loads(target_trace_path.read_text(encoding="utf-8"))
        target_run = DomainRun(
            domain=target_cfg["domain"],
            subset=target_cfg["subset"],
            num_samples=int(target_cfg["num_samples"]),
            run_id=target_trace["run_id"],
            snapshot_root=base_root,
            output_dir=Path(target_trace["output_dir"]),
            predictions_path=Path(target_trace["output_dir"]) / "predictions.csv",
            trace_path=target_trace_path,
            trace=target_trace,
            question_id_col=_domain_info(target_cfg["domain"])[0],
        )
    else:
        if target_output_dir.exists():
            output_root.joinpath("runs", target_cfg["domain"]).mkdir(parents=True, exist_ok=True)
            copied = output_root / "runs" / target_cfg["domain"] / target_output_dir.name
            if copied.exists():
                shutil.rmtree(copied)
            shutil.copytree(target_output_dir, copied)
        target_run = _trace_domain_run(
            snapshot_root=base_root,
            domain=target_cfg["domain"],
            subset=target_cfg["subset"],
            num_samples=int(target_cfg["num_samples"]),
            model=config["eval_model"],
            num_workers=int(config["num_workers"]),
            save_interval=int(config["save_interval"]),
            output_root=output_root,
        )
        target_trace = target_run.trace

    target_examples = _load_domain_examples(target_run)

    atlas: dict[str, Any] = {
        "config_path": str(config_path),
        "base_root": str(base_root),
        "eval_model": config["eval_model"],
        "target": {
            "domain": target_run.domain,
            "subset": target_run.subset,
            "num_samples": target_run.num_samples,
            "output_dir": str(target_run.output_dir),
        },
        "candidate_sources": [],
        "excluded_domains": config.get("excluded_domains", []),
    }

    for source_cfg in config["candidate_sources"]:
        source_run = _trace_domain_run(
            snapshot_root=base_root,
            domain=source_cfg["domain"],
            subset=source_cfg["subset"],
            num_samples=int(source_cfg["num_samples"]),
            model=config["eval_model"],
            num_workers=int(config["num_workers"]),
            save_interval=int(config["save_interval"]),
            output_root=output_root,
        )
        source_examples = _load_domain_examples(source_run)
        shared_symbols = _pair_shared_symbols(source_run.trace, target_trace)
        source_summary = _summarize_examples(source_examples, shared_symbols)
        target_summary = _summarize_examples(target_examples, shared_symbols)
        allowlist_path = _build_pair_allowlist(
            config=config,
            source_domain=source_cfg["domain"],
            source_trace=source_run.trace,
            target_trace=target_trace,
            output_path=Path(config["generated_allowlist_dir"]) / f"{source_cfg['domain']}_to_{target_cfg['domain']}.yaml",
        )
        overlap = _score_overlap(
            source_summary=source_summary,
            target_summary=target_summary,
            weights=config["overlap"]["symptom_weights"],
        )
        atlas["candidate_sources"].append(
            {
                "domain": source_cfg["domain"],
                "subset": source_cfg["subset"],
                "num_samples": int(source_cfg["num_samples"]),
                "task_type": source_cfg["task_type"],
                "approximate_cost": source_cfg["approximate_cost"],
                "plausibility_note": source_cfg["plausibility_note"],
                "trace_path": str(source_run.trace_path),
                "output_dir": str(source_run.output_dir),
                "allowlist_path": str(allowlist_path),
                "shared_executed_symbols": sorted(shared_symbols),
                "shared_summary": source_summary,
                "target_summary": target_summary,
                "overlap": overlap,
            }
        )

    return atlas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    atlas = build_failure_overlap_atlas(args.config)
    config = _load_yaml(args.config)
    _write_json(Path(config["atlas_json"]), atlas)

    print(json.dumps(atlas, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
