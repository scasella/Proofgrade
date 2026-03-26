"""Trace which transfer-relevant codepaths execute on a frozen target run."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable


DEFAULT_MANIFEST_PATH = Path(
    "/Users/scasella/Downloads/hyperagents/HyperAgents/analysis/outputs/first_transfer_pilot/transfer_delta_manifest.json"
)


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _clear_snapshot_modules() -> None:
    prefixes = (
        "domains",
        "utils",
        "agent",
        "task_agent",
        "meta_agent",
        "run_meta_agent",
        "generate_loop",
        "select_next_parent",
    )
    for name in list(sys.modules):
        if name == "analysis.trace_transfer_codepaths":
            continue
        if name in prefixes or any(name.startswith(prefix + ".") for prefix in prefixes):
            del sys.modules[name]


def _new_entry(symbol: str, changed: bool) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "changed_symbol": changed,
        "call_count": 0,
        "domains": [],
        "return_labels": [],
        "executed": False,
    }


def _append_unique(values: list[str], item: str | None) -> None:
    if item is None:
        return
    if item not in values:
        values.append(item)


def _wrap_function(
    module: Any,
    func_name: str,
    trace_key: str,
    trace_data: dict[str, dict[str, Any]],
    changed_symbols: set[str],
    domain_extractor: Callable[[tuple[Any, ...], dict[str, Any]], str | None] | None = None,
    return_extractor: Callable[[Any], str | None] | None = None,
) -> None:
    original = getattr(module, func_name)
    trace_data.setdefault(trace_key, _new_entry(trace_key, trace_key in changed_symbols))

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        entry = trace_data[trace_key]
        entry["executed"] = True
        entry["call_count"] += 1
        if domain_extractor is not None:
            _append_unique(entry["domains"], domain_extractor(args, kwargs))
        result = original(*args, **kwargs)
        if return_extractor is not None:
            _append_unique(entry["return_labels"], return_extractor(result))
        return result

    setattr(module, func_name, wrapped)


def _extract_inputs_domain(args: tuple[Any, ...], _: dict[str, Any]) -> str | None:
    if not args:
        return None
    inputs = args[0]
    if isinstance(inputs, dict):
        return inputs.get("domain")
    return None


def _extract_domain_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    if args:
        return str(args[0])
    value = kwargs.get("domain")
    return str(value) if value is not None else None


def _extract_parse_label(result: Any) -> str | None:
    label = getattr(result, "label", None)
    return str(label) if label is not None else None


def _load_changed_symbols(manifest_path: Path) -> set[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        f"{change['file_path']}::{change['symbol']}"
        for change in manifest.get("changes", [])
    }


def _ensure_trace_entry(trace_data: dict[str, dict[str, Any]], key: str, changed_symbols: set[str]) -> None:
    trace_data.setdefault(key, _new_entry(key, key in changed_symbols))


def run_trace(
    snapshot_root: Path,
    manifest_path: Path,
    domain: str,
    run_id: str,
    subset: str,
    num_samples: int,
    num_workers: int,
    save_interval: int,
    model: str,
    output_path: Path,
) -> dict[str, Any]:
    changed_symbols = _load_changed_symbols(manifest_path)
    trace_data: dict[str, dict[str, Any]] = {}

    with _pushd(snapshot_root):
        sys.path.insert(0, str(snapshot_root))
        try:
            _clear_snapshot_modules()
            prediction_contracts = importlib.import_module("utils.prediction_contracts")
            harness_mod = importlib.import_module("domains.harness")

            symbol_specs = [
                ("utils/prediction_contracts.py::build_task_instruction", prediction_contracts, "build_task_instruction", _extract_inputs_domain, None),
                ("utils/prediction_contracts.py::parse_prediction_output", prediction_contracts, "parse_prediction_output", _extract_domain_arg, _extract_parse_label),
                ("utils/prediction_contracts.py::get_prediction_contract", prediction_contracts, "get_prediction_contract", _extract_domain_arg, None),
                ("utils/prediction_contracts.py::_build_paper_review_instruction", prediction_contracts, "_build_paper_review_instruction", _extract_inputs_domain, None),
                ("utils/prediction_contracts.py::_parse_paper_review_output", prediction_contracts, "_parse_paper_review_output", lambda *_: "paper_review", _extract_parse_label),
                ("utils/prediction_contracts.py::_build_imo_grading_instruction", prediction_contracts, "_build_imo_grading_instruction", _extract_inputs_domain, None),
                ("utils/prediction_contracts.py::_parse_imo_grading_output", prediction_contracts, "_parse_imo_grading_output", lambda *_: "imo_grading", _extract_parse_label),
            ]

            for trace_key, module, func_name, domain_extractor, return_extractor in symbol_specs:
                _wrap_function(
                    module=module,
                    func_name=func_name,
                    trace_key=trace_key,
                    trace_data=trace_data,
                    changed_symbols=changed_symbols,
                    domain_extractor=domain_extractor,
                    return_extractor=return_extractor,
                )

            for domain_name, build_name, parse_name in [
                ("paper_review", "_build_paper_review_instruction", "_parse_paper_review_output"),
                ("imo_grading", "_build_imo_grading_instruction", "_parse_imo_grading_output"),
            ]:
                contract = prediction_contracts.CONTRACTS[domain_name]
                prediction_contracts.CONTRACTS[domain_name] = dataclasses.replace(
                    contract,
                    build_instruction=getattr(prediction_contracts, build_name),
                    parse_output=getattr(prediction_contracts, parse_name),
                )

            harness_output = harness_mod.harness(
                agent_path="./task_agent.py",
                output_dir="./outputs",
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

    for key in [
        "utils/prediction_contracts.py::build_task_instruction",
        "utils/prediction_contracts.py::parse_prediction_output",
        "utils/prediction_contracts.py::get_prediction_contract",
        "utils/prediction_contracts.py::_build_paper_review_instruction",
        "utils/prediction_contracts.py::_parse_paper_review_output",
        "utils/prediction_contracts.py::_build_imo_grading_instruction",
        "utils/prediction_contracts.py::_parse_imo_grading_output",
    ]:
        _ensure_trace_entry(trace_data, key, changed_symbols)

    changed_executed = sorted(key for key, entry in trace_data.items() if entry["changed_symbol"] and entry["executed"])
    changed_not_executed = sorted(key for key, entry in trace_data.items() if entry["changed_symbol"] and not entry["executed"])

    summary = {
        "snapshot_root": str(snapshot_root),
        "run_id": run_id,
        "domain": domain,
        "subset": subset,
        "num_samples": num_samples,
        "model": model,
        "harness_output": str(Path(harness_output).resolve()),
        "changed_symbols_executed": changed_executed,
        "changed_symbols_not_executed": changed_not_executed,
        "trace": trace_data,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace transfer-relevant codepaths on a frozen target run.")
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--domain", type=str, default="imo_grading")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--subset", type=str, default="_filtered_100_val")
    parser.add_argument("--num-samples", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = run_trace(
        snapshot_root=args.snapshot_root,
        manifest_path=args.manifest,
        domain=args.domain,
        run_id=args.run_id,
        subset=args.subset,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        save_interval=args.save_interval,
        model=args.model,
        output_path=args.output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
