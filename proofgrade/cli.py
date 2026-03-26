from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import uvicorn

from proofgrade.api import create_app
from proofgrade.benchmark import run_benchmark
from proofgrade.config import load_settings, settings_to_json
from proofgrade.exceptions import ProofgradeError
from proofgrade.grader import grade_submission
from proofgrade.logging import configure_logging
from proofgrade.schemas import GradeRequest
from proofgrade.version import __version__, get_git_sha


def _read_text(path: str) -> str:
    return Path(path).read_text().strip()


def _grade_request_from_args(args: argparse.Namespace) -> GradeRequest:
    return GradeRequest(
        problem=_read_text(args.problem_file),
        solution=_read_text(args.solution_file),
        grading_guidelines=_read_text(args.guidelines_file),
        student_answer=_read_text(args.answer_file),
        prompt_variant=args.prompt_variant,
        model=args.model,
    )


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _handle_grade(args: argparse.Namespace) -> int:
    settings = load_settings(args.config, {"model": args.model, "prompt_variant": args.prompt_variant})
    configure_logging(settings.log_level)
    result = grade_submission(_grade_request_from_args(args), settings)
    _print_json(result.response.model_dump())
    return 0


def _handle_batch_grade(args: argparse.Namespace) -> int:
    settings = load_settings(args.config, {"model": args.model, "prompt_variant": args.prompt_variant})
    configure_logging(settings.log_level)
    input_path = Path(args.input)
    if input_path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    else:
        loaded = json.loads(input_path.read_text())
        if not isinstance(loaded, list):
            raise ProofgradeError("Batch input must be a JSON array or JSONL file.")
        records = loaded
    responses = []
    for record in records:
        request = GradeRequest.model_validate(
            {
                "problem": record["problem"],
                "solution": record["solution"],
                "grading_guidelines": record["grading_guidelines"],
                "student_answer": record["student_answer"],
                "prompt_variant": record.get("prompt_variant") or args.prompt_variant,
                "model": record.get("model") or args.model,
            }
        )
        responses.append(grade_submission(request, settings).response.model_dump())
    payload = {
        "count": len(responses),
        "items": responses,
        "version": __version__,
        "git_sha": get_git_sha(),
    }
    _print_json(payload)
    return 0


def _handle_version(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    _print_json(
        {
            "package": "proofgrade",
            "version": __version__,
            "git_sha": get_git_sha(),
            "settings": json.loads(settings_to_json(settings)),
        }
    )
    return 0


def _handle_benchmark(args: argparse.Namespace) -> int:
    run_benchmark(args.config)
    print(f"Benchmark workflow completed for {args.config}")
    return 0


def _handle_serve(args: argparse.Namespace) -> int:
    settings = load_settings(args.config, {"model": args.model, "prompt_variant": args.prompt_variant})
    configure_logging(settings.log_level)
    app = create_app(settings)
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_level=settings.log_level.lower())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Proofgrade CLI for rubric-aware proof grading.")
    parser.add_argument("--config", default=None, help="Optional path to a runtime YAML config.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    grade_parser = subparsers.add_parser("grade", help="Grade one proof from local text files.")
    grade_parser.add_argument("--problem-file", required=True)
    grade_parser.add_argument("--solution-file", required=True)
    grade_parser.add_argument("--guidelines-file", required=True)
    grade_parser.add_argument("--answer-file", required=True)
    grade_parser.add_argument("--prompt-variant", default=None)
    grade_parser.add_argument("--model", default=None)
    grade_parser.set_defaults(handler=_handle_grade)

    batch_parser = subparsers.add_parser("batch-grade", help="Grade a JSON array or JSONL batch input.")
    batch_parser.add_argument("--input", required=True)
    batch_parser.add_argument("--prompt-variant", default=None)
    batch_parser.add_argument("--model", default=None)
    batch_parser.set_defaults(handler=_handle_batch_grade)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run one of the frozen benchmark/repro workflows.")
    benchmark_parser.add_argument("--config", required=True, help="Path to a frozen benchmark config.")
    benchmark_parser.set_defaults(handler=_handle_benchmark)

    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI grading service.")
    serve_parser.add_argument("--prompt-variant", default=None)
    serve_parser.add_argument("--model", default=None)
    serve_parser.set_defaults(handler=_handle_serve)

    version_parser = subparsers.add_parser("version", help="Print version and active defaults.")
    version_parser.set_defaults(handler=_handle_version)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except ProofgradeError as exc:
        parser.exit(status=2, message=f"proofgrade error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
