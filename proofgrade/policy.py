from __future__ import annotations

from typing import Any

from proofgrade._frozen_imo_policy import (
    build_imo_grading_instruction_for_variant,
    extract_last_prediction_json,
    get_default_imo_grading_variant,
    get_imo_grading_variants,
    parse_imo_grading_output,
)


IMO_ALLOWED_LABELS = ("incorrect", "partial", "almost", "correct")


def supported_prompt_variants() -> tuple[str, ...]:
    return get_imo_grading_variants()


def default_prompt_variant() -> str:
    return get_default_imo_grading_variant()


def build_instruction(
    *,
    problem: str,
    solution: str,
    grading_guidelines: str,
    student_answer: str,
    prompt_variant: str,
) -> str:
    return build_imo_grading_instruction_for_variant(
        {
            "domain": "imo_grading",
            "problem": problem,
            "solution": solution,
            "grading_guidelines": grading_guidelines,
            "student_answer": student_answer,
        },
        prompt_variant,
    )


def parse_grade_output(raw_text: str):
    return parse_imo_grading_output(raw_text)


def extract_grade_metadata(raw_text: str) -> dict[str, Any]:
    extracted = extract_last_prediction_json(raw_text)
    return extracted if isinstance(extracted, dict) else {}
