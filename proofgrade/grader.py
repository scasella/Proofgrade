from __future__ import annotations

import uuid
from dataclasses import dataclass

from proofgrade.config import RuntimeSettings
from proofgrade.exceptions import ProviderError
from proofgrade.policy import build_instruction, extract_grade_metadata, parse_grade_output
from proofgrade.providers import complete, model_name, provider_name
from proofgrade.schemas import GradeRequest, GradeResponse
from proofgrade.version import __version__, get_git_sha


@dataclass(frozen=True)
class GradeResult:
    response: GradeResponse
    raw_text: str


def _review_recommended(label: str, parse_source: str) -> bool:
    return label in {"partial", "almost"} or parse_source != "json:label"


def grade_submission(request: GradeRequest, settings: RuntimeSettings) -> GradeResult:
    prompt_variant = request.prompt_variant or settings.prompt_variant
    model = request.model or settings.model
    prompt = build_instruction(
        problem=request.problem,
        solution=request.solution,
        grading_guidelines=request.grading_guidelines,
        student_answer=request.student_answer,
        prompt_variant=prompt_variant,
    )
    completion = complete(prompt, model=model)
    parsed = parse_grade_output(completion.text)
    if parsed.label is None:
        raise ProviderError(
            "The model returned an invalid grading label. "
            "Try again or review the raw provider output in the server logs."
        )
    metadata = extract_grade_metadata(completion.text)
    response = GradeResponse(
        label=parsed.label,
        rationale=metadata.get("rationale"),
        matched_guideline=metadata.get("matched_guideline"),
        confidence=metadata.get("confidence"),
        review_recommended=_review_recommended(parsed.label, parsed.source),
        prompt_variant=prompt_variant,
        model_provider=provider_name(model),
        model_name=model_name(model),
        parse_source=parsed.source,
        latency_ms=completion.latency_ms,
        version=__version__,
        git_sha=get_git_sha(),
        request_id=str(uuid.uuid4()),
    )
    return GradeResult(response=response, raw_text=completion.text)

