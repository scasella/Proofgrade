from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


PromptVariant = Literal[
    "baseline",
    "guideline_gate_v1",
    "guideline_gate_almost_boundary_v1",
    "guideline_gate_no_top_end_guard_v1",
]
GradeLabel = Literal["incorrect", "partial", "almost", "correct"]


class GradeRequest(BaseModel):
    problem: str = Field(..., min_length=1)
    solution: str = Field(..., min_length=1)
    grading_guidelines: str = Field(..., min_length=1)
    student_answer: str = Field(..., min_length=1)
    prompt_variant: PromptVariant | None = None
    model: str | None = None


class GradeResponse(BaseModel):
    label: GradeLabel
    rationale: str | None = None
    matched_guideline: str | None = None
    confidence: float | None = None
    review_recommended: bool
    prompt_variant: str
    model_provider: str
    model_name: str
    parse_source: str
    latency_ms: int
    version: str
    git_sha: str | None = None
    request_id: str


class BatchGradeRequest(BaseModel):
    items: list[GradeRequest]


class BatchGradeResponse(BaseModel):
    count: int
    items: list[GradeResponse]
    version: str
    git_sha: str | None = None


class HealthResponse(BaseModel):
    status: str
    default_prompt_variant: str
    model_provider: str
    model_name: str
    version: str
    git_sha: str | None = None


class VersionResponse(BaseModel):
    package: str
    version: str
    git_sha: str | None = None
    default_prompt_variant: str
    model_provider: str
    model_name: str
