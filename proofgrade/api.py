from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from proofgrade.config import RuntimeSettings, load_settings
from proofgrade.exceptions import ConfigurationError, ProofgradeError
from proofgrade.grader import grade_submission
from proofgrade.providers import model_name, provider_name, validate_runtime_credentials
from proofgrade.schemas import (
    BatchGradeRequest,
    BatchGradeResponse,
    GradeRequest,
    GradeResponse,
    HealthResponse,
    VersionResponse,
)
from proofgrade.version import __version__, get_git_sha


def create_app(settings: RuntimeSettings | None = None, *, validate_credentials_on_startup: bool = True) -> FastAPI:
    runtime = settings or load_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if validate_credentials_on_startup:
            validate_runtime_credentials(runtime.model)
        yield

    app = FastAPI(
        title="proofgrade",
        version=__version__,
        summary="Frozen rubric-aware proof grading service.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            default_prompt_variant=runtime.prompt_variant,
            model_provider=provider_name(runtime.model),
            model_name=model_name(runtime.model),
            version=__version__,
            git_sha=get_git_sha(),
        )

    @app.get("/version", response_model=VersionResponse)
    def version() -> VersionResponse:
        return VersionResponse(
            package="proofgrade",
            version=__version__,
            git_sha=get_git_sha(),
            default_prompt_variant=runtime.prompt_variant,
            model_provider=provider_name(runtime.model),
            model_name=model_name(runtime.model),
        )

    @app.post("/grade", response_model=GradeResponse)
    def grade(request: GradeRequest) -> GradeResponse:
        try:
            result = grade_submission(request, runtime)
        except ConfigurationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ProofgradeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return result.response

    @app.post("/batch-grade", response_model=BatchGradeResponse)
    def batch_grade(request: BatchGradeRequest) -> BatchGradeResponse:
        items = []
        for item in request.items:
            try:
                result = grade_submission(item, runtime)
            except ConfigurationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except ProofgradeError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            items.append(result.response)
        return BatchGradeResponse(count=len(items), items=items, version=__version__, git_sha=get_git_sha())

    return app


app = create_app()
