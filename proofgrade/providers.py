from __future__ import annotations

import os
import time
from dataclasses import dataclass

from proofgrade._gemini_backend import (
    GEMINI_FLASH_PREVIEW_MODEL,
    get_model_name,
    get_model_provider,
    get_response_from_llm,
    normalize_model_name,
    validate_model_credentials,
)
from proofgrade.exceptions import ConfigurationError, ProviderError


SUPPORTED_MODELS = {
    "gemini-3-flash-preview": GEMINI_FLASH_PREVIEW_MODEL,
    GEMINI_FLASH_PREVIEW_MODEL: GEMINI_FLASH_PREVIEW_MODEL,
}


@dataclass(frozen=True)
class CompletionResult:
    text: str
    provider: str
    model: str
    latency_ms: int


def resolve_model(model: str) -> str:
    normalized = SUPPORTED_MODELS.get(model, normalize_model_name(model))
    if normalized != GEMINI_FLASH_PREVIEW_MODEL:
        raise ConfigurationError(
            "v0.1.0 officially supports only gemini-3-flash-preview for the published runtime "
            "and benchmark reproduction."
        )
    return normalized


def validate_runtime_credentials(model: str) -> None:
    resolved = resolve_model(model)
    try:
        validate_model_credentials(resolved)
    except Exception as exc:  # pragma: no cover - exact provider error text is not critical
        raise ConfigurationError(
            "Missing model credentials. Set GEMINI_API_KEY or GOOGLE_API_KEY before grading."
        ) from exc


def provider_name(model: str) -> str:
    return get_model_provider(resolve_model(model))


def model_name(model: str) -> str:
    return get_model_name(resolve_model(model))


def complete(prompt: str, *, model: str) -> CompletionResult:
    resolved = resolve_model(model)
    validate_runtime_credentials(resolved)
    started = time.perf_counter()
    try:
        text, _, info = get_response_from_llm(msg=prompt, model=resolved, temperature=0.0)
    except Exception as exc:
        raise ProviderError(f"Provider request failed for model '{resolved}': {exc}") from exc
    latency_ms = int((time.perf_counter() - started) * 1000)
    provider = info.get("provider") or get_model_provider(resolved)
    return CompletionResult(text=text, provider=provider, model=resolved, latency_ms=latency_ms)


def credentials_present() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
