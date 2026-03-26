from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from proofgrade.exceptions import ConfigurationError, UnsupportedVariantError
from proofgrade._frozen_imo_policy import (
    DEFAULT_IMO_GRADING_VARIANT,
    get_imo_grading_variants,
)


DEFAULT_MODEL = "gemini-3-flash-preview"
ENV_PREFIX = "PROOFGRADE_"


@dataclass(frozen=True)
class RuntimeSettings:
    model: str = DEFAULT_MODEL
    prompt_variant: str = DEFAULT_IMO_GRADING_VARIANT
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    request_timeout_seconds: int = 180
    app_name: str = "proofgrade"
    supported_provider: str = "gemini"

    def validate(self) -> "RuntimeSettings":
        if self.prompt_variant not in get_imo_grading_variants():
            raise UnsupportedVariantError(
                f"Unsupported prompt variant '{self.prompt_variant}'. "
                f"Supported variants: {', '.join(get_imo_grading_variants())}."
            )
        if self.api_port <= 0:
            raise ConfigurationError("API port must be a positive integer.")
        return self


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"Config file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"Config file must contain a mapping: {path}")
    return data


def _env_key(field_name: str) -> str:
    return f"{ENV_PREFIX}{field_name.upper()}"


def _coerce(field_name: str, value: Any) -> Any:
    field_type = {item.name: item.type for item in fields(RuntimeSettings)}[field_name]
    if field_type is int:
        return int(value)
    return value


def settings_to_dict(settings: RuntimeSettings) -> dict[str, Any]:
    return asdict(settings)


def settings_to_json(settings: RuntimeSettings) -> str:
    return json.dumps(settings_to_dict(settings), indent=2, sort_keys=True)


def load_settings(config_path: str | None = None, overrides: dict[str, Any] | None = None) -> RuntimeSettings:
    values: dict[str, Any] = settings_to_dict(RuntimeSettings())

    if config_path:
        values.update(_load_yaml(Path(config_path)))

    for field in fields(RuntimeSettings):
        env_value = os.getenv(_env_key(field.name))
        if env_value not in (None, ""):
            values[field.name] = _coerce(field.name, env_value)

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                values[key] = value

    settings = RuntimeSettings(
        model=str(values["model"]),
        prompt_variant=str(values["prompt_variant"]),
        log_level=str(values["log_level"]),
        api_host=str(values["api_host"]),
        api_port=int(values["api_port"]),
        request_timeout_seconds=int(values["request_timeout_seconds"]),
        app_name=str(values["app_name"]),
        supported_provider=str(values["supported_provider"]),
    )
    return settings.validate()
