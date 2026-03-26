from __future__ import annotations

import json
import os
from typing import Tuple

import backoff
import requests

GEMINI_FLASH_PREVIEW_MODEL = "gemini/gemini-3-flash-preview"
GEMINI_PREVIEW_MAX_OUTPUT_TOKENS = 2048

MODEL_ALIASES = {
    "gemini-3-flash-preview": GEMINI_FLASH_PREVIEW_MODEL,
}


def normalize_model_name(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def get_model_provider(model: str) -> str:
    normalized = normalize_model_name(model)
    if "/" in normalized:
        return normalized.split("/", 1)[0]
    return "unknown"


def get_model_name(model: str) -> str:
    normalized = normalize_model_name(model)
    if "/" in normalized:
        return normalized.split("/", 1)[1]
    return normalized


def validate_model_credentials(model: str) -> None:
    normalized = normalize_model_name(model)
    if _is_rest_gemini_model(normalized):
        _get_gemini_api_key()


def _is_rest_gemini_model(model: str) -> bool:
    return model == GEMINI_FLASH_PREVIEW_MODEL


def _get_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set for Gemini preview models.")
    return api_key


def _convert_messages_for_gemini(msg: str, msg_history: list[dict]) -> list[dict]:
    contents = []
    normalized_history = [
        {**entry, "content": entry.get("content", entry.get("text", ""))}
        for entry in msg_history
    ]
    normalized_history.append({"role": "user", "content": msg})
    for entry in normalized_history:
        text = entry.get("content") or ""
        if not text:
            continue
        role = entry.get("role", "user")
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": str(text)}]})
    return contents


def _infer_gemini_response_schema(msg: str) -> dict | None:
    if '"domain": "imo_grading"' in msg:
        return {
            "type": "OBJECT",
            "properties": {
                "label": {
                    "type": "STRING",
                    "enum": ["incorrect", "partial", "almost", "correct"],
                },
                "rationale": {"type": "STRING"},
                "decision_basis": {"type": "STRING"},
                "missing_piece": {"type": "STRING"},
                "matched_guideline": {"type": "STRING"},
            },
            "required": ["label"],
        }

    if "<json>" in msg or "Return exactly one JSON object" in msg:
        return {
            "type": "OBJECT",
            "properties": {
                "label": {"type": "STRING"},
                "response": {"type": "STRING"},
                "rationale": {"type": "STRING"},
                "confidence": {"type": "NUMBER"},
            },
        }

    return None


def _get_response_from_gemini_rest(
    msg: str,
    model: str,
    temperature: float,
    max_tokens: int,
    msg_history: list[dict],
) -> Tuple[str, list, dict]:
    api_key = _get_gemini_api_key()
    model_name = model.split("/", 1)[1] if "/" in model else model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    generation_config = {
        "temperature": temperature,
        "maxOutputTokens": min(max_tokens, GEMINI_PREVIEW_MAX_OUTPUT_TOKENS),
    }
    response_schema = _infer_gemini_response_schema(msg)
    if response_schema is not None:
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = response_schema

    payload = {
        "contents": _convert_messages_for_gemini(msg=msg, msg_history=msg_history),
        "generationConfig": generation_config,
    }
    response = requests.post(url, params={"key": api_key}, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise ValueError(f"Gemini preview response contained no candidates: {data}")
    parts = candidates[0].get("content", {}).get("parts", [])
    response_text = "".join(part.get("text", "") for part in parts if "text" in part)
    if not response_text:
        raise ValueError(f"Gemini preview response contained no text parts: {data}")

    new_msg_history = [
        {**entry, "content": entry.get("content", entry.get("text", ""))}
        for entry in msg_history
    ]
    new_msg_history.append({"role": "user", "content": msg})
    new_msg_history.append({"role": "assistant", "content": response_text})
    new_msg_history = [
        {**entry, "text": entry.pop("content")} if "content" in entry else entry
        for entry in new_msg_history
    ]
    return response_text, new_msg_history, {"provider": "gemini_rest"}


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, json.JSONDecodeError, KeyError),
    max_time=600,
    max_value=60,
)
def get_response_from_llm(
    msg: str,
    model: str = GEMINI_FLASH_PREVIEW_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    msg_history=None,
) -> Tuple[str, list, dict]:
    if msg_history is None:
        msg_history = []
    model = normalize_model_name(model)

    if _is_rest_gemini_model(model):
        return _get_response_from_gemini_rest(
            msg=msg,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            msg_history=msg_history,
        )

    raise ValueError(f"Unsupported model for proofgrade v0.1.0: {model}")
