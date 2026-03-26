import backoff
import os
from typing import Tuple
import requests
import litellm
from dotenv import load_dotenv
import json

load_dotenv()

MAX_TOKENS = 16384

CLAUDE_MODEL = "anthropic/claude-sonnet-4-5-20250929"
CLAUDE_HAIKU_MODEL = "anthropic/claude-3-haiku-20240307"
CLAUDE_35NEW_MODEL = "anthropic/claude-3-5-sonnet-20241022"
OPENAI_MODEL = "openai/gpt-4o"
OPENAI_MINI_MODEL = "openai/gpt-4o-mini"
OPENAI_O3_MODEL = "openai/o3"
OPENAI_O3MINI_MODEL = "openai/o3-mini"
OPENAI_O4MINI_MODEL = "openai/o4-mini"
OPENAI_GPT52_MODEL = "openai/gpt-5.2"
OPENAI_GPT5_MODEL = "openai/gpt-5"
OPENAI_GPT5MINI_MODEL = "openai/gpt-5-mini"
GEMINI_3_MODEL = "gemini/gemini-3-pro-preview"
GEMINI_MODEL = "gemini/gemini-2.5-pro"
GEMINI_FLASH_MODEL = "gemini/gemini-2.5-flash"
GEMINI_FLASH_PREVIEW_MODEL = "gemini/gemini-3-flash-preview"
GEMINI_PREVIEW_MAX_OUTPUT_TOKENS = 2048

litellm.drop_params=True

MODEL_ALIASES = {
    # Public repo configs still reference a few internal-style model names.
    "claude-4-sonnet-genai": CLAUDE_MODEL,
    "gpt-4o-mini-genai": OPENAI_MINI_MODEL,
    "gpt-o4-mini-genai": OPENAI_O4MINI_MODEL,
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

    if '"domain": "paper_review"' in msg:
        return {
            "type": "OBJECT",
            "properties": {
                "label": {
                    "type": "STRING",
                    "enum": ["accept", "reject"],
                },
                "overall_score": {"type": "INTEGER"},
                "confidence": {"type": "INTEGER"},
                "strengths": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                },
                "weaknesses": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                },
                "rationale": {"type": "STRING"},
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
        # The preview model can spend a long time thinking unless we cap
        # the output budget. These task agents only need a short JSON label.
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
    model: str = OPENAI_MODEL,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
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

    # Convert text to content, compatible with LITELLM API
    msg_history = [
        {**msg, "content": msg.pop("text")} if "text" in msg else msg
        for msg in msg_history
    ]

    new_msg_history = msg_history + [{"role": "user", "content": msg}]

    # Build kwargs - handle model-specific requirements
    completion_kwargs = {
        "model": model,
        "messages": new_msg_history,
    }

    # GPT-5 and GPT-5-mini only support default temperature (1), skip it
    # GPT-5.2 supports temperature
    if model in ["openai/gpt-5", "openai/gpt-5-mini"]:
        pass  # Don't set temperature
    else:
        completion_kwargs["temperature"] = temperature

    # GPT-5 models require max_completion_tokens instead of max_tokens
    if "gpt-5" in model:
        completion_kwargs["max_completion_tokens"] = max_tokens
    else:
        # Claude Haiku has a 4096 token limit
        if "claude-3-haiku" in model:
            completion_kwargs["max_tokens"] = min(max_tokens, 4096)
        else:
            completion_kwargs["max_tokens"] = max_tokens

    response = litellm.completion(**completion_kwargs)
    response_text = response['choices'][0]['message']['content']  # pyright: ignore
    new_msg_history.append({"role": "assistant", "content": response['choices'][0]['message']['content']})

    # Convert content to text, compatible with MetaGen API
    new_msg_history = [
        {**msg, "text": msg.pop("content")} if "content" in msg else msg
        for msg in new_msg_history
    ]

    return response_text, new_msg_history, {}


if __name__ == "__main__":
    msg = 'Hello there!'
    models = [
        ("CLAUDE_MODEL", CLAUDE_MODEL),
        ("CLAUDE_HAIKU_MODEL", CLAUDE_HAIKU_MODEL),
        ("CLAUDE_35NEW_MODEL", CLAUDE_35NEW_MODEL),
        ("OPENAI_MODEL", OPENAI_MODEL),
        ("OPENAI_MINI_MODEL", OPENAI_MINI_MODEL),
        ("OPENAI_O3_MODEL", OPENAI_O3_MODEL),
        ("OPENAI_O3MINI_MODEL", OPENAI_O3MINI_MODEL),
        ("OPENAI_O4MINI_MODEL", OPENAI_O4MINI_MODEL),
        ("OPENAI_GPT52_MODEL", OPENAI_GPT52_MODEL),
        ("OPENAI_GPT5_MODEL", OPENAI_GPT5_MODEL),
        ("OPENAI_GPT5MINI_MODEL", OPENAI_GPT5MINI_MODEL),
        ("GEMINI_3_MODEL", GEMINI_3_MODEL),
        ("GEMINI_MODEL", GEMINI_MODEL),
        ("GEMINI_FLASH_MODEL", GEMINI_FLASH_MODEL),
    ]
    for name, model in models:
        print(f"\n{'='*50}")
        print(f"Testing {name}: {model}")
        print('='*50)
        try:
            output_msg, msg_history, info = get_response_from_llm(msg, model=model)
            print(f"OK: {output_msg[:100]}...")
        except Exception as e:
            print(f"FAIL: {str(e)[:200]}")
