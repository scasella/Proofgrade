from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

JSON_LABEL_KEYS = ("label", "decision", "prediction", "response")
DEFAULT_IMO_GRADING_VARIANT = "guideline_gate_almost_boundary_v1"


@dataclass(frozen=True)
class ParseResult:
    label: str | None
    source: str
    raw_value: str | None = None
    reason: str | None = None


def _extract_jsons(response: str) -> list[Any]:
    patterns = [
        r"<json>(.*?)</json>",
        r"```json(.*?)```",
    ]
    extracted_jsons = []
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            try:
                extracted_jsons.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                continue
    return extracted_jsons


def extract_prediction_json_objects(raw_text: str) -> list[dict[str, Any]]:
    return _extract_json_objects(raw_text)


def extract_last_prediction_json(raw_text: str) -> dict[str, Any] | None:
    objects = extract_prediction_json_objects(raw_text)
    return objects[-1] if objects else None


def _format_inputs(inputs: dict[str, Any]) -> str:
    return json.dumps(inputs, indent=2, ensure_ascii=True, sort_keys=True)


def _extract_json_objects(raw_text: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    seen: set[str] = set()

    for obj in _extract_jsons(raw_text):
        if isinstance(obj, dict):
            key = json.dumps(obj, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                objects.append(obj)

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw_text):
        try:
            candidate, _ = decoder.raw_decode(raw_text[match.start():])
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, dict):
            continue
        key = json.dumps(candidate, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            objects.append(candidate)

    return objects


def _extract_json_label_candidate(raw_text: str) -> tuple[str | int | float | None, str]:
    objects = _extract_json_objects(raw_text)
    for obj in reversed(objects):
        for key in JSON_LABEL_KEYS:
            if key in obj:
                return obj[key], f"json:{key}"
    return None, "none"


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def _canonical_label_text(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"[_-]+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    return _normalize_whitespace(lowered)


def _normalize_numeric_label(value: str | int | float | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = int(value)
    else:
        text = str(value).strip().lower()
        match = re.fullmatch(r"(0|1|6|7)", text)
        if match is None:
            match = re.fullmatch(r"(0|1|6|7)\s*(?:/|out of)\s*7", text)
        if not match:
            return None
        number = int(match.group(1))
    return {
        0: "incorrect",
        1: "partial",
        6: "almost",
        7: "correct",
    }.get(number)


def _normalize_imo_label_value(value: Any) -> str | None:
    numeric = _normalize_numeric_label(value)
    if numeric is not None:
        return numeric

    if value is None:
        return None
    text = _canonical_label_text(str(value))
    exact_map = {
        "incorrect": "incorrect",
        "partial": "partial",
        "partial progress": "partial",
        "almost": "almost",
        "almost correct": "almost",
        "correct": "correct",
    }
    if text in exact_map:
        return exact_map[text]

    label_patterns = {
        "incorrect": [
            r"\blabel\s*[:\-]?\s*incorrect\b",
            r"\bfinal label\s*[:\-]?\s*incorrect\b",
            r"\bclassification\s*[:\-]?\s*incorrect\b",
            r"\bgrade\s*[:\-]?\s*incorrect\b",
            r"\b(?:the\s+)?final answer is\s*incorrect\b",
            r"\b(?:the\s+)?final answer is(?:\s+boxed)?(?:\s+text)?\s*incorrect\b",
            r"\bboxed\s*incorrect\b",
            r"\bboxed(?:\s+text)?\s*incorrect\b",
            r"\bverdict\s*[:\-]?\s*incorrect\b",
        ],
        "partial": [
            r"\blabel\s*[:\-]?\s*partial\b",
            r"\bfinal label\s*[:\-]?\s*partial\b",
            r"\bclassification\s*[:\-]?\s*partial\b",
            r"\bgrade\s*[:\-]?\s*partial\b",
            r"\bpartial progress\b",
            r"\b(?:the\s+)?final answer is\s*partial\b",
            r"\b(?:the\s+)?final answer is(?:\s+boxed)?(?:\s+text)?\s*partial\b",
            r"\bboxed\s*partial\b",
            r"\bboxed(?:\s+text)?\s*partial\b",
            r"\bverdict\s*[:\-]?\s*partial\b",
        ],
        "almost": [
            r"\blabel\s*[:\-]?\s*almost\b",
            r"\bfinal label\s*[:\-]?\s*almost\b",
            r"\bclassification\s*[:\-]?\s*almost\b",
            r"\bgrade\s*[:\-]?\s*almost\b",
            r"\balmost correct\b",
            r"\b(?:the\s+)?final answer is\s*almost\b",
            r"\b(?:the\s+)?final answer is(?:\s+boxed)?(?:\s+text)?\s*almost\b",
            r"\bboxed\s*almost\b",
            r"\bboxed(?:\s+text)?\s*almost\b",
            r"\bverdict\s*[:\-]?\s*almost\b",
        ],
        "correct": [
            r"\blabel\s*[:\-]?\s*correct\b",
            r"\bfinal label\s*[:\-]?\s*correct\b",
            r"\bclassification\s*[:\-]?\s*correct\b",
            r"\bgrade\s*[:\-]?\s*correct\b",
            r"\b(?:the\s+)?final answer is\s*correct\b",
            r"\b(?:the\s+)?final answer is(?:\s+boxed)?(?:\s+text)?\s*correct\b",
            r"\bboxed\s*correct\b",
            r"\bboxed(?:\s+text)?\s*correct\b",
            r"\bverdict\s*[:\-]?\s*correct\b",
        ],
    }
    matched = [label for label, patterns in label_patterns.items() if any(re.search(pattern, text) for pattern in patterns)]
    if len(matched) == 1:
        return matched[0]
    return None


def _extract_imo_points_label(raw_text: str) -> str | None:
    points_match = re.search(r"<points>\s*(.*?)\s*</points>", raw_text, re.IGNORECASE | re.DOTALL)
    if points_match:
        return _normalize_numeric_label(points_match.group(1))

    score_patterns = [
        r"\bfinal score\s*[:\-]?\s*(0|1|6|7)\s*(?:/|out of)\s*7\b",
        r"\bscore\s*[:\-]?\s*(0|1|6|7)\s*(?:/|out of)\s*7\b",
        r"\baward(?:ed)?\s*(0|1|6|7)\s*points?\b",
        r"\baward(?:ed)?\s*(0|1|6|7)\s*point\b",
    ]
    for pattern in score_patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            return _normalize_numeric_label(match.group(1))
    return None


def _parse_imo_grading_output(raw_text: str) -> ParseResult:
    candidate, source = _extract_json_label_candidate(raw_text)
    label = _normalize_imo_label_value(candidate)
    if label is not None:
        return ParseResult(label=label, source=source, raw_value=str(candidate))

    points_label = _extract_imo_points_label(raw_text)
    if points_label is not None:
        return ParseResult(label=points_label, source="points", raw_value=raw_text)

    free_text_label = _normalize_imo_label_value(raw_text)
    if free_text_label is not None:
        return ParseResult(label=free_text_label, source="free_text", raw_value=raw_text)

    return ParseResult(label=None, source="invalid", raw_value=str(candidate) if candidate is not None else None, reason="no_unambiguous_imo_label")


def _build_imo_grading_instruction_baseline(inputs: dict[str, Any]) -> str:
    schema = """<json>
{
  "label": "incorrect | partial | almost | correct",
  "rationale": "12 words or fewer"
}
</json>"""

    rules = [
        "You are grading a student's IMO-style solution into exactly one label.",
        "Allowed labels: \"incorrect\", \"partial\", \"almost\", \"correct\".",
        "Think silently. Output only one JSON object inside <json> tags.",
        "Do not output bullets, analysis, markdown, or any text outside the JSON object.",
        "The first field must be `label`, and it must be exactly one allowed label in lowercase.",
        "Keep `rationale` to 12 words or fewer.",
        "Do not output numeric points in the final answer.",
        "Use `almost` only for solutions that are almost correct.",
        "Use `partial` only for genuine partial progress supported by the grading guidelines.",
        "Stop immediately after </json>.",
    ]

    return (
        "\n".join(rules)
        + "\n\nTask input:\n```json\n"
        + _format_inputs(inputs)
        + "\n```\n\nRespond with:\n"
        + schema
    )


def _build_imo_grading_instruction_guideline_gate(inputs: dict[str, Any]) -> str:
    schema = """<json>
{
  "label": "incorrect | partial | almost | correct",
  "matched_guideline": "none | partial | almost | complete",
  "rationale": "12 words or fewer"
}
</json>"""

    rules = [
        "You are grading a student's IMO-style solution into exactly one label.",
        "Use the official solution and grading guidelines as the source of truth.",
        "Allowed labels: \"incorrect\", \"partial\", \"almost\", \"correct\".",
        "Think silently. Output only one JSON object inside <json> tags.",
        "Apply this hidden decision gate before labeling:",
        "1. If the student does not clearly achieve any substantial progress from the grading guidelines or an equivalent key step, choose `incorrect`.",
        "2. If the work shows genuine but incomplete progress, choose `partial` unless it is almost complete.",
        "3. Choose `almost` only when the proof follows the right strategy and only one concrete gap remains.",
        "4. Choose `correct` only when nothing important is missing.",
        "A different proof is acceptable, but only if it is complete and justified.",
        "If the answer uses examples, hand-waving, or says a proof is omitted, it cannot be `correct`.",
        "The first field must be `label`, and it must be exactly one allowed label in lowercase.",
        "Set `matched_guideline` to the highest level actually earned: `none`, `partial`, `almost`, or `complete`.",
        "Keep `rationale` to 12 words or fewer.",
        "If uncertain, choose the lower label.",
        "Stop immediately after </json>.",
    ]

    return (
        "\n".join(rules)
        + "\n\nTask input:\n```json\n"
        + _format_inputs(inputs)
        + "\n```\n\nRespond with:\n"
        + schema
    )


def _build_imo_grading_instruction_guideline_gate_almost_boundary(inputs: dict[str, Any]) -> str:
    schema = """<json>
{
  "label": "incorrect | partial | almost | correct",
  "matched_guideline": "none | partial | almost | complete",
  "rationale": "12 words or fewer"
}
</json>"""

    rules = [
        "You are grading a student's IMO-style solution into exactly one label.",
        "Use the official solution and grading guidelines as the source of truth.",
        "Allowed labels: \"incorrect\", \"partial\", \"almost\", \"correct\".",
        "Think silently. Output only one JSON object inside <json> tags.",
        "Apply this hidden decision gate before labeling:",
        "1. If the student does not clearly achieve any substantial progress from the grading guidelines or an equivalent key step, choose `incorrect`.",
        "2. If the work shows genuine but incomplete progress, choose `partial` unless it is truly near-complete.",
        "3. Choose `almost` only when the proof uses the right core strategy, covers all major cases, and would become complete after one concrete repair.",
        "4. If more than one serious gap remains, or a required case is still missing, do not use `almost`.",
        "5. Choose `correct` only when nothing important is missing.",
        "A different proof is acceptable, but only if it is complete and justified.",
        "If the answer uses examples, hand-waving, or says a proof is omitted, it cannot be `correct`.",
        "The first field must be `label`, and it must be exactly one allowed label in lowercase.",
        "Set `matched_guideline` to the highest level actually earned: `none`, `partial`, `almost`, or `complete`.",
        "If uncertain, choose the lower label.",
        "Keep `rationale` to 12 words or fewer.",
        "Stop immediately after </json>.",
    ]

    return (
        "\n".join(rules)
        + "\n\nTask input:\n```json\n"
        + _format_inputs(inputs)
        + "\n```\n\nRespond with:\n"
        + schema
    )


def _build_imo_grading_instruction_guideline_gate_no_top_end_guard(inputs: dict[str, Any]) -> str:
    schema = """<json>
{
  "label": "incorrect | partial | almost | correct",
  "matched_guideline": "none | partial | almost | complete",
  "rationale": "12 words or fewer"
}
</json>"""

    rules = [
        "You are grading a student's IMO-style solution into exactly one label.",
        "Use the official solution and grading guidelines as the source of truth.",
        "Allowed labels: \"incorrect\", \"partial\", \"almost\", \"correct\".",
        "Think silently. Output only one JSON object inside <json> tags.",
        "Apply this hidden decision gate before labeling:",
        "1. If the student does not clearly achieve any substantial progress from the grading guidelines or an equivalent key step, choose `incorrect`.",
        "2. If the work shows genuine but incomplete progress, choose `partial` unless it is truly near-complete.",
        "3. Choose `almost` only when the proof uses the right core strategy, covers all major cases, and would become complete after one concrete repair.",
        "4. If more than one serious gap remains, or a required case is still missing, do not use `almost`.",
        "A different proof is acceptable, but only if it is complete and justified.",
        "The first field must be `label`, and it must be exactly one allowed label in lowercase.",
        "Set `matched_guideline` to the highest level actually earned: `none`, `partial`, `almost`, or `complete`.",
        "If uncertain, choose the lower label.",
        "Keep `rationale` to 12 words or fewer.",
        "Stop immediately after </json>.",
    ]

    return (
        "\n".join(rules)
        + "\n\nTask input:\n```json\n"
        + _format_inputs(inputs)
        + "\n```\n\nRespond with:\n"
        + schema
    )


IMO_GRADING_VARIANT_BUILDERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "baseline": _build_imo_grading_instruction_baseline,
    "default": _build_imo_grading_instruction_guideline_gate,
    "guideline_gate_v1": _build_imo_grading_instruction_guideline_gate,
    "guideline_gate_almost_boundary_v1": _build_imo_grading_instruction_guideline_gate_almost_boundary,
    "guideline_gate_no_top_end_guard_v1": _build_imo_grading_instruction_guideline_gate_no_top_end_guard,
}


def get_default_imo_grading_variant() -> str:
    return DEFAULT_IMO_GRADING_VARIANT


def get_imo_grading_variants() -> tuple[str, ...]:
    return tuple(IMO_GRADING_VARIANT_BUILDERS.keys())


def build_imo_grading_instruction_for_variant(inputs: dict[str, Any], variant: str | None = None) -> str:
    normalized = (variant or DEFAULT_IMO_GRADING_VARIANT).strip().lower()
    builder = IMO_GRADING_VARIANT_BUILDERS.get(normalized, _build_imo_grading_instruction_guideline_gate)
    return builder(inputs)


def parse_imo_grading_output(raw_text: str) -> ParseResult:
    return _parse_imo_grading_output(raw_text)
