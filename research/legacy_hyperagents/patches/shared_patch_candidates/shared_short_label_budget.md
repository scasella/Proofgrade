# shared_short_label_budget

- Changed file: `agent/llm.py`
- Changed symbol: `_get_response_from_gemini_rest`
- Mechanism: clamp the output budget for label-style tasks on the shared Gemini REST path.
- Shared-path rationale: both domains use the same request function when evaluated with `gemini-3-flash-preview`.
- Source hypothesis: label tasks should finish with a short structured answer instead of a long response that gets cut off.
