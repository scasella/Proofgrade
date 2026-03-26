# API

The API is a thin HTTP surface over the frozen grading runtime.

## Base service

- default local address: `http://127.0.0.1:8000`
- interactive docs: `/docs`
- OpenAPI schema: `/openapi.json`

## Endpoints

### `GET /health`

Returns service health and the active default runtime settings.

Example response:

```json
{
  "status": "ok",
  "default_prompt_variant": "guideline_gate_almost_boundary_v1",
  "model_provider": "gemini",
  "model_name": "gemini-3-flash-preview",
  "version": "0.1.0",
  "git_sha": "abc123def456"
}
```

### `GET /version`

Returns package version, git SHA, default prompt variant, and default model metadata.

### `POST /grade`

Grades a single proof response.

Request body:

```json
{
  "problem": "Problem statement",
  "solution": "Official solution",
  "grading_guidelines": "(Partial) ... (Almost) ...",
  "student_answer": "Student proof",
  "prompt_variant": "guideline_gate_almost_boundary_v1",
  "model": "gemini-3-flash-preview"
}
```

Response body:

```json
{
  "label": "partial",
  "rationale": "Key construction found, final justification still missing",
  "matched_guideline": "partial",
  "confidence": null,
  "review_recommended": true,
  "prompt_variant": "guideline_gate_almost_boundary_v1",
  "model_provider": "gemini",
  "model_name": "gemini-3-flash-preview",
  "parse_source": "json:label",
  "latency_ms": 842,
  "version": "0.1.0",
  "git_sha": "abc123def456",
  "request_id": "..."
}
```

### `POST /batch-grade`

Grades a list of requests with the same schema as `/grade`.

## Error behavior

- `400` for invalid config or unsupported prompt/model selection
- `502` for provider failures or invalid model outputs

## Stability

The public API is intentionally minimal in `v0.1.0`. The grading label, prompt variant, model metadata, and version fields are part of the intended stable response contract.

