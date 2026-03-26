# Deployment

## Local Python runtime

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
proofgrade serve --config configs/runtime/default.yaml
```

## Docker

Build:

```bash
docker build -t proofgrade:0.1.0 .
```

Run:

```bash
docker run --rm -p 8000:8000 --env-file .env proofgrade:0.1.0
```

## Required environment

- `GEMINI_API_KEY` or `GOOGLE_API_KEY`

Optional runtime overrides:

- `PROOFGRADE_MODEL`
- `PROOFGRADE_PROMPT_VARIANT`
- `PROOFGRADE_LOG_LEVEL`
- `PROOFGRADE_API_HOST`
- `PROOFGRADE_API_PORT`

## Operational expectations

- grading requests are remote-model calls, so latency depends on provider response time
- batch grading is sequential in `v0.1.0` for simplicity and auditability
- benchmark reproduction can take much longer than ordinary grading requests

## Safe usage expectations

- keep a human in the loop for consequential grading decisions
- log or store the returned prompt variant and model metadata with any downstream grade record
- do not silently swap models if you need published-result comparability
