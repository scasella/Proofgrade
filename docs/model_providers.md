# Model Providers

## Officially supported in v0.1.0

Published benchmark and release claims are tied to:

- provider family: Gemini
- runtime model: `gemini-3-flash-preview`

## Required credentials

Set one of:

- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`

If neither is set, the CLI and API fail early with a clear configuration error.

## Reproducing the published result

Use the frozen configs under `configs/baseline_freeze/` with the model above.

Key commands:

```bash
PYTHONPATH=. .venv/bin/python analysis/build_imo_result_tables.py --config configs/baseline_freeze/final_imo_release.yaml
PYTHONPATH=. .venv/bin/python analysis/build_imo_casebook.py --config configs/baseline_freeze/final_imo_release.yaml
GEMINI_API_KEY=... GOOGLE_API_KEY=... PYTHONPATH=. .venv/bin/python analysis/run_final_imo_lockbox_test.py --config configs/baseline_freeze/final_imo_lockbox_test.yaml
```

## What may differ across models

- strictness near `correct` vs `almost`
- sensitivity to small rubric ambiguities
- response-format stability
- latency and provider-side retries

## Release policy

`v0.1.0` does not promise benchmark-equivalent behavior on arbitrary model swaps. If a stronger model is evaluated later, it should be tested against the frozen winner prompt on a new untouched pack, not by retuning the prompt on the old benchmark line.
