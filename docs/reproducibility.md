# Reproducibility

## Frozen release line

The public release freezes three grading-policy variants:

- `baseline`
- `guideline_gate_v1`
- `guideline_gate_almost_boundary_v1`

The shipped default is `guideline_gate_almost_boundary_v1`.

## Core frozen configs

- `configs/baseline_freeze/final_imo_lock.yaml`
- `configs/baseline_freeze/final_imo_lockbox_test.yaml`
- `configs/baseline_freeze/final_imo_release.yaml`
- `configs/baseline_freeze/fresh_generalization_eval.yaml`

## Rebuild the public result package from saved artifacts

```bash
PYTHONPATH=. .venv/bin/python analysis/build_imo_result_tables.py --config configs/baseline_freeze/final_imo_release.yaml
PYTHONPATH=. .venv/bin/python analysis/build_imo_casebook.py --config configs/baseline_freeze/final_imo_release.yaml
```

## Re-run the validation ablation

```bash
PYTHONPATH=. .venv/bin/python analysis/run_final_imo_ablation.py --config configs/baseline_freeze/final_imo_lock.yaml
PYTHONPATH=. .venv/bin/python analysis/build_final_imo_remaining_error_atlas.py --config configs/baseline_freeze/final_imo_lock.yaml
```

## Re-run the one-time lockbox test

```bash
GEMINI_API_KEY=... GOOGLE_API_KEY=... \
PYTHONPATH=. .venv/bin/python analysis/run_final_imo_lockbox_test.py \
  --config configs/baseline_freeze/final_imo_lockbox_test.yaml
```

## Re-run the fresh 512-response evaluation

```bash
GEMINI_API_KEY=... GOOGLE_API_KEY=... \
PYTHONPATH=. .venv/bin/python analysis/run_fresh_generalization_eval.py \
  --config configs/baseline_freeze/fresh_generalization_eval.yaml
```

## What counts as the published result

The official release claims are based on:

- the frozen prompt variants named above
- the released model family (`gemini-3-flash-preview`)
- the frozen grading policy and parser logic vendored into the `proofgrade` runtime package, which mirrors the locked benchmark line
- the curated release artifacts in `artifacts/release/v0.1.0/`

## Caveats

- The fresh 512-example evaluation is fresh-response generalization, not fresh-problem-family generalization.
- This repository does not make a validated cross-domain transfer claim.
- The product release is intentionally narrower than the original HyperAgents research framing.
