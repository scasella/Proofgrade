# Final IMO Result Package

## Locked release

- Model/provider: `gemini-3-flash-preview`
- Parser version: `frozen proofgrade parser and grading policy package`
- Baseline env switch: `HYPERAGENTS_IMO_GRADING_PROMPT_VARIANT=baseline`
- Final winner env switch: `HYPERAGENTS_IMO_GRADING_PROMPT_VARIANT=guideline_gate_almost_boundary_v1`
- Validation ablation summary: `artifacts/release/v0.1.0/ablation_summary.json`
- Remaining-error summary: `artifacts/release/v0.1.0/final_remaining_error_summary.json`
- Lockbox test summary: `artifacts/release/v0.1.0/lockbox_summary.json`

## Reproduction commands

- `PYTHONPATH=. .venv/bin/python analysis/run_final_imo_ablation.py --config configs/baseline_freeze/final_imo_lock.yaml`
- `PYTHONPATH=. .venv/bin/python analysis/build_final_imo_remaining_error_atlas.py --config configs/baseline_freeze/final_imo_lock.yaml`
- `GEMINI_API_KEY=... GOOGLE_API_KEY=... PYTHONPATH=. .venv/bin/python analysis/run_final_imo_lockbox_test.py --config configs/baseline_freeze/final_imo_lockbox_test.yaml`
- `GEMINI_API_KEY=... GOOGLE_API_KEY=... PYTHONPATH=. .venv/bin/python analysis/run_fresh_generalization_eval.py --config configs/baseline_freeze/fresh_generalization_eval.yaml`
- `PYTHONPATH=. .venv/bin/python analysis/build_imo_result_tables.py --config configs/baseline_freeze/final_imo_release.yaml`
- `PYTHONPATH=. .venv/bin/python analysis/build_imo_casebook.py --config configs/baseline_freeze/final_imo_release.yaml`

## Main result table

| Split | Variant | Accuracy | Normalized grading error | Valid-label rate |
| --- | --- | --- | --- | --- |
| Held-out validation (100) | baseline | 0.590 | 0.251 | 0.990 |
| Held-out validation (100) | guideline_gate_almost_boundary_v1 | 0.700 | 0.141 | 1.000 |
| Untouched test (100) | baseline | 0.640 | 0.219 | 0.990 |
| Untouched test (100) | guideline_gate_almost_boundary_v1 | 0.770 | 0.133 | 1.000 |
| Fresh filtered remainder (512) | baseline | 0.627 | 0.208 | 0.986 |
| Fresh filtered remainder (512) | guideline_gate_almost_boundary_v1 | 0.697 | 0.134 | 0.998 |

## Mechanism and ablation table

| Comparison | Acc delta | Error delta | Changed | Better | Worse | Corrected overcredit |
| --- | --- | --- | --- | --- | --- | --- |
| baseline -> guideline_gate_v1 | 0.060 | -0.096 | 20 | 9 | 3 | 13 |
| guideline_gate_v1 -> guideline_gate_almost_boundary_v1 | 0.050 | -0.014 | 8 | 5 | 0 | 2 |
| guideline_gate_almost_boundary_v1 -> guideline_gate_no_top_end_guard_v1 | -0.030 | 0.013 | 11 | 2 | 5 | 2 |

## Remaining-error summary

| Remaining-error bucket | Count | Share of remaining errors |
| --- | --- | --- |
| overgenerous_full_credit | 14 | 0.467 |
| rubric_ambiguity | 13 | 0.433 |
| almost_vs_partial_boundary | 2 | 0.067 |
| reasoning_or_comprehension_failure | 1 | 0.033 |

## Fresh generalization status

- Fresh filtered remainder: baseline `0.627` -> winner `0.697` accuracy; error `0.208` -> `0.134`.

## Saved artifacts

- Markdown report: `artifacts/release/v0.1.0/generated/final_imo_result_package.md`
- Result tables JSON: `artifacts/release/v0.1.0/generated/result_tables.json`
- Main table snippet: `artifacts/release/v0.1.0/generated/main_result_table.md`
- Mechanism table snippet: `artifacts/release/v0.1.0/generated/mechanism_ablation_table.md`
- Error-bucket table snippet: `artifacts/release/v0.1.0/generated/error_bucket_table.md`
- Casebook JSON: `artifacts/release/v0.1.0/generated/casebook.json`
- Casebook markdown: `artifacts/release/v0.1.0/generated/casebook.md`

