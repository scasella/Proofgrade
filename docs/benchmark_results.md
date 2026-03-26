# Benchmark Results

## Headline table

| Split | Baseline accuracy | Winner accuracy | Baseline error | Winner error | Winner valid-label rate |
| --- | --- | --- | --- | --- | --- |
| Held-out validation (100) | 0.590 | 0.700 | 0.251 | 0.141 | 1.000 |
| Untouched lockbox test (100) | 0.640 | 0.770 | 0.219 | 0.133 | 1.000 |
| Fresh filtered remainder (512) | 0.627 | 0.697 | 0.208 | 0.134 | 0.998 |

## Mechanism summary

The gain is attributable to two policy changes:

1. less over-generous full credit
2. tighter `almost` vs `partial` calibration

Validation ablation summary:

| Comparison | Accuracy delta | Error delta | Corrected overcredit |
| --- | --- | --- | --- |
| baseline -> guideline_gate_v1 | +0.060 | -0.096 | 13 |
| guideline_gate_v1 -> guideline_gate_almost_boundary_v1 | +0.050 | -0.014 | 2 |
| winner -> no-top-end-guard ablation | -0.030 | +0.013 | 2 |

## Lockbox result

Untouched test result:

- accuracy: `0.64 -> 0.77`
- normalized grading error: `0.219 -> 0.133`
- valid-label rate: `0.99 -> 1.00`

## Fresh response-level generalization

Fresh filtered remainder result:

- accuracy: `0.627 -> 0.697`
- normalized grading error: `0.208 -> 0.134`
- valid-label rate: `0.986 -> 0.998`

This is a positive generalization result, but it is weaker than the lockbox gain and remains within the same task family. Problem IDs overlap with the benchmark line.

## Remaining error buckets for the frozen winner

Locked benchmark remaining errors:

- `overgenerous_full_credit`: 14
- `rubric_ambiguity`: 13
- `almost_vs_partial_boundary`: 2
- `reasoning_or_comprehension_failure`: 1

Fresh 512-response remaining errors shift somewhat toward broader rubric ambiguity:

- `rubric_ambiguity`: 76
- `overgenerous_full_credit`: 69
- `almost_vs_partial_boundary`: 5
- `reasoning_or_comprehension_failure`: 2
- `unlikely_prompt_fix`: 3

