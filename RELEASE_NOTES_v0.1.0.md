# proofgrade v0.1.0

## What shipped

This release turns the repository into a narrow, public-facing proof grading package:

- frozen `proofgrade` runtime package
- CLI and FastAPI service
- Docker path for local deployment
- curated reproducibility package for the locked `imo_grading` result
- public docs and GitHub workflows

## Headline results

- Held-out validation: `0.59 -> 0.70` accuracy, `0.251 -> 0.141` normalized grading error
- Untouched lockbox test: `0.64 -> 0.77` accuracy, `0.219 -> 0.133` normalized grading error
- Fresh filtered remainder (512): `0.627 -> 0.697` accuracy, `0.208 -> 0.134` normalized grading error

## Mechanism

The frozen winner works for understandable reasons:

- it is less over-generous with full credit
- it separates `almost` from `partial` more cleanly

## What changed from the research repo shape

- the public runtime now centers on a proof-grading product, not a broad HyperAgents platform claim
- the frozen `imo_grading` winner is the shipped default policy
- release docs, CI, templates, and deployment files are now included
- stale transfer-oriented top-level artifacts were removed or archived

## Example usage

```bash
proofgrade grade \
  --problem-file examples/problem.txt \
  --solution-file examples/solution.txt \
  --guidelines-file examples/guidelines.txt \
  --answer-file examples/student_answer.txt
```

```bash
proofgrade serve --config configs/runtime/default.yaml
```

## Limitations

- This release does **not** claim validated cross-domain transfer.
- This release does **not** claim to reproduce the full HyperAgents thesis.
- Fresh 512-example evaluation is fresh-response generalization within the same task family, not fresh-problem-family generalization.
- Human supervision is still recommended for consequential grading workflows.

