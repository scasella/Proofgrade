# Limitations

## Product limitations

- `proofgrade` is a narrow grader for the released `imo_grading` task family.
- It is intended for human-supervised evaluation workflows, not unattended high-stakes grading.
- It depends on a remote model provider and inherits provider latency and reliability limits.

## Benchmark limitations

- The public benchmark line is frozen around one task family.
- The fresh 512-example result uses unseen responses, but not unseen problem IDs.
- The repository does not contain a validated cross-domain transfer success result.

## Modeling limitations

- The remaining misses still cluster around rubric ambiguity and top-end generosity.
- Some further narrow policy room may remain, but large future gains are more likely to come from a stronger model than more prompt churn.
- This system is not a formal proof verifier. It grades with a learned model plus rubric scaffolding.

## Repo-shape limitations

- Legacy research code remains in the repository for provenance.
- Only the `proofgrade` runtime and the frozen IMO release line are intended as supported surfaces in `v0.1.0`.

