# Casebook

This casebook is a compact guide to how the frozen winner changes grading behavior.

## Typical improvements

The frozen winner most often helps by:

- pushing obviously incomplete proofs down from `correct`
- separating near-complete work from ordinary partial progress more cleanly
- keeping outputs structurally valid while doing so

Representative improvements and regressions are also exported under:

- `artifacts/release/v0.1.0/casebook.json`
- `artifacts/release/v0.1.0/casebook.md`

## High-signal examples

### Over-generous full credit corrected downward

- `GB-0658`: baseline `correct`, frozen winner `partial`, gold `incorrect`
- `GB-0346`: baseline `correct`, frozen winner `partial`, gold `incorrect`

### Near-complete work preserved as near-complete

- `GB-0367`: baseline `correct`, frozen winner `almost`, gold `partial`

### Remaining hard misses

- `GB-0064`: frozen winner stays `partial` while gold is `correct`
- `GB-0669`: frozen winner remains too generous at `correct` while gold is `partial`

## What to look for

If you inspect raw cases, the most important questions are:

- did the model award full credit too easily?
- did it mistake a single missing repair for a broader incompleteness?
- is the remaining miss a rubric-boundary issue or a deeper reasoning failure?

