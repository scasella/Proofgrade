# Product Positioning

## Target users

- Researchers who want a stable proof-grading baseline
- Engineers building human-in-the-loop grading workflows
- Benchmark authors who want a reproducible rubric-aware evaluator
- Contest or education teams experimenting with supervised grading copilot flows

## What this project is

`proofgrade` is a narrow evaluator product:

- frozen prompt-policy line
- explicit rubric-aware grading
- stable CLI and API
- reproducible benchmark package

## What this project is not

- not a broad autonomous agent platform
- not a validated transfer-learning system
- not a replacement for human graders in high-stakes settings
- not a claim of general mathematical proof understanding

## Strongest credible claims

- The shipped winner improves the `imo_grading` benchmark over a frozen baseline.
- That gain survives a one-time untouched lockbox test.
- It also survives a fresh 512-response check within the same filtered task family.
- The mechanism is understandable and auditable.

## Near-term roadmap

- keep the current prompt-policy line frozen
- improve packaging and deployment ergonomics
- evaluate one stronger model against the same frozen policy on a new untouched pack
- expand casebook and operational guidance for human-supervised use

## Why the narrow scope is still useful

The value here is trust. A modest, reproducible evaluator with clear caveats is more useful than a larger, fuzzier system that overclaims what it can do.

