# Release Audit

## Public release goal

Public `v0.1.0` is a narrow evaluator release:

- a repaired HyperAgents-derived scaffold
- plus a frozen, rubric-aware proof grading engine
- plus a reproducible benchmark package for `imo_grading`

It is **not** a public claim that the full HyperAgents thesis was reproduced or validated.

## What belongs in the public release

- `proofgrade/` runtime package
- `configs/runtime/` and `configs/baseline_freeze/` needed for the frozen release line
- `domains/imo/` datasets needed for examples and reproducibility
- final IMO analysis scripts and curated result artifacts
- product docs, deployment docs, benchmark docs, and community files
- CI, Docker, and GitHub templates

## What stays but is clearly labeled as research-only

- archived multi-domain benchmark code under `research/legacy_hyperagents/`
- old agent/self-improvement scaffolding kept under `research/legacy_hyperagents/`
- archived research notes under `research/legacy_hyperagents/docs_archive/`

These are retained for context, not as part of the supported runtime.

## What should not ship as part of the release candidate

- raw `outputs/` eval dumps
- raw `analysis/outputs/` run dumps
- split output archive fragments
- stale top-level research memos that imply validated transfer or autonomous self-improvement
- docs or reports containing local absolute paths or provider secrets

## Public-safety scrub summary

- Removed split output archive fragments from the repo root
- Archived old transfer/task-repair docs and legacy code under `research/legacy_hyperagents/`
- Removed raw run dumps and bulky generated outputs from the public repo surface
- Replaced the old noncommercial software license with Apache-2.0 for the public release
- Reframed the repo away from broad HyperAgents claims and toward the frozen proof grader

## Remaining caution areas

- Legacy research code is still present in-repo under `research/legacy_hyperagents/` and must stay downplayed in README, CI, and release docs.
- Published benchmark claims must remain tied only to the frozen `imo_grading` line.
- The fresh 512-example result must continue to be described as response-level generalization within the same task family.
