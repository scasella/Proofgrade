# Release Final Report

## What changed

- Added a public `proofgrade` runtime package with CLI and FastAPI surfaces.
- Reframed the repository around the frozen `imo_grading` release line.
- Added reproducibility, deployment, API, benchmark, and positioning docs.
- Added GitHub issue templates, PR template, CI, release, dependency review, and CodeQL workflows.
- Replaced the old noncommercial license with Apache-2.0.
- Replaced the root Dockerfile with a slim CPU API image.

## Repo structure after cleanup

- `proofgrade/`: supported public runtime package
- `configs/runtime/`: runtime defaults
- `configs/baseline_freeze/`: frozen benchmark/repro configs
- `docs/`: public docs
- `analysis/`: only the frozen IMO release scripts
- `domains/imo/`: only the frozen IMO grading datasets used by the release line
- `artifacts/release/v0.1.0/`: curated release artifact bundle
- `research/legacy_hyperagents/`: archived research-only legacy code, notes, and configs

## Removed or archived

- removed split output archive fragments from the repo root
- moved legacy agent, transfer, and multi-domain code under `research/legacy_hyperagents/`
- removed raw eval dumps and bulky generated outputs from the public repo surface

## What is now production-ready

- slim `proofgrade` wheel install from `pyproject.toml`
- frozen grading runtime with explicit prompt-variant selection
- CLI and API product surfaces
- Docker local deployment path
- health and version endpoints
- public docs and reproducibility package
- CI and release workflows

## What is still research-only

- archived legacy multi-domain benchmark code
- archived self-improvement and transfer scaffolding
- archived exploratory reports and notes

## Release readiness

The repo is ready for a public GitHub release once:

- local verification commands in `PUBLISH_CHECKLIST.md` pass
- GitHub private vulnerability reporting is enabled
- the release branch is pushed and CI is green

## Manual steps remaining

- push the release branch
- confirm CI green on GitHub
- enable or verify private vulnerability reporting
- create tag `v0.1.0`
- publish the GitHub release using `RELEASE_NOTES_v0.1.0.md`
