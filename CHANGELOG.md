# Changelog

All notable changes to this project will be documented in this file.

## v0.1.0 - 2026-03-25

### Added

- Public `proofgrade` runtime package with a stable proof-grading namespace
- CLI commands for single grading, batch grading, benchmark orchestration, API serving, and version inspection
- FastAPI service with `/health`, `/version`, `/grade`, and `/batch-grade`
- Frozen runtime config under `configs/runtime/default.yaml`
- Public release docs for architecture, API, deployment, reproducibility, benchmark results, limitations, and product positioning
- GitHub issue templates, PR template, CI workflow, release workflow, dependency review workflow, and CodeQL workflow
- Curated release artifact bundle under `artifacts/release/v0.1.0`

### Changed

- Reframed the repository around a narrow proof-grading product instead of a broad self-improving-agent claim
- Made `guideline_gate_almost_boundary_v1` the shipped default proof-grading policy
- Replaced the noncommercial Creative Commons license with Apache-2.0 for the public software release
- Replaced the old GPU-oriented root Dockerfile with a slim CPU API image for local deployment

### Removed or archived

- Removed split output archive fragments from the public release path
- Archived old transfer/task-repair docs under `docs/archive/`
- Moved the stale top-level transfer memo out of the main repo surface

