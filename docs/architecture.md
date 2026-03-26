# Architecture

## Runtime shape

The public runtime is intentionally small.

1. `proofgrade.cli` handles CLI argument parsing and dispatch.
2. `proofgrade.api` exposes the same grading path through FastAPI.
3. `proofgrade.config` resolves defaults, YAML config, environment variables, and runtime overrides.
4. `proofgrade.policy` calls the frozen IMO grading instruction builders and parser.
5. `proofgrade.providers` calls a thin Gemini adapter that preserves the locked release behavior.
6. `proofgrade.grader` returns a stable grading response with label, metadata, and version information.

## Frozen grading path

- task family: `imo_grading`
- shipped default policy: `guideline_gate_almost_boundary_v1`
- comparison variants still exposed: `baseline`, `guideline_gate_v1`
- published release model: `gemini-3-flash-preview`

The runtime does not silently change prompt variant or model. Both are explicit in config and returned in CLI/API output.

## Benchmark and reproducibility path

The benchmark path is separate from the product runtime:

- `configs/baseline_freeze/` locks the published evaluation settings
- `analysis/` contains the frozen result scripts
- `artifacts/release/v0.1.0/` contains the curated release artifact bundle
- `proofgrade benchmark` expects this full repo checkout; the slim runtime wheel and Docker image are product surfaces, not reproducibility bundles

This keeps the shipped product surface small while preserving auditability.

## Legacy research code

Legacy multi-domain and self-improvement code remains in the repository for provenance, but it is not part of the supported runtime or public product claim.
