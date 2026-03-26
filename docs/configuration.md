# Configuration

## Precedence

Runtime configuration resolves in this order:

1. package defaults
2. YAML config
3. environment variables
4. CLI overrides

## Runtime config file

Default runtime config: `configs/runtime/default.yaml`

```yaml
model: gemini-3-flash-preview
prompt_variant: guideline_gate_almost_boundary_v1
log_level: INFO
api_host: 0.0.0.0
api_port: 8000
```

## Environment variables

Provider credentials:

- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`

Optional runtime overrides:

- `PROOFGRADE_MODEL`
- `PROOFGRADE_PROMPT_VARIANT`
- `PROOFGRADE_LOG_LEVEL`
- `PROOFGRADE_API_HOST`
- `PROOFGRADE_API_PORT`

## Supported prompt variants

- `baseline`
- `guideline_gate_v1`
- `guideline_gate_almost_boundary_v1`

The shipped default is `guideline_gate_almost_boundary_v1`.

## Supported model path in v0.1.0

Officially supported for release claims:

- `gemini-3-flash-preview`

Other model names may remain in the underlying legacy code, but they are not supported for the public `v0.1.0` claim set.

