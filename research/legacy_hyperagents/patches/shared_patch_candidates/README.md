# Shared Patch Candidates

These candidates are intentionally small and only target the already-approved shared surface traced by the transfer-eligibility gate.

Each candidate has one mechanism:

- `shared_truncated_json_label_salvage`
  - Shared parser fallback.
  - Tries to recover a clear label from a truncated JSON prefix such as `"label": "accept"` when the full object never closes.

- `shared_instruction_wrapper_tightening`
  - Shared task-instruction wrapper.
  - Adds a global rule that the answer must be the smallest valid JSON object, with `label` first and no extra text.

- `shared_compact_input_format`
  - Shared input-serialization change.
  - Makes the task input JSON more compact before it is sent to the model.

- `shared_short_label_budget`
  - Shared Gemini request-configuration change.
  - Reduces the allowed output size for label-style tasks so the model is pushed toward shorter structured answers.

The tracked config for the sprint is:

- `/Users/scasella/Downloads/hyperagents/HyperAgents/configs/baseline_freeze/shared_patch_variants.yaml`

The benchmark runner materializes each candidate into a copy of the frozen baseline snapshot, validates transfer eligibility, and saves the exact generated diff under:

- `analysis/outputs/shared_patch_sprint/candidates/<variant_id>/candidate.diff`

The source of truth for whether a candidate is allowed is still the shared-path allowlist:

- `/Users/scasella/Downloads/hyperagents/HyperAgents/configs/baseline_freeze/shared_path_allowlist.yaml`
