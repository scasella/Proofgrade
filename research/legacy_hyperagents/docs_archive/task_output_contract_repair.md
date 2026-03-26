# Task Output Contract Repair

## Why this work is necessary

The current baseline is scientifically unusable for transfer analysis because the task agent often fails before the evaluation harness can even measure task skill.

The saved public baseline runs already show that the main blocker is the prediction contract:

- `paper_review` expects exactly one of:
  - `accept`
  - `reject`
- `imo_grading` expects exactly one of:
  - `incorrect`
  - `partial`
  - `almost`
  - `correct`

The current `task_agent.py` asks for a generic JSON payload with `{"response": ...}` and then returns that field directly. That is too weak for classification tasks with a fixed label space.

## Current failure modes

### `paper_review`

Observed in `outputs/initial_paper_review_filtered_100_{train,val}_0`:

- The model often returns long free-form review text in `response`.
- Those outputs are not normalized into the required binary label space.
- The saved baseline report is degenerate:
  - accuracy is `0.0`
  - the prediction distribution is mostly unique essay-like strings rather than labels

### `imo_grading`

Observed in `outputs/initial_imo_grading_filtered_100_{train,val}_0`:

- The model often returns prose judgments or otherwise invalid outputs instead of the exact four labels.
- Invalid or missing predictions receive full error under the harness metric.
- The saved baseline report is degenerate:
  - accuracy is `0.0`
  - normalized MAE is `1.0`

## Desired contract

For `paper_review` and `imo_grading`, the task agent should use an explicit domain-aware contract.

Preferred JSON shape:

```json
{
  "label": "...",
  "rationale": "...",
  "confidence": 0.0
}
```

Rules:

- `label` is the only field consumed by the harness.
- `label` must be lowercase and must belong to the exact domain label set.
- `rationale` is optional and should stay short.
- `confidence` is optional and should be numeric when present.
- The task agent should remain backward-compatible with the old `{"response": ...}` output if it appears.

## Normalization and repair rules

Repair must be deterministic and reproducible.

Required parsing order:

1. Try structured JSON extraction first.
2. Read `label` if present.
3. Fall back to compatible fields such as `decision`, `prediction`, or `response`.
4. Normalize obvious casing and punctuation variants.
5. Map only unambiguous variants into the exact label space.
6. If the intent is unclear, return an explicit invalid result rather than guessing.

Examples that should normalize:

### `paper_review`

- `Accept` -> `accept`
- `REJECT` -> `reject`
- `I would reject this paper.` -> `reject`
- `{"label":"accept"}` -> `accept`
- `{"response":"reject"}` -> `reject`

### `imo_grading`

- `correct` -> `correct`
- `almost correct` -> `almost`
- `partial progress` -> `partial`
- `incorrect` -> `incorrect`
- `{"label":"almost"}` -> `almost`
- `<points>6 out of 7</points>` -> `almost`

Examples that should stay invalid unless explicitly justified:

- `Outcome 1: COMPLETE PROOF`
- broad prose with no single clear label
- outputs containing conflicting labels

## Success criteria for this repair

The repo is past the degenerate-baseline blocker only if the repaired task contract produces:

- at least `95%` valid-label rate on small real pilots for both domains
- `paper_review` accuracy above `0.0`
- `imo_grading` normalized MAE below `1.0`

Only after that is it scientifically meaningful to revisit transfer questions.
