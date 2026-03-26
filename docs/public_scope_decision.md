# Public Scope Decision

## Narrow product being released

The public product is:

> A frozen, rubric-aware proof grading engine and grading copilot for IMO-style responses.

It is intended for human-supervised evaluator workflows where an engineer, researcher, or contest organizer wants a stable grading baseline with explicit rubric handling and reproducible benchmark evidence.

## Strongest credible public claims

- Derived from work inside a repaired HyperAgents scaffold
- Ships a frozen proof-grading policy line with explicit prompt-variant selection
- Improves `imo_grading` materially over the shipped baseline
- Holds on untouched lockbox test
- Shows positive fresh response-level generalization on additional filtered examples
- Has an auditable mechanism: less over-generous full credit and better `almost` vs `partial` calibration

## Non-claims

- No validated cross-domain transfer success
- No claim of general autonomous self-improvement
- No claim that the full HyperAgents research thesis was reproduced
- No claim of fresh problem-family generalization

## Keep / archive / remove decision

- Keep and support: `proofgrade` runtime, frozen IMO configs, final IMO release scripts, curated result artifacts, public docs
- Keep but archive or downplay: legacy multi-domain research code and planning docs under `research/legacy_hyperagents/`
- Remove from the public product surface: raw eval dumps, bulky generated outputs, stale transfer memos, and local-path-heavy internal-style reports
