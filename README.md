# Proofgrade

`proofgrade` is a frozen, rubric-aware proof grading engine for Olympiad-style math proofs.

It takes four inputs:

- the problem
- the official solution
- the grading rubric
- the student's proof

and returns a grading label with a short rationale.

This release is narrow on purpose. It is meant to be a serious grading engine and grading copilot, not a general autonomous agent platform. The project grew out of a repaired HyperAgents scaffold, but what ships here is a locked proof grader with a small runtime surface, reproducible benchmark evidence, and an auditable policy story.

## Why this exists

Proof grading is a place where generic LLM prompting often breaks in predictable ways:

- it gives full credit too easily
- it blurs `almost` and `partial`
- it returns output that is hard to use consistently in a real workflow

`proofgrade` exists to turn one strong, locked grading line into something the community can actually inspect, reproduce, and use.

## What Proofgrade contributes

What is new here is not a new base model. The core model is still an off-the-shelf LLM.

What this project adds is:

- a **frozen grading policy** with named variants, rather than an undocumented prompt that keeps changing
- a **real benchmark win** on `imo_grading`, not just anecdotal examples
- a **one-time untouched lockbox test result**, so the main claim is not only based on development slices
- a **fresh response-level generalization check** on 512 additional examples from the same task family
- a **clear mechanism story** for the improvement:
  - less over-generous full credit
  - better `almost` vs `partial` calibration
- a **usable product surface**:
  - Python package
  - CLI
  - FastAPI service
  - Docker path
- a **reproducibility package** with frozen configs, result tables, and casebook artifacts

In practical terms, this means the community gets a proof grader that is:

- stronger than the baseline it started from
- honest about what it does and does not prove
- easy to run locally
- easy to inspect and compare

## What problem it solves

Many human-in-the-loop grading workflows need a tool that can do a disciplined first pass before a person reviews edge cases. That tool needs to be:

- more consistent than ad hoc prompting
- explicit about its rubric behavior
- stable enough to benchmark and integrate

`proofgrade` is built for that role.

It is especially aimed at:

- researchers who want a reproducible proof-grading baseline
- benchmark authors who want a locked evaluator rather than a moving prompt target
- engineers building grading copilots or supervised evaluation workflows
- contest and education teams experimenting with AI-assisted rubric-based grading

## Headline results

| Evaluation | Baseline | Frozen winner |
| --- | --- | --- |
| Held-out validation (100) | 0.59 accuracy / 0.251 grading error | 0.70 accuracy / 0.141 grading error |
| Untouched lockbox test (100) | 0.64 accuracy / 0.219 grading error | 0.77 accuracy / 0.133 grading error |
| Fresh filtered remainder (512) | 0.627 accuracy / 0.208 grading error | 0.697 accuracy / 0.134 grading error |

Valid-label rate stayed effectively perfect across the locked winner path:

- Validation: `1.00`
- Lockbox test: `1.00`
- Fresh 512-response check: `0.998`

In plain English:

- the frozen winner is right more often than the baseline
- when it misses, it tends to be less wrong
- it almost always returns a usable grading label

Here, **grading error** means how severe the mistake is on average. Lower is better.

Important caveat: the fresh 512-example result is **fresh response-level generalization within the same task family**, not new-problem-family generalization. Problem IDs still overlap with the benchmark line.

## Why it improved

The gain is understandable, not mysterious.

Two policy changes did most of the work:

1. **Less over-generous full credit**
   The grader became stricter about when `correct` is allowed.
2. **Better top-end boundary calibration**
   The grader became more disciplined about when a proof is truly `almost` complete versus merely `partial`.

That matters because it makes the release easier to trust. This is not just “we found a better prompt somehow.” It is a more auditable grading policy with a measurable effect.

## How to use Proofgrade

### 1. Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Then set one provider credential in your environment:

- `GEMINI_API_KEY`
- or `GOOGLE_API_KEY`

The published benchmark line is tied to:

- provider: `gemini`
- model: `gemini-3-flash-preview`
- default shipped policy: `guideline_gate_almost_boundary_v1`

### 2. Check the runtime

```bash
proofgrade version
```

This should show the active default model, prompt variant, package version, and git SHA when available.

### 3. Grade one proof from files

```bash
proofgrade grade \
  --problem-file examples/problem.txt \
  --solution-file examples/solution.txt \
  --guidelines-file examples/guidelines.txt \
  --answer-file examples/student_answer.txt
```

Example output:

```json
{
  "label": "partial",
  "rationale": "The main construction is present, but the final justification is still missing.",
  "matched_guideline": "partial",
  "review_recommended": true,
  "prompt_variant": "guideline_gate_almost_boundary_v1",
  "model_provider": "gemini",
  "model_name": "gemini-3-flash-preview",
  "version": "0.1.0"
}
```

### 4. Grade through the API

Start the server:

```bash
proofgrade serve --config configs/runtime/default.yaml
```

Then send one request:

```bash
curl -X POST http://127.0.0.1:8000/grade \
  -H "Content-Type: application/json" \
  -d '{
    "problem": "Prove that ...",
    "solution": "Official solution ...",
    "grading_guidelines": "(Partial) ... (Almost) ...",
    "student_answer": "Student proof ..."
  }'
```

Useful local endpoints:

- `GET /health`
- `GET /version`
- `GET /docs`

### 5. Grade a batch

```bash
proofgrade batch-grade --input examples/batch.jsonl
```

### 6. Reproduce the published result package

Build the public tables and casebook from frozen artifacts:

```bash
PYTHONPATH=. .venv/bin/python analysis/build_imo_result_tables.py \
  --config configs/baseline_freeze/final_imo_release.yaml

PYTHONPATH=. .venv/bin/python analysis/build_imo_casebook.py \
  --config configs/baseline_freeze/final_imo_release.yaml
```

If you want to rerun the final untouched test workflow itself:

```bash
GEMINI_API_KEY=... GOOGLE_API_KEY=... \
PYTHONPATH=. .venv/bin/python analysis/run_final_imo_lockbox_test.py \
  --config configs/baseline_freeze/final_imo_lockbox_test.yaml
```

`proofgrade benchmark` is for a full repo checkout with the frozen configs and analysis scripts present. The slim package and Docker image are for runtime grading use, not for replaying the full research history.

## What ships in v0.1.0

- a frozen proof-grading runtime package: [`proofgrade`](proofgrade)
- a CLI:
  - `proofgrade grade`
  - `proofgrade batch-grade`
  - `proofgrade benchmark`
  - `proofgrade serve`
  - `proofgrade version`
- a small FastAPI service with `/health`, `/version`, `/grade`, and `/batch-grade`
- frozen benchmark configs under [`configs/baseline_freeze`](configs/baseline_freeze)
- curated release artifacts under [`artifacts/release/v0.1.0`](artifacts/release/v0.1.0)
- public docs for architecture, API, deployment, reproducibility, benchmark results, and limitations
- archived legacy research context under [`research/legacy_hyperagents`](research/legacy_hyperagents)

## How Proofgrade can be improved

The current benchmark line is locked. The next sensible improvements are not open-ended prompt tweaking.

The most credible next steps are:

- test the same frozen policy on a stronger model
- evaluate on a new untouched pack instead of reusing old benchmark slices
- expand the casebook and operational guidance for human-supervised grading workflows
- possibly try one more very narrow policy refinement only if it is justified by new untouched evidence

What is **not** the plan:

- not reopening broad transfer claims
- not reviving a “self-improving agent” story
- not chasing more validation gains on the same benchmark slices

## Documentation

### Use it

- [Quick start](QUICKSTART.md)
- [API](docs/api.md)
- [Configuration](docs/configuration.md)
- [Deployment](docs/deployment.md)

### Trust it

- [Benchmark results](docs/benchmark_results.md)
- [Reproducibility](docs/reproducibility.md)
- [Casebook](docs/casebook.md)
- [Limitations](docs/limitations.md)

### Understand it

- [Architecture](docs/architecture.md)
- [Model providers](docs/model_providers.md)
- [Product positioning](docs/product_positioning.md)

### Contribute safely

- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)

## What this project is not

- Not a validated cross-domain transfer result
- Not a general autonomous agent platform
- Not a claim of fresh-problem-family generalization
- Not a replacement for human oversight on high-stakes grading
- Not a formal proof verifier

## Contributing and support

Please read [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md) before opening a pull request or reporting a problem.

## License

Apache-2.0. See [LICENSE](LICENSE).
