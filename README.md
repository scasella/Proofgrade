# Proofgrade

Proofgrade grades Olympiad-style math proofs against an official solution and rubric.

It is built for one job: helping a person review proofs more consistently. You give it the problem, the official solution, the grading rubric, and a student's proof. It returns one label, a short reason, and enough metadata to audit what happened.

This project publishes something that is still rare in public LLM grading repos: not just a prompt, but a frozen grading engine with named policy versions, lockbox results, fresh response-level generalization evidence, a casebook, a CLI, an API, and reproducible release artifacts.

Proofgrade grew out of cleanup work inside a HyperAgents-derived scaffold, but the public release is much narrower than that history. What ships here is a proof grader, not a general agent platform.

## Why this matters

A lot of proof-grading experiments stop at "here is a model and a prompt." That is hard to trust and hard to compare. The prompt changes, the behavior drifts, and the benchmark story gets blurry.

Proofgrade is meant to fix that.

It gives the community a grading line that is:

- frozen at a named policy version
- benchmarked against a clear baseline
- checked on an untouched test set
- checked again on 512 fresh responses from the same task family
- packaged so another engineer can run it without reconstructing the research process

That is the main contribution of this repo. It turns a promising grading setup into something inspectable, reproducible, and usable.

## What the grader actually does

The grader reads four inputs:

- the problem
- the official solution
- the grading rubric
- the student's proof

It then returns one of four labels:

- `incorrect`
- `partial`
- `almost`
- `correct`

Along with the label, it returns a short rationale and the policy/model metadata used for the decision.

The current release uses:

- provider: `gemini`
- model: `gemini-3-flash-preview`
- default policy: `guideline_gate_almost_boundary_v1`

This is not a custom-trained model. The improvement comes from a tighter grading policy wrapped around an off-the-shelf LLM, plus stable parsing and packaging around that policy.

## What improved

The benchmark gain came from two concrete changes:

1. The grader became less generous with full credit.
2. The grader got stricter about the line between `almost` and `partial`.

That sounds small, but it matters a lot in proof grading. Many of the old errors came from giving `correct` too easily or treating an unfinished proof as nearly complete.

This is why the result is easier to audit than a vague "prompt engineering win." We can point to the policy change and show the effect it had.

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

In plain terms, the frozen winner is right more often than the baseline, and when it is wrong it tends to miss by less.

Here, **grading error** means how far off the grade is on average. Lower is better.

Important caveat: the fresh 512-example result is fresh response-level evidence in the same task family. It is not evidence of new-problem-family generalization.

## Who this is for

Proofgrade is useful if you want:

- a first-pass proof grader for a human-supervised workflow
- a stable rubric-aware evaluator for research or benchmarking
- a reproducible baseline for grading-policy experiments
- a small service or CLI you can plug into internal grading tools

It is not built to replace human judgment on high-stakes decisions.

## Quick start

### 1. Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Then add one provider key to `.env` or your shell:

- `GEMINI_API_KEY`
- or `GOOGLE_API_KEY`

### 2. Confirm the runtime

```bash
proofgrade version
```

You should see the package version, default policy, provider, and model.

### 3. Grade one proof from files

```bash
proofgrade grade \
  --problem-file examples/problem.txt \
  --solution-file examples/solution.txt \
  --guidelines-file examples/guidelines.txt \
  --answer-file examples/student_answer.txt
```

Example response:

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

### 4. Run the API

Start the server:

```bash
proofgrade serve --config configs/runtime/default.yaml
```

Send one request:

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

## Reproduce the published result package

Build the public tables and casebook from the frozen release artifacts:

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

- the `proofgrade` runtime package
- a CLI for single grading, batch grading, benchmarking, serving, and version inspection
- a small FastAPI service with `/health`, `/version`, `/grade`, and `/batch-grade`
- frozen benchmark configs under [`configs/baseline_freeze`](configs/baseline_freeze)
- curated release artifacts under [`artifacts/release/v0.1.0`](artifacts/release/v0.1.0)
- docs for API, configuration, deployment, reproducibility, benchmark results, and limitations
- archived research context under [`research/legacy_hyperagents`](research/legacy_hyperagents)

## How Proofgrade could improve from here

The current benchmark line is locked. The next sensible gains are likely to come from cleaner evaluation and stronger models, not from endless reuse of the same validation slices.

The most credible next steps are:

- test the same frozen policy on a stronger model
- evaluate on a new untouched pack
- expand the casebook and operational guidance for human reviewers
- try one more narrow policy change only if new untouched evidence points to a specific boundary problem

What is not on the roadmap for this release line:

- reopening transfer claims
- reviving a "self-improving agent" story
- broad prompt sweeps on the same benchmark slices

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

- It is not a validated cross-domain transfer result.
- It is not a general autonomous agent platform.
- It is not a claim of fresh-problem-family generalization.
- It is not a replacement for human oversight on important grading decisions.
- It is not a formal proof verifier.

## Contributing and support

Read [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md) before opening a pull request or reporting a problem.

## License

Apache-2.0. See [LICENSE](LICENSE).
