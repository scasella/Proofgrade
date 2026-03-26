# Quick Start

## 1. Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Set one of these provider credentials:

- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`

## 2. Check the runtime

```bash
proofgrade version
```

You should see:

- package version `0.1.0`
- default prompt variant `guideline_gate_almost_boundary_v1`
- default model `gemini-3-flash-preview`

## 3. Grade one proof from files

```bash
proofgrade grade \
  --problem-file examples/problem.txt \
  --solution-file examples/solution.txt \
  --guidelines-file examples/guidelines.txt \
  --answer-file examples/student_answer.txt
```

## 4. Grade a batch

Use a JSON array or JSONL file with records like:

```json
{
  "problem": "Problem text",
  "solution": "Official solution",
  "grading_guidelines": "(Partial) ... (Almost) ...",
  "student_answer": "Student proof"
}
```

Then run:

```bash
proofgrade batch-grade --input examples/batch.jsonl
```

## 5. Start the API

```bash
proofgrade serve --config configs/runtime/default.yaml
```

Open:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/version`
- `http://127.0.0.1:8000/docs`

## 6. Build the published result package

```bash
PYTHONPATH=. .venv/bin/python analysis/build_imo_result_tables.py --config configs/baseline_freeze/final_imo_release.yaml
PYTHONPATH=. .venv/bin/python analysis/build_imo_casebook.py --config configs/baseline_freeze/final_imo_release.yaml
```

These reproducibility commands assume a full clone of this repository. The slim runtime package and Docker image intentionally ship only the grading service surface.

## 7. Run in Docker

```bash
docker build -t proofgrade:0.1.0 .
docker run --rm -p 8000:8000 --env-file .env proofgrade:0.1.0
```

## 8. Read the right docs

- Product shape: [docs/product_positioning.md](docs/product_positioning.md)
- API surface: [docs/api.md](docs/api.md)
- Deployment: [docs/deployment.md](docs/deployment.md)
- Reproduction: [docs/reproducibility.md](docs/reproducibility.md)
- Caveats: [docs/limitations.md](docs/limitations.md)
