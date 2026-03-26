# Repo vs Paper Gap Audit

## What the public repo currently does
- The initial task agent and meta agent match the paper’s appendix very closely:
  - `task_agent.py` is a single-call JSON responder.
  - `meta_agent.py` starts from “Modify any part of the codebase”.
- The main loop is in `generate_loop.py`, not `meta_agent.py`.
- The normal parent-selection path uses `utils.gl_utils.select_parent`.
- Transfer utilities already exist in lightweight form:
  - `analysis/transfer_utils.py`
  - `process_meta_patch_files(...)` in `utils/gl_utils.py`
  - `--reset_task_agent` / `--reset_meta_agent` hooks in `generate_loop.py`
- Existing analysis scripts include hard-coded path lists and paper-number summaries for some published figures.

## What the paper claims
- The paper’s main experiments use a fixed score-and-child-count parent selector in the outer loop.
- Transfer source agents are chosen by a discounted descendant-growth rule with `gamma = 0.6` and at least 3 descendants.
- The DGM-H learns transferable meta-level capabilities such as:
  - performance tracking,
  - persistent memory,
  - evaluation analysis,
  - compute-aware planning.
- Main transfer claims are evaluated on held-out `imo_grading`.
- Main compounding claim continues training from transferred hyperagents in the new domain.

## Concrete gaps and simplifications

### 1. Archived logs are not actually present locally
- `outputs_os_parts.zip` and `outputs_os_parts.z01` ... `z08` are Git LFS pointer files in this workspace, not the underlying archives.
- Impact:
  - exact reproduction from raw public logs is blocked locally,
  - the pipeline must explicitly handle “missing raw runs” instead of assuming they exist.

### 2. The paper’s parent-selection description is only approximately reflected in released code
- The paper appendix describes a novelty term close to `1 / (1 + n_i)`.
- The repo’s main active heuristic in `utils.gl_utils.select_parent(..., method="score_child_prop")` uses a stronger exponential-style child penalty.
- `select_next_parent.py` itself is random and is only used in a separate “editable parent selection” path.
- Impact:
  - parent-selection claims should distinguish:
    - paper-described fixed selector,
    - public default heuristic,
    - editable selector experiments.

### 3. Structured meta-memory is described in the paper, but not implemented in the public baseline
- The paper shows examples of persistent memory and performance tracking.
- The released baseline code does not include a reusable structured memory module.
- Impact:
  - adding `utils/meta_memory.py` is a genuine follow-on intervention, not a repo refactor.

### 4. The repo does not ship a first-class transfer-attribution pipeline
- There is no built-in lineage dataframe, deterministic patch taxonomy, transfer-mode filtering, or selector comparison script.
- Impact:
  - attribution questions in this project require new analysis modules rather than small prompt edits.

### 5. The released task-agent baseline is under-specified for label-only tasks
- `task_agent.py` asks for a generic JSON `"response"` rather than domain-specific labels such as:
  - `accept` / `reject` for `paper_review`
  - `incorrect` / `partial` / `almost` / `correct` for `imo_grading`
- Fresh baseline runs in this workspace complete successfully but score poorly:
  - `paper_review` is `0.0` accuracy on the first 10 train/val/test examples,
  - `imo_grading` is `0.0` accuracy and `1.0` normalized MAE on the first 10 train/val/test examples.
- Impact:
  - the public baseline is runnable, but weaker and noisier than the paper narrative may suggest,
  - the first self-improvement pilot should be interpreted partly as a test of whether the system can learn the output contract itself.

### 6. The public meta-agent prompt is much less directed than the paper’s qualitative story suggests
- In the released code, the core instruction was effectively just “Modify any part of the codebase”.
- Fresh pilots on March 24, 2026 showed that this under-specification matters in practice:
  - one run got stuck repeatedly listing the repo root,
  - another only began inspecting relevant files after an explicit repeated-tool-call guard was added,
  - and even then the agent drifted toward shallow CLI help-text edits instead of substantive self-improvement changes.
- Impact:
  - the released public meta-agent path is runnable, but not yet reliably purposeful,
  - prompt and interaction quality are part of the real repo/paper gap, not just an implementation detail.

### 7. Some datasets required for full reproduction are absent locally
- `domains/paper_review/dataset.csv` is present.
- `domains/imo/*.csv` files are absent until `domains/imo/setup.sh` is run.
- Genesis setup clearly assumes GPU-backed execution.
- Impact:
  - local smoke validation can be complete,
  - full paper-style target experiments still require dataset setup and, for Genesis source runs, a GPU path.

### 8. The public repo is less “fully editable” than the paper framing suggests
- `setup_initial_gen(...)` excludes `analysis/`, `baselines/`, and non-selected domains from the copied repo.
- Domains are copied back in selectively.
- Impact:
  - actual self-modification scope in the released implementation is narrower than the broad conceptual framing.

## Experimental design consequences
- Use a local-first, bounded-recovery protocol for archived runs.
- Make reproduction scripts report unavailable evidence explicitly.
- Keep the main comparison scientifically clean:
  - fixed-meta-agent transfer attribution first,
  - continued self-improvement second,
  - parent-selection changes as appendix/secondary.
- Treat structured meta-memory as the main new intervention.
- Treat selector comparison as the cleanest fallback claim if full memory experiments cannot be run.
