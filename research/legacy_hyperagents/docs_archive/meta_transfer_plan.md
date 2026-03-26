# Meta-Transfer Implementation Plan

## Goal
Turn the paper’s transfer thesis into a reproducible repo-native pipeline that can answer:

1. What part of cross-domain gains comes from task patches versus meta-level infrastructure?
2. Does structured run memory improve zero-shot transfer and continued self-improvement?
3. Which transfer-source selector is most transferable?
4. Do gains survive judge or evaluator changes when those artifacts are available?

## Implementation Order

### 1. Audit and reproduction boundary
- Use the paper source and public repo as the source of truth.
- Treat bundled `outputs_os_parts.*` as unavailable until actual LFS content is recovered.
- Treat `paper_review` and `genesis_go2walking` as source domains.
- Treat `imo_grading` as the primary held-out target.

### 2. Core analysis artifacts
- `utils/lineage.py` normalizes archive parsing, tree structure, descendant counts, and growth scores.
- `analysis/build_lineage_dataset.py` emits a lineage dataframe with patch metadata, score maps, and transfer labels.
- `analysis/classify_patch_types.py` provides deterministic patch taxonomy labels.
- `analysis/select_transfer_agents.py` ranks source agents by score, growth, meta density, random selection, or hybrids.

### 3. Main intervention
- `utils/meta_memory.py` builds compact structured memory from prior runs.
- `run_meta_agent.py` now writes inspectable `meta_memory.json` and `meta_memory.txt` artifacts when enabled.
- `meta_agent.py` consumes the structured memory only when explicitly requested.
- `generate_loop.py` exposes opt-in flags so default behavior stays unchanged.

### 4. Transfer-attribution experiments
- `analysis/run_transfer_ablation.py` prepares filtered transfer patches for:
  - `initial_baseline`
  - `full_transfer`
  - `meta_only_transfer`
  - `task_only_transfer`
  - `search_only_transfer`
  - `memory_only_transfer`
  - `random_source_transfer`
- The same entrypoint also summarizes completed runs into attribution plots and tables.

### 5. Reproduction and reporting
- `analysis/reproduce_transfer_figures.py` rebuilds transfer-style summaries when runs exist.
- If real runs are absent, it writes an explicit closest-possible report instead of inventing numbers.
- `REPORT_meta_transfer.md` captures the current strongest claim supported by available evidence.

## Verification Strategy
- Use `analysis/smoke_data.py` to create a tiny synthetic archive suite.
- Smoke checks must cover:
  - lineage dataset generation,
  - transfer-source ranking,
  - meta-memory creation,
  - ablation manifest generation,
  - reproduction reporting and plots.
- Use Modal for GPU-dependent Genesis work when real runs are launched.

## Default Assumptions
- No hidden data is assumed.
- No test-set tuning is allowed.
- Main attribution studies should prefer fixed-meta-agent transfer first, then continued self-improvement as a secondary comparison.
- Judge-shift stays secondary until the main transfer-attribution story is stable.
