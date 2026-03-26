# Meta-Transfer Report

## 1. Problem framing
This follow-on project asks whether HyperAgents become more scientifically useful when transfer is measured and improved at the meta level rather than inferred from headline target scores alone. The main question is not “can a transferred source agent help once,” but “what part of that help is genuinely transferable infrastructure?”

## 2. Public-repo audit and paper/repo mismatch notes
- The public baseline matches the paper’s initial agent prompts closely.
- The public repo does not include a first-class transfer-attribution pipeline.
- The bundled paper logs are not actually present in this workspace; they are Git LFS pointers.
- Local `imo_grading` CSVs are absent until `domains/imo/setup.sh` is run.
- The released parent-selection implementation differs from the appendix description in important details.
- The paper discusses persistent memory and performance tracking as learned behaviors, but the public baseline does not ship a reusable structured memory module.

## 3. Hypotheses
- H1: Transfer gains are driven more by meta-level infrastructure than by domain-specific task patches alone.
- H2: Compact structured meta-memory should improve transfer more reliably than unstructured prior-run dumps.
- H3: Descendant-growth-aware source selection should outperform raw best-score and random source selection.

## 4. Methods
- Added deterministic patch taxonomy and lineage parsing.
- Added transfer-source ranking rules:
  - best score,
  - descendant growth,
  - random valid,
  - meta patch density,
  - hybrid growth plus meta density,
  - diversity-aware optional ranking.
- Added structured meta-memory with compact JSON/text outputs.
- Added transfer-mode filtering for:
  - full,
  - meta-only,
  - task-only,
  - search-only,
  - memory-only,
  - random-source transfer.
- Added smoke fixtures so the entire pipeline can be verified without hidden logs or paid runs.

## 5. Reproduction from archived logs
Exact paper-style reproduction is not currently possible from this workspace alone because the shipped run archive is only a Git LFS pointer, not the underlying logs. The new reproduction script therefore does two things:

1. rebuilds transfer summaries when real runs exist,
2. writes an explicit closest-possible report when the raw runs are missing.

This keeps the analysis honest and prevents silent over-claiming.

Follow-up execution confirmed that the public recovery path is exhausted for now:
- `git-lfs` was installed successfully,
- the repo correctly identifies `outputs_os_parts.*` as LFS-tracked files,
- but `git lfs fetch origin main --include='outputs_os_parts*'` returns `404 Object does not exist on the server` for all archive objects.

This upgrades the earlier “missing in workspace” caveat to a stronger statement: the historical logs are not recoverable from the current public LFS endpoint.

## 6. New interventions
The main intervention is structured meta-memory:
- compact prior-run state,
- trend summaries,
- best generations by domain,
- regressions and over-corrections,
- failure-mode counts,
- patch-category gain summaries,
- candidate next hypotheses,
- transferable versus domain-specific cues.

This intervention is fully opt-in through new CLI flags and does not change default behavior.

## 7. Main results
Current evidence in this workspace supports an engineering claim, not yet a real-experiment scientific claim:
- the repo now has a reproducible transfer-analysis and attribution pipeline,
- the runtime can consume inspectable structured meta-memory,
- the new selectors and transfer filters are runnable,
- smoke fixtures validate the end-to-end plumbing.
- the IMO grading dataset can now be materialized locally via the existing setup path.
- the public runtime had a real compatibility bug around internal-style model aliases; this is now fixed by normalizing those aliases before the API call.
- both the source-domain (`paper_review`) and target-domain (`imo_grading`) public harnesses now complete real API-backed runs in this workspace.
- the exact initial baseline folders expected by `setup_initial_gen(...)` now exist for both domains, so the self-improvement loop can start from real local artifacts rather than placeholders.
- a lightweight CPU loop image now exists, so the public self-improvement loop can be piloted on non-GPU domains without the heavyweight default build.
- the first full loop pilot now exists as a real runtime result rather than a hypothetical next step:
  - the loop can launch,
  - the meta agent can run inside the container,
  - patch export works,
  - and per-generation metadata is written locally.

The workspace still does **not** support a real-data claim that structured meta-memory improves transfer, because the raw transfer logs remain unavailable and the fresh pilots to date have not produced a meaningful improvement patch.

## 8. Negative results / failed ideas
- Exact public-log reproduction is blocked locally by missing LFS content.
- Real `imo_grading` reruns were initially blocked by missing benchmark CSVs; that data blocker is now cleared.
- Real Genesis reruns require a GPU path; this is why the full scripts point to Modal rather than assuming local CUDA.
- The first fresh 10-sample baseline runs for both `paper_review` and `imo_grading` are uniformly poor:
  - `paper_review` is `0.0` accuracy on train, val, and test.
  - `imo_grading` is `0.0` accuracy and `1.0` normalized MAE on train, val, and test.
- This means the fresh public baseline is currently functional but weak, so the first self-improvement pilot needs to be interpreted as a runtime and attribution check before it can support a strong empirical claim.
- The first completed loop pilot (`paper_review`, one generation, March 24, 2026) exposed a real infrastructure bug: the recorded patch consisted of Python cache files rather than source changes.
- That bug is now fixed by ignoring cache artifacts in diff capture and disabling bytecode generation in the container.
- After that fix, the public meta-agent still showed weak behavior with `openai/gpt-4o-mini`:
  - one pilot progressed beyond the root directory only after a repeated-tool-call guard was added,
  - another pilot, with a more goal-directed prompt, made real source edits but they were shallow help-text edits rather than changes likely to improve transfer or evaluation quality.
- This is an honest negative result: the loop is now runnable, but the current public meta-agent behavior is not yet producing scientifically useful self-improvements on `paper_review`.

## 9. Threats to validity
- Smoke fixtures only validate correctness of the pipeline, not the paper’s empirical claims.
- The public repo’s released heuristics do not perfectly match every appendix detail.
- Transfer attribution can be confounded if many variables change at once, which is why the new ablation modes isolate patch families explicitly.

## 10. Next experiment to run
1. Keep the runtime fixes and rerun the one-generation `paper_review` pilot with only one substantive change at a time:
   - either enable structured meta-memory,
   - or switch to a stronger meta model,
   - but not both at once.
2. Use the reproduction script on any recovered runs before launching any new experiments.
3. Run fixed-meta-agent transfer attribution on:
   - `initial_baseline`
   - `full_transfer`
   - `meta_only_transfer`
   - `task_only_transfer`
   - `memory_only_transfer`
4. If source Genesis reruns are needed, run them through `scripts/modal_gpu.py`.

## Direct answers

### What actually transfers?
Not yet established from real runs in this workspace. The pipeline is now set up to separate task, meta, search, and memory transfers once the raw artifacts are available.

### What intervention improved transfer the most?
Not yet established from real runs. The primary candidate remains structured meta-memory, and the strongest fallback claim is still source-agent selection by descendant growth.

### Did structured meta-memory help?
Not yet measured on real runs here. It is implemented, inspectable, and ready for controlled ablation.

### Are gains meta-level or just task-specific?
Not yet measured from real transfer runs here. The new ablation modes were added specifically to answer this cleanly.

### What is the cleanest publishable claim supported by the evidence?
The cleanest supported claim today is an engineering one: the public HyperAgents repo did not previously provide a rigorous, reproducible transfer-attribution pipeline, and it now does, with explicit handling for repo/paper mismatches, missing public artifacts, fresh runnable baselines for the main source/target domains, and verified loop-level infrastructure fixes discovered during real pilots.
