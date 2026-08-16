<!-- qx:shared:begin header v1 -->
# Agent Instructions

This is the QuixiCore CUDA backend. Kernel work must be correctness-first,
measurement-driven, and recorded in the performance notebook.
<!-- qx:shared:end header -->

<!-- qx:shared:begin read-first v1 -->
## Read First

- User-facing overview: `README.md`.
- Repository layout: `docs/repository-structure.md`.
- Performance operating guide: `perf/perf.md`.
- Established findings: `perf/findings.md` — read before proposing any
  experiment; do not re-derive or retry anything listed there without new
  evidence.
- Idea backlog: `perf/backlog.md` — pick the next experiment from here; update
  it when you finish one.
- Optimization notebook: `perf/optimization_status.md`.
- Baseline index: `perf/baseline_status.md`.
- Kernel metadata: `.quixicore/kernels.yaml` and
  `.quixicore/quant-formats.yaml`.
<!-- qx:shared:end read-first -->

<!-- qx:backend:begin read-first-extras -->
<!-- qx:backend:end read-first-extras -->

<!-- qx:backend:begin backend-notes -->
<!-- qx:backend:end backend-notes -->

<!-- qx:shared:begin perf-gate v1 -->
## Performance Optimization Requirement

Before committing any kernel implementation, kernel routing change, benchmark
change, or performance claim, the agent must complete at least one focused
performance optimization run on an affected kernel.

A valid run includes:

- The kernel, public route, dtype/format, and shape set.
- Correctness for the touched path.
- Baseline/current timing and candidate timing when testing a variant.
- GPU model, driver, CUDA toolkit, library/framework versions, command line,
  warmups, iterations, median, and variance or min/max.
- A keep/reject decision in `perf/optimization_status.md`.

If a suitable CUDA GPU/runtime is unavailable, do not commit a kernel
optimization or speedup claim. Stop and report the blocker, or restrict the
commit to docs/scaffolding with no performance claim.

Pure documentation and metadata-only commits may skip the kernel perf run, but
they must not claim a performance improvement.
<!-- qx:shared:end perf-gate -->

<!-- qx:shared:begin evidence-rules v1 -->
## Evidence Rules

- A run that times out or crashes is INCONCLUSIVE — never a rejection and never
  a win. Record it as INCONCLUSIVE with the timeout or failure mode.
- Only completed runs count as measurements. Never extrapolate from partial
  iterations or a killed benchmark.
- Report median plus variance or min/max from the stated warmup/iteration
  counts. If the baseline and candidate ranges overlap, there is no claim —
  rerun with more iterations or record INCONCLUSIVE.
- Never compare numbers taken on different hosts, drivers, or toolchains
  without saying so in the same sentence.
- Notebook verdict vocabulary: LANDED, KEPT, REJECTED, CANDIDATE, DEFERRED,
  INCONCLUSIVE, MIXED, RECORDED.
<!-- qx:shared:end evidence-rules -->

<!-- qx:shared:begin how-to-optimize v1 -->
## How To Optimize

- Start from `perf/perf.md`; form a bottleneck hypothesis before editing.
- Change one meaningful factor at a time: tile shape, launch geometry, memory
  layout, fusion, `cp.async`/shared-memory staging, dequant strategy, routing
  threshold, or specialization.
- Compare against cuBLAS/cuBLASLt, PyTorch, naive decompositions, and current
  CUDA kernels where relevant.
- Keep only wins that pass correctness, improve realistic priority shapes, and
  do not regress supported edge shapes or tolerances.
- Store raw output under `perf/results/`; copy durable conclusions into
  `perf/optimization_status.md`. Do not commit large Nsight traces.
<!-- qx:shared:end how-to-optimize -->

<!-- qx:shared:begin idea-selection v1 -->
## Idea Selection (The Beam)

- Keep `perf/backlog.md` to 3-5 active idea families, best first. Each family
  carries a parent result, hypothesis, evidence so far, next action, and kill
  criteria.
- Where measurable, give each family a quantitative target derived from
  recorded data — a percentage of the measured roofline, or beating a named
  baseline by a stated margin — never an invented number. A numeric target is
  what makes a long unattended session steerable.
- Pick the next experiment from the top of the beam unless the user directs
  otherwise. Do not run a single-incumbent hill climb on one idea.
- Do not kill a family on one failed singleton. If two ideas are individually
  neutral but touch independent costs, try the combination before retiring
  either.
- Keep one low-priority cleanup lane — dead code, archives, simplification,
  compile time — that never consumes the whole search. Schedule an occasional
  simplification pass after a run of landed wins.
- When an experiment concludes, update the beam (advance or kill) and promote
  durable conclusions to `perf/findings.md`. When a kill criterion fires,
  record the kill and its reason in `perf/findings.md` so it is never retried.
- New ideas that surface mid-run go to the backlog, not into scope creep.
<!-- qx:shared:end idea-selection -->

<!-- qx:shared:begin useful-commands v1 -->
## Useful Commands

```bash
.venv/bin/python perf/bench_kernels.py --phase all --kernel <kernel>
.venv/bin/python perf/bench_kernels.py --phase run --kernel <kernel> --gpus 0
CUDA_VISIBLE_DEVICES=0 bash perf/sweep_quant.sh
cd kernels/tm_cuda && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q
```

Use Nsight Compute/Systems when a timing A/B does not explain a result. Record
the trace path and conclusion, not the full trace.
<!-- qx:shared:end useful-commands -->

<!-- qx:shared:begin escalation v1 -->
## Escalation Protocol

- Stop and report to the user instead of improvising when required
  hardware/runtime is unavailable, when branch or upstream state is ambiguous,
  or when a correctness gate and a perf result conflict.
- Use subagents for bulk reading (notebooks over ~1,000 lines), broad reference
  searches, and multi-config sweeps. Subagents report numbers with full
  provenance; the primary agent writes the notebook entry.
- After two consecutive INCONCLUSIVE runs on one hypothesis, stop and present
  the data rather than rerunning a third time.
- When stuck (repeated rejects or sub-bar wins), re-read `perf/findings.md`,
  profile before guessing again, and prefer one structural change from a
  different beam family over another parameter retune.
- Still stuck after that: consult an advisor model. Hand a stronger or
  fresh-context model (for example a headless `claude -p` call) the latest
  profile, `perf/findings.md`, and the current beam, and ask for new idea
  FAMILIES — not variants of the incumbent. Advisor output feeds
  `perf/backlog.md`; it is ideas, never evidence.
<!-- qx:shared:end escalation -->

<!-- qx:shared:begin standing-questions v1 -->
## Standing Questions

Run through these before picking the next experiment; they exist to break
tunnel vision:

- What does the latest profile actually say, and does it match the assumed
  bottleneck? When did we last profile?
- What information are we missing, and what is the cheapest way to get it?
- What fusion candidates exist — values that round-trip through memory
  between launches, or multiple reductions sweeping the same data?
- What runtime signal can become a static signal — shapes, dtypes, formats,
  or alignments worth specializing or templating on?
- Is there numerical slack — precision the tolerance contract allows that we
  are not spending?
- Could a bug, not a bottleneck, be costing time (wrong launch geometry, an
  accidental sync, dead work)? A subagent bug hunt is cheap.
- What does the input data actually look like, and is there structure —
  sparsity, low rank, repeated values, outlier channels — a specialized path
  could exploit? (Distributions: `registry/benchmark-shapes.yaml`.)
<!-- qx:shared:end standing-questions -->

<!-- qx:shared:begin git-policy v1 -->
## Git Publishing Policy

- Never create, switch, rename, or choose a branch unless the user explicitly
  requests it. The user owns branch selection.
- Never create a pull request unless the user explicitly requests one.
- When asked to "commit and push," commit on the branch that is already checked
  out and push that same branch to its configured upstream.
- If the current branch has no upstream, pushing it is blocked, or branch
  selection is ambiguous, stop and ask the user rather than creating or
  switching branches or opening a pull request.
<!-- qx:shared:end git-policy -->

<!-- qx:shared:begin build-policy v1 -->
## Build Artifact Policy

- Keep generated build artifacts under one repository-local `build/` root, with
  incompatible configurations isolated as `build/<profile>/`.
- Do not create top-level task-, agent-, experiment-, or architecture-named
  build directories.
- Add a reusable preset or wrapper for a lasting configuration; use
  `build/scratch/` for a temporary experiment and remove it afterward.
- Build trees are disposable. Never treat one as durable correctness or
  performance evidence; record evidence under the documented `perf/` paths.
<!-- qx:shared:end build-policy -->

<!-- qx:shared:begin notebook-discipline v1 -->
## Notebook Discipline

- `perf/optimization_status.md` is append-only and ordered oldest-first. Never
  rewrite or reorder existing entries.
- New entry headings use the format `## YYYY-MM-DD: <kernel or pass> — VERDICT`
  with a verdict from the Evidence Rules vocabulary.
- After appending an entry, regenerate the index:
  `python3 ../tools/perf_notebook.py index perf/optimization_status.md`
  (run from this repository root; adjust the path if the umbrella checkout is
  a sibling rather than the parent directory).
- Distilled truths go to `perf/findings.md`; queued ideas go to
  `perf/backlog.md`; the notebook holds the full evidence.
<!-- qx:shared:end notebook-discipline -->

<!-- qx:shared:begin hygiene v1 -->
## Engineering Hygiene

- Check `git status` before editing. Do not revert user changes.
- Keep backend-local optimizations behind the public QuixiCore contract.
- Update metadata, tests, docs, and bindings when changing public behavior.
- Do not import reference implementation code unless licensing and provenance
  have been reviewed.
- Keep commits scoped and descriptive.
<!-- qx:shared:end hygiene -->
