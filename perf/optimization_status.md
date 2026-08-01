# QuixiCore CUDA Optimization Status

This is the running notebook for CUDA kernel optimization. Raw output belongs
under `perf/results/`; stable conclusions belong here. Historical and baseline
snapshots live in `perf/baseline_status.md`.

## Entry Template

Use this structure for every kernel family or optimization pass:

```text
## YYYY-MM-DD: <kernel or pass name>

Status: not started | baselining | experimenting | candidate | landed | deferred.
Current implementation:
Current public route:
References inspected:
Correctness:
Baseline:
Experiments:
Decision:
Open questions:
Raw results:
```

Record enough context to reproduce the run: GPU, driver, CUDA toolkit, PyTorch
or library versions, command, git commit or working-tree label, dtype, shape,
quant format, warmups, iterations, median, variance, correctness tolerance, and
observed error.

## 2026-07-06: Shared Performance Documentation Alignment

Status: landed documentation scaffold.

Added the shared optimization notebook expected by `perf/perf.md`. The CUDA
handbook already contained the full measurement and experiment discipline; this
file is the tracked place for ongoing kernel-specific decisions, rejected
experiments, and final performance tables.

Current baseline material remains in `perf/baseline_status.md`:

- RTX 3090 / SM86 environment and roofline notes.
- Build/run baseline index.
- Framework reference numbers.
- Ampere port status and known performance debts.
- Existing result paths under `perf/results/`.

## Current High-Leverage Backlog

Status: baselined, experiments pending or partially complete.

| Area | Current finding | Next experiment |
|---|---|---|
| i-quant GEMV | lookup-bound from divergent table access | Move hot grids/tables into shared memory and remeasure format sweep |
| Quant GEMM prefill | dequant-to-fp16 plus cuBLAS is current route | Prototype fused dequant-in-pipeline GEMM and compare by `M` |
| MX/NV GEMV | below q8_0 effective bandwidth | Audit block-scale load coalescing and metadata placement |
| Attention backward D=128 | correctness landed, register pressure visible | Profile spills/occupancy and sweep split geometry |
| Decode attention/MLA/GDN/selective scan | low occupancy in one-row/one-state mappings | Batch more rows/states per block or add partition/chunk parallelism |
| Quant MoE GEMM | 32-row M-blocking landed | Add cp.async staging and compare grouped-library route at prefill sizes |

## Open Questions

- Which CUDA host should be treated as the canonical baseline for SM90/SM100
  kernels that cannot execute on the RTX 3090 box?
- Which results should be promoted from historical notes into compact
  per-kernel tables here?
- Should `perf/bench_kernels.py` grow a single normalized JSON schema for all
  standalone CUDA harnesses?

## A100-vs-MI300X campaign

Goal: 4x A100 TP4 beats 2x MI300X (82/141/176/260/297/408 aggregate tok/s at
1/4/8/16/32/64 conns; real prompts, temp 0, natural stops, tokens/drain-time).

### mla_decode_fp8_v: vectorized all-fp8 row loads (2026-08-01)

- Hypothesis: 6.75 us per serial iteration measured at BOTH ctx 20 (135 us/call,
  tlen 20) and ctx 1300 (1.73 ms/call, 256-iteration partitions) = the inner
  loop's QPL=18 rounds of one-byte-per-lane loads, each an uncoalesced 32 B
  transaction at full latency. MLA was 74% of GPU busy at ctx 1300.
- Change: NFP8==QW slots (GLM geometry) read 4 bytes/lane/round -> VW/128
  coalesced 128 B rounds + 2-byte tail rounds (rope, score-only); q loads uint2;
  float4/uint2 epilogue stores to the unchanged canonical layout (reduce kernel
  untouched). Scalar path kept for mixed-layout instantiations.
- Correctness: fp64 harness, all 3 geometries PASS (GLM fp8 rel 2.9e-3).
- Measured (bs=1 TP4 no-spec, ms/token at gen len 64/128/256/512):
  44.2/57.6/85.4/110.2 -> 27.2/29.3/34.7/39.6 = 1.63x/1.97x/2.46x/2.78x.
- Verdict: KEEP. Matrix delta (TP4 agg 1/4/8/16/32/64):
  23.4/57.8/82.7/105.0/185.9/216.9 -> 47.8/94.5/147.6/165.8/277.2/312.3
  (up to 2.04x); TP8 now beats the 2x MI300X reference at 8/16/32/64 conns.
- Open follow-ups filed: per-iteration idx/block-table prefetch (dependent-load
  chain still serial), 2 warps/block occupancy experiment.

Campaign log: 2026-08-01 iter1 vectorized MLA loads KEPT; matrix jumped up to 2x; new anomaly filed: TP4 bs=16 dent (165.8 vs 147.6@8 / 277.2@32).

### TP4 bs=16 "dent": not reproduced (2026-08-01, iter 2)

- Fine sweep bs 12/16/20/24/32, spec and no-spec, same methodology as the
  matrix: smooth monotonic scaling both ways (spec: 163.9/181.3/223.0/253.2/
  271.8). The matrix's 165.8@16 was workload-mix variance on a convex curve,
  not a cliff. Verdict: CLOSED, no defect. Matrix-row single runs carry ~9%
  column variance at mid batch; only cross-column shape claims need repeats.

### Fused allreduce+residual+RMSNorm: LOSER (2026-08-01, iter 3)

- Correct (residual bitwise vs unfused sequence; norm <= 6.0e-3 rel; ws 4+8)
  but end-to-end wash: bs1 +0.7% (3 reps), bs8 +0.9%, bs32 -0.7% vs bars of
  >=5%/>=3%. Root cause of the miss: at decode sizes the one-shot AR's ~20%
  step share is signal/sync LATENCY, not bandwidth or launches; the fused
  epilogue removes ~156 norm launches/step (few hundred us under graphs)
  and none of the latency. Reducing AR cost needs FEWER reduces, not fusion.
- Kept behind VLLM_FUSED_AR_NORM=1 (default OFF; baseline byte-identical when
  unset). Covers both per-layer decode ARs at ws 2/4/6/8; not prefill/draft.
  Aux-hidden-state tap layers (2,20,39,58,75) reduce explicitly first.
Campaign log: 2026-08-01 iter3 AR+norm fusion LOSER (noise-level); flag-gated
code committed; next: [f] DSpark draft overhead at bs=1.

### Split-K load balance + verify-width GEMV tile: KEEP (2026-08-01, iter 4)

- All three prior hypotheses wrong (draft IS graph-captured incl. Markov loop;
  mhc fallbacks not in path; eager pre/post 0.87 ms). Real cost: target verify
  at 4 tokens, dominated by (1) mla_decode_fp8_v split-K using FIXED partition
  spans -- ctx<256 ran entirely in partition 0, 16 warps on 108 SMs, and the
  0.44 ms/ctx-token step growth; now spans are ceil(len/P) per request,
  device-side, graph-safe: standalone 2.4-6.1x at len 32-512, identical at
  2048; (2) mmvq rows_per_block 2->4 for ncols_dst==4 (CUDA-only): dense GEMV
  step cost -13%.
- Correctness: fp64 harness extended w/ balanced-partition case, all PASS
  (<=2.9e-3); q8_0 GEMV <0.008 maxrel across 7 shapes; acceptance 2.6-2.7.
- A/B bs=1 natural 44.6 -> 63.8 (+43%), fixed-128 +23%, bs=32 -0.6% (band).
  Per-step MLA 9.6-19.1 ms -> 0.6-0.9 ms; profiled step 48.1 -> 33.9 ms.
- Remaining headroom filed: MoE vec +4.7 ms at verify width (expert-grouped
  batching), ~12.5 ms small-kernel latency floor in the 4300-kernel graph.
Campaign log: 2026-08-01 iter4 split-K balance + GEMV tile KEPT (+43% bs=1);
full matrix rerun for README in flight.

### iter5: high-batch stability (2026-08-01) — CLOSED, no regression
- TP4 3x repeats: bs=32 232/252/278 (mean 254), bs=64 304/329/285 (mean 306).
  bs=64 mean matches pre-iter4 (312); matrix3's 264 was a low draw. Natural-
  stop single runs at 32/64 carry +/-8-10%; treat README high-batch cells as
  mean-of-3 going forward.
Campaign log: iter5 stability CLOSED (variance, no mmvq regression); next [j].
