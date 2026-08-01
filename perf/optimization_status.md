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
