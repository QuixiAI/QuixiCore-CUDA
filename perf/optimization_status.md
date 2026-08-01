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

### iter6: MoE at verify width — KEEP, config-only (2026-08-01)
- w1 vec->MMQ crossover 64->16 on CUDA (fused_moe.py one-liner): -10-12%
  ms/decode-step at bs=8 (acceptance-corrected metric, +/-0.5%); bs=1 and
  bs=32 paths unchanged. Kernel-level: MMQ wins from 32 tokens (604 vs 720us).
- LOSERS w/ numbers: expert-sorted vec 0.95-1.03x (L2 already dedups; vec is
  per-row compute-bound); rows_per_block retune (heuristic already optimal);
  MMQ at bs=1 -2.7%. Correctness: vec==MMQ rel L2 to 5 decimals, K=256 safe.
- NEGATIVE: bs=1 MoE at practical floor (~40.1 ms/step, ~61 tok/s at typical
  acceptance) -- 82 target needs [k] small-kernel floor or acceptance gains,
  not MoE scheduling.
Campaign log: iter6 KEEP (w1 crossover 16); next [k] kernel-count floor.

### iter7: decode-graph kernel-count reduction — KEEP (2026-08-01)
- Profile-first top-20 table (4508 kernels/step, small bucket 9.66 ms) drove 5
  fusions: fill_ removal before GGUF matmuls (-563 launches, CUDA-gated),
  one-launch sparse_topk_tlen (-156), per-step block-table gather cache (-77),
  split-q MLA (kills 78 cats/step, BITWISE equal), fused moe_weighted_sum
  (-75). Result: 3560 kernels/step (-21%), small bucket 7.02 ms.
- A/B ms/step: bs=1 40.01->37.58 (-6.1%), bs=8 -4.2%, bs=32 322.5->291.9
  (-9.5%). Natural-stop bs=1 mean-of-4 69.8. Correctness: exact/bitwise where
  applicable; allocator-poison runs prove fill removals safe; smoke coherent.
- Remaining headroom filed: quantize_q8_1 562x/1.42 ms (quant-once
  restructuring), router-adjacent copy_ 76x, AR 1.30 ms.
Campaign log: iter7 KEEP (kernel floor -21%); fresh matrix for README next.

### iter8: dequant+cuBLAS route for wide q8_0 — KEEP, default on (2026-08-01)
- Crossover measured per shape: cuBLAS wins from rows>=96 (N<8192) / >=160
  (wide incl. lm_head); gate at those bounds, hatches VLLM_GGUF_CUBLAS=0 and
  _MIN_BATCH. Per-shape persistent scratch (~230 MB/GPU dense + 476 MB vocab),
  ggml_dequantize_into (no-alloc no-fill, USE_ROCM-gated), graph-safe.
- Correctness: rel L2 2.4-2.9e-3 -- about HALF of mmq_v2's 5.6e-3.
- A/B mean-of-3 natural: bs=32 270.2->299.0 (+10.7%, beats 297 target),
  bs=64 328.3->360.4 (+9.8%). Fixed-length guards: bs=1 +2.7%, bs=8 -1.1%
  (natural-stop bs=8 dip shown to be content/stop-length artifact).
Campaign log: iter8 KEEP; TP4 now beats reference at bs=8 and bs=32; fresh
matrix next; remaining gaps bs=1/4/16/64.

### iter9: MoE MMQ padding skip + w1 crossover 8 — KEEP (2026-08-01)
- Attribution at bs=8 verify width: MoE Q2_K MMQ = 63.7% of an 85 ms step;
  draft only 1.5 ms. ncu: DRAM near-perfect (21% peak) but L1TEX 57% -->
  LSU-bound computing padding: moe_align pads experts to mmq_x=4 blocks, so
  only 256/656 tile columns were real at 256 routed rows. Candidate list
  (cuBLAS<96, quantize_q8_1) was wrong again -- profile-first vindicated.
- Fix: skip_padding template predicate (warp-uniform, k-sweep skip), host
  gate rows < experts*mmq_x, hatch VLLM_GGUF_MOE_SKIP_PAD=0; BITWISE
  identical outputs at all widths. Plus w1 vec->MMQ crossover 16->8
  (MMQ now 331 vs 360 us at 16 tokens).
- ms/step (median-of-3 fixed-1024): bs=1 -3.2%, bs=4 -9.7%, bs=8 -13.2%,
  bs=16 -13.0%, bs=32 -1.3% (bit-identical path). Kernel: w1 1.23-1.44x,
  w2 1.28-1.56x at mid widths.
- Follow-ups filed: dense mmq_v2 next at 15.3% (~300 GB/s at 32-64 rows);
  re-sweep MOE_X=8 now that padding is free.
Campaign log: iter9 KEEP; fresh matrix for README + stop-check next.
