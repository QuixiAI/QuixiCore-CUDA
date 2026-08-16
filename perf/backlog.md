# CUDA Optimization Backlog

The beam: 3-5 active idea families, best first. Pick from the top. Update
after every concluded experiment. Kill criteria are binding — when one fires,
record the kill in `perf/findings.md` and remove the family.

Where measurable, each family carries a quantitative target derived from
recorded data — a percentage of the measured roofline, or beating a named
baseline by a stated margin — set from `perf/findings.md` or
`perf/baseline_status.md`, never invented. The backend's aggregate score
lives in `perf/scoreboard.md`.

## Beam

### 1. Build plumbing + A100 kernel re-baseline (perf.md Open Items #1–2)
- Parent result: swapping the arch flag to `compute_80,code=sm_80` was the
  only change needed for 70/70 tk-independent tests green on this host
  (perf.md Host Reality, 2026-07-30); yet every canonical kernel number is
  still 3090-measured and no kernel-harness baseline exists on A100.
- Hypothesis: parameterizing ARCH in `kernels/tm_cuda/{setup.py,build_ext.sh}`
  and the harness headers, fixing the hard-coded `.venv` interpreter paths
  (`build_ext.sh:6`, `bench_kernels.py:304`) and the torch
  `LD_LIBRARY_PATH`/rpath, makes the full measurement stack runnable here —
  and the re-baseline will reorder the 3090-era priority ranking
  (FP32:bandwidth moved ~4x; see findings.md).
- Evidence so far: sm_80 build+correctness proven 2026-07-30; measured A100
  rooflines exist (1750 GB/s DRAM, 239 TFLOP/s bf16 cuBLAS, 353 TOP/s int8).
- Next action: parameterize the arch flag in `build_ext.sh`/`setup.py`, then
  run the 29-format `perf/sweep_quant.sh` qgemv sweep on one idle A100 and
  record it in `perf/optimization_status.md`.
- Kill criteria: none — prerequisite work; it concludes (rather than dies)
  when the format sweep, the MetalForge serving table, and the
  `qgemm_ksplit` threshold are recorded on A100. If the A100 host is lost,
  park the entire beam. Open Item #3 (baseline_status correction) is
  completed by this change (2026-08-15).

### 2. Q8_0 Marlin-route GEMM (kU8B128 relabel) + load-time layout repack
- Parent result: Q8_0 mmvq measured 53–60% of DRAM peak (SlimServe MI300X
  profile, cited in a100_glm52_design.md §1); Q8_0's A100 op budget is 5.92
  ops/weight and the Marlin `kU8B128` instantiation already exists — design
  §2.7/§6 items 1–2 expect 85–90%, touching 872 tensors.
- Hypothesis: +128 at repack, group_size=32, fp16 compute (`prmt` dequant +
  `__hsub2`, 0.5–1.0 op/weight) reaches 85–90% of roofline on the
  dense/attention GGUF matmuls; the layout-only repack (mma-fragment order,
  no byte growth) is the shared prerequisite with the Q2_K path.
- Evidence so far: budget arithmetic + measured 33x tensor-MAC headroom and
  the §2.2 probe showing the unpack is free; no A100 A/B yet.
- Next action: implement the layout-only repack for Q8_0 per design §2.5,
  then A/B against the current q8_0 route at registry `quant_matmul` shapes
  on one idle A100.
- Kill criteria: <8–10% end-to-end at serving shapes (complexity-adding
  bar), or fp16 numerics outside the `registry/tolerances.yaml` quantized
  bound.

### 3. Q2_K IMMA path for M≥2 + activation-scale coarsening 32→256
- Parent result: the landed q2k_ampere dp4a kernel is compute-bound from
  M≥2 — time scales linearly with M (M=1 775 GB/s = 44% of ceiling, M=8
  159 GB/s = 9%); the IMMA crossover is M≈2, not "prefill"
  (a100_glm52_design.md §2.5b, measured correction to the roadmap).
- Hypothesis: moving the M dimension onto `m16n8k32` int8 tensor cores (512
  MACs/instr vs dp4a's 4) restores bandwidth-bound scaling at decode batch;
  coarsening the activation scale group 32→256 cuts the float epilogue term
  ~10x (design §6 items 3/3b).
- Evidence so far: measured (NR,QB) sweep recorded in §2.5b (register
  pressure dominates tile choice; NR=2 gives 1081 GB/s at pure M=1);
  bank-conflict fix measured no change.
- Next action: implement the `m16n8k32` M-tile in
  `kernels/quant/q2k_ampere.cuh` and measure the M∈{1,2,4,8} sweep vs the
  NR=4/QB=2 baseline in the same harness (cross-harness reads 10–15%
  apart — compare within one harness).
- Kill criteria: IMMA fails to beat dp4a by ≥8–10% at M∈{2,4,8}, or the
  32→256 coarsening fails a measured accuracy eval (then keep group 32 and
  re-cost the epilogue).

### 4. Decode small-kernel/launch floor (quant-once, residual pass, node count)
- Parent result: iter7 cut 4508→3560 kernels/step (−21%, bs=1 −6.1%) and
  filed quantize_q8_1 at 562 launches / 1.42 ms per step; the Campaign
  closed entry (2026-08-01) left unstarted leads: quantize_q8_1 quant-once
  (~4–5% bs=1), residual small-kernel pass, bs=64 residual (~4%). Graph
  launch is 1.05 vs 3.01 us streamed [M], and at TP8 a graph-resident
  decode step is launch-bound (0.53 ms launch vs 0.28 ms bandwidth,
  a100_glm52_design.md §3, §6 item 4).
- Hypothesis: quantizing activations once per step instead of per-matmul,
  plus a residual small-kernel fusion pass, recovers most of the remaining
  ~7 ms small-kernel bucket at bs=1 and ~4% at bs=64.
- Evidence so far: iter7's profile-first top-20-table method is proven;
  small bucket now 7.02 ms; no experiment started on the leads.
- Next action: regenerate the top-20 kernel table at bs=1 on the current
  build, then implement quant-once for quantize_q8_1.
- Kill criteria: quant-once <3% at bs=1 (low-risk bar), or it breaks CUDA
  graph capture / the bitwise-identical routing guarantees.

### 5. Scalar-dequant debt: i-quant GEMV smem grids + fused dequant-in-pipeline GEMM
- Parent result: the 2026-07-06 backlog entry recorded i-quant GEMV as
  lookup-bound from divergent table access (15–71 GB/s on the 3090) and
  quant GEMM prefill routed via dequant-to-fp16 + cuBLAS; perf.md Host
  Reality (2026-07-30) shows both became MORE valuable on A100
  (FP32:bandwidth fell ~4x).
- Hypothesis: moving i-quant `__constant__` grids to smem removes the
  divergent-index constant-cache defeat; a cp.async-ring
  dequant-in-pipeline GEMM beats the separate dequant pass at M≥64 (the
  extra DRAM round-trip costs proportionally more against 239 TFLOP/s
  cuBLAS).
- Evidence so far: 3090-era ranking only; blocked on family 1's A100
  re-baseline for current numbers.
- Next action: after the family-1 sweep lands, A/B smem-resident grids for
  the iq2/iq3/iq4 formats at N=512/K=4096 plus registry shapes.
- Kill criteria: the A100 re-baseline shows i-quants already at their
  format-adjusted bandwidth ceiling, or smem grids <3% on the format sweep;
  for the fused GEMM, <8–10% vs cuBLAS-on-dequant at prefill M.

## Parked (not on the beam)

- perf.md Open Item #4: `fftconv_pc` now runnable on A100 (163 KB smem vs
  its 112 KB need) — free win awaiting a session.
- perf.md Open Item #5: register the port families in the
  `bench_kernels.py` registry (only 4/45 entries runnable today), or state
  plainly they are measured outside `results.jsonl`.
- perf.md Open Item #6: seed `optimization_status.md` per-kernel entries
  from `thundermittens_ampere_port.md` + the MetalForge table.
- perf.md Open Item #7: `dequant8` coverage for the 10 formats still on the
  generic path.
- perf.md Open Item #8: audit `shared_to_register.cuh` on H100 + int
  shared<->register unit-test coverage.
- perf.md Open Item #9: restore/vendor the `tk` module (unblocks the full
  pytest gate, golden regeneration, `bench_vs_torch.py`).
- perf.md Open Item #10: gitignore rule for the committed ELF binaries in
  `perf/tools/`.
- 2026-07-06 backlog rows not promoted: MX/NV GEMV block-scale coalescing;
  attention backward D=128 spill/occupancy + split geometry; decode
  attention/MLA/GDN/selective-scan occupancy batching (MLA decode largely
  addressed by campaign iter1/iter4); quant MoE cp.async staging +
  grouped-library comparison at prefill sizes.
- a100_glm52_design.md §6 items 5–7: sparse-MLA decode (gather-dequant to
  bf16 scratch then dense MQA); indexer logits kernel (replaces the
  pure-torch stub); stream-K / grouped MoE tile enumeration. §7: read the
  checkpoint's `index_topk_freq`/`index_topk_pattern` before any
  long-context work.
- Multi-GPU BAR1-P2P + `parallel/` kernels: parked TRACK with its own
  resume-point doc, `backlog_parallel.md` (3090 box; not duplicated here).

## Migrated sources

- (a) `perf/optimization_status.md` 2026-07-06 "Current High-Leverage
  Backlog — RECORDED": i-quant GEMV + fused dequant-in-pipeline GEMM →
  family 5; the other four rows → Parked. That entry is now annotated as
  superseded-as-a-queue by this file (2026-08-15).
- (b) `perf/perf.md` "Open Items On This Host" #1–2 → family 1; #3
  completed by the 2026-08-15 `baseline_status.md` restructure; #4–10 →
  Parked.
- (c) `perf/optimization_status.md` 2026-08-01 "Campaign closed" unstarted
  leads (quantize_q8_1 quant-once, residual small-kernel pass, bs=64
  residual) → family 4.
- (d) `perf/a100_glm52_design.md` §6 roadmap: items 1–2 → family 2, items
  3/3b → family 3, item 4 → family 4, items 5–7 → Parked.
