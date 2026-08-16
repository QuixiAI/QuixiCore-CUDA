# CUDA Established Findings — Do Not Re-Derive

Distilled from `perf/optimization_status.md` through 2026-08-01. Treat as
current truth until re-measured; every entry names its date and notebook
entry so it can be challenged with new data.

## Environment anchor

- 8x NVIDIA A100-SXM4-80GB (GA100, SM 8.0), 80 GB HBM2e each; driver
  595.45.04, CUDA toolkit 13.2, PyTorch 2.13.0+cu130; 108 SMs, 40 MB L2,
  164 KB smem/SM (163 KB max dynamic per block); full NV12 NVLink mesh.
  Source: perf.md Host Reality (2026-07-30).
- The 2026-08-01 campaign findings below are end-to-end serving measurements
  on this host (SlimServe stack, GLM Q2_K/Q8_0 GGUF, TP4/TP8, CUDA graphs,
  DSpark k=3 draft, fp8 KV) plus standalone kernel A/Bs.
- All 3090-era numbers (perf.md historical section, `baseline_status.md`
  "Superseded (historical)") are a DIFFERENT target's results (RTX 3090,
  SM 8.6) — the port's acceptance evidence, not this host's baseline. Never
  mix the two hosts' numbers in one comparison.

## Wins

All rows are from the 2026-08-01 "A100-vs-MI300X campaign" entry in
`perf/optimization_status.md`; "iterN" names the campaign subsection.

| finding | effect | date | notebook entry |
|---|---|---|---|
| mla_decode_fp8_v: vectorized all-fp8 row loads (4 B/lane/round coalesced, uint2 q loads, float4/uint2 epilogue) | bs=1 TP4 ms/token 44.2/57.6/85.4/110.2 → 27.2/29.3/34.7/39.6 (1.63–2.78x); serving matrix up to 2.04x | 2026-08-01 | campaign iter1 |
| Split-K load balance: device-side ceil(len/P) partition spans (fixed spans ran ctx<256 entirely in partition 0 = 16 warps on 108 SMs); plus mmvq rows_per_block 2→4 at ncols_dst==4 | standalone MLA 2.4–6.1x at len 32–512; dense GEMV step −13%; bs=1 natural 44.6→63.8 (+43%) | 2026-08-01 | campaign iter4 |
| MoE w1 vec→MMQ crossover 64→16 (config-only; MMQ wins from 32 tokens, 604 vs 720 us) | −10–12% ms/decode-step at bs=8; bs=1 and bs=32 paths unchanged | 2026-08-01 | campaign iter6 |
| Decode-graph kernel-count reduction: 5 fusions off a profile-first top-20 table (fill_ removal, one-launch sparse_topk_tlen, block-table gather cache, split-q MLA, fused moe_weighted_sum) | 4508→3560 kernels/step (−21%), small bucket 9.66→7.02 ms; bs=1 −6.1%, bs=8 −4.2%, bs=32 −9.5% | 2026-08-01 | campaign iter7 |
| dequant+cuBLAS route for wide q8_0 (gate: rows≥96 for N<8192, ≥160 wide incl. lm_head; per-shape persistent scratch; graph-safe) | bs=32 +10.7% (270.2→299.0), bs=64 +9.8%; rel L2 ~half of mmq_v2's | 2026-08-01 | campaign iter8 |
| MoE MMQ padding skip (skip_padding predicate, BITWISE identical outputs) + w1 crossover 16→8 | bs=4 −9.7%, bs=8 −13.2%, bs=16 −13.0% ms/step; kernel w1 1.23–1.44x, w2 1.28–1.56x | 2026-08-01 | campaign iter9 |
| mmq_v2 floor (not ceil) split-K nsplit + MoE Y=64 width gate at ≥1024 routed rows | up to 30% kernel-level at serving shapes; bs=8 −5.8%, bs=16 −8.1%, bs=32 −5.3%, bs=64 −4.4% | 2026-08-01 | campaign iter10 |
| TurboQuant Triton→CUDA port (bitwise vs Triton for the serving config; numeric pins lifted from Triton PTX) | speed-neutral (~1.2% at matched acceptance); enables triton-free opt-in KV memory saving (~22%/token); defaults unchanged | 2026-08-01 | campaign iter12 |

Campaign end state (2026-08-01, "Campaign closed" entry): final matrix TP4
66/138/204/273/326/392 vs 2x MI300X reference 82/141/176/260/297/408 — wins
at 8/16/32 conns, 98%/96% at 4/64, 80% at bs=1; TP8 beats the reference
everywhere.

## Rejected — with the reason, so they are not retried

**AR+norm fusion — REJECTED (2026-08-01, campaign iter3).** Correct (residual
bitwise, norm ≤6.0e-3 rel) but an end-to-end wash: bs1 +0.7%, bs8 +0.9%,
bs32 −0.7% against ≥5%/≥3% bars. Root cause: at decode sizes the one-shot
allreduce's ~20% step share is signal/sync LATENCY, not bandwidth or launch
count; the fused epilogue removes ~156 norm launches/step and none of the
latency. Generalized rule: **reducing AR cost needs FEWER reduces, not
fusion.** Code kept behind `VLLM_FUSED_AR_NORM=1`, default OFF.

**Expert-sorted vec MoE — REJECTED (2026-08-01, campaign iter6).**
0.95–1.03x: L2 already dedups expert weight reads and the vec path is
per-row compute-bound. Rule: do not sort for locality a 40 MB L2 already
provides.

**MoE micro-retunes at bs=1 — REJECTED (2026-08-01, campaign iter6).** MMQ at
bs=1 −2.7%; rows_per_block retune found the heuristic already optimal. The
NEGATIVE finding: bs=1 MoE is at a practical floor (~40.1 ms/step) — the
bs=1 gap needs the small-kernel floor or acceptance gains, not MoE
scheduling.

**MOE_X=8 tile — REJECTED (2026-08-01, campaign iter10).** Loser even with
padding free, and X=8/NW=4 is numerically INVALID (the y-scale loop covers
only nwarps columns — caught in the sweep). Rule: keep a correctness check
inside every config sweep.

**Spec depth k>3 at bs=1 — REJECTED (2026-08-01, campaign iter11).** ms/step
36.6/46.3/50.2/53.7 at k=3/4/5/6 vs acceptance ~2.4/2.9/2.8/2.8 — every k>3
loses net (mean tok/s 66/62/~57/53). k=3 confirmed optimal on A100. Rule:
the bs=1 gap requires step-time cuts, not more speculation.

**TurboQuant draft-KV as default — REJECTED (2026-08-01, campaign iter12).**
A/B lost ~6–7% mean, acceptance-driven (2.24 vs 2.42 at bs=1; 4k parity did
not hold on this Q2_K stack). Fail-closed: draft-KV default stays bf16. The
kernel port itself is speed-neutral; TQ's win is opt-in KV memory.

## Patterns and generalized rules

- Profile-first beats candidate lists: iter4's three prior hypotheses were
  all wrong, and iter9's candidate list was wrong again — profile/ncu
  attribution (e.g. LSU-bound padding compute) found the real cost both
  times. (2026-08-01, campaign iter4/iter9)
- Fill the 108 SMs before tuning anything else: fixed split-K partition
  spans left 16 warps on 108 SMs (iter4); ceil-semantics nsplit pushed
  tiles×nsplit past 108 SMs into a straggler wave (iter10). (2026-08-01,
  campaign iter4/iter10)
- Do not compute padding: moe_align's mmq_x=4 padding meant only 256/656
  tile columns were real at 256 routed rows. (2026-08-01, campaign iter9)
- Variance discipline: natural-stop single runs at bs=32/64 carry ±8–10%
  (iter5) and matrix-row single runs ~9% column variance at mid batch
  (iter2) — high-batch claims need mean-of-3; only cross-column shape
  claims need repeats. (2026-08-01, campaign iter2/iter5)
- Routing crossovers move after every structural fix: the MoE w1 vec→MMQ
  crossover went 64→16 (iter6) → 8 once padding was free (iter9).
  Re-measure crossovers after any kernel win. (2026-08-01, campaign
  iter6/iter9)
- FP32:bandwidth fell ~4x moving 3090→A100 (38 → 9.6 FLOP/byte): any
  scalar/FP32-bound dequant path sits ~4x further from roofline here, so
  i-quant lookup work and fused dequant-in-pipeline GEMM became MORE
  valuable, not less. Source: perf.md Host Reality (2026-07-30).
- Tensor:bandwidth doubled moving 3090→A100 (76 → 153 FLOP/byte):
  tensor-core paths have more room to hide memory. Source: perf.md Host
  Reality (2026-07-30).
- No 3090-derived threshold survives the ratio shift: `qgemm_ksplit`'s
  `tiles < 832`, the mmvq/mmq crossovers, and every tile config need
  re-deriving on A100, not rescaling. Source: perf.md Host Reality
  (2026-07-30); a100_glm52_design.md §6 "what not to do".

## Open contradictions

- perf.md Host Reality (2026-07-30) says "no performance baseline exists on
  this host" — yet the 2026-08-01 campaign recorded extensive A100 serving
  numbers. Both are true at different layers: the campaign measured the
  SlimServe serving stack; the kernel-harness baselines (29-format qgemv
  sweep, MetalForge table, `qgemm_ksplit` threshold) still do not exist on
  A100. Resolved by: backlog family 1's re-baseline run.
- a100_glm52_design.md §2.2 probe predicted 80–88% end-to-end for the Q2_K
  integer route; the landed kernel (§2.5b) measured 44% at M=1 and is
  compute-bound from M≥2. The doc self-corrects. Resolved by: measuring the
  IMMA m16n8k32 path (design §6 item 3), which the corrected analysis names
  as the M≥2 answer.
- baseline_status.md's "Hardware Gap" and old Per-Kernel Status Table
  claimed only layernorm executes on Ampere, contradicted by the Ampere
  Port Status table in the same file. Resolved 2026-08-15 by the
  baseline_status.md restructure (perf.md Open Item #3) — no measurement
  needed.
- perf.md Recording Format still says "optimization_status.md currently has
  zero per-kernel entries", but the notebook now holds the 2026-08-01
  campaign. Doc staleness; resolved by a perf.md edit, no measurement
  needed.
- `qgemm_ksplit` split threshold: 3090-derived `tiles < 832` (≈10 waves ×
  82 SMs); the naive A100 rescale is ≈1100, but perf.md warns the 40 MB L2
  and shifted compute:bandwidth ratio may move the crossover independently
  of SM count. Resolved by: an M/tiles crossover sweep on an idle A100.
