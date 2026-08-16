# QuixiCore CUDA Baseline Status

Baseline snapshots for CUDA kernel performance work. Method and harness are
described in `perf/perf.md`; distilled established results live in
`perf/findings.md`; the active experiment queue is `perf/backlog.md`; ongoing
optimization decisions live in `perf/optimization_status.md`. Raw results
live under `perf/results/` (git-ignored — dead on any clone; copy summaries
into the notebook instead of citing run paths as evidence).

Restructured 2026-08-15 to implement `perf/perf.md` Open Item #3: the current
host is the A100 box below, and everything 3090-era or pre-port moved
verbatim (headings demoted one level, bodies unchanged) to
[Superseded (historical)](#superseded-historical). Nothing was deleted and no
numbers were added or changed in the move.

## Environment

Current validation host, from `perf/perf.md` Host Reality (2026-07-30):

- 8x NVIDIA A100-SXM4-80GB (GA100, **SM 8.0**), 80 GB HBM2e each
- Driver 595.45.04, CUDA toolkit 13.2, PyTorch 2.13.0+cu130
- 108 SMs; 40 MB L2; 164 KB smem/SM (163 KB max dynamic per block);
  65536 regs/block; 2048 threads/SM
- NVLink: full NV12 mesh between all 8 GPUs (~600 GB/s bidirectional per
  GPU). `kernels/parallel/` stays blocked on `multimem` (SM90+), not on the
  fabric.
- No `.venv` in this repo — point at an interpreter with torch explicitly.
- OS / kernel version: TBD (record on next A100 session)
- Repo commit for the 2026-07-30 validation: TBD (record on next A100 session)

The 3090-era numbers in Superseded below are a different target's results
(RTX 3090, SM 8.6), not this host's baseline.

## Build + gate (the standing invariant)

- **Arch flags are host-dependent.** `-gencode arch=compute_80,code=sm_80`
  on A100, `compute_86,code=sm_86` on the 3090, or a multi-arch list.
  `code=sm_86` binaries fail to launch on SM 8.0.
  `kernels/tm_cuda/{setup.py,build_ext.sh}` and the per-kernel harness
  headers still hard-code `sm_86`; parameterizing them is `perf/backlog.md`
  family 1.
- The sm_80 build is proven sufficient: swapping the arch flag was the only
  change needed for the 9-TU extension to build and run on this host
  (2026-07-30) — 70/70 tk-independent tests green, all 117 bound ops load.
- The end-to-end gate is bare pytest (13 test files), with torch's `lib/`
  on `LD_LIBRARY_PATH`:

  ```bash
  cd kernels/tm_cuda && LD_LIBRARY_PATH=<torch>/lib \
    CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q
  ```

  Until the `tk` module is restored (perf.md Open Item #9), the 3
  tk-dependent files cannot even collect; the interim gate is the 10-file
  explicit list in perf.md "Build And Correctness".
- For `include/` substrate changes, also run `make -C tests` and the
  `perf/tools/sm80_*_smoke.cu` primitive smokes.

## Kernel roofline snapshot (dated)

Measured rooflines on one idle A100 (2026-07-30, CUDA events, warmed) — from
perf.md Host Reality:

| metric | measured | spec | % of spec |
|---|--:|--:|--:|
| DRAM copy (read+write) | **1750 GB/s** | 2039 | 86% |
| DRAM read (reduction) | 1655 GB/s | 2039 | 81% |
| cuBLAS bf16 GEMM 8192³ | **239 TFLOP/s** | 312 | 77% |
| cuBLAS bf16 GEMM 4096³ | 218 TFLOP/s | 312 | 70% |
| int8 tensor GEMM 8192³ | 353 TOP/s | 624 | 57% |
| TF32 GEMM 8192³ | 141 TFLOP/s | 156 | 90% |
| FP32 GEMM 8192³ | 19.1 TFLOP/s | 19.5 | 98% |

Framework baselines on one idle A100 (2026-07-30, same source):

| op | shape | measured |
|---|---|--:|
| SDPA (flash, causal, bf16) | B4 H32 N4096 D128 | 153 TFLOP/s |
| SDPA (flash, causal, bf16) | B8 H32 N2048 D128 | 142 TFLOP/s |
| SDPA (flash, causal, bf16) | B8 H32 N2048 D64 | 128 TFLOP/s |
| `rms_norm` bf16 | 65536×4096 | 1504 GB/s |
| `rms_norm` bf16 | 16384×4096 | 1392 GB/s |
| `silu` bf16 | 65536×4096 | 1687 GB/s |
| `softmax` bf16 | 65536×4096 | 742 GB/s |

End-to-end serving snapshot — the latest campaign matrix (earlier matrices
are in Superseded). Method as recorded 2026-08-01: real varied prompts,
temp 0, natural stops, tokens/drain-time, max_num_seqs 64, fp8 KV, DSpark
k=3 draft-TP1, CUDA graphs, no Triton.

### A100 matrix — 2026-08-01 post iter10 (mean-of-3 at 1/32/64)
| conns | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| 4x TP4 | 65.7 | 138.3 | 204.0 | 272.5 | 325.6 | 391.7 |
| 8x TP8 | 80.6 | 166.7 | 304.8 | 333.1 | 550.0 | 626.4 |
TP4 beats reference at 8/16/32; 98% at bs=4, 96% at bs=64, 80% at bs=1.

The campaign-final README matrix (SlimServe 63e639171, recorded in the
2026-08-01 "Campaign closed" notebook entry) matches this within rounding:
TP4 66/138/204/273/326/392 vs 2x MI300X reference 82/141/176/260/297/408.

## Per-family status

Status on this host (A100, SM 8.0). "TBD" means not yet measured here —
see `perf/backlog.md` family 1.

| family | status on A100 | source |
|---|---|---|
| tm_cuda extension (quant, serving, moe_quant, elementwise, lin_attn_tm, moe) | correct: 70/70 tk-independent tests, 117 ops load (2026-07-30); kernel-harness perf baseline TBD | perf.md Host Reality |
| Serving decode path (MLA fp8, split-K, GGUF GEMV/MMQ routing, MoE Q2_K/Q8_0) | optimized end-to-end through the 2026-08-01 campaign (iters 1–12); wins and rejections distilled in `perf/findings.md` | optimization_status.md 2026-08-01 |
| q2k_ampere (integer/dp4a route, new) | landed: repack bit-exact, GEMV 0.61% rel; M=1 44% of ceiling, compute-bound from M≥2 | a100_glm52_design.md §2.5b |
| Standalone TK kernels (layernorm, mha_ampere, gemm int8/fp8_ampere, rotary, flux, based, hedgehog, linear_attention, mamba2, fftconv_non_pc) | SM86-validated (see Superseded); A100 runtime numbers TBD | Superseded: Ampere (SM86) Port Status |
| fftconv_pc | newly runnable here (163 KB smem vs its 112 KB need); not yet run — TBD | perf.md Open Item #4 |
| SM90+/Blackwell kernels (mha_h100*, gemm/*_b200, bf16_b300*) | build-only on this host (unchanged) | perf.md |
| parallel/ (13 kernels) | capability_gated: NVLink fabric now present, `multimem` still SM90+ | perf.md Host Reality; backlog_parallel.md |

## Deferred (bigger projects, flagged not faked)

Not scheduled on the beam; recorded here so absence of numbers is never
mistaken for measurement. Details in `perf/backlog.md` "Parked".

- A100 kernel-harness re-baseline (29-format qgemv sweep, MetalForge
  serving table, `qgemm_ksplit` threshold) — until it lands, per-kernel
  A100 numbers are TBD, not inferred from the 3090.
- Restore/vendor the `tk` module (perf.md Open Item #9) — unblocks the full
  pytest gate, golden regeneration, and `bench_vs_torch.py`.
- Register the port families in the `bench_kernels.py` registry (Open Item
  #5) — only 4/45 entries runnable today.
- `shared_to_register.cuh` audit on H100 + int shared<->register unit tests
  (Open Item #8).
- Multi-GPU BAR1-P2P + `parallel/` track — parked with its own resume doc,
  `backlog_parallel.md` (3090 box).

## Decision log

- 2026-07-03: Added `perf/` harness (`bench_kernels.py`, `tk_bench_layernorm.py`,
  `bench_reference.py`, `perf.md`, this file). `perf/results/` git-ignored.
- 2026-07-03: Fixed `include/pyutils/torchutils.cuh` to guard PGL /
  `TKParallelTensor` machinery behind `KITTENS_SM90/SM10X/SM120` (it broke every
  SM80 PyTorch-extension build) and to include `<torch/csrc/utils/pybind.h>`
  directly (was inherited transitively via `parallel_tensor.cuh`). SM90+ builds
  verified unaffected (all pytorch-config kernels rebuilt cleanly after the
  change).
- 2026-07-03: Fixed missing `#include <chrono>` in
  `kernels/attention/mha_h100_lcf/mha_h100_lcf.cu` (pre-existing build failure).
- 2026-07-03: Rebuilding arch-portable kernels with `ARCH=SM80` for execution on
  Ampere is done automatically by the harness (`sm80_ok` registry flag).
- 2026-08-15: Restructured this file per perf.md Open Item #3 — added the
  A100 host block; moved the Hardware Gap section, the old Per-Kernel Status
  Table, the 3090-era baselines, the earlier post-iterN matrices, and the
  Run Index verbatim to Superseded. Added `perf/findings.md` (distilled
  results) and `perf/backlog.md` (experiment queue); the notebook's
  2026-07-06 backlog entry is annotated as superseded-as-a-queue. No new
  measurements.

<a name="superseded-historical"></a>
## Superseded (historical)

Everything below is moved, not deleted: headings demoted one level, bodies
verbatim, each with a reason line. Do not cite these as this host's current
baseline.

**Reason: 2026-07-03 RTX 3090 host block — applies only to the 3090-era
baselines below; the current host block is the Environment section above.**

### Environment (all numbers below)

- Host: 8x NVIDIA GeForce RTX 3090 (SM 8.6 / Ampere, 24 GB), driver 580.65.06
- CUDA toolkit 12.9 (`/usr/local/cuda`), PyTorch 2.12.1+cu130, Python 3.12.3
- Repo: commit `02e9acbd` + perf harness + two small fixes (see Decision Log)
- Date: 2026-07-03

**Reason: pre-dates the Ampere port — perf.md Open Items #3. Contradicted by
the "Ampere (SM86) Port Status" table below, which is the correct record.**

### The Hardware Gap (read first)

Every kernel in this repo except `layernorm` and the cuBLAS baseline programs
uses TMA and/or WGMMA/tcgen05, which require SM 9.0+ (H100) or SM 10.x (B200/
B300). The multi-GPU `parallel/` kernels additionally use `multimem`. **None of
these can execute on SM 8.6 hardware.** The runtime baseline on this host is
therefore: `layernorm` (rebuilt for SM80) plus cuBLAS reference GEMMs plus
PyTorch framework references. Every other kernel has a compile-only baseline:
build health and ptxas register/spill statistics for its declared arch.
Completing runtime baselines requires an H100/B200/B300 host; the harness
(`perf/bench_kernels.py`) will pick up the runnable set automatically there —
the standalone GEMM/attention kernels self-benchmark and their output is
captured/parsed by the run phase.

**Reason: 3090-era build baseline (commit `02e9acbd`, 2026-07-03). Kept:
SM86 is a declared target.**

### Build Baseline — 42/42 kernels compile cleanly

Run: `perf/results/2026-07-03/baseline-build` (+ `lcf-rebuild`). Highlights,
worst kernel per directory (`regs` = max registers across entry points,
`spill` = total spill bytes reported by ptxas):

| kernel | arch | regs | spill B | note |
|---|---|---|---|---|
| attention/mha_h100 | SM90 | 252 | 1464 | spills: backward pass candidates |
| attention/mha_h100_lcf | SM90 | 254 | 0 | fixed missing `<chrono>` include |
| attention/bf16_b300_mha_causal | SM103 | 128 | 640 | |
| attention/bf16_b300_mha_noncausal | SM103 | 128 | 96 | |
| based | SM90 | 255 | 0 | at register ceiling |
| fftconv | SM90 | 168 | 760 | |
| flux | SM90 | 232 | 1488 | largest spill count |
| hedgehog | SM90 | 252 | 0 | |
| layernorm | SM80 | 96 | 0 | built SM80 to run on this host |
| linear_attention | SM90 | 255 | 0 | at register ceiling |
| mamba2 | SM90 | 168 | 0 | |
| rotary | SM90 | 168 | 144 | |
| gemm/bf16_h100, fp8_h100, fp8_h100_scaled | SM90 | 168 | 0 | |
| gemm/int8_h100 | SM90 | 168 | 32 | |
| gemm/bf16_b200, fp8_b200 | SM100 | 255 | 0 | |
| gemm/int8_b200 | SM100 | 255 | 1752 | largest GEMM spill |
| gemm/mxfp8_b200 | SM100 | 186 | 0 | |
| gemm/nvfp4_b200 | SM100 | 174 | 0 | |
| gemm/educational_{h100,b200} | SM90/100 | 32/30 | 0 | level 01 only |
| gemm/baselines/* (cuBLAS/Lt) | SM80/SM100 | ~30 | 0 | bf16+int8 rebuilt SM80 |
| parallel/* (13 kernels) | SM90/SM100 | 8–255 | ≤16 | all compile |

Full per-kernel table: `perf/results/2026-07-03/baseline-build/results.jsonl`
and `summary.md` there.

**Reason: 3090-era runtime baseline. Kept: SM86 is a declared target.**

### Runtime Baseline — layernorm (the one executable TK kernel)

`fused_layernorm` = dropout + residual-add + LayerNorm, bf16, `d_model=1024`
(hard-coded), seq divisible by 16. Built `ARCH=SM80`, run on one RTX 3090.
Benchmark: `perf/tk_bench_layernorm.py` (CUDA events, 10 warmup, 50 iters x 3
repeats, median). Archived: `perf/results/2026-07-03/baseline-run/`.

Correctness (dropout_p=0 deterministic path): max abs err out = 0.052,
resid = 0.016 vs fp32 reference — passes bf16 tolerance 0.15.

Shapes `b=16, d=1024`, dropout_p=0.1; GB/s assumes 8 B/element moved:

| n | TK ms | TK GB/s | torch eager ms | triton fused ms | TK vs torch | TK vs triton |
|---|---|---|---|---|---|---|
| 1024 | 0.258 | 520 | 0.814 | 0.256 | 3.2x | 0.99x |
| 2048 | 0.482 | 557 | 1.972 | 0.510 | 4.1x | 1.06x |
| 4096 | 1.183 | 454 | 4.001 | 1.166 | 3.4x | 0.99x |
| 8192 | 2.387 | 450 | 8.332 | 2.391 | 3.5x | 1.00x |
| 16384 | 4.766 | 451 | 16.653 | 4.605 | 3.5x | 0.97x |

Reading: TK fused layernorm is ~3.2–4.1x faster than the eager PyTorch
composite and statistically tied with the FlashAttention Triton fused kernel
(±3%). At ~450–557 GB/s it reaches 48–60% of the 3090's 936 GB/s DRAM peak —
plausible for a dropout+curand+two-output kernel, and a reasonable
optimization target to revisit on real target hardware rather than here.

Note: in `baseline-run-parallel` (three GEMM sweeps running on neighboring
GPUs simultaneously) the same kernel read 381–418 GB/s — a ~10–15% haircut
from shared power/host contention. The isolated `layernorm-run` numbers above
are canonical; the relative ordering (TK ≈ triton >> torch eager) is identical
in both runs.

**Reason: 3090-era cuBLAS reference. Kept: SM86 is a declared target. The
A100 cuBLAS references are in the roofline snapshot above.**

### Reference GEMM Baseline — cuBLAS on RTX 3090

`gemm/baselines/{bf16_cublas,bf16_cublas_lt,int8_cublas_lt}` rebuilt
`ARCH=SM80`, 500 warmup / 100 iters each. Canonical numbers below are from
`baseline-run-parallel` (each benchmark alone on an idle GPU):

| M=N=K | bf16 cuBLAS TFLOP/s | bf16 cuBLASLt TFLOP/s | int8 cuBLASLt TOP/s |
|---|---|---|---|
| 1024 | 46.5 | 46.8 | 102.1 |
| 2048 | 49.2 | 49.7 | 88.5 |
| 4096 | 54.8 | 54.7 | 87.6 |
| 8192 | 57.5 | 57.4 | 125.8 |
| 16384 | 56.4 | 56.3 | 122.3 |

Peak observed: bf16 ~57.5 TFLOP/s (~81% of the 3090's ~71 TFLOP/s dense bf16
peak); int8 ~126 TOP/s (~89% of the ~142 TOP/s dense int8 peak). cuBLAS and
cuBLASLt are equivalent for bf16 on this device.

Measurement-hygiene lesson (kept because it will bite again): the first,
serial run executed all four benchmarks back-to-back on GPU 0 and produced
wild swings — bf16 cuBLAS at 16384^3 read 56.0 then 32.3 TFLOP/s across runs,
and cuBLASLt read 15–30 TFLOP/s at small sizes — pure thermal/clock and
contention artifacts, not library behavior. Judge large-size numbers only
from an idle, cool GPU, and re-measure anything surprising before believing
it.

**Reason: 3090-era framework reference. Kept: SM86 is a declared target. The
A100 framework references are in the roofline snapshot above.**

### Framework Reference — PyTorch on RTX 3090

`perf/bench_reference.py`, archived at
`perf/results/reference/rtx3090_reference.json`.

- bf16 `torch.matmul`: 45–60 TFLOP/s on square 1024–16384; LLM projection
  shapes (`M=2048, N=11008/14336, K=4096`) 55–56 TFLOP/s; small-M decode
  (`M=16`) drops to ~9 TFLOP/s (memory-bound).
- bf16 SDPA (flash backend): 49–55 TFLOP/s non-causal at N>=1024 (D=64 and
  128); causal 33–48 TFLOP/s.
- bf16 `nn.LayerNorm` (norm only, 4 B/element): 705–785 GB/s at n<=2048,
  ~318–338 GB/s at n>=4096.

These are the numbers TK kernels must beat (or match with better fusion) on
this device class.

**Reason: pre-dates the A100 port — perf.md Open Items #3.**

### Per-Kernel Status Table

Status legend: `build-only` = compiles for declared arch, cannot execute here;
`baselined` = runtime numbers recorded on this host.

| kernel | status | runtime blocker |
|---|---|---|
| layernorm | **baselined** | — |
| gemm/baselines/bf16_cublas | **baselined** | — |
| gemm/baselines/bf16_cublas_lt | **baselined** | — |
| gemm/baselines/int8_cublas_lt | **baselined** (int8 IMMA works on Ampere) | — |
| attention/* | build-only | TMA/WGMMA (SM90) or SM103 |
| based, hedgehog, linear_attention, mamba2 | build-only | TMA/WGMMA (SM90) |
| fftconv, flux, rotary | build-only | TMA/WGMMA (SM90) |
| gemm/*_h100 | build-only | TMA/WGMMA (SM90) |
| gemm/*_b200 | build-only | tcgen05/TMA (SM100) |
| gemm/baselines/{fp8,mxfp8,nvfp4}_cublas_lt | build-only | fp8/fp4 hw (SM89/SM100+) |
| parallel/* (13) | build-only | multimem + TMA (SM90+), NVLink fabric |

**Reason: 3090 (SM 8.6) numbers — a different target's results, superseded
as this host's baseline. Kept: SM86 is a declared target and this table
remains the port's acceptance evidence.**

### Ampere (SM86) Port Status — 2026-07-03

The SM90+ kernels are being ported to run on this box's 3090s (plan:
`~/.claude/plans/all-the-kernels-that-serene-storm.md`). Library groundwork
(KITTENS_SM86 target, cp.async producer primitives, WGMMA emulation in
`include/ops/group/mma/warpgroup_sm80.cuh`) is landed and validated; SM90
compilation output is unchanged (PTX-verified). Validation tools live in
`perf/tools/` (sm80_*_smoke.cu).

| kernel | SM86 status | numbers (3090; * = under external GPU contention) |
|---|---|---|
| rotary (same-source) | CORRECT | IDLE GPUs: peak 763 GB/s (d64 n4096), 480–600 GB/s other shapes |
| attention/mha_ampere fwd (new dir; full mha_h100 parity) | CORRECT: causal+GQA+L, D=64/128 (o err <=1.1e-3, L err <=6e-4 vs fp32 ref) | IDLE GPUs: 48.7 TFLOP/s D=64 non-causal harness |
| attention/mha_ampere bwd (NEW kernel, no Ampere ancestor) | CORRECT: dQ/dK/dV <=7.5e-4 max vs torch fp32 autograd, all configs (D 64/128 x causal x GQA) | untuned; D=128 spills (1 wg/block) |
| gemm/bf16_h100 (same-source, M2N2 stages3) | CORRECT | IDLE GPUs: 44.0 TFLOP/s @4096³ = 80% cuBLAS (54.9) |
| flux/flux_gate (same-source, 128x128x64) | CORRECT (err count 0) | IDLE GPUs: 44.8 TFLOP/s |
| flux/flux_gelu (same-source) | builds; run pending | — |
| gemm/int8_ampere (new dir) | EXACT (err 0) | IDLE, tuned: 50.4 @4096³ / 51.5 @8192³ TOP/s (per-size configs: staged-writeback Nb=64 small, direct-register-writeback Nb=128 large; was 41/36) vs cuBLASLt 85.6/123 |
| mamba2 (same-source, NWG=1 on Ampere) | CORRECT | 0.39% mean rel err vs fp32 ssd_minimal |
| fftconv_non_pc (same-source + chrono fix) | PASSED (0/16.7M violations) | 15.1 ms* full B=4 H=1024 N=4096 |
| fftconv_pc | A100-only (112KB scratch > 99KB) | untested — no A100 on this box |
| flux/flux_gelu (same-source) | CORRECT (err count 0) | IDLE GPUs: 47.3 TFLOP/s |
| based/lin_attn_ampere (resurrected 50ee1f0a 4090 kernel) | CORRECT: randn max 0.164 / avg 0.0149 = exactly the bf16-state precision floor (validated vs exact-precision emulation) | 17.2 TFLOP/s* B16 H16 N1024 |
| linear_attention_ampere (new copy: single-buffered, no full-k tile, merged o store, 96.5KB) | CORRECT first run: rel_diff 0.127%, max 0.0078 | 209 us* B1 H8 N1024 D=F=128 |
| hedgehog_ampere (new copy: 1-slot k/v with cross-block carry, aliased kv/scratch staging, 97.5KB) | CORRECT first run (randn): O rel 0.81%, KV state rel 1.0%, K state rel 0.18% | 517 us* B2 H2 N1024 |
| gemm/fp8_ampere (new dir: fp8e4m3 STORAGE, Marlin bit-trick dequant to fp16 in registers, fp16 mma) | EXACT (max err 0 vs fp32 ref) | IDLE: 23.1 TFLOP/s @8192³ (fast dequant = 5x over __nv_cvt) |
| gemm/fp8_ampere scaled (per-row/col scales, port of fp8_h100_scaled) | PASS (max err 1.5e-5) | IDLE: 23.7 TFLOP/s @8192³ |
| BAR1 P2P driver (.reference/open-gpu-kernel-modules-580.65.06, tinygrad 9e39420bc4cb adapted to 580: 4 files) | all 5 .ko BUILT for 6.8.0-110-generic | awaiting user install: BIOS ReBAR (BAR1 now 256MiB) + sudo; see README-BAR1-P2P.md |
| unit tests ARCH=SM86 | 2467/2468 | 1 pre-existing warp complex-mma fail |

Upstream bugs found & fixed during the port (affect SM90 correctness too
where noted):
1. `kernels/rotary/rotary.cu`: sin/cos loaded with an element-typed
   `kittens::coord` (reads rows w..w+15 instead of 16w..) — wrong on all
   arches; the shipped test_correctness.py's large "tk diff" was this.
2. `include/ops/thread/util/util.cuh`: `move<int8_4>/<uint8_4>` (ldmatrix
   wrappers) were inside the fp8 SM90+ guard; int8 is an SM80 feature.
3. `include/ops/group/memory/tile/shared_to_register.cuh`: the 32-bit
   row-major register<->shared fast path (hand-rolled swizzle+blit
   addressing) writes addresses inconsistent with `st::idx()` (proven with a
   pattern-fill test on SM86: element (0,28) landed at (1,0)). SM80 family
   now uses canonical idx()-based accesses; SM90 keeps the fast path.
   NOTE: worth auditing on H100 too — unit tests don't cover int
   shared<->register.
4. Missing `#include <chrono>` in mha_h100_lcf.cu, the 4090 harness, and
   flux_gate.cu (RUN_MAIN paths).
5. based test harness (50ee1f0a `lin_attn_4090_harness.impl`, now
   `harness_ampere.impl`): head-replication used modulus `ATTN_B*ATTN_H`
   (=256) instead of elements-per-head, so the device saw v rows 0-3 tiled
   everywhere; and the check indexed `o_ref[i]` out of bounds for heads > 0
   (o_ref holds one head). The old "based is broken" impressions likely trace
   here — the kernel matches an exact-precision emulation to 7.8e-3 max.
6. based test generator (`generate_tests_ampere.py`, from the old repo):
   `pytorch_test(q, k, v, TESTNAME)` passes TESTNAME positionally into
   `add_scale`, forcing the scaled reference (and making every per-term test
   file identical to all-terms). The kernel computes the UNSCALED convention;
   fixed by calling with `add_scale=False, TESTNAME=TESTNAME`.

**Reason: run dirs are git-ignored and dead on clones — superseded as
evidence pointers; conclusions live in the notebook.**

### Run Index

| run | contents |
|---|---|
| `perf/results/2026-07-03/baseline-build/` | build phase, all 42 kernels, ptxas stats |
| `perf/results/2026-07-03/lcf-rebuild/` | mha_h100_lcf rebuild after chrono fix |
| `perf/results/2026-07-03/baseline-run/` | serial run phase (superseded; thermal artifacts) |
| `perf/results/2026-07-03/baseline-run-parallel/` | **canonical run**: one benchmark per idle GPU (4 ok / 38 skip) |
| `perf/results/2026-07-03/int8-rerun/` | int8_cublas_lt re-record with TOP/s parsing |
| `perf/results/2026-07-03/layernorm-run/` | first archived layernorm run |
| `perf/results/reference/rtx3090_reference.json` | torch matmul/SDPA/layernorm sweep |

**Reason: 2026-07-03 questions. The canonical-host question is answered for
SM80 by the A100 host; still open for SM90/SM100 (tracked in perf.md Open
Items).**

### Open Questions

- Which target device should runtime baselines be recorded on for the
  SM90/SM100 kernels — is an H100/B200 host available to this project?
- The educational GEMM levels 02–08 are compile-checked at level 01 only;
  sweep them if they become optimization targets.
- `parallel/` kernels additionally need a multi-GPU fabric with multicast
  support; 8x 3090 (PCIe/pairwise NVLink) cannot validate them even if the
  arch gap were closed.

**Reason: superseded by the post-iter10 snapshot above; kept for
cross-iteration history.**

### A100 end-to-end matrix — 2026-08-01, post vectorized-MLA (agg tok/s)
Method: real varied prompts, temp 0, natural stops, tokens/drain-time,
max_num_seqs 64, fp8 KV, DSpark k=3 draft-TP1, CUDA graphs, no Triton.
| conns | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| 4x A100 TP4 | 47.8 | 94.5 | 147.6 | 165.8 | 277.2 | 312.3 |
| 8x A100 TP8 | 57.9 | 125.2 | 214.2 | 264.4 | 418.0 | 514.7 |
| 2x MI300X (ref) | 82 | 141 | 176 | 260 | 297 | 408 |
TP8 beats the 2x MI300X reference at 8/16/32/64. TP4 bs=16 dent (165.8,
barely above bs=8) flagged as a scheduling anomaly.

**Reason: superseded by the post-iter10 snapshot above; kept for
cross-iteration history.**

### A100 matrix — 2026-08-01 post iter4 (balanced split-K + GEMV tile)
| conns | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| 4x TP4 | 68.2 | 125.3 | 163.1 | 184.1 | 267.5 | 263.6 |
| 8x TP8 | 68.5 | 140.0 | 248.7 | 302.0 | 368.3 | 548.8 |
NOTE: high-batch columns swing +/-12-15% between full runs (TP8@32: 418 then
368; TP4@64: 312 then 264) -- single-run numbers at 32/64 are not stable
enough to attribute to code changes without repeats. Filed as lead [l].

**Reason: superseded by the post-iter10 snapshot above; kept for
cross-iteration history.**

### A100 matrix — 2026-08-01 post iters 6+7 (mean-of-3 at bs 1/32/64)
| conns | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| 4x TP4 | 66.2 | 110.9 | 183.3 | 239.5 | 275.1 | 307.7 |
| 8x TP8 | 84.6 | 166.6 | 276.2 | 315.5 | 464.3 | 562.3 |
| 2x MI300X ref | 82 | 141 | 176 | 260 | 297 | 408 |
TP8 beats the reference at EVERY column (incl. bs=1). TP4 beats it at bs=8;
remaining TP4 gaps: 81%/79%/-/92%/93%/75%.

**Reason: superseded by the post-iter10 snapshot above; kept for
cross-iteration history.**

### A100 matrix — 2026-08-01 post iter8 (mean-of-3 at 1/32/64)
| conns | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| 4x TP4 | 73.1 | 110.0 | 140.6 | 201.7 | 299.6 | 360.7 |
| 8x TP8 | 79.1 | 141.4 | 286.1 | 258.5 | 494.5 | 552.4 |
TP4 beats the 297 target at bs=32; bs=64 361 vs 408. CAVEAT: single-run
mid-batch cells (4/8/16) now swing +/-15-25% across matrices because iter8's
logit changes alter generation lengths (drain-efficiency artifact, not speed:
fixed-length guards within 2%). Mid-batch cells need fixed-workload or
mean-of-3 treatment before further cross-matrix claims.

**Reason: superseded by the post-iter10 snapshot above; kept for
cross-iteration history.**

### A100 matrix — 2026-08-01 post iter9 (mean-of-3 at 1/32/64)
| conns | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| 4x TP4 | 65.2 | 130.9 | 203.4 | 259.1 | 310.7 | 352.7 |
| 8x TP8 | 85.2 | 144.7 | 317.5 | 411.1 | 515.2 | 548.9 |
TP4 beats the reference at bs=8/32; bs=16 at 99.7% (iter9 ms/step -13%
corroborates). TP8 beats at every column. TP4 gaps: bs=1 79%, bs=4 93%,
bs=64 86%.
