# QuixiCore CUDA Performance Handbook

This is the operating guide for baselining and optimizing QuixiCore CUDA kernels.
It mirrors the hardware-independent discipline of the QuixiCore Metal and ROCm
handbooks while keeping the backend notes specific to NVIDIA CUDA.

The goal is not to collect tricks. Run a disciplined loop: find references, form a
bottleneck hypothesis, measure a clean baseline, run controlled experiments, keep
only verified wins, and record enough detail that the next pass starts from
evidence instead of memory.

- Method (this file): how to measure and decide.
- Results notebook: `perf/optimization_status.md` (per-kernel entries).
- Baseline snapshots: `perf/baseline_status.md`.

**Read this first if you are new to the tree:** the current validation host
changed from RTX 3090 (SM 8.6) to A100-SXM4-80GB (SM 8.0). Cubins built
`code=sm_86` **cannot launch on SM 8.0**, so every build recipe needs an explicit
arch for the host you are on. See [Host Reality](#host-reality) and
[Build And Correctness](#build-and-correctness). Also note that
`perf/baseline_status.md` lines 15–27 and its per-kernel status table are
**stale**: they pre-date the Ampere port (written 2026-07-03 against commit
`02e9acbd`, before the port landed) and claim nothing but `layernorm` can execute
on Ampere. That is contradicted 150 lines later in the same file by the
"Ampere (SM86) Port Status" table, which is the correct record.

## Principles

Optimization starts from correctness and measurement. A change is not a win until
it passes the kernel's correctness test (pytest and/or the standalone fp64/golden
harness), improves the target metric on realistic shapes, and does not regress a
supported edge shape or the numeric tolerance.

Numeric tolerances are contract-level, not per-kernel opinion — see
`registry/tolerances.yaml` in the umbrella repo (fp32 rtol 1e-5, fp16 1e-3,
bf16 2e-3, fp8 2e-2, quantized 3e-2; norms/softmax/quantization must be
deterministic, sampling may be stochastic). Exact-integer and RNG bit-parity
kernels have a stricter contract: exactness *is* the contract.

Attack a specific, named bottleneck:

- **Memory-bound** (most decode/elementwise/quant-GEMV): reduce bytes moved,
  coalesce global loads, exploit L2/constant-cache reuse, use narrower formats,
  kill extra global passes. Judge against the measured ~1750 GB/s achievable
  roofline, not the 2039 GB/s spec number.
- **Compute-bound** (prefill GEMM, large-M): raise arithmetic intensity, feed the
  `m16n8k16` tensor cores densely, cut scalar side work, fuse epilogues. Judge
  against measured cuBLAS (~218–239 TFLOP/s bf16), not the 312 spec number.
- **Latency-bound** (serial recurrences, tiny shapes): grow resident work, fix
  launch geometry, unroll serial loops, cut per-lane divergence.
- **Occupancy-bound** (decode kernels with 1 warp/block): more resident warps per
  SM — batch multiple rows/(head,batch) per block, split-K across the grid, shrink
  register/smem footprint. The chip is **108 SMs**; a grid of 128 warps leaves it
  nearly idle (this is what `qgemm_ksplit` fixed on the 3090 — its threshold needs
  re-deriving here, see below).
- **Sync/smem-bound**: remove needless `__syncthreads`, cut smem traffic and bank
  conflicts, prefer warp-shuffle reductions when cross-warp sharing does not pay.

## The Ampere Baseline Assumption

Do not blindly port H100/Blackwell machinery, and do not port Metal simdgroup or
ROCm MFMA/LDS idioms either. They can suggest experiments; they are not CUDA
design rules. Ampere kernels should be written and measured in terms of
Ampere-native mechanisms.

Ampere (SM 8.0/8.6) has **no `tma::`/tensor-maps and no warpgroup MMA** (SM90+),
and **no fp8/fp4/mxfp tensor units** (SM89/Blackwell). So:

- Async copy is `cp.async` (`cp.async.ca/cg`, `commit_group`/`wait_group`), not
  TMA. The `wgmma`/warpgroup path is emulated at warp scope — see
  `include/ops/group/mma/warpgroup_sm80.cuh`, which lowers each warp's 16-row
  slice to `mma.sync` (`m16n8k16`/`m16n8k32`) via `ldmatrix` and turns
  fence/commit/wait into no-ops. That header does **not** support fp8 operands.
- Tensor-core matmul is the `m16n8k16` fp16/bf16 `mma.sync`
  (`kernels/quant/tm_qmm.cuh::mma16816`, with `load_wfrag`/`load_xfrag`), fp32
  accumulate.
- Quantized compute rides that fp16 mma or native **`dp4a`/IMMA** int8 — the
  manifesto: *storage format is bits at rest; dequant is software in the latency
  shadow; compute rides fp16 mma / dp4a.* Never say "unsupported."
- Watch the register file (255 regs/thread hard cap, 65536 regs/block) and the
  smem budget. Spills are real on D=128 backward and the wide linear-attention
  kernels.

**SM 8.0 and SM 8.6 are not interchangeable.** This repo declares
`sm80, sm86, sm89, sm90, sm100` as targets (`.quixicore/backend.yaml`), and the
two Ampere tiers differ in ways that change tuning decisions *and* binary
compatibility:

| | SM 8.0 (A100, GA100) | SM 8.6 (RTX 3090, GA102) |
|---|---|---|
| smem per SM | **164 KB** (163 KB opt-in per block) | 100 KB (99 KB usable) |
| L2 | **40 MB** | 6 MB |
| FP32 | 19.5 TFLOP/s | **35.6** TFLOP/s (2× FP32 lanes) |
| BF16 tensor | **312** TFLOP/s | 71 TFLOP/s |
| DRAM | **2039 GB/s** HBM2e | 936 GB/s GDDR6X |

SASS compatibility is upward-only within a major version: `code=sm_86` binaries
**fail to launch on SM 8.0** with "no kernel image is available for execution on
the device." Always pass the arch for the host you are on, or a multi-arch
`-gencode` list.

**The consequence nobody has internalized yet:** A100 has 2.2× the bandwidth of
the 3090 but *less* FP32 throughput. The FP32-to-bandwidth ratio falls from
**38 to 9.6 FLOP/byte** — so any kernel whose dequant path is scalar/FP32-bound
sits **~4× further from roofline on A100 than it did on the 3090**. Meanwhile the
tensor-to-bandwidth ratio doubles (76 → 153 FLOP/byte), so tensor-core paths have
*more* room to hide memory. Net effect on priorities: the i-quant lookup problem
and the fused dequant-in-pipeline GEMM both become **more** valuable here, not
less. Re-measure before assuming any 3090-era ranking still holds.

**Known substrate trap** (found during the port): the 32-bit row-major
shared→register fast path in
`include/ops/group/memory/tile/shared_to_register.cuh` computes addresses
inconsistent with `st::idx()` on Ampere; the SM80 family uses plain `idx()` scalar
accesses (`:108-110`, store side `:282-283`). It was caught with a pattern fill
(element (0,28) landed at (1,0)). Don't "re-optimize" it back without re-verifying
that way. **Open risk:** per `baseline_status.md:212-213` this fast path is worth
auditing on H100 too, because the unit tests do not cover int shared↔register.

<a name="host-reality"></a>
## Host Reality

Regenerate this block rather than trusting it — it has rotted once already.

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
nvcc --version | tail -2
python3 -c "import torch;p=torch.cuda.get_device_properties(0);print(torch.__version__,p.name,p.major,p.minor,p.multi_processor_count)"
```

**Current validation host (2026-07-30):**

- **8× NVIDIA A100-SXM4-80GB** (GA100, **SM 8.0**), 80 GB HBM2e each,
  driver 595.45.04, CUDA toolkit **13.2**, PyTorch **2.13.0+cu130**.
- 108 SMs; 40 MB L2; 164 KB smem/SM (**163 KB** max dynamic per block);
  65536 regs/block; 2048 threads/SM.
- **NVLink present**: full NV12 mesh between all 8 GPUs (12 links × 25 GB/s per
  direction ≈ 600 GB/s bidirectional per GPU). This invalidates the old "no
  NVLink" note — but `kernels/parallel/` stays blocked because `multimem` is
  SM90+, not because the fabric is missing. The harness's `n_gpus < 8` gate now
  passes, so those entries will attempt to run; expect them to fail on the
  instruction, and keep them `capability_gated`.
- **There is no `.venv` in this repo.** Commands below use `python3`; if you want
  an interpreter with torch, point at one explicitly
  (e.g. `/home/ubuntu/SlimServe/.venv/bin/python`). Note
  `perf/bench_kernels.py:304` hard-codes `REPO_ROOT/.venv/bin/python` for torch
  metadata and will silently record `torch: null` until fixed.

**Measured rooflines on one idle A100** (2026-07-30, CUDA events, warmed):

| metric | measured | spec | % of spec |
|---|--:|--:|--:|
| DRAM copy (read+write) | **1750 GB/s** | 2039 | 86% |
| DRAM read (reduction) | 1655 GB/s | 2039 | 81% |
| cuBLAS bf16 GEMM 8192³ | **239 TFLOP/s** | 312 | 77% |
| cuBLAS bf16 GEMM 4096³ | 218 TFLOP/s | 312 | 70% |
| int8 tensor GEMM 8192³ | 353 TOP/s | 624 | 57% |
| TF32 GEMM 8192³ | 141 TFLOP/s | 156 | 90% |
| FP32 GEMM 8192³ | 19.1 TFLOP/s | 19.5 | 98% |

**Framework baselines on one idle A100** — these are the "framework baseline" leg
of the three-baseline rule below:

| op | shape | measured |
|---|---|--:|
| SDPA (flash, causal, bf16) | B4 H32 N4096 D128 | 153 TFLOP/s |
| SDPA (flash, causal, bf16) | B8 H32 N2048 D128 | 142 TFLOP/s |
| SDPA (flash, causal, bf16) | B8 H32 N2048 D64 | 128 TFLOP/s |
| `rms_norm` bf16 | 65536×4096 | 1504 GB/s |
| `rms_norm` bf16 | 16384×4096 | 1392 GB/s |
| `silu` bf16 | 65536×4096 | 1687 GB/s |
| `softmax` bf16 | 65536×4096 | 742 GB/s |

**Validation status on this host — first sm80 result, 2026-07-30:**

The port **builds and runs correctly on SM 8.0**. Verified by building the
9-TU extension with `-gencode arch=compute_80,code=sm_80` (the only change needed
was the arch flag) and running the `tk`-independent test modules:

```
70 passed in 3.41s     # test_{elementwise,gdn,m4,m5,m6,mf_followups,
                       #        moe_quant,norm_quant,selective_scan,w6}.py
```

All 117 bound ops load, and spot-checked kernels produce correct results on-device
(e.g. `rms_norm` within bf16 tolerance). So the SM86→SM80 move needs **no source
changes** — the `warpgroup_sm80.cuh` emulation, the `shared_to_register.cuh`
Ampere path, and the quant/serving/MoE families are all genuinely SM80-portable,
not just SM86-tuned.

Two gaps remain before this is a complete validation record:

- `test_tk_cuda.py`, `test_step4.py`, `test_serving.py` (the quant-format and
  serving suites) cannot run — they need the absent `tk` module, see
  [Build And Correctness](#build-and-correctness).
- **No performance baseline exists on this host.** Correctness ≠ perf: every
  number in this file is still 3090-measured. `perf/results/` contains only its
  `.gitignore`. See [Open Items](#open-items).

### Historical: RTX 3090 (SM 8.6) validation record

These numbers are real and were the port's acceptance evidence, measured on an
idle 3090 (GA102, 936 GB/s, 82 SMs, driver 580.65.06, CUDA 12.9). **Treat them as
a different target's results, not as this host's baseline.** Keep them: SM 8.6 is
a declared target, and they are the only end-to-end validation the port has.

On that host the full port was SM86-native and green: 29 quant formats bit-exact;
qgemv/qgemm/qflux/lm_head; the serving stack (paged v1/v2, MLA, rope_kv, attn_q,
varlen, window, beam, spec); elementwise/norm/training; MoE; linear attention; and
the MetalForge ports (quantized-MoE GEMMs, norm/activation quant, GDN, varlen
Mamba scan, sampler zoo, EAGLE). Unit tests 2467/2468 at `ARCH=SM86`. Only the
SM90+/Blackwell-native kernels (`attention/mha_h100*`, `gemm/*_b200`,
`attention/bf16_b300*`, TMA/wgmma demos) were build-only.

Canonical idle-3090 figures (perf floor to beat on that target, not a ceiling):

- Weight-only quant **GEMV** (N=512, K=4096): q8_0 **283 GB/s**, e5m2 288, fp8
  260, mxfp8 254, fp8_block 252, q6_K 212, mxfp6 201, q5_0 171, q4_1 160,
  q4_0/q4_K 156/152, nvfp4 141, mxfp4 136, kU4 137, hqq 145, bitnet/q2_K 91,
  iq4_nl 71, q3_K/iq4_xs 59, iq3_xxs 48, iq2_xxs 36, iq2_xs/iq1_s ~15.
- Quant **GEMM** decode shape (M=64,N=512,K=4096) via `qgemm_ksplit`: q8_0/q4_0
  ~7–8, nvfp4/mxfp4 ~7 TFLOP/s (~2.4× the single-pass tiled kernel).
- Quant **GEMM** prefill (M≥64) routes through dequant-to-fp16 + cuBLAS:
  **~28–30 TFLOP/s** end-to-end vs ~7–9 for the naive per-tile kernel.
- Dense **bf16 GEMM** 44.0 TFLOP/s @4096³ (80% cuBLAS); **mha_ampere** fwd 48.7
  TFLOP/s; **flux** 44.8/47.3; **int8** 50–51 TOP/s; **rotary** peak 763 GB/s;
  **W8A8** 528 GB/s (native dp4a).

MetalForge serving-kernel baselines (idle 3090, 2026-07-05, `bench_metalforge.py`):

| kernel | shape | measured | note |
|---|---|--:|---|
| `moe_gemm_fp8` | E=8, N=K=4096, 2048 rows | **10.5 TFLOP/s** | per-tile e4m3 dequant in the mma path |
| `moe_gemm_wna16` int4 | same | **7.0 TFLOP/s** | uint32 de-interleave + per-group scale/zp |
| `moe_gemm_wna16` int8 | same | **5.7 TFLOP/s** | |
| `gdn_linear_attention` | 128 seqs, GQA 2/8, D=128, decode | **345k tok/s** · 360 GB/s state | one warp per (req,hv,dv); state-bandwidth-bound |
| `selective_scan_varlen` | dim=2048, dstate=16, one 2048-tok seq | **21 GB/s** | serial recurrence, latency-bound |
| `rms_norm_quant` (fp8 dyn) | 8192×4096 | **304 GB/s** in | multi-warp block/row + folded amax |
| `silu_and_mul_quant` (fp8) | 8192×4096 | **374 GB/s** in | |
| `merge_attn_states` | 8192 tok × 8 heads × 128 | **415 GB/s** | 2-way LSE combine |
| `indexer_k_quant_and_cache` | 8192 tok × 128 | 92 GB/s in | tiny kernel, launch-bound at this size |
| `fwht_rotate` (fwd) | 65536 rows × 128 | **815 GB/s** | ~87% of that peak — warp-shuffle butterflies |

Two wins landed on that host after the first baseline, both worth re-checking here:

- **Quant MoE GEMMs — 32-row M-blocking (1.4–1.6×).** The mma path re-loaded and
  re-dequantized the weight `B` fragment once per 16-row tile. The expert schedule
  is already per-32-row (`expert_of_tile`), so each block now owns the full 32-row
  tile and reuses every dequantized `B` fragment across both 16-row M-subtiles.
  fp8 6.5→10.5, int4 4.9→7.0, int8 3.7→5.7 TFLOP/s.
- **`rms_norm_quant` — multi-warp block per row + folded amax (1.7×).** One warp
  per row capped occupancy at ~33%; rewritten to one 256-thread block per row with
  the dynamic-scale `max|v·w|` folded into the sum-of-squares sweep (3 passes → 2).
  181→304 GB/s.

## Contract Conformance

This backend implements the umbrella QuixiCore contract, which includes benchmark
methodology — not just op names. Conform to it:

- **`docs/benchmarking.md`** (umbrella) — required reporting fields and the rule
  that cross-backend numbers are comparable only at identical operation
  semantics, shape, dtype, quant format, and measurement policy. Backend-specific
  optimized variants are reported separately.
- **`registry/benchmark-shapes.yaml`** — the canonical shape families
  (`decode_small`, `decode_large`, `prefill`, `quant_matmul`, `moe`). Local
  exploratory shapes are fine, but **contract compatibility is measured against
  registry shapes**, so every kernel family needs at least those covered.
- **`registry/tolerances.yaml`** — per-dtype rtol/atol and the
  deterministic-vs-stochastic policy.
- **`.quixicore/kernels.yaml`, `.quixicore/quant-formats.yaml`** — the
  machine-readable inventory `scripts/coverage-report` consumes. Prefer these over
  any hand-maintained prose list (including the one below, which has drifted
  before).

Every benchmark record should carry: backend repo + commit, contract version,
hardware target, driver/runtime/compiler versions, kernel family + operation +
public entry point, shape name and concrete dims, input/output dtypes, quant
format, warmup and measured iteration counts, latency summary (median + min/max
or p20/p80 + CoV), derived throughput, correctness tolerance and observed max
abs/rel error, and the raw output path.

## Repo Facts To Preserve

Authoritative inventory is `.quixicore/kernels.yaml`. The map below is orientation
only.

- **Quant format layer**: `kernels/quant/quant_formats.cuh` (**18** bit-arithmetic
  format structs + encoders + swizzle), `quant_formats_tables.cuh` (**11** GGUF
  k-/i-quant structs) + `quant_tables.cuh`, `dequant8<FMT>` span helpers,
  `tm_qmm.cuh` (`load_wfrag`/`load_xfrag`/`mma16816`).
- **Format count is 29.** The authoritative registry is the `TMQ_FORMATS`
  X-macro at `kernels/tm_cuda/tm_cuda_ext.cu:16-21`: 8 legacy + 7 fp8/fp4/mx +
  3 (mxfp6×2, bitnet) + 5 k-quants + 6 i-quants = 29 exposed formats. The "30"
  that appears in `README.md:88` and `thundermittens_ampere_port.md` is folklore —
  it double-counts `fp8_raw`, a codes-only internal storage mode
  (`quant_formats.cuh:273`, used by `qgemm_blockscale`) that is not an exposed
  format; `mxfp6` is one template instantiated `<true>`/`<false>`. Use **29**.
- **Quant matmul/decode**: `qgemv.cu`, `qgemm.cu` (contains `qgemm_ksplit`, a
  kernel, not a file), `qgemm_variants.cu` (`qgemm_actorder`, `qgemm_blockscale`),
  `qflux.cu`, `qgemv_int.cu` (w8a8/w2a8, dp4a), `quant_rt.cu`, `lm_head.cu` +
  `lm_head_topkp.cu`, `tm_kernels.cuh` (consolidated), `tm_rng.cuh` (RNG — in
  `kernels/quant/`, not `serving/`), `turboquant.cuh` (**M6**: FWHT rotation,
  `tq_encode` codec, `moe_lora_align`, `permute_cols`).
- **Quantized MoE (M1)**: `kernels/moe_quant/tm_moe_quant_kernels.cuh` —
  `moe_gemm_fp8` (rowwise), `moe_gemm_nvfp4` (dual-fp4, swizzled A-scale),
  `moe_gemm_wna16<BIT>` (int4/int8), fused `silu_and_mul_quant`,
  `per_token_group_quant_fp8`, `nvfp4_experts_quant`, `moe_route_scored`.
- **Non-quant MoE (W6)**: `kernels/moe/tm_moe_kernels.cuh` — `moe_route_topk`,
  histogram/scan/scatter permute, 32-pad schedule, gather, grouped GEMM, finalize.
- **Norm/activation quant (M2)**: `kernels/elementwise/tm_norm_quant_kernels.cuh`
  — `rms_norm_quant<FP8,DYN,RESID>`, `azp_int8_quant`, `per_token_group_int8`.
- **Serving/decode**: `kernels/serving/` — `kv_cache` (also hosts `attn_window`),
  `paged_attn_v2` (v1/partition/reduce/gqa_staged/cascade), `mla`
  (bf16/fp8/sparse/partitioned + insert), `rope_kv`, `attn_q`, `attn_varlen`,
  `beam_xcache`, `sampling`, `spec_beam`, `logits_proc_kernels.cuh` (sampler zoo),
  `eagle_kernels.cuh`, and **`sparse_serving_kernels.cuh`** (M5: MInference
  vertical/slash index builder, `tau_tail`, DeepSeek-V3.2
  `indexer_k_quant_and_cache` + `cp_gather_indexer_k_quant_cache`). Shared warp
  helpers in `tm_warp.cuh`.
- **Elementwise/norm/training**: `kernels/elementwise/tm_elementwise_kernels.cuh`
  (rms/layernorm fwd+bwd, add_norm+fp8, softmax, gelu, glu×6, dropout,
  cross_entropy, embedding, hadamard, adamw, add). Standalone `kernels/layernorm/`
  is the one kernel with a wired harness runner; `kernels/flux/` produces the
  44.8/47.3 TFLOP/s figures.
- **Linear/state-space**: `kernels/lin_attn_tm/` (`tm_linattn_kernels.cuh` holds
  linear_attn/causal/chunk-parallel/cmplx_matmul; **GDN** in `gdn_kernels.cuh`),
  `kernels/mamba2/selective_scan_kernels.cuh` (tile-SSD + varlen scan + APC),
  `kernels/based/`, `kernels/hedgehog/`, `kernels/linear_attention/`.
- **Dense/attention base**: `kernels/attention/mha_ampere/`,
  `kernels/gemm/{int8_ampere,fp8_ampere}/`, `kernels/fftconv/`, `kernels/rotary/`.
  **There is no `gemm/bf16_ampere`** — the Ampere bf16 GEMM is `gemm/bf16_h100`
  rebuilt with `ARCH=SM80`.
- **SM90+/Blackwell (build-only on Ampere)**: `attention/mha_h100`,
  `attention/mha_h100_lcf`, `attention/bf16_b300_mha_{causal,noncausal}`,
  `gemm/{fp8_h100,fp8_h100_scaled,bf16_b200,fp8_b200,int8_b200,mxfp8_b200,nvfp4_b200}`,
  `kernels/parallel/` (multimem). Registry marks these with a min-CC gate.
- **Contract layer**: `include/quixicore/{contract/kernel_abi.hpp,
  contract/operations.hpp, cuda/contract.hpp, cuda/contract_stubs.hpp}`.
- **Substrate tests**: `tests/` (`unit_tests.cu`, `group/`, `thread/`, `Makefile`)
  — source of the 2467/2468 figure. Run via `scripts/build tests` or
  `make -C tests`.

<a name="build-and-correctness"></a>
## Build And Correctness

**Arch flags are host-dependent — this is the most common failure here.** Use
`-gencode arch=compute_80,code=sm_80` on A100, `compute_86,code=sm_86` on the
3090, or list both. `kernels/tm_cuda/setup.py:27` and
`kernels/tm_cuda/build_ext.sh:20` currently hard-code `sm_86` only and must be
parameterized before anything runs on this host; the same applies to the
standalone-harness recipe in [Final Verification](#final-verification).

Build the python extension with the committed script (**9 TUs**), not the
"5-TU line in the session notes" (which does not exist):

```bash
bash kernels/tm_cuda/build_ext.sh    # fix its ARCH + interpreter path first
```

`setup.py`'s header documents a `cpp_extension` workaround for a torch-cu130
vs system-CUDA-12.9 mismatch. That reason is **obsolete**: this host has CUDA 13.2
and torch 2.13.0+cu130 (same major), so `python3 setup.py build_ext` may now work.
Re-test before propagating the workaround. `scripts/build tm_cuda` uses the
`setup.py` path.

Every standalone kernel has a self-contained fp64/oracle harness with its build
command in the header comment (e.g. `kernels/quant/qgemm.cu`,
`kernels/moe_quant/moe_quant_test.cu`, `kernels/serving/logits_proc_test.cu`).
Golden data comes from `kernels/quant/gen_golden*.py`, which import the Metal
`quant.py` reference — **currently unavailable** (`~/ThunderMittens` is absent),
so quant golden regeneration is not reproducible here; the committed `golden/` and
`golden_int/` sets still work.

End-to-end gate — use bare `pytest` so new test files are never silently missed:

```bash
cd kernels/tm_cuda && LD_LIBRARY_PATH=<torch>/lib \
  CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q
```

There are **13** test files. An explicit list previously omitted `test_m5.py`,
`test_m6.py`, and `test_mf_followups.py` even though their TUs are in the build —
that is exactly why this gate is a bare `pytest`. Note `scripts/test tm_cuda`
runs only `test_tk_cuda.py`, a much weaker gate.

**Two environment prerequisites, both of which currently bite (verified
2026-07-30):**

1. **`LD_LIBRARY_PATH` must include torch's `lib/`.** The `.so` is linked against
   `libtorch.so` without an rpath, so importing `tk_cuda._C` otherwise fails with
   `ImportError: libtorch.so: cannot open shared object file`.
2. **`pytest -q` cannot even *collect* without the `tk` module.**
   `test_tk_cuda.py`, `test_step4.py`, and `test_serving.py` do
   `from tk.quant import ...`, which resolves to
   `~/ThunderMittens/ThunderMittens/kernels/tk/quant.py` — **absent on this
   host**. A collection error in one module aborts the whole run, so the bare gate
   yields *zero* tests until `tk` is on `PYTHONPATH`. The other **10** test files
   are `tk`-independent and can be named explicitly as an interim gate:

   ```bash
   python3 -m pytest -q test_elementwise.py test_gdn.py test_m4.py test_m5.py \
     test_m6.py test_mf_followups.py test_moe_quant.py test_norm_quant.py \
     test_selective_scan.py test_w6.py
   ```

   Restoring `tk` (see [Open Items](#open-items)) is what makes the real gate
   usable again — it also unblocks `gen_golden*.py` and `bench_vs_torch.py`, which
   share the dependency.

## Reference Search Protocol

**`.reference/` is an operator-supplied, git-ignored local checkout, and it is
absent on this host.** The searches below do nothing until you populate it. It is
in `.gitignore` by design ("local reference checkouts"), so a fresh clone never
has it.

```bash
mkdir -p .reference && cd .reference
git clone --depth 1 https://github.com/vllm-project/vllm
git clone --depth 1 https://github.com/flashinfer-ai/flashinfer
git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM
git clone --depth 1 https://github.com/NVIDIA/cutlass
```

Marlin dequant tricks live inside the vllm checkout
(`.reference/vllm/csrc/.../quantization/marlin/`). MetalForge and the sibling
QuixiCore Metal/ROCm trees are useful for operation behavior and for serving
kernels being ported; add them if the task needs them.

Record the exact files inspected in `perf/optimization_status.md`. Bucket every
reference idea into: **portable algorithm** (consider), **SM90+/Blackwell
mechanism** (translate only if Ampere has a real analogue — usually cp.async +
`m16n8k16` mma + dp4a), or **benchmark/shape idea** (usually adopt).

**Do not import implementation code from references into this repository** unless
a license and provenance review explicitly allows it. Ideas and shapes are fine;
copied code is not.

```bash
rg -n "cp.async|mma.sync|dp4a|ldmatrix|__shfl|cvta" .reference kernels
rg -n "moe_wna16|marlin|dequant|per_token_group|nvfp4" .reference
```

## Measurement Harness

Benchmark against **three** baselines, not one: the framework/library path
(cuBLAS, SDPA, torch — see the table in [Host Reality](#host-reality)), a naive
decomposed path (e.g. `dequantize(wq) @ x`), and the current QuixiCore CUDA
implementation before the change.

- **`perf/bench_kernels.py`** (or its wrapper `scripts/bench`) — registry-driven.
  Each entry declares `arch`, `min_cc`, `sm80_ok`, `config`
  (`standalone`|`python`|`pytorch`), `make_vars`, `out`. Build phase compiles for
  the declared arch (SM80 when arch-portable and host CC < 90) and parses ptxas
  registers/spills/smem; run phase executes only what the host can run and what
  has a runner. Schedules single-GPU benches one per device (`--gpus`), multi-GPU
  last.
  ```bash
  python3 perf/bench_kernels.py --phase all --kernel all
  python3 perf/bench_kernels.py --phase run  --kernel layernorm --gpus 0
  ```
  **Know its real coverage before trusting it.** The registry has 45 entries and
  **only two runner types** (`stdout`, `layernorm`), so it can actually *run* 4 of
  45; `attention/mha_ampere`, `gemm/int8_ampere`, and `gemm/fp8_ampere` are
  Ampere-native but skipped as "no automated runner wired." More importantly the
  registry has **no entries at all** for `quant/`, `serving/`, `moe_quant/`,
  `elementwise/`, `lin_attn_tm/`, or `moe/` — i.e. the entire QuixiCore/MetalForge
  port. Those families are benchmarked by their standalone `.out` harnesses plus
  `sweep_quant.sh` and `bench_metalforge.py`, and **those results do not flow into
  `results.jsonl`.** Closing that gap is tracked in
  [Open Items](#open-items).
- **`perf/sweep_quant.sh`** — the 29-format qgemv/qgemm GB/s + TFLOP/s table.
  Iterates `golden/` files (not the format list directly) and greps `GB/s` /
  `TFLOP/s`; defaults output to `/tmp/quant_sweep.md`.
- **`perf/bench_metalforge.py`** — MetalForge serving-layer kernels. Needs
  `tk_cuda._C` built.
- **`perf/bench_vs_torch.py`** — quant GEMV/GEMM vs `torch.matmul` fp16 at matched
  shapes. **Currently broken**: it imports from a hard-coded
  `~/ThunderMittens/.../kernels` path that does not exist here, plus `tk_cuda._C`.
- **`perf/bench_reference.py`** — cuBLAS/SDPA/LayerNorm framework roofline sweep.
- **`perf/tk_bench_layernorm.py`** — the one wired non-stdout runner.
- Per-kernel standalone `.out` harnesses print their own timing + correctness.

Writes `perf/results/YYYY-MM-DD/<run-id>/{run.json,results.jsonl,summary.md,
build/*.log,run/*.log}`. `perf/results/` is git-ignored (twice), so **archived run
directories cited in `baseline_status.md` are dead on any clone** — copy summary
snippets into the notebook instead of citing run paths as evidence.

**Timing rules (CUDA):** CUDA-event timing around each call,
`cudaDeviceSynchronize` outside the timed region, warm up first (cuBLAS
heuristics, autotuners). Do not regenerate inputs in the timed region. Record
min/max; if the spread exceeds ~10%, re-run on an idle GPU before trusting an A/B.

Two recorded methodology traps, both of which produced fake results before:

- **Contention.** A shared-GPU "2.7× vs torch" at M=1 corrected to ~1.4× idle.
  Absolutes need an idle GPU; ratios on a shared GPU are still fair.
- **Per-sync latency floor.** Timing one small kernel per sync measures dispatch,
  not the kernel. Batch several calls per sample and divide. (The sibling Metal
  handbook records a layernorm that "improved" 0.26 ms → 0.084 ms purely from
  fixing this.)

Derived metrics:

```text
GEMM FLOPs            = 2*M*N*K
attention fwd FLOPs   ~= 4*B*H*N^2*D        (halve for causal)
quant decode          = effective_GBps = packed_weight_bytes_read / time / 1e9
norm/pointwise bytes  = conservative required reads + writes (note cache reuse)
```

State when an estimate ignores cache reuse, repeated passes, metadata reads, or
write allocation. With 40 MB of L2 on this host, a "bandwidth" number on a small
working set may be cache-resident — that means "L2-resident", not "magic".

**Profiling:** `kernels/common.mk` provides `make ncu` (Nsight Compute, full
sections) and `make nsys` per kernel. Use them when a timing A/B can't explain a
result — achieved occupancy, memory throughput %, warp-stall reasons,
`smsp__inst_executed` for dp4a/mma issue, bank conflicts. Don't commit profiler
artifacts; record their path + a summary.

## Shape Strategy

Cover the umbrella registry shapes first (`registry/benchmark-shapes.yaml`:
`decode_small`, `decode_large`, `prefill`, `quant_matmul`, `moe`) — those are what
cross-backend comparability is measured on. Then add local coverage: small/edge,
tile-aligned, tile-ragged, real-model, and stress shapes. Log skips with a reason.

- **Dense GEMM**: 1024/2048/4096/8192/16384 squares + LLM rectangles
  (`K=4096,N=11008/14336`, small-M prefill/decode).
- **Quant GEMV/GEMM**: LLM projections `N=3840/13824/2560, K=2560/6912`; batch
  sweep `M∈{1,2,4,8,16,32,64,128}` (find the GEMV→ksplit→cuBLAS crossovers);
  all 29 formats.
- **MoE**: E∈{8,16,64}, top-k∈{2,8}, H/inter LLM sizes; padded-schedule tile
  counts vs **108 SMs**.
- **Attention/decode**: `(B,H,N,D)` D∈{64,128}, N∈{512…4096+}, causal/non-causal;
  paged decode context 128…8192 with `partition_size` sweeps; GQA ratios.
- **Norm/pointwise/sampler**: rows ∈{4096,16384,65536}, hidden ∈{1024…8192},
  vocab ∈{32k,128k,256k} for lm_head/samplers.
- **Linear/SSM**: `(B,H,N,D)` with N long (the recurrence occupancy stress).

## Per-Kernel Optimization Loop

1. **Inventory** — entry points, dtypes, shape contract, tests, existing bench.
2. **References** — `.reference/` (populate it first), marlin/vllm/cutlass,
   QuixiCore Metal/ROCm, MetalForge originals.
3. **Baseline** — correctness first (pytest + standalone harness), then the
   harness on the shape set vs the three baselines. Raw output under
   `perf/results/`, summary into the notebook.
4. **Classify** — bytes, FLOPs, achieved vs measured roofline, occupancy from
   ptxas regs/smem + `ncu`, variance.
5. **Experiments** — one factor at a time. Hypothesis before edit.
6. **Execute** — small, reversible; focused correctness test first, then the same
   shape matrix, then broader tests for a candidate.
7. **Decide** — keep only if it beats targets by ≥3% (low-risk) / ≥8–10%
   (complexity-adding), regresses nothing that matters, and has a
   counter-backed explanation.
8. **Record** — update `perf/optimization_status.md` (status, tables, decision
   log, rejected alternatives). Commit only when asked; sole author Eric
   Hartford, no AI attribution.

## Experiment Catalogue (Ampere)

**Launch geometry / occupancy.** Grid should saturate **108 SMs** with enough
resident warps to hide latency. For 1-warp-per-row decode kernels, batch multiple
rows per block or split-K across `blockIdx.z`. `qgemm_ksplit`'s heuristic
(`tiles < ~832` → split) was derived as ≈10 waves × 82 SMs on the 3090; the
equivalent constant here is ≈**1100**, but **re-measure rather than rescaling
blindly** — A100's larger L2 and different compute:bandwidth ratio may move the
crossover independently of SM count. Sweep threads/block (multiple of 32),
warps/block, rows/block. Watch tail effects when grid size isn't a multiple of SM
count and wasted work on partial edge tiles.

**cp.async pipelines.** For GEMM/attention, double/triple-buffer global→smem with
`cp.async.cg` + `commit_group`/`wait_group`; tune ring depth (2 vs 3 stages) vs
smem budget and occupancy. **163 KB usable smem/block here vs 99 KB on SM86** —
ring depths and tile sizes that were smem-blocked on the 3090 may now fit. The
int8 ring is the proven template. `gemm_staged`'s lesson still holds: staging only
wins if reuse beats the added smem traffic and occupancy loss.

**mma tiling & fragments.** Sweep BM/BN/BK; feed `m16n8k16` fragments densely
(fewer partial-K tiles). For quant, prefer **dequant-direct-to-fragment**
(`load_wfrag<FMT>`) over a full-dequant global round-trip when weights are used
once; route superblock formats through the fp16 materialize + cuBLAS only at
prefill M.

**Dequant strategy (highest-leverage for quant, and more so on A100).** Because
the FP32:bandwidth ratio dropped 4× on this host, scalar dequant work that was
hidden on the 3090 may now dominate. Hoist the block scale / sub-scale / grid
entry to **once per 8-span** (`dequant8<FMT>` — **19** explicit specializations
exist: 8 in `quant_formats.cuh`, 11 in `quant_formats_tables.cuh`, plus a generic
fallback). The formats still on the generic path — `q4_1`, `q5_0`, `q5_1`, `kU4`,
`kU4B8`, `hqq`, `fp8_e4m3`, `e5m2`, `fp8_block`, `bitnet` — are a concrete open
item. Move i-quant `__constant__` grids to **smem** (divergent indices defeat the
constant cache). Branchless bit-tricks for fp8/fp4 (Marlin: sign |
mantissa-shift + one pow-2 mul). Predecode scales per K-block.

**Memory layout & vectorization.** Ensure adjacent lanes read adjacent addresses
on hot paths. Use `half2`/`float4`/128-bit (`ld.global.v4`) loads where aligned.
`ldmatrix` for register↔shared fragment loads. Compare row-major vs swizzled smem
to kill bank conflicts.

**Reductions & numerics.** `__shfl_xor_sync` warp reductions before smem
reductions; smem cross-warp fixup only when it pays. Keep fp32 accumulation for
softmax/norm/attention/long-K unless a measured lower-precision variant passes
the `registry/tolerances.yaml` bound. Integer/exact kernels: exactness is contract.

**Fusion.** Fuse bias/residual/scale/gelu/gate/norm epilogues that would
round-trip through global (flux/qflux, add_norm, silu_and_mul_quant). Fuse
dequant into matmul/attention when the dequantized value is used once. Split a
fused kernel if register pressure or occupancy loss dominates the saved traffic.

**Branch/scalar hoisting.** Template on format/D/causal/block-k so decisions leave
the inner loop; precompute base offsets and use increments; specialize D=64/128
and aligned K. A little scalar decode can erase the byte savings in qgemv/qgemm/
attn_q/int paths — and costs 4× more here than on the 3090.

**Routing & crossovers.** Find the GEMV→ksplit→cuBLAS-prefill crossover by
sweeping M; all such thresholds are 3090-derived and need re-deriving on A100.
Route tiny elementwise to a single fused kernel; batch decode rows.

## Per-Kernel Starting Hypotheses

All numbers below are 3090-measured. Re-baseline on A100 before acting.

- **`qgemv`** (top decode priority). Effective packed-weight GB/s should scale
  with bits/weight. i-quants (15–71 GB/s on 3090) are lookup-bound → **smem-resident
  grids**; also try 2 output rows/warp, `half2` X loads, wider vectorized packed
  loads. Expect the i-quant gap to widen on A100 (scalar-bound), making this the
  clearest first win. MX/NV (136–254) vs q8_0 (283): check block-scale coalescing.
- **`qgemm`/`qflux`**. Decode uses `qgemm_ksplit`; prefill uses cuBLAS-on-dequant.
  The open win: a **fused cp.async-ring dequant-in-pipeline GEMM** (dequant into
  fragments inside the ring) to beat the separate dequant pass at M≥64 — more
  attractive here, since cuBLAS bf16 is 239 TFLOP/s and the extra dequant pass now
  costs proportionally more bandwidth-time. Also: actorder gather cost,
  `fp8_block` 2D-scale path.
- **`moe_gemm_{fp8,nvfp4,wna16}` (M1)**. 32-row M-blocked `mma16816`, no cp.async.
  Add the **cp.async ring + `_shared_b` 64×64 staging** (the 163 KB smem budget
  helps), per-expert scale hoist, N-blocking to 32×32 for A-reuse, and measure vs
  a dequant-to-fp16 + cuBLAS grouped path at prefill.
- **`qgemv_int`/`w8a8`** (528 GB/s on 3090). Native dp4a; tune output-rows/block,
  activation layout, split-K. Note A100 int8 measured 353 TOP/s vs 3090's ~51 —
  the int path has far more headroom here.
- **`mha_ampere`** (fwd 48.7 TFLOP/s on 3090; bwd D=128 spills). A100's SDPA
  reference is 128–153 TFLOP/s, so the bar is much higher here. Sweep seq block
  size, D-specialize, cp.async double-buffer K/V, dS^T staging, logsumexp storage;
  attack the D=128 backward register spills.
- **`attn_q`** (quantized-KV prefill). Save bandwidth only if K/V dequant doesn't
  dominate — K-dequant-to-shared vs V-dequant-to-register; causal/non-causal
  specialize; format sweep.
- **paged decode (`paged_attention` v1/v2/`gqa_staged`, `mla_decode*`)**. KV-
  bandwidth + occupancy bound. Sweep `partition_size` per context; measure the
  partition-grid axis (v2) and `gqa_staged` smem KV reuse vs v1; batch multiple
  (head,batch) per block to raise occupancy; fp8-cache dequant-on-read cost vs
  bf16.
- **`gdn`/`selective_scan`/`lin_attn*`**. Serial recurrences, one warp per
  (req,hv,dv)/(dim,batch)/(b,h) — low occupancy at short seq, and worse on 108 SMs
  than 82. Try chunk-parallel decomposition, state layout in smem/registers,
  `exp2` vs `exp` where valid, larger blocks covering more (dv/dim) lanes.
- **norm/`softmax`/`gelu`/`rotary`/samplers/`logits_proc`**. Bandwidth/reduction
  bound. Framework bars on A100: rms_norm 1504 GB/s, silu 1687, softmax 742 —
  note softmax is the weak framework path and therefore the best target.
  Vectorized contiguous loads, rows/block sweep, warp-only reductions for small
  hidden, fp32 accum, hidden/vocab specialization.
- **MoE permute (`kernels/moe/`: route/pad/gather/finalize)**. Atomic contention
  in histogram/scatter; vectorize gather; measure grouped-GEMM vs per-expert
  dispatch crossover at small E.
- **`cmplx_matmul`/`fftconv`**. Register pressure + intermediate traffic; complex
  tile layout, pointwise-complex-mul fusion, fewer inter-stage global writes.

## Decision Rules

**Candidate win** when it: passes the focused correctness test; improves median on
priority realistic shapes by ≥3% (low-risk) or ≥8–10% (complexity-adding);
regresses no required correctness shape or `registry/tolerances.yaml` bound;
regresses no secondary perf shape beyond the agreed tolerance unless the routing
intentionally narrows the target; and has a bytes/FLOPs/counter-backed explanation.

**Reject/defer** when: the win is inside noise; it appears only on toy shapes; it
adds substantial complexity without a durable real-shape win; it depends on an
SM90+/Blackwell feature this host lacks; or it breaks a numeric contract
(exact-int kernels, RNG bit-parity samplers).

## Recording Format

Each kernel section in `perf/optimization_status.md`:

- Status: not started / baselining / experimenting / candidate / landed / deferred.
- Current best impl + current public route (e.g. "qgemm M≥64 → cuBLAS-on-dequant").
- References inspected (exact files).
- Correctness command + last result.
- Baseline table (framework / naive / current CUDA implementation on the shape set).
- Experiment table (one factor per row, before/after, keep/reject + why).
- Decision log + open questions.

**`optimization_status.md` currently has zero per-kernel entries** — no kernel has
ever been recorded in the format this handbook mandates, and the MetalForge
baselines ended up in this file instead, violating the method/results split. The
de-facto ledger is `thundermittens_ampere_port.md`. Seeding the notebook from it
is [an open item](#open-items).

<a name="final-verification"></a>
## Final Verification Before Landing A Win

```bash
# focused correctness (standalone harness for the touched kernel)
# NOTE: set the arch for YOUR host — sm_80 on A100, sm_86 on RTX 3090
cd kernels/<family> && nvcc <kernel>_test.cu -std=c++17 -O2 \
  -gencode arch=compute_80,code=sm_80 -o <kernel>_test.out -I../quant -I../serving
CUDA_VISIBLE_DEVICES=0 ./<kernel>_test.out
# format sweep if quant
CUDA_VISIBLE_DEVICES=0 bash perf/sweep_quant.sh
# full end-to-end regression (must not drop below the current passing count)
cd kernels/tm_cuda && CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q
# golden dequant-exactness for any quant_formats.cuh change
cd kernels/quant && for f in golden/*; do ./qgemv.out "$f"; done | grep -c EXACT
```

For `include/` substrate changes, also run the broader unit tests
(`scripts/build tests`, or `make -C tests`) and the
`perf/tools/sm80_*_smoke.cu` primitive smokes. Those are
`sm80_{wgmma,pipeline,lcsf,transpose,int8,int8_pipeline,a0,a0_mirror,based_cross,based_local}_smoke.cu`
plus `sm80_addr_probe.cu` — more than the four usually cited.

<a name="open-items"></a>
## Open Items On This Host

Ordered by leverage. The first three are prerequisites for any measurement.

1. **Parameterize the arch flags.** `kernels/tm_cuda/{setup.py,build_ext.sh}` and
   the per-kernel harness headers emit `sm_86` only; nothing runs on this A100
   until they take the host arch (or a multi-arch list). **Confirmed sufficient:**
   swapping to `compute_80,code=sm_80` was the only change needed to get 70/70
   green here, so this is a build-plumbing fix, not a porting effort. Also fix
   `build_ext.sh:6`'s `../../.venv/bin/python` and
   `bench_kernels.py:304`'s `REPO_ROOT/.venv/bin/python`, and either add an rpath
   to torch's `lib/` or document `LD_LIBRARY_PATH`.
2. **Re-baseline everything on A100 (SM 8.0).** All canonical numbers in this file
   are 3090/SM86. Priorities: the 29-format qgemv sweep, `qgemm_ksplit`'s split
   threshold, and the MetalForge serving table. For the GLM-5.2-Vision GGUF
   workload specifically, the measured A100 budget analysis and the resulting
   kernel design are in **`perf/a100_glm52_design.md`** — including a validated
   Q2_K structure that holds 94% of DRAM roofline versus today's 12–32%.
3. **Correct `perf/baseline_status.md`.** Mark lines 15–27 and the per-kernel
   status table as superseded (they pre-date the port), and add the A100 host
   block.
4. **`fftconv_pc` is now runnable.** It was blocked on SM86 by a 112 KB scratch
   requirement against a 99 KB limit; A100 gives 163 KB. Free win.
5. **Register the port families in the harness**, or state plainly that
   `quant/ serving/ moe_quant/ elementwise/ lin_attn_tm/ moe/` are measured
   outside `results.jsonl`. Only 4 of 45 registry entries are runnable today, and
   `mha_ampere`/`int8_ampere`/`fp8_ampere` lack runners despite being Ampere-native.
6. **Seed `optimization_status.md`** from `thundermittens_ampere_port.md` and the
   MetalForge table, and move the 3090 results out of this handbook.
7. **Finish `dequant8` coverage** for the 10 formats still on the generic path.
8. **Audit `shared_to_register.cuh` on H100** and add int shared↔register unit
   test coverage (`baseline_status.md:212-213`).
9. **Restore the `tk` module dependency** — one missing package blocks three
   things at once: the primary pytest gate (`test_tk_cuda.py`, `test_step4.py`,
   `test_serving.py` — the quant-format and serving suites), the quant
   golden-generation path (`gen_golden*.py`), and `bench_vs_torch.py`. It is
   expected at `~/ThunderMittens/ThunderMittens/kernels/tk/quant.py`; vendoring
   the reference quantizer into this repo would remove the external-path
   dependency permanently.
10. **Housekeeping**: ~10 committed ELF binaries in `perf/tools/` escape
    `.gitignore` because they have no extension; add a rule.

## External References

- NVIDIA **CUDA C++ Programming Guide** & **Best Practices Guide** — occupancy,
  coalescing, cp.async, warp primitives, `__launch_bounds__`.
- **PTX ISA** — `mma.sync.m16n8k16`, `cp.async`, `dp4a`, `ldmatrix`, `cvta`.
- NVIDIA **A100 Tuning Guide** (SM80) and **Ampere GA10x Tuning Guide** (SM86) —
  smem/register budgets, tensor/int throughput, async-copy. Read the one matching
  your host; they differ (see the SM80/SM86 table above).
- **Marlin** (inside a populated `.reference/vllm/`) and **CUTLASS** — quant
  dequant tricks, mma fragment layouts, ring pipelines, grouped/MoE GEMM.
- **Nsight Compute / Systems** docs — occupancy, memory-throughput, stall-reason,
  and roofline sections.
- Umbrella contract: `QuixiCore/docs/benchmarking.md`,
  `registry/benchmark-shapes.yaml`, `registry/tolerances.yaml`.
- Sibling handbooks: QuixiCore Metal and QuixiCore ROCm `perf/perf.md` — shared
  contracts, shape sets, and recording discipline.
- In-repo history: `thundermittens_ampere_port.md` (the de-facto port ledger) and
  `metalforge_gap_analysis.md` (measured gaps, M1–M6 status).
