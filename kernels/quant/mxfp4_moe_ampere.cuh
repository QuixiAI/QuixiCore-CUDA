/**
 * @file
 * @brief MXFP4 (OCP MX / GGUF type 39) MoE kernel family for NVIDIA Ampere
 *        (SM 8.0+): fused per-route decode, grouped tensor-core tile, and the
 *        segmented (permutation-based) pipeline.
 *
 * Provenance: ported from the SlimServe DSV4-on-A100 serving tree, where these
 * kernels were developed, validated and measured on the live profiles:
 *   csrc/quixicore/quant/dsv4_mxfp4_moe_ampere.cuh   (fused decode path)
 *   csrc/quixicore/quant/dsv4_mxfp4_mmq_ampere.cuh   (tensor-core tile)
 *   csrc/quixicore/quant/dsv4_mxfp4_seg_ampere.cuh   (segmented pipeline)
 * plus the minimal int8 mma.sync machinery those kernels consume from the
 * vendored mmq_v2 port (MIT, llama.cpp-adapted; attribution below).
 *
 * Design lineage: the QuixiCore-XPU grouped_qgemm/glu_quant review supplied
 * the segmented launch (device-side count/prefix/scatter route grouping, a
 * static worst-case grid walked through a per-expert prefix table, and the
 * fused SwiGLU + route-weight + requantize epilogue); the fused decode path
 * mirrors the tuned ROCm IQ2_XXS/Q2_K per-route strategy; the tile kernel
 * reuses the dense Q8_0 MMQ v2 shared-memory layout so its vec dot runs
 * unmodified over decoded MXFP4.
 *
 * Measured on A100 (SlimServe DSV4 profiles): the tensor-core tile is
 * 8.8-57x the dp4a grouped tile at prefill widths; end to end, mxfp4-4 12K
 * prefill +50%, c8 ~2x from the tile, and the segmented pipeline a further
 * +33% at cold-c8.
 *
 * Weight format (GGUF MXFP4, 32 values in 17 bytes): one e8m0 scale byte then
 * 16 bytes of e2m1 nibble pairs. The LOW nibbles supply values [0,16) and the
 * HIGH nibbles [16,32) -- not an even/odd interleave. All three routes share
 * one integer decode: the e2m1 table holds 2x the true magnitudes so the int8
 * dot stays exact, and the 0.5 factor is folded into the e8m0 block scale.
 */
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace tmq_a100 {

// ---------------------------------------------------------------- q8_1 blocks
// GGUF activation block: 32 int8 quants behind a half2 (d, sum) header. d is
// the dequant scale; sum is the pre-quantization value sum (unused by the
// q8_0-style vec dots here but kept so the layout matches the GGUF stack).
constexpr int QK8_1 = 32;
constexpr int QI8_1 = 8;  // int32 words per block
constexpr int QI8_0 = 8;

struct block_q8_1 {
  half2 ds;  // ds.x = delta, ds.y = sum
  int8_t qs[QK8_1];
};
static_assert(sizeof(block_q8_1) == 36, "unexpected block_q8_1 size");

// ------------------------------------------------------- int8 mma primitives
//
// MIT License
//
// Copyright (c) 2023-2024 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
//
// The tile fragments, ldmatrix/mma wrappers and the q8_0 x q8_1 vec dot below
// are adapted from llama.cpp ggml/src/ggml-cuda/{mma,mmq}.cuh at commit
// 9b2a088819cda774bdbf713168ee1eee8498cda5, reduced to the int8 tile shapes,
// the sm80 instructions and the single tile-config family (I=128, 8 warps,
// 256 K values per iteration) that this file uses.

// Fragment of an int32 matrix distributed over one warp. The (i, j) element of
// logical tile <I, J> lives in lane get_i(l)/get_j(l) register x[l], matching
// the PTX mma.sync operand layout.
template <int I_, int J_>
struct tile {
  static constexpr int I = I_;
  static constexpr int J = J_;
  static constexpr int ne = I * J / 32;
  int x[ne] = {0};

  static __device__ __forceinline__ int get_i(const int l) {
    if constexpr (I == 8 && J == 8) {
      return threadIdx.x / 4;
    } else if constexpr (I == 16 && J == 8) {
      return ((l / 2) * 8) + (threadIdx.x / 4);
    } else {
      return -1;
    }
  }

  static __device__ __forceinline__ int get_j(const int l) {
    if constexpr (I == 8 && J == 8) {
      return (l * 4) + (threadIdx.x % 4);
    } else if constexpr (I == 16 && J == 8) {
      return ((threadIdx.x % 4) * 2) + (l % 2);
    } else {
      return -1;
    }
  }
};

template <int I, int J>
static __device__ __forceinline__ void load_generic(tile<I, J>& t,
                                                    const int* __restrict__ xs0,
                                                    const int stride) {
#pragma unroll
  for (int l = 0; l < t.ne; ++l) {
    t.x[l] = xs0[t.get_i(l) * stride + t.get_j(l)];
  }
}

// Requires a 16 byte aligned shared memory address; the tile strides are
// padded to guarantee it.
static __device__ __forceinline__ void load_ldmatrix(
    tile<16, 8>& t, const int* __restrict__ xs0, const int stride) {
  int* xi = (int*)t.x;
  const int* xs = xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * 4;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
               : "l"(xs));
}

static __device__ __forceinline__ void mma(tile<16, 8>& D, const tile<16, 8>& A,
                                           const tile<8, 8>& B) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
      : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]), "r"(B.x[0]),
        "r"(B.x[1]));
}

// ------------------------------------------------- q8_0 mmq tile layout + dot
constexpr int MMQ_NTHREADS = 256;
constexpr int MMQ_NWARPS = MMQ_NTHREADS / 32;
constexpr int MMQ_I = 128;
constexpr int MMQ_TILE_NE_K = 32;
// block_q8_1_mmq shape per column: 4 fp32 scales + 32 int32 of quants.
constexpr int MMQ_TILE_Y_K = MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1;
// 2*32 quant words + 2*32/QI8_0 scale words + 4 padding; %8==4 keeps every
// ldmatrix address 16 byte aligned.
constexpr int MMQ_SRAM_STRIDE_Q8_0 =
    2 * MMQ_TILE_NE_K + 2 * MMQ_TILE_NE_K / QI8_0 + 4;

static constexpr __host__ __device__ int mmq_rows_per_warp(int J) {
  return (J >= 48 && J % 16 == 0) ? 32 : 16;
}

// x tile: MMQ_I rows of [64 quant ints | 8 fp32 block scales | pad], stride
// MMQ_SRAM_STRIDE_Q8_0. y tile: J columns of [4 fp32 scales | 32 quant ints],
// stride MMQ_TILE_Y_K. k00 selects which 128-value half of the x tile the
// (reloaded) y tile currently covers.
template <int J>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_mma(
    const int* __restrict__ x, const int* __restrict__ y,
    float* __restrict__ sum, const int k00) {
  typedef tile<16, 8> tile_A;
  typedef tile<8, 8> tile_B;
  typedef tile<16, 8> tile_C;

  constexpr int sram_stride = MMQ_SRAM_STRIDE_Q8_0;
  constexpr int rows_per_warp = mmq_rows_per_warp(J);
  constexpr int ntx = rows_per_warp / tile_C::I;

  y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

  const int* x_qs = (const int*)x;
  const float* x_df = (const float*)x_qs + 2 * MMQ_TILE_NE_K;
  const int* y_qs = (const int*)y + 4;
  const float* y_df = (const float*)y;

  tile_A A[ntx][MMQ_TILE_NE_K / QI8_0];
  float dA[ntx][tile_C::ne / 2][MMQ_TILE_NE_K / QI8_0];

  const int i0 = (threadIdx.y / ntx) * rows_per_warp;

#pragma unroll
  for (int n = 0; n < ntx; ++n) {
#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
      load_ldmatrix(A[n][k01 / QI8_0],
                    x_qs + (i0 + n * tile_A::I) * sram_stride + k00 + k01,
                    sram_stride);
    }

#pragma unroll
    for (int l = 0; l < tile_C::ne / 2; ++l) {
      const int i = i0 + n * tile_A::I + tile_C::get_i(2 * l);
#pragma unroll
      for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
        dA[n][l][k01 / QI8_0] = x_df[i * sram_stride + (k00 + k01) / QI8_0];
      }
    }
  }

#pragma unroll
  for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
      tile_B B;
      float dB[tile_C::ne / 2];

      // load_generic beats load_ldmatrix for the B fragment here
      load_generic(B, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

#pragma unroll
      for (int l = 0; l < tile_C::ne / 2; ++l) {
        const int j = j0 + tile_C::get_j(l);
        dB[l] = y_df[j * MMQ_TILE_Y_K + k01 / QI8_1];
      }

#pragma unroll
      for (int n = 0; n < ntx; ++n) {
        tile_C C;
        mma(C, A[n][k01 / QI8_0], B);

#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
          sum[(j0 / tile_C::J + n) * tile_C::ne + l] +=
              C.x[l] * dA[n][l / 2][k01 / QI8_0] * dB[l % 2];
        }
      }
    }
  }
}

// ============================================================================
// Fused per-route decode path.
//
// The MXFP4 build stores both W1 ([expert, gate|up, packed]) and W2
// ([expert, out_row, packed]) as OCP MXFP4: 32 values in 17 bytes (one e8m0
// scale byte + 16 bytes of e2m1 nibbles). The generic MoE route dequantizes
// through the MMVQ vector kernels one projection at a time with fp32
// intermediates; on A100 that measured ~32 tok/s at TP4 decode.
//
// This path mirrors the tuned IQ2_XXS/Q2_K pipeline (SlimServe
// dsv4_moe_ampere.cuh, itself ported from the optimized ROCm strategy):
// quantize the activation to Q8_1 once, compute paired gate/up rows against
// the staged activation, apply SwiGLU and the route weight, emit Q8_1
// directly, and consume it with a weighted MXFP4 down kernel that accumulates
// all routes into the output. One warp owns one intermediate row, so the Q8_1
// emission is a warp reduction with no cross-block coordination.
//
// The raw GGUF layout interleaves the scale byte with the codes (17-byte
// blocks), which breaks aligned vectorized loads. A byte-neutral repack
// splits each expert into a scale array followed by a 16-byte-aligned code
// array (same total bytes, same expert stride), so the inner loop issues one
// uint4 per block. e2m1 decode goes through the same integer table as the
// MMVQ kernel: table values are 2x the true magnitudes and the 0.5 factor is
// folded into the block scale, keeping the inner loop on __dp4a.
// ============================================================================

__device__ __forceinline__ float mxfp4_scale_to_fp32(uint8_t x) {
  // e8m0: the byte is the fp32 exponent field; 0 is the smallest normal.
  const uint32_t bits = (x == 0) ? 0x00400000u : ((uint32_t)x << 23);
  float r;
  memcpy(&r, &bits, 4);
  return r;
}

// Expand 8 packed e2m1 codes (one int) into two ints of 2x-value int8 lanes.
// NOTE: CUDA's __byte_perm takes 4-bit nibble selectors (16-bit total), not
// AMD-style per-byte selectors, so the ROCm/XPU byte-permute table trick
// needs a selector repack that erodes its advantage; the scalar table loop
// compiles to predicated selects and measured fine at decode widths.
__device__ __forceinline__ void mxfp4_expand8(int q4, int& lo, int& hi) {
  static constexpr int8_t kValues[16] = {0, 1,  2,  3,  4,  6,  8,  12,
                                         0, -1, -2, -3, -4, -6, -8, -12};
  const uint32_t l = (uint32_t)q4, h = ((uint32_t)q4 >> 4);
  int8_t bl[4], bh[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    bl[i] = kValues[(l >> (8 * i)) & 0xF];
    bh[i] = kValues[(h >> (8 * i)) & 0xF];
  }
  memcpy(&lo, bl, 4);
  memcpy(&hi, bh, 4);
}

// One token's staged Q8_1 block held in registers so gate and up rows can
// both dot against a single set of activation loads (the shared-activation
// pattern from the tuned XPU nvfp4_row_dot_pair).
struct Q8Block {
  int v[8];
  float d;
};

__device__ __forceinline__ void mxfp4_load_q8(const block_q8_1* q8,
                                              Q8Block& r) {
  const int* q8i = reinterpret_cast<const int*>(q8->qs);
#pragma unroll
  for (int i = 0; i < 8; ++i) r.v[i] = q8i[i];
  r.d = __low2float(q8->ds);
}

// One MXFP4 block (32 values) against a register-staged Q8_1 block. `codes`
// holds the 16 nibble bytes; `scale` is the raw e8m0 byte.
__device__ __forceinline__ float mxfp4_block_dot_regs(const uint4 codes,
                                                      const uint8_t scale,
                                                      const Q8Block& q8) {
  int sumi = 0;
  int lo, hi;
  mxfp4_expand8((int)codes.x, lo, hi);
  sumi = __dp4a(lo, q8.v[0], sumi);
  sumi = __dp4a(hi, q8.v[4], sumi);
  mxfp4_expand8((int)codes.y, lo, hi);
  sumi = __dp4a(lo, q8.v[1], sumi);
  sumi = __dp4a(hi, q8.v[5], sumi);
  mxfp4_expand8((int)codes.z, lo, hi);
  sumi = __dp4a(lo, q8.v[2], sumi);
  sumi = __dp4a(hi, q8.v[6], sumi);
  mxfp4_expand8((int)codes.w, lo, hi);
  sumi = __dp4a(lo, q8.v[3], sumi);
  sumi = __dp4a(hi, q8.v[7], sumi);
  return mxfp4_scale_to_fp32(scale) * 0.5f * q8.d * float(sumi);
}

__device__ __forceinline__ float mxfp4_block_dot(const uint4 codes,
                                                 const uint8_t scale,
                                                 const block_q8_1* q8) {
  Q8Block r;
  mxfp4_load_q8(q8, r);
  return mxfp4_block_dot_regs(codes, scale, r);
}

// Fetch one block's codes+scale from either layout. `nblocks` is the number
// of MXFP4 blocks per expert (repacked scale-region size in bytes).
template <bool REPACKED>
__device__ __forceinline__ void mxfp4_load_block(
    const char* __restrict__ expert_base, const int64_t nblocks,
    const int64_t block_index, uint4& codes, uint8_t& scale) {
  if constexpr (REPACKED) {
    scale = reinterpret_cast<const uint8_t*>(expert_base)[block_index];
    codes = reinterpret_cast<const uint4*>(expert_base + nblocks)[block_index];
  } else {
    const char* block = expert_base + block_index * 17;
    scale = *reinterpret_cast<const uint8_t*>(block);
    memcpy(&codes, block + 1, 16);
  }
}

// Row dot for one (gate,up) pair. Eight lanes cooperate on a row; the caller
// reduces across the 8-lane group.
template <bool REPACKED>
__device__ __forceinline__ void mxfp4_gate_up_row_dot(
    const char* __restrict__ expert_base, const block_q8_1* __restrict__ input,
    const int64_t nblocks, const int hidden, const int input_token_blocks,
    const int intermediate, const int row, const int token, const int lane8,
    float& gate, float& up) {
  const int blocks_per_row = hidden / 32;
  const block_q8_1* token_input = input + int64_t(token) * input_token_blocks;
  const int64_t gate_base = int64_t(row) * blocks_per_row;
  const int64_t up_base = int64_t(row + intermediate) * blocks_per_row;
  for (int k = lane8; k < blocks_per_row; k += 8) {
    Q8Block q8;
    mxfp4_load_q8(token_input + k, q8);
    uint4 codes;
    uint8_t scale;
    mxfp4_load_block<REPACKED>(expert_base, nblocks, gate_base + k, codes,
                               scale);
    gate += mxfp4_block_dot_regs(codes, scale, q8);
    mxfp4_load_block<REPACKED>(expert_base, nblocks, up_base + k, codes,
                               scale);
    up += mxfp4_block_dot_regs(codes, scale, q8);
  }
}

// Decode W1: grid (intermediate/32, tokens*top_k), 256 threads = 32 rows x 8
// lanes. Each 8-lane group owns one intermediate row (its gate and up rows),
// applies SwiGLU and the route weight, and lane groups then cooperate on the
// 32-wide Q8_1 output block. Mirrors iq2_xxs_gate_up_swiglu_q8_1_decode.
template <int TOP_K, bool REPACKED>
__global__ __launch_bounds__(256, 2) void mxfp4_gate_up_swiglu_q8_1_decode(
    const void* __restrict__ weights, const block_q8_1* __restrict__ input,
    block_q8_1* __restrict__ output, const int* __restrict__ topk_ids,
    const float* __restrict__ route_weights, const int64_t expert_stride_bytes,
    const int hidden, const int input_token_blocks, const int intermediate,
    const int tokens, const int experts, const float swiglu_limit) {
  const int lane8 = threadIdx.x & 7;
  const int row_lane = threadIdx.x >> 3;
  const int route = blockIdx.y;
  const int token = route / TOP_K;
  const int row = blockIdx.x * 32 + row_lane;
  const int expert = route < tokens * TOP_K ? topk_ids[route] : -1;
  const int blocks_per_mid = intermediate / QK8_1;
  const int64_t nblocks = expert_stride_bytes / 17;

  float gate = 0.0f;
  float up = 0.0f;
  if (expert >= 0 && expert < experts && row < intermediate) {
    const char* expert_base = reinterpret_cast<const char*>(weights) +
                              int64_t(expert) * expert_stride_bytes;
    mxfp4_gate_up_row_dot<REPACKED>(expert_base, input, nblocks, hidden,
                                    input_token_blocks, intermediate, row,
                                    token, lane8, gate, up);
  }
#pragma unroll
  for (int mask = 4; mask > 0; mask >>= 1) {
    gate += __shfl_down_sync(0xffffffffu, gate, mask);
    up += __shfl_down_sync(0xffffffffu, up, mask);
  }

  __shared__ float values[32];
  if (lane8 == 0) {
    float value = 0.0f;
    if (expert >= 0 && expert < experts && row < intermediate) {
      if (swiglu_limit > 0.0f) {
        gate = fminf(gate, swiglu_limit);
        up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
      }
      value = (gate / (1.0f + expf(-gate))) * up * route_weights[route];
      if (!isfinite(value)) value = 0.0f;
    }
    values[row_lane] = value;
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    const float value = values[threadIdx.x];
    float amax = fabsf(value);
    float sum = value;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
      sum += __shfl_xor_sync(0xffffffffu, sum, mask);
    }
    const float scale = amax / 127.0f;
    const int8_t quant =
        amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / scale));
    block_q8_1* out = output + int64_t(route) * blocks_per_mid + blockIdx.x;
    out->qs[threadIdx.x] = quant;
    if (threadIdx.x == 0) out->ds = __floats2half2_rn(scale, sum);
  }
}

// Down: out[token, j] = sum_r dot(W2[e_r, j, :], mid_q8[route_r, :]).  The
// route weight was already folded into the Q8_1 mid activation by the W1
// kernel, so the down pass is an unweighted accumulation over the token's
// routes. Grid (out_row/32, tokens), 256 threads = 32 rows x 8 lanes.
template <typename scalar_t, int TOP_K, bool REPACKED>
__global__ __launch_bounds__(256, 2) void mxfp4_down_sum(
    const void* __restrict__ weights, const block_q8_1* __restrict__ mid,
    const int* __restrict__ topk_ids, scalar_t* __restrict__ out,
    const int64_t expert_stride_bytes, const int intermediate,
    const int out_row, const int tokens, const int experts) {
  const int lane8 = threadIdx.x & 7;
  const int row_lane = threadIdx.x >> 3;
  const int token = blockIdx.y;
  const int row = blockIdx.x * 32 + row_lane;
  const int blocks_per_row = intermediate / 32;
  const int mid_blocks_per_route = intermediate / QK8_1;
  const int64_t nblocks = expert_stride_bytes / 17;
  if (row >= out_row) return;

  float acc = 0.0f;
#pragma unroll 1
  for (int r = 0; r < TOP_K; ++r) {
    const int route = token * TOP_K + r;
    const int expert = topk_ids[route];
    if (expert < 0 || expert >= experts) continue;
    const char* expert_base = reinterpret_cast<const char*>(weights) +
                              int64_t(expert) * expert_stride_bytes;
    const block_q8_1* route_mid = mid + int64_t(route) * mid_blocks_per_route;
    const int64_t row_base = int64_t(row) * blocks_per_row;
    for (int k = lane8; k < blocks_per_row; k += 8) {
      uint4 codes;
      uint8_t scale;
      mxfp4_load_block<REPACKED>(expert_base, nblocks, row_base + k, codes,
                                 scale);
      acc += mxfp4_block_dot(codes, scale, route_mid + k);
    }
  }
#pragma unroll
  for (int mask = 4; mask > 0; mask >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, mask);
  }
  if (lane8 == 0) {
    out[int64_t(token) * out_row + row] = scalar_t(acc);
  }
}

// Byte-neutral AoS(17) -> SoA(scales | 16B codes) repack. One thread per
// block; reads raw, writes split. Output has the same shape/stride as input.
static __global__ void repack_mxfp4_experts(const uint8_t* __restrict__ raw,
                                            uint8_t* __restrict__ packed,
                                            const int64_t nblocks,
                                            const int64_t expert_stride_bytes) {
  const int expert = blockIdx.y;
  const int64_t block = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (block >= nblocks) return;
  const uint8_t* src = raw + int64_t(expert) * expert_stride_bytes + block * 17;
  uint8_t* dst = packed + int64_t(expert) * expert_stride_bytes;
  dst[block] = src[0];
  uint4 codes;
  memcpy(&codes, src + 1, 16);
  reinterpret_cast<uint4*>(dst + nblocks)[block] = codes;
}

inline void launch_repack_mxfp4_experts(const void* raw, void* packed,
                                        const int experts,
                                        const int64_t nblocks,
                                        const int64_t expert_stride_bytes,
                                        cudaStream_t stream) {
  const dim3 grid((unsigned)((nblocks + 255) / 256), experts);
  repack_mxfp4_experts<<<grid, 256, 0, stream>>>(
      static_cast<const uint8_t*>(raw), static_cast<uint8_t*>(packed), nblocks,
      expert_stride_bytes);
}

inline void launch_mxfp4_gate_up_swiglu_q8_1_decode(
    const void* weights, const void* input, void* output, const int* topk_ids,
    const float* route_weights, const int64_t expert_stride_bytes,
    const int hidden, const int input_token_blocks, const int intermediate,
    const int tokens, const int top_k, const int experts,
    const float swiglu_limit, const bool repacked, cudaStream_t stream) {
  const dim3 grid(intermediate / 32, tokens * top_k);
  const auto in = static_cast<const block_q8_1*>(input);
  const auto out = static_cast<block_q8_1*>(output);
#define LAUNCH_MXFP4_W1(TOPK, REPACKED)                                     \
  mxfp4_gate_up_swiglu_q8_1_decode<TOPK, REPACKED>                          \
      <<<grid, 256, 0, stream>>>(weights, in, out, topk_ids, route_weights, \
                                 expert_stride_bytes, hidden,               \
                                 input_token_blocks, intermediate, tokens,  \
                                 experts, swiglu_limit)
  if (top_k == 6) {
    if (repacked) LAUNCH_MXFP4_W1(6, true);
    else LAUNCH_MXFP4_W1(6, false);
  } else {
    if (repacked) LAUNCH_MXFP4_W1(8, true);
    else LAUNCH_MXFP4_W1(8, false);
  }
#undef LAUNCH_MXFP4_W1
}

template <typename scalar_t>
inline void launch_mxfp4_down_sum(const void* weights, const void* mid,
                                  const int* topk_ids, scalar_t* out,
                                  const int64_t expert_stride_bytes,
                                  const int intermediate, const int out_row,
                                  const int tokens, const int top_k,
                                  const int experts, const bool repacked,
                                  cudaStream_t stream) {
  const dim3 grid((out_row + 31) / 32, tokens);
  const auto mid_blocks = static_cast<const block_q8_1*>(mid);
#define LAUNCH_MXFP4_W2(TOPK, REPACKED)                              \
  mxfp4_down_sum<scalar_t, TOPK, REPACKED>                           \
      <<<grid, 256, 0, stream>>>(weights, mid_blocks, topk_ids, out, \
                                 expert_stride_bytes, intermediate,  \
                                 out_row, tokens, experts)
  if (top_k == 6) {
    if (repacked) LAUNCH_MXFP4_W2(6, true);
    else LAUNCH_MXFP4_W2(6, false);
  } else {
    if (repacked) LAUNCH_MXFP4_W2(8, true);
    else LAUNCH_MXFP4_W2(8, false);
  }
#undef LAUNCH_MXFP4_W2
}

// ============================================================================
// Grouped tensor-core MoE GEMM (moe_align padded-metadata route).
//
// The dp4a grouped tile (SlimServe moe.cuh moe_mxfp4) is MOE_X=4 columns wide
// on CUDA, so prefill re-streams every expert's full weight tile per 4 routed
// rows; measured e2e that leaves the MXFP4 profiles at ~25% of the hybrid
// quant's throughput at c8 when activated-byte parity predicts ~59%. This
// kernel gives the wide path real weight reuse: 128 expert rows x 64 routed
// columns per tile, K in 256-value iterations, int8 mma.sync via the mmq_v2
// machinery.
//
// The trick that keeps it small: e2m1 decode goes through the same 2x-value
// int8 table as the fused decode path, and the decoded tile is written in
// exactly the shared-memory layout of the dense Q8_0 MMQ v2 kernel (64 quant
// ints + 8 fp32 block scales per row, stride 76). vec_dot_q8_0_q8_1_mma then
// runs unmodified; only the tile loader (MXFP4 -> int8 smem), the activation
// gather (per sorted_token_ids, like moe.cuh moe_q), and the write-back
// (scatter to routed columns, skip padding slots) are MoE/MXFP4 specific.
// The 0.5 factor of the 2x table is folded into the e8m0 block scale.
//
// Weights are read in either layout: raw GGUF AoS (17-byte blocks: scale
// byte + 16 code bytes) or the byte-neutral SoA split (scales | 16B-aligned
// codes) produced by repack_mxfp4_experts.
//
// Column tiles follow moe_align_block_size metadata with block size 64.
// Padding slots hold sorted_token_ids >= ncols_dst: their gather is zeroed
// and the write-back drops them; the mma still computes them, which at DSV4
// prefill widths is a few percent of the tile.
// ============================================================================

constexpr int MXMMQ_J = 64;   // routed columns per tile (= align block size)
constexpr int MXMMQ_I = 128;  // expert rows per tile
constexpr int MXMMQ_NWARPS = 8;
constexpr int MXMMQ_NTHREADS = MXMMQ_NWARPS * 32;
constexpr int MXMMQ_ITER_K = 256;                          // values per K iter
constexpr int MXMMQ_BLOCKS_PER_ITER = MXMMQ_ITER_K / 32;   // 8 MXFP4 blocks
// Must equal MMQ_SRAM_STRIDE_Q8_0: 64 quant ints + 8 scale floats + 4 pad.
constexpr int MXMMQ_X_STRIDE = 2 * 32 + 8 + 4;
// Per-column ints in the y tile: 4 scale floats + 32 quant ints (one span of
// 128 activation values). Must equal MMQ_TILE_Y_K.
constexpr int MXMMQ_Y_STRIDE = 32 + 4;

static_assert(MXMMQ_X_STRIDE == MMQ_SRAM_STRIDE_Q8_0,
              "x tile layout must match the dense v2 vec dot");
static_assert(MXMMQ_Y_STRIDE == MMQ_TILE_Y_K,
              "y tile layout must match the dense v2 vec dot");

// Decode one iteration's weight tile into the q8_0-mmq shared layout.
// 128 rows x 8 blocks = 1024 (row, block) pairs; 256 threads, 4 rounds.
template <bool REPACKED, bool ROW_CLAMP>
__device__ __forceinline__ void mxmmq_load_x(
    const char* __restrict__ expert_base, int* __restrict__ x_tile,
    const int64_t nblocks, const int blocks_per_row, const int kb0,
    const int row_x0, const int i_max) {
  float* x_df = reinterpret_cast<float*>(x_tile + 2 * 32);
  const int tid = threadIdx.y * 32 + threadIdx.x;
#pragma unroll
  for (int r = 0; r < MXMMQ_I * MXMMQ_BLOCKS_PER_ITER / MXMMQ_NTHREADS; ++r) {
    const int pair = tid + r * MXMMQ_NTHREADS;
    const int i = pair / MXMMQ_BLOCKS_PER_ITER;
    const int kb = pair % MXMMQ_BLOCKS_PER_ITER;
    int row = row_x0 + i;
    if (ROW_CLAMP) row = min(row, i_max);
    uint4 codes;
    uint8_t scale;
    mxfp4_load_block<REPACKED>(expert_base, nblocks,
                               int64_t(row) * blocks_per_row + kb0 + kb, codes,
                               scale);
    int lo, hi;
    int* qs = x_tile + i * MXMMQ_X_STRIDE + kb * 8;
    mxfp4_expand8((int)codes.x, lo, hi);
    qs[0] = lo;
    qs[4] = hi;
    mxfp4_expand8((int)codes.y, lo, hi);
    qs[1] = lo;
    qs[5] = hi;
    mxfp4_expand8((int)codes.z, lo, hi);
    qs[2] = lo;
    qs[6] = hi;
    mxfp4_expand8((int)codes.w, lo, hi);
    qs[3] = lo;
    qs[7] = hi;
    x_df[i * MXMMQ_X_STRIDE + kb] = mxfp4_scale_to_fp32(scale) * 0.5f;
  }
}

// Gather one 128-value activation span (4 block_q8_1 blocks) for each of the
// tile's 64 columns into the block_q8_1_mmq-shaped y tile.
__device__ __forceinline__ void mxmmq_load_y(
    const block_q8_1* __restrict__ y, int* __restrict__ tile_y,
    const int* __restrict__ token_offs, const int q8b0,
    const int blocks_per_col_y, const int ncols_dst, const int top_k) {
  const int tid = threadIdx.y * 32 + threadIdx.x;
#pragma unroll
  for (int l0 = 0; l0 < MXMMQ_J * MXMMQ_Y_STRIDE; l0 += MXMMQ_NTHREADS) {
    const int l = l0 + tid;
    if (l >= MXMMQ_J * MXMMQ_Y_STRIDE) break;
    const int c = l / MXMMQ_Y_STRIDE;
    const int m = l % MXMMQ_Y_STRIDE;
    const int id = token_offs[c];
    if (id >= ncols_dst) {
      // Padding slot: zero the scales so discarded columns stay finite.
      if (m < 4) tile_y[l] = 0;
      continue;
    }
    const block_q8_1* col = y + int64_t(id / top_k) * blocks_per_col_y + q8b0;
    if (m < 4) {
      const float d = __low2float(col[m].ds);
      tile_y[l] = __float_as_int(d);
    } else {
      const int qi = m - 4;
      tile_y[l] = reinterpret_cast<const int*>(col[qi / 8].qs)[qi % 8];
    }
  }
}

template <typename scalar_t, bool REPACKED, bool ROW_CLAMP>
__global__ __launch_bounds__(MXMMQ_NTHREADS, 1) void moe_mxfp4_mmq_v2(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const int* __restrict__ num_tokens_post_padded,
    const int64_t exp_stride_bytes, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int ncols_pad) {
  const int blocks_per_row = ncols_x / 32;
  const int blocks_per_col_y = nrows_y / 32;
  const int ncols_dst = ncols_y * top_k;
  const int row_x0 = blockIdx.x * MXMMQ_I;
  const int col0 = blockIdx.y * MXMMQ_J;
  const int i_max = nrows_x - 1;
  const int tid = threadIdx.y * 32 + threadIdx.x;

  __shared__ int token_offs[MXMMQ_J];
  // ldmatrix requires 16-byte-aligned addresses; the row stride (76 ints =
  // 304 bytes) and the k offsets (multiples of 32 bytes) preserve base
  // alignment, so aligning the arrays is sufficient.
  __shared__ __align__(16) int tile_y[MXMMQ_J * MXMMQ_Y_STRIDE];
  __shared__ __align__(16) int tile_x[MXMMQ_I * MXMMQ_X_STRIDE];

  if (tid < MXMMQ_J) {
    // The sorted array length need not be a multiple of the tile width;
    // clamped reads land on padding slots which the write-back drops.
    token_offs[tid] = sorted_token_ids[min(col0 + tid, ncols_pad - 1)];
  }
  __syncthreads();

  const int exp_idx = expert_ids[blockIdx.y];
  if (exp_idx < 0) {
    // Callers no longer pre-fill dst; zero this tile's real columns.
    for (int j = tid; j < MXMMQ_J; j += MXMMQ_NTHREADS) {
      const int col_dst = token_offs[j];
      if (col_dst >= ncols_dst) continue;
      for (int i = 0; i < MXMMQ_I; ++i) {
        const int row = row_x0 + i;
        if (row < nrows_dst)
          dst[int64_t(col_dst) * nrows_dst + row] = scalar_t(0);
      }
    }
    return;
  }
  if (col0 >= num_tokens_post_padded[0]) return;

  const char* expert_base =
      reinterpret_cast<const char*>(vx) + int64_t(exp_idx) * exp_stride_bytes;
  const int64_t nblocks = exp_stride_bytes / 17;
  const block_q8_1* y = reinterpret_cast<const block_q8_1*>(vy);

  float sum[MXMMQ_J * MXMMQ_I / (MXMMQ_NTHREADS)] = {0.0f};

  for (int kb0 = 0; kb0 < blocks_per_row; kb0 += MXMMQ_BLOCKS_PER_ITER) {
    mxmmq_load_x<REPACKED, ROW_CLAMP>(expert_base, tile_x, nblocks,
                                      blocks_per_row, kb0, row_x0, i_max);
    mxmmq_load_y(y, tile_y, token_offs, kb0, blocks_per_col_y, ncols_dst,
                 top_k);
    __syncthreads();
    vec_dot_q8_0_q8_1_mma<MXMMQ_J>(tile_x, tile_y, sum, 0);
    __syncthreads();
    mxmmq_load_y(y, tile_y, token_offs, kb0 + 4, blocks_per_col_y, ncols_dst,
                 top_k);
    __syncthreads();
    vec_dot_q8_0_q8_1_mma<MXMMQ_J>(tile_x, tile_y, sum, 32);
    __syncthreads();
  }

  // Write back with the dense v2 fragment mapping, scattered to routed
  // columns; padding slots and row overhang are dropped.
  typedef tile<16, 8> tile_C;
  constexpr int rows_per_warp = mmq_rows_per_warp(MXMMQ_J);
  constexpr int ntx = rows_per_warp / tile_C::I;
  const int i0 = (threadIdx.y / ntx) * rows_per_warp;

#pragma unroll
  for (int j0 = 0; j0 < MXMMQ_J; j0 += ntx * tile_C::J) {
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
      for (int l = 0; l < tile_C::ne; ++l) {
        const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
        const int col_dst = token_offs[j];
        if (col_dst >= ncols_dst) continue;
        const int row = row_x0 + i0 + n * tile_C::I + tile_C::get_i(l);
        if (ROW_CLAMP && row >= nrows_dst) continue;
        dst[int64_t(col_dst) * nrows_dst + row] =
            scalar_t(sum[(j0 / tile_C::J + n) * tile_C::ne + l]);
      }
    }
  }
}

// True when this kernel can serve the shape: K a whole number of 256-value
// iterations (the loader has no partial-iteration path).
inline bool moe_mxfp4_mmq_v2_supported(int64_t ncols_x) {
  return ncols_x % MXMMQ_ITER_K == 0;
}

template <typename scalar_t>
inline void launch_moe_mxfp4_mmq_v2(
    const void* quant_x, const void* weights, scalar_t* dst,
    const int* sorted_token_ids, const int* expert_ids,
    const int* num_tokens_post_padded, const int64_t exp_stride_bytes,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_dst, const int top_k, const int ncols_pad,
    const bool repacked, cudaStream_t stream) {
  const dim3 grid((nrows_x + MXMMQ_I - 1) / MXMMQ_I,
                  (ncols_pad + MXMMQ_J - 1) / MXMMQ_J);
  const dim3 block(32, MXMMQ_NWARPS);
  const bool row_clamp = (nrows_x % MXMMQ_I) != 0;
#define LAUNCH_MXMMQ(REPACKED, ROW_CLAMP)                             \
  moe_mxfp4_mmq_v2<scalar_t, REPACKED, ROW_CLAMP>                     \
      <<<grid, block, 0, stream>>>(                                   \
          weights, quant_x, dst, sorted_token_ids, expert_ids,        \
          num_tokens_post_padded, exp_stride_bytes, ncols_x, nrows_x, \
          ncols_y, nrows_y, nrows_dst, top_k, ncols_pad)
  if (repacked) {
    if (row_clamp) LAUNCH_MXMMQ(true, true);
    else LAUNCH_MXMMQ(true, false);
  } else {
    if (row_clamp) LAUNCH_MXMMQ(false, true);
    else LAUNCH_MXMMQ(false, false);
  }
#undef LAUNCH_MXMMQ
}

// ============================================================================
// Segmented (permutation-based) MXFP4 MoE pipeline, replacing the moe_align
// padded-metadata route for the DSV4 wide path.
//
// Design ported from the QuixiCore-XPU grouped_qgemm review (2026-08-10):
// routes are grouped per expert by a device-side count/prefix/scatter chain
// (no host sync, no sorted+padded arrays), and the GEMM launches a STATIC
// worst-case grid of ceil(M/J)+E column tiles; every block rebuilds a small
// per-expert prefix table from rows_per_expert in shared memory and maps its
// linear tile index to (expert, local column range), exiting early past the
// real tile count. The grid depends only on (M, E), so the whole pipeline is
// CUDA-graph-capture-safe with varying routing.
//
// The W1 kernel additionally fuses the between-GEMM elementwise work (the
// XPU glu_quant idea): its epilogue spills the fp32 accumulator tile to the
// (dead) weight-staging smem region, applies SwiGLU + the route weight, and
// emits the Q8_1 mid activation directly -- the [routes, 2I] half
// intermediate, the separate activation pass, and the separate quantize pass
// all disappear. Pairing gate row r with up row I+r inside one tile is
// arranged by the row map: tile row i < 64 -> gate row g0+i, else up row
// I+g0+i-64.
//
// W2 consumes the Q8_1 mid (route-indexed, route weight already folded, the
// decode-path convention) through the same segmented gather and writes
// per-route output rows; a small deterministic reduce sums each token's
// routes in fixed j order (no atomics -- the XPU review flagged relaxed
// atomic accumulation as run-to-run nondeterministic).
//
// The mma stage reuses vec_dot_q8_0_q8_1_mma via the same smem layout as the
// grouped tile above. J is a template axis: 64 for prefill widths, 16 for
// small routed counts where a 64-wide tile is mostly masked slack (per-expert
// tile tails are masked, not padded slots, but the mma quantum is still J
// columns -- J=16 quarters that waste).
// ============================================================================

constexpr int SEG_MAX_EXPERTS = 256;

// ------------------------------------------------------------- cp.async
// The y gathers stream through cp.async so the next span's global loads
// overlap the current span's mma (these CTAs run at 1-2 per SM; there is no
// cross-warp surplus to hide gather latency otherwise). block_q8_1's 36-byte
// stride defeats 16-byte alignment, so qs words move at int granularity;
// sizes below 16 require the .ca (L1-allocating) variant. Pre-sm80 falls
// back to a synchronous copy with no-op fences, preserving semantics.
__device__ __forceinline__ void seg_cp_async4(void* smem, const void* glob) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s),
               "l"(glob));
#else
  *static_cast<int*>(smem) = *static_cast<const int*>(glob);
#endif
}
__device__ __forceinline__ void seg_cp_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N>
__device__ __forceinline__ void seg_cp_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

// ------------------------------------------------------------ perm metadata
// rows_per_expert/cursors must be zeroed before the histogram (launcher does
// a cudaMemsetAsync). Invalid expert ids (<0 or >=experts) are skipped, so
// perm slots cover only valid routes; the reduce re-checks validity.
static __global__ void seg_histogram(const int* __restrict__ topk_ids,
                                     int* __restrict__ rows_per_expert,
                                     const int routes, const int experts) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= routes) return;
  const int e = topk_ids[r];
  if (e < 0 || e >= experts) return;
  atomicAdd(&rows_per_expert[e], 1);
}

// Exclusive prefix into cursors plus the row/tile prefix tables the GEMM
// blocks consume (built once here instead of per-block; single block,
// serial, E <= 256).
static __global__ void seg_prefix(const int* __restrict__ rows_per_expert,
                                  int* __restrict__ cursors,
                                  int* __restrict__ rowseg,
                                  int* __restrict__ tseg, const int experts,
                                  const int J) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int rows = 0, tiles = 0;
  for (int e = 0; e < experts; ++e) {
    cursors[e] = rows;
    rowseg[e] = rows;
    tseg[e] = tiles;
    const int rpe = rows_per_expert[e];
    rows += rpe;
    tiles += (rpe + J - 1) / J;
  }
  rowseg[experts] = rows;
  tseg[experts] = tiles;
}

static __global__ void seg_scatter(const int* __restrict__ topk_ids,
                                   int* __restrict__ cursors,
                                   int* __restrict__ perm_ids,
                                   const int routes, const int experts) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= routes) return;
  const int e = topk_ids[r];
  if (e < 0 || e >= experts) return;
  const int slot = atomicAdd(&cursors[e], 1);
  perm_ids[slot] = r;
}

// Cooperatively load the precomputed row/tile prefix tables into scratch
// shared memory (the caller lends its weight-staging tile, dead until the K
// loop) and map this block's linear column tile to (expert, slot range).
// Returns false when the tile is past the real tile count (grid slack).
// scratch must hold 2*(experts+1) ints; the caller must __syncthreads()
// after this returns true, before overwriting the scratch.
template <int J>
__device__ __forceinline__ bool seg_locate(
    const int* __restrict__ g_rowseg, const int* __restrict__ g_tseg,
    const int experts, const int tile, int* __restrict__ scratch, int& expert,
    int& slot0, int& ncols) {
  int* rowseg = scratch;
  int* tseg = scratch + experts + 1;
  const int tid = threadIdx.y * 32 + threadIdx.x;
  for (int e = tid; e <= experts; e += MXMMQ_NTHREADS) {
    rowseg[e] = g_rowseg[e];
    tseg[e] = g_tseg[e];
  }
  __syncthreads();
  if (tile >= tseg[experts]) return false;
  // Warp-uniform linear walk (E <= 256; every thread does the same walk).
  int e = 0;
  while (tseg[e + 1] <= tile) ++e;
  const int local_tile = tile - tseg[e];
  const int rows_e = rowseg[e + 1] - rowseg[e];
  expert = e;
  slot0 = rowseg[e] + local_tile * J;
  ncols = min(J, rows_e - local_tile * J);
  return true;
}

// Gather one 128-value activation span for the tile's J columns, reading
// token activations through perm_ids (route -> token). qs words stream via
// cp.async; the four per-block scales convert half->float in registers after
// the async issues. Pad columns keep stale qs and zero scales (the mma
// multiplies by the zero scale; the operands are integers, so stale bytes
// cannot poison the sum). Caller owns seg_cp_commit/seg_cp_wait and the
// __syncthreads before the mma reads the tile.
template <int J>
__device__ __forceinline__ void seg_load_y_tokens(
    const block_q8_1* __restrict__ y, int* __restrict__ tile_y,
    const int* __restrict__ token_routes, const int ncols, const int q8b0,
    const int blocks_per_col_y, const int top_k) {
  const int tid = threadIdx.y * 32 + threadIdx.x;
  for (int l = tid; l < J * 32; l += MXMMQ_NTHREADS) {
    const int c = l >> 5;
    const int qi = l & 31;
    if (c >= ncols) continue;
    const block_q8_1* col =
        y + int64_t(token_routes[c] / top_k) * blocks_per_col_y + q8b0;
    seg_cp_async4(&tile_y[c * MXMMQ_Y_STRIDE + 4 + qi],
                  reinterpret_cast<const int*>(col[qi >> 3].qs) + (qi & 7));
  }
  for (int l = tid; l < J * 4; l += MXMMQ_NTHREADS) {
    const int c = l >> 2;
    const int m = l & 3;
    const block_q8_1* col =
        y + int64_t(token_routes[c] / top_k) * blocks_per_col_y + q8b0;
    tile_y[c * MXMMQ_Y_STRIDE + m] =
        c < ncols ? __float_as_int(__low2float(col[m].ds)) : 0;
  }
}

// Same, but the y rows are the Q8_1 mid activation indexed directly by route.
template <int J>
__device__ __forceinline__ void seg_load_y_mid(
    const block_q8_1* __restrict__ mid, int* __restrict__ tile_y,
    const int* __restrict__ token_routes, const int ncols, const int q8b0,
    const int blocks_per_col_y) {
  const int tid = threadIdx.y * 32 + threadIdx.x;
  for (int l = tid; l < J * 32; l += MXMMQ_NTHREADS) {
    const int c = l >> 5;
    const int qi = l & 31;
    if (c >= ncols) continue;
    const block_q8_1* col =
        mid + int64_t(token_routes[c]) * blocks_per_col_y + q8b0;
    seg_cp_async4(&tile_y[c * MXMMQ_Y_STRIDE + 4 + qi],
                  reinterpret_cast<const int*>(col[qi >> 3].qs) + (qi & 7));
  }
  for (int l = tid; l < J * 4; l += MXMMQ_NTHREADS) {
    const int c = l >> 2;
    const int m = l & 3;
    const block_q8_1* col =
        mid + int64_t(token_routes[c]) * blocks_per_col_y + q8b0;
    tile_y[c * MXMMQ_Y_STRIDE + m] =
        c < ncols ? __float_as_int(__low2float(col[m].ds)) : 0;
  }
}

// W1 weight loader over the paired row map: tile row i < 64 -> gate row
// g0 + i, else up row I + g0 + (i - 64). g0 + 63 < I is guaranteed by
// I % 64 == 0, so no clamp is needed.
template <bool REPACKED>
__device__ __forceinline__ void seg_load_x_w1(
    const char* __restrict__ expert_base, int* __restrict__ x_tile,
    const int64_t nblocks, const int blocks_per_row, const int kb0,
    const int g0, const int intermediate) {
  float* x_df = reinterpret_cast<float*>(x_tile + 2 * 32);
  const int tid = threadIdx.y * 32 + threadIdx.x;
#pragma unroll
  for (int r = 0; r < MXMMQ_I * MXMMQ_BLOCKS_PER_ITER / MXMMQ_NTHREADS; ++r) {
    const int pair = tid + r * MXMMQ_NTHREADS;
    const int i = pair / MXMMQ_BLOCKS_PER_ITER;
    const int kb = pair % MXMMQ_BLOCKS_PER_ITER;
    const int row = (i < 64) ? (g0 + i) : (intermediate + g0 + (i - 64));
    uint4 codes;
    uint8_t scale;
    mxfp4_load_block<REPACKED>(expert_base, nblocks,
                               int64_t(row) * blocks_per_row + kb0 + kb, codes,
                               scale);
    int lo, hi;
    int* qs = x_tile + i * MXMMQ_X_STRIDE + kb * 8;
    mxfp4_expand8((int)codes.x, lo, hi);
    qs[0] = lo;
    qs[4] = hi;
    mxfp4_expand8((int)codes.y, lo, hi);
    qs[1] = lo;
    qs[5] = hi;
    mxfp4_expand8((int)codes.z, lo, hi);
    qs[2] = lo;
    qs[6] = hi;
    mxfp4_expand8((int)codes.w, lo, hi);
    qs[3] = lo;
    qs[7] = hi;
    x_df[i * MXMMQ_X_STRIDE + kb] = mxfp4_scale_to_fp32(scale) * 0.5f;
  }
}

// ------------------------------------------------------------------- W1
// Grid: (intermediate/64 row-pair tiles, ceil(M/J)+E column tiles).
// Epilogue: spill C to smem, SwiGLU + route weight, emit Q8_1 mid blocks
// (mid[route, I] in 32-value blocks along I; this tile owns blocks
// rt*2 and rt*2+1 of each column's route).
template <int J, bool REPACKED>
__global__ __launch_bounds__(MXMMQ_NTHREADS, 1) void moe_mxfp4_seg_w1(
    const void* __restrict__ weights, const block_q8_1* __restrict__ input,
    block_q8_1* __restrict__ mid, const int* __restrict__ perm_ids,
    const int* __restrict__ g_rowseg, const int* __restrict__ g_tseg,
    const float* __restrict__ route_weights, const int64_t exp_stride_bytes,
    const int hidden, const int input_token_blocks, const int intermediate,
    const int experts, const int top_k, const float swiglu_limit) {
  // Dynamic smem: the double-buffered y tile pushes J=64 past the 48 KiB
  // static limit. Layout: routes | y0 | y1 | x (every offset is int-counted
  // and 16-byte aligned for J in {16, 64}).
  extern __shared__ int seg_w1_smem[];
  int* token_routes = seg_w1_smem;
  int* tile_y0 = token_routes + J;
  int* tile_y1 = tile_y0 + J * MXMMQ_Y_STRIDE;
  int* tile_x = tile_y1 + J * MXMMQ_Y_STRIDE;
  static_assert(2 * (SEG_MAX_EXPERTS + 1) <= MXMMQ_I * MXMMQ_X_STRIDE,
                "seg tables borrow the weight-staging tile");

  int expert, slot0, ncols;
  if (!seg_locate<J>(g_rowseg, g_tseg, experts, blockIdx.y, tile_x, expert,
                     slot0, ncols))
    return;

  const int tid = threadIdx.y * 32 + threadIdx.x;
  if (tid < J) {
    token_routes[tid] = tid < ncols ? perm_ids[slot0 + tid] : -1;
  }
  __syncthreads();

  const char* expert_base = reinterpret_cast<const char*>(weights) +
                            int64_t(expert) * exp_stride_bytes;
  const int64_t nblocks = exp_stride_bytes / 17;
  const int blocks_per_row = hidden / 32;
  const int g0 = blockIdx.x * 64;

  float sum[J * MXMMQ_I / MXMMQ_NTHREADS] = {0.0f};

  // Software pipeline: span s of superblock kb0 lives in tile_y[s&1]; the
  // next span's cp.async group is always in flight while the current mma
  // runs, and the x decode's global loads overlap the pending y group.
  // seg_cp_wait<1> retires exactly the oldest of the two pending groups.
  seg_load_y_tokens<J>(input, tile_y0, token_routes, ncols, 0,
                       input_token_blocks, top_k);
  seg_cp_commit();
  for (int kb0 = 0; kb0 < blocks_per_row; kb0 += MXMMQ_BLOCKS_PER_ITER) {
    seg_load_x_w1<REPACKED>(expert_base, tile_x, nblocks, blocks_per_row, kb0,
                            g0, intermediate);
    seg_load_y_tokens<J>(input, tile_y1, token_routes, ncols, kb0 + 4,
                         input_token_blocks, top_k);
    seg_cp_commit();
    seg_cp_wait<1>();
    __syncthreads();
    vec_dot_q8_0_q8_1_mma<J>(tile_x, tile_y0, sum, 0);
    __syncthreads();
    if (kb0 + MXMMQ_BLOCKS_PER_ITER < blocks_per_row) {
      seg_load_y_tokens<J>(input, tile_y0, token_routes, ncols,
                           kb0 + MXMMQ_BLOCKS_PER_ITER, input_token_blocks,
                           top_k);
      seg_cp_commit();
      seg_cp_wait<1>();
    } else {
      seg_cp_wait<0>();
    }
    __syncthreads();
    vec_dot_q8_0_q8_1_mma<J>(tile_x, tile_y1, sum, 32);
    __syncthreads();
  }

  // Spill C into the dead weight-staging smem: C[i][c] at tile_x[i*J + c],
  // fp32 (128*J*4 bytes <= the tile_x region for J <= 76).
  static_assert(J <= MXMMQ_X_STRIDE, "C spill must fit in tile_x");
  float* c_spill = reinterpret_cast<float*>(tile_x);
  {
    typedef tile<16, 8> tile_C;
    constexpr int rows_per_warp = mmq_rows_per_warp(J);
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
#pragma unroll
      for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
          const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
          const int i = i0 + n * tile_C::I + tile_C::get_i(l);
          c_spill[i * J + j] = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
        }
      }
    }
  }
  __syncthreads();

  // SwiGLU + route weight + Q8_1 emission. Warp tasks: (column, block b) with
  // b in {0,1} covering mid rows [b*32, b*32+32) of this tile's 64 pairs.
  const int mid_blocks_per_route = intermediate / 32;
  for (int task = threadIdx.y; task < ncols * 2; task += MXMMQ_NWARPS) {
    const int c = task >> 1;
    const int b = task & 1;
    const int m_local = b * 32 + threadIdx.x;
    float gate = c_spill[m_local * J + c];
    float up = c_spill[(64 + m_local) * J + c];
    if (swiglu_limit > 0.0f) {
      gate = fminf(gate, swiglu_limit);
      up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
    }
    const int route = token_routes[c];
    float value = (gate / (1.0f + expf(-gate))) * up * route_weights[route];
    if (!isfinite(value)) value = 0.0f;

    float amax = fabsf(value);
    float vsum = value;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
      vsum += __shfl_xor_sync(0xffffffffu, vsum, mask);
    }
    const float scale = amax / 127.0f;
    const int8_t quant =
        amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / scale));
    block_q8_1* out = mid + int64_t(route) * mid_blocks_per_route +
                      (int64_t(blockIdx.x) * 2 + b);
    out->qs[threadIdx.x] = quant;
    if (threadIdx.x == 0) out->ds = __floats2half2_rn(scale, vsum);
  }
}

// ------------------------------------------------------------------- W2
// Standard segmented tile over K = intermediate, y = Q8_1 mid (route rows),
// output per-route rows [routes, out_row] in the activation dtype.
template <typename scalar_t, int J, bool REPACKED>
__global__ __launch_bounds__(MXMMQ_NTHREADS, 1) void moe_mxfp4_seg_w2(
    const void* __restrict__ weights, const block_q8_1* __restrict__ mid,
    scalar_t* __restrict__ w2out, const int* __restrict__ perm_ids,
    const int* __restrict__ g_rowseg, const int* __restrict__ g_tseg,
    const int64_t exp_stride_bytes, const int intermediate, const int out_row,
    const int experts) {
  extern __shared__ int seg_w2_smem[];
  int* token_routes = seg_w2_smem;
  int* tile_y0 = token_routes + J;
  int* tile_y1 = tile_y0 + J * MXMMQ_Y_STRIDE;
  int* tile_x = tile_y1 + J * MXMMQ_Y_STRIDE;
  static_assert(2 * (SEG_MAX_EXPERTS + 1) <= MXMMQ_I * MXMMQ_X_STRIDE,
                "seg tables borrow the weight-staging tile");

  int expert, slot0, ncols;
  if (!seg_locate<J>(g_rowseg, g_tseg, experts, blockIdx.y, tile_x, expert,
                     slot0, ncols))
    return;

  const int tid = threadIdx.y * 32 + threadIdx.x;
  if (tid < J) {
    token_routes[tid] = tid < ncols ? perm_ids[slot0 + tid] : -1;
  }
  __syncthreads();

  const char* expert_base = reinterpret_cast<const char*>(weights) +
                            int64_t(expert) * exp_stride_bytes;
  const int64_t nblocks = exp_stride_bytes / 17;
  const int blocks_per_row = intermediate / 32;
  const int row_x0 = blockIdx.x * MXMMQ_I;

  float sum[J * MXMMQ_I / MXMMQ_NTHREADS] = {0.0f};

  seg_load_y_mid<J>(mid, tile_y0, token_routes, ncols, 0, blocks_per_row);
  seg_cp_commit();
  for (int kb0 = 0; kb0 < blocks_per_row; kb0 += MXMMQ_BLOCKS_PER_ITER) {
    mxmmq_load_x<REPACKED, false>(expert_base, tile_x, nblocks, blocks_per_row,
                                  kb0, row_x0, out_row - 1);
    seg_load_y_mid<J>(mid, tile_y1, token_routes, ncols, kb0 + 4,
                      blocks_per_row);
    seg_cp_commit();
    seg_cp_wait<1>();
    __syncthreads();
    vec_dot_q8_0_q8_1_mma<J>(tile_x, tile_y0, sum, 0);
    __syncthreads();
    if (kb0 + MXMMQ_BLOCKS_PER_ITER < blocks_per_row) {
      seg_load_y_mid<J>(mid, tile_y0, token_routes, ncols,
                        kb0 + MXMMQ_BLOCKS_PER_ITER, blocks_per_row);
      seg_cp_commit();
      seg_cp_wait<1>();
    } else {
      seg_cp_wait<0>();
    }
    __syncthreads();
    vec_dot_q8_0_q8_1_mma<J>(tile_x, tile_y1, sum, 32);
    __syncthreads();
  }

  typedef tile<16, 8> tile_C;
  constexpr int rows_per_warp = mmq_rows_per_warp(J);
  constexpr int ntx = rows_per_warp / tile_C::I;
  const int i0 = (threadIdx.y / ntx) * rows_per_warp;
#pragma unroll
  for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
      for (int l = 0; l < tile_C::ne; ++l) {
        const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
        if (j >= ncols) continue;
        const int row = row_x0 + i0 + n * tile_C::I + tile_C::get_i(l);
        w2out[int64_t(token_routes[j]) * out_row + row] =
            scalar_t(sum[(j0 / tile_C::J + n) * tile_C::ne + l]);
      }
    }
  }
}

// Deterministic unpermute-reduce: out[t, h] = sum_j w2out[t*top_k+j, h] over
// valid routes, fixed j order. Route weights were folded into the mid.
template <typename scalar_t>
static __global__ void seg_reduce(const scalar_t* __restrict__ w2out,
                                  const int* __restrict__ topk_ids,
                                  scalar_t* __restrict__ out, const int out_row,
                                  const int top_k, const int experts) {
  const int t = blockIdx.x;
  for (int h = threadIdx.x; h < out_row; h += blockDim.x) {
    float acc = 0.0f;
    for (int j = 0; j < top_k; ++j) {
      const int route = t * top_k + j;
      const int e = topk_ids[route];
      if (e < 0 || e >= experts) continue;
      acc += float(w2out[int64_t(route) * out_row + h]);
    }
    out[int64_t(t) * out_row + h] = scalar_t(acc);
  }
}

// ------------------------------------------------------------------ launch
inline int seg_col_tiles(const int routes, const int experts, const int J) {
  return (routes + J - 1) / J + experts;
}

// routes | y0 | y1 | x, int-counted (double-buffered y for the cp.async
// pipeline; J=64 exceeds the 48 KiB static limit, hence dynamic smem +
// opt-in below).
inline int seg_smem_bytes(const int J, const int x_stride) {
  return (J + 2 * J * MXMMQ_Y_STRIDE + MXMMQ_I * x_stride) * (int)sizeof(int);
}

template <typename KernelT>
inline void seg_maybe_opt_in_smem(KernelT kernel, const int smem) {
  if (smem > 48 * 1024) {
    // Idempotent and cheap; per-device/per-J bookkeeping not worth it.
    cudaFuncSetAttribute((const void*)kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  }
}

// meta scratch layout (ints): rows_per_expert[E] | cursors[E] |
// rowseg[E+1] | tseg[E+1] | perm_ids[routes].
inline int64_t seg_meta_ints(const int experts, const int routes) {
  return int64_t(2) * experts + 2 * (experts + 1) + routes;
}

template <typename scalar_t>
inline void launch_moe_mxfp4_seg(
    const void* quant_x, const void* w1, const void* w2, void* mid,
    scalar_t* w2out, scalar_t* out, const int* topk_ids,
    const float* route_weights, int* meta, const int64_t w1_stride_bytes,
    const int64_t w2_stride_bytes, const int hidden,
    const int input_token_blocks, const int intermediate, const int out_row,
    const int tokens, const int top_k, const int experts,
    const float swiglu_limit, const bool use_j16, cudaStream_t stream) {
  const int routes = tokens * top_k;
  int* rows_per_expert = meta;
  int* cursors = meta + experts;
  int* rowseg = meta + 2 * experts;
  int* tseg = rowseg + experts + 1;
  int* perm_ids = tseg + experts + 1;
  cudaMemsetAsync(rows_per_expert, 0, experts * sizeof(int), stream);
  {
    const int threads = 256;
    const int blocks = (routes + threads - 1) / threads;
    seg_histogram<<<blocks, threads, 0, stream>>>(topk_ids, rows_per_expert,
                                                  routes, experts);
    seg_prefix<<<1, 1, 0, stream>>>(rows_per_expert, cursors, rowseg, tseg,
                                    experts, use_j16 ? 16 : 64);
    seg_scatter<<<blocks, threads, 0, stream>>>(topk_ids, cursors, perm_ids,
                                                routes, experts);
  }
  const auto in = static_cast<const block_q8_1*>(quant_x);
  const auto mid_blocks = static_cast<block_q8_1*>(mid);
  const dim3 block(32, MXMMQ_NWARPS);
#define LAUNCH_SEG(J)                                                       \
  do {                                                                      \
    const int smem = seg_smem_bytes(J, MXMMQ_X_STRIDE);                     \
    const dim3 g1(intermediate / 64, seg_col_tiles(routes, experts, J));    \
    seg_maybe_opt_in_smem(moe_mxfp4_seg_w1<J, false>, smem);                \
    moe_mxfp4_seg_w1<J, false><<<g1, block, smem, stream>>>(                   \
        w1, in, mid_blocks, perm_ids, rowseg, tseg, route_weights,          \
        w1_stride_bytes, hidden, input_token_blocks, intermediate, experts, \
        top_k, swiglu_limit);                                               \
    const dim3 g2((out_row + MXMMQ_I - 1) / MXMMQ_I,                        \
                  seg_col_tiles(routes, experts, J));                       \
    seg_maybe_opt_in_smem(moe_mxfp4_seg_w2<scalar_t, J, false>, smem);     \
    moe_mxfp4_seg_w2<scalar_t, J, false><<<g2, block, smem, stream>>>(      \
        w2, mid_blocks, w2out, perm_ids, rowseg, tseg, w2_stride_bytes,     \
        intermediate, out_row, experts);                                    \
  } while (0)
  if (use_j16) LAUNCH_SEG(16);
  else LAUNCH_SEG(64);
#undef LAUNCH_SEG
  seg_reduce<scalar_t><<<tokens, 256, 0, stream>>>(w2out, topk_ids, out,
                                                   out_row, top_k, experts);
}

}  // namespace tmq_a100
