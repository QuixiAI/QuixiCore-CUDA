/**
 * @file
 * @brief Fused Q4_K (GGUF type 12) MoE decode pair for NVIDIA Ampere
 *        (SM 6.1+ dp4a): warp-per-row gate/up GEMV with fused SwiGLU +
 *        route weight + Q8_1 emission, and a Q8xQ4_K weighted down sum.
 *
 * Provenance: ported from the SlimServe DSV4-on-A100 serving tree
 * (csrc/quixicore/quant/dsv4_q4k_moe_ampere.cuh), where the pair serves the
 * hybrid artifact's Q4_K tail layers. It replaces the generic GGUF MoE
 * decode route (vec W1, SwiGLU pass, requantize pass, four moe_align
 * metadata launches, MMQ W2, weighted reduce) with two launches over raw
 * GGUF block_q4_K rows -- no repack; the layout is 4-byte aligned at every
 * access.
 *
 * Validated in SlimServe against an fp64 dequant reference across shard
 * sizes {256, 512, 1024} and tokens {1..64} incl. masked (-1) experts;
 * residuals decompose into +-1-quantum Q8_1 mid boundary flips (a
 * single-route lstsq decomposition showed every delta at v/scale frac ~= .5),
 * so parity metrics must use the mean relative error, not elementwise
 * bounds -- see the test harness.
 *
 * The Q4_K x Q8_1 dot (vec_dot_q4_K_q8_1 + scale unpacking) is adapted from
 * llama.cpp's CUDA backend via vLLM's vendored copy (MIT).
 *
 * Weight format (GGUF Q4_K, 256 values in 144 bytes): half2 (d, dmin), 12
 * bytes of packed 6-bit sub-block scales/mins, 128 bytes of 4-bit quants in
 * 32-value groups (low nibbles then high nibbles per 64-byte half).
 */
#pragma once

#include "mxfp4_moe_ampere.cuh"  // block_q8_1, warp helpers

namespace quixi_q4k {

using tmq_a100::block_q8_1;

constexpr int QK_K_Q4K = 256;
constexpr int Q8_BLOCK = 32;

struct block_q4_K {
  half2 dm;             // super-block scale / min
  uint8_t scales[12];   // 6-bit sub-block scales and mins
  uint8_t qs[128];      // 4-bit quants
};
static_assert(sizeof(block_q4_K) == 144, "unexpected block_q4_K size");

// llama.cpp-adapted contiguous-nibble dot: one call covers 32 weight values
// (8 low-nibble + 8 high-nibble bytes against two Q8_1 blocks).
__device__ __forceinline__ float vec_dot_q4_K_q8_1_impl(
    const int* __restrict__ v, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m,
    const half2& dm4, const float* __restrict__ d8) {
  float sumf_d = 0.0f;
  float sumf_m = 0.0f;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
    const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
    const int dot1 =
        __dp4a(v1i, u[2 * i + 1], __dp4a(v0i, u[2 * i + 0], 0));
    const int dot2 =
        __dp4a(0x01010101, u[2 * i + 1], __dp4a(0x01010101, u[2 * i + 0], 0));
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * m[i]);
  }
  const float2 dm4f = __half22float2(dm4);
  return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

// iqs in {0, 2, .., 30}; 16 positions tile one superblock.
__device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const block_q4_K* __restrict__ bq4_K,
    const block_q8_1* __restrict__ bq8_1, const int iqs) {
  int v[2];
  int u[4];
  float d8[2];

  const int bq8_offset = 2 * ((iqs / 2) / 4);
  const int* q4 =
      (const int*)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  v[0] = q4[0];
  v[1] = q4[4];

  const uint16_t* scales = (const uint16_t*)bq4_K->scales;
  uint16_t aux[2];
  const int j = bq8_offset / 2;
  if (j < 2) {
    aux[0] = scales[j + 0] & 0x3f3f;
    aux[1] = scales[j + 2] & 0x3f3f;
  } else {
    aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
    aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
  }
  const uint8_t* sc = (const uint8_t*)aux;
  const uint8_t* m = sc + 2;

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
    d8[i] = __low2float(bq8i->ds);
    const int* q8 = (const int*)bq8i->qs + ((iqs / 2) % 4);
    u[2 * i + 0] = q8[0];
    u[2 * i + 1] = q8[4];
  }
  return vec_dot_q4_K_q8_1_impl(v, u, sc, m, bq4_K->dm, d8);
}

__device__ __forceinline__ float q4k_half_warp_sum(float value) {
  const int lane = threadIdx.x & 31;
  const unsigned mask = lane < 16 ? 0x0000ffffu : 0xffff0000u;
#pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

// One warp owns one intermediate row (gate row `row`, up row
// `intermediate + row`).  Half-warps split the superblocks; within a
// superblock the 16 lanes cover the 16 vec_dot positions (iqs = 0,2..30).
__device__ __forceinline__ void q4_k_gate_up_row_dot(
    const block_q4_K* __restrict__ expert_weights,
    const block_q8_1* __restrict__ input_row, const int blocks_per_row,
    const int intermediate, const int row, float& gate, float& up) {
  const int lane = threadIdx.x & 31;
  const int half_id = lane >> 4;
  const int half_lane = lane & 15;
  const block_q4_K* gate_row = expert_weights + row * blocks_per_row;
  const block_q4_K* up_row =
      expert_weights + (intermediate + row) * blocks_per_row;

  gate = 0.0f;
  up = 0.0f;
  for (int block = half_id; block < blocks_per_row; block += 2) {
    const block_q8_1* q8 = input_row + block * (QK_K_Q4K / Q8_BLOCK);
    gate += vec_dot_q4_K_q8_1(gate_row + block, q8, 2 * half_lane);
    up += vec_dot_q4_K_q8_1(up_row + block, q8, 2 * half_lane);
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    gate += __shfl_down_sync(0xffffffffu, gate, offset);
    up += __shfl_down_sync(0xffffffffu, up, offset);
  }
}

// A 32-warp CTA computes one complete Q8_1 output block; warp 0 performs the
// SwiGLU/quant epilogue with the route weight folded in.
template <int TOP_K>
__global__ __launch_bounds__(1024, 1) void q4_k_gate_up_swiglu_q8_1_decode(
    const void* __restrict__ weights, const block_q8_1* __restrict__ input,
    block_q8_1* __restrict__ output, const int* __restrict__ topk_ids,
    const float* __restrict__ route_weights,
    const int64_t expert_stride_bytes, const int hidden,
    const int intermediate, const int tokens, const int experts,
    const float swiglu_limit) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int route = blockIdx.y;
  const int token = route / TOP_K;
  const int row = blockIdx.x * 32 + warp;
  const int expert = route < tokens * TOP_K ? topk_ids[route] : -1;
  const int blocks_per_mid = intermediate / Q8_BLOCK;
  const int blocks_per_row = hidden / QK_K_Q4K;
  const int q8_blocks_per_row = hidden / Q8_BLOCK;

  float gate = 0.0f;
  float up = 0.0f;
  if (expert >= 0 && expert < experts && row < intermediate) {
    const block_q4_K* expert_weights = reinterpret_cast<const block_q4_K*>(
        reinterpret_cast<const char*>(weights) +
        int64_t(expert) * expert_stride_bytes);
    q4_k_gate_up_row_dot(expert_weights,
                         input + int64_t(token) * q8_blocks_per_row,
                         blocks_per_row, intermediate, row, gate, up);
  }

  __shared__ float values[32];
  if (lane == 0) {
    float value = 0.0f;
    if (expert >= 0 && expert < experts && row < intermediate) {
      if (swiglu_limit > 0.0f) {
        gate = fminf(gate, swiglu_limit);
        up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
      }
      value = (gate / (1.0f + expf(-gate))) * up * route_weights[route];
      if (!isfinite(value)) {
        value = 0.0f;
      }
    }
    values[warp] = value;
  }
  __syncthreads();

  if (warp == 0) {
    const float value = values[lane];
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
    out->qs[lane] = quant;
    if (lane == 0) {
      out->ds = __floats2half2_rn(scale, sum);
    }
  }
}

// Down leg: route weights are already folded into quant_mid, so each half
// warp computes one output row across all routes and writes the final token
// result directly (no [token, route, hidden] tensor is materialized).
template <typename out_t, int TOP_K>
__global__ __launch_bounds__(256, 1) void q4_k_down_weighted_sum(
    const void* __restrict__ vw, const block_q8_1* __restrict__ quant_mid,
    const int* __restrict__ topk_ids, out_t* __restrict__ output,
    const int64_t exp_stride, const int intermediate, const int out_rows,
    const int tokens, const int experts) {
  const int token = blockIdx.y;
  const int half_lane = threadIdx.x & 15;
  const int row_lane = threadIdx.x >> 4;
  const int blocks_per_weight_row = intermediate / QK_K_Q4K;
  const int blocks_per_mid = intermediate / Q8_BLOCK;

#pragma unroll
  for (int row_step = 0; row_step < 4; ++row_step) {
    const int row = blockIdx.x * 64 + row_lane + row_step * 16;
    if (token >= tokens || row >= out_rows) {
      continue;
    }
    float total = 0.0f;
#pragma unroll
    for (int slot = 0; slot < TOP_K; ++slot) {
      const int route = token * TOP_K + slot;
      const int expert = topk_ids[route];
      float value = 0.0f;
      if (expert >= 0 && expert < experts) {
        const block_q4_K* weight = reinterpret_cast<const block_q4_K*>(
            static_cast<const char*>(vw) + int64_t(expert) * exp_stride);
        const block_q8_1* input = quant_mid + route * blocks_per_mid;
        for (int block = 0; block < blocks_per_weight_row; ++block) {
          value += vec_dot_q4_K_q8_1(
              weight + row * blocks_per_weight_row + block,
              input + block * (QK_K_Q4K / Q8_BLOCK), 2 * half_lane);
        }
      }
      value = q4k_half_warp_sum(value);
      if (half_lane == 0) {
        total += isfinite(value) ? value : 0.0f;
      }
    }
    if (half_lane == 0) {
      output[int64_t(token) * out_rows + row] = out_t(total);
    }
  }
}

inline void launch_q4_k_gate_up_swiglu_q8_1_decode(
    const void* weights, const void* input, void* output,
    const int* topk_ids, const float* route_weights,
    const int64_t expert_stride_bytes, const int hidden,
    const int intermediate, const int tokens, const int top_k,
    const int experts, const float swiglu_limit, cudaStream_t stream) {
  const dim3 grid((intermediate + 31) / 32, tokens * top_k, 1);
  if (top_k == 6) {
    q4_k_gate_up_swiglu_q8_1_decode<6><<<grid, 1024, 0, stream>>>(
        weights, static_cast<const block_q8_1*>(input),
        static_cast<block_q8_1*>(output), topk_ids, route_weights,
        expert_stride_bytes, hidden, intermediate, tokens, experts,
        swiglu_limit);
  } else {
    q4_k_gate_up_swiglu_q8_1_decode<8><<<grid, 1024, 0, stream>>>(
        weights, static_cast<const block_q8_1*>(input),
        static_cast<block_q8_1*>(output), topk_ids, route_weights,
        expert_stride_bytes, hidden, intermediate, tokens, experts,
        swiglu_limit);
  }
}

template <typename out_t>
inline void launch_q4_k_down_weighted_sum(
    const void* w, const void* quant_mid, const int* topk_ids, out_t* output,
    const int64_t exp_stride, const int intermediate, const int out_rows,
    const int tokens, const int top_k, const int experts,
    cudaStream_t stream) {
  const dim3 grid((out_rows + 63) / 64, tokens, 1);
  if (top_k == 6) {
    q4_k_down_weighted_sum<out_t, 6><<<grid, 256, 0, stream>>>(
        w, static_cast<const block_q8_1*>(quant_mid), topk_ids, output,
        exp_stride, intermediate, out_rows, tokens, experts);
  } else {
    q4_k_down_weighted_sum<out_t, 8><<<grid, 256, 0, stream>>>(
        w, static_cast<const block_q8_1*>(quant_mid), topk_ids, output,
        exp_stride, intermediate, out_rows, tokens, experts);
  }
}

}  // namespace quixi_q4k
