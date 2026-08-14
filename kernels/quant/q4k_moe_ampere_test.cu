/**
 * @file
 * @brief Correctness harness for the fused Q4_K MoE decode pair.
 *
 * Build:
 *   nvcc q4k_moe_ampere_test.cu -std=c++17 -O3 -arch=sm_80 -o q4k_moe_ampere_test.out
 *   CUDA_VISIBLE_DEVICES=0 ./q4k_moe_ampere_test.out
 *
 * Compares the two-kernel pipeline (gate/up + SwiGLU + route weight + Q8_1
 * emit, then Q8xQ4_K weighted down) against an fp64 host reference that
 * reproduces the quantization semantics (fp32 Q8_1 scale computation, fp16
 * stored scale, route weight folded before the mid quant).
 *
 * Metric: MEAN relative error. Recomputing the mid activation in fp64 flips
 * round(v/scale) by +-1 near .5 boundaries relative to the kernel's fp32
 * value; each flip perturbs one whole output row by ~1e-3 of the mean and
 * flips stack, so elementwise bounds cannot pass for ANY correct kernel
 * (established by lstsq flip decomposition in the SlimServe tree). Wrong
 * indexing or scales produce O(1) mean error; the flip floor is ~1e-3.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "q4k_moe_ampere.cuh"

using tmq_a100::block_q8_1;
using quixi_q4k::block_q4_K;

static constexpr int E = 8, TOP_K = 6, HIDDEN = 1024, INTER = 256,
                     OUT_ROW = 512;

#define CHECK(x)                                                     \
  do {                                                               \
    cudaError_t e = (x);                                             \
    if (e != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
              cudaGetErrorString(e), __FILE__, __LINE__);            \
      exit(1);                                                       \
    }                                                                \
  } while (0)

static void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d,
                             uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

static void dequant_q4k_row(const block_q4_K* row, int blocks, double* out) {
  for (int b = 0; b < blocks; ++b) {
    const block_q4_K& x = row[b];
    const float d = __half2float(__low2half(x.dm));
    const float dmin = __half2float(__high2half(x.dm));
    const uint8_t* q = x.qs;
    int is = 0;
    double* y = out + b * 256;
    for (int j = 0; j < 256; j += 64) {
      uint8_t sc, m;
      get_scale_min_k4(is + 0, x.scales, &sc, &m);
      const double d1 = double(d) * sc, m1 = double(dmin) * m;
      get_scale_min_k4(is + 1, x.scales, &sc, &m);
      const double d2 = double(d) * sc, m2 = double(dmin) * m;
      for (int l = 0; l < 32; ++l) y[l] = d1 * (q[l] & 0xF) - m1;
      for (int l = 0; l < 32; ++l) y[32 + l] = d2 * (q[l] >> 4) - m2;
      y += 64;
      q += 32;
      is += 2;
    }
  }
}

// Host Q8_1: fp32 scale for rounding, fp16 stored scale for dequant --
// exactly the device pipeline's semantics.
static void q8_1_quant(const double* x, int n, std::vector<block_q8_1>& out,
                       std::vector<double>* roundtrip) {
  const int blocks = n / 32;
  out.resize(blocks);
  if (roundtrip) roundtrip->resize(n);
  for (int b = 0; b < blocks; ++b) {
    float amax = 0.0f;
    for (int l = 0; l < 32; ++l)
      amax = fmaxf(amax, fabsf(float(x[b * 32 + l])));
    const float d = amax / 127.0f;
    const float d16 = __half2float(__float2half_rn(d));
    float sum = 0.0f;
    for (int l = 0; l < 32; ++l) {
      const int8_t q =
          amax == 0.0f ? 0 : (int8_t)roundf(float(x[b * 32 + l]) / d);
      out[b].qs[l] = q;
      sum += float(x[b * 32 + l]);
      if (roundtrip) (*roundtrip)[b * 32 + l] = double(q) * d16;
    }
    out[b].ds = __floats2half2_rn(d, sum);
  }
}

int main() {
  std::mt19937 rng(7);
  const int w1_blocks = HIDDEN / 256, w2_blocks = INTER / 256;
  std::vector<block_q4_K> w1(size_t(E) * 2 * INTER * w1_blocks);
  std::vector<block_q4_K> w2(size_t(E) * OUT_ROW * w2_blocks);
  auto fill = [&](std::vector<block_q4_K>& w) {
    for (auto& blk : w) {
      uint8_t* raw = reinterpret_cast<uint8_t*>(&blk);
      for (size_t i = 0; i < sizeof(block_q4_K); ++i)
        raw[i] = uint8_t(rng());
      blk.dm = __floats2half2_rn(0.02f, 0.02f);
    }
  };
  fill(w1);
  fill(w2);

  block_q4_K *d_w1, *d_w2;
  CHECK(cudaMalloc(&d_w1, w1.size() * sizeof(block_q4_K)));
  CHECK(cudaMalloc(&d_w2, w2.size() * sizeof(block_q4_K)));
  CHECK(cudaMemcpy(d_w1, w1.data(), w1.size() * sizeof(block_q4_K),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_w2, w2.data(), w2.size() * sizeof(block_q4_K),
                   cudaMemcpyHostToDevice));

  // Dequantized experts for the reference.
  std::vector<double> w1d(size_t(E) * 2 * INTER * HIDDEN);
  std::vector<double> w2d(size_t(E) * OUT_ROW * INTER);
  for (int e = 0; e < E; ++e)
    for (int r = 0; r < 2 * INTER; ++r)
      dequant_q4k_row(&w1[(size_t(e) * 2 * INTER + r) * w1_blocks], w1_blocks,
                      &w1d[(size_t(e) * 2 * INTER + r) * HIDDEN]);
  for (int e = 0; e < E; ++e)
    for (int r = 0; r < OUT_ROW; ++r)
      dequant_q4k_row(&w2[(size_t(e) * OUT_ROW + r) * w2_blocks], w2_blocks,
                      &w2d[(size_t(e) * OUT_ROW + r) * INTER]);

  int failures = 0;
  std::normal_distribution<double> gauss(0.0, 0.3);
  std::uniform_real_distribution<float> uw(0.25f, 1.25f);
  for (int tokens : {1, 5}) {
    for (int mask_first : {0, 1}) {
      std::vector<double> x(size_t(tokens) * HIDDEN);
      for (auto& v : x) v = gauss(rng);
      std::vector<int> ids(size_t(tokens) * TOP_K);
      std::vector<float> rw(size_t(tokens) * TOP_K);
      for (int t = 0; t < tokens; ++t)
        for (int s = 0; s < TOP_K; ++s) {
          ids[t * TOP_K + s] = mask_first && s == 0 ? -1 : int(rng() % E);
          rw[t * TOP_K + s] = uw(rng);
        }

      // Input Q8_1 (per token row) + roundtrip for the reference.
      std::vector<block_q8_1> xq;
      std::vector<double> xrt(size_t(tokens) * HIDDEN);
      for (int t = 0; t < tokens; ++t) {
        std::vector<block_q8_1> row;
        std::vector<double> rt;
        q8_1_quant(&x[size_t(t) * HIDDEN], HIDDEN, row, &rt);
        xq.insert(xq.end(), row.begin(), row.end());
        std::copy(rt.begin(), rt.end(), xrt.begin() + size_t(t) * HIDDEN);
      }

      block_q8_1 *d_xq, *d_mid;
      int* d_ids;
      float *d_rw, *d_out;
      CHECK(cudaMalloc(&d_xq, xq.size() * sizeof(block_q8_1)));
      CHECK(cudaMalloc(&d_mid, size_t(tokens) * TOP_K * (INTER / 32) *
                                   sizeof(block_q8_1)));
      CHECK(cudaMalloc(&d_ids, ids.size() * sizeof(int)));
      CHECK(cudaMalloc(&d_rw, rw.size() * sizeof(float)));
      CHECK(cudaMalloc(&d_out, size_t(tokens) * OUT_ROW * sizeof(float)));
      CHECK(cudaMemcpy(d_xq, xq.data(), xq.size() * sizeof(block_q8_1),
                       cudaMemcpyHostToDevice));
      CHECK(cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(int),
                       cudaMemcpyHostToDevice));
      CHECK(cudaMemcpy(d_rw, rw.data(), rw.size() * sizeof(float),
                       cudaMemcpyHostToDevice));

      quixi_q4k::launch_q4_k_gate_up_swiglu_q8_1_decode(
          d_w1, d_xq, d_mid, d_ids, d_rw,
          int64_t(2 * INTER) * w1_blocks * sizeof(block_q4_K), HIDDEN, INTER,
          tokens, TOP_K, E, 0.0f, nullptr);
      quixi_q4k::launch_q4_k_down_weighted_sum<float>(
          d_w2, d_mid, d_ids, d_out,
          int64_t(OUT_ROW) * w2_blocks * sizeof(block_q4_K), INTER, OUT_ROW,
          tokens, TOP_K, E, nullptr);
      CHECK(cudaDeviceSynchronize());

      std::vector<float> got(size_t(tokens) * OUT_ROW);
      CHECK(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));

      // fp64 reference with the kernel's quantization semantics.
      std::vector<double> want(size_t(tokens) * OUT_ROW, 0.0);
      for (int t = 0; t < tokens; ++t) {
        for (int s = 0; s < TOP_K; ++s) {
          const int e = ids[t * TOP_K + s];
          if (e < 0) continue;
          std::vector<double> v(INTER);
          for (int r = 0; r < INTER; ++r) {
            double gate = 0.0, up = 0.0;
            const double* g = &w1d[(size_t(e) * 2 * INTER + r) * HIDDEN];
            const double* u = &w1d[(size_t(e) * 2 * INTER + INTER + r) *
                                   HIDDEN];
            for (int k = 0; k < HIDDEN; ++k) {
              gate += g[k] * xrt[size_t(t) * HIDDEN + k];
              up += u[k] * xrt[size_t(t) * HIDDEN + k];
            }
            v[r] = (gate / (1.0 + exp(-gate))) * up *
                   double(rw[t * TOP_K + s]);
          }
          std::vector<block_q8_1> mq;
          std::vector<double> mrt;
          q8_1_quant(v.data(), INTER, mq, &mrt);
          for (int r = 0; r < OUT_ROW; ++r) {
            double acc = 0.0;
            const double* wrow = &w2d[(size_t(e) * OUT_ROW + r) * INTER];
            for (int k = 0; k < INTER; ++k) acc += wrow[k] * mrt[k];
            want[size_t(t) * OUT_ROW + r] += acc;
          }
        }
      }

      double err_sum = 0.0, mag_sum = 0.0;
      for (size_t i = 0; i < want.size(); ++i) {
        err_sum += fabs(double(got[i]) - want[i]);
        mag_sum += fabs(want[i]);
      }
      const double mean_rel = err_sum / (mag_sum > 0 ? mag_sum : 1.0);
      const bool ok = mean_rel < 5e-3;
      printf("tokens=%d masked=%d mean_rel=%.2e %s\n", tokens, mask_first,
             mean_rel, ok ? "PASS" : "FAIL");
      failures += !ok;

      cudaFree(d_xq);
      cudaFree(d_mid);
      cudaFree(d_ids);
      cudaFree(d_rw);
      cudaFree(d_out);
    }
  }
  printf(failures == 0 ? "ALL PASS\n" : "FAILURES: %d\n", failures);
  return failures != 0;
}
