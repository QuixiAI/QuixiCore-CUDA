/**
 * @file
 * @brief Correctness + timing harness for the Ampere MXFP4 MoE kernel family.
 *
 * Build (set the arch for YOUR host: sm_80 on A100, sm_86 on RTX 3090):
 *   nvcc mxfp4_moe_ampere_test.cu -std=c++17 -O3 -arch=sm_80 -o mxfp4_moe_ampere_test.out
 *   CUDA_VISIBLE_DEVICES=0 ./mxfp4_moe_ampere_test.out
 *
 * Checks, in order:
 *   1. fused decode path (W1 gate/up + SwiGLU + Q8_1 emit, weighted down) vs
 *      an fp64 host reference over random MXFP4 weights, raw AND repacked
 *      layouts, tokens {1, 8, 48}
 *   2. grouped tensor-core tile (moe_align-style padded metadata) vs the same
 *      reference
 *   3. segmented pipeline (histogram/prefix/scatter + fused-SwiGLU W1 + W2 +
 *      deterministic reduce) at tokens {65, 300}, including -1 routes; covers
 *      both the J=16 and J=64 tiles
 *   4. segmented pipeline timing at the DSV4 TP4 shape
 *      (E=256, hidden=4096, I=512, tokens=2048)
 *
 * Tolerance: rel max-error < 0.03 vs fp64 (the Q8_1 activation and Q8_1 mid
 * quantization dominate; the MXFP4 integer decode itself is exact).
 */
#include "mxfp4_moe_ampere.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

using namespace tmq_a100;

#define CK(x)                                                                 \
    do {                                                                      \
        cudaError_t e = (x);                                                  \
        if (e != cudaSuccess) {                                               \
            printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// True e2m1 magnitudes (the device table is 2x these with 0.5 folded into the
// block scale).
static const double kMxfp4Values[16] = {0.0, 0.5,  1.0,  1.5,  2.0,  3.0,
                                        4.0, 6.0,  -0.0, -0.5, -1.0, -1.5,
                                        -2.0, -3.0, -4.0, -6.0};

// Random raw MXFP4 tensor [experts, rows, cols/32*17] plus its fp64
// dequantization. Layout matches the GGUF MMVQ convention (and the SlimServe
// test _random_mxfp4): per 17-byte block, one e8m0 scale byte then 16 code
// bytes whose LOW nibbles supply values [0,16) and HIGH nibbles [16,32).
struct MxTensor {
    int experts, rows, cols;
    std::vector<uint8_t> raw;   // [experts][rows*cols/32*17]
    std::vector<double> deq;    // [experts][rows][cols] (empty if !want_deq)
    int64_t expert_stride() const { return (int64_t)rows * (cols / 32) * 17; }
};

static MxTensor random_mxfp4(int experts, int rows, int cols, unsigned seed,
                             bool want_deq = true) {
    MxTensor t;
    t.experts = experts;
    t.rows = rows;
    t.cols = cols;
    const int blocks = cols / 32;
    t.raw.resize((size_t)experts * rows * blocks * 17);
    if (want_deq) t.deq.resize((size_t)experts * rows * cols);
    std::mt19937 rng(seed);
    for (size_t bi = 0; bi < (size_t)experts * rows * blocks; ++bi) {
        uint8_t* b = t.raw.data() + bi * 17;
        // Scales near 1.0 (e8m0 exponents around 127) keep values sane.
        b[0] = (uint8_t)(121 + rng() % 10);
        for (int j = 0; j < 16; ++j) b[1 + j] = (uint8_t)(rng() & 0xFF);
        if (!want_deq) continue;
        const double scale = ldexp(1.0, (int)b[0] - 127);
        double* d = t.deq.data() + bi * 32;
        for (int j = 0; j < 16; ++j) {
            d[j] = scale * kMxfp4Values[b[1 + j] & 0xF];
            d[16 + j] = scale * kMxfp4Values[b[1 + j] >> 4];
        }
    }
    return t;
}

static half2 host_half2(float a, float b) {
    __half h[2] = {__float2half(a), __float2half(b)};
    half2 r;
    memcpy(&r, h, 4);
    return r;
}

// Host Q8_1 quantization: 32-value blocks, d = amax/127, ds = (d, sum).
static std::vector<block_q8_1> quantize_q8_1_host(const std::vector<float>& x,
                                                  int rows, int cols) {
    const int blocks = cols / 32;
    std::vector<block_q8_1> out((size_t)rows * blocks);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < blocks; ++b) {
            const float* v = x.data() + (size_t)r * cols + b * 32;
            float amax = 0.f, sum = 0.f;
            for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(v[j]));
            const float d = amax / 127.f;
            const float id = amax > 0.f ? 127.f / amax : 0.f;
            block_q8_1& blk = out[(size_t)r * blocks + b];
            for (int j = 0; j < 32; ++j) {
                blk.qs[j] = (int8_t)lrintf(v[j] * id);
                sum += v[j];
            }
            blk.ds = host_half2(d, sum);
        }
    }
    return out;
}

// fp64 MoE reference: SwiGLU(W1 x) with route weight, then W2, accumulated
// per token over valid routes. Matches the SlimServe pytest reference.
static std::vector<double> moe_ref(const MxTensor& w1, const MxTensor& w2,
                                   const std::vector<float>& x,
                                   const std::vector<int>& ids,
                                   const std::vector<float>& rw, int tokens,
                                   int top_k, int hidden, int inter) {
    std::vector<double> out((size_t)tokens * hidden, 0.0);
    std::vector<double> act(inter);
    for (int t = 0; t < tokens; ++t) {
        for (int k = 0; k < top_k; ++k) {
            const int e = ids[(size_t)t * top_k + k];
            if (e < 0) continue;
            const double* w1e = w1.deq.data() + (size_t)e * (2 * inter) * hidden;
            const double* w2e = w2.deq.data() + (size_t)e * hidden * inter;
            for (int i = 0; i < inter; ++i) {
                double gate = 0.0, up = 0.0;
                for (int h = 0; h < hidden; ++h) {
                    const double xv = x[(size_t)t * hidden + h];
                    gate += w1e[(size_t)i * hidden + h] * xv;
                    up += w1e[(size_t)(inter + i) * hidden + h] * xv;
                }
                act[i] = (gate / (1.0 + exp(-gate))) * up *
                         (double)rw[(size_t)t * top_k + k];
            }
            for (int h = 0; h < hidden; ++h) {
                double acc = 0.0;
                for (int i = 0; i < inter; ++i)
                    acc += w2e[(size_t)h * inter + i] * act[i];
                out[(size_t)t * hidden + h] += acc;
            }
        }
    }
    return out;
}

static double rel_max_err(const std::vector<float>& got,
                          const std::vector<double>& want) {
    double worst = 0.0, mag = 1.0;
    for (size_t i = 0; i < want.size(); ++i) {
        if (!std::isfinite((double)got[i])) return 1e30;
        worst = fmax(worst, fabs((double)got[i] - want[i]));
        mag = fmax(mag, fabs(want[i]));
    }
    return worst / mag;
}

// Random distinct top-k expert ids per token (like torch.randperm[:top_k]).
static std::vector<int> random_topk(int tokens, int top_k, int experts,
                                    std::mt19937& rng) {
    std::vector<int> ids((size_t)tokens * top_k);
    std::vector<int> perm(experts);
    std::iota(perm.begin(), perm.end(), 0);
    for (int t = 0; t < tokens; ++t) {
        for (int k = 0; k < top_k; ++k) {
            std::uniform_int_distribution<int> d(k, experts - 1);
            std::swap(perm[k], perm[d(rng)]);
            ids[(size_t)t * top_k + k] = perm[k];
        }
    }
    return ids;
}

int main() {
    const int E = 16, TOPK = 6, HID = 256, INTER = 256;
    printf("MXFP4 MoE Ampere harness: E=%d top_k=%d hidden=%d intermediate=%d\n",
           E, TOPK, HID, INTER);

    // ---- shared small-shape weights: W1 [E, 2I, HID], W2 [E, HID, I]
    MxTensor w1 = random_mxfp4(E, 2 * INTER, HID, 1);
    MxTensor w2 = random_mxfp4(E, HID, INTER, 2);
    uint8_t *d_w1, *d_w2, *d_w1p, *d_w2p;
    CK(cudaMalloc(&d_w1, w1.raw.size()));
    CK(cudaMalloc(&d_w2, w2.raw.size()));
    CK(cudaMalloc(&d_w1p, w1.raw.size()));
    CK(cudaMalloc(&d_w2p, w2.raw.size()));
    CK(cudaMemcpy(d_w1, w1.raw.data(), w1.raw.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w2, w2.raw.data(), w2.raw.size(), cudaMemcpyHostToDevice));
    launch_repack_mxfp4_experts(d_w1, d_w1p, E, w1.expert_stride() / 17,
                                w1.expert_stride(), 0);
    launch_repack_mxfp4_experts(d_w2, d_w2p, E, w2.expert_stride() / 17,
                                w2.expert_stride(), 0);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 0.3f);
    bool all_pass = true;

    // ---- 1. fused decode path (raw + repacked) vs fp64 reference
    printf("  [1] fused decode (W1 SwiGLU Q8_1 emit + weighted down):\n");
    for (int tokens : {1, 8, 48}) {
        const int routes = tokens * TOPK;
        std::vector<float> x((size_t)tokens * HID);
        for (auto& v : x) v = nd(rng);
        auto q8 = quantize_q8_1_host(x, tokens, HID);
        auto ids = random_topk(tokens, TOPK, E, rng);
        std::vector<float> rw((size_t)routes);
        std::uniform_real_distribution<float> ud(0.25f, 1.25f);
        for (auto& v : rw) v = ud(rng);

        block_q8_1 *d_in, *d_mid;
        int* d_ids;
        float *d_rw, *d_out;
        CK(cudaMalloc(&d_in, q8.size() * sizeof(block_q8_1)));
        CK(cudaMalloc(&d_mid, (size_t)routes * (INTER / 32) * sizeof(block_q8_1)));
        CK(cudaMalloc(&d_ids, ids.size() * 4));
        CK(cudaMalloc(&d_rw, rw.size() * 4));
        CK(cudaMalloc(&d_out, (size_t)tokens * HID * 4));
        CK(cudaMemcpy(d_in, q8.data(), q8.size() * sizeof(block_q8_1),
                      cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_rw, rw.data(), rw.size() * 4, cudaMemcpyHostToDevice));

        const auto ref = moe_ref(w1, w2, x, ids, rw, tokens, TOPK, HID, INTER);
        std::vector<float> got((size_t)tokens * HID);
        for (int repacked = 0; repacked < 2; ++repacked) {
            launch_mxfp4_gate_up_swiglu_q8_1_decode(
                repacked ? d_w1p : d_w1, d_in, d_mid, d_ids, d_rw,
                w1.expert_stride(), HID, HID / 32, INTER, tokens, TOPK, E, 0.0f,
                repacked, 0);
            launch_mxfp4_down_sum<float>(repacked ? d_w2p : d_w2, d_mid, d_ids,
                                         d_out, w2.expert_stride(), INTER, HID,
                                         tokens, TOPK, E, repacked, 0);
            CK(cudaDeviceSynchronize());
            CK(cudaGetLastError());
            CK(cudaMemcpy(got.data(), d_out, got.size() * 4,
                          cudaMemcpyDeviceToHost));
            const double rel = rel_max_err(got, ref);
            const bool ok = rel < 0.03;
            all_pass &= ok;
            printf("      tokens=%2d %-8s rel %.4f  %s\n", tokens,
                   repacked ? "repacked" : "raw", rel, ok ? "PASS" : "FAIL");
        }
        CK(cudaFree(d_in));
        CK(cudaFree(d_mid));
        CK(cudaFree(d_ids));
        CK(cudaFree(d_rw));
        CK(cudaFree(d_out));
    }

    // ---- 2. grouped tensor-core tile through moe_align-style metadata
    {
        const int tokens = 65, routes = tokens * TOPK, nrows = 2 * INTER;
        std::vector<float> x((size_t)tokens * HID);
        for (auto& v : x) v = nd(rng);
        auto q8 = quantize_q8_1_host(x, tokens, HID);
        auto ids = random_topk(tokens, TOPK, E, rng);

        // moe_align_block_size equivalent: per-expert route lists padded to 64.
        std::vector<std::vector<int>> per_e(E);
        for (int r = 0; r < routes; ++r) per_e[ids[r]].push_back(r);
        std::vector<int> sorted_ids, expert_ids;
        for (int e = 0; e < E; ++e) {
            if (per_e[e].empty()) continue;
            const int tiles = (int)(per_e[e].size() + 63) / 64;
            for (int t = 0; t < tiles; ++t) expert_ids.push_back(e);
            for (int i = 0; i < tiles * 64; ++i)
                sorted_ids.push_back(i < (int)per_e[e].size() ? per_e[e][i]
                                                             : routes);
        }
        const int ncols_pad = (int)sorted_ids.size();
        int num_post[1] = {ncols_pad};

        block_q8_1* d_in;
        int *d_sorted, *d_eids, *d_npost;
        float* d_dst;
        CK(cudaMalloc(&d_in, q8.size() * sizeof(block_q8_1)));
        CK(cudaMalloc(&d_sorted, sorted_ids.size() * 4));
        CK(cudaMalloc(&d_eids, expert_ids.size() * 4));
        CK(cudaMalloc(&d_npost, 4));
        CK(cudaMalloc(&d_dst, (size_t)routes * nrows * 4));
        CK(cudaMemcpy(d_in, q8.data(), q8.size() * sizeof(block_q8_1),
                      cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_sorted, sorted_ids.data(), sorted_ids.size() * 4,
                      cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_eids, expert_ids.data(), expert_ids.size() * 4,
                      cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_npost, num_post, 4, cudaMemcpyHostToDevice));

        launch_moe_mxfp4_mmq_v2<float>(d_in, d_w1, d_dst, d_sorted, d_eids,
                                       d_npost, w1.expert_stride(), HID, nrows,
                                       tokens, HID, nrows, TOPK, ncols_pad,
                                       false, 0);
        CK(cudaDeviceSynchronize());
        CK(cudaGetLastError());

        std::vector<float> got((size_t)routes * nrows);
        CK(cudaMemcpy(got.data(), d_dst, got.size() * 4,
                      cudaMemcpyDeviceToHost));
        std::vector<double> ref((size_t)routes * nrows, 0.0);
        for (int r = 0; r < routes; ++r) {
            const int e = ids[r], t = r / TOPK;
            const double* we = w1.deq.data() + (size_t)e * nrows * HID;
            for (int row = 0; row < nrows; ++row) {
                double acc = 0.0;
                for (int h = 0; h < HID; ++h)
                    acc += we[(size_t)row * HID + h] * x[(size_t)t * HID + h];
                ref[(size_t)r * nrows + row] = acc;
            }
        }
        const double rel = rel_max_err(got, ref);
        const bool ok = rel < 0.03;
        all_pass &= ok;
        printf("  [2] mma tile (128x64, tokens=%d, %d col tiles): rel %.4f  %s\n",
               tokens, ncols_pad / 64, rel, ok ? "PASS" : "FAIL");
        CK(cudaFree(d_in));
        CK(cudaFree(d_sorted));
        CK(cudaFree(d_eids));
        CK(cudaFree(d_npost));
        CK(cudaFree(d_dst));
    }

    // ---- 3. segmented pipeline (J=16 and J=64), with invalid routes
    printf("  [3] segmented pipeline (perm build + fused W1 + W2 + reduce):\n");
    for (int tokens : {65, 300}) {
        const int routes = tokens * TOPK;
        const bool use_j16 = routes < 1536;
        std::vector<float> x((size_t)tokens * HID);
        for (auto& v : x) v = nd(rng);
        auto q8 = quantize_q8_1_host(x, tokens, HID);
        auto ids = random_topk(tokens, TOPK, E, rng);
        // Drop ~10% of routes: -1 must contribute exactly nothing.
        const int drops = routes / 10;
        for (int i = 0; i < drops; ++i)
            ids[(size_t)(rng() % routes)] = -1;
        std::vector<float> rw((size_t)routes);
        std::uniform_real_distribution<float> ud(0.25f, 1.25f);
        for (auto& v : rw) v = ud(rng);

        block_q8_1 *d_in, *d_mid;
        int *d_ids, *d_meta;
        float *d_rw, *d_w2out, *d_out;
        CK(cudaMalloc(&d_in, q8.size() * sizeof(block_q8_1)));
        CK(cudaMalloc(&d_mid, (size_t)routes * (INTER / 32) * sizeof(block_q8_1)));
        CK(cudaMalloc(&d_ids, ids.size() * 4));
        CK(cudaMalloc(&d_meta, seg_meta_ints(E, routes) * 4));
        CK(cudaMalloc(&d_rw, rw.size() * 4));
        CK(cudaMalloc(&d_w2out, (size_t)routes * HID * 4));
        CK(cudaMalloc(&d_out, (size_t)tokens * HID * 4));
        CK(cudaMemcpy(d_in, q8.data(), q8.size() * sizeof(block_q8_1),
                      cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_rw, rw.data(), rw.size() * 4, cudaMemcpyHostToDevice));

        launch_moe_mxfp4_seg<float>(d_in, d_w1, d_w2, d_mid, d_w2out, d_out,
                                    d_ids, d_rw, d_meta, w1.expert_stride(),
                                    w2.expert_stride(), HID, HID / 32, INTER,
                                    HID, tokens, TOPK, E, 0.0f, use_j16, 0);
        CK(cudaDeviceSynchronize());
        CK(cudaGetLastError());

        std::vector<float> got((size_t)tokens * HID);
        CK(cudaMemcpy(got.data(), d_out, got.size() * 4,
                      cudaMemcpyDeviceToHost));
        const auto ref = moe_ref(w1, w2, x, ids, rw, tokens, TOPK, HID, INTER);
        const double rel = rel_max_err(got, ref);
        const bool ok = rel < 0.03;
        all_pass &= ok;
        printf("      tokens=%3d J=%-2d dropped=%2d: rel %.4f  %s\n", tokens,
               use_j16 ? 16 : 64, drops, rel, ok ? "PASS" : "FAIL");
        CK(cudaFree(d_in));
        CK(cudaFree(d_mid));
        CK(cudaFree(d_ids));
        CK(cudaFree(d_meta));
        CK(cudaFree(d_rw));
        CK(cudaFree(d_w2out));
        CK(cudaFree(d_out));
    }
    if (!all_pass) return 1;

    // ---- 4. segmented pipeline timing at the DSV4 TP4 shape
    {
        const int bE = 256, bTOPK = 8, bHID = 4096, bI = 512;
        const int tokens = 2048, routes = tokens * bTOPK;
        MxTensor bw1 = random_mxfp4(bE, 2 * bI, bHID, 11, /*want_deq=*/false);
        MxTensor bw2 = random_mxfp4(bE, bHID, bI, 12, /*want_deq=*/false);
        printf("  [4] segmented timing, DSV4 TP4 shape: E=%d top_k=%d "
               "hidden=%d I=%d tokens=%d (%.0f MiB W1 + %.0f MiB W2)\n",
               bE, bTOPK, bHID, bI, tokens,
               (double)bw1.raw.size() / (1 << 20),
               (double)bw2.raw.size() / (1 << 20));

        std::vector<float> x((size_t)tokens * bHID);
        for (auto& v : x) v = nd(rng);
        auto q8 = quantize_q8_1_host(x, tokens, bHID);
        auto ids = random_topk(tokens, bTOPK, bE, rng);
        std::vector<float> rw((size_t)routes);
        std::uniform_real_distribution<float> ud(0.25f, 1.25f);
        for (auto& v : rw) v = ud(rng);

        uint8_t *d_bw1, *d_bw2;
        block_q8_1 *d_in, *d_mid;
        int *d_ids, *d_meta;
        float *d_rw, *d_w2out, *d_out;
        CK(cudaMalloc(&d_bw1, bw1.raw.size()));
        CK(cudaMalloc(&d_bw2, bw2.raw.size()));
        CK(cudaMalloc(&d_in, q8.size() * sizeof(block_q8_1)));
        CK(cudaMalloc(&d_mid, (size_t)routes * (bI / 32) * sizeof(block_q8_1)));
        CK(cudaMalloc(&d_ids, ids.size() * 4));
        CK(cudaMalloc(&d_meta, seg_meta_ints(bE, routes) * 4));
        CK(cudaMalloc(&d_rw, rw.size() * 4));
        CK(cudaMalloc(&d_w2out, (size_t)routes * bHID * 4));
        CK(cudaMalloc(&d_out, (size_t)tokens * bHID * 4));
        CK(cudaMemcpy(d_bw1, bw1.raw.data(), bw1.raw.size(),
                      cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_bw2, bw2.raw.data(), bw2.raw.size(),
                      cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_in, q8.data(), q8.size() * sizeof(block_q8_1),
                      cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_rw, rw.data(), rw.size() * 4, cudaMemcpyHostToDevice));

        const bool use_j16 = routes < 1536;  // false here: J=64 prefill tile
        auto run = [&] {
            launch_moe_mxfp4_seg<float>(d_in, d_bw1, d_bw2, d_mid, d_w2out,
                                        d_out, d_ids, d_rw, d_meta,
                                        bw1.expert_stride(), bw2.expert_stride(),
                                        bHID, bHID / 32, bI, bHID, tokens,
                                        bTOPK, bE, 0.0f, use_j16, 0);
        };
        run();
        CK(cudaDeviceSynchronize());
        CK(cudaGetLastError());
        {   // sanity: output must be finite
            std::vector<float> got((size_t)tokens * bHID);
            CK(cudaMemcpy(got.data(), d_out, got.size() * 4,
                          cudaMemcpyDeviceToHost));
            for (float v : got)
                if (!std::isfinite(v)) {
                    printf("      non-finite output at bench shape -> FAIL\n");
                    return 1;
                }
        }

        // Weight bytes actually streamed: each real column tile reads its
        // expert's full W1 (across the I/64 row tiles) and full W2 (across
        // the out_row/128 row tiles) once.
        std::vector<int> cnt(bE, 0);
        for (int id : ids) ++cnt[id];
        long tiles_real = 0;
        double streamed = 0.0;
        const double we_bytes =
            (double)bw1.expert_stride() + (double)bw2.expert_stride();
        for (int e = 0; e < bE; ++e) {
            const int te = (cnt[e] + 63) / 64;
            tiles_real += te;
            streamed += te * we_bytes;
        }
        const int tiles_static = seg_col_tiles(routes, bE, 64);

        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        const int iters = 20;
        for (int i = 0; i < 5; ++i) run();
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for (int i = 0; i < iters; ++i) run();
        CK(cudaEventRecord(e1));
        CK(cudaDeviceSynchronize());
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        ms /= iters;
        printf("      column tiles: %ld real / %d static grid (J=64)\n",
               tiles_real, tiles_static);
        printf("      %8s %14s %14s\n", "ms/iter", "weightGB/s", "MoE tok/s");
        printf("      %8.3f %14.0f %14.0f\n", ms, streamed / (ms / 1e3) / 1e9,
               tokens / (ms / 1e3));
    }

    printf("ALL PASS\n");
    return 0;
}
