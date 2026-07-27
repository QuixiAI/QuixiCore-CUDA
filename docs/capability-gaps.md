# CUDA Capability Gaps

Inventory date: 2026-07-27.

This file compares the CUDA backend with the **263-operation semantic
union** and **17 exact quant-format IDs** recorded in the
[QuixiCore umbrella capability map](https://github.com/QuixiAI/QuixiCore/blob/main/matrices/capability-map.md).

Backend snapshot: `main` @ `d959679b0163`.

Normalized adapter stubs, including planned practical inference and fused
operations, are indexed in `.quixicore/kernel-stubs.yaml` and declared in
`include/quixicore/cuda/contract_stubs.hpp`. Their counts differ from this
exact-ID evidence comparison because aliases are collapsed and planned
contracts are included.

Union source revisions: CUDA d959679b0163; Metal a6d984377288; ROCm 636ae5ae983f; XPU 67c70fe4dc0c; CPU 0159223979db.

## How gaps are classified

- **family-only metadata**: this backend marks the family implemented but
  does not publish the exact operation ID. This is an enumeration/evidence
  gap, not proof that the semantic kernel is missing.
- **partial-family coverage**: the exact ID is absent and the backend marks
  the family partial.
- **capability-gated**: the family or operation depends on hardware/runtime
  conditions and is not an unconditional capability.
- **planned family**, **no family claim**, **partial operation**, and
  **experimental operation** are implementation or maturity gaps relative
  to a fully evidenced union capability.

Exact accelerator stage/layout aliases remain separate because the umbrella
map preserves published operation IDs. A backend may close a metadata gap
by documenting a proven semantic collapse instead of adding duplicate code.

## Summary

| Measure | Count |
|---|---:|
| Union operation capabilities | 263 |
| Fully implemented or semantically mapped | 0 |
| Operation gaps or enumeration gaps | 263 |
| capability-gated | 4 |
| family-only metadata | 222 |
| no family claim | 37 |
| Union quant-format IDs | 17 |
| Fully declared quant-format IDs | 6 |
| Quant-format gaps or missing declarations | 11 |
| quant: no exact format declaration | 11 |

## Operation gap list

| Union family | Capability | Gap class |
|---|---|---|
| Norms | `layernorm` | family-only metadata |
| Norms | `norm_quant` | family-only metadata |
| Norms | `qk_norm_rope` | family-only metadata |
| Norms | `qk_norm_rope_kv_f16` | family-only metadata |
| Norms | `qk_norm_rope_positioned` | family-only metadata |
| Norms | `rms_norm` | family-only metadata |
| Norms | `rms_norm_residual_next` | family-only metadata |
| Norms | `rms_residual_next` | family-only metadata |
| Norms | `rmsnorm` | family-only metadata |
| Activations | `elementwise` | family-only metadata |
| Activations | `gelu` | family-only metadata |
| Activations | `gelu_backward` | family-only metadata |
| Activations | `glu` | family-only metadata |
| Activations | `leaky_relu` | family-only metadata |
| Activations | `sigmoid_mul` | family-only metadata |
| Activations | `sigmoid_mul_backward` | family-only metadata |
| Activations | `silu` | family-only metadata |
| Activations | `silu_backward` | family-only metadata |
| Activations | `softmax` | family-only metadata |
| Activations | `softmax_backward` | family-only metadata |
| Activations | `unary` | family-only metadata |
| Activations | `value_clip` | family-only metadata |
| Attention | `attention` | family-only metadata |
| Attention | `attn_composites` | family-only metadata |
| Attention | `attn_decode_bh` | family-only metadata |
| Attention | `attn_fwd_sg_d256` | family-only metadata |
| Attention | `biased_attention` | family-only metadata |
| Attention | `cross_attention` | family-only metadata |
| Attention | `decode_cache_attention` | family-only metadata |
| Attention | `gqa` | family-only metadata |
| Attention | `gqa_backward` | family-only metadata |
| Attention | `gqa_causal` | family-only metadata |
| Attention | `gqa_causal_backward` | family-only metadata |
| Attention | `gqa_swa` | family-only metadata |
| Attention | `mrope` | family-only metadata |
| Attention | `paged_attention_q8_0` | family-only metadata |
| Attention | `rope` | family-only metadata |
| Attention | `rope_variants` | family-only metadata |
| Attention | `rotary` | family-only metadata |
| Attention | `rotary_positioned` | family-only metadata |
| Linear attention | `based` | family-only metadata |
| Linear attention | `gated_linear_attention` | family-only metadata |
| Linear attention | `gdn_gate_beta` | family-only metadata |
| Linear attention | `gdn_gated_rmsnorm` | family-only metadata |
| Linear attention | `gdn_qkv_prepare` | family-only metadata |
| Linear attention | `gdn_recur` | family-only metadata |
| Linear attention | `gdn_recurrence` | family-only metadata |
| Linear attention | `gdn_short_conv` | family-only metadata |
| Linear attention | `hedgehog` | family-only metadata |
| Linear attention | `linear_attention_unnormalized` | family-only metadata |
| Linear attention | `linear_attn` | family-only metadata |
| Linear attention | `rwkv_wkv6` | family-only metadata |
| Linear attention | `rwkv_wkv7` | family-only metadata |
| State-space models | `dsv4_hc_comb` | family-only metadata |
| State-space models | `dsv4_hc_post` | family-only metadata |
| State-space models | `dsv4_hc_pre` | family-only metadata |
| State-space models | `fftconv` | family-only metadata |
| State-space models | `mamba2_backward` | family-only metadata |
| State-space models | `selective_scan` | family-only metadata |
| State-space models | `ssd_chunked_backward` | family-only metadata |
| State-space models | `ssd_decode` | family-only metadata |
| Dense matmul and projections | `bf16fp32_matmul` | family-only metadata |
| Dense matmul and projections | `complex_gemm` | family-only metadata |
| Dense matmul and projections | `decode_linear` | family-only metadata |
| Dense matmul and projections | `decode_linear_epilogue` | family-only metadata |
| Dense matmul and projections | `decode_linear_epilogue_dense` | family-only metadata |
| Dense matmul and projections | `decode_linear_epilogue_packed` | family-only metadata |
| Dense matmul and projections | `decode_linear_q8` | family-only metadata |
| Dense matmul and projections | `decode_linear_residual` | family-only metadata |
| Dense matmul and projections | `decode_swiglu` | family-only metadata |
| Dense matmul and projections | `decode_swiglu_dense` | family-only metadata |
| Dense matmul and projections | `decode_swiglu_packed` | family-only metadata |
| Dense matmul and projections | `dense_gemm` | family-only metadata |
| Dense matmul and projections | `flux` | family-only metadata |
| Dense matmul and projections | `fp8fp32_matmul` | family-only metadata |
| Dense matmul and projections | `gemm_gate_residual` | family-only metadata |
| Dense matmul and projections | `gemm_staged` | family-only metadata |
| Dense matmul and projections | `grouped_gemm` | family-only metadata |
| Dense matmul and projections | `int8_matmul` | family-only metadata |
| Dense matmul and projections | `lora_apply` | family-only metadata |
| Dense matmul and projections | `lora_apply_direct_f16` | family-only metadata |
| Dense matmul and projections | `matmul_custom` | family-only metadata |
| Dense matmul and projections | `mxfp8_matmul` | family-only metadata |
| Dense matmul and projections | `nvfp4_matmul` | family-only metadata |
| Dense matmul and projections | `scaled_matmul` | family-only metadata |
| Quantization | `act_quant_int8` | family-only metadata |
| Quantization | `base_q_dequant` | family-only metadata |
| Quantization | `base_q_embedding` | family-only metadata |
| Quantization | `base_q_fused_consumers` | family-only metadata |
| Quantization | `base_q_gemm` | family-only metadata |
| Quantization | `base_q_gemv` | family-only metadata |
| Quantization | `base_q_gemv_qkv` | family-only metadata |
| Quantization | `base_q_gemv_swiglu` | family-only metadata |
| Quantization | `base_q_lm_head_argmax` | family-only metadata |
| Quantization | `base_q_moe_gemm` | family-only metadata |
| Quantization | `base_q_moe_swiglu` | family-only metadata |
| Quantization | `base_q_qkv` | family-only metadata |
| Quantization | `base_q_swiglu` | family-only metadata |
| Quantization | `calibration_absmax` | family-only metadata |
| Quantization | `dequant_gather` | family-only metadata |
| Quantization | `fake_quant_float8` | family-only metadata |
| Quantization | `fake_quant_int8` | family-only metadata |
| Quantization | `fp8_gemm` | family-only metadata |
| Quantization | `gguf_gemv` | family-only metadata |
| Quantization | `lm_head` | family-only metadata |
| Quantization | `lm_head_beam_advance` | family-only metadata |
| Quantization | `lm_head_candidates` | family-only metadata |
| Quantization | `lm_head_masked` | family-only metadata |
| Quantization | `mxfp4_gemv` | family-only metadata |
| Quantization | `nvfp4_gemv` | family-only metadata |
| Quantization | `qgeglu` | family-only metadata |
| Quantization | `qgemm` | family-only metadata |
| Quantization | `qgemm_backward_input` | family-only metadata |
| Quantization | `qgemm_int` | family-only metadata |
| Quantization | `qgemm_int8` | family-only metadata |
| Quantization | `qgemm_q4q8` | family-only metadata |
| Quantization | `qgemv` | family-only metadata |
| Quantization | `qgemv_int4` | family-only metadata |
| Quantization | `qgemv_q4_0_f32_qkv` | family-only metadata |
| Quantization | `qgemv_q4_0_f32_up_gate` | family-only metadata |
| Quantization | `qgemv_q4_0_f32_up_gate_gelu` | family-only metadata |
| Quantization | `qkv_proj_fused` | family-only metadata |
| Quantization | `quant_rt` | family-only metadata |
| Quantization | `quantize_int4_group` | family-only metadata |
| Quantization | `quantized_embedding` | family-only metadata |
| Quantization | `quantized_embedding_bag` | family-only metadata |
| Quantization | `ternary_code_flip_count` | family-only metadata |
| Quantization | `ternary_pack` | family-only metadata |
| Quantization | `ternary_stats` | family-only metadata |
| Quantization | `ternary_unpack` | family-only metadata |
| Quantization | `tq2_0_pack` | family-only metadata |
| Quantization | `tq2_0_unpack` | family-only metadata |
| Quantization | `turboquant` | family-only metadata |
| Mixture of experts | `moe` | family-only metadata |
| Mixture of experts | `moe_finalize_backward` | family-only metadata |
| Mixture of experts | `moe_gather_backward` | family-only metadata |
| Mixture of experts | `moe_grouped_gemm_backward_input` | family-only metadata |
| Mixture of experts | `moe_grouped_gemm_backward_weight` | family-only metadata |
| Mixture of experts | `moe_grouped_qgemm` | family-only metadata |
| Mixture of experts | `moe_grouped_qswiglu` | family-only metadata |
| Mixture of experts | `moe_quant` | family-only metadata |
| Mixture of experts | `moe_route_grouped` | family-only metadata |
| Mixture of experts | `moe_route_topk` | family-only metadata |
| Sampling | `argmax` | family-only metadata |
| Sampling | `logits_softcap` | family-only metadata |
| Sampling | `sample_categorical` | family-only metadata |
| Sampling | `top_k_renorm` | family-only metadata |
| Sampling | `top_k_sample` | family-only metadata |
| Sampling | `top_p_renorm` | family-only metadata |
| Serving and caches | `embedding_lookup` | family-only metadata |
| Serving and caches | `embedding_lookup_types` | family-only metadata |
| Serving and caches | `kv_cache_copy_blocks_q8_0` | family-only metadata |
| Serving and caches | `kv_cache_gather` | family-only metadata |
| Serving and caches | `kv_cache_gather_bitnet_kv3` | family-only metadata |
| Serving and caches | `kv_cache_gather_q8_0` | family-only metadata |
| Serving and caches | `kv_cache_q8_0` | family-only metadata |
| Serving and caches | `kv_cache_scatter` | family-only metadata |
| Serving and caches | `kv_cache_scatter_bitnet_kv3` | family-only metadata |
| Serving and caches | `kv_cache_scatter_q8_0` | family-only metadata |
| Serving and caches | `masked_mean_pool_rms_l2` | family-only metadata |
| Serving and caches | `mean_pool_rms_l2` | family-only metadata |
| Serving and caches | `paged_attention_advanced` | family-only metadata |
| Serving and caches | `paged_attention_bitnet_kv3` | family-only metadata |
| Serving and caches | `paged_attention_turboquant` | family-only metadata |
| Serving and caches | `quantized_attention` | family-only metadata |
| Serving and caches | `serving` | family-only metadata |
| Optimizers | `adamw` | family-only metadata |
| Optimizers | `adamw_masked` | family-only metadata |
| Optimizers | `sgd` | family-only metadata |
| Collectives | `broadcast` | capability-gated |
| Collectives | `fp8_gemm_collectives` | capability-gated |
| Collectives | `reduce_sum` | capability-gated |
| Collectives | `standalone_collectives` | capability-gated |
| Vision | `add_relative_position_2d` | no family claim |
| Vision | `avg_pool2d_tokens` | no family claim |
| Vision | `edge_mlp_256x7` | no family claim |
| Vision | `extract_patches_2d` | no family claim |
| Vision | `extract_patches_3d` | no family claim |
| Vision | `factorized_position_2d` | no family claim |
| Vision | `get_relative_position` | no family claim |
| Vision | `interpolate_position_2d` | no family claim |
| Vision | `patch_merge_layer_norm` | no family claim |
| Vision | `pool_tokens_by_position` | no family claim |
| Vision | `qwen_vision_rope_2d` | no family claim |
| Vision | `space_to_depth_norm_linear` | no family claim |
| Vision | `timestep_embedding` | no family claim |
| Vision | `upscale_nearest_2d` | no family claim |
| Vision | `vision_patch_projection` | no family claim |
| Vision | `vision_patch_projection_3d` | no family claim |
| Vision | `vision_rope_2d` | no family claim |
| Vision | `window_partition` | no family claim |
| Vision | `window_unpartition` | no family claim |
| Audio | `audio_causal_depthwise_conv1d` | no family claim |
| Audio | `audio_conv1d` | no family claim |
| Audio | `audio_conv1d_direct` | no family claim |
| Audio | `audio_depthwise_conv1d` | no family claim |
| Audio | `audio_relative_attention` | no family claim |
| Convolution | `col2im_1d` | no family claim |
| Convolution | `col2im_2d` | no family claim |
| Convolution | `conv2d` | no family claim |
| Convolution | `conv3d` | no family claim |
| Convolution | `conv_transpose_1d` | no family claim |
| Convolution | `conv_transpose_2d` | no family claim |
| Convolution | `depthwise_conv2d` | no family claim |
| Convolution | `im2col_2d` | no family claim |
| Convolution | `im2col_3d` | no family claim |
| Convolution | `pool1d` | no family claim |
| Convolution | `pool2d` | no family claim |
| Convolution | `pool2d_backward` | no family claim |
| Pooling | `pool_mean_rms_l2` | no family claim |
| Utilities and training | `accumulate` | family-only metadata |
| Utilities and training | `add_id` | family-only metadata |
| Utilities and training | `add_scalar` | family-only metadata |
| Utilities and training | `arange` | family-only metadata |
| Utilities and training | `argsort` | family-only metadata |
| Utilities and training | `clamp` | family-only metadata |
| Utilities and training | `concat` | family-only metadata |
| Utilities and training | `cosine` | family-only metadata |
| Utilities and training | `count_equal` | family-only metadata |
| Utilities and training | `cross_entropy` | family-only metadata |
| Utilities and training | `cumulative_sum` | family-only metadata |
| Utilities and training | `diag_embed` | family-only metadata |
| Utilities and training | `diag_mask` | family-only metadata |
| Utilities and training | `divide` | family-only metadata |
| Utilities and training | `dropout` | family-only metadata |
| Utilities and training | `fill` | family-only metadata |
| Utilities and training | `group_norm` | family-only metadata |
| Utilities and training | `hadamard` | family-only metadata |
| Utilities and training | `kd_ce_fused_bwd` | family-only metadata |
| Utilities and training | `kd_ce_fused_fwd` | family-only metadata |
| Utilities and training | `kd_kl_dense_bwd` | family-only metadata |
| Utilities and training | `kd_kl_dense_fwd` | family-only metadata |
| Utilities and training | `kd_kl_topk_bwd` | family-only metadata |
| Utilities and training | `kd_kl_topk_fwd` | family-only metadata |
| Utilities and training | `l2_normalize` | family-only metadata |
| Utilities and training | `logarithm` | family-only metadata |
| Utilities and training | `marginal` | family-only metadata |
| Utilities and training | `multiply` | family-only metadata |
| Utilities and training | `outer_product` | family-only metadata |
| Utilities and training | `pad_2d` | family-only metadata |
| Utilities and training | `pad_reflect_1d` | family-only metadata |
| Utilities and training | `reduce_mean` | family-only metadata |
| Utilities and training | `reduce_sum_all` | family-only metadata |
| Utilities and training | `repeat_2d` | family-only metadata |
| Utilities and training | `repeat_backward_2d` | family-only metadata |
| Utilities and training | `roll_2d` | family-only metadata |
| Utilities and training | `scale` | family-only metadata |
| Utilities and training | `set_rows` | family-only metadata |
| Utilities and training | `sine` | family-only metadata |
| Utilities and training | `solve_lower_triangular` | family-only metadata |
| Utilities and training | `square` | family-only metadata |
| Utilities and training | `square_root` | family-only metadata |
| Utilities and training | `subtract` | family-only metadata |
| Utilities and training | `tensor_copy` | family-only metadata |
| Utilities and training | `tensor_set_4d` | family-only metadata |
| Utilities and training | `threshold_topk_indices` | family-only metadata |
| Utilities and training | `triangular_fill` | family-only metadata |
| Attention | `attention_with_lse` | family-only metadata |
| Utilities and training | `cross_entropy_backward` | family-only metadata |
| Serving and caches | `embedding_backward` | family-only metadata |
| Serving and caches | `indexer_k_gather` | family-only metadata |
| Norms | `rms_norm_backward` | family-only metadata |
| Activations | `swiglu_oai` | family-only metadata |

## Quant-format gap list

| Format ID | Gap class |
|---|---|
| `awq` | no exact format declaration |
| `base_qn` | no exact format declaration |
| `int4_group` | no exact format declaration |
| `int8` | no exact format declaration |
| `mxfp4` | no exact format declaration |
| `mxfp6` | no exact format declaration |
| `mxfp8` | no exact format declaration |
| `nvfp4` | no exact format declaration |
| `q8_0_kv` | no exact format declaration |
| `tq2_0` | no exact format declaration |
| `turboquant` | no exact format declaration |

## Evidence rule

Removing an implementation or maturity gap requires the backend's native
path, correctness coverage, focused performance evidence, and an updated
manifest/status entry. Removing a family-only metadata gap requires an exact
operation entry or a documented semantic alias backed by the existing tests
and performance notebook. Directory presence alone is not sufficient.

Evidence remains backend-owned in `perf/optimization_status.md`,
`perf/baseline_status.md`, `perf/results/`, and the
backend correctness tests.
