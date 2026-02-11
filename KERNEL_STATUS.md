# Kernel Status

Status of all HipKittens kernels. Tested on AMD Instinct MI325X (gfx942, CDNA3) with ROCm 7.2.

## Status Key

- **Tested** — Compiled, loads, passes parity test against PyTorch reference, benchmarked
- **Compiled** — Compiled and loads as a Python module, but not functionally tested on GPU
- **Source only** — Source code included, does not compile yet (missing HipKittens features or complex dependencies)

---

## Tested (11 kernels)

These kernels are fully validated: correct output verified against PyTorch, benchmarked against AITER.

| Kernel | Module | Functions | Notes |
|--------|--------|-----------|-------|
| RMSNorm | `rmsnorm_tk` | `rmsnorm_fwd(x, w, out, eps, N)` | 3-4x faster than AITER Triton at D=128 |
| Fused Add+RMSNorm | `rmsnorm_tk` | `fused_add_rmsnorm_fwd(x, res, w, out, res_out, eps, N)` | 3-6x faster than AITER Triton |
| LayerNorm | `layernorm_tk` | `layernorm_fwd(x, w, b, out, eps, N)` | ~0.7x vs AITER CK (CK is faster) |
| Fused Add+LayerNorm | `layernorm_tk` | `fused_add_layernorm_fwd(x, res, w, b, out, res_out, eps, N)` | ~0.6-0.9x vs AITER CK |
| SiLU | `activation_tk` | `silu_fwd(x, out)` | 2-3x faster than PyTorch |
| GeLU | `activation_tk` | `gelu_fwd(x, out)` | 2-3x faster than PyTorch |
| SiLU+Mul (gated) | `activation_tk` | `silu_and_mul_fwd(xg, out)` | 1.5-1.8x faster than AITER HIP at D<=256 |
| GeLU+Mul (gated) | `activation_tk` | `gelu_and_mul_fwd(xg, out)` | 1.5-1.8x faster than AITER HIP at D<=256 |
| Softmax | `softmax_tk` | `softmax_fwd(x, out)` | 3-4x faster than AITER Triton |
| RoPE | `rope_tk` | `rope_fwd(x, out, cos_half, sin_half)` | 1.1-1.5x faster than AITER Triton |
| Top-K | `topk_tk` | `topk_fwd(x, out_vals, out_idx, K)` | Slower than AITER at K>10 |

Additional tested activation functions in `activation_tk`: `relu_fwd`, `tanh_fwd`, `gelu_tanh_fwd`

### Tested kernel limitations

- **Normalization**: D=128 only (hardcoded tile size). AITER CK/Triton supports arbitrary D.
- **Gated activations**: HK wins at D<=256, AITER HIP wins at D>=8192.
- **Top-K**: Degrades badly at K=32 (1630us vs AITER's 106us). Fine at K<=10.
- **Softmax**: HK faster than AITER Triton but slower than PyTorch at large batch sizes.

---

## Compiled but untested (60 kernels)

These compile and load as Python modules. The pybind11 bindings expose functions, but they have not been run with real GPU data to verify correctness. They need specialized test harnesses due to complex interfaces.

### GEMM — Basic (10 kernels)

All expose `dispatch(A, B, ...)`. Tile sizes may restrict valid input dimensions.

| Kernel | Module | Quantization |
|--------|--------|-------------|
| gemm_a16w16 | `gemm_a16w16_tk` | BF16 x BF16 |
| gemm_a16w16_atomic | `gemm_a16w16_atomic_tk` | BF16 x BF16 (atomic accumulate) |
| gemm_a16w16_gated | `gemm_a16w16_gated_tk` | BF16 x BF16 (gated) |
| gemm_a8w8 | `gemm_a8w8_tk` | INT8 x INT8 |
| gemm_a8w8_blockscale | `gemm_a8w8_blockscale_tk` | INT8 x INT8 (block-scale) |
| gemm_a8w8_per_token_scale | `gemm_a8w8_per_token_scale_tk` | INT8 x INT8 (per-token-scale) |
| gemm_a16w8_blockscale | `gemm_a16w8_blockscale_tk` | BF16 x INT8 (block-scale) |
| gemm_a16wfp4 | `gemm_a16wfp4_tk` | BF16 x FP4 |
| gemm_a8wfp4 | `gemm_a8wfp4_tk` | INT8 x FP4 |
| gemm_afp4wfp4 | `gemm_afp4wfp4_tk` | FP4 x FP4 |

### GEMM — Batched (4 kernels)

| Kernel | Module |
|--------|--------|
| batched_gemm_bf16 | `batched_gemm_bf16_tk` |
| batched_gemm_a8w8 | `batched_gemm_a8w8_tk` |
| batched_gemm_a16wfp4 | `batched_gemm_a16wfp4_tk` |
| batched_gemm_afp4wfp4 | `batched_gemm_afp4wfp4_tk` |

### GEMM — Fused (4 kernels)

| Kernel | Module | Pipeline |
|--------|--------|----------|
| fused_gemm_a8w8_blockscale_a16w16 | `fused_gemm_a8w8_blockscale_a16w16_tk` | INT8 GEMM then BF16 GEMM |
| fused_gemm_a8w8_blockscale_mul_add | `fused_gemm_a8w8_blockscale_mul_add_tk` | INT8 GEMM then mul+add |
| fused_gemm_afp4wfp4_a16w16 | `fused_gemm_afp4wfp4_a16w16_tk` | FP4 GEMM then BF16 GEMM |
| fused_gemm_afp4wfp4_mul_add | `fused_gemm_afp4wfp4_mul_add_tk` | FP4 GEMM then mul+add |

### MoE GEMM (10 kernels)

| Kernel | Module | Function |
|--------|--------|----------|
| moe_op | `moe_op_tk` | `moe_gemm(...)` |
| moe_op_e2e | `moe_op_e2e_tk` | `moe_e2e(...)` |
| moe_op_gelu | `moe_op_gelu_tk` | `moe_gelu(...)` |
| moe_op_silu_fused | `moe_op_silu_fused_tk` | `moe_silu_fused(...)` |
| moe_op_gemm_a4w4 | `moe_op_gemm_a4w4_tk` | `moe_a4w4(...)` |
| moe_op_gemm_a8w4 | `moe_op_gemm_a8w4_tk` | `moe_a8w4(...)` |
| moe_op_gemm_a8w8 | `moe_op_gemm_a8w8_tk` | `moe_a8w8(...)` |
| moe_op_gemm_a8w8_blockscale | `moe_op_gemm_a8w8_blockscale_tk` | `moe_a8w8_blockscale(...)` |
| moe_op_mxfp4 | `moe_op_mxfp4_tk` | `moe_mxfp4(...)` |
| moe_op_mxfp4_silu_fused | `moe_op_mxfp4_silu_fused_tk` | `moe_mxfp4_silu_fused(...)` |

### MoE Routing (4 kernels)

| Kernel | Module | Functions |
|--------|--------|-----------|
| moe_align_block_size | `moe_align_block_size_tk` | `moe_align_block_size(...)` |
| moe_bitmatrix | `moe_bitmatrix_tk` | `sum_bitmatrix_rows(...)`, `sum_bitmatrix_rows_fused(...)` |
| moe_expt_data | `moe_expt_data_tk` | `expt_data_compute(...)`, `expt_data_stage1(...)`, `expt_data_stage2(...)`, `expt_data_stage2_fused(...)` |
| moe_topk | `moe_topk_tk` | `moe_topk(...)` |

### MoE Misc (2 kernels)

| Kernel | Module | Functions |
|--------|--------|-----------|
| moe_routing_sigmoid_top1_fused | `moe_routing_sigmoid_top1_fused_tk` | `moe_routing_sigmoid_top1_fused(...)` |
| quant_moe | `quant_moe_tk` | `downcast_to_mxfp(...)`, `downcast_to_static_fp8(...)`, `upcast_from_mxfp(...)` |

### Feedforward Fusions (5 kernels + 1 empty)

| Kernel | Module | Functions |
|--------|--------|-----------|
| ff_fused_gated | `ff_fused_gated_tk` | `dispatch_4096(...)`, `dispatch_8192(...)` |
| ff_fused_ungated | `ff_fused_ungated_tk` | `dispatch_4096(...)`, `dispatch_8192(...)`, `dispatch_4096_no_activation(...)`, `dispatch_8192_no_activation(...)` |
| fused_kv_cache | `fused_kv_cache_tk` | `dispatch_bf16_128_16(...)`, `dispatch_bf16_128_32(...)`, `dispatch_fp16_128_16(...)`, `dispatch_fp16_128_32(...)` |
| fused_mul_add | `fused_mul_add_tk` | `dispatch_bf16_4096(...)`, `dispatch_bf16_8192(...)`, `dispatch_fp16_4096(...)`, `dispatch_fp16_8192(...)` |
| fused_qk_concat | `fused_qk_concat_tk` | `dispatch_bf16_128_128(...)`, `dispatch_bf16_64_128(...)`, etc. |
| fused_bmm_rope_kv_cache | `fused_bmm_rope_kv_cache_tk` | *(loads but exports 0 functions — empty module)* |

### Attention (2 kernels)

| Kernel | Module | Functions |
|--------|--------|-----------|
| mla_decode_rope | `mla_decode_rope_tk` | `mla_decode(...)` |
| unified_attn_sparse_mla | `unified_attn_sparse_mla_tk` | `sparse_mla(...)` |

### GDR Decode (5 kernels)

| Kernel | Module | Functions |
|--------|--------|-----------|
| fused_recurrent | `fused_recurrent_tk` | `fused_recurrent_fwd(...)` |
| fused_sigmoid_gating_recurrent | `fused_sigmoid_gating_recurrent_tk` | `fused_sigmoid_gating_recurrent(...)` |
| causal_conv1d_split_qkv | `causal_conv1d_split_qkv_tk` | `causal_conv1d_update_split_qkv(...)` |
| fused_qkvzba_split | `fused_qkvzba_split_tk` | `fused_qkvzba_split(...)` |
| gdr_utils | `gdr_utils_tk` | `abs_diff(...)`, `bf16_to_float(...)`, `float_to_bf16(...)`, `l2_normalize(...)`, `max_reduce(...)`, `squared_diff(...)`, `sum_reduce(...)` |

### GDR Prefill (11 kernels)

| Kernel | Module | Functions |
|--------|--------|-----------|
| chunk | `chunk_tk` | `chunk_compute_A(...)`, `chunk_cumsum(...)`, `chunk_pipeline(...)`, `chunk_recompute_w_u(...)`, `chunk_solve_tril(...)` |
| chunk_delta_h | `chunk_delta_h_tk` | `chunk_delta_h_fwd(...)` |
| chunk_o | `chunk_o_tk` | `chunk_fwd_o(...)` |
| cumsum | `cumsum_tk` | `cumsum_scalar(...)`, `cumsum_vector(...)` |
| fused_cumsum_kkt | `fused_cumsum_kkt_tk` | `fused_cumsum_kkt(...)` |
| fused_gdn_gating_prefill | `fused_gdn_gating_prefill_tk` | `fused_gdn_gating(...)` |
| index | `index_tk` | `get_max_num_splits(...)`, `gpu_prepare_chunk_indices(...)`, `prepare_chunk_offsets(...)`, `prepare_cu_seqlens_from_lens(...)`, `prepare_lens(...)`, `prepare_position_ids(...)`, `prepare_sequence_ids(...)` |
| l2norm | `l2norm_tk` | `l2norm_fwd(...)`, `l2norm_bwd(...)` |
| solve_tril | `solve_tril_tk` | `solve_tril(...)` |
| wy_representation | `wy_representation_tk` | `chunk_scaled_dot_kkt(...)`, `recompute_w_u(...)` |
| causal_conv1d_fwd_split_qkv | `causal_conv1d_fwd_split_qkv_tk` | `causal_conv1d_fwd_split(...)` |

### Normalization — Extra (1 kernel)

| Kernel | Module | Functions |
|--------|--------|-----------|
| fused_add_rmsnorm_pad | `fused_add_rmsnorm_pad_tk` | `rmsnorm_pad(...)`, `fused_add_rmsnorm_pad(...)` |

### Quantization (3 kernels)

| Kernel | Module | Functions |
|--------|--------|-----------|
| fused_fp8_quant | `fused_fp8_quant_tk` | `fused_rmsnorm_fp8_quant_fwd(...)` |
| fused_mxfp4_quant | `fused_mxfp4_quant_tk` | `fused_rmsnorm_mxfp4_quant_fwd(...)` |
| quant | `quant_tk` | `per_token_quant_fwd(...)` |

### Rope/Activations — Extra (2 kernels)

| Kernel | Module | Functions |
|--------|--------|-----------|
| causal_conv1d | `causal_conv1d_tk` | `causal_conv1d_fwd(...)`, `causal_conv1d_bias_silu_fwd(...)` |
| fused_qkv_split_qk_rope | `fused_qkv_split_qk_rope_tk` | `fused_qkv_split_qk_rope_fwd(...)` |

---

## Source only — not compiling (24 kernels)

These have source code (`kernel.cpp`) but fail to compile due to missing HipKittens features, complex template issues, or unresolved dependencies.

### Attention Forward (5)
`extend_attn`, `hstu_attn`, `mha`, `prefill_attn`, `unified_attn`

### Attention Backward (9)
`flash_attn_bwd`, `flash_attn_fwd_decode`, `flash_attn_fwd_prefill`, `fp8_mqa_logits`, `lean_atten`, `lean_atten_paged`, `mha_fused_bwd`, `mha_onekernel_bwd`, `sage_attention`

### Attention Paged (5)
`chunked_pa_prefill`, `pa_decode`, `pa_mqa_logits`, `pa_prefill`, `pod_attn`

### GEMM (3)
`batched_gemm_a8w8_ptgq`, `fused_gemm_a8w8_blockscale_split_cat`, `fused_gemm_afp4wfp4_split_cat`

### MoE Routing (1)
`moe_routing`

### Quantization (1)
`gmm` (grouped matrix multiply)

---

## Summary

| Status | Count | Percentage |
|--------|-------|------------|
| Tested and benchmarked | 11 | 12% |
| Compiled (untested) | 60 | 63% |
| Source only | 24 | 25% |
| **Total** | **95** | 100% |
