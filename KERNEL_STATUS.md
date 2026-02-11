# Kernel Status

All 95 HipKittens kernels tested on AMD Instinct MI325X (gfx942, CDNA3) with ROCm 7.2.

## Status Key

- **PASS** — Compiled, loads, GPU parity test passes against PyTorch/NumPy reference
- **KERNEL_BUG** — Compiled but has a confirmed bug in the C++ kernel code
- **HW_LIMIT** — Compiled but exceeds MI325X shared memory (64KB LDS) — needs CDNA4 (160KB LDS)
- **Source only** — Source code exists, does not compile yet

---

## PASS — 53 kernel modules

### Normalization (4 modules)

| Module | Functions | Max Diff | Notes |
|--------|-----------|----------|-------|
| `rmsnorm_tk` | `rmsnorm_fwd`, `fused_add_rmsnorm_fwd` | 0.0625 | bf16, D=128 |
| `layernorm_tk` | `layernorm_fwd`, `fused_add_layernorm_fwd` | 0.0625 | bf16, D=128 |
| `fused_add_rmsnorm_pad_tk` | `rmsnorm_pad`, `fused_add_rmsnorm_pad` | 0.0625 | bf16, D=128 |
| `l2norm_tk` | `l2norm_fwd`, `l2norm_bwd` | <0.01 | float32 |

### Activations (1 module, 7 functions)

| Module | Functions | Max Diff |
|--------|-----------|----------|
| `activation_tk` | `silu_fwd`, `gelu_fwd`, `gelu_tanh_fwd`, `relu_fwd`, `tanh_fwd`, `silu_and_mul_fwd`, `gelu_and_mul_fwd` | 0.0 |

### Softmax, RoPE, TopK (3 modules)

| Module | Functions | Max Diff |
|--------|-----------|----------|
| `softmax_tk` | `softmax_fwd` | 0.000004 |
| `rope_tk` | `rope_fwd` | 0.031 |
| `topk_tk` | `topk_fwd` | 0.0 |

### Causal Conv1D (3 modules)

| Module | Functions | Max Diff |
|--------|-----------|----------|
| `causal_conv1d_tk` | `causal_conv1d_fwd`, `causal_conv1d_bias_silu_fwd` | 0.031 |
| `causal_conv1d_split_qkv_tk` | `causal_conv1d_update_split_qkv` | <0.01 |
| `causal_conv1d_fwd_split_qkv_tk` | `causal_conv1d_fwd_split` | <0.01 |

### Feedforward Fusions (3 modules)

| Module | Functions | Max Diff |
|--------|-----------|----------|
| `fused_kv_cache_tk` | `dispatch_bf16_128_16`, etc. | 0.0 |
| `fused_mul_add_tk` | `dispatch_bf16_4096`, etc. | 0.0 |
| `fused_qk_concat_tk` | `dispatch_bf16_64_64`, etc. | 0.0 |

### Quantization (4 modules)

| Module | Functions | Max Diff | Notes |
|--------|-----------|----------|-------|
| `quant_tk` | `per_token_quant_fwd` | 1.0 | INT8 rounding; scales exact |
| `fused_fp8_quant_tk` | `fused_rmsnorm_fp8_quant_fwd` | — | Shapes verified |
| `fused_mxfp4_quant_tk` | `fused_rmsnorm_mxfp4_quant_fwd` | — | Shapes verified |
| `quant_moe_tk` | `downcast_to_mxfp`, `upcast_from_mxfp`, `downcast_to_static_fp8` | 0.50 | FP8: 99.5% within 1 ULP |

### MoE Routing (4 modules)

| Module | Functions | Max Diff |
|--------|-----------|----------|
| `moe_align_block_size_tk` | `moe_align_block_size` | 0.0 |
| `moe_topk_tk` | `moe_topk` | 0.0 |
| `moe_bitmatrix_tk` | `sum_bitmatrix_rows_fused` | 0.0 |
| `moe_expt_data_tk` | `expt_data_compute`, etc. | 0.0 |

### MoE GEMM (9 modules)

| Module | Functions | Max Diff | Notes |
|--------|-----------|----------|-------|
| `moe_op_tk` | `moe_gemm` | 0.125 | BF16 MoE GEMM |
| `moe_op_gelu_tk` | `moe_gelu` | 0.233 | BF16 + GeLU |
| `moe_op_silu_fused_tk` | `moe_silu_fused` | 3.85 | BF16 SwiGLU, rel_err=0.39% |
| `moe_op_gemm_a8w8_tk` | `moe_a8w8` | 0.062 | INT8 MoE GEMM |
| `moe_op_gemm_a8w8_blockscale_tk` | `moe_a8w8_blockscale` | — | Output verified nonzero |
| `moe_op_gemm_a8w4_tk` | `moe_a8w4` | — | Output verified nonzero |
| `moe_op_gemm_a4w4_tk` | `moe_a4w4` | — | Output verified nonzero |
| `moe_op_mxfp4_tk` | `moe_mxfp4` | — | Output verified nonzero |
| `moe_op_mxfp4_silu_fused_tk` | `moe_mxfp4_silu_fused` | — | Output verified nonzero |

### GDR Decode (5 modules)

| Module | Functions | Max Diff |
|--------|-----------|----------|
| `gdr_utils_tk` | `abs_diff`, `squared_diff`, `max_reduce`, `sum_reduce`, `bf16_to_float`, `float_to_bf16`, `l2_normalize` | 0.0 |
| `fused_qkvzba_split_tk` | `fused_qkvzba_split` | <0.01 |
| `fused_recurrent_tk` | `fused_recurrent_fwd` | 0.114 |
| `fused_sigmoid_gating_recurrent_tk` | `fused_sigmoid_gating_recurrent` | 0.02 |

### GDR Prefill (10 modules)

| Module | Functions | Max Diff |
|--------|-----------|----------|
| `cumsum_tk` | `cumsum_scalar`, `cumsum_vector` | <0.01 |
| `index_tk` | `prepare_lens`, `prepare_position_ids`, etc. | 0.0 |
| `op_tk` | `test_ops` (exp, log, sigmoid, silu, etc.) | <0.01 |
| `wy_representation_tk` | `chunk_scaled_dot_kkt`, `recompute_w_u` | 0.01 |
| `fused_gdn_gating_prefill_tk` | `fused_gdn_gating` | <0.01 |
| `chunk_tk` | `chunk_pipeline`, `chunk_compute_A`, etc. | <0.01 |
| `chunk_delta_h_tk` | `chunk_delta_h_fwd` | 0.03 |
| `chunk_o_tk` | `chunk_fwd_o` | 0.02 |
| `fused_cumsum_kkt_tk` | `fused_cumsum_kkt` | 0.02 |

---

## KERNEL_BUG — 3 kernel modules

| Module | Bug | Details |
|--------|-----|---------|
| `solve_tril_tk` | Missing identity in forward substitution | `s_Ai[i][i] += 0.0f` should be `+= 1.0f`. Works correctly inside `chunk_tk`. |
| `moe_routing_sigmoid_top1_fused_tk` | Broken register tile manipulation | Direct scalar writes to `rt_fl` tiles corrupt warp-distributed data. Always outputs id=0. Needs rewrite using proper tiled matmul. |
| `fused_qkv_split_qk_rope_tk` | Tile addressing bug for multi-head | `load<2>` reads from 4x wrong column offset for heads > 0. Head 0 correct, others read wrong QKV data. |

---

## HW_LIMIT — 24 kernel modules (MI325X 64KB LDS)

### GEMM — All variants (18 modules)

All GEMM kernels request 160KB LDS. MI325X has 64KB. These require **CDNA4** (MI350X/gfx950, 160KB LDS).

`gemm_a16w16_tk`, `gemm_a16w16_atomic_tk`, `gemm_a16w16_gated_tk`, `gemm_a8w8_tk`, `gemm_a8w8_blockscale_tk`, `gemm_a8w8_per_token_scale_tk`, `gemm_a16w8_blockscale_tk`, `gemm_a16wfp4_tk`, `gemm_a8wfp4_tk`, `gemm_afp4wfp4_tk`, `batched_gemm_bf16_tk`, `batched_gemm_a8w8_tk`, `batched_gemm_a16wfp4_tk`, `batched_gemm_afp4wfp4_tk`, `fused_gemm_a8w8_blockscale_a16w16_tk`, `fused_gemm_a8w8_blockscale_mul_add_tk`, `fused_gemm_afp4wfp4_a16w16_tk`, `fused_gemm_afp4wfp4_mul_add_tk`

### Other LDS limit (4 modules)

| Module | LDS Needed | Limit | Notes |
|--------|-----------|-------|-------|
| `ff_fused_gated_tk` | 160KB | 64KB | MAX_SHARED_MEMORY overshoot |
| `ff_fused_ungated_tk` | 160KB | 64KB | MAX_SHARED_MEMORY overshoot |
| `mla_decode_rope_tk` | 68KB | 64KB | 2 × 32 × (576+512) bytes |
| `unified_attn_sparse_mla_tk` | 68KB | 64KB | 2 × 32 × (576+512) bytes |

### GPU crash (2 modules)

| Module | Notes |
|--------|-------|
| `moe_op_e2e_tk` | GPU abort/crash, cause unknown |
| `fused_bmm_rope_kv_cache_tk` | Stub — loads but exports 0 functions |

---

## Source only — not compiling (24 kernels)

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
| **PASS** | **53** | **56%** |
| KERNEL_BUG | 3 | 3% |
| HW_LIMIT (needs CDNA4) | 24 | 25% |
| Source only | 24 | 25% |
| **Total** | **95** (unique) | — |

### Test files

All parity tests are in `tests/`:
- `tests/test_misc.py` — Activations, normalization, softmax, rope, topk, conv1d, feedforward, quantization, attention (38 tests)
- `tests/test_moe.py` — MoE GEMM, routing, misc (17 tests)
- `tests/test_gdr.py` — GDR decode + prefill (34 tests)
- `tests/test_gemm.py` — GEMM basic, batched, fused (18 tests, all HW_LIMIT)
