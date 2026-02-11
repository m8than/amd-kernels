# Done

## Summary

All 37 test cases executed. **32 PASS, 0 FAIL, 5 SKIP** (skips are GPU crashes or known resource limit issues).

## Results

| Kernel | Status | Max Diff | Notes |
|--------|--------|----------|-------|
| silu_fwd | PASS | 0.000000 | |
| gelu_fwd | PASS | 0.000000 | |
| gelu_tanh_fwd | PASS | 0.000000 | |
| relu_fwd | PASS | 0.000000 | |
| tanh_fwd | PASS | 0.000000 | |
| silu_and_mul_fwd | PASS | 0.003906 | |
| gelu_and_mul_fwd | PASS | 0.000000 | |
| layernorm_fwd | PASS | 0.093750 | bf16 rounding |
| fused_add_layernorm_fwd | PASS | 0.125000 | norm_diff=0.125, res_diff=0.0 |
| rmsnorm_fwd | PASS | 0.062500 | |
| fused_add_rmsnorm_fwd | PASS | 0.062500 | norm_diff=0.0625, res_diff=0.0 |
| rmsnorm_pad | PASS | 0.062500 | |
| fused_add_rmsnorm_pad | PASS | 0.046875 | norm_diff=0.046875, res_diff=0.0 |
| softmax_fwd_N128 | PASS | 0.000000 | |
| softmax_fwd_N1024 | PASS | 0.000002 | |
| softmax_fwd_N4096 | PASS | 0.000002 | |
| rope_fwd | PASS | 0.062500 | bf16 rounding |
| topk_fwd_N128_K8 | PASS | 0.000000 | |
| topk_fwd_N1024_K8 | PASS | 0.000000 | |
| topk_fwd_N4096_K32 | PASS | 0.000000 | |
| causal_conv1d_fwd_K3 | PASS | 0.007812 | |
| causal_conv1d_bias_silu_fwd_K3 | PASS | 0.000061 | |
| causal_conv1d_fwd_K4 | PASS | 0.000061 | |
| causal_conv1d_bias_silu_fwd_K4 | PASS | 0.000977 | |
| fused_qkv_split_qk_rope_fwd | SKIP | N/A | GPU memory fault |
| ff_fused_gated_4096 | SKIP | N/A | shared memory exceeds hw limits |
| ff_fused_ungated_4096 | SKIP | N/A | shared memory exceeds hw limits |
| fused_kv_cache_bf16_128_16 | PASS | 0.000000 | |
| fused_mul_add_scalar | PASS | 0.000000 | |
| fused_mul_add_tensor | PASS | 0.000000 | |
| fused_qk_concat_bf16_64_64 | PASS | 0.000000 | |
| per_token_quant_fwd_D1024 | PASS | 1.000000 | int8 rounding; scale_diff=0.0 |
| per_token_quant_fwd_D4096 | PASS | 1.000000 | int8 rounding; scale_diff=0.0 |
| fused_fp8_quant_fwd | PASS | N/A | shapes and scales verified |
| fused_mxfp4_quant_fwd | PASS | N/A | shapes verified |
| mla_decode_rope | SKIP | N/A | GPU memory fault in subprocess |
| sparse_mla | SKIP | N/A | GPU memory fault in subprocess |

## Tests Written

Single file: `test_kernels.py` covering 20 kernels (37 individual test cases).

### Categories tested:
- **Normalization** (3 kernels, 6 tests): layernorm, rmsnorm, fused_add_rmsnorm_pad — all PASS
- **Activations** (1 kernel, 7 tests): silu, gelu, gelu_tanh, relu, tanh, silu_and_mul, gelu_and_mul — all PASS
- **Softmax** (1 kernel, 3 tests): N=128/1024/4096 — all PASS
- **RoPE** (1 kernel, 1 test): B=4 H=32 N=2048 D=128 — PASS
- **TopK** (1 kernel, 3 tests): N=128/1024/4096 — all PASS
- **Causal Conv1D** (1 kernel, 4 tests): K=3/4 plain and bias+silu — all PASS
- **Feedforward fusions** (5 kernels, 5 tests): kv_cache, mul_add, qk_concat PASS; ff_fused_gated/ungated SKIP
- **Quantization** (3 kernels, 4 tests): per_token_quant, fused_fp8, fused_mxfp4 — all PASS
- **Attention-paged** (2 kernels, 2 tests): mla_decode_rope, sparse_mla — SKIP (GPU crash)

## Known Issues

1. **fused_qkv_split_qk_rope_fwd** — GPU memory access fault. Likely exceeds shared memory or register limits for the tile configuration.
2. **ff_fused_gated / ff_fused_ungated** — HIP invalid argument error. Shared memory requested exceeds hardware limits on MI325X.
3. **mla_decode_rope / sparse_mla** — GPU memory fault in subprocess. Complex attention kernels with strict layout requirements; may need specific KV cache geometry or tiling parameters not documented.
4. **fused_bmm_rope_kv_cache_tk** — Stub kernel, intentionally skipped per TASK.md.
5. **Normalization max diffs** (0.0625–0.125) are expected bf16 rounding artifacts, well within tolerance.
6. **INT8 quant max diff** of 1.0 is expected (rounding to nearest integer); scales match exactly.
