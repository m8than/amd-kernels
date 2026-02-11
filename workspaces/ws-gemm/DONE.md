# Done

## Summary

All 18 GEMM kernels (10 basic, 4 batched, 4 fused) fail to launch on the AMD MI325X (gfx942) GPU. The root cause is a shared memory (LDS) mismatch: the kernels request 160KB of dynamic shared memory (`MAX_SHARED_MEMORY = 160000` defined in `HipKittens/include/common/util.cuh:104`), but the MI325X hardware only supports 64KB per workgroup (`shared_memory_per_block: 65536`).

This is not a test setup issue — it's a fundamental hardware incompatibility. The HipKittens GEMM kernels were designed for CDNA4 (MI350X, gfx950) which has 160KB LDS, and cannot run on CDNA3 (MI325X, gfx942) with 64KB LDS.

## Results

| Kernel | Status | Max Diff | Notes |
|--------|--------|----------|-------|
| gemm_a16w16 | FAIL | N/A | HIP invalid configuration argument (160KB LDS > 64KB limit) |
| gemm_a16w16_atomic | FAIL | N/A | HIP invalid configuration argument (160KB LDS > 64KB limit) |
| gemm_a16w16_gated | FAIL | N/A | HIP invalid configuration argument (160KB LDS > 64KB limit) |
| gemm_a8w8 | FAIL | N/A | HIP invalid configuration argument (160KB LDS > 64KB limit) |
| gemm_a8w8_blockscale | FAIL | N/A | HIP invalid configuration argument (160KB LDS > 64KB limit) |
| gemm_a8w8_per_token_scale | FAIL | N/A | HIP invalid configuration argument (160KB LDS > 64KB limit) |
| gemm_a16w8_blockscale | FAIL | N/A | HIP invalid configuration argument (160KB LDS > 64KB limit) |
| gemm_a16wfp4 | FAIL | N/A | HIP invalid configuration argument (160KB LDS > 64KB limit) |
| gemm_a8wfp4 | FAIL | N/A | HIP invalid configuration argument (160KB LDS > 64KB limit) |
| gemm_afp4wfp4 | FAIL | N/A | HIP invalid configuration argument (160KB LDS > 64KB limit) |
| batched_gemm_bf16 | FAIL | N/A | HIP invalid configuration argument (160KB LDS > 64KB limit) |
| batched_gemm_a8w8 | FAIL | N/A | HIP invalid configuration argument (160KB LDS > 64KB limit) |
| batched_gemm_a16wfp4 | FAIL | N/A | HIP invalid configuration argument (160KB LDS > 64KB limit) |
| batched_gemm_afp4wfp4 | FAIL | N/A | HIP invalid configuration argument (160KB LDS > 64KB limit) |
| fused_gemm_a8w8_blockscale_a16w16 | FAIL | N/A | HIP invalid configuration argument (160KB LDS > 64KB limit) |
| fused_gemm_a8w8_blockscale_mul_add | FAIL | N/A | HIP invalid configuration argument (160KB LDS > 64KB limit) |
| fused_gemm_afp4wfp4_a16w16 | FAIL | N/A | HIP invalid configuration argument (160KB LDS > 64KB limit) |
| fused_gemm_afp4wfp4_mul_add | FAIL | N/A | HIP invalid configuration argument (160KB LDS > 64KB limit) |

## Tests Written

- `test_kernels.py` — 18 tests with subprocess isolation per kernel, correct pybind11 dispatch arg orders, proper tensor shapes/dtypes/sizes per kernel.cpp specs

## Known Issues

### Root Cause: Shared Memory Mismatch

- **Hardware**: AMD MI325X (gfx942, CDNA3) — 64KB LDS per workgroup
- **Kernel request**: `MAX_SHARED_MEMORY = 160000` (156KB) defined in `HipKittens/include/common/util.cuh:104`
- **Error**: `hipErrorInvalidConfiguration` when `hipFuncSetAttribute` tries to set 160KB dynamic shared memory
- **Resolution needed**: Kernels must be recompiled with `MAX_SHARED_MEMORY <= 65536` or tile sizes reduced to fit within 64KB LDS. Alternatively, use CDNA4 hardware (MI350X/gfx950).

### Dispatch Argument Orders (verified from pybind11 bindings)

For reference, the correct dispatch() argument orders are:

- `gemm_a16w16`: `dispatch(A, B, C)`
- `gemm_a16w16_atomic`: `dispatch(A, B, C)` — C is fp32
- `gemm_a16w16_gated`: `dispatch(A, B, C)` — B is [2N, K], C is [M, N]
- `gemm_a8w8`: `dispatch(A_bf16, B_bf16, a_scale, b_scale, C)`
- `gemm_a8w8_blockscale`: `dispatch(A_bf16, B_bf16, C, a_scale, b_scale)`
- `gemm_a8w8_per_token_scale`: `dispatch(A_bf16, B_bf16, C, a_scale, b_scale)`
- `gemm_a16w8_blockscale`: `dispatch(A, B_bf16, C, b_scale)`
- `gemm_a16wfp4`: `dispatch(A, B, C, b_scales)`
- `gemm_a8wfp4`: `dispatch(A, B, a_scale, C)`
- `gemm_afp4wfp4`: `dispatch(A, B, C)`
- `batched_gemm_bf16`: `dispatch(A_hk, B_hk, C_hk, bias_hk)` — 4D (1,B,M/N,K/N)
- `batched_gemm_a8w8`: `dispatch(A_hk, B_hk, a_scale_hk, b_scale_hk, C_hk, bias_hk)`
- `batched_gemm_a16wfp4`: `dispatch(A_hk, B_hk, C_hk)`
- `batched_gemm_afp4wfp4`: `dispatch(A_hk, B_hk, C_hk)`
- `fused_*_a16w16`: `dispatch(a_q, b_q, c_q, a_scale, b_scale, a_bf16, b_bf16, c_bf16)`
- `fused_*_mul_add`: `dispatch(a, b, c, a_scale, b_scale, c_a, c_b)`
