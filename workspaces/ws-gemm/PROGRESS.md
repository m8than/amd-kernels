# Progress

## Phase 2: Fix GEMM kernels for MI325X

### Investigation Complete
- Root cause: `MAX_SHARED_MEMORY = 160000` in util.cuh causes all kernels to request 160KB LDS
- MI325X only supports 64KB per workgroup
- Most kernels actually only USE 64KB or less — they just REQUEST too much

### Fix Applied
- Changed `dynamic_shared_memory()` return value from `MAX_SHARED_MEMORY` (160KB) to actual LDS needed
- 7 out of 10 kernels successfully recompiled with correct LDS size

### Compilation Results (7/7 SUCCESS)
- gemm_a16w16_atomic (32KB LDS) — COMPILED
- gemm_a8w8 (64KB LDS) — COMPILED
- gemm_a8w8_per_token_scale (64KB LDS) — COMPILED
- gemm_a16wfp4 (64KB LDS) — COMPILED
- gemm_a8wfp4 (64KB LDS) — COMPILED
- gemm_afp4wfp4 (64KB LDS) — COMPILED
- gemm_a16w16_gated (64KB LDS) — COMPILED

### Cannot Fix Without Tile Size Changes
- gemm_a16w16 (128KB LDS, 256x256 tiles)
- gemm_a8w8_blockscale (128KB LDS, BLOCK_K=128)
- gemm_a16w8_blockscale (128KB LDS, BLOCK_K=128)

### Next: Test kernels on GPU
