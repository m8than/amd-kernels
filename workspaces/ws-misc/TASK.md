# Task: Fix failing/skipping misc kernels

Several kernels either crash or produce wrong results. Fix what you can.

## Environment
- Python: `/root/aiter-hipkittens/amd-kernels/.venv/bin/python`
- Kernels dir: `/root/aiter-hipkittens/amd-kernels/kernels/`
- GPU: AMD MI325X (gfx942), ROCm 7.2
- Load .so files via: `importlib.util.spec_from_file_location(name, path)`

## Previous results

PASS (32 tests): All activations, normalization, softmax, rope, topk, causal_conv1d, fused_kv_cache, fused_mul_add, fused_qk_concat, quantization kernels pass.

SKIP/CRASH (5 kernels, need investigation):
1. `fused_qkv_split_qk_rope_tk` — GPU memory access fault. Source at `kernels/rope-activations/fused_qkv_split_qk_rope/`
2. `ff_fused_gated_tk` — HIP invalid argument, shared memory exceeds hw limits. Source at `kernels/feedforward-fusions/ff_fused_gated/`
3. `ff_fused_ungated_tk` — Same shared memory issue. Source at `kernels/feedforward-fusions/ff_fused_ungated/`
4. `mla_decode_rope_tk` — GPU memory fault. Source at `kernels/attention-paged/mla_decode_rope/`
5. `unified_attn_sparse_mla_tk` — GPU memory fault. Source at `kernels/attention-paged/unified_attn_sparse_mla/`

## Instructions

1. First, reproduce the previous passing tests: write test_kernels.py with tests for ALL kernels (activations, normalization, softmax, rope, topk, causal_conv1d, fused_kv_cache, fused_mul_add, fused_qk_concat, quantization). Make sure those still pass.

2. For each SKIP/CRASH kernel, investigate:
   - Read kernel.cpp to understand shared memory usage
   - Read test_parity.py for expected interface
   - For shared memory issues: check if the kernel has multiple dispatch functions for different tile sizes. Maybe a smaller variant exists.
   - For GPU memory faults: try smaller input sizes, check if tensor layout is wrong
   - Try running in a subprocess to avoid taking down the whole test suite

3. The ff_fused_gated/ungated kernels use very large tile sizes. Look at the kernel.cpp globals struct for `dynamic_shared_memory()` — if it returns more than 65536, the kernel CANNOT run on MI325X (64KB LDS limit). In that case, document it as hardware-limited and move on.

4. For fused_qkv_split_qk_rope: there are multiple dispatch configurations (32/8, 64/8, 32/32 for QH/KVH). Try each one.

5. For mla_decode and sparse_mla: these are complex attention kernels. Try the exact test setup from test_parity.py. If they still crash, document as hardware-limited.

6. Write DONE.md with results.

## DONE.md Format
```
# Done
## Summary
## Results
| Kernel | Status | Max Diff | Notes |
## Fixes Applied
## Known Issues
```
