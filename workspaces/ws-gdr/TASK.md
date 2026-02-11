# Task: Fix solve_tril kernel and verify all GDR tests pass

You need to write and run GPU parity tests for all GDR kernels, and investigate the solve_tril failure.

## Environment
- Python: `/root/aiter-hipkittens/amd-kernels/.venv/bin/python`
- Kernels dir: `/root/aiter-hipkittens/amd-kernels/kernels/`
- GPU: AMD MI325X (gfx942), ROCm 7.2
- Load .so files via: `importlib.util.spec_from_file_location(name, path)`

## Previous results (from another worker, files lost)

All GDR kernels PASS except `solve_tril_tk` which fails at BT=16/32/64 with max_diff=1.0. It works correctly inside `chunk_tk`. The previous worker made fixes:
- fused_recurrent: Fixed reference to use delta rule, reduced input magnitudes to 0.1
- fused_sigmoid_gating_recurrent: Same delta rule fix, corrected output shape
- chunk/solve_tril: Relaxed tolerance to 1e-2 for matrix inverse

## Kernels to test

### gdr-decode (5 kernels)
1. `gdr_utils_tk` — utility functions (abs_diff, squared_diff, max_reduce, sum_reduce, bf16_to_float, float_to_bf16, l2_normalize)
2. `causal_conv1d_split_qkv_tk` — causal conv1d + split QKV for decode
3. `fused_qkvzba_split_tk` — tensor reshuffling
4. `fused_recurrent_tk` — recurrent gated delta rule
5. `fused_sigmoid_gating_recurrent_tk` — fused sigmoid gating + recurrent

### gdr-prefill (12 kernels)
1. `l2norm_tk` — L2 normalization fwd/bwd
2. `cumsum_tk` — chunk-local cumulative sum
3. `index_tk` — sequence indexing utilities
4. `op_tk` — element-wise math ops
5. `solve_tril_tk` — triangular matrix inverse (I+A)^{-1} ← **THIS IS THE FAILING ONE**
6. `wy_representation_tk` — KKT + w/u compute
7. `fused_gdn_gating_prefill_tk` — fused gating
8. `chunk_tk` — orchestrator for chunk-based GDR
9. `chunk_delta_h_tk` — recurrent hidden state
10. `chunk_o_tk` — output computation
11. `causal_conv1d_fwd_split_qkv_tk` — causal conv1d for prefill
12. `fused_cumsum_kkt_tk` — fused cumsum + KKT

## Instructions

1. Write `test_kernels.py` with tests for ALL 17 GDR kernels. Read each kernel's test_parity.py and kernel.cpp for exact interfaces.
2. Run the tests.
3. For solve_tril: investigate why standalone fails but chunk_tk works. Read the chunk_tk code to see how it calls solve_tril internally. Maybe the standalone version has different semantics.
4. Write DONE.md with results.

IMPORTANT: If a kernel crashes, skip it and note it. Don't get stuck.

## DONE.md Format
```
# Done
## Summary
## Results
| Kernel | Status | Max Diff | Notes |
## Fixes Applied
## Known Issues
```
