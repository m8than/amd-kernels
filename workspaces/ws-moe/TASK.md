# Task: Fix failing MoE kernel tests

You have an existing `test_kernels.py` that tests MoE kernels. Several are failing. Fix them.

## Environment
- Python: `/root/aiter-hipkittens/amd-kernels/.venv/bin/python`
- Kernels dir: `/root/aiter-hipkittens/amd-kernels/kernels/`
- GPU: AMD MI325X (gfx942), ROCm 7.2
- Load .so files via: `importlib.util.spec_from_file_location(name, path)`

## Current test results (from running test_kernels.py)

PASS: moe_align_block_size, moe_topk, moe_bitmatrix, moe_expt_data, moe_op_gemm_a8w8, moe_op_gemm_a8w8_blockscale, moe_op_gemm_a8w4, moe_op_gemm_a4w4, moe_op_mxfp4, moe_op_mxfp4_silu_fused, quant_moe (mxfp roundtrip)

FAIL (need fixing):
1. `moe_op_tk` → max_diff=56.58 — BF16 MoE GEMM. The routing setup (sorted_token_ids, expert_ids, num_tokens_post_padded) is likely wrong.
2. `moe_op_gelu_tk` → max_diff=62.08 — Same routing issue.
3. `moe_op_silu_fused_tk` → max_diff=1403 — Same routing issue, plus SwiGLU gate has 2x width.
4. `quant_moe_tk` (fp8) → 0% match rate — The `downcast_to_static_fp8` function likely expects different scale semantics.
5. `moe_routing_sigmoid_top1_fused_tk` → 7.4% index match — Function signature or semantics wrong.

SKIP: moe_op_e2e_tk (GPU crash — skip this one)

## How to fix

1. Read the EXISTING `test_kernels.py` to understand the current test setup.
2. For each failing kernel, **read the kernel's `test_parity.py`** in its source directory to understand the expected interface. The test_parity.py files have reference implementations.
3. Also **read the kernel.cpp** pybind11 module definition to understand exact arg order and types.
4. The BF16 MoE GEMMs (moe_op, moe_op_gelu, moe_op_silu_fused) use `sorted_token_ids`, `expert_ids`, `num_tokens_post_padded` for routing. The test setup must match exactly what the kernel expects. Study how the existing passing MoE tests (moe_op_gemm_a8w8) set up their routing data — the quantized variants use a DIFFERENT routing API (GatherIndx/ExptHist/ExptOffs/ExptData).
5. Fix the tests, run them, confirm PASS.
6. Write DONE.md with results.

## Key directories
- `kernels/moe-gemm/moe_op/` — kernel.cpp + test_parity.py for moe_op
- `kernels/moe-gemm/moe_op_gelu/` — kernel.cpp + test_parity.py
- `kernels/moe-gemm/moe_op_silu_fused/` — kernel.cpp + test_parity.py
- `kernels/moe-misc/quant_moe/` — kernel.cpp + test_parity.py
- `kernels/moe-misc/moe_routing_sigmoid_top1_fused/` — kernel.cpp + test_parity.py

IMPORTANT: Don't waste turns on passing tests. Focus ONLY on the 5 failing ones. If after thorough investigation a kernel genuinely has a bug (not a test issue), document it and move on.

## DONE.md Format
```
# Done
## Summary
## Results
| Kernel | Status | Max Diff | Notes |
## Fixes Applied
## Known Issues
```
