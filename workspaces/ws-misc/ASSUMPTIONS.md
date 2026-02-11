# Assumptions & Decisions

## Tolerance thresholds
- **Normalization kernels** (rmsnorm, layernorm, fused_add_rmsnorm_pad): Used 0.1 tolerance instead of 0.05 because these kernels compute entirely in bf16 registers, where 0.0625 (bf16 epsilon at moderate scales) is the expected quantization error.
- **Activation kernels**: 0.05 tolerance (all hit 0.0 actual diff).
- **Softmax**: 0.05 tolerance.
- **RoPE**: 0.05 tolerance (actual 0.03125 from bf16 precision).
- **TopK**: 0.1 tolerance (values match exactly in practice).
- **Causal conv1d**: 0.1 tolerance (accumulation of bf16 rounding across kernel width).
- **Quantization (INT8)**: Allow +-1 for rounding differences.
- **Quantization (FP8/MXFP4)**: Verified non-zero output only (no exact reference available for packed quantized formats).

## fused_mul_add scalar mode
The `gl<T, -1, -1, -1, N>` template requires the last tensor dimension to match N (4096). For scalar broadcast mode (`a_is_scalar=True`), used `torch.full((1,1,1,N), value)` tensors instead of scalar (1,1,1,1) tensors to satisfy the gl dimension constraint.

## Crashing kernel investigation approach
- Ran all crash-prone kernels in subprocesses to prevent GPU faults from killing the test harness.
- For shared memory issues: computed exact smem requirements from kernel source and compared to MI325X 64KB LDS limit.
- For fused_qkv_split_qk_rope_tk: ran with identity RoPE (cos=1, sin=0) and sequential/constant head values to trace data flow and identify the 4x addressing bug.

## Output tensor layouts
- Used the exact shapes implied by the kernel's `make_gl()` calls and pybind11 `get_tensor_shape()` helpers.
- For kernels using `kittens::py::bind_function`, matched the globals struct field types.
- For manual pybind11 wrappers, matched the tensor shapes documented in the kernel comments.

## MI325X LDS limit
- Used 65,536 bytes (64KB) as the per-workgroup LDS limit for AMD MI325X (gfx942). This is the standard limit for CDNA3 architecture.
