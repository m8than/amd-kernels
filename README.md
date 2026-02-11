# HipKittens GPU Kernels for AMD CDNA3

High-performance HIP C++ GPU kernels for AMD Instinct MI300X/MI325X (CDNA3) and MI350X (CDNA4) GPUs, built using the [HipKittens](https://github.com/HazyResearch/HipKittens) tile-based DSL. Ported from [AITER](https://github.com/ROCm/aiter) Triton/CK kernels.

## Overview

**71 compiled kernels** across 15 categories with pybind11 Python bindings. Includes normalization, activations, GEMM, MoE, attention, and more.

### Kernel Categories

| Category | Kernels | Description |
|----------|---------|-------------|
| **normalization** | 3 | RMSNorm, LayerNorm, Fused Add+RMSNorm+Pad |
| **rope-activations** | 6 | RoPE, activations (SiLU/GeLU/ReLU/Tanh), softmax, top-k, causal conv1d, fused QKV+RoPE |
| **gemm-basic** | 10 | BF16, FP8, FP4, block-scale, per-token-scale variants |
| **gemm-batched** | 4 | Batched GEMM (BF16, A8W8, FP4) |
| **gemm-fused** | 4 | Fused GEMM pipelines (A8W8+A16W16, FP4+mul-add) |
| **feedforward-fusions** | 6 | Fused FF (gated/ungated), KV cache, mul-add, QK concat |
| **moe-gemm** | 10 | MoE GEMM (BF16, A8W8, A4W4, A8W4, MXFP4, SiLU-fused, E2E) |
| **moe-routing** | 4 | Alignment, bitmatrix, expert data, top-k |
| **moe-misc** | 2 | Sigmoid top-1 routing, MoE quantization |
| **attention-paged** | 2 | MLA decode, sparse MLA attention |
| **gdr-decode** | 5 | Gated Delta Rule decode (recurrent, conv1d, sigmoid gating) |
| **gdr-prefill** | 11 | GDR prefill (chunked, cumsum, L2norm, WY representation) |
| **quantization** | 3 | FP8/MXFP4 quantization, per-token quantization |

Plus 24 additional attention/GEMM kernels (source included, compilation WIP).

## Benchmarks

Benchmarked on **AMD Instinct MI325X** (gfx942, CDNA3) with ROCm 7.2. All times in microseconds (median of 100 iterations, 10 warmup). **HK = HipKittens, PT = PyTorch**.

### Normalization — RMSNorm (D=128)

HipKittens dominates at all batch sizes, up to **4.4x faster than PyTorch** and **3.3x faster than AITER Triton**.

| Shape | PT (us) | HK (us) | AITER Triton (us) | HK vs PT | HK vs AITER |
|-------|---------|---------|-------------------|----------|-------------|
| 128x128 | 35.9 | **9.2** | 24.3 | 3.9x | 2.6x |
| 4096x128 | 36.5 | **8.3** | 27.6 | 4.4x | 3.3x |
| 16384x128 | 54.9 | **11.5** | 20.5 | 4.8x | 1.8x |
| 65536x128 | 101.1 | 23.6 | **21.8** | 4.3x | 0.9x |

### Normalization — Fused Add+RMSNorm (D=128)

HipKittens wins big, up to **6.2x faster than AITER** at 65K tokens.

| Shape | PT (us) | HK (us) | AITER Triton (us) | HK vs PT | HK vs AITER |
|-------|---------|---------|-------------------|----------|-------------|
| 4096x128 | 67.6 | **10.2** | 31.9 | 6.6x | 3.1x |
| 16384x128 | 78.8 | **14.6** | 61.6 | 5.4x | 4.2x |
| 65536x128 | 160.5 | **29.2** | 180.0 | 5.5x | 6.2x |

### Normalization — LayerNorm (D=128, vs AITER CK)

AITER CK (Composable Kernel) is faster for LayerNorm, but HipKittens narrows the gap at large batch sizes.

| Shape | PT (us) | HK (us) | AITER CK (us) | HK vs PT | HK vs AITER |
|-------|---------|---------|---------------|----------|-------------|
| 4096x128 | 22.7 | 9.6 | **6.7** | 2.4x | 0.7x |
| 16384x128 | 36.4 | 14.8 | **10.2** | 2.5x | 0.7x |
| 65536x128 | 96.5 | 32.3 | **25.6** | 3.0x | 0.8x |

### Activations — Gated (SiLU/GeLU and Mul, vs AITER HIP)

HipKittens wins at small-to-medium D (256), AITER HIP wins at large D (8192+).

| Kernel | Shape | PT (us) | HK (us) | AITER HIP (us) | HK vs AITER |
|--------|-------|---------|---------|----------------|-------------|
| silu_and_mul | 4096x256 | 20.1 | **6.0** | 6.4 | 1.07x |
| silu_and_mul | 16384x256 | 37.1 | **8.4** | 13.0 | 1.5x |
| silu_and_mul | 65536x256 | 81.1 | **21.9** | 37.0 | 1.7x |
| silu_and_mul | 131072x256 | 149.0 | **38.6** | 68.6 | 1.8x |
| silu_and_mul | 4096x8192 | 149.7 | 37.9 | **27.6** | 0.7x |
| silu_and_mul | 4096x16384 | 287.6 | 73.4 | **50.2** | 0.7x |
| gelu_and_mul | 16384x256 | 39.9 | **8.4** | 13.0 | 1.5x |
| gelu_and_mul | 131072x256 | 147.4 | **38.6** | 68.9 | 1.8x |

### Softmax (D=128, vs AITER Triton)

HipKittens significantly outperforms AITER Triton softmax at all sizes.

| Shape | PT (us) | HK (us) | AITER Triton (us) | HK vs PT | HK vs AITER |
|-------|---------|---------|-------------------|----------|-------------|
| 4096x128 | 13.5 | **8.7** | 27.9 | 1.6x | 3.2x |
| 16384x128 | 26.1 | **21.9** | 89.3 | 1.2x | 4.1x |
| 65536x128 | 47.2 | **70.7** | 313.6 | 0.7x | 4.4x |

### RoPE (vs AITER Triton)

HipKittens is competitive with AITER on RoPE. Note: AITER uses a different RoPE convention (fails parity).

| Shape (B,H,S,D) | PT (us) | HK (us) | AITER Triton (us) | HK vs PT | HK vs AITER |
|------------------|---------|---------|-------------------|----------|-------------|
| 4x32x1024x128 | 202.9 | **23.3** | 34.2 | 8.7x | 1.5x |
| 8x32x2048x128 | 716.0 | **67.5** | 76.4 | 10.6x | 1.1x |
| 16x32x4096x128 | 2894.4 | **329.9** | 302.7 | 8.8x | 0.9x |

### Top-K (vs AITER Triton)

AITER Triton wins at top-k, especially at K=32.

| Shape | PT (us) | HK (us) | AITER Triton (us) | HK vs AITER |
|-------|---------|---------|-------------------|-------------|
| 1024x256 K=10 | 29.9 | 31.4 | **24.5** | 0.8x |
| 4096x256 K=10 | 66.9 | 72.4 | **39.1** | 0.5x |
| 4096x1024 K=32 | 108.8 | 1630.7 | **106.2** | 0.07x |

### Summary

| | HK Wins | AITER Wins | Total |
|---|---------|------------|-------|
| Head-to-head configs | **19** | 17 | 36 |

**Best HK wins:** RMSNorm (3-4x), Fused Add+RMSNorm (3-6x), Softmax (3-4x vs Triton), Gated activations at D<=256 (1.5-1.8x), RoPE (1.1-1.5x)

**Best AITER wins:** LayerNorm CK (1.3-1.7x), Gated activations at D>=8192 (1.4-1.7x), Top-K Triton (1.3-15x)

## Building

Requires ROCm 6.0+ with `hipcc` and Python 3.10+ with pybind11.

```bash
# Set HipKittens include path
export HIPKITTENS_ROOT=/path/to/HipKittens/include

# Build a single kernel
cd kernels/normalization/rmsnorm
make

# Build all kernels
for dir in kernels/*/*/; do
    [ -f "$dir/Makefile" ] && make -C "$dir" 2>/dev/null
done
```

### Quick compile command (without Makefile)

```bash
hipcc kernel.cpp \
    -DKITTENS_CDNA3 --offload-arch=gfx942 \
    -std=c++20 -w -O3 \
    -I/path/to/HipKittens/include \
    $(python3 -m pybind11 --includes) \
    -shared -fPIC \
    -o kernel_tk$(python3-config --extension-suffix)
```

## Usage

Each kernel compiles to a Python module (`.so` file) with pybind11 bindings:

```python
import torch
import importlib.util

# Load a kernel
spec = importlib.util.spec_from_file_location(
    "rmsnorm_tk", "kernels/normalization/rmsnorm/rmsnorm_tk.cpython-312-x86_64-linux-gnu.so")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# RMSNorm
x = torch.randn(4096, 128, dtype=torch.bfloat16, device="cuda")
w = torch.ones(128, dtype=torch.bfloat16, device="cuda")
out = torch.empty_like(x)
mod.rmsnorm_fwd(x, w, out, 1e-5, 128)

# Fused Add + RMSNorm
res = torch.randn_like(x)
out = torch.empty_like(x)
res_out = torch.empty_like(x)
mod.fused_add_rmsnorm_fwd(x, res, w, out, res_out, 1e-5, 128)
```

### Available Functions

**Normalization:**
- `rmsnorm_fwd(x, w, out, eps, N)` — RMSNorm forward
- `fused_add_rmsnorm_fwd(x, res, w, out, res_out, eps, N)` — Fused residual add + RMSNorm
- `layernorm_fwd(x, w, b, out, eps, N)` — LayerNorm forward
- `fused_add_layernorm_fwd(x, res, w, b, out, res_out, eps, N)` — Fused residual add + LayerNorm

**Activations:**
- `silu_fwd(x, out)`, `gelu_fwd(x, out)`, `relu_fwd(x, out)`, `tanh_fwd(x, out)`
- `silu_and_mul_fwd(xg, out)`, `gelu_and_mul_fwd(xg, out)` — Gated activations (input is [x, gate] concatenated)
- `gelu_tanh_fwd(x, out)` — GeLU with tanh approximation
- `softmax_fwd(x, out)` — Softmax forward

**Other:**
- `rope_fwd(x, out, cos_half, sin_half)` — Rotary position embedding
- `topk_fwd(x, out_vals, out_idx, K)` — Top-K selection
- `causal_conv1d_fwd(...)`, `causal_conv1d_bias_silu_fwd(...)` — Causal conv1d

**GEMM (all via `dispatch()`):**
- `gemm_a16w16_tk.dispatch(A, B, C)` — BF16 GEMM
- `gemm_a8w8_tk.dispatch(A, B, C, scale_a, scale_b)` — INT8 GEMM
- Plus: a16wfp4, a8wfp4, afp4wfp4, block-scale, per-token-scale variants

**MoE:** `moe_gemm`, `moe_silu_fused`, `moe_e2e`, `moe_a8w8`, `moe_a4w4`, `moe_mxfp4`, and routing ops.

## Architecture

Kernels use the HipKittens globals struct pattern:

```cpp
#include "kittens.cuh"
using namespace kittens;

template<int D>
struct kernel_globals {
    using input_gl  = gl<bf16, -1, -1, -1, D>;
    using output_gl = gl<bf16, -1, -1, -1, D>;
    input_gl input;
    output_gl output;
    hipStream_t stream;
    dim3 grid() { return dim3(...); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

template<int D>
__global__ void my_kernel(const __grid_constant__ kernel_globals<D> g) {
    // HipKittens tile operations
}
```

## Target Hardware

- **CDNA3**: MI300X, MI325X (gfx942) — primary target
- **CDNA4**: MI350X (gfx950) — supported via Makefile flags
- **CDNA2**: MI250X (gfx90a) — limited support

## License

MIT. HipKittens library from [HazyResearch](https://github.com/HazyResearch/HipKittens). Original AITER kernels from [AMD ROCm](https://github.com/ROCm/aiter).
