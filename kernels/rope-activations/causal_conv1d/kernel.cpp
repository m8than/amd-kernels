// Causal 1D Convolution HipKittens Kernel
// Ported from reference/triton/causal_conv1d.py
//
// Implements causal 1D convolution used in Mamba/SSM architectures:
//   output[t] = sum_{k=0}^{K-1} weight[k] * input[t - K + 1 + k] + bias
// where input before t=0 is padded with zeros (or conv_state initial state).
//
// Optional SiLU activation on the output.
//
// Input layout:
//   x: (B, D, L) bf16 — batch, channels/dim, sequence length
//   w: (D, K) bf16 — weights, K = kernel width (typically 3 or 4)
//   bias: (D,) bf16 — optional bias
//
// Output layout:
//   o: (B, D, L) bf16
//
// Each thread block handles one (batch, dim_chunk) pair.
// Templated on KERNEL_WIDTH for compile-time unrolling.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>

constexpr int WARP_SIZE = 64;

// SiLU activation
__device__ __forceinline__ float device_silu(float x) {
    return x / (1.0f + expf(-x));
}

// Causal conv1d kernel
// Each block handles one batch element, processes BLOCK_D channels at a time
// Processes all L time steps sequentially per channel block
template<int KERNEL_WIDTH, bool HAS_BIAS, bool SILU_ACTIVATION>
__global__ void causal_conv1d_kernel(
    const __hip_bfloat16* __restrict__ x_ptr,    // (B, D, L)
    const __hip_bfloat16* __restrict__ w_ptr,    // (D, K)
    const __hip_bfloat16* __restrict__ bias_ptr, // (D,) or nullptr
    __hip_bfloat16* __restrict__ o_ptr,          // (B, D, L)
    int B,
    int D,
    int L
) {
    int batch = blockIdx.x;
    int dim_offset = blockIdx.y * blockDim.x + threadIdx.x;

    if (dim_offset >= D) return;

    // Base pointers for this (batch, dim) slice
    const __hip_bfloat16* x_base = x_ptr + (batch * D + dim_offset) * L;
    __hip_bfloat16* o_base = o_ptr + (batch * D + dim_offset) * L;

    // Load weights for this channel
    float w[KERNEL_WIDTH];
    #pragma unroll
    for (int k = 0; k < KERNEL_WIDTH; k++) {
        w[k] = __bfloat162float(w_ptr[dim_offset * KERNEL_WIDTH + k]);
    }

    // Load bias
    float bias = 0.0f;
    if constexpr (HAS_BIAS) {
        bias = __bfloat162float(bias_ptr[dim_offset]);
    }

    // Sliding window state: holds the last (KERNEL_WIDTH-1) values
    float state[KERNEL_WIDTH - 1];
    #pragma unroll
    for (int k = 0; k < KERNEL_WIDTH - 1; k++) {
        state[k] = 0.0f;  // Zero-padded initial state
    }

    // Process each time step
    for (int t = 0; t < L; t++) {
        float x_val = __bfloat162float(x_base[t]);

        // Compute convolution: output = bias + sum(w[k] * input[t-K+1+k])
        float acc = bias;

        // Accumulate from state (past values)
        #pragma unroll
        for (int k = 0; k < KERNEL_WIDTH - 1; k++) {
            acc += w[k] * state[k];
        }
        // Current value
        acc += w[KERNEL_WIDTH - 1] * x_val;

        // Apply SiLU if requested
        if constexpr (SILU_ACTIVATION) {
            acc = device_silu(acc);
        }

        o_base[t] = __float2bfloat16(acc);

        // Shift state window
        #pragma unroll
        for (int k = 0; k < KERNEL_WIDTH - 2; k++) {
            state[k] = state[k + 1];
        }
        if constexpr (KERNEL_WIDTH > 1) {
            state[KERNEL_WIDTH - 2] = x_val;
        }
    }
}

extern "C" {

// Launch for kernel_width=3, no bias, no activation (most common for Mamba)
void launch_causal_conv1d_k3(
    const __hip_bfloat16* x, const __hip_bfloat16* w,
    __hip_bfloat16* o, int B, int D, int L, hipStream_t stream
) {
    int threads = 256;
    dim3 grid(B, (D + threads - 1) / threads);
    causal_conv1d_kernel<3, false, false><<<grid, threads, 0, stream>>>(
        x, w, nullptr, o, B, D, L);
}

// kernel_width=4, no bias, no activation
void launch_causal_conv1d_k4(
    const __hip_bfloat16* x, const __hip_bfloat16* w,
    __hip_bfloat16* o, int B, int D, int L, hipStream_t stream
) {
    int threads = 256;
    dim3 grid(B, (D + threads - 1) / threads);
    causal_conv1d_kernel<4, false, false><<<grid, threads, 0, stream>>>(
        x, w, nullptr, o, B, D, L);
}

// kernel_width=4, with bias, with SiLU (full Mamba config)
void launch_causal_conv1d_k4_bias_silu(
    const __hip_bfloat16* x, const __hip_bfloat16* w, const __hip_bfloat16* bias,
    __hip_bfloat16* o, int B, int D, int L, hipStream_t stream
) {
    int threads = 256;
    dim3 grid(B, (D + threads - 1) / threads);
    causal_conv1d_kernel<4, true, true><<<grid, threads, 0, stream>>>(
        x, w, bias, o, B, D, L);
}

// kernel_width=3, with bias, with SiLU
void launch_causal_conv1d_k3_bias_silu(
    const __hip_bfloat16* x, const __hip_bfloat16* w, const __hip_bfloat16* bias,
    __hip_bfloat16* o, int B, int D, int L, hipStream_t stream
) {
    int threads = 256;
    dim3 grid(B, (D + threads - 1) / threads);
    causal_conv1d_kernel<3, true, true><<<grid, threads, 0, stream>>>(
        x, w, bias, o, B, D, L);
}

} // extern "C"

// ---- pybind11 bindings ----
#include <pybind11/pybind11.h>

static uint64_t _get_ptr(pybind11::object t) {
    return t.attr("data_ptr")().cast<uint64_t>();
}
static int _get_dim(pybind11::object t, int d) {
    return pybind11::cast<int>(t.attr("shape").cast<pybind11::tuple>()[d]);
}

// x: (B, D, L), w: (D, K), o: (B, D, L)
void causal_conv1d_fwd(pybind11::object x, pybind11::object w, pybind11::object o, int kernel_width) {
    int B = _get_dim(x, 0), D = _get_dim(x, 1), L = _get_dim(x, 2);
    auto xp = (const __hip_bfloat16*)_get_ptr(x);
    auto wp = (const __hip_bfloat16*)_get_ptr(w);
    auto op = (__hip_bfloat16*)_get_ptr(o);
    if (kernel_width == 3)
        launch_causal_conv1d_k3(xp, wp, op, B, D, L, 0);
    else if (kernel_width == 4)
        launch_causal_conv1d_k4(xp, wp, op, B, D, L, 0);
    else
        throw std::runtime_error("Unsupported kernel_width: " + std::to_string(kernel_width));
}

// x: (B, D, L), w: (D, K), bias: (D,), o: (B, D, L)
void causal_conv1d_bias_silu_fwd(pybind11::object x, pybind11::object w, pybind11::object bias,
                                  pybind11::object o, int kernel_width) {
    int B = _get_dim(x, 0), D = _get_dim(x, 1), L = _get_dim(x, 2);
    auto xp = (const __hip_bfloat16*)_get_ptr(x);
    auto wp = (const __hip_bfloat16*)_get_ptr(w);
    auto bp = (const __hip_bfloat16*)_get_ptr(bias);
    auto op = (__hip_bfloat16*)_get_ptr(o);
    if (kernel_width == 3)
        launch_causal_conv1d_k3_bias_silu(xp, wp, bp, op, B, D, L, 0);
    else if (kernel_width == 4)
        launch_causal_conv1d_k4_bias_silu(xp, wp, bp, op, B, D, L, 0);
    else
        throw std::runtime_error("Unsupported kernel_width: " + std::to_string(kernel_width));
}

PYBIND11_MODULE(causal_conv1d_tk, m) {
    m.doc() = "HipKittens causal conv1d kernels";
    m.def("causal_conv1d_fwd", &causal_conv1d_fwd, "Causal conv1d forward (no bias, no activation)");
    m.def("causal_conv1d_bias_silu_fwd", &causal_conv1d_bias_silu_fwd, "Causal conv1d forward (bias + SiLU)");
}
