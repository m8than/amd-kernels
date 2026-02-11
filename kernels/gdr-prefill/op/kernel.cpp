// SPDX-License-Identifier: MIT
// Operation utilities ported from Triton op.py to HIP C++
// Provides device-side math function wrappers used by other kernels.
// In the Triton version, these are aliases to tl.exp, tl.log, etc.
// In HipKittens, these map to HIP intrinsics.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define HIP_CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Device math function wrappers
// These correspond to the Triton op.py aliases:
//   exp, exp2, log, log2 (with optional fast_math variants)
// ============================================================================

namespace gdr_ops {

// Standard precision math functions (IEEE mode)
__device__ __forceinline__ float op_exp(float x) { return expf(x); }
__device__ __forceinline__ float op_exp2(float x) { return exp2f(x); }
__device__ __forceinline__ float op_log(float x) { return logf(x); }
__device__ __forceinline__ float op_log2(float x) { return log2f(x); }

// Fast math variants (matching FLA_USE_FAST_OPS=1)
__device__ __forceinline__ float op_fast_exp(float x) { return __expf(x); }
__device__ __forceinline__ float op_fast_log(float x) { return __logf(x); }
__device__ __forceinline__ float op_fast_log2(float x) { return __log2f(x); }

// safe_exp: clamp positive values to -inf before exp (used in fused_cumsum_kkt)
__device__ __forceinline__ float safe_exp(float x) {
    return expf(x <= 0.0f ? x : -INFINITY);
}

// Softplus: log(1 + exp(beta * x)) / beta
__device__ __forceinline__ float softplus(float x, float beta = 1.0f, float threshold = 20.0f) {
    float bx = beta * x;
    if (bx > threshold) return x;
    return logf(1.0f + expf(bx)) / beta;
}

// Sigmoid: 1 / (1 + exp(-x))
__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// SiLU (Swish): x * sigmoid(x)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

} // namespace gdr_ops

// ============================================================================
// Test kernel: verify math functions produce correct results
// ============================================================================
__global__ void test_ops_kernel(
    const float* __restrict__ input,
    float* __restrict__ out_exp,
    float* __restrict__ out_exp2,
    float* __restrict__ out_log,
    float* __restrict__ out_log2,
    float* __restrict__ out_safe_exp,
    float* __restrict__ out_softplus,
    float* __restrict__ out_sigmoid,
    float* __restrict__ out_silu,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = input[idx];
    out_exp[idx] = gdr_ops::op_exp(x);
    out_exp2[idx] = gdr_ops::op_exp2(x);

    // Only compute log for positive inputs
    float pos_x = fabsf(x) + 0.001f;
    out_log[idx] = gdr_ops::op_log(pos_x);
    out_log2[idx] = gdr_ops::op_log2(pos_x);

    out_safe_exp[idx] = gdr_ops::safe_exp(x);
    out_softplus[idx] = gdr_ops::softplus(x, 1.0f, 20.0f);
    out_sigmoid[idx] = gdr_ops::sigmoid(x);
    out_silu[idx] = gdr_ops::silu(x);
}

// ============================================================================
// Test harness
// ============================================================================
// ============================================================================
// Pybind11 bindings
// ============================================================================
#include <pybind11/pybind11.h>

static uint64_t get_data_ptr(pybind11::object t) {
    return t.attr("data_ptr")().cast<uint64_t>();
}
static std::array<int,4> get_tensor_shape(pybind11::object t) {
    std::array<int,4> s = {1,1,1,1};
    auto shape = t.attr("shape").cast<pybind11::tuple>();
    int nd = shape.size();
    for (int i = 0; i < nd && i < 4; i++)
        s[4-nd+i] = pybind11::cast<int>(shape[i]);
    return s;
}

void test_ops_wrapper(pybind11::object input,
                      pybind11::object out_exp, pybind11::object out_exp2,
                      pybind11::object out_log, pybind11::object out_log2,
                      pybind11::object out_safe_exp, pybind11::object out_softplus,
                      pybind11::object out_sigmoid, pybind11::object out_silu, int N) {
    dim3 grid((N + 255) / 256);
    dim3 block(256);
    hipLaunchKernelGGL(test_ops_kernel, grid, block, 0, 0,
        (const float*)get_data_ptr(input),
        (float*)get_data_ptr(out_exp), (float*)get_data_ptr(out_exp2),
        (float*)get_data_ptr(out_log), (float*)get_data_ptr(out_log2),
        (float*)get_data_ptr(out_safe_exp), (float*)get_data_ptr(out_softplus),
        (float*)get_data_ptr(out_sigmoid), (float*)get_data_ptr(out_silu), N);
}

PYBIND11_MODULE(op_tk, m) {
    m.doc() = "GDR math operation utilities (exp, log, sigmoid, silu, softplus, etc.)";
    m.def("test_ops", &test_ops_wrapper,
          "Apply all math ops: exp, exp2, log, log2, safe_exp, softplus, sigmoid, silu",
          pybind11::arg("input"),
          pybind11::arg("out_exp"), pybind11::arg("out_exp2"),
          pybind11::arg("out_log"), pybind11::arg("out_log2"),
          pybind11::arg("out_safe_exp"), pybind11::arg("out_softplus"),
          pybind11::arg("out_sigmoid"), pybind11::arg("out_silu"),
          pybind11::arg("N"));
}
