// SPDX-License-Identifier: MIT
// Fused GDN gating + sigmoid kernel for prefill
// Ported from Triton fused_gdn_gating_prefill.py
//
// Computes:
//   g = -exp(A_log) * softplus(a + dt_bias)
//   beta = sigmoid(b)
//
// Input: A_log [H], a [S, H], b [S, H], dt_bias [H]
// Output: g [S, H], beta [S, H]

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
// Fused gating kernel
// Each block handles one sequence position (row)
// ============================================================================
__global__ void fused_gdn_gating_sigmoid_kernel(
    const float* __restrict__ A_log,    // [H]
    const float* __restrict__ a,        // [S, H]
    const float* __restrict__ b,        // [S, H]
    const float* __restrict__ dt_bias,  // [H]
    float* __restrict__ g,              // [S, H]
    float* __restrict__ beta_out,       // [S, H]
    int S, int H,
    float softplus_beta_param,
    float softplus_threshold
) {
    int i_s = blockIdx.x;   // sequence position
    int tid = threadIdx.x;  // head index

    if (i_s >= S) return;

    for (int i_h = tid; i_h < H; i_h += blockDim.x) {
        int off = i_s * H + i_h;

        float a_val = a[off];
        float b_val = b[off];
        float A_log_val = A_log[i_h];
        float dt_bias_val = dt_bias[i_h];

        // g = -exp(A_log) * softplus(a + dt_bias)
        float x = a_val + dt_bias_val;
        float bx = softplus_beta_param * x;
        float sp;
        if (bx <= softplus_threshold) {
            sp = (1.0f / softplus_beta_param) * logf(1.0f + expf(bx));
        } else {
            sp = x;
        }
        float g_val = -expf(A_log_val) * sp;

        // beta = sigmoid(b)
        float beta_val = 1.0f / (1.0f + expf(-b_val));

        g[off] = g_val;
        beta_out[off] = beta_val;
    }
}

// ============================================================================
// Host dispatch
// ============================================================================
void dispatch_fused_gdn_gating(
    const float* A_log, const float* a, const float* b, const float* dt_bias,
    float* g, float* beta_out,
    int S, int H,
    float softplus_beta, float softplus_threshold,
    hipStream_t stream
) {
    int threads = min(256, H);
    if (threads < 1) threads = 1;
    dim3 grid(S);
    dim3 block(threads);
    hipLaunchKernelGGL(fused_gdn_gating_sigmoid_kernel, grid, block, 0, stream,
        A_log, a, b, dt_bias, g, beta_out, S, H, softplus_beta, softplus_threshold);
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

void fused_gdn_gating_wrapper(pybind11::object A_log, pybind11::object a,
                               pybind11::object b, pybind11::object dt_bias,
                               pybind11::object g, pybind11::object beta_out,
                               int S, int H,
                               float softplus_beta, float softplus_threshold) {
    dispatch_fused_gdn_gating(
        (const float*)get_data_ptr(A_log),
        (const float*)get_data_ptr(a),
        (const float*)get_data_ptr(b),
        (const float*)get_data_ptr(dt_bias),
        (float*)get_data_ptr(g),
        (float*)get_data_ptr(beta_out),
        S, H, softplus_beta, softplus_threshold, 0);
}

PYBIND11_MODULE(fused_gdn_gating_prefill_tk, m) {
    m.doc() = "Fused GDN gating + sigmoid for prefill";
    m.def("fused_gdn_gating", &fused_gdn_gating_wrapper,
          "g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b)",
          pybind11::arg("A_log"), pybind11::arg("a"), pybind11::arg("b"),
          pybind11::arg("dt_bias"), pybind11::arg("g"), pybind11::arg("beta_out"),
          pybind11::arg("S"), pybind11::arg("H"),
          pybind11::arg("softplus_beta") = 1.0f, pybind11::arg("softplus_threshold") = 20.0f);
}
