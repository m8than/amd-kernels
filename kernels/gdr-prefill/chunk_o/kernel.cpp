// SPDX-License-Identifier: MIT
// Chunk output computation kernel (Forward only)
// Ported from Triton chunk_o.py
//
// Computes: o = scale * (Q @ H + causal_mask(Q @ K^T) @ V) * exp(g)
// Where H is the per-chunk hidden state, and the second term is intra-chunk attention.
//
// Input: q [B,T,H,K], k [B,T,H,K], v [B,T,H,V], h [B,NT,H,K,V], g [B,T,H]
// Output: o [B,T,H,V]

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
// Chunk output forward kernel
// Each block handles one (chunk, batch, head, v_block) tuple
// ============================================================================
__global__ void chunk_fwd_o_kernel(
    const float* __restrict__ q,      // [B, T, H, K]
    const float* __restrict__ k,      // [B, T, H, K]
    const float* __restrict__ v,      // [B, T, H, V]
    const float* __restrict__ h,      // [B, NT, H, K, V]
    const float* __restrict__ g,      // [B, T, H] or nullptr
    float* __restrict__ o,            // [B, T, H, V]
    float scale,
    int T, int H, int K, int V, int BT,
    bool use_g
) {
    int i_v = blockIdx.x;    // V block index
    int i_t = blockIdx.y;    // chunk index
    int i_bh = blockIdx.z;   // batch * H
    int i_b = i_bh / H;
    int i_h = i_bh % H;
    int tid = threadIdx.x;

    int bos = i_b * T;
    int cs = i_t * BT;
    int NT = (T + BT - 1) / BT;
    int i_tg = i_b * NT + i_t;

    // Process one row of the chunk at a time
    for (int row = tid; row < BT && cs + row < T; row += blockDim.x) {
        int ti = cs + row;

        // Part 1: Q @ H contribution (inter-chunk)
        // o[ti, v_col] = sum_k q[ti, k] * h[chunk_idx, k, v_col]
        float o_val = 0.0f;
        for (int dk = 0; dk < K; dk++) {
            float q_val = q[((bos + ti) * H + i_h) * K + dk];
            float h_val = h[((long long)(i_tg * H + i_h) * K + dk) * V + i_v];
            o_val += q_val * h_val;
        }

        // Apply gating to inter-chunk part
        if (use_g) {
            float gi = g[(bos + ti) * H + i_h];
            o_val *= expf(gi);
        }

        // Part 2: Causal Q @ K^T @ V contribution (intra-chunk)
        float intra = 0.0f;
        for (int j = 0; j <= row; j++) {
            int tj = cs + j;
            if (tj >= T) break;

            // Compute q[ti] @ k[tj]
            float qk = 0.0f;
            for (int dk = 0; dk < K; dk++) {
                qk += q[((bos + ti) * H + i_h) * K + dk] *
                      k[((bos + tj) * H + i_h) * K + dk];
            }

            // Apply gating
            if (use_g) {
                float gi = g[(bos + ti) * H + i_h];
                float gj = g[(bos + tj) * H + i_h];
                qk *= expf(gi - gj);
            }

            intra += qk * v[((bos + tj) * H + i_h) * V + i_v];
        }

        o[((bos + ti) * H + i_h) * V + i_v] = (o_val + intra) * scale;
    }
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

void chunk_fwd_o_wrapper(pybind11::object q, pybind11::object k, pybind11::object v,
                         pybind11::object h, pybind11::object g, pybind11::object o,
                         float scale, int B, int T, int H, int K, int V, int BT, bool use_g) {
    int NT = (T + BT - 1) / BT;
    dim3 grid(V, NT, B * H);
    dim3 block(BT);
    hipLaunchKernelGGL(chunk_fwd_o_kernel, grid, block, 0, 0,
        (const float*)get_data_ptr(q),
        (const float*)get_data_ptr(k),
        (const float*)get_data_ptr(v),
        (const float*)get_data_ptr(h),
        use_g ? (const float*)get_data_ptr(g) : nullptr,
        (float*)get_data_ptr(o),
        scale, T, H, K, V, BT, use_g);
}

PYBIND11_MODULE(chunk_o_tk, m) {
    m.doc() = "Chunk output computation: o = scale * (Q @ H + causal(Q @ K^T) @ V) * exp(g)";
    m.def("chunk_fwd_o", &chunk_fwd_o_wrapper,
          "Compute chunk forward output",
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
          pybind11::arg("h"), pybind11::arg("g"), pybind11::arg("o"),
          pybind11::arg("scale"), pybind11::arg("B"), pybind11::arg("T"),
          pybind11::arg("H"), pybind11::arg("K"), pybind11::arg("V"),
          pybind11::arg("BT"), pybind11::arg("use_g") = true);
}
