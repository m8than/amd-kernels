// SPDX-License-Identifier: MIT
// Chunk-based hidden state computation for gated delta rule (Forward only)
// Ported from Triton chunk_delta_h.py
//
// Computes hidden states h and new values v_new using recurrence:
//   For each chunk t:
//     Store h[t] = current state
//     v_new = u - w @ h  (residual)
//     h = h * exp(g_last) + k^T @ v_new  (state update)
//
// Input: k [B,T,H,K], w [B,T,H,K], u [B,T,H,V], g [B,T,H]
// Output: h [B,NT,H,K,V], v_new [B,T,H,V], final_state [N,H,K,V]

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
// Forward kernel: sequential recurrence over chunks
// Each block handles one (batch, head) pair, iterating over V columns
// ============================================================================
__global__ void chunk_gated_delta_rule_fwd_h_kernel(
    const float* __restrict__ k,      // [B, T, H, K]
    const float* __restrict__ w,      // [B, T, H, K]
    const float* __restrict__ u,      // [B, T, H, V]  (input values, already beta-scaled)
    const float* __restrict__ g,      // [B, T, H] (cumulative gate)
    const float* __restrict__ h0,     // [N, H, K, V] initial state or nullptr
    float* __restrict__ h,            // [B, NT, H, K, V] per-chunk states
    float* __restrict__ v_new,        // [B, T, H, V] new values
    float* __restrict__ ht,           // [N, H, K, V] final state or nullptr
    int T, int H, int K, int V, int BT,
    bool use_g, bool use_initial_state, bool store_final_state, bool save_new_value
) {
    int i_v = blockIdx.x;    // V block index
    int i_nh = blockIdx.y;   // batch * H
    int i_n = i_nh / H;
    int i_h = i_nh % H;

    int BV = 1;  // Process one V column at a time for simplicity
    int v_col = i_v;
    if (v_col >= V) return;

    int bos = i_n * T;
    int NT = (T + BT - 1) / BT;

    // Hidden state: h_state[k_dim] for this (head, v_col)
    // We process one v column at a time
    float h_state[256];  // K <= 256
    for (int d = 0; d < K; d++) h_state[d] = 0.0f;

    if (use_initial_state) {
        for (int d = 0; d < K; d++) {
            h_state[d] = h0[((i_n * H + i_h) * K + d) * V + v_col];
        }
    }

    for (int i_t = 0; i_t < NT; i_t++) {
        // Store current state h[i_t]
        for (int d = 0; d < K; d++) {
            h[((long long)((i_n * NT + i_t) * H + i_h) * K + d) * V + v_col] = h_state[d];
        }

        // Process chunk: for each time step in chunk
        int cs = i_t * BT;
        int ce = min(cs + BT, T);

        for (int t = cs; t < ce; t++) {
            // v_residual = u[t] - w[t] @ h_state
            float v_res = u[((bos + t) * H + i_h) * V + v_col];
            for (int d = 0; d < K; d++) {
                v_res -= w[((bos + t) * H + i_h) * K + d] * h_state[d];
            }

            if (save_new_value) {
                v_new[((bos + t) * H + i_h) * V + v_col] = v_res;
            }
        }

        // State update for the chunk:
        // First apply gating: h *= exp(g_last)
        int last_t = ce - 1;
        if (use_g) {
            float g_last = g[(bos + last_t) * H + i_h];
            float decay = expf(g_last);
            for (int d = 0; d < K; d++) {
                h_state[d] *= decay;
            }

            // Then accumulate: h += sum over t in chunk of k[t]^T @ v_new[t] * exp(g_last - g[t])
            for (int t = cs; t < ce; t++) {
                float v_res = u[((bos + t) * H + i_h) * V + v_col];
                for (int d = 0; d < K; d++) {
                    v_res -= w[((bos + t) * H + i_h) * K + d] *
                             h[((long long)((i_n * NT + i_t) * H + i_h) * K + d) * V + v_col];
                }
                // Note: we should use h at start of chunk, not current h_state
                // The Triton kernel recomputes this differently, but the effect is the same
                float g_t = g[(bos + t) * H + i_h];
                float scale = expf(g_last - g_t);
                for (int d = 0; d < K; d++) {
                    h_state[d] += k[((bos + t) * H + i_h) * K + d] * v_res * scale;
                }
            }
        } else {
            // No gating: h += k^T @ v_new
            for (int t = cs; t < ce; t++) {
                float v_res = u[((bos + t) * H + i_h) * V + v_col];
                for (int d = 0; d < K; d++) {
                    v_res -= w[((bos + t) * H + i_h) * K + d] *
                             h[((long long)((i_n * NT + i_t) * H + i_h) * K + d) * V + v_col];
                }
                for (int d = 0; d < K; d++) {
                    h_state[d] += k[((bos + t) * H + i_h) * K + d] * v_res;
                }
            }
        }
    }

    // Store final state
    if (store_final_state) {
        for (int d = 0; d < K; d++) {
            ht[((i_n * H + i_h) * K + d) * V + v_col] = h_state[d];
        }
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

void chunk_delta_h_wrapper(pybind11::object k, pybind11::object w,
                           pybind11::object u, pybind11::object g,
                           pybind11::object h0, pybind11::object h,
                           pybind11::object v_new, pybind11::object ht,
                           int B, int T, int H, int K, int V, int BT,
                           bool use_g, bool use_initial_state,
                           bool store_final_state, bool save_new_value) {
    dim3 grid(V, B * H);
    dim3 block(1);

    hipLaunchKernelGGL(chunk_gated_delta_rule_fwd_h_kernel,
        grid, block, 0, 0,
        (const float*)get_data_ptr(k),
        (const float*)get_data_ptr(w),
        (const float*)get_data_ptr(u),
        use_g ? (const float*)get_data_ptr(g) : nullptr,
        use_initial_state ? (const float*)get_data_ptr(h0) : nullptr,
        (float*)get_data_ptr(h),
        save_new_value ? (float*)get_data_ptr(v_new) : nullptr,
        store_final_state ? (float*)get_data_ptr(ht) : nullptr,
        T, H, K, V, BT,
        use_g, use_initial_state, store_final_state, save_new_value);
}

PYBIND11_MODULE(chunk_delta_h_tk, m) {
    m.doc() = "Chunk-based hidden state computation for gated delta rule";
    m.def("chunk_delta_h_fwd", &chunk_delta_h_wrapper,
          "Forward pass: compute hidden states and new values using recurrence",
          pybind11::arg("k"), pybind11::arg("w"),
          pybind11::arg("u"), pybind11::arg("g"),
          pybind11::arg("h0"), pybind11::arg("h"),
          pybind11::arg("v_new"), pybind11::arg("ht"),
          pybind11::arg("B"), pybind11::arg("T"), pybind11::arg("H"),
          pybind11::arg("K"), pybind11::arg("V"), pybind11::arg("BT"),
          pybind11::arg("use_g") = true, pybind11::arg("use_initial_state") = false,
          pybind11::arg("store_final_state") = true, pybind11::arg("save_new_value") = true);
}
