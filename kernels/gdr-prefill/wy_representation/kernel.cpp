// SPDX-License-Identifier: MIT
// WY representation kernels for chunk gated delta rule
// Ported from Triton wy_representation.py
//
// Two kernels:
// 1. chunk_scaled_dot_kkt_fwd: Compute beta * K * K^T (with optional gating) for each chunk
//    Input: k [B,T,H,K], beta [B,T,H], g [B,T,H] (optional)
//    Output: A [B,T,H,BT] strictly lower triangular
//
// 2. recompute_w_u_fwd: Compute w = A @ (k * beta * g), u = A @ (v * beta)
//    Input: k [B,T,H,K], v [B,T,H,V], beta [B,T,H], A [B,T,H,BT], g [B,T,H] (optional)
//    Output: w [B,T,H,K], u [B,T,H,V]

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
// Kernel 1: chunk_scaled_dot_kkt_fwd
// Computes: A[i,j] = beta[i] * sum_k(K[i,k] * K[j,k]) * exp(g[i] - g[j])  for i > j
// ============================================================================
__global__ void chunk_scaled_dot_kkt_fwd_kernel(
    const float* __restrict__ k,      // [B, T, H, K]
    const float* __restrict__ g,      // [B, T, H] or nullptr
    const float* __restrict__ beta,   // [B, T, H]
    float* __restrict__ A,            // [B, T, H, BT]
    int T, int H, int K, int BT,
    bool use_g
) {
    int i_t = blockIdx.x;   // chunk index
    int i_bh = blockIdx.y;
    int i_b = i_bh / H;
    int i_h = i_bh % H;
    int tid = threadIdx.x;

    int bos = i_b * T;
    int chunk_start = i_t * BT;

    // Each thread handles one (i, j) pair within the BT x BT output
    for (int idx = tid; idx < BT * BT; idx += blockDim.x) {
        int i = idx / BT;
        int j = idx % BT;
        int ti = chunk_start + i;
        int tj = chunk_start + j;

        float val = 0.0f;
        if (i > j && ti < T && tj < T) {
            // Compute dot product k[ti] . k[tj]
            float dot = 0.0f;
            for (int d = 0; d < K; d++) {
                float ki = k[((bos + ti) * H + i_h) * K + d];
                float kj = k[((bos + tj) * H + i_h) * K + d];
                dot += ki * kj;
            }

            // Apply gating
            if (use_g) {
                float gi = g[(bos + ti) * H + i_h];
                float gj = g[(bos + tj) * H + i_h];
                dot *= expf(gi - gj);
            }

            // Apply beta
            float b = beta[(bos + ti) * H + i_h];
            val = b * dot;
        }

        if (ti < T) {
            A[((bos + ti) * H + i_h) * BT + j] = val;
        }
    }
}

// ============================================================================
// Kernel 2: recompute_w_u_fwd
// w = A @ (k * beta * exp(g)), u = A @ (v * beta)
// Where A is the solved triangular matrix
// ============================================================================
__global__ void recompute_w_u_fwd_kernel(
    const float* __restrict__ k,      // [B, T, H, K]
    const float* __restrict__ v,      // [B, T, H, V]
    const float* __restrict__ beta,   // [B, T, H]
    const float* __restrict__ A,      // [B, T, H, BT]
    const float* __restrict__ g,      // [B, T, H] or nullptr
    float* __restrict__ w,            // [B, T, H, K]
    float* __restrict__ u,            // [B, T, H, V]
    int T, int H, int K, int V, int BT,
    bool use_g
) {
    int i_t = blockIdx.x;   // chunk index
    int i_bh = blockIdx.y;
    int i_b = i_bh / H;
    int i_h = i_bh % H;
    int tid = threadIdx.x;

    int bos = i_b * T;
    int chunk_start = i_t * BT;
    int chunk_end = min(chunk_start + BT, T);

    // For each row i in the chunk, compute:
    //   u[i] = sum_j A[i,j] * (v[j] * beta[j])
    //   w[i] = sum_j A[i,j] * (k[j] * beta[j] * exp(g[j]))

    for (int i = tid; i < BT; i += blockDim.x) {
        int ti = chunk_start + i;
        if (ti >= T) continue;

        // Load A row
        // Compute u[i, :V]
        for (int d = 0; d < V; d++) {
            float sum_u = 0.0f;
            for (int j = 0; j < BT && chunk_start + j < T; j++) {
                int tj = chunk_start + j;
                float a_ij = A[((bos + ti) * H + i_h) * BT + j];
                float vj = v[((bos + tj) * H + i_h) * V + d];
                float bj = beta[(bos + tj) * H + i_h];
                sum_u += a_ij * vj * bj;
            }
            u[((bos + ti) * H + i_h) * V + d] = sum_u;
        }

        // Compute w[i, :K]
        for (int d = 0; d < K; d++) {
            float sum_w = 0.0f;
            for (int j = 0; j < BT && chunk_start + j < T; j++) {
                int tj = chunk_start + j;
                float a_ij = A[((bos + ti) * H + i_h) * BT + j];
                float kj = k[((bos + tj) * H + i_h) * K + d];
                float bj = beta[(bos + tj) * H + i_h];
                float gj_factor = 1.0f;
                if (use_g) {
                    gj_factor = expf(g[(bos + tj) * H + i_h]);
                }
                sum_w += a_ij * kj * bj * gj_factor;
            }
            w[((bos + ti) * H + i_h) * K + d] = sum_w;
        }
    }
}

// ============================================================================
// Host dispatch
// ============================================================================
void dispatch_kkt(
    const float* k, const float* g, const float* beta, float* A,
    int B, int T, int H, int K, int BT, bool use_g,
    hipStream_t stream
) {
    int NT = (T + BT - 1) / BT;
    dim3 grid(NT, B * H);
    dim3 block(256);
    hipLaunchKernelGGL(chunk_scaled_dot_kkt_fwd_kernel, grid, block, 0, stream,
        k, g, beta, A, T, H, K, BT, use_g);
}

void dispatch_recompute_w_u(
    const float* k, const float* v, const float* beta,
    const float* A, const float* g,
    float* w, float* u,
    int B, int T, int H, int K, int V, int BT, bool use_g,
    hipStream_t stream
) {
    int NT = (T + BT - 1) / BT;
    dim3 grid(NT, B * H);
    dim3 block(64);
    hipLaunchKernelGGL(recompute_w_u_fwd_kernel, grid, block, 0, stream,
        k, v, beta, A, g, w, u, T, H, K, V, BT, use_g);
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

void kkt_wrapper(pybind11::object k, pybind11::object g, pybind11::object beta,
                 pybind11::object A,
                 int B, int T, int H, int K, int BT, bool use_g) {
    dispatch_kkt(
        (const float*)get_data_ptr(k),
        use_g ? (const float*)get_data_ptr(g) : nullptr,
        (const float*)get_data_ptr(beta),
        (float*)get_data_ptr(A),
        B, T, H, K, BT, use_g, 0);
}

void recompute_w_u_wrapper(pybind11::object k, pybind11::object v, pybind11::object beta,
                           pybind11::object A, pybind11::object g,
                           pybind11::object w, pybind11::object u,
                           int B, int T, int H, int K, int V, int BT, bool use_g) {
    dispatch_recompute_w_u(
        (const float*)get_data_ptr(k),
        (const float*)get_data_ptr(v),
        (const float*)get_data_ptr(beta),
        (const float*)get_data_ptr(A),
        use_g ? (const float*)get_data_ptr(g) : nullptr,
        (float*)get_data_ptr(w),
        (float*)get_data_ptr(u),
        B, T, H, K, V, BT, use_g, 0);
}

PYBIND11_MODULE(wy_representation_tk, m) {
    m.doc() = "WY representation kernels: KKT and recompute_w_u";
    m.def("chunk_scaled_dot_kkt", &kkt_wrapper,
          "Compute beta * K * K^T with optional gating",
          pybind11::arg("k"), pybind11::arg("g"), pybind11::arg("beta"),
          pybind11::arg("A"),
          pybind11::arg("B"), pybind11::arg("T"), pybind11::arg("H"),
          pybind11::arg("K"), pybind11::arg("BT"), pybind11::arg("use_g"));
    m.def("recompute_w_u", &recompute_w_u_wrapper,
          "Recompute w and u using solved A matrix",
          pybind11::arg("k"), pybind11::arg("v"), pybind11::arg("beta"),
          pybind11::arg("A"), pybind11::arg("g"),
          pybind11::arg("w"), pybind11::arg("u"),
          pybind11::arg("B"), pybind11::arg("T"), pybind11::arg("H"),
          pybind11::arg("K"), pybind11::arg("V"), pybind11::arg("BT"), pybind11::arg("use_g"));
}
