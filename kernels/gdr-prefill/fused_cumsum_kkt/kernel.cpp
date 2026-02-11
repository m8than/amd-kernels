// SPDX-License-Identifier: MIT
// Fused cumsum + K*K^T kernel for gated delta rule prefill
// Ported from Triton fused_cumsum_kkt.py
//
// Fuses two operations into one kernel:
// 1. g_cumsum = cumsum(g) within each chunk
// 2. A[i,j] = beta[i] * (K[i] @ K[j]^T) * safe_exp(g_cumsum[i] - g_cumsum[j])
//    for i > j (strictly lower triangular)
//
// Input: g [B,T,H], k [B,T,Hg,K], beta [B,T,H]
// Output: g_cumsum [B,T,H], A [B,T,H,BT]

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

__device__ __forceinline__ float safe_exp(float x) {
    return expf(x <= 0.0f ? x : -INFINITY);
}

// ============================================================================
// Fused cumsum + KKT kernel
// Each block handles one chunk for one (batch, head) pair
// ============================================================================
__global__ void fused_cumsum_kkt_kernel(
    const float* __restrict__ g,          // [B, T, H]
    const float* __restrict__ k,          // [B, T, Hg, K]
    const float* __restrict__ beta,       // [B, T, H]
    float* __restrict__ g_cumsum,         // [B, T, H]
    float* __restrict__ A,                // [B, T, H, BT]
    int T, int H, int Hg, int K, int BT
) {
    int i_t = blockIdx.x;   // chunk index
    int i_bh = blockIdx.y;  // batch * H
    int i_b = i_bh / H;
    int i_h = i_bh % H;
    int tid = threadIdx.x;

    int bos = i_b * T;
    int cs = i_t * BT;
    int T_seq = T;

    // GQA: map attention head to key-value head
    int i_hg = i_h / (H / Hg);

    // Step 1: Compute cumulative sum of g within the chunk
    __shared__ float s_g_cumsum[128]; // max BT
    if (tid == 0) {
        float running = 0.0f;
        for (int i = 0; i < BT && cs + i < T_seq; i++) {
            running += g[(bos + cs + i) * H + i_h];
            s_g_cumsum[i] = running;
        }
    }
    __syncthreads();

    // Write g_cumsum to global memory
    for (int i = tid; i < BT && cs + i < T_seq; i += blockDim.x) {
        g_cumsum[(bos + cs + i) * H + i_h] = s_g_cumsum[i];
    }

    // Step 2: Compute A = beta * K @ K^T * safe_exp(g_diff), strictly lower triangular
    for (int idx = tid; idx < BT * BT; idx += blockDim.x) {
        int i = idx / BT;
        int j = idx % BT;
        int ti = cs + i;
        int tj = cs + j;
        float val = 0.0f;

        if (i > j && ti < T_seq && tj < T_seq) {
            // K dot product
            float dot = 0.0f;
            for (int d = 0; d < K; d++) {
                float ki = k[((bos + ti) * Hg + i_hg) * K + d];
                float kj = k[((bos + tj) * Hg + i_hg) * K + d];
                dot += ki * kj;
            }

            // Gate difference using cumulative sum
            float g_diff = s_g_cumsum[i] - s_g_cumsum[j];
            dot *= safe_exp(g_diff);

            // Apply beta
            val = beta[(bos + ti) * H + i_h] * dot;
        }

        if (ti < T_seq) {
            A[((bos + ti) * H + i_h) * BT + j] = val;
        }
    }
}

// ============================================================================
// Host dispatch
// ============================================================================
void dispatch_fused_cumsum_kkt(
    const float* g, const float* k, const float* beta,
    float* g_cumsum, float* A,
    int B, int T, int H, int Hg, int K, int BT,
    hipStream_t stream
) {
    int NT = (T + BT - 1) / BT;
    dim3 grid(NT, B * H);
    dim3 block(256);
    hipLaunchKernelGGL(fused_cumsum_kkt_kernel, grid, block, 0, stream,
        g, k, beta, g_cumsum, A, T, H, Hg, K, BT);
}

// ============================================================================
// Pybind11 bindings
// ============================================================================
#include <pybind11/pybind11.h>

static uint64_t _get_ptr(pybind11::object t) {
    return t.attr("data_ptr")().cast<uint64_t>();
}

void fused_cumsum_kkt_wrapper(
    pybind11::object g, pybind11::object k, pybind11::object beta,
    pybind11::object g_cumsum, pybind11::object A,
    int B, int T, int H, int Hg, int K, int BT)
{
    dispatch_fused_cumsum_kkt(
        (const float*)_get_ptr(g),
        (const float*)_get_ptr(k),
        (const float*)_get_ptr(beta),
        (float*)_get_ptr(g_cumsum),
        (float*)_get_ptr(A),
        B, T, H, Hg, K, BT, 0);
}

PYBIND11_MODULE(fused_cumsum_kkt_tk, m) {
    m.doc() = "Fused cumsum + K*K^T kernel for gated delta rule";
    m.def("fused_cumsum_kkt", &fused_cumsum_kkt_wrapper,
          "Fused cumsum and KKT computation");
}

#if 0 // Test harness (disabled)
int main_test() {
    int B = 1, T = 128, H = 4, Hg = 4, K = 64, BT = 64;

    size_t sz_g = B * T * H;
    size_t sz_k = B * T * Hg * K;
    size_t sz_A = B * T * H * BT;

    std::vector<float> h_g(sz_g), h_k(sz_k), h_beta(sz_g);
    std::vector<float> h_g_cumsum(sz_g), h_A(sz_A, 0.0f);
    std::vector<float> ref_g_cumsum(sz_g), ref_A(sz_A, 0.0f);

    srand(42);
    auto randf = []() { return (float)(rand() % 1000 - 500) / 1000.0f; };
    for (auto& x : h_g) x = randf() * 0.1f;
    for (auto& x : h_k) x = randf();
    for (auto& x : h_beta) x = randf() * 0.5f;

    // CPU reference
    int NT = (T + BT - 1) / BT;
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++) {
            int hg = h / (H / Hg);
            for (int nt = 0; nt < NT; nt++) {
                int cs = nt * BT;
                // cumsum
                float running = 0.0f;
                for (int i = 0; i < BT && cs + i < T; i++) {
                    running += h_g[(b * T + cs + i) * H + h];
                    ref_g_cumsum[(b * T + cs + i) * H + h] = running;
                }
                // KKT
                for (int i = 0; i < BT && cs + i < T; i++)
                    for (int j = 0; j < i; j++) {
                        float dot = 0.0f;
                        for (int d = 0; d < K; d++)
                            dot += h_k[((b * T + cs + i) * Hg + hg) * K + d] *
                                   h_k[((b * T + cs + j) * Hg + hg) * K + d];
                        float gd = ref_g_cumsum[(b * T + cs + i) * H + h] -
                                   ref_g_cumsum[(b * T + cs + j) * H + h];
                        float se = (gd <= 0.0f) ? expf(gd) : 0.0f;
                        ref_A[((b * T + cs + i) * H + h) * BT + j] = h_beta[(b * T + cs + i) * H + h] * dot * se;
                    }
            }
        }

    // GPU
    float *d_g, *d_k, *d_beta, *d_g_cumsum, *d_A;
    HIP_CHECK(hipMalloc(&d_g, sz_g * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_k, sz_k * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_beta, sz_g * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_g_cumsum, sz_g * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_A, sz_A * sizeof(float)));
    HIP_CHECK(hipMemset(d_A, 0, sz_A * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_g, h_g.data(), sz_g * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_k, h_k.data(), sz_k * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_beta, h_beta.data(), sz_g * sizeof(float), hipMemcpyHostToDevice));

    dispatch_fused_cumsum_kkt(d_g, d_k, d_beta, d_g_cumsum, d_A, B, T, H, Hg, K, BT, 0);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_g_cumsum.data(), d_g_cumsum, sz_g * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_A.data(), d_A, sz_A * sizeof(float), hipMemcpyDeviceToHost));

    // Check cumsum
    float max_err_g = 0.0f;
    for (size_t i = 0; i < sz_g; i++) {
        float err = fabsf(h_g_cumsum[i] - ref_g_cumsum[i]);
        if (err > max_err_g) max_err_g = err;
    }
    printf("fused_cumsum: max_error=%.6e %s\n", max_err_g, max_err_g < 1e-4 ? "PASS" : "FAIL");

    // Check A
    float max_err_A = 0.0f;
    for (size_t i = 0; i < sz_A; i++) {
        float err = fabsf(h_A[i] - ref_A[i]);
        if (err > max_err_A) max_err_A = err;
    }
    printf("fused_kkt: max_error=%.6e %s\n", max_err_A, max_err_A < 1e-3 ? "PASS" : "FAIL");

    HIP_CHECK(hipFree(d_g));
    HIP_CHECK(hipFree(d_k));
    HIP_CHECK(hipFree(d_beta));
    HIP_CHECK(hipFree(d_g_cumsum));
    HIP_CHECK(hipFree(d_A));

    return 0;
}
#endif
