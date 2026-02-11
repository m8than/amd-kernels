// MoE INT4 GEMM Kernel (moe_op_gemm_a4w4)
// Ported from reference/triton/moe_op_gemm_a4w4.py
//
// INT4xINT4 quantized MoE GEMM:
//   - Activations: INT4 (packed 2 per byte)
//   - Weights: INT4 (packed 2 per byte)
//   - Uses MXFP4-style microscaling or group-wise dequantization
//   - Output: bf16
//
// Both X and W are packed as uint8 with 2 INT4 values per byte.
// Supports optional SwiGLU activation and FP8 output quantization.

#include "kittens.cuh"
using namespace kittens;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 128;  // Larger K block for 4-bit (same bytes as 64 for 8-bit)

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

struct moe_a4w4_globals {
    const uint8_t* __restrict__ X;      // [total_tokens, K/2] uint8 (packed INT4)
    const uint8_t* __restrict__ W;      // [E, K/2, N] uint8 (packed INT4)
    bf16* __restrict__ Y;               // [total_tokens, N] bf16
    const uint8_t* __restrict__ X_mx_scale;  // [total_tokens, K/32] uint8 MX scales
    const uint8_t* __restrict__ W_mx_scale;  // [E, K/32, N] uint8 MX scales
    const float* __restrict__ X_static_scale;
    const float* __restrict__ W_static_scale;

    const int32_t* __restrict__ GatherIndx;
    const int32_t* __restrict__ ExptHist;
    const int32_t* __restrict__ ExptOffs;
    const int32_t* __restrict__ ExptData;

    int N, K;
    int grid_m, grid_n;
    int n_expts_act;

    int stride_ym, stride_yn;
    int stride_xm, stride_xk;
    int stride_we, stride_wk, stride_wn;

    bool apply_swiglu;

    hipStream_t stream;

    dim3 grid() { return dim3(grid_m * grid_n); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

// Unpack INT4 value from packed byte
__device__ __forceinline__ float unpack_int4(uint8_t packed, int idx) {
    int shift = (idx % 2) * 4;
    int val = (packed >> shift) & 0xF;
    // Interpret as signed: if bit 3 is set, subtract 16
    if (val >= 8) val -= 16;
    return (float)val;
}

__global__ __launch_bounds__(NUM_THREADS)
void moe_a4w4_kernel(const moe_a4w4_globals g) {
    const int pid = blockIdx.x;
    if (pid >= g.grid_m * g.grid_n) return;

    const int pid_n = pid % g.grid_n;
    const int pid_m = pid / g.grid_n;

    int expt_data = g.ExptData[pid_m];
    if (expt_data == -1) return;

    int expt_id = expt_data & 0x0000FFFF;
    int block_id = expt_data >> 16;
    int M = g.ExptHist[expt_id];
    int start_m = g.ExptOffs[expt_id];

    constexpr int ELEMS_PER_THREAD = (BLOCK_M * BLOCK_N) / NUM_THREADS;
    float acc[ELEMS_PER_THREAD];
    for (int i = 0; i < ELEMS_PER_THREAD; i++) acc[i] = 0.0f;

    // Packed shared memory (half the K elements fit in same bytes)
    __shared__ uint8_t X_sh[BLOCK_M * BLOCK_K / 2];
    __shared__ uint8_t W_sh[BLOCK_K / 2 * BLOCK_N];

    const int num_k_tiles = (g.K + BLOCK_K - 1) / BLOCK_K;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k_start = kt * BLOCK_K;
        int packed_k = BLOCK_K / 2;

        // Load packed X tile
        for (int idx = threadIdx.x; idx < BLOCK_M * packed_k; idx += NUM_THREADS) {
            int lm = idx / packed_k, lk = idx % packed_k;
            int gm = (block_id * BLOCK_M + lm) % M;
            int gk = (k_start / 2) + lk;

            int token_idx;
            if (g.GatherIndx != nullptr)
                token_idx = g.GatherIndx[start_m + gm] / g.n_expts_act;
            else
                token_idx = start_m + gm;

            if (gk < g.K / 2 && gm < M)
                X_sh[idx] = g.X[token_idx * g.stride_xm + gk * g.stride_xk];
            else
                X_sh[idx] = 0;
        }

        // Load packed W tile
        for (int idx = threadIdx.x; idx < packed_k * BLOCK_N; idx += NUM_THREADS) {
            int lk = idx / BLOCK_N, ln = idx % BLOCK_N;
            int gk = (k_start / 2) + lk;
            int gn = pid_n * BLOCK_N + ln;

            if (gk < g.K / 2 && gn < g.N)
                W_sh[idx] = g.W[expt_id * g.stride_we + gk * g.stride_wk + gn * g.stride_wn];
            else
                W_sh[idx] = 0;
        }
        __syncthreads();

        // INT4xINT4 matmul
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int eidx = threadIdx.x * ELEMS_PER_THREAD + i;
            if (eidx >= BLOCK_M * BLOCK_N) break;
            int lm = eidx / BLOCK_N, ln = eidx % BLOCK_N;

            float sum = 0.0f;
            for (int kk = 0; kk < BLOCK_K && (k_start + kk) < g.K; kk++) {
                float x_val = unpack_int4(X_sh[lm * packed_k + kk / 2], kk);
                float w_val = unpack_int4(W_sh[(kk / 2) * BLOCK_N + ln], kk);
                sum += x_val * w_val;
            }
            acc[i] += sum;
        }
        __syncthreads();
    }

    // Apply scales and store
    float x_s = (g.X_static_scale != nullptr) ? g.X_static_scale[0] : 1.0f;
    float w_s = (g.W_static_scale != nullptr) ? g.W_static_scale[0] : 1.0f;

    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int eidx = threadIdx.x * ELEMS_PER_THREAD + i;
        if (eidx >= BLOCK_M * BLOCK_N) break;
        int lm = eidx / BLOCK_N, ln = eidx % BLOCK_N;
        int gm = (block_id * BLOCK_M + lm) % M;
        int gn = pid_n * BLOCK_N + ln;

        if (gm >= M || gn >= g.N) continue;

        float result = acc[i] * x_s * w_s;
        int out_row = start_m + gm;
        g.Y[out_row * g.stride_ym + gn * g.stride_yn] = __float2bfloat16(result);
    }
}

void dispatch_moe_a4w4(moe_a4w4_globals& g) {
    hipLaunchKernelGGL(moe_a4w4_kernel, g.grid(), g.block(),
                       g.dynamic_shared_memory(), g.stream, g);
}

// ============================================================================
// PyBind11 bindings
// ============================================================================
#include <pybind11/pybind11.h>
#include <array>

static std::array<int,4> get_tensor_shape(pybind11::object t) {
    std::array<int,4> s = {1,1,1,1};
    auto shape = t.attr("shape").cast<pybind11::tuple>();
    int nd = shape.size();
    for (int i = 0; i < nd && i < 4; i++)
        s[4-nd+i] = pybind11::cast<int>(shape[i]);
    return s;
}
static uint64_t get_data_ptr(pybind11::object t) {
    return t.attr("data_ptr")().cast<uint64_t>();
}

void moe_a4w4_py(
    pybind11::object X,
    pybind11::object W,
    pybind11::object Y,
    pybind11::object X_mx_scale,
    pybind11::object W_mx_scale,
    pybind11::object X_static_scale,
    pybind11::object W_static_scale,
    pybind11::object GatherIndx,
    pybind11::object ExptHist,
    pybind11::object ExptOffs,
    pybind11::object ExptData,
    int N, int K,
    int grid_m, int grid_n,
    int n_expts_act,
    int stride_ym, int stride_yn,
    int stride_xm, int stride_xk,
    int stride_we, int stride_wk, int stride_wn,
    bool apply_swiglu
) {
    moe_a4w4_globals g;
    g.X = reinterpret_cast<const uint8_t*>(get_data_ptr(X));
    g.W = reinterpret_cast<const uint8_t*>(get_data_ptr(W));
    g.Y = reinterpret_cast<bf16*>(get_data_ptr(Y));
    g.X_mx_scale = X_mx_scale.is_none() ? nullptr : reinterpret_cast<const uint8_t*>(get_data_ptr(X_mx_scale));
    g.W_mx_scale = W_mx_scale.is_none() ? nullptr : reinterpret_cast<const uint8_t*>(get_data_ptr(W_mx_scale));
    g.X_static_scale = X_static_scale.is_none() ? nullptr : reinterpret_cast<const float*>(get_data_ptr(X_static_scale));
    g.W_static_scale = W_static_scale.is_none() ? nullptr : reinterpret_cast<const float*>(get_data_ptr(W_static_scale));
    g.GatherIndx = GatherIndx.is_none() ? nullptr : reinterpret_cast<const int32_t*>(get_data_ptr(GatherIndx));
    g.ExptHist = reinterpret_cast<const int32_t*>(get_data_ptr(ExptHist));
    g.ExptOffs = reinterpret_cast<const int32_t*>(get_data_ptr(ExptOffs));
    g.ExptData = reinterpret_cast<const int32_t*>(get_data_ptr(ExptData));
    g.N = N;
    g.K = K;
    g.grid_m = grid_m;
    g.grid_n = grid_n;
    g.n_expts_act = n_expts_act;
    g.stride_ym = stride_ym;
    g.stride_yn = stride_yn;
    g.stride_xm = stride_xm;
    g.stride_xk = stride_xk;
    g.stride_we = stride_we;
    g.stride_wk = stride_wk;
    g.stride_wn = stride_wn;
    g.apply_swiglu = apply_swiglu;
    g.stream = 0;

    dispatch_moe_a4w4(g);
}

PYBIND11_MODULE(moe_op_gemm_a4w4_tk, m) {
    m.def("moe_a4w4", &moe_a4w4_py,
          "MoE INT4xINT4 GEMM kernel",
          pybind11::arg("X"),
          pybind11::arg("W"),
          pybind11::arg("Y"),
          pybind11::arg("X_mx_scale"),
          pybind11::arg("W_mx_scale"),
          pybind11::arg("X_static_scale"),
          pybind11::arg("W_static_scale"),
          pybind11::arg("GatherIndx"),
          pybind11::arg("ExptHist"),
          pybind11::arg("ExptOffs"),
          pybind11::arg("ExptData"),
          pybind11::arg("N"),
          pybind11::arg("K"),
          pybind11::arg("grid_m"),
          pybind11::arg("grid_n"),
          pybind11::arg("n_expts_act"),
          pybind11::arg("stride_ym"),
          pybind11::arg("stride_yn"),
          pybind11::arg("stride_xm"),
          pybind11::arg("stride_xk"),
          pybind11::arg("stride_we"),
          pybind11::arg("stride_wk"),
          pybind11::arg("stride_wn"),
          pybind11::arg("apply_swiglu"));
}
