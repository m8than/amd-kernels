// MoE INT8xINT4 GEMM Kernel (moe_op_gemm_a8w4)
// Ported from reference/triton/moe_op_gemm_a8w4.py
//
// Mixed-precision MoE GEMM:
//   - Activations: INT8
//   - Weights: INT4 (packed 2 per byte)
//   - Output: bf16 after dequantization with per-group scales
//
// Weight packing: 2 INT4 values packed per uint8
//   w_unpacked = (w_packed >> shift) & 0xF, shift = (col % 2) * 4
// Dequantization: (w_int4 - zero_point) * scale

#include "kittens.cuh"
using namespace kittens;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 64;

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

struct moe_a8w4_globals {
    const int8_t* __restrict__ X;       // [total_tokens, K] int8
    const uint8_t* __restrict__ W;      // [E, K/2, N] uint8 (packed INT4)
    bf16* __restrict__ Y;               // [total_tokens, N] bf16
    const float* __restrict__ X_scale;  // per-tensor
    const float* __restrict__ W_scale;  // [E, K/group_size, N] per-group
    const float* __restrict__ W_zp;     // [E, K/group_size, N] zero points (optional)

    const int32_t* __restrict__ GatherIndx;
    const int32_t* __restrict__ ExptHist;
    const int32_t* __restrict__ ExptOffs;
    const int32_t* __restrict__ ExptData;

    int N, K;
    int group_size;   // Quantization group size along K
    int grid_m, grid_n;
    int n_expts_act;

    int stride_ym, stride_yn;
    int stride_xm, stride_xk;
    int stride_we, stride_wk, stride_wn;  // W packed strides
    int stride_wse, stride_wsk, stride_wsn;

    bool has_zero_point;

    hipStream_t stream;

    dim3 grid() { return dim3(grid_m * grid_n); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

__global__ __launch_bounds__(NUM_THREADS)
void moe_a8w4_kernel(const moe_a8w4_globals g) {
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

    __shared__ int8_t X_sh[BLOCK_M * BLOCK_K];
    __shared__ uint8_t W_sh[BLOCK_K / 2 * BLOCK_N];  // Packed

    const int num_k_tiles = (g.K + BLOCK_K - 1) / BLOCK_K;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k_start = kt * BLOCK_K;

        // Load X tile
        for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_K; idx += NUM_THREADS) {
            int lm = idx / BLOCK_K, lk = idx % BLOCK_K;
            int gm = (block_id * BLOCK_M + lm) % M;
            int gk = k_start + lk;

            int token_idx;
            if (g.GatherIndx != nullptr)
                token_idx = g.GatherIndx[start_m + gm] / g.n_expts_act;
            else
                token_idx = start_m + gm;

            if (gk < g.K && gm < M)
                X_sh[idx] = g.X[token_idx * g.stride_xm + gk * g.stride_xk];
            else
                X_sh[idx] = 0;
        }

        // Load packed W tile [K/2, N]
        const int packed_k = BLOCK_K / 2;
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

        // Compute with INT4 unpacking
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int eidx = threadIdx.x * ELEMS_PER_THREAD + i;
            if (eidx >= BLOCK_M * BLOCK_N) break;
            int lm = eidx / BLOCK_N, ln = eidx % BLOCK_N;

            float sum = 0.0f;
            for (int kk = 0; kk < BLOCK_K && (k_start + kk) < g.K; kk++) {
                int8_t x_val = X_sh[lm * BLOCK_K + kk];

                // Unpack INT4: 2 values per byte
                uint8_t packed = W_sh[(kk / 2) * BLOCK_N + ln];
                int shift = (kk % 2) * 4;
                int w_int4 = (packed >> shift) & 0xF;

                // Dequantize weight
                int k_group = (k_start + kk) / g.group_size;
                int gn = pid_n * BLOCK_N + ln;
                float scale = g.W_scale[expt_id * g.stride_wse +
                                         k_group * g.stride_wsk +
                                         gn * g.stride_wsn];

                float w_dequant;
                if (g.has_zero_point) {
                    float zp = g.W_zp[expt_id * g.stride_wse +
                                       k_group * g.stride_wsk +
                                       gn * g.stride_wsn];
                    w_dequant = ((float)w_int4 - zp) * scale;
                } else {
                    w_dequant = ((float)w_int4 - 8.0f) * scale;  // Default ZP=8 for INT4
                }

                sum += (float)x_val * w_dequant;
            }
            acc[i] += sum;
        }
        __syncthreads();
    }

    // Apply X scale and store
    float x_scale = (g.X_scale != nullptr) ? g.X_scale[0] : 1.0f;

    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int eidx = threadIdx.x * ELEMS_PER_THREAD + i;
        if (eidx >= BLOCK_M * BLOCK_N) break;
        int lm = eidx / BLOCK_N, ln = eidx % BLOCK_N;
        int gm = (block_id * BLOCK_M + lm) % M;
        int gn = pid_n * BLOCK_N + ln;

        if (gm >= M || gn >= g.N) continue;

        int out_row = start_m + gm;
        g.Y[out_row * g.stride_ym + gn * g.stride_yn] =
            __float2bfloat16(acc[i] * x_scale);
    }
}

void dispatch_moe_a8w4(moe_a8w4_globals& g) {
    hipLaunchKernelGGL(moe_a8w4_kernel, g.grid(), g.block(),
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

void moe_a8w4_py(
    pybind11::object X,
    pybind11::object W,
    pybind11::object Y,
    pybind11::object X_scale,
    pybind11::object W_scale,
    pybind11::object W_zp,
    pybind11::object GatherIndx,
    pybind11::object ExptHist,
    pybind11::object ExptOffs,
    pybind11::object ExptData,
    int N, int K,
    int group_size,
    int grid_m, int grid_n,
    int n_expts_act,
    int stride_ym, int stride_yn,
    int stride_xm, int stride_xk,
    int stride_we, int stride_wk, int stride_wn,
    int stride_wse, int stride_wsk, int stride_wsn,
    bool has_zero_point
) {
    moe_a8w4_globals g;
    g.X = reinterpret_cast<const int8_t*>(get_data_ptr(X));
    g.W = reinterpret_cast<const uint8_t*>(get_data_ptr(W));
    g.Y = reinterpret_cast<bf16*>(get_data_ptr(Y));
    g.X_scale = X_scale.is_none() ? nullptr : reinterpret_cast<const float*>(get_data_ptr(X_scale));
    g.W_scale = reinterpret_cast<const float*>(get_data_ptr(W_scale));
    g.W_zp = W_zp.is_none() ? nullptr : reinterpret_cast<const float*>(get_data_ptr(W_zp));
    g.GatherIndx = GatherIndx.is_none() ? nullptr : reinterpret_cast<const int32_t*>(get_data_ptr(GatherIndx));
    g.ExptHist = reinterpret_cast<const int32_t*>(get_data_ptr(ExptHist));
    g.ExptOffs = reinterpret_cast<const int32_t*>(get_data_ptr(ExptOffs));
    g.ExptData = reinterpret_cast<const int32_t*>(get_data_ptr(ExptData));
    g.N = N;
    g.K = K;
    g.group_size = group_size;
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
    g.stride_wse = stride_wse;
    g.stride_wsk = stride_wsk;
    g.stride_wsn = stride_wsn;
    g.has_zero_point = has_zero_point;
    g.stream = 0;

    dispatch_moe_a8w4(g);
}

PYBIND11_MODULE(moe_op_gemm_a8w4_tk, m) {
    m.def("moe_a8w4", &moe_a8w4_py,
          "MoE INT8xINT4 GEMM kernel",
          pybind11::arg("X"),
          pybind11::arg("W"),
          pybind11::arg("Y"),
          pybind11::arg("X_scale"),
          pybind11::arg("W_scale"),
          pybind11::arg("W_zp"),
          pybind11::arg("GatherIndx"),
          pybind11::arg("ExptHist"),
          pybind11::arg("ExptOffs"),
          pybind11::arg("ExptData"),
          pybind11::arg("N"),
          pybind11::arg("K"),
          pybind11::arg("group_size"),
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
          pybind11::arg("stride_wse"),
          pybind11::arg("stride_wsk"),
          pybind11::arg("stride_wsn"),
          pybind11::arg("has_zero_point"));
}
