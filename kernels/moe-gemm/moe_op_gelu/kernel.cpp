// MoE GEMM with GeLU Activation (moe_op_gelu)
// Ported from reference/triton/moe_op_gelu.py
//
// Same as moe_op but applies GeLU (tanh approximation) to the GEMM output
// before writing back. The activation is applied when MUL_ROUTED_WEIGHT=false
// (i.e., for the first layer of a gated MLP before combining with routing weights).
//
// C = GeLU(A @ B[expert])
//
// GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

#include "kittens.cuh"
using namespace kittens;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 32;

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

struct moe_gelu_globals {
    const bf16* __restrict__ A;
    const bf16* __restrict__ B;
    bf16* __restrict__ C;
    const float* __restrict__ topk_weights;
    const int32_t* __restrict__ sorted_token_ids;
    const int32_t* __restrict__ expert_ids;
    const int32_t* __restrict__ num_tokens_post_padded;

    int N, K;
    int num_valid_tokens;
    int top_k;

    int stride_am, stride_ak;
    int stride_be, stride_bk, stride_bn;
    int stride_cm, stride_cn;

    bool mul_routed_weight;  // If true, multiply by weight. GeLU only when false.

    hipStream_t stream;

    dim3 grid() {
        int est = num_valid_tokens * 2;
        return dim3(((est + BLOCK_M - 1) / BLOCK_M) * ((N + BLOCK_N - 1) / BLOCK_N));
    }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

// GeLU tanh approximation
__device__ __forceinline__ float gelu_tanh(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ __launch_bounds__(NUM_THREADS)
void moe_gelu_kernel(const moe_gelu_globals g) {
    const int pid = blockIdx.x;
    const int num_tokens_post_padded = g.num_tokens_post_padded[0];

    const int num_pid_m = (num_tokens_post_padded + BLOCK_M - 1) / BLOCK_M;
    const int num_pid_n = (g.N + BLOCK_N - 1) / BLOCK_N;
    const int total_tiles = num_pid_m * num_pid_n;
    if (pid >= total_tiles) return;

    constexpr int GROUP_SIZE_M = 8;
    const int num_pid_in_group = GROUP_SIZE_M * num_pid_n;
    const int group_id = pid / num_pid_in_group;
    const int first_pid_m = group_id * GROUP_SIZE_M;
    const int group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M);
    const int pid_m = first_pid_m + (pid % group_size_m);
    const int pid_n = (pid % num_pid_in_group) / group_size_m;

    const int offs_token_base = pid_m * BLOCK_M;
    const int off_expert = g.expert_ids[pid_m];

    if (off_expert == -1) {
        // Write zeros for missing expert
        for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_N; idx += NUM_THREADS) {
            int lm = idx / BLOCK_N, ln = idx % BLOCK_N;
            int gn = pid_n * BLOCK_N + ln;
            if (gn >= g.N) continue;
            int tid = g.sorted_token_ids[offs_token_base + lm];
            if (tid < g.num_valid_tokens)
                g.C[tid * g.stride_cm + gn * g.stride_cn] = __float2bfloat16(0.0f);
        }
        return;
    }

    constexpr int ELEMS_PER_THREAD = (BLOCK_M * BLOCK_N) / NUM_THREADS;
    float acc[ELEMS_PER_THREAD];
    for (int i = 0; i < ELEMS_PER_THREAD; i++) acc[i] = 0.0f;

    __shared__ bf16 A_sh[BLOCK_M * BLOCK_K];
    __shared__ bf16 B_sh[BLOCK_K * BLOCK_N];

    const int num_k_tiles = (g.K + BLOCK_K - 1) / BLOCK_K;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k_start = kt * BLOCK_K;

        for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_K; idx += NUM_THREADS) {
            int lm = idx / BLOCK_K, lk = idx % BLOCK_K;
            int gk = k_start + lk;
            int tid = g.sorted_token_ids[offs_token_base + lm];
            int orig = tid / g.top_k;
            if (tid < g.num_valid_tokens && gk < g.K)
                A_sh[idx] = g.A[orig * g.stride_am + gk * g.stride_ak];
            else
                A_sh[idx] = __float2bfloat16(0.0f);
        }

        for (int idx = threadIdx.x; idx < BLOCK_K * BLOCK_N; idx += NUM_THREADS) {
            int lk = idx / BLOCK_N, ln = idx % BLOCK_N;
            int gk = k_start + lk, gn = pid_n * BLOCK_N + ln;
            if (gk < g.K && gn < g.N)
                B_sh[idx] = g.B[off_expert * g.stride_be + gk * g.stride_bk + gn * g.stride_bn];
            else
                B_sh[idx] = __float2bfloat16(0.0f);
        }
        __syncthreads();

        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int eidx = threadIdx.x * ELEMS_PER_THREAD + i;
            if (eidx >= BLOCK_M * BLOCK_N) break;
            int lm = eidx / BLOCK_N, ln = eidx % BLOCK_N;
            float sum = 0.0f;
            for (int kk = 0; kk < BLOCK_K; kk++)
                sum += __bfloat162float(A_sh[lm * BLOCK_K + kk]) *
                       __bfloat162float(B_sh[kk * BLOCK_N + ln]);
            acc[i] += sum;
        }
        __syncthreads();
    }

    // Apply routing weights OR GeLU activation
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int eidx = threadIdx.x * ELEMS_PER_THREAD + i;
        if (eidx >= BLOCK_M * BLOCK_N) break;
        int lm = eidx / BLOCK_N, ln = eidx % BLOCK_N;
        int gn = pid_n * BLOCK_N + ln;
        if (gn >= g.N) continue;

        int tid = g.sorted_token_ids[offs_token_base + lm];
        if (tid >= g.num_valid_tokens) continue;

        float result = acc[i];

        if (g.mul_routed_weight) {
            result *= g.topk_weights[tid];
        } else {
            // Apply GeLU when not multiplying by routing weight
            result = gelu_tanh(result);
        }

        g.C[tid * g.stride_cm + gn * g.stride_cn] = __float2bfloat16(result);
    }
}

void dispatch_moe_gelu(moe_gelu_globals& g) {
    hipLaunchKernelGGL(moe_gelu_kernel, g.grid(), g.block(),
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

void moe_gelu_py(
    pybind11::object A,
    pybind11::object B,
    pybind11::object C,
    pybind11::object topk_weights,
    pybind11::object sorted_token_ids,
    pybind11::object expert_ids,
    pybind11::object num_tokens_post_padded,
    int N, int K,
    int num_valid_tokens,
    int top_k,
    int stride_am, int stride_ak,
    int stride_be, int stride_bk, int stride_bn,
    int stride_cm, int stride_cn,
    bool mul_routed_weight
) {
    moe_gelu_globals g;
    g.A = reinterpret_cast<const bf16*>(get_data_ptr(A));
    g.B = reinterpret_cast<const bf16*>(get_data_ptr(B));
    g.C = reinterpret_cast<bf16*>(get_data_ptr(C));
    g.topk_weights = reinterpret_cast<const float*>(get_data_ptr(topk_weights));
    g.sorted_token_ids = reinterpret_cast<const int32_t*>(get_data_ptr(sorted_token_ids));
    g.expert_ids = reinterpret_cast<const int32_t*>(get_data_ptr(expert_ids));
    g.num_tokens_post_padded = reinterpret_cast<const int32_t*>(get_data_ptr(num_tokens_post_padded));
    g.N = N;
    g.K = K;
    g.num_valid_tokens = num_valid_tokens;
    g.top_k = top_k;
    g.stride_am = stride_am;
    g.stride_ak = stride_ak;
    g.stride_be = stride_be;
    g.stride_bk = stride_bk;
    g.stride_bn = stride_bn;
    g.stride_cm = stride_cm;
    g.stride_cn = stride_cn;
    g.mul_routed_weight = mul_routed_weight;
    g.stream = 0;

    dispatch_moe_gelu(g);
}

PYBIND11_MODULE(moe_op_gelu_tk, m) {
    m.def("moe_gelu", &moe_gelu_py,
          "MoE GEMM with GeLU activation (bf16)",
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("C"),
          pybind11::arg("topk_weights"),
          pybind11::arg("sorted_token_ids"),
          pybind11::arg("expert_ids"),
          pybind11::arg("num_tokens_post_padded"),
          pybind11::arg("N"),
          pybind11::arg("K"),
          pybind11::arg("num_valid_tokens"),
          pybind11::arg("top_k"),
          pybind11::arg("stride_am"),
          pybind11::arg("stride_ak"),
          pybind11::arg("stride_be"),
          pybind11::arg("stride_bk"),
          pybind11::arg("stride_bn"),
          pybind11::arg("stride_cm"),
          pybind11::arg("stride_cn"),
          pybind11::arg("mul_routed_weight"));
}
