// MoE GEMM with Fused SiLU Activation (moe_op_silu_fused)
// Ported from reference/triton/moe_op_silu_fused.py
//
// Fused MoE GEMM with SiLU-and-mul (SwiGLU-style gated activation):
//   1. Compute A @ B[expert] -> [M, N] where N = 2*output_dim
//   2. Split result into gate (first half) and up (second half)
//   3. Apply SiLU to gate: activated = SiLU(gate) * up
//   4. Write activated [M, N/2] to output
//
// The weight matrix B has interleaved gate/up columns for efficient access:
//   B columns: [gate[0], up[0], gate[1], up[1], ...]
// This allows SiLU-and-mul without a separate permute step.

#include "kittens.cuh"
using namespace kittens;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;  // Full N (gate+up interleaved)
constexpr int BLOCK_K = 32;

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

struct moe_silu_globals {
    const bf16* __restrict__ A;
    const bf16* __restrict__ B;
    bf16* __restrict__ C;
    const float* __restrict__ topk_weights;
    const int32_t* __restrict__ sorted_token_ids;
    const int32_t* __restrict__ expert_ids;
    const int32_t* __restrict__ num_tokens_post_padded;

    int N;  // Full N (= 2 * output_dim)
    int K;
    int num_valid_tokens;
    int top_k;

    int stride_am, stride_ak;
    int stride_be, stride_bk, stride_bn;
    int stride_cm, stride_cn;

    bool mul_routed_weight;

    hipStream_t stream;

    dim3 grid() {
        int est = num_valid_tokens * 2;
        return dim3(((est + BLOCK_M - 1) / BLOCK_M) * ((N + BLOCK_N - 1) / BLOCK_N));
    }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

// SiLU using exp2 approximation (matches Triton _silu_exp2)
__device__ __forceinline__ float silu_exp2(float x) {
    return x / (1.0f + exp2f(-1.44269504089f * x));
}

__global__ __launch_bounds__(NUM_THREADS)
void moe_silu_kernel(const moe_silu_globals g) {
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
        // Write zeros for output (N/2 columns)
        const int BLOCK_N_HALF = BLOCK_N / 2;
        for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_N_HALF; idx += NUM_THREADS) {
            int lm = idx / BLOCK_N_HALF, ln = idx % BLOCK_N_HALF;
            int gn = pid_n * BLOCK_N_HALF + ln;
            if (gn >= g.N / 2) continue;
            int tid = g.sorted_token_ids[offs_token_base + lm];
            if (tid < g.num_valid_tokens)
                g.C[tid * g.stride_cm + gn * g.stride_cn] = __float2bfloat16(0.0f);
        }
        return;
    }

    const int N_HALF = g.N / 2;
    constexpr int BLOCK_N_HALF = BLOCK_N / 2;

    constexpr int ELEMS_PER_THREAD = (BLOCK_M * BLOCK_N) / NUM_THREADS;
    float acc[ELEMS_PER_THREAD];
    for (int i = 0; i < ELEMS_PER_THREAD; i++) acc[i] = 0.0f;

    __shared__ bf16 A_sh[BLOCK_M * BLOCK_K];
    __shared__ bf16 B_sh[BLOCK_K * BLOCK_N];

    const int num_k_tiles = (g.K + BLOCK_K - 1) / BLOCK_K;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k_start = kt * BLOCK_K;

        // Load A tile
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

        // Load B tile with interleaved gate/up indexing
        for (int idx = threadIdx.x; idx < BLOCK_K * BLOCK_N; idx += NUM_THREADS) {
            int lk = idx / BLOCK_N, ln = idx % BLOCK_N;
            int gk = k_start + lk;

            // Interleaved indexing: even->gate[i], odd->up[i]
            int i_floor = ln / 2;
            int half_col = (pid_n * BLOCK_N_HALF + i_floor) % N_HALF;
            int b_col = (half_col + (ln % 2) * N_HALF) % g.N;

            if (gk < g.K && b_col < g.N)
                B_sh[idx] = g.B[off_expert * g.stride_be + gk * g.stride_bk + b_col * g.stride_bn];
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

    // Apply routing weights first (if applicable)
    if (g.mul_routed_weight) {
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int eidx = threadIdx.x * ELEMS_PER_THREAD + i;
            if (eidx >= BLOCK_M * BLOCK_N) break;
            int lm = eidx / BLOCK_N;
            int tid = g.sorted_token_ids[offs_token_base + lm];
            if (tid < g.num_valid_tokens) {
                acc[i] *= g.topk_weights[tid];
            }
        }
    }

    // Apply SiLU-and-mul: interleaved [gate0, up0, gate1, up1, ...]
    // For each pair (even, odd): output = silu(gate) * up
    for (int i = 0; i < ELEMS_PER_THREAD; i += 2) {
        int eidx = threadIdx.x * ELEMS_PER_THREAD + i;
        if (eidx >= BLOCK_M * BLOCK_N) break;
        int lm = eidx / BLOCK_N;
        int ln = eidx % BLOCK_N;

        if (ln % 2 != 0) continue;  // Only process pairs starting at even

        float gate_val = acc[i];
        float up_val = (i + 1 < ELEMS_PER_THREAD) ? acc[i + 1] : 0.0f;
        float activated = silu_exp2(gate_val) * up_val;

        int gn = pid_n * BLOCK_N_HALF + ln / 2;
        if (gn >= N_HALF) continue;

        int tid = g.sorted_token_ids[offs_token_base + lm];
        if (tid >= g.num_valid_tokens) continue;

        g.C[tid * g.stride_cm + gn * g.stride_cn] = __float2bfloat16(activated);
    }
}

void dispatch_moe_silu(moe_silu_globals& g) {
    hipLaunchKernelGGL(moe_silu_kernel, g.grid(), g.block(),
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

void moe_silu_py(
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
    moe_silu_globals g;
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

    dispatch_moe_silu(g);
}

PYBIND11_MODULE(moe_op_silu_fused_tk, m) {
    m.def("moe_silu_fused", &moe_silu_py,
          "MoE GEMM with fused SiLU activation (bf16, SwiGLU)",
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
