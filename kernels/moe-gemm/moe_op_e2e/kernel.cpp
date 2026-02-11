// MoE End-to-End Kernel (moe_op_e2e)
// Ported from reference/triton/moe_op_e2e.py
//
// End-to-end MoE: performs both layers of a gated MLP expert:
//   intermediate = SiLU(A @ W1_gate) * (A @ W1_up)   [gated MLP layer 1]
//   output = intermediate @ W2                         [layer 2]
//
// Key data flow:
//   - A: input tokens [num_tokens, K] (bf16)
//   - W1: expert gate+up weights [E, N, K] where N = 2*hidden (gate || up)
//   - W2: expert down weights [E, K, N//2]
//   - C: output [num_tokens_padded, K] (bf16)
//
// The W1 output is interleaved: even columns from gate, odd from up-projection,
// enabling fused SiLU-and-mul without a permute step.

#include "kittens.cuh"
using namespace kittens;

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 128;   // W1 output (gate+up interleaved)
constexpr int BLOCK_K1 = 32;   // K-loop for layer 1
constexpr int BLOCK_K2 = 32;   // K-loop for layer 2

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

struct moe_e2e_globals {
    const bf16* __restrict__ A;
    const bf16* __restrict__ W1;
    const bf16* __restrict__ W2;
    bf16* __restrict__ C;
    const float* __restrict__ topk_weights;
    const int32_t* __restrict__ sorted_token_ids;
    const int32_t* __restrict__ expert_ids;
    const int32_t* __restrict__ num_tokens_post_padded;

    int N;  // W1 output dim (2 * hidden_dim)
    int K;  // input/output dim

    int num_valid_tokens;
    int top_k;

    int stride_am, stride_ak;
    int stride_w1e, stride_w1n, stride_w1k;
    int stride_w2e, stride_w2n, stride_w2k;
    int stride_cm;

    bool mul_routed_weight;

    hipStream_t stream;

    dim3 grid() {
        int num_tokens_padded_est = num_valid_tokens * 2;
        int num_pid_m = (num_tokens_padded_est + BLOCK_M - 1) / BLOCK_M;
        int num_pid_n = (N + BLOCK_N - 1) / BLOCK_N;
        return dim3(num_pid_m * num_pid_n);
    }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
// Using exp2 approximation: silu(x) = x / (1 + exp2(-1.44269504089 * x))
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + exp2f(-1.44269504089f * x));
}

__global__ __launch_bounds__(NUM_THREADS)
void moe_e2e_kernel(const moe_e2e_globals g) {
    const int pid = blockIdx.x;
    const int num_tokens_post_padded = g.num_tokens_post_padded[0];

    const int num_pid_m = (num_tokens_post_padded + BLOCK_M - 1) / BLOCK_M;
    const int num_pid_n = (g.N + BLOCK_N - 1) / BLOCK_N;
    const int total_tiles = num_pid_m * num_pid_n;
    if (pid >= total_tiles) return;

    const int pid_m = pid / num_pid_n;
    const int pid_n = pid % num_pid_n;

    if (pid_m * BLOCK_M >= num_tokens_post_padded) return;

    const int off_expert = g.expert_ids[pid_m];
    if (off_expert == -1) return;

    const int offs_token_base = pid_m * BLOCK_M;
    const int N_HALF = g.N / 2;
    const int BLOCK_N_HALF = BLOCK_N / 2;

    // === Layer 1: intermediate = SiLU(A @ W1_gate) * (A @ W1_up) ===

    __shared__ bf16 A_sh[BLOCK_M * BLOCK_K1];
    __shared__ bf16 W1_sh[BLOCK_K1 * BLOCK_N];

    // Accumulator for W1 output (interleaved gate/up)
    // Each thread handles a subset of elements
    constexpr int L1_ELEMS = (BLOCK_M * BLOCK_N + NUM_THREADS - 1) / NUM_THREADS;
    float l1_acc[L1_ELEMS];
    for (int i = 0; i < L1_ELEMS; i++) l1_acc[i] = 0.0f;

    const int num_k1_tiles = (g.K + BLOCK_K1 - 1) / BLOCK_K1;

    for (int kt = 0; kt < num_k1_tiles; kt++) {
        int k_start = kt * BLOCK_K1;

        // Load A tile
        for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_K1; idx += NUM_THREADS) {
            int lm = idx / BLOCK_K1, lk = idx % BLOCK_K1;
            int gk = k_start + lk;
            int token_id = g.sorted_token_ids[offs_token_base + lm];
            int orig = token_id / g.top_k;
            if (token_id < g.num_valid_tokens && gk < g.K)
                A_sh[idx] = g.A[orig * g.stride_am + gk * g.stride_ak];
            else
                A_sh[idx] = __float2bfloat16(0.0f);
        }

        // Load W1 tile with interleaved gate/up indexing
        // W1 layout: [E, N, K] where N = gate_dim || up_dim
        // Interleave: even cols -> gate[i], odd cols -> up[i]
        for (int idx = threadIdx.x; idx < BLOCK_K1 * BLOCK_N; idx += NUM_THREADS) {
            int lk = idx / BLOCK_N, ln = idx % BLOCK_N;
            int gk = k_start + lk;

            // Compute interleaved W1 column index
            int i_floor = ln / 2;
            int half_col = (pid_n * BLOCK_N_HALF + i_floor) % N_HALF;
            int w1_col = (half_col + (ln % 2) * N_HALF) % g.N;

            if (gk < g.K && w1_col < g.N)
                W1_sh[idx] = g.W1[off_expert * g.stride_w1e +
                                   gk * g.stride_w1k +
                                   w1_col * g.stride_w1n];
            else
                W1_sh[idx] = __float2bfloat16(0.0f);
        }
        __syncthreads();

        for (int i = 0; i < L1_ELEMS; i++) {
            int eidx = threadIdx.x * L1_ELEMS + i;
            if (eidx >= BLOCK_M * BLOCK_N) break;
            int lm = eidx / BLOCK_N, ln = eidx % BLOCK_N;
            float sum = 0.0f;
            for (int kk = 0; kk < BLOCK_K1 && (k_start + kk) < g.K; kk++)
                sum += __bfloat162float(A_sh[lm * BLOCK_K1 + kk]) *
                       __bfloat162float(W1_sh[kk * BLOCK_N + ln]);
            l1_acc[i] += sum;
        }
        __syncthreads();
    }

    // Apply SiLU-and-mul: split interleaved into gate/up, apply SiLU to gate, multiply
    // The interleaving is: [gate0, up0, gate1, up1, ...]
    // After reshape: silu_acc = l1_acc[::2], mul_acc = l1_acc[1::2]
    __shared__ bf16 intermediate[BLOCK_M * BLOCK_N_HALF];

    for (int i = 0; i < L1_ELEMS; i++) {
        int eidx = threadIdx.x * L1_ELEMS + i;
        if (eidx >= BLOCK_M * BLOCK_N) break;
        int lm = eidx / BLOCK_N;
        int ln = eidx % BLOCK_N;

        // Only process even-indexed columns (gate values)
        if (ln % 2 == 0) {
            int pair_idx = eidx + 1;  // corresponding up value
            float gate_val = l1_acc[i];
            float up_val = (i + 1 < L1_ELEMS && pair_idx < BLOCK_M * BLOCK_N)
                           ? l1_acc[i + 1] : 0.0f;
            float activated = silu(gate_val) * up_val;
            int half_n = ln / 2;
            intermediate[lm * BLOCK_N_HALF + half_n] = __float2bfloat16(activated);
        }
    }
    __syncthreads();

    // === Layer 2: output = intermediate @ W2, with atomic add across N-tiles ===

    __shared__ bf16 W2_sh[BLOCK_N_HALF * BLOCK_K2];

    const int num_k2_tiles = (g.K + BLOCK_K2 - 1) / BLOCK_K2;

    for (int kt = 0; kt < num_k2_tiles; kt++) {
        int k_start = kt * BLOCK_K2;

        constexpr int L2_ELEMS = (BLOCK_M * BLOCK_K2 + NUM_THREADS - 1) / NUM_THREADS;
        float l2_acc[L2_ELEMS];
        for (int i = 0; i < L2_ELEMS; i++) l2_acc[i] = 0.0f;

        // Load W2 tile [BLOCK_N_HALF, BLOCK_K2]
        int w2n_base = pid_n * BLOCK_N_HALF;
        for (int idx = threadIdx.x; idx < BLOCK_N_HALF * BLOCK_K2; idx += NUM_THREADS) {
            int ln = idx / BLOCK_K2, lk = idx % BLOCK_K2;
            int gn = w2n_base + ln;
            int gk = k_start + lk;
            if (gn < g.N / 2 && gk < g.K)
                W2_sh[idx] = g.W2[off_expert * g.stride_w2e +
                                   gn * g.stride_w2n +
                                   gk * g.stride_w2k];
            else
                W2_sh[idx] = __float2bfloat16(0.0f);
        }
        __syncthreads();

        // Compute intermediate @ W2 slice
        for (int i = 0; i < L2_ELEMS; i++) {
            int eidx = threadIdx.x * L2_ELEMS + i;
            if (eidx >= BLOCK_M * BLOCK_K2) break;
            int lm = eidx / BLOCK_K2, lk = eidx % BLOCK_K2;
            int gk = k_start + lk;
            if (gk >= g.K) continue;

            float sum = 0.0f;
            for (int nn = 0; nn < BLOCK_N_HALF; nn++)
                sum += __bfloat162float(intermediate[lm * BLOCK_N_HALF + nn]) *
                       __bfloat162float(W2_sh[nn * BLOCK_K2 + lk]);

            // Apply routing weight
            int token_id = g.sorted_token_ids[offs_token_base + lm];
            if (token_id >= g.num_valid_tokens) continue;

            if (g.mul_routed_weight) {
                float weight = g.topk_weights[offs_token_base + lm];
                sum *= weight;
            }

            // Atomic add to output (multiple N-tiles contribute)
            atomicAdd(reinterpret_cast<float*>(&g.C[token_id * g.stride_cm + gk]),
                      sum);
        }
        __syncthreads();
    }
}

void dispatch_moe_e2e(moe_e2e_globals& g) {
    hipLaunchKernelGGL(moe_e2e_kernel, g.grid(), g.block(),
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

void moe_e2e_py(
    pybind11::object A,
    pybind11::object W1,
    pybind11::object W2,
    pybind11::object C,
    pybind11::object topk_weights,
    pybind11::object sorted_token_ids,
    pybind11::object expert_ids,
    pybind11::object num_tokens_post_padded,
    int N, int K,
    int num_valid_tokens,
    int top_k,
    int stride_am, int stride_ak,
    int stride_w1e, int stride_w1n, int stride_w1k,
    int stride_w2e, int stride_w2n, int stride_w2k,
    int stride_cm,
    bool mul_routed_weight
) {
    moe_e2e_globals g;
    g.A = reinterpret_cast<const bf16*>(get_data_ptr(A));
    g.W1 = reinterpret_cast<const bf16*>(get_data_ptr(W1));
    g.W2 = reinterpret_cast<const bf16*>(get_data_ptr(W2));
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
    g.stride_w1e = stride_w1e;
    g.stride_w1n = stride_w1n;
    g.stride_w1k = stride_w1k;
    g.stride_w2e = stride_w2e;
    g.stride_w2n = stride_w2n;
    g.stride_w2k = stride_w2k;
    g.stride_cm = stride_cm;
    g.mul_routed_weight = mul_routed_weight;
    g.stream = 0;

    dispatch_moe_e2e(g);
}

PYBIND11_MODULE(moe_op_e2e_tk, m) {
    m.def("moe_e2e", &moe_e2e_py,
          "MoE end-to-end kernel (gated MLP with SiLU)",
          pybind11::arg("A"),
          pybind11::arg("W1"),
          pybind11::arg("W2"),
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
          pybind11::arg("stride_w1e"),
          pybind11::arg("stride_w1n"),
          pybind11::arg("stride_w1k"),
          pybind11::arg("stride_w2e"),
          pybind11::arg("stride_w2n"),
          pybind11::arg("stride_w2k"),
          pybind11::arg("stride_cm"),
          pybind11::arg("mul_routed_weight"));
}
