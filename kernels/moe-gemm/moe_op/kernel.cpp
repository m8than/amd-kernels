// MoE GEMM Core Kernel (moe_op)
// Ported from reference/triton/moe_op.py
//
// Fused MoE computation: for each expert e, computes C[e] = A_tokens @ B[e]
// where tokens are routed to experts via sorted_token_ids and expert_ids.
//
// Key data flow:
//   - A: input tokens [num_tokens, K] (bf16)
//   - B: expert weights [E, K, N] (bf16, stored as K x N per expert)
//   - C: output [num_tokens_padded, N] (bf16)
//   - sorted_token_ids: maps block rows -> original token indices
//   - expert_ids: maps each M-block -> expert index
//   - topk_weights: optional per-token routing weights
//   - num_tokens_post_padded: total padded token count
//
// The kernel processes tiles of [BLOCK_M, BLOCK_N] output by iterating
// over the K dimension in BLOCK_K steps, accumulating in fp32.

#include "kittens.cuh"
using namespace kittens;

// Tile sizes
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 32;

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

// Globals descriptor for the MoE GEMM
struct moe_gemm_globals {
    // Tensor pointers (raw pointers for flexible indexing with MoE routing)
    const bf16* __restrict__ A;        // [num_tokens, K]
    const bf16* __restrict__ B;        // [E, K, N]
    bf16* __restrict__ C;              // [num_tokens_padded, N]
    const float* __restrict__ topk_weights;  // [num_tokens_padded]
    const int32_t* __restrict__ sorted_token_ids;  // [num_tokens_padded]
    const int32_t* __restrict__ expert_ids;         // [num_m_blocks]
    const int32_t* __restrict__ num_tokens_post_padded;  // scalar

    int N;
    int K;
    int num_valid_tokens;
    int top_k;

    // Strides (in elements)
    int stride_am;   // A row stride
    int stride_ak;   // A col stride (usually 1)
    int stride_be;   // B expert stride
    int stride_bk;   // B K-dim stride
    int stride_bn;   // B N-dim stride
    int stride_cm;   // C row stride
    int stride_cn;   // C col stride (usually 1)

    bool mul_routed_weight;

    hipStream_t stream;

    dim3 grid() {
        // Upper bound: launched tiles = num_m_blocks * num_n_blocks
        // Actual bounds checked inside kernel via num_tokens_post_padded
        int num_tokens_padded_est = num_valid_tokens * 2;  // overestimate
        int num_pid_m = (num_tokens_padded_est + BLOCK_M - 1) / BLOCK_M;
        int num_pid_n = (N + BLOCK_N - 1) / BLOCK_N;
        return dim3(num_pid_m * num_pid_n);
    }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

// MoE GEMM kernel
// Each thread block computes one [BLOCK_M, BLOCK_N] output tile.
// The M dimension indexes into sorted tokens (with expert routing).
__global__ __launch_bounds__(NUM_THREADS)
void moe_gemm_kernel(const moe_gemm_globals g) {
    const int pid = blockIdx.x;

    // Load num_tokens_post_padded (runtime constant)
    const int num_tokens_post_padded = g.num_tokens_post_padded[0];

    const int num_pid_m = (num_tokens_post_padded + BLOCK_M - 1) / BLOCK_M;
    const int num_pid_n = (g.N + BLOCK_N - 1) / BLOCK_N;
    const int total_tiles = num_pid_m * num_pid_n;

    if (pid >= total_tiles) return;

    // Grouped ordering for L2 reuse
    constexpr int GROUP_SIZE_M = 8;
    const int num_pid_in_group = GROUP_SIZE_M * num_pid_n;
    const int group_id = pid / num_pid_in_group;
    const int first_pid_m = group_id * GROUP_SIZE_M;
    const int group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M);
    const int pid_m = first_pid_m + (pid % group_size_m);
    const int pid_n = (pid % num_pid_in_group) / group_size_m;

    const int warp_id = kittens::warpid();
    const int lane_id = kittens::laneid();

    // Load token indices for this M-block
    const int offs_token_base = pid_m * BLOCK_M;

    // Load expert ID for this M-block
    const int off_expert = g.expert_ids[pid_m];

    // If expert_id == -1, write zeros (expert not in this rank)
    if (off_expert == -1) {
        // Cooperatively zero out the output block
        const int elems_per_thread = (BLOCK_M * BLOCK_N + NUM_THREADS - 1) / NUM_THREADS;
        for (int i = 0; i < elems_per_thread; i++) {
            int idx = threadIdx.x + i * NUM_THREADS;
            if (idx >= BLOCK_M * BLOCK_N) break;
            int local_m = idx / BLOCK_N;
            int local_n = idx % BLOCK_N;
            int global_n = pid_n * BLOCK_N + local_n;
            if (global_n >= g.N) continue;

            int token_id = g.sorted_token_ids[offs_token_base + local_m];
            if (token_id < g.num_valid_tokens) {
                g.C[token_id * g.stride_cm + global_n * g.stride_cn] = __float2bfloat16(0.0f);
            }
        }
        return;
    }

    // Accumulator in fp32 (each thread holds a portion)
    // We use a simple thread-level accumulation approach
    // Each thread processes multiple (m, n) output elements
    constexpr int ELEMS_PER_THREAD = (BLOCK_M * BLOCK_N) / NUM_THREADS;

    float acc[ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        acc[i] = 0.0f;
    }

    // Shared memory for cooperative loading of A and B tiles
    __shared__ bf16 A_shared[BLOCK_M * BLOCK_K];
    __shared__ bf16 B_shared[BLOCK_K * BLOCK_N];

    const int num_k_tiles = (g.K + BLOCK_K - 1) / BLOCK_K;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int k_start = k_tile * BLOCK_K;

        // Cooperative load of A tile [BLOCK_M, BLOCK_K]
        // A is indexed by sorted_token_ids with top_k division
        {
            const int total_elems = BLOCK_M * BLOCK_K;
            for (int idx = threadIdx.x; idx < total_elems; idx += NUM_THREADS) {
                int local_m = idx / BLOCK_K;
                int local_k = idx % BLOCK_K;
                int global_k = k_start + local_k;

                int token_id = g.sorted_token_ids[offs_token_base + local_m];
                int orig_token = token_id / g.top_k;  // Map back to original token

                if (token_id < g.num_valid_tokens && global_k < g.K) {
                    A_shared[idx] = g.A[orig_token * g.stride_am + global_k * g.stride_ak];
                } else {
                    A_shared[idx] = __float2bfloat16(0.0f);
                }
            }
        }

        // Cooperative load of B tile [BLOCK_K, BLOCK_N]
        // B is indexed by expert_id
        {
            const int total_elems = BLOCK_K * BLOCK_N;
            for (int idx = threadIdx.x; idx < total_elems; idx += NUM_THREADS) {
                int local_k = idx / BLOCK_N;
                int local_n = idx % BLOCK_N;
                int global_k = k_start + local_k;
                int global_n = pid_n * BLOCK_N + local_n;

                if (global_k < g.K && global_n < g.N) {
                    B_shared[idx] = g.B[off_expert * g.stride_be +
                                        global_k * g.stride_bk +
                                        global_n * g.stride_bn];
                } else {
                    B_shared[idx] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();

        // Compute partial matmul: acc += A_shared @ B_shared
        // Each thread computes its assigned output elements
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int elem_idx = threadIdx.x * ELEMS_PER_THREAD + i;
            if (elem_idx >= BLOCK_M * BLOCK_N) break;
            int local_m = elem_idx / BLOCK_N;
            int local_n = elem_idx % BLOCK_N;

            float sum = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < BLOCK_K; kk++) {
                float a_val = __bfloat162float(A_shared[local_m * BLOCK_K + kk]);
                float b_val = __bfloat162float(B_shared[kk * BLOCK_N + local_n]);
                sum += a_val * b_val;
            }
            acc[i] += sum;
        }

        __syncthreads();
    }

    // Apply routing weights if needed
    // Store results
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int elem_idx = threadIdx.x * ELEMS_PER_THREAD + i;
        if (elem_idx >= BLOCK_M * BLOCK_N) break;
        int local_m = elem_idx / BLOCK_N;
        int local_n = elem_idx % BLOCK_N;
        int global_n = pid_n * BLOCK_N + local_n;

        if (global_n >= g.N) continue;

        int token_id = g.sorted_token_ids[offs_token_base + local_m];
        if (token_id >= g.num_valid_tokens) continue;

        float result = acc[i];

        // Multiply by routing weight
        if (g.mul_routed_weight) {
            float weight = g.topk_weights[token_id];
            result *= weight;
        }

        // Convert and store
        g.C[token_id * g.stride_cm + global_n * g.stride_cn] = __float2bfloat16(result);
    }
}

// Dispatch function
void dispatch_moe_gemm(moe_gemm_globals& g) {
    hipLaunchKernelGGL(
        moe_gemm_kernel,
        g.grid(),
        g.block(),
        g.dynamic_shared_memory(),
        g.stream,
        g
    );
}

// Template instantiation not needed (no template params)

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

void moe_gemm_py(
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
    moe_gemm_globals g;
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

    dispatch_moe_gemm(g);
}

PYBIND11_MODULE(moe_op_tk, m) {
    m.def("moe_gemm", &moe_gemm_py,
          "MoE GEMM kernel (bf16)",
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
