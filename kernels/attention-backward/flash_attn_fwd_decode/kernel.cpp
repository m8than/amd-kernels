/**
 * Flash Attention Forward Decode Kernel (HipKittens Port)
 *
 * Implements split-K flash attention for the decode case where seqlen_q is
 * small (typically 1) while seqlen_k can be large.
 *
 * Algorithm (Split-K approach):
 * 1. Split the K/V sequence across multiple workgroups (num_splits)
 * 2. Each workgroup computes a partial attention result for its K/V chunk:
 *    - Load Q (small, e.g., 1-64 tokens)
 *    - Iterate K/V blocks in assigned range
 *    - Standard online softmax within the chunk
 *    - Store partial: O_partial, m_partial, l_partial
 * 3. Reduce kernel: combine partial results across splits
 *    - For each split: rescale using max correction
 *    - Final O = sum(O_partial * alpha) / sum(l_partial * alpha)
 *
 * Two kernels:
 *   splitk_attn_fwd_decode_kernel  -- per-split partial attention
 *   splitk_reduce_kernel           -- cross-split reduction
 *
 * Reference: reference/triton/flash_attn_triton_amd/fwd_decode.py
 */

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

// ============================================================================
// Configuration
// ============================================================================

constexpr int BLOCK_M  = 64;   // Q block size along sequence dim
constexpr int BLOCK_N  = 64;   // K/V block size along sequence dim
constexpr int ATTN_D   = 128;  // head dimension

// Thread configuration for the splitk kernel
// 4 warps = 256 threads
#define SPLITK_NUM_WARPS 4
#define SPLITK_NUM_THREADS (kittens::WARP_THREADS * SPLITK_NUM_WARPS)

// Thread configuration for the reduce kernel
#define REDUCE_NUM_WARPS 4
#define REDUCE_NUM_THREADS (kittens::WARP_THREADS * REDUCE_NUM_WARPS)

// LOG2E for computing softmax in log2 space (use exp2 instead of exp)
constexpr float LOG2E = 1.44269504089f;
constexpr float INV_LOG2E = 0.69314718056f;  // ln(2)

// ============================================================================
// Global Memory Layout Types
// ============================================================================

// Q, K, V, O: bf16 [B, seqlen, H, D]
using bf16_gl = gl<bf16, -1, -1, -1, -1>;
// Partial O, metadata: float
using f32_gl  = gl<float, -1, -1, -1, -1>;

// ============================================================================
// Tile type aliases
// ============================================================================

// Helper: reinterpret a register tile's layout tag without changing data.
// This is valid because row_l and col_l rt_base tiles with the same shape
// have identical packed_per_thread and data array sizes on CDNA3.
template<typename T, int R, int C, ducks::rt_layout::all from_layout,
         ducks::rt_layout::all to_layout = col_l,
         ducks::rt_shape::all shape = ducks::rt_shape::rt_16x16>
__device__ inline rt<T, R, C, to_layout, shape>&
as_layout(rt<T, R, C, from_layout, shape>& t) {
    return reinterpret_cast<rt<T, R, C, to_layout, shape>&>(t);
}

// Register tiles for Q, O accumulator, K, V, and attention scores
using q_tile_t   = rt_fl<BLOCK_M, ATTN_D>;   // [BLOCK_M, D] float
using q_bf_t     = rt_bf<BLOCK_M, ATTN_D>;    // [BLOCK_M, D] bf16
using kv_tile_t  = rt_bf<BLOCK_N, ATTN_D>;    // [BLOCK_N, D] bf16
using acc_tile_t = rt_fl<BLOCK_M, ATTN_D>;    // [BLOCK_M, D] float
using qk_tile_t  = rt_fl<BLOCK_M, BLOCK_N>;   // [BLOCK_M, BLOCK_N] float
using p_bf_t     = rt_bf<BLOCK_M, BLOCK_N>;    // [BLOCK_M, BLOCK_N] bf16

// Col vector: one element per row (length BLOCK_M) -- for per-query-position stats
using col_vec_t  = typename q_tile_t::col_vec;

// ============================================================================
// SplitK Kernel Globals
// ============================================================================

struct splitk_globals {
    bf16_gl Q;              // [B, seqlen_q, H_q, D]
    bf16_gl K;              // [B, seqlen_k, H_kv, D]
    bf16_gl V;              // [B, seqlen_k, H_kv, D]
    f32_gl  O_partial;      // [B*H_q, num_splits, M_ceil, D]
    f32_gl  Metadata;       // [B*H_q, 2, num_splits, M_ceil]
    int num_heads_q;
    int num_heads_kv;
    int seqlen_q;
    int seqlen_k;
    int num_splits;
    float sm_scale;

    hipStream_t stream;

    dim3 grid() {
        int batch = Q.batch();
        int grid_m = (seqlen_q + BLOCK_M - 1) / BLOCK_M;
        return dim3(grid_m, batch * num_heads_q, num_splits);
    }
    dim3 block() { return dim3(SPLITK_NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

// ============================================================================
// Reduce Kernel Globals
// ============================================================================

struct reduce_globals {
    f32_gl  O_partial;      // [B*H_q, num_splits, M_ceil, D]
    f32_gl  Metadata;       // [B*H_q, 2, num_splits, M_ceil]
    bf16_gl O;              // [B, seqlen_q, H_q, D]
    f32_gl  LSE;            // [B*H_q, seqlen_q, 1, 1]
    int num_heads_q;
    int seqlen_q;
    int num_splits;

    hipStream_t stream;

    dim3 grid() {
        int batch = O.batch();
        return dim3((seqlen_q + BLOCK_M - 1) / BLOCK_M, batch * num_heads_q, 1);
    }
    dim3 block() { return dim3(REDUCE_NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

// ============================================================================
// SplitK Attention Forward Decode Kernel
// ============================================================================

/**
 * Each workgroup processes a chunk of the K/V sequence for a given
 * (batch, head, m_block) triple. It performs online softmax attention
 * within its assigned [lo, hi) range and writes partial results.
 *
 * Grid: (cdiv(seqlen_q, BLOCK_M), B * H_q, num_splits)
 */
__launch_bounds__(SPLITK_NUM_THREADS, 1)
__global__ void splitk_attn_fwd_decode_kernel(const splitk_globals g) {

    const int pid_m      = blockIdx.x;
    const int pid_bh     = blockIdx.y;
    const int pid_splitk = blockIdx.z;

    const int batch_idx   = pid_bh / g.num_heads_q;
    const int head_q_idx  = pid_bh % g.num_heads_q;
    const int group_size  = g.num_heads_q / g.num_heads_kv;
    const int head_kv_idx = head_q_idx / group_size;

    // K/V range for this split
    const int split_size = (g.seqlen_k + g.num_splits - 1) / g.num_splits;
    const int lo = pid_splitk * split_size;
    const int hi = min((pid_splitk + 1) * split_size, g.seqlen_k);

    const int m_start = pid_m * BLOCK_M;
    const int tid = threadIdx.x;

    // Handle empty splits: write sentinel values
    if (lo >= hi) {
        for (int m = tid; m < BLOCK_M; m += SPLITK_NUM_THREADS) {
            int m_idx = m_start + m;
            if (m_idx < g.seqlen_q) {
                g.Metadata[{pid_bh, 0, pid_splitk, m_idx}] = -INFINITY;
                g.Metadata[{pid_bh, 1, pid_splitk, m_idx}] = 0.0f;
            }
        }
        for (int idx = tid; idx < BLOCK_M * ATTN_D; idx += SPLITK_NUM_THREADS) {
            int m = idx / ATTN_D;
            int d = idx % ATTN_D;
            int m_idx = m_start + m;
            if (m_idx < g.seqlen_q) {
                g.O_partial[{pid_bh, pid_splitk, m_idx, d}] = 0.0f;
            }
        }
        return;
    }

    // Scale by log2(e) so we can use exp2 throughout
    const float qk_scale = g.sm_scale * LOG2E;

    // ---- Load Q into registers and prescale ----
    // Q: [B, seqlen_q, H_q, D]
    // load<1> strides tile rows along gl dim 1 (seqlen_q)
    q_tile_t q_reg;
    load<1>(q_reg, g.Q, {batch_idx, m_start, head_q_idx, 0});
    mul(q_reg, q_reg, qk_scale);

    // Convert to bf16 for MMA
    q_bf_t q_bf;
    copy(q_bf, q_reg);

    // ---- Initialize online softmax accumulators ----
    acc_tile_t acc;
    zero(acc);

    // m_i: per-row running max (in log2 space), init to -inf
    // l_i: per-row running sum of exp, init to 0
    col_vec_t m_i, l_i;
    #pragma unroll
    for (int o = 0; o < m_i.outer_dim; ++o) {
        #pragma unroll
        for (int i = 0; i < m_i.inner_dim; ++i) {
            m_i.data[o][i] = -INFINITY;
        }
    }
    zero(l_i);

    // ---- Main loop: iterate K/V blocks in [lo, hi) ----
    for (int start_n = lo; start_n < hi; start_n += BLOCK_N) {

        // Load K block: [BLOCK_N, D]
        // K: [B, seqlen_k, H_kv, D], load<1> strides along seqlen_k
        kv_tile_t k_reg;
        load<1>(k_reg, g.K, {batch_idx, start_n, head_kv_idx, 0});

        // Compute QK^T: [BLOCK_M, BLOCK_N] = Q_bf @ K_bf^T
        // mma_ABt requires D,C = col_l; use as_layout to reinterpret
        qk_tile_t qk;
        zero(qk);
        mma_ABt(as_layout(qk), q_bf, k_reg, as_layout(qk));

        // ---- Online softmax update ----
        // Compute per-row max of current QK block
        col_vec_t qk_row_max;
        row_max(qk_row_max, qk);

        // New running max: m_i_new = max(m_i, qk_row_max)
        col_vec_t m_i_new;
        max(m_i_new, m_i, qk_row_max);

        // Correction factor: alpha = exp2(m_i_old - m_i_new)
        col_vec_t alpha;
        sub(alpha, m_i, m_i_new);
        exp2(alpha, alpha);

        // Subtract new max from QK scores (per row)
        sub_col(qk, qk, m_i_new);

        // P = exp2(QK - max)
        exp2(qk, qk);

        // Update running sum: l_i = l_i * alpha + sum(P, axis=1)
        mul(l_i, l_i, alpha);
        col_vec_t p_row_sum;
        row_sum(p_row_sum, qk);
        add(l_i, l_i, p_row_sum);

        // Update running max
        copy(m_i, m_i_new);

        // Rescale accumulator with correction factor
        mul_col(acc, acc, alpha);

        // Convert P to bf16 for MMA
        p_bf_t p_bf;
        copy(p_bf, qk);

        // Load V block: [BLOCK_N, D]
        kv_tile_t v_reg;
        load<1>(v_reg, g.V, {batch_idx, start_n, head_kv_idx, 0});

        // Accumulate: acc += P @ V
        // mma_AB requires D,C = col_l, B = col_l; reinterpret layouts
        mma_AB(as_layout(acc), p_bf, as_layout(v_reg), as_layout(acc));
    }

    // ---- Store partial results ----
    // O_partial: [B*H_q, num_splits, M_ceil, D]
    // store<2> strides tile rows along gl dim 2 (M_ceil)
    store<2>(g.O_partial, acc, {pid_bh, pid_splitk, m_start, 0});

    // Metadata: [B*H_q, 2, num_splits, M_ceil]
    // Store m_i vector at dim 3 (M_ceil) starting at m_start
    store(g.Metadata, m_i, {pid_bh, 0, pid_splitk, m_start});
    store(g.Metadata, l_i, {pid_bh, 1, pid_splitk, m_start});
}

// ============================================================================
// Reduce Kernel
// ============================================================================

/**
 * Combines partial results from all splits for each (batch, head, m) position.
 * Each thread block handles one (bh, m_block). Each thread handles D elements.
 *
 * Grid: (cdiv(seqlen_q, BLOCK_M), B * H_q, 1)
 */
__launch_bounds__(REDUCE_NUM_THREADS, 1)
__global__ void splitk_reduce_kernel(const reduce_globals g) {

    const int pid_m  = blockIdx.x;
    const int pid_bh = blockIdx.y;
    const int batch_idx  = pid_bh / g.num_heads_q;
    const int head_q_idx = pid_bh % g.num_heads_q;
    const int m_start = pid_m * BLOCK_M;
    const int tid = threadIdx.x;

    // Each thread processes one or more (m, d) elements
    // With REDUCE_NUM_THREADS=256 and ATTN_D=128, we can process 2 rows per wave
    for (int m_local = 0; m_local < BLOCK_M; m_local++) {
        int m_idx = m_start + m_local;
        if (m_idx >= g.seqlen_q) break;

        // Step 1: Find global max across all splits
        float g_m = -INFINITY;
        for (int s = 0; s < g.num_splits; s++) {
            float m_val = g.Metadata[{pid_bh, 0, s, m_idx}];
            g_m = fmaxf(g_m, m_val);
        }

        // Step 2: Compute weighted denominator
        float g_sum = 0.0f;
        for (int s = 0; s < g.num_splits; s++) {
            float m_val = g.Metadata[{pid_bh, 0, s, m_idx}];
            float l_val = g.Metadata[{pid_bh, 1, s, m_idx}];
            float alpha = (m_val > -INFINITY) ? exp2f(m_val - g_m) : 0.0f;
            g_sum += l_val * alpha;
        }
        float g_sum_safe = (g_sum > 0.0f) ? g_sum : 1.0f;

        // Step 3: Compute and store output for each d
        for (int d = tid; d < ATTN_D; d += REDUCE_NUM_THREADS) {
            float acc_d = 0.0f;
            for (int s = 0; s < g.num_splits; s++) {
                float m_val = g.Metadata[{pid_bh, 0, s, m_idx}];
                float alpha = (m_val > -INFINITY) ? exp2f(m_val - g_m) : 0.0f;
                float o_val = g.O_partial[{pid_bh, s, m_idx, d}];
                acc_d += o_val * alpha;
            }
            float result = acc_d / g_sum_safe;
            // O: [B, seqlen_q, H_q, D]
            g.O[{batch_idx, m_idx, head_q_idx, d}] = __float2bfloat16(result);
        }

        // Step 4: Store LSE (only thread 0)
        if (tid == 0) {
            float lse_val;
            if (g_sum > 0.0f) {
                lse_val = (g_m + log2f(g_sum)) * INV_LOG2E;  // convert from log2 to ln
            } else {
                lse_val = g_m;
            }
            // LSE: [B*H_q, seqlen_q, 1, 1]
            g.LSE[{pid_bh, m_idx, 0, 0}] = lse_val;
        }
    }
}

// ============================================================================
// Dispatch Functions
// ============================================================================

void dispatch_splitk(splitk_globals g) {
    unsigned long mem = g.dynamic_shared_memory();
    if (mem > 0) {
        hipFuncSetAttribute((void*)splitk_attn_fwd_decode_kernel,
            hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    }
    splitk_attn_fwd_decode_kernel<<<g.grid(), g.block(), mem, g.stream>>>(g);
}

void dispatch_reduce(reduce_globals g) {
    unsigned long mem = g.dynamic_shared_memory();
    if (mem > 0) {
        hipFuncSetAttribute((void*)splitk_reduce_kernel,
            hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    }
    splitk_reduce_kernel<<<g.grid(), g.block(), mem, g.stream>>>(g);
}

// ============================================================================
// Python Bindings
// ============================================================================

PYBIND11_MODULE(flash_attn_fwd_decode, m) {
    m.doc() = "Flash Attention Forward Decode (HipKittens Split-K)";

    // SplitK kernel:
    //   dispatch_splitk(Q, K, V, O_partial, Metadata,
    //                   num_heads_q, num_heads_kv,
    //                   seqlen_q, seqlen_k, num_splits, sm_scale)
    //
    // Q:         [B, seqlen_q, H_q, D]        bf16
    // K:         [B, seqlen_k, H_kv, D]       bf16
    // V:         [B, seqlen_k, H_kv, D]       bf16
    // O_partial: [B*H_q, num_splits, M_ceil, D] float
    // Metadata:  [B*H_q, 2, num_splits, M_ceil]  float
    py::bind_function<dispatch_splitk>(m, "dispatch_splitk",
        &splitk_globals::Q,
        &splitk_globals::K,
        &splitk_globals::V,
        &splitk_globals::O_partial,
        &splitk_globals::Metadata,
        &splitk_globals::num_heads_q,
        &splitk_globals::num_heads_kv,
        &splitk_globals::seqlen_q,
        &splitk_globals::seqlen_k,
        &splitk_globals::num_splits,
        &splitk_globals::sm_scale
    );

    // Reduce kernel:
    //   dispatch_reduce(O_partial, Metadata, O, LSE,
    //                   num_heads_q, seqlen_q, num_splits)
    //
    // O_partial: [B*H_q, num_splits, M_ceil, D] float
    // Metadata:  [B*H_q, 2, num_splits, M_ceil]  float
    // O:         [B, seqlen_q, H_q, D]         bf16
    // LSE:       [B*H_q, seqlen_q, 1, 1]       float
    py::bind_function<dispatch_reduce>(m, "dispatch_reduce",
        &reduce_globals::O_partial,
        &reduce_globals::Metadata,
        &reduce_globals::O,
        &reduce_globals::LSE,
        &reduce_globals::num_heads_q,
        &reduce_globals::seqlen_q,
        &reduce_globals::num_splits
    );
}
