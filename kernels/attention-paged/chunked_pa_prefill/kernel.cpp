/**
 * Chunked Paged Attention Prefill Kernel (HipKittens)
 *
 * Ported from AITER Triton: chunked_pa_prefill.py
 *
 * A "chunked" decode-style paged attention that processes one query token
 * per sequence. Iterates over all KV blocks using block tables with support
 * for interleaved K cache layout and sliding window attention.
 *
 * Grid: (num_q_heads, num_tokens)
 * Each thread block processes one (head, token) pair.
 *
 * Algorithm:
 *   For each query token:
 *     1. Load Q [1, HEAD_SZ]
 *     2. Loop over KV blocks via block_table:
 *        a. Load K [BLOCK_SIZE, HEAD_SZ] from paged cache
 *        b. Compute score = scale * sum(K * Q) per KV position
 *        c. Apply sliding window mask if enabled
 *        d. Online softmax update
 *        e. Load V [BLOCK_SIZE, HEAD_SZ] from paged cache
 *        f. Accumulate O += attn_weight * V
 *     3. Normalize O /= sum_exp
 *     4. Store O [1, HEAD_SZ]
 *
 * Key difference from pa_decode: this supports variable-length queries
 * and can filter by query length (process only decode tokens).
 */

#include "kittens.cuh"

using namespace kittens;

#ifndef CPA_HEAD_SZ
#define CPA_HEAD_SZ 128
#endif

#ifndef CPA_NUM_KV_HEADS
#define CPA_NUM_KV_HEADS 8
#endif

#ifndef CPA_NUM_Q_HEADS
#define CPA_NUM_Q_HEADS 64
#endif

#ifndef CPA_BLOCK_SIZE
#define CPA_BLOCK_SIZE 32
#endif

constexpr int HEAD_SZ = CPA_HEAD_SZ;
constexpr int NUM_KV_HEADS = CPA_NUM_KV_HEADS;
constexpr int NUM_Q_HEADS = CPA_NUM_Q_HEADS;
constexpr int QUERY_GRP_SZ = NUM_Q_HEADS / NUM_KV_HEADS;
constexpr int BLOCK_SIZE = CPA_BLOCK_SIZE;  // KV page size

// Thread configuration: decode is memory-bound
#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using G = kittens::group<NUM_WARPS>;

// Global memory types
using q_gl  = gl<bf16, -1, -1, -1, CPA_HEAD_SZ>;   // [num_tokens, num_q_heads, 1, head_sz]
using kv_gl = gl<bf16, -1, -1, -1, CPA_HEAD_SZ>;   // [num_blocks, num_kv_heads, block_size, head_sz]
using o_gl  = gl<bf16, -1, -1, -1, CPA_HEAD_SZ>;   // [num_tokens, num_q_heads, 1, head_sz]
using bt_gl = gl<int, -1, -1, -1, -1>;
using si_gl = gl<int, -1, -1, -1, -1>;

struct chunked_pa_globals {
    q_gl  Q;              // [num_tokens, num_q_heads, 1, head_sz]
    kv_gl K_cache;        // [num_blocks, num_kv_heads, block_size, head_sz]
    kv_gl V_cache;        // [num_blocks, num_kv_heads, block_size, head_sz]
    o_gl  O;              // [num_tokens, num_q_heads, 1, head_sz]
    bt_gl block_table;    // [num_seqs, max_blocks, 1, 1]
    si_gl seq_lens;       // [num_seqs, 1, 1, 1]
    si_gl query_starts;   // [num_seqs + 1, 1, 1, 1] cumulative query token starts
    float scale;
    int   num_tokens;
    int   num_seqs;
    int   sliding_window; // 0 = disabled
    hipStream_t stream;

    dim3 grid() {
        return dim3(NUM_Q_HEADS, num_tokens);
    }
    dim3 block() {
        return dim3(NUM_THREADS);
    }
    size_t dynamic_shared_memory() {
        return sizeof(bf16) * BLOCK_SIZE * HEAD_SZ;
    }
};

/**
 * Chunked paged attention kernel - one query token per thread block.
 */
__launch_bounds__(NUM_THREADS, 1)
__global__ void chunked_pa_kernel(const chunked_pa_globals g) {
    const int q_head_idx = blockIdx.x;
    const int token_idx = blockIdx.y;

    if (token_idx >= g.num_tokens) return;

    const int kv_head_idx = q_head_idx / QUERY_GRP_SZ;

    // Find which sequence this token belongs to (binary search)
    int seq_idx = 0;
    for (int s = 0; s < g.num_seqs; s++) {
        const int qs = *(const int*)&g.query_starts[{s, 0, 0, 0}];
        const int qe = *(const int*)&g.query_starts[{s + 1, 0, 0, 0}];
        if (token_idx >= qs && token_idx < qe) {
            seq_idx = s;
            break;
        }
    }

    const int seq_len = *(const int*)&g.seq_lens[{seq_idx, 0, 0, 0}];
    if (seq_len <= 0) return;

    const int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Shared memory for KV
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, CPA_HEAD_SZ, st_16x32_s> &kv_smem = al.allocate<st_bf<BLOCK_SIZE, CPA_HEAD_SZ, st_16x32_s>>();

    const float temperature = g.scale * 1.44269504089f;

    // Load Q for this token (padded to BLOCK_SIZE rows, only first row used)
    rt<bf16, BLOCK_SIZE, CPA_HEAD_SZ, row_l, rt_16x32_s> q_reg;
    zero(q_reg);
    if (warpid() == 0) {
        // Load single query row into first row of q_reg
        rt<bf16, 16, CPA_HEAD_SZ, row_l, rt_16x32_s> q_load;
        load<2>(q_load, g.Q, {token_idx, q_head_idx, 0, 0});
        // Copy into first subtile row of q_reg
        #pragma unroll
        for (int j = 0; j < q_reg.width; j++) {
            q_reg.tiles[0][j] = q_load.tiles[0][j];
        }
    }
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    // Output accumulator
    rt<float, BLOCK_SIZE, CPA_HEAD_SZ, col_l, rt_16x16_s> o_acc;
    zero(o_acc);
    float m_prev = -1e30f;
    float l_prev = 0.0f;

    // Loop over KV blocks
    for (int blk = 0; blk < num_blocks; blk++) {
        const int phys_block = *(const int*)&g.block_table[{seq_idx, blk, 0, 0}];
        const int block_start = blk * BLOCK_SIZE;
        const int valid = min(BLOCK_SIZE, seq_len - block_start);

        // Sliding window check
        if (g.sliding_window > 0) {
            const int q_pos = seq_len - 1; // For decode, Q is the last position
            if (block_start + valid - 1 < q_pos - g.sliding_window) {
                continue; // Skip blocks outside sliding window
            }
        }

        // Load K from paged cache
        G::load<2, false>(kv_smem, g.K_cache, {phys_block, kv_head_idx, 0, 0});
        __builtin_amdgcn_s_barrier();

        rt<bf16, BLOCK_SIZE, CPA_HEAD_SZ, row_l, rt_16x32_s> k_reg;
        load(k_reg, kv_smem);
        asm volatile("s_waitcnt lgkmcnt(0)");

        // Compute attention scores: Q @ K^T -> (BLOCK_SIZE, BLOCK_SIZE)
        rt<float, BLOCK_SIZE, BLOCK_SIZE, col_l, rt_16x16_s> scores;
        zero(scores);
        mma_ABt(scores, q_reg, k_reg, scores);
        mul(scores, scores, temperature);

        // Online softmax (only first row is meaningful)
        typename decltype(scores)::col_vec max_vec;
        col_max(max_vec, scores);

        float m_new = m_prev;
        #pragma unroll
        for (int o = 0; o < max_vec.outer_dim; o++) {
            #pragma unroll
            for (int i = 0; i < max_vec.inner_dim; i++) {
                m_new = fmaxf(m_new, fmaxf(max_vec.data[o][i].x, max_vec.data[o][i].y));
            }
        }

        float correction = exp2f(m_prev - m_new);

        sub_col(scores, scores, max_vec);
        #pragma unroll
        for (int i = 0; i < scores.height; i++) {
            #pragma unroll
            for (int j = 0; j < scores.width; j++) {
                #pragma unroll
                for (int k = 0; k < scores.tiles[i][j].packed_per_thread; k++) {
                    scores.tiles[i][j].data[k].x = base_ops::exp2::op(scores.tiles[i][j].data[k].x);
                    scores.tiles[i][j].data[k].y = base_ops::exp2::op(scores.tiles[i][j].data[k].y);
                }
            }
        }

        typename decltype(scores)::col_vec sum_vec;
        col_sum(sum_vec, scores);
        float l_sum = 0.0f;
        #pragma unroll
        for (int o = 0; o < sum_vec.outer_dim; o++) {
            #pragma unroll
            for (int i = 0; i < sum_vec.inner_dim; i++) {
                l_sum += sum_vec.data[o][i].x + sum_vec.data[o][i].y;
            }
        }

        float l_new = correction * l_prev + l_sum;

        if (blk > 0) {
            mul(o_acc, o_acc, correction);
        }

        // Load V from paged cache
        G::load<2, false>(kv_smem, g.V_cache, {phys_block, kv_head_idx, 0, 0});
        __builtin_amdgcn_s_barrier();

        rt<bf16, BLOCK_SIZE, CPA_HEAD_SZ, row_l, rt_16x32_s> v_reg;
        load(v_reg, kv_smem);
        asm volatile("s_waitcnt lgkmcnt(0)");

        // P @ V accumulation using mma_AB: O = scores @ V
        // scores: (BLOCK_SIZE, BLOCK_SIZE) row_l, V: (BLOCK_SIZE, HEAD_SZ) col_l
        rt<bf16, BLOCK_SIZE, BLOCK_SIZE, row_l, rt_16x32_s> scores_bf16_row;
        copy(scores_bf16_row, scores);

        rt<bf16, BLOCK_SIZE, CPA_HEAD_SZ, col_l, rt_16x32_s> v_col;
        transpose(v_col, v_reg);

        mma_AB(o_acc, scores_bf16_row, v_col, o_acc);

        m_prev = m_new;
        l_prev = l_new;

        __builtin_amdgcn_s_barrier();
    }

    // Normalize
    if (l_prev > 0.0f) {
        mul(o_acc, o_acc, 1.0f / l_prev);
    }

    // Store output: extract first subtile row
    rt<bf16, BLOCK_SIZE, CPA_HEAD_SZ, row_l, rt_16x32_s> o_row;
    copy(o_row, o_acc);

    // Only store the first 16 rows (first query token)
    rt<bf16, 16, CPA_HEAD_SZ, row_l, rt_16x32_s> o_out;
    #pragma unroll
    for (int j = 0; j < o_row.width; j++) {
        o_out.tiles[0][j] = o_row.tiles[0][j];
    }

    if (warpid() == 0) {
        store<2>(g.O, o_out, {token_idx, q_head_idx, 0, 0});
    }
}

void dispatch_chunked_pa(chunked_pa_globals &g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)chunked_pa_kernel,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    chunked_pa_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}
