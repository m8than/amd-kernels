/**
 * Paged Attention Prefill Kernel (HipKittens)
 *
 * Ported from AITER Triton: pa_prefill.py
 *
 * Computes attention for prefill (full query sequence) with a paged KV cache.
 * Two-phase approach:
 *   Phase 1 (Context): Q attends to KV tokens from the cached prefix (paged cache).
 *   Phase 2 (Self): Q attends to the new query tokens themselves (causal mask).
 *
 * Grid: (num_q_heads, num_q_blocks, num_seqs)
 * Each thread block processes one Q tile of BLOCK_M query tokens.
 *
 * Algorithm:
 *   1. Load Q tile [BLOCK_M, HEAD_SZ]
 *   2. Phase 1 - Context attention:
 *      Loop over all KV pages from the paged cache:
 *        Load K [BLOCK_N, HEAD_SZ] from paged cache via block table
 *        Compute S = Q @ K^T, apply sliding window mask
 *        Online softmax update
 *        Load V [BLOCK_N, HEAD_SZ], accumulate O += P @ V
 *   3. Phase 2 - Self attention:
 *      Loop over new query tokens with causal mask:
 *        Load K, V from new tokens (contiguous)
 *        Compute S = Q @ K^T, apply causal mask
 *        Online softmax update, accumulate O += P @ V
 *   4. Store O [BLOCK_M, HEAD_SZ]
 */

#include "kittens.cuh"

using namespace kittens;

#ifndef PA_HEAD_SZ
#define PA_HEAD_SZ 128
#endif

#ifndef PA_NUM_KV_HEADS
#define PA_NUM_KV_HEADS 8
#endif

#ifndef PA_NUM_Q_HEADS
#define PA_NUM_Q_HEADS 64
#endif

#ifndef PA_KV_BLK_SZ
#define PA_KV_BLK_SZ 16
#endif

constexpr int HEAD_SZ = PA_HEAD_SZ;
constexpr int NUM_KV_HEADS = PA_NUM_KV_HEADS;
constexpr int NUM_Q_HEADS = PA_NUM_Q_HEADS;
constexpr int QUERY_GRP_SZ = NUM_Q_HEADS / NUM_KV_HEADS;
constexpr int KV_BLK_SZ = PA_KV_BLK_SZ;

// Tile sizes for prefill (larger than decode since compute-bound)
constexpr int BLOCK_M = 32;  // Q tile: 32 query tokens
constexpr int BLOCK_N = 32;  // KV tile: 32 KV tokens per iteration

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using G = kittens::group<NUM_WARPS>;

// Global memory types
using q_gl   = gl<bf16, -1, -1, -1, PA_HEAD_SZ>;  // [total_tokens, num_heads, 1, head_sz]
using kv_gl  = gl<bf16, -1, -1, -1, PA_HEAD_SZ>;  // [total_tokens, num_kv_heads, 1, head_sz] or paged
using kvc_gl = gl<bf16, -1, -1, -1, PA_HEAD_SZ>;  // [num_blocks, num_kv_heads, kv_blk_sz, head_sz]
using o_gl   = gl<bf16, -1, -1, -1, PA_HEAD_SZ>;  // [total_tokens, num_heads, 1, head_sz]
using bt_gl  = gl<int, -1, -1, -1, -1>;
using si_gl  = gl<int, -1, -1, -1, -1>;

struct pa_prefill_globals {
    q_gl   Q;           // [total_tokens, num_q_heads, 1, head_sz]
    kv_gl  K;           // [total_tokens, num_kv_heads, 1, head_sz] (new tokens)
    kv_gl  V;           // [total_tokens, num_kv_heads, 1, head_sz] (new tokens)
    kvc_gl K_cache;     // [num_blocks, num_kv_heads, kv_blk_sz, head_sz] (paged)
    kvc_gl V_cache;     // [num_blocks, num_kv_heads, kv_blk_sz, head_sz] (paged)
    o_gl   O;           // [total_tokens, num_q_heads, 1, head_sz]
    bt_gl  block_table; // [num_seqs, max_blocks, 1, 1]
    si_gl  seq_start;   // [num_seqs + 1, 1, 1, 1] cumulative start positions
    si_gl  seq_lens;    // [num_seqs, 1, 1, 1] total seq lengths (context + new)
    si_gl  ctx_lens;    // [num_seqs, 1, 1, 1] context (cached) lengths
    float  scale;
    hipStream_t stream;

    dim3 grid() {
        // Compute max possible Q blocks (conservative estimate)
        // Actual Q length varies per sequence
        return dim3(NUM_Q_HEADS, 64, 1);  // Will early-exit invalid blocks
    }
    dim3 block() {
        return dim3(NUM_THREADS);
    }
    size_t dynamic_shared_memory() {
        return MAX_SHARED_MEMORY;
    }
};

// Tile type aliases matching the GQA reference pattern
template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s>
using q_tile = rt<T, BLOCK_M, D, L, S>;

template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s>
using kv_tile = rt<T, BLOCK_N, D, L, S>;

template<int D, typename T=float, typename L=col_l, typename S=rt_16x32_4_s>
using attn_tile = rt<T, BLOCK_N, BLOCK_M, L, S>;

__launch_bounds__(NUM_THREADS, 2)
__global__ void pa_prefill_kernel(const pa_prefill_globals g) {
    const int q_head_idx = blockIdx.x;
    const int q_block_idx = blockIdx.y;
    const int seq_idx = blockIdx.z;

    const int kv_head_idx = q_head_idx / QUERY_GRP_SZ;

    // Get sequence metadata
    const int seq_start = *(const int*)&g.seq_start[{seq_idx, 0, 0, 0}];
    const int next_seq_start = *(const int*)&g.seq_start[{seq_idx + 1, 0, 0, 0}];
    const int num_new_tokens = next_seq_start - seq_start;
    const int ctx_len = *(const int*)&g.ctx_lens[{seq_idx, 0, 0, 0}];
    const int total_seq_len = *(const int*)&g.seq_lens[{seq_idx, 0, 0, 0}];

    // Check if this Q block is valid
    const int q_start = q_block_idx * BLOCK_M;
    if (q_start >= num_new_tokens) return;

    // Allocate shared memory
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_N, PA_HEAD_SZ, st_32x32_s> (&kv_smem)[2] = al.allocate<st_bf<BLOCK_N, PA_HEAD_SZ, st_32x32_s>, 2>();

    const float temperature = g.scale * 1.44269504089f; // scale * log2(e)

    // Load Q tile [BLOCK_M, HEAD_SZ] into registers
    // Q is contiguous for the new tokens of this sequence
    q_tile<PA_HEAD_SZ, bf16> q_reg;
    q_tile<PA_HEAD_SZ, float> q_reg_fl;

    // Each warp loads a portion of Q
    const int wid = warpid();
    const int global_q_offset = seq_start + q_start;

    // Load Q - each warp handles its portion
    load<2>(q_reg_fl, g.Q, {global_q_offset + wid * (BLOCK_M / NUM_WARPS), q_head_idx, 0, 0});
    asm volatile("s_waitcnt vmcnt(0)");

    // Scale Q
    mul(q_reg_fl, q_reg_fl, temperature);
    copy(q_reg, q_reg_fl);

    // No need for transposed Q - we'll use mma_ABt

    // Output accumulator
    rt<float, PA_HEAD_SZ, BLOCK_M, col_l, rt_32x32_s> o_reg;
    zero(o_reg);

    // Online softmax state vectors (one per Q row)
    typename attn_tile<PA_HEAD_SZ, float, col_l, rt_32x32_s>::row_vec max_vec, norm_vec, max_vec_prev, scale_vec;
    #pragma unroll
    for (int o = 0; o < max_vec.outer_dim; ++o) {
        #pragma unroll
        for (int i = 0; i < max_vec.inner_dim; ++i) {
            max_vec.data[o][i] = -INFINITY;
        }
    }
    zero(norm_vec);

    // Prefill swizzled offsets
    uint32_t swizzled_offsets[8]; // Sized for typical memcpy_per_tile
    G::prefill_swizzled_offsets<2, false>(kv_smem[0], g.K_cache, swizzled_offsets);

    // ============================================
    // Phase 1: Context attention (paged KV cache)
    // ============================================
    const int num_ctx_blocks = (ctx_len + KV_BLK_SZ - 1) / KV_BLK_SZ;

    for (int ctx_blk = 0; ctx_blk < num_ctx_blocks; ctx_blk++) {
        const int phys_block = *(const int*)&g.block_table[{seq_idx, ctx_blk, 0, 0}];
        const int ctx_start = ctx_blk * KV_BLK_SZ;
        const int valid_ctx = min(KV_BLK_SZ, ctx_len - ctx_start);

        // Load K from paged cache
        G::load<2, false>(kv_smem[0], g.K_cache, {phys_block, kv_head_idx, 0, 0});
        __builtin_amdgcn_s_barrier();

        kv_tile<PA_HEAD_SZ, bf16> k_reg;
        load(k_reg, kv_smem[0]);
        asm volatile("s_waitcnt lgkmcnt(0)");

        // Compute S = Q @ K^T
        attn_tile<PA_HEAD_SZ, float, col_l, rt_32x32_s> att;
        zero(att);
        mma_ABt(att, q_reg, k_reg, att);

        // Online softmax update
        copy(max_vec_prev, max_vec);
        col_max(max_vec, att, max_vec_prev);
        sub(scale_vec, max_vec_prev, max_vec);
        exp2(scale_vec, scale_vec);
        mul_col(o_reg, o_reg, scale_vec);
        mul(norm_vec, norm_vec, scale_vec);
        sub_col(att, att, max_vec);

        // exp2 on attention tiles
        #pragma unroll
        for (int i = 0; i < att.height; i++) {
            #pragma unroll
            for (int j = 0; j < att.width; j++) {
                #pragma unroll
                for (int k = 0; k < att.tiles[i][j].packed_per_thread; k++) {
                    att.tiles[i][j].data[k] = base_ops::exp2::op(att.tiles[i][j].data[k]);
                }
            }
        }
        col_sum(norm_vec, att, norm_vec);

        // Load V from paged cache
        G::load<2, false>(kv_smem[1], g.V_cache, {phys_block, kv_head_idx, 0, 0});
        __builtin_amdgcn_s_barrier();

        kv_tile<PA_HEAD_SZ, bf16, col_l, rt_16x32_4_s> v_reg;
        load(v_reg, kv_smem[1]);
        asm volatile("s_waitcnt lgkmcnt(0)");

        // Convert attention to bf16
        rt<bf16, BLOCK_M, BLOCK_N, row_l, rt_16x32_4_s> att_bf16;
        copy(att_bf16, att);

        // Accumulate O += P @ V
        mma_AB(o_reg, att_bf16, v_reg, o_reg);

        __builtin_amdgcn_s_barrier();
    }

    // ============================================
    // Phase 2: Self-attention (new query tokens)
    // ============================================
    const int num_new_kv_blocks = (num_new_tokens + BLOCK_N - 1) / BLOCK_N;

    for (int kv_blk = 0; kv_blk < num_new_kv_blocks; kv_blk++) {
        const int kv_start = kv_blk * BLOCK_N;

        // Load K from new tokens (contiguous layout)
        const int global_kv_offset = seq_start + kv_start;
        G::load<2, false>(kv_smem[0], g.K, {global_kv_offset, kv_head_idx, 0, 0});
        __builtin_amdgcn_s_barrier();

        kv_tile<PA_HEAD_SZ, bf16> k_reg;
        load(k_reg, kv_smem[0]);
        asm volatile("s_waitcnt lgkmcnt(0)");

        // Compute S = Q @ K^T
        attn_tile<PA_HEAD_SZ, float, col_l, rt_32x32_s> att;
        zero(att);
        mma_ABt(att, q_reg, k_reg, att);

        // Apply causal mask: positions where q_pos < kv_pos get -inf
        // q_pos = q_start + row_idx (within new tokens), kv_pos = kv_start + col_idx
        // For causal: q_start + row >= kv_start + col
        if (kv_start >= q_start) {
            // Some positions need masking
            make_causal(att, -1e10f);
        }

        // Online softmax update
        copy(max_vec_prev, max_vec);
        col_max(max_vec, att, max_vec_prev);
        sub(scale_vec, max_vec_prev, max_vec);
        exp2(scale_vec, scale_vec);
        mul_col(o_reg, o_reg, scale_vec);
        mul(norm_vec, norm_vec, scale_vec);
        sub_col(att, att, max_vec);

        // exp2 on attention tiles
        #pragma unroll
        for (int i = 0; i < att.height; i++) {
            #pragma unroll
            for (int j = 0; j < att.width; j++) {
                #pragma unroll
                for (int k = 0; k < att.tiles[i][j].packed_per_thread; k++) {
                    att.tiles[i][j].data[k] = base_ops::exp2::op(att.tiles[i][j].data[k]);
                }
            }
        }
        col_sum(norm_vec, att, norm_vec);

        // Load V from new tokens
        G::load<2, false>(kv_smem[1], g.V, {global_kv_offset, kv_head_idx, 0, 0});
        __builtin_amdgcn_s_barrier();

        kv_tile<PA_HEAD_SZ, bf16, col_l, rt_16x32_4_s> v_reg;
        load(v_reg, kv_smem[1]);
        asm volatile("s_waitcnt lgkmcnt(0)");

        attn_tile<PA_HEAD_SZ, bf16, col_l, rt_16x32_4_s> att_bf16;
        copy(att_bf16, att);

        mma_AtB(o_reg, v_reg, att_bf16, o_reg);

        __builtin_amdgcn_s_barrier();
    }

    // Final normalization
    div_col(o_reg, o_reg, norm_vec);

    // Store output - o_reg is already col_l, copy to row_l tile for store
    rt<float, BLOCK_M, PA_HEAD_SZ, row_l, rt_32x32_s> o_reg_row;
    copy(o_reg_row, o_reg);

    rt<bf16, BLOCK_M, PA_HEAD_SZ, row_l, rt_32x16_s> o_out;
    copy(o_out, o_reg_row);

    store<2>(g.O, o_out, {global_q_offset + warpid() * (BLOCK_M / NUM_WARPS), q_head_idx, 0, 0});
}

void dispatch_pa_prefill(pa_prefill_globals &g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)pa_prefill_kernel,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    pa_prefill_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}
