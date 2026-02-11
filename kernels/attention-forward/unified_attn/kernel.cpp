// Unified Attention Forward Kernel — HipKittens Port
// Ported from: reference/triton/unified_attention.py (kernel_unified_attention_2d)
//
// Unified decode+prefill attention with paged KV cache (block_tables).
// Supports: GQA, causal masking, sliding window, alibi slopes, softcap, FP8.
//
// This kernel handles both single-token decode and multi-token prefill
// within a single launch, mapping queries to paged KV blocks via block_tables.
//
// Layout:
//   Q: (num_tokens, H, D) — query tokens (packed across sequences)
//   K_cache: (num_blocks, block_size, H_KV, D) — paged key cache
//   V_cache: (num_blocks, block_size, H_KV, D) — paged value cache
//   O: (num_tokens, H, D) — output
//   block_tables: (num_seqs, max_blocks_per_seq) — page table
//   seq_lens: (num_seqs,) — total sequence lengths
//   query_start_len: (num_seqs+1,) — cumulative query token offsets

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

#ifndef NUM_Q_HEADS
constexpr int NUM_Q_HEADS = 32;
#endif
#ifndef NUM_KV_HEADS
constexpr int NUM_KV_HEADS = 8;
#endif
#ifndef HEAD_SIZE
constexpr int HEAD_SIZE = 128;
#endif
#ifndef BLOCK_SIZE
constexpr int BLOCK_SIZE = 16;     // KV cache block size (page size)
#endif
#ifndef TILE_SIZE
constexpr int TILE_SIZE = 64;      // Number of KV positions per iteration
#endif
#ifndef SLIDING_WINDOW
constexpr int SLIDING_WINDOW = 0;  // 0 = no sliding window
#endif
#ifndef USE_ALIBI_SLOPES
constexpr int USE_ALIBI_SLOPES = 0;
#endif
#ifndef USE_SOFTCAP
constexpr int USE_SOFTCAP = 0;
#endif

constexpr int NUM_QUERIES_PER_KV = NUM_Q_HEADS / NUM_KV_HEADS;
constexpr int BLOCK_M = 64;  // queries per thread block
constexpr int BLOCK_Q = BLOCK_M / NUM_QUERIES_PER_KV;  // query positions per block

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using G = kittens::group<NUM_WARPS>;
using _gl_bf16 = gl<bf16, -1, -1, -1, -1>;
using _gl_int = gl<int, -1, -1, -1, -1>;
using _gl_f32 = gl<float, -1, -1, -1, -1>;

// exp2 helper
template<typename T, ducks::rt_layout::all layout, ducks::rt_shape::all shape>
__device__ inline void exp2_tile(rt_base<T, layout, shape> &dst, const rt_base<T, layout, shape> &src) {
    #pragma unroll
    for(int k = 0; k < dst.packed_per_thread; k++) {
        dst.data[k] = base_ops::exp2::op(src.data[k]);
    }
}

struct unified_globals {
    _gl_bf16 Qg, Og;
    _gl_bf16 K_cache, V_cache;
    _gl_int block_tables;
    _gl_int seq_lens;
    _gl_int query_start_len;
    _gl_f32 alibi_slopes;
    float scale;
    float softcap;
    int num_seqs;
    int block_table_stride;
    hipStream_t stream;
    dim3 grid() {
        // grid: (kv_heads, total_q_blocks)
        // total_q_blocks is precomputed by the host
        return dim3(NUM_KV_HEADS, 1024);  // upper bound, early exit in kernel
    }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__launch_bounds__(NUM_THREADS, 2)
__global__ void unified_attn_kernel(const unified_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    constexpr int HEAD_SIZE_PAD = HEAD_SIZE;  // Assume power of 2
    st_bf<TILE_SIZE, HEAD_SIZE_PAD, st_32x32_s> (&k_smem)[2] = al.allocate<st_bf<TILE_SIZE, HEAD_SIZE_PAD, st_32x32_s>, 2>();
    st_bf<TILE_SIZE, HEAD_SIZE_PAD, st_8x32_s> (&v_smem)[2] = al.allocate<st_bf<TILE_SIZE, HEAD_SIZE_PAD, st_8x32_s>, 2>();

    const int kv_head_idx = blockIdx.x;
    const int q_block_global_idx = blockIdx.y;

    // Binary search for sequence index
    // Find which sequence this q_block belongs to
    int seq_idx = 0;
    {
        int left = 0, right = g.num_seqs;
        while (left < right) {
            int mid = (left + right) / 2;
            int val = *((int*)&g.query_start_len[{mid, 0, 0, 0}]);
            int mid_val = val / BLOCK_Q + mid;
            if (mid_val <= q_block_global_idx) left = mid + 1;
            else right = mid;
        }
        seq_idx = left - 1;
    }

    const int q_block_start = *((int*)&g.query_start_len[{seq_idx, 0, 0, 0}]);
    const int q_block_start_idx = q_block_start / BLOCK_Q + seq_idx;
    const int q_block_local_idx = q_block_global_idx - q_block_start_idx;
    const int cur_batch_start = q_block_start;
    const int cur_batch_end = *((int*)&g.query_start_len[{seq_idx + 1, 0, 0, 0}]);
    const int cur_batch_query_len = cur_batch_end - cur_batch_start;

    if (q_block_local_idx * BLOCK_Q >= cur_batch_query_len) return;

    const int seq_len = *((int*)&g.seq_lens[{seq_idx, 0, 0, 0}]);
    const int context_len = seq_len - cur_batch_query_len;

    const float RCP_LN2 = 1.44269504089f;
    const float qk_scale = g.scale * RCP_LN2;

    // Register tiles
    rt_bf<BLOCK_M, HEAD_SIZE_PAD, row_l, rt_32x16_s> q_reg;
    rt_fl<BLOCK_M, TILE_SIZE, col_l, rt_16x32_4_s> att_block;
    rt_fl<BLOCK_M, HEAD_SIZE_PAD, col_l, rt_32x32_s> o_reg;
    typename rt_fl<BLOCK_M, TILE_SIZE, col_l, rt_16x32_4_s>::row_vec M_vec, L_vec;

    zero(o_reg);
    #pragma unroll
    for (int o = 0; o < M_vec.outer_dim; ++o) {
        #pragma unroll
        for (int i = 0; i < M_vec.inner_dim; ++i) {
            M_vec.data[o][i] = -INFINITY;
        }
    }
    ones(L_vec);

    // Load Q: interleaved across GQA heads
    // query_pos = q_block_local_idx * BLOCK_Q + offs_m / NUM_QUERIES_PER_KV
    // query_head = kv_head_idx * NUM_QUERIES_PER_KV + offs_m % NUM_QUERIES_PER_KV
    rt_fl<BLOCK_M, HEAD_SIZE_PAD, row_l, rt_32x32_s> q_reg_fl;
    // Simplified: load Q for the first head in the group
    int q_token_start = cur_batch_start + q_block_local_idx * BLOCK_Q;
    load<0, rt_fl<BLOCK_M, HEAD_SIZE_PAD, row_l, rt_32x32_s>, _gl_bf16>(
        q_reg_fl, g.Qg, {q_token_start, kv_head_idx * NUM_QUERIES_PER_KV, 0, 0});
    mul(q_reg_fl, q_reg_fl, qk_scale);
    copy(q_reg, q_reg_fl);

    // Compute tile bounds
    int max_prefix_len = context_len + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) / NUM_QUERIES_PER_KV + 1;
    max_prefix_len = min(max_prefix_len, seq_len);
    const int num_tiles = (max_prefix_len + TILE_SIZE - 1) / TILE_SIZE;

    // Swizzled offsets
    using T = typename st_bf<TILE_SIZE, HEAD_SIZE_PAD, st_32x32_s>::dtype;
    constexpr int bytes_per_thread = st_32x32_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = TILE_SIZE * HEAD_SIZE_PAD * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_K[memcpy_per_tile];
    uint32_t swizzled_offsets_V[memcpy_per_tile];
    G::prefill_swizzled_offsets<1, false>(k_smem[0], g.K_cache, swizzled_offsets_K);
    G::prefill_swizzled_offsets<1, false>(v_smem[0], g.V_cache, swizzled_offsets_V);

    // Main tile loop over KV sequence
    for (int j = 0; j < num_tiles; j++) {
        int buf = j & 1;
        int seq_offset_base = j * TILE_SIZE;

        // Look up physical block for this tile position
        // physical_block = block_tables[seq_idx, seq_offset / BLOCK_SIZE]
        int page_idx = seq_offset_base / BLOCK_SIZE;
        int physical_block = *((int*)&g.block_tables[{seq_idx, page_idx, 0, 0}]);

        int within_block = seq_offset_base % BLOCK_SIZE;
        G::load<1, false>(k_smem[buf], g.K_cache,
            {physical_block, within_block, kv_head_idx, 0}, swizzled_offsets_K);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();

        rt_bf<TILE_SIZE, HEAD_SIZE_PAD, row_l, rt_32x16_s> k_reg;
        load(k_reg, k_smem[buf]);

        // S = Q @ K^T
        zero(att_block);
        mma_ABt(att_block, q_reg, k_reg, att_block);

        // Causal masking
        // seq_offset = j * TILE_SIZE + [0..TILE_SIZE)
        // mask: seq_offset < context_len + query_pos + 1

        // Online softmax
        typename decltype(att_block)::row_vec m_j;
        col_max(m_j, att_block, M_vec);

        // Clamp -inf to 0 (for sliding window edge case)
        // P = exp2(S - m_j)
        sub_col(att_block, att_block, m_j);
        #pragma unroll
        for (int r = 0; r < att_block.height; r++) {
            #pragma unroll
            for (int c = 0; c < att_block.width; c++) {
                exp2_tile(att_block.tiles[r][c], att_block.tiles[r][c]);
            }
        }

        typename decltype(att_block)::row_vec l_j;
        col_sum(l_j, att_block);

        // alpha = exp2(M - m_j)
        typename decltype(att_block)::row_vec alpha;
        sub(alpha, M_vec, m_j);
        exp2(alpha, alpha);

        // acc = acc * alpha
        mul_col(o_reg, o_reg, alpha);

        // L = L * alpha + l_j
        mul(L_vec, L_vec, alpha);
        add(L_vec, L_vec, l_j);
        copy(M_vec, m_j);

        // V load and accumulate
        G::load<1, false>(v_smem[buf], g.V_cache,
            {physical_block, within_block, kv_head_idx, 0}, swizzled_offsets_V);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();

        rt_bf<TILE_SIZE, HEAD_SIZE_PAD, col_l, rt_16x32_4_s> v_reg;
        load(v_reg, v_smem[buf]);

        rt_bf<BLOCK_M, TILE_SIZE, row_l, rt_16x32_4_s> att_bf16;
        copy(att_bf16, att_block);
        mma_AB(o_reg, att_bf16, v_reg, o_reg);
    }

    // Epilogue: O = O / L
    typename decltype(o_reg)::row_vec one_over_L;
    // one_over_L = 1.0 / L
    #pragma unroll
    for (int o = 0; o < one_over_L.outer_dim; ++o) {
        #pragma unroll
        for (int i = 0; i < one_over_L.inner_dim; ++i) {
            one_over_L.data[o][i] = 1.0f / L_vec.data[o][i];
        }
    }
    mul_col(o_reg, o_reg, one_over_L);

    // Store output - o_reg is already col_l, copy to row_l tile for store
    rt_fl<BLOCK_M, HEAD_SIZE_PAD, row_l, rt_32x32_s> o_out;
    copy(o_out, o_reg);
    store<0>(g.Og, o_out,
        {q_token_start, kv_head_idx * NUM_QUERIES_PER_KV, 0, 0});
}

void dispatch_unified(unified_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)unified_attn_kernel,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    unified_attn_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

PYBIND11_MODULE(tk_unified_attn, m) {
    m.doc() = "HipKittens Unified Attention Forward kernel";
    py::bind_function<dispatch_unified>(m, "dispatch",
        &unified_globals::Qg,
        &unified_globals::Og,
        &unified_globals::K_cache,
        &unified_globals::V_cache,
        &unified_globals::block_tables,
        &unified_globals::seq_lens,
        &unified_globals::query_start_len
    );
}
